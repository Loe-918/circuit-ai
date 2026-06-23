"""
Circuit AI Learning Platform
AI Tutor + Community Q&A + Circuit Simulator + Tools
"""

import os
import json
import sqlite3
import hashlib
import secrets
import uuid
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import httpx
from fastapi import FastAPI, HTTPException, Depends, Request, Query, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from jose import jwt, JWTError
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
DATABASE_PATH = os.getenv("DATABASE_PATH", "circuit_learn.db")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Ensure upload directory exists
UPLOAD_DIR.mkdir(exist_ok=True)
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(title="Circuit AI", description="电路学习平台", version="1.0.0")
security = HTTPBearer(auto_error=False)

# Password helpers
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# ============================================================
# DATABASE
# ============================================================

@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with get_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                avatar_url TEXT DEFAULT '',
                bio TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reputation INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT DEFAULT '',
                circuit_data TEXT DEFAULT '',
                image_urls TEXT DEFAULT '[]',
                view_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                image_urls TEXT DEFAULT '[]',
                is_accepted INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                question_id INTEGER,
                answer_id INTEGER,
                value INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE,
                FOREIGN KEY (answer_id) REFERENCES answers(id) ON DELETE CASCADE,
                UNIQUE(user_id, question_id, answer_id)
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT DEFAULT '新对话',
                messages TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_questions_user ON questions(user_id);
            CREATE INDEX IF NOT EXISTS idx_questions_created ON questions(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_answers_question ON answers(question_id);
            CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history(user_id);
        """)
        # Migration: add image_urls columns if they don't exist
        for col, table in [("image_urls", "questions"), ("image_urls", "answers")]:
            try:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT DEFAULT '[]'")
            except sqlite3.OperationalError:
                pass  # Column already exists

init_db()

# ============================================================
# AUTH HELPERS
# ============================================================

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[dict]:
    if not credentials:
        return None
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            return None
        with get_db() as db:
            user = db.execute("SELECT * FROM users WHERE id=?", (int(user_id),)).fetchone()
            return dict(user) if user else None
    except JWTError:
        return None

def require_user(user: Optional[dict] = Depends(get_current_user)) -> dict:
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    return user

# ============================================================
# PYDANTIC MODELS
# ============================================================

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=30)
    email: str = Field(..., max_length=100)
    password: str = Field(..., min_length=6)

class LoginRequest(BaseModel):
    email: str
    password: str

class QuestionCreate(BaseModel):
    title: str = Field(..., min_length=5, max_length=200)
    content: str = Field(..., min_length=10)
    tags: str = ""
    circuit_data: str = ""
    image_urls: str = "[]"  # JSON array of image URLs

class AnswerCreate(BaseModel):
    content: str = Field(..., min_length=5)
    image_urls: str = "[]"  # JSON array of image URLs

class VoteCreate(BaseModel):
    value: int = Field(1, ge=-1, le=1)

class ChatMessage(BaseModel):
    role: str  # user / assistant / system
    content: str

class ChatRequest(BaseModel):
    chat_id: Optional[int] = None
    message: str
    history: List[ChatMessage] = []

class SimulationRequest(BaseModel):
    components: List[Dict[str, Any]]  # [{type, name, value, node1, node2, ...}]
    analysis_type: str = "dc"  # dc / ac / transient
    frequency: Optional[float] = None  # Hz, for AC
    time_start: float = 0.0  # for transient
    time_stop: float = 0.01
    time_step: float = 1e-4

class SpiceRequest(BaseModel):
    netlist: str
    analysis_type: str = "dc"
    frequency: Optional[float] = None
    time_start: float = 0.0
    time_stop: float = 0.01
    time_step: float = 1e-4

class ToolRequest(BaseModel):
    tool: str
    params: Dict[str, float] = {}

# ============================================================
# AUTH ENDPOINTS
# ============================================================

@app.post("/api/auth/register")
def register(req: RegisterRequest):
    with get_db() as db:
        existing = db.execute(
            "SELECT id FROM users WHERE email=? OR username=?",
            (req.email, req.username)
        ).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="邮箱或用户名已存在")
        password_hash = hash_password(req.password)
        cursor = db.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?,?,?)",
            (req.username, req.email, password_hash)
        )
        user_id = cursor.lastrowid
        token = create_access_token({"sub": str(user_id)})
        return {
            "token": token,
            "user": {"id": user_id, "username": req.username, "email": req.email}
        }

@app.post("/api/auth/login")
def login(req: LoginRequest):
    with get_db() as db:
        user = db.execute("SELECT * FROM users WHERE email=?", (req.email,)).fetchone()
        if not user or not verify_password(req.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="邮箱或密码错误")
        token = create_access_token({"sub": str(user["id"])})
        return {
            "token": token,
            "user": {
                "id": user["id"], "username": user["username"],
                "email": user["email"], "avatar_url": user["avatar_url"],
                "bio": user["bio"], "reputation": user["reputation"]
            }
        }

@app.get("/api/auth/me")
def me(user: dict = Depends(require_user)):
    return {"user": {k: user[k] for k in ["id","username","email","avatar_url","bio","reputation","created_at"]}}

# ============================================================
# FILE UPLOAD
# ============================================================

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), user: dict = Depends(require_user)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式，仅支持: {', '.join(ALLOWED_EXTENSIONS)}")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail=f"文件太大，最大 {MAX_UPLOAD_SIZE // (1024*1024)}MB")

    # Generate unique filename
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = UPLOAD_DIR / filename
    filepath.write_bytes(contents)

    url = f"/uploads/{filename}"
    return {"url": url, "filename": file.filename, "size": len(contents)}

@app.post("/api/upload/multi")
async def upload_multiple(files: list[UploadFile] = File(...), user: dict = Depends(require_user)):
    urls = []
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_SIZE:
            continue
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = UPLOAD_DIR / filename
        filepath.write_bytes(contents)
        urls.append({"url": f"/uploads/{filename}", "filename": file.filename, "size": len(contents)})
    return {"files": urls}

# ============================================================
# COMMUNITY ENDPOINTS
# ============================================================

@app.get("/api/questions")
def list_questions(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    tag: str = Query(""),
    search: str = Query(""),
    sort: str = Query("newest")
):
    with get_db() as db:
        where = []
        params = []
        if tag:
            where.append("tags LIKE ?")
            params.append(f"%{tag}%")
        if search:
            where.append("(title LIKE ? OR content LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])

        where_clause = "WHERE " + " AND ".join(where) if where else ""

        order = "q.created_at DESC" if sort == "newest" else \
                "q.view_count DESC" if sort == "views" else \
                "(SELECT COUNT(*) FROM votes WHERE question_id=q.id AND value=1) DESC"

        questions = db.execute(f"""
            SELECT q.*, u.username, u.avatar_url,
                   (SELECT COUNT(*) FROM answers WHERE question_id=q.id) as answer_count,
                   (SELECT COALESCE(SUM(value),0) FROM votes WHERE question_id=q.id) as vote_score
            FROM questions q
            JOIN users u ON q.user_id = u.id
            {where_clause}
            ORDER BY {order}
            LIMIT ? OFFSET ?
        """, params + [per_page, (page-1)*per_page]).fetchall()

        total = db.execute(f"""
            SELECT COUNT(*) FROM questions q {where_clause}
        """, params).fetchone()[0]

        return {
            "questions": [dict(q) for q in questions],
            "total": total,
            "page": page,
            "per_page": per_page
        }

@app.get("/api/questions/{question_id}")
def get_question(question_id: int):
    with get_db() as db:
        db.execute("UPDATE questions SET view_count = view_count + 1 WHERE id=?", (question_id,))
        q = db.execute("""
            SELECT q.*, u.username, u.avatar_url, u.reputation
            FROM questions q JOIN users u ON q.user_id = u.id
            WHERE q.id=?
        """, (question_id,)).fetchone()
        if not q:
            raise HTTPException(status_code=404, detail="问题不存在")

        answers = db.execute("""
            SELECT a.*, u.username, u.avatar_url, u.reputation,
                   (SELECT COALESCE(SUM(value),0) FROM votes WHERE answer_id=a.id) as vote_score
            FROM answers a JOIN users u ON a.user_id = u.id
            WHERE a.question_id=?
            ORDER BY a.is_accepted DESC, vote_score DESC, a.created_at ASC
        """, (question_id,)).fetchall()

        return {"question": dict(q), "answers": [dict(a) for a in answers]}

@app.post("/api/questions")
def create_question(req: QuestionCreate, user: dict = Depends(require_user)):
    with get_db() as db:
        cursor = db.execute(
            "INSERT INTO questions (user_id, title, content, tags, circuit_data, image_urls) VALUES (?,?,?,?,?,?)",
            (user["id"], req.title, req.content, req.tags, req.circuit_data, req.image_urls)
        )
        return {"id": cursor.lastrowid, "title": req.title}

@app.put("/api/questions/{question_id}")
def update_question(question_id: int, req: QuestionCreate, user: dict = Depends(require_user)):
    with get_db() as db:
        q = db.execute("SELECT * FROM questions WHERE id=?", (question_id,)).fetchone()
        if not q:
            raise HTTPException(status_code=404, detail="问题不存在")
        if q["user_id"] != user["id"]:
            raise HTTPException(status_code=403, detail="无权修改")
        db.execute(
            "UPDATE questions SET title=?, content=?, tags=?, circuit_data=?, image_urls=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (req.title, req.content, req.tags, req.circuit_data, req.image_urls, question_id)
        )
        return {"ok": True}

@app.delete("/api/questions/{question_id}")
def delete_question(question_id: int, user: dict = Depends(require_user)):
    with get_db() as db:
        q = db.execute("SELECT * FROM questions WHERE id=?", (question_id,)).fetchone()
        if not q:
            raise HTTPException(status_code=404)
        if q["user_id"] != user["id"]:
            raise HTTPException(status_code=403)
        db.execute("DELETE FROM questions WHERE id=?", (question_id,))
        return {"ok": True}

@app.post("/api/questions/{question_id}/answers")
def create_answer(question_id: int, req: AnswerCreate, user: dict = Depends(require_user)):
    with get_db() as db:
        q = db.execute("SELECT id FROM questions WHERE id=?", (question_id,)).fetchone()
        if not q:
            raise HTTPException(status_code=404, detail="问题不存在")
        cursor = db.execute(
            "INSERT INTO answers (question_id, user_id, content, image_urls) VALUES (?,?,?,?)",
            (question_id, user["id"], req.content, req.image_urls)
        )
        return {"id": cursor.lastrowid}

@app.put("/api/answers/{answer_id}/accept")
def accept_answer(answer_id: int, user: dict = Depends(require_user)):
    with get_db() as db:
        ans = db.execute("""
            SELECT a.*, q.user_id as q_user_id
            FROM answers a JOIN questions q ON a.question_id = q.id
            WHERE a.id=?
        """, (answer_id,)).fetchone()
        if not ans:
            raise HTTPException(status_code=404, detail="答案不存在")
        if ans["q_user_id"] != user["id"]:
            raise HTTPException(status_code=403, detail="只有提问者可以采纳")
        db.execute("UPDATE answers SET is_accepted=0 WHERE question_id=?", (ans["question_id"],))
        db.execute("UPDATE answers SET is_accepted=1 WHERE id=?", (answer_id,))
        return {"ok": True}

@app.post("/api/votes")
def create_vote(req: VoteCreate, user: dict = Depends(require_user)):
    with get_db() as db:
        existing = db.execute(
            "SELECT id FROM votes WHERE user_id=? AND question_id=? AND answer_id IS NULL",
            (user["id"], req.question_id)
        ).fetchone() if hasattr(req, 'question_id') else None
        # Simplified: upsert vote
        db.execute("""
            INSERT INTO votes (user_id, question_id, answer_id, value)
            VALUES (?,?,?,?)
            ON CONFLICT(user_id, question_id, answer_id)
            DO UPDATE SET value=?
        """, (user["id"], getattr(req, 'question_id', None), getattr(req, 'answer_id', None), req.value, req.value))
        return {"ok": True}

# ============================================================
# AI CHAT (DeepSeek)
# ============================================================

SYSTEM_PROMPT = """你是一个专业的电路分析导师，名字叫「电路小AI」。你的职责是帮助学生学习电路知识。

## 你的能力
- 解答电路理论问题（基尔霍夫定律、欧姆定律、戴维南定理、诺顿定理等）
- 分析直流电路和交流电路
- 解释运算放大器、二极管、晶体管电路
- 指导 RC/RL/RLC 暂态分析
- 讲解滤波器、谐振电路
- 帮助理解电路图和各种元件

## 回答风格
- 用中文回答，解释要清晰易懂
- 给初学者解释时要类比生活例子
- 给出解题步骤，逐步推导
- 如果适合，可以画文本电路图（ASCII art）
- 提供 SPICE 网表让用户去模拟器里跑
- 鼓励用户自己思考，不要只给答案

## 重要
- 如果问题涉及危险（高压、市电等），要明确警告
- 遇到不确定的回答，诚实说明
- 用 LaTeX 格式写公式：$V = IR$"""

@app.post("/api/chat")
async def chat(req: ChatRequest, user: dict = Depends(require_user)):
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=503, detail="AI 服务未配置，请设置 DEEPSEEK_API_KEY 环境变量")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Add previous history
    for msg in req.history[-20:]:  # last 20 messages for context
        messages.append({"role": msg.role, "content": msg.content})
    # Add current message
    messages.append({"role": "user", "content": req.message})

    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
        )
        reply = response.choices[0].message.content

        # Save chat history
        with get_db() as db:
            history = req.history + [
                ChatMessage(role="user", content=req.message),
                ChatMessage(role="assistant", content=reply)
            ]
            if req.chat_id:
                db.execute(
                    "UPDATE chat_history SET messages=?, updated_at=CURRENT_TIMESTAMP WHERE id=? AND user_id=?",
                    (json.dumps([m.dict() for m in history], ensure_ascii=False), req.chat_id, user["id"])
                )
            else:
                title = req.message[:50] + ("..." if len(req.message) > 50 else "")
                cursor = db.execute(
                    "INSERT INTO chat_history (user_id, title, messages) VALUES (?,?,?)",
                    (user["id"], title, json.dumps([m.dict() for m in history], ensure_ascii=False))
                )
                req.chat_id = cursor.lastrowid

        return {"reply": reply, "chat_id": req.chat_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 服务错误: {str(e)}")

@app.get("/api/chats")
def list_chats(user: dict = Depends(require_user)):
    with get_db() as db:
        chats = db.execute(
            "SELECT id, title, updated_at FROM chat_history WHERE user_id=? ORDER BY updated_at DESC LIMIT 50",
            (user["id"],)
        ).fetchall()
        return {"chats": [dict(c) for c in chats]}

@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: int, user: dict = Depends(require_user)):
    with get_db() as db:
        chat = db.execute(
            "SELECT * FROM chat_history WHERE id=? AND user_id=?",
            (chat_id, user["id"])
        ).fetchone()
        if not chat:
            raise HTTPException(status_code=404, detail="对话不存在")
        return {"chat": dict(chat)}

@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: int, user: dict = Depends(require_user)):
    with get_db() as db:
        db.execute("DELETE FROM chat_history WHERE id=? AND user_id=?", (chat_id, user["id"]))
        return {"ok": True}

# ============================================================
# CIRCUIT SIMULATION - MNA SOLVER
# ============================================================

class MNASolver:
    """Modified Nodal Analysis solver for DC, AC, and Transient analysis"""

    @staticmethod
    def dc_solve(components: List[Dict]) -> Dict:
        """DC operating point analysis"""
        # Find max node number
        nodes = set()
        vsources = []
        for c in components:
            nodes.add(c.get("node1", 0))
            nodes.add(c.get("node2", 0))
            if c["type"] == "voltage_source_dc":
                vsources.append(c)
        if 0 in nodes:
            nodes.remove(0)  # ground node
        n = len(nodes)
        m = len(vsources)
        node_map = {node: i for i, node in enumerate(sorted(nodes))}

        size = n + m
        M = np.zeros((size, size))
        rhs = np.zeros(size)

        # Stamp components
        for c in components:
            if c["type"] == "resistor":
                n1, n2 = c.get("node1", 0), c.get("node2", 0)
                g = 1.0 / c["value"]
                if n1 != 0:
                    i1 = node_map[n1]
                    M[i1, i1] += g
                    if n2 != 0:
                        M[i1, node_map[n2]] -= g
                if n2 != 0:
                    i2 = node_map[n2]
                    M[i2, i2] += g
                    if n1 != 0:
                        M[i2, node_map[n1]] -= g

            elif c["type"] == "current_source_dc":
                n1, n2 = c.get("node1", 0), c.get("node2", 0)
                val = c["value"]
                if n1 != 0:
                    rhs[node_map[n1]] -= val
                if n2 != 0:
                    rhs[node_map[n2]] += val

        # Stamp voltage sources
        for k, vs in enumerate(vsources):
            n1, n2 = vs.get("node1", 0), vs.get("node2", 0)
            # B matrix
            if n1 != 0:
                M[node_map[n1], n + k] = 1
                M[n + k, node_map[n1]] = 1
            if n2 != 0:
                M[node_map[n2], n + k] = -1
                M[n + k, node_map[n2]] = -1
            rhs[n + k] = vs["value"]

        try:
            x = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return {"error": "电路矩阵奇异，请检查电路连接", "type": "dc"}

        # Extract results
        node_voltages = {}
        for node, idx in node_map.items():
            node_voltages[f"V({node})"] = round(float(x[idx]), 6)

        # Calculate branch currents
        branch_currents = []
        for c in components:
            n1, n2 = c.get("node1", 0), c.get("node2", 0)
            v1 = node_voltages.get(f"V({n1})", 0) if n1 != 0 else 0
            v2 = node_voltages.get(f"V({n2})", 0) if n2 != 0 else 0
            v_diff = v1 - v2
            if c["type"] == "resistor":
                current = v_diff / c["value"]
                branch_currents.append({
                    "component": c.get("name", "R"),
                    "voltage": round(v_diff, 6),
                    "current": round(current, 6),
                    "power": round(abs(v_diff * current), 6)
                })

        return {
            "type": "dc",
            "node_voltages": node_voltages,
            "branch_currents": branch_currents,
            "matrix_size": size
        }

    @staticmethod
    def ac_solve(components: List[Dict], frequency: float) -> Dict:
        """AC frequency domain analysis"""
        omega = 2 * np.pi * frequency
        nodes = set()
        vsources = []
        for c in components:
            nodes.add(c.get("node1", 0))
            nodes.add(c.get("node2", 0))
            if c["type"] in ("voltage_source_dc", "voltage_source_ac"):
                vsources.append(c)
        if 0 in nodes:
            nodes.remove(0)
        n = len(nodes)
        m = len(vsources)
        node_map = {node: i for i, node in enumerate(sorted(nodes))}

        size = n + m
        M = np.zeros((size, size), dtype=complex)
        rhs = np.zeros(size, dtype=complex)

        for c in components:
            n1, n2 = c.get("node1", 0), c.get("node2", 0)
            i1 = node_map.get(n1) if n1 != 0 else None
            i2 = node_map.get(n2) if n2 != 0 else None

            if c["type"] == "resistor":
                g = 1.0 / c["value"]
                if i1 is not None:
                    M[i1, i1] += g
                    if i2 is not None: M[i1, i2] -= g
                if i2 is not None:
                    M[i2, i2] += g
                    if i1 is not None: M[i2, i1] -= g

            elif c["type"] == "capacitor":
                g = 1j * omega * c["value"]
                if i1 is not None:
                    M[i1, i1] += g
                    if i2 is not None: M[i1, i2] -= g
                if i2 is not None:
                    M[i2, i2] += g
                    if i1 is not None: M[i2, i1] -= g

            elif c["type"] == "inductor":
                g = 1.0 / (1j * omega * c["value"])
                if i1 is not None:
                    M[i1, i1] += g
                    if i2 is not None: M[i1, i2] -= g
                if i2 is not None:
                    M[i2, i2] += g
                    if i1 is not None: M[i2, i1] -= g

            elif c["type"] == "current_source_ac":
                val = c["value"] * (np.cos(c.get("phase", 0)) + 1j * np.sin(c.get("phase", 0)))
                if i1 is not None: rhs[i1] -= val
                if i2 is not None: rhs[i2] += val
            elif c["type"] == "current_source_dc":
                if i1 is not None: rhs[i1] -= c["value"]
                if i2 is not None: rhs[i2] += c["value"]

        # Stamp voltage sources
        for k, vs in enumerate(vsources):
            n1, n2 = vs.get("node1", 0), vs.get("node2", 0)
            val = vs["value"]
            if vs["type"] == "voltage_source_ac":
                val = val * (np.cos(vs.get("phase", 0)) + 1j * np.sin(vs.get("phase", 0)))
            if n1 != 0:
                M[node_map[n1], n + k] = 1
                M[n + k, node_map[n1]] = 1
            if n2 != 0:
                M[node_map[n2], n + k] = -1
                M[n + k, node_map[n2]] = -1
            rhs[n + k] = val

        try:
            x = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return {"error": "电路矩阵奇异", "type": "ac"}

        node_voltages = {}
        for node, idx in node_map.items():
            v = x[idx]
            node_voltages[f"V({node})"] = {
                "magnitude": round(abs(v), 6),
                "phase_deg": round(np.angle(v, deg=True), 2),
                "real": round(v.real, 6),
                "imag": round(v.imag, 6)
            }

        return {
            "type": "ac",
            "frequency": frequency,
            "omega": round(omega, 2),
            "node_voltages": node_voltages
        }

    @staticmethod
    def transient_solve(components: List[Dict], t_start: float, t_stop: float, t_step: float) -> Dict:
        """Transient analysis using Backward Euler"""
        t_points = np.arange(t_start, t_stop + t_step/2, t_step)
        n_points = len(t_points)
        if n_points > 5000:
            return {"error": "时间点太多（>5000），请增大步长或缩小时间范围", "type": "transient"}

        nodes = set()
        vsources = []
        caps = []
        inds = []
        for c in components:
            nodes.add(c.get("node1", 0))
            nodes.add(c.get("node2", 0))
            if c["type"] in ("voltage_source_dc", "voltage_source_ac", "voltage_source_pulse"):
                vsources.append(c)
            elif c["type"] == "capacitor":
                caps.append(c)
            elif c["type"] == "inductor":
                inds.append(c)
        if 0 in nodes:
            nodes.remove(0)
        n = len(nodes)
        node_map = {node: i for i, node in enumerate(sorted(nodes))}

        results = {"time": [], "node_voltages": {f"V({node})": [] for node in sorted(nodes)}}
        v_prev = {node: 0.0 for node in nodes}  # initial capacitor voltages
        i_prev = {c.get("name", f"L{idx}"): 0.0 for idx, c in enumerate(inds)}  # initial inductor currents

        for ti, t in enumerate(t_points):
            m = len(vsources)
            size = n + m
            M = np.zeros((size, size))
            rhs = np.zeros(size)

            # Stamp resistors
            for c in components:
                if c["type"] != "resistor":
                    continue
                n1, n2 = c.get("node1", 0), c.get("node2", 0)
                g = 1.0 / c["value"]
                if n1 != 0:
                    M[node_map[n1], node_map[n1]] += g
                    if n2 != 0:
                        M[node_map[n1], node_map[n2]] -= g
                if n2 != 0:
                    M[node_map[n2], node_map[n2]] += g
                    if n1 != 0:
                        M[node_map[n2], node_map[n1]] -= g

            # Stamp capacitors (Backward Euler: companion model = conductance + current source)
            for c in caps:
                n1, n2 = c.get("node1", 0), c.get("node2", 0)
                geq = c["value"] / t_step  # C/dt
                ieq = geq * v_prev.get(n1, 0) - geq * v_prev.get(n2, 0)
                if n1 != 0:
                    M[node_map[n1], node_map[n1]] += geq
                    if n2 != 0:
                        M[node_map[n1], node_map[n2]] -= geq
                    rhs[node_map[n1]] += ieq
                if n2 != 0:
                    M[node_map[n2], node_map[n2]] += geq
                    if n1 != 0:
                        M[node_map[n2], node_map[n1]] -= geq
                    rhs[node_map[n2]] -= ieq

            # Stamp inductors (Backward Euler: companion model)
            for idx, c in enumerate(inds):
                n1, n2 = c.get("node1", 0), c.get("node2", 0)
                req = c["value"] / t_step  # L/dt
                veq = req * i_prev.get(c.get("name", f"L{idx}"), 0)
                # Treat as voltage source
                vsources.append({"node1": n1, "node2": n2, "value": veq, "type": "inductor_companion"})

            # Stamp voltage sources (including inductor companion models)
            for k, vs in enumerate(vsources):
                n1, n2 = vs.get("node1", 0), vs.get("node2", 0)
                val = vs["value"]
                if vs["type"] == "voltage_source_pulse":
                    period = vs.get("period", 0.001)
                    duty = vs.get("duty", 0.5)
                    val = vs["value"] if (t % period) < (duty * period) else vs.get("v_low", 0)
                elif vs["type"] == "voltage_source_ac":
                    freq = vs.get("frequency", 1000) if "frequency" in vs else vs.get("freq", 1000)
                    val = vs["value"] * np.sin(2 * np.pi * freq * t + vs.get("phase", 0))

                if n1 != 0:
                    M[node_map[n1], n + k] = 1
                    M[n + k, node_map[n1]] = 1
                if n2 != 0:
                    M[node_map[n2], n + k] = -1
                    M[n + k, node_map[n2]] = -1
                rhs[n + k] = val

            try:
                x = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                continue

            results["time"].append(round(t, 10))
            for node in sorted(nodes):
                v = float(x[node_map[node]])
                results["node_voltages"][f"V({node})"].append(round(v, 6))
                v_prev[node] = v

            # Update inductor currents
            for idx, c in enumerate(inds):
                k = len(vsources) - len(inds) + idx
                if n + k < len(x):
                    i_prev[c.get("name", f"L{idx}")] = float(x[n + k])

        return {"type": "transient", "data": results, "points": len(t_points)}

# ============================================================
# SPICE NETLIST PARSER
# ============================================================

def parse_spice(netlist: str) -> List[Dict]:
    """Parse a simple SPICE netlist into component dicts"""
    components = []
    for line in netlist.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("*") or line.startswith("."):
            # Skip comments and control lines for now
            continue
        parts = line.split()
        if not parts:
            continue
        first = parts[0][0].upper()
        name = parts[0]
        try:
            if first == "R" and len(parts) >= 4:
                components.append({
                    "type": "resistor", "name": name,
                    "node1": int(parts[1]), "node2": int(parts[2]), "value": float(parts[3])
                })
            elif first == "C" and len(parts) >= 4:
                components.append({
                    "type": "capacitor", "name": name,
                    "node1": int(parts[1]), "node2": int(parts[2]), "value": float(parts[3])
                })
            elif first == "L" and len(parts) >= 4:
                components.append({
                    "type": "inductor", "name": name,
                    "node1": int(parts[1]), "node2": int(parts[2]), "value": float(parts[3])
                })
            elif first == "V" and len(parts) >= 4:
                comp_type = "voltage_source_dc"
                if len(parts) >= 5 and parts[4].upper() == "AC":
                    comp_type = "voltage_source_ac"
                components.append({
                    "type": comp_type, "name": name,
                    "node1": int(parts[1]), "node2": int(parts[2]), "value": float(parts[3])
                })
            elif first == "I" and len(parts) >= 4:
                components.append({
                    "type": "current_source_dc", "name": name,
                    "node1": int(parts[1]), "node2": int(parts[2]), "value": float(parts[3])
                })
        except (ValueError, IndexError):
            continue
    return components

# ============================================================
# CIRCUIT SIMULATION ENDPOINTS
# ============================================================

@app.post("/api/simulate")
def simulate(req: SimulationRequest):
    solver = MNASolver()
    components = req.components
    if req.analysis_type == "ac" and req.frequency:
        return solver.ac_solve(components, req.frequency)
    elif req.analysis_type == "transient":
        return solver.transient_solve(components, req.time_start, req.time_stop, req.time_step)
    else:
        return solver.dc_solve(components)

@app.post("/api/simulate/spice")
def simulate_spice(req: SpiceRequest):
    components = parse_spice(req.netlist)
    if not components:
        raise HTTPException(status_code=400, detail="无法解析网表，请检查格式")
    solver = MNASolver()
    if req.analysis_type == "ac" and req.frequency:
        return solver.ac_solve(components, req.frequency)
    elif req.analysis_type == "transient":
        return solver.transient_solve(components, req.time_start, req.time_stop, req.time_step)
    else:
        return solver.dc_solve(components)

# ============================================================
# CIRCUIT TEMPLATES
# ============================================================

CIRCUIT_TEMPLATES = {
    "voltage_divider": {
        "name": "电阻分压器",
        "description": "两个电阻串联，输出电压 = Vin × R2/(R1+R2)",
        "components": [
            {"type": "voltage_source_dc", "name": "Vin", "node1": 1, "node2": 0, "value": 10},
            {"type": "resistor", "name": "R1", "node1": 1, "node2": 2, "value": 1000},
            {"type": "resistor", "name": "R2", "node1": 2, "node2": 0, "value": 1000},
        ],
        "params": {"Vin": 10, "R1": 1000, "R2": 1000},
        "output_node": 2
    },
    "rc_lowpass": {
        "name": "RC 低通滤波器",
        "description": "截止频率 fc = 1/(2πRC)",
        "components": [
            {"type": "voltage_source_ac", "name": "Vin", "node1": 1, "node2": 0, "value": 1},
            {"type": "resistor", "name": "R1", "node1": 1, "node2": 2, "value": 1000},
            {"type": "capacitor", "name": "C1", "node1": 2, "node2": 0, "value": 1e-6},
        ],
        "params": {"Vin_ac": 1, "R1": 1000, "C1": 1e-6},
        "output_node": 2
    },
    "rc_highpass": {
        "name": "RC 高通滤波器",
        "description": "截止频率 fc = 1/(2πRC)",
        "components": [
            {"type": "voltage_source_ac", "name": "Vin", "node1": 1, "node2": 0, "value": 1},
            {"type": "capacitor", "name": "C1", "node1": 1, "node2": 2, "value": 1e-6},
            {"type": "resistor", "name": "R1", "node1": 2, "node2": 0, "value": 1000},
        ],
        "params": {"Vin_ac": 1, "C1": 1e-6, "R1": 1000},
        "output_node": 2
    },
    "rlc_series": {
        "name": "RLC 串联谐振",
        "description": "谐振频率 f0 = 1/(2π√(LC))",
        "components": [
            {"type": "voltage_source_ac", "name": "Vin", "node1": 1, "node2": 0, "value": 1},
            {"type": "resistor", "name": "R1", "node1": 1, "node2": 2, "value": 100},
            {"type": "inductor", "name": "L1", "node1": 2, "node2": 3, "value": 0.01},
            {"type": "capacitor", "name": "C1", "node1": 3, "node2": 0, "value": 1e-6},
        ],
        "params": {"Vin_ac": 1, "R1": 100, "L1": 0.01, "C1": 1e-6},
        "output_node": 3
    },
    "wheatstone_bridge": {
        "name": "惠斯通电桥",
        "description": "平衡条件：R1/R2 = R3/R4",
        "components": [
            {"type": "voltage_source_dc", "name": "Vin", "node1": 1, "node2": 0, "value": 5},
            {"type": "resistor", "name": "R1", "node1": 1, "node2": 2, "value": 1000},
            {"type": "resistor", "name": "R2", "node1": 2, "node2": 0, "value": 1000},
            {"type": "resistor", "name": "R3", "node1": 1, "node2": 3, "value": 1000},
            {"type": "resistor", "name": "R4", "node1": 3, "node2": 0, "value": 1100},
        ],
        "params": {"Vin": 5, "R1": 1000, "R2": 1000, "R3": 1000, "R4": 1100},
        "output_nodes": [2, 3]
    },
    "opamp_inverting": {
        "name": "反相放大器",
        "description": "增益 Av = -Rf/Rin，Vout = -Vin × Rf/Rin",
        "components": [
            {"type": "voltage_source_dc", "name": "Vin", "node1": 1, "node2": 0, "value": 1},
            {"type": "resistor", "name": "Rin", "node1": 1, "node2": 2, "value": 1000},
            {"type": "resistor", "name": "Rf", "node1": 2, "node2": 3, "value": 10000},
            {"type": "resistor", "name": "Rload", "node1": 3, "node2": 0, "value": 10000},
        ],
        "params": {"Vin": 1, "Rin": 1000, "Rf": 10000},
        "output_node": 3,
        "note": "理想运放需手动用 VCVS 替代，此处为简化版"
    },
    "transistor_ce": {
        "name": "共射极放大器",
        "description": "基本 BJT 共射极放大电路",
        "components": [
            {"type": "voltage_source_dc", "name": "Vcc", "node1": 4, "node2": 0, "value": 12},
            {"type": "voltage_source_dc", "name": "Vin", "node1": 1, "node2": 0, "value": 0.7},
            {"type": "resistor", "name": "Rb", "node1": 1, "node2": 2, "value": 100000},
            {"type": "resistor", "name": "Rc", "node1": 4, "node2": 3, "value": 4700},
            {"type": "resistor", "name": "Re", "node1": 2, "node2": 0, "value": 1000},
        ],
        "params": {"Vcc": 12, "Vin": 0.7, "Rb": 100000, "Rc": 4700, "Re": 1000},
        "output_node": 3,
        "note": "简化版，实际 BJT 模型需非线性元件"
    }
}

@app.get("/api/templates")
def list_templates():
    return {
        "templates": [
            {"id": k, "name": v["name"], "description": v["description"], "params": v.get("params", {})}
            for k, v in CIRCUIT_TEMPLATES.items()
        ]
    }

@app.get("/api/templates/{template_id}")
def get_template(template_id: str):
    if template_id not in CIRCUIT_TEMPLATES:
        raise HTTPException(status_code=404, detail="模板不存在")
    return CIRCUIT_TEMPLATES[template_id]

# ============================================================
# CIRCUIT TOOLS
# ============================================================

@app.post("/api/tools")
def run_tool(req: ToolRequest):
    p = req.params
    if req.tool == "ohms_law":
        # V=IR, calculate the missing one
        if "V" in p and "I" in p:
            return {"result": {"R": p["V"] / p["I"]}, "formula": "R = V/I"}
        elif "V" in p and "R" in p:
            return {"result": {"I": p["V"] / p["R"]}, "formula": "I = V/R"}
        elif "I" in p and "R" in p:
            return {"result": {"V": p["I"] * p["R"]}, "formula": "V = IR"}
        else:
            raise HTTPException(status_code=400, detail="请提供 V, I, R 中的两个")

    elif req.tool == "voltage_divider":
        vin = p.get("Vin", 5)
        r1 = p.get("R1", 1000)
        r2 = p.get("R2", 1000)
        vout = vin * r2 / (r1 + r2)
        return {
            "result": {"Vout": round(vout, 4), "I": round(vin / (r1 + r2), 6)},
            "formula": "Vout = Vin × R2/(R1+R2)"
        }

    elif req.tool == "parallel_resistors":
        values = [v for k, v in p.items() if k.startswith("R")]
        if not values:
            raise HTTPException(status_code=400, detail="请提供至少一个电阻值")
        req_val = 1 / sum(1/v for v in values if v > 0)
        return {
            "result": {"Req": round(req_val, 4)},
            "formula": "1/Req = 1/R1 + 1/R2 + ..."
        }

    elif req.tool == "series_resistors":
        values = [v for k, v in p.items() if k.startswith("R")]
        req_val = sum(values)
        return {
            "result": {"Req": round(req_val, 4)},
            "formula": "Req = R1 + R2 + ..."
        }

    elif req.tool == "rc_time_constant":
        r = p.get("R", 1000)
        c = p.get("C", 1e-6)
        tau = r * c
        return {
            "result": {
                "tau": tau,
                "tau_ms": round(tau * 1000, 4),
                "fc": round(1 / (2 * np.pi * tau), 4),
                "charge_5tau": f"5τ = {round(5*tau*1000,2)}ms (充电至99.3%)"
            },
            "formula": "τ = RC, fc = 1/(2πτ)"
        }

    elif req.tool == "lc_resonance":
        l_val = p.get("L", 0.01)
        c_val = p.get("C", 1e-6)
        f0 = 1 / (2 * np.pi * np.sqrt(l_val * c_val))
        return {
            "result": {
                "f0": round(f0, 4),
                "f0_kHz": round(f0/1000, 4),
                "omega0": round(2*np.pi*f0, 4)
            },
            "formula": "f0 = 1/(2π√(LC))"
        }

    elif req.tool == "resistor_color":
        # 4-band resistor color code calculator
        color_map = {
            "black": 0, "brown": 1, "red": 2, "orange": 3, "yellow": 4,
            "green": 5, "blue": 6, "violet": 7, "gray": 8, "white": 9
        }
        if "bands" in p:
            return {"result": {"message": "色环计算器：bands 参数为颜色数组，如 [\"brown\",\"black\",\"red\",\"gold\"]"}}
        return {"result": dict(color_map), "info": "可用颜色: " + ", ".join(color_map.keys())}

    elif req.tool == "power":
        v = p.get("V", 0)
        i = p.get("I", 0)
        r = p.get("R", 0)
        results = {}
        if v and i:
            results["P_VI"] = v * i
        if v and r:
            results["P_V2R"] = v * v / r
        if i and r:
            results["P_I2R"] = i * i * r
        return {"result": results, "formula": "P = VI = V²/R = I²R"}

    else:
        available = ["ohms_law", "voltage_divider", "parallel_resistors", "series_resistors",
                     "rc_time_constant", "lc_resonance", "resistor_color", "power"]
        return {"error": f"未知工具，可用: {', '.join(available)}"}

# ============================================================
# DIAGRAM SVG GENERATION
# ============================================================

@app.post("/api/diagram")
def generate_diagram(components: List[Dict[str, Any]] = None, template_id: str = None):
    """Generate SVG circuit diagram"""
    if template_id and template_id in CIRCUIT_TEMPLATES:
        comps = CIRCUIT_TEMPLATES[template_id]["components"]
    elif components:
        comps = components
    else:
        raise HTTPException(status_code=400, detail="请提供 components 或 template_id")

    # Generate simple SVG
    svg_parts = []
    svg_parts.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400" style="background:#0a0a0a;">')
    svg_parts.append('<defs><style>text{fill:#00ff41;font-family:monospace;font-size:14px} line{stroke:#00ff41;stroke-width:2} .comp{stroke:#00ffff;stroke-width:2;fill:none}</style></defs>')

    # Calculate node positions (simple grid layout)
    node_positions = {}
    nodes_seen = set()
    for c in comps:
        for n in [c.get("node1", 0), c.get("node2", 0)]:
            nodes_seen.add(n)
    sorted_nodes = sorted(nodes_seen)
    cols = min(4, len(sorted_nodes))
    for i, n in enumerate(sorted_nodes):
        col = i % cols
        row = i // cols
        x = 100 + col * 150
        y = 80 + row * 120
        node_positions[n] = (x, y)

    # Draw components
    for c in comps:
        n1 = c.get("node1", 0)
        n2 = c.get("node2", 0)
        if n1 not in node_positions or n2 not in node_positions:
            continue
        x1, y1 = node_positions[n1]
        x2, y2 = node_positions[n2]
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        name = c.get("name", "")
        value = c.get("value", "")

        if c["type"] == "resistor":
            # Draw zigzag
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            if length < 1: length = 1
            ux, uy = dx / length, dy / length
            nx, ny = -uy, ux  # perpendicular
            zigzag = f'M {x1} {y1} '
            segments = 6
            for i in range(1, segments):
                frac = i / segments
                px = x1 + dx * frac + (nx * 12 if i % 2 == 0 else -nx * 12)
                py = y1 + dy * frac + (ny * 12 if i % 2 == 0 else -ny * 12)
                zigzag += f'L {px} {py} '
            zigzag += f'L {x2} {y2}'
            svg_parts.append(f'<path d="{zigzag}" class="comp"/>')
        elif c["type"] == "capacitor":
            svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{mid_x-8}" y2="{y1}" class="comp"/>')
            svg_parts.append(f'<line x1="{mid_x-8}" y1="{y1-15}" x2="{mid_x-8}" y2="{y1+15}" class="comp"/>')
            svg_parts.append(f'<line x1="{mid_x+8}" y1="{y1-15}" x2="{mid_x+8}" y2="{y1+15}" class="comp"/>')
            svg_parts.append(f'<line x1="{mid_x+8}" y1="{y1}" x2="{x2}" y2="{y2}" class="comp"/>')
        elif c["type"] == "inductor":
            # Draw coil
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            ux, uy = dx / length, dy / length if length > 0 else (1, 0)
            coil = f'M {x1} {y1} '
            for i in range(1, 8):
                frac = i / 8
                px = x1 + dx * frac + (np.sin(frac * 6 * np.pi) * 12)
                py = y1 + dy * frac + (np.cos(frac * 6 * np.pi) * 12)
                coil += f'L {px} {py} '
            coil += f'L {x2} {y2}'
            svg_parts.append(f'<path d="{coil}" class="comp"/>')
        elif "voltage_source" in c["type"]:
            svg_parts.append(f'<circle cx="{mid_x}" cy="{mid_y}" r="18" class="comp"/>')
            sign = "+" if "ac" not in c["type"] else "~"
            svg_parts.append(f'<text x="{mid_x-3}" y="{mid_y-5}" text-anchor="end">+</text>')
            svg_parts.append(f'<text x="{mid_x+3}" y="{mid_y+10}" text-anchor="start">-</text>')
            svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{mid_x-18}" y2="{mid_y}" class="comp"/>')
            svg_parts.append(f'<line x1="{mid_x+18}" y1="{mid_y}" x2="{x2}" y2="{y2}" class="comp"/>')
        elif "current_source" in c["type"]:
            svg_parts.append(f'<circle cx="{mid_x}" cy="{mid_y}" r="18" class="comp"/>')
            svg_parts.append(f'<text x="{mid_x}" y="{mid_y+5}" text-anchor="middle">I</text>')
            svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{mid_x-18}" y2="{mid_y}" class="comp"/>')
            svg_parts.append(f'<line x1="{mid_x+18}" y1="{mid_y}" x2="{x2}" y2="{y2}" class="comp"/>')
        else:
            # Generic: just a line
            svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="comp"/>')

        # Label
        if name:
            svg_parts.append(f'<text x="{mid_x}" y="{mid_y-25}" text-anchor="middle" fill="#00ffff" font-size="12">{name}={value}</text>')

    # Draw nodes
    for n, (x, y) in node_positions.items():
        color = "#00ff41" if n == 0 else "#ff6600"
        label = "GND" if n == 0 else str(n)
        if n == 0:
            svg_parts.append(f'<line x1="{x-15}" y1="{y}" x2="{x+15}" y2="{y}" stroke="#00ff41" stroke-width="2"/>')
            svg_parts.append(f'<line x1="{x-10}" y1="{y+5}" x2="{x+10}" y2="{y+5}" stroke="#00ff41" stroke-width="2"/>')
            svg_parts.append(f'<line x1="{x-5}" y1="{y+10}" x2="{x+5}" y2="{y+10}" stroke="#00ff41" stroke-width="2"/>')
        svg_parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
        svg_parts.append(f'<text x="{x}" y="{y-12}" text-anchor="middle" fill="{color}" font-size="12">{label}</text>')

    svg_parts.append('</svg>')
    svg = "\n".join(svg_parts)
    return {"svg": svg}

# ============================================================
# STATIC FILES
# ============================================================

@app.get("/")
def index():
    return FileResponse("static/index.html")

# Mount static after defining all routes
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  Circuit AI Learning Platform")
    print("  - AI Tutor + Community + Simulator + Tools")
    print("=" * 60)
    print(f"  API:            http://localhost:8000")
    print(f"  Docs:           http://localhost:8000/docs")
    print(f"  AI Chat:        {'DeepSeek ready' if DEEPSEEK_API_KEY else 'No API key set'}")
    print(f"  Community:      Q&A forum ready")
    print(f"  Simulator:      DC/AC/Transient ready")
    print(f"  Tools:          8 circuit calculators")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
