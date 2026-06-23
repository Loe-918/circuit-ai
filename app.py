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

            CREATE TABLE IF NOT EXISTS quiz_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                icon TEXT DEFAULT '📚'
            );

            CREATE TABLE IF NOT EXISTS quiz_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                option_a TEXT NOT NULL,
                option_b TEXT NOT NULL,
                option_c TEXT NOT NULL,
                option_d TEXT NOT NULL,
                correct_answer TEXT NOT NULL CHECK(correct_answer IN ('A','B','C','D')),
                explanation TEXT DEFAULT '',
                difficulty INTEGER DEFAULT 1 CHECK(difficulty BETWEEN 1 AND 3),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (category_id) REFERENCES quiz_categories(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS quiz_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                category_id INTEGER,
                score INTEGER DEFAULT 0,
                total INTEGER DEFAULT 0,
                answers_json TEXT DEFAULT '[]',
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_questions_user ON questions(user_id);
            CREATE INDEX IF NOT EXISTS idx_questions_created ON questions(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_answers_question ON answers(question_id);
            CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history(user_id);
            CREATE INDEX IF NOT EXISTS idx_quiz_q_category ON quiz_questions(category_id);
            CREATE INDEX IF NOT EXISTS idx_quiz_results_user ON quiz_results(user_id);
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

def parse_spice(netlist: str) -> dict:
    """Parse SPICE netlist into components + analysis directives.
    Returns {'components': [...], 'analysis': {...}, 'warnings': [...]}"""
    import re
    components = []
    analysis = {"type": "dc"}
    warnings = []

    for line in netlist.strip().split("\n"):
        line_orig = line
        line = line.strip()
        if not line or line.startswith("*"):
            continue

        # Handle continuations (lines starting with +)
        # Already unrolled for simplicity

        parts = line.split()
        if not parts:
            continue

        first = parts[0].upper()
        name = parts[0]

        # ─── Control lines ───
        if first == ".OP":
            analysis["type"] = "dc"
        elif first == ".DC":
            analysis["type"] = "dc_sweep"
            analysis["source"] = parts[1] if len(parts) > 1 else ""
            try:
                analysis["start"] = float(parts[2]) if len(parts) > 2 else 0
                analysis["stop"] = float(parts[3]) if len(parts) > 3 else 5
                analysis["step"] = float(parts[4]) if len(parts) > 4 else 0.1
            except ValueError:
                pass
        elif first == ".AC":
            analysis["type"] = "ac"
            # .AC DEC|OCT|LIN N fstart fstop
            mode = parts[1].upper() if len(parts) > 1 else "DEC"
            try:
                analysis["ac_points"] = int(parts[2]) if len(parts) > 2 else 10
                analysis["ac_start"] = float(parts[3]) if len(parts) > 3 else 1
                analysis["ac_stop"] = float(parts[4]) if len(parts) > 4 else 1000
                analysis["ac_mode"] = mode
            except ValueError:
                pass
        elif first == ".TRAN":
            analysis["type"] = "transient"
            try:
                analysis["tran_step"] = float(parts[1]) if len(parts) > 1 else 1e-6
                analysis["tran_stop"] = float(parts[2]) if len(parts) > 2 else 0.001
                analysis["tran_start"] = float(parts[3]) if len(parts) > 3 else 0
            except ValueError:
                pass
        elif first == ".IC":
            analysis["ic"] = {}
            for p in parts[1:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    try: analysis["ic"][k] = float(v)
                    except ValueError: pass

        # ─── Components ───
        elif first == "R" and len(parts) >= 4:
            components.append({
                "type": "resistor", "name": name,
                "node1": int(parts[1]), "node2": int(parts[2]), "value": scale_spice_value(parts[3])
            })
        elif first == "C" and len(parts) >= 4:
            val = scale_spice_value(parts[3])
            ic = None
            if len(parts) >= 5 and parts[4].upper().startswith("IC="):
                try: ic = float(parts[4].split("=",1)[1])
                except ValueError: pass
            components.append({
                "type": "capacitor", "name": name,
                "node1": int(parts[1]), "node2": int(parts[2]), "value": val, "ic": ic
            })
        elif first == "L" and len(parts) >= 4:
            val = scale_spice_value(parts[3])
            ic = None
            if len(parts) >= 5 and parts[4].upper().startswith("IC="):
                try: ic = float(parts[4].split("=",1)[1])
                except ValueError: pass
            components.append({
                "type": "inductor", "name": name,
                "node1": int(parts[1]), "node2": int(parts[2]), "value": val, "ic": ic
            })
        elif first == "D" and len(parts) >= 4:
            components.append({
                "type": "diode", "name": name, "model": parts[3] if len(parts) > 3 else "D1N4148",
                "node1": int(parts[1]), "node2": int(parts[2])
            })
            warnings.append(f"{name}: 二极管为简化模型（理想二极管 + 0.7V 正向压降）")
        elif first == "Q" and len(parts) >= 5:
            components.append({
                "type": "bjt_npn", "name": name,
                "node_c": int(parts[1]), "node_b": int(parts[2]), "node_e": int(parts[3]),
                "model": parts[4] if len(parts) > 4 else "2N2222"
            })
            warnings.append(f"{name}: BJT 为简化小信号模型")
        elif first == "V" and len(parts) >= 4:
            comp = {"type": "voltage_source_dc", "name": name,
                    "node1": int(parts[1]), "node2": int(parts[2]), "value": 0}
            # Check for DC value, AC magnitude, SIN/PULSE
            dc_val = None
            ac_mag = None
            sin_params = None
            pulse_params = None
            rest = parts[3:]
            i = 0
            while i < len(rest):
                token = rest[i].upper()
                if token in ("DC",):
                    if i + 1 < len(rest):
                        try: dc_val = float(rest[i+1]); i += 2; continue
                        except ValueError: pass
                elif token == "AC":
                    if i + 1 < len(rest):
                        try: ac_mag = float(rest[i+1]); i += 2; continue
                        except ValueError: pass
                elif token == "SIN":
                    sin_params = []
                    for j in range(i+1, min(i+7, len(rest))):
                        try: sin_params.append(float(rest[j]))
                        except ValueError: break
                    break
                elif token == "PULSE":
                    pulse_params = []
                    for j in range(i+1, min(i+8, len(rest))):
                        try: pulse_params.append(float(rest[j]))
                        except ValueError: break
                    break
                else:
                    try:
                        dc_val = float(rest[i])
                    except ValueError:
                        pass
                i += 1

            if dc_val is not None:
                comp["value"] = dc_val
            if ac_mag is not None:
                comp["type"] = "voltage_source_ac"
                comp["value"] = ac_mag
            elif sin_params is not None and len(sin_params) >= 2:
                comp["type"] = "voltage_source_sin"
                comp["offset"] = sin_params[0] if len(sin_params) > 0 else 0
                comp["amplitude"] = sin_params[1] if len(sin_params) > 1 else 0
                comp["frequency"] = sin_params[2] if len(sin_params) > 2 else 1000
            elif pulse_params is not None and len(pulse_params) >= 2:
                comp["type"] = "voltage_source_pulse"
                comp["v_low"] = pulse_params[0] if len(pulse_params) > 0 else 0
                comp["v_high"] = pulse_params[1] if len(pulse_params) > 1 else 5
                comp["tdelay"] = pulse_params[2] if len(pulse_params) > 2 else 0
                comp["trise"] = pulse_params[3] if len(pulse_params) > 3 else 1e-9
                comp["tfall"] = pulse_params[4] if len(pulse_params) > 4 else 1e-9
                comp["tpw"] = pulse_params[5] if len(pulse_params) > 5 else 0.001
                comp["tper"] = pulse_params[6] if len(pulse_params) > 6 else 0.002

            components.append(comp)

        elif first == "I" and len(parts) >= 4:
            dc_val = None
            try: dc_val = float(parts[3])
            except ValueError: pass
            if dc_val is None:
                try: dc_val = scale_spice_value(parts[3])
                except (ValueError, TypeError): dc_val = 0
            components.append({
                "type": "current_source_dc", "name": name,
                "node1": int(parts[1]), "node2": int(parts[2]), "value": dc_val or 0
            })
        elif first == "K" and len(parts) >= 4:
            # Mutual inductance / transformer
            components.append({
                "type": "mutual_inductance", "name": name,
                "l1": parts[1], "l2": parts[2], "k": float(parts[3]) if len(parts) > 3 else 1.0
            })
        elif first == "X":
            # Subcircuit call — simplified
            warnings.append(f"{name}: 子电路调用已跳过（简化解析器不支持）")
        elif first == ".":
            # Other control lines
            pass
        else:
            pass  # Unknown line

    return {"components": components, "analysis": analysis, "warnings": warnings}


def scale_spice_value(token: str) -> float:
    """Parse SPICE-style values: 1k, 1MEG, 1u, 1n, 1p, etc."""
    import re
    token = token.upper().strip()
    # Remove trailing unit chars
    unit_map = {
        "F": 1, "H": 1, "OHM": 1, "OHMS": 1, "Ω": 1,
        "V": 1, "A": 1, "HZ": 1,
    }
    scale_map = {
        "T": 1e12, "G": 1e9, "MEG": 1e6, "M": 1e-3,  # MEG is mega, M alone is milli
        "K": 1e3, "U": 1e-6, "N": 1e-9, "P": 1e-12, "F": 1e-15,
        "MIL": 25.4e-6,
        "DB": 1,  # not supported directly
    }
    # Try to extract number + suffix
    match = re.match(r'^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([A-Z]*)$', token)
    if match:
        num = float(match.group(1))
        suffix = match.group(2)
        if not suffix:
            return num
        # Check if it's MEG first
        if suffix == "MEG":
            return num * 1e6
        elif suffix == "M":
            # M is ambiguous: in resistors it's mega, in capacitors it's milli
            # Default to milli (SPICE convention: M = milli)
            return num * 1e-3
        elif suffix in scale_map:
            return num * scale_map[suffix]
        return num
    return float(token) if token.replace('.','').replace('-','').replace('e','').replace('E','').replace('+','').isdigit() else 0

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
    parsed = parse_spice(req.netlist)
    components = parsed["components"]
    analysis = parsed["analysis"]
    if not components:
        raise HTTPException(status_code=400, detail="无法解析网表，请检查格式")
    solver = MNASolver()

    # Use analysis directives from netlist if not overridden by request
    atype = req.analysis_type if req.analysis_type else analysis.get("type", "dc")

    if atype in ("ac", "ac_sweep") and (req.frequency or analysis.get("ac_start")):
        f = req.frequency if req.frequency else analysis.get("ac_start", 1000)
        return solver.ac_solve(components, f)
    elif atype == "transient":
        ts = req.time_start if req.time_start else analysis.get("tran_start", 0)
        tstop = req.time_stop if req.time_stop else analysis.get("tran_stop", 0.001)
        tstep = req.time_step if req.time_step else analysis.get("tran_step", 1e-6)
        return solver.transient_solve(components, ts, tstop, tstep)
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
# QUIZ SEED DATA
# ============================================================

def seed_quiz_data():
    with get_db() as db:
        count = db.execute("SELECT COUNT(*) FROM quiz_categories").fetchone()[0]
        if count > 0:
            return  # Already seeded

        categories = [
            (1, "直流电路基础", "欧姆定律、基尔霍夫定律、串并联", "⚡"),
            (2, "交流电路分析", "相量、阻抗、功率因数", "〰️"),
            (3, "模拟电子", "运放、二极管、晶体管", "🔌"),
            (4, "数字电路", "逻辑门、触发器、时序", "🔢"),
            (5, "滤波器与谐振", "RC/RL/RLC、频率响应", "📡"),
        ]
        for cid, name, desc, icon in categories:
            db.execute("INSERT INTO quiz_categories (id, name, description, icon) VALUES (?,?,?,?)",
                       (cid, name, desc, icon))

        questions = [
            # DC Circuit Basics (cat 1)
            (1, "欧姆定律的正确表达式是？", "V = IR", "V = I/R", "V = R/I", "I = V²R", "A", "欧姆定律描述电压、电流和电阻之间的线性关系：V = IR。", 1),
            (1, "基尔霍夫电流定律 (KCL) 的核心内容是？", "流入节点的电流之和等于流出之和", "回路电压之和为零", "功率守恒", "电阻之和为零", "A", "KCL 基于电荷守恒：任何时刻流入节点的总电流等于流出节点的总电流。", 1),
            (1, "两个 100Ω 电阻并联后的等效电阻是？", "200Ω", "50Ω", "100Ω", "150Ω", "B", "并联公式：1/Req = 1/R1 + 1/R2 = 2/100，Req = 50Ω。", 1),
            (1, "基尔霍夫电压定律 (KVL) 的内容是？", "电流在节点处守恒", "闭合回路中电压代数和为零", "功率守恒", "电阻分压与阻值成正比", "B", "KVL 基于能量守恒：沿任意闭合回路一周，电压降的代数和为零。", 1),
            (1, "一个 12V 电源连接 3kΩ 电阻，通过该电阻的电流是？", "36mA", "4mA", "0.25mA", "3mA", "B", "I = V/R = 12/3000 = 0.004A = 4mA。", 1),
            (1, "电路中「节点」的定义是什么？", "两个元件之间的连接点", "三个或以上元件的连接点", "电源的正极", "接地端", "B", "节点是电路中三个或更多电路元件连接在一起的点。", 2),
            (1, "戴维南定理将一个线性有源二端网络等效为？", "一个电流源并联电阻", "一个电压源串联电阻", "一个受控源", "一个理想变压器", "B", "戴维南定理：任何线性二端网络可等效为一个电压源 Vth 串联一个电阻 Rth。", 2),
            (1, "一个 5V 电源，内阻为 0.5Ω，最大输出功率发生在？", "负载电阻 = 0Ω", "负载电阻 = 0.5Ω", "负载电阻 = 1Ω", "负载电阻 = ∞", "B", "最大功率传输定理：当负载电阻等于电源内阻 (0.5Ω) 时，传输功率最大。", 3),

            # AC Circuit (cat 2)
            (2, "交流电路中，纯电阻的阻抗相角是？", "90°", "-90°", "0°", "45°", "C", "纯电阻元件中电压和电流同相，阻抗角为 0°。", 1),
            (2, "电容在交流电路中的阻抗随频率如何变化？", "与频率成正比", "与频率成反比", "与频率无关", "随频率平方变化", "B", "容抗 Xc = 1/(2πfC)，与频率成反比。频率越高，容抗越小。", 1),
            (2, "一个 50Hz 交流正弦波，周期是多少？", "0.02s", "0.05s", "0.01s", "0.1s", "A", "周期 T = 1/f = 1/50 = 0.02s。", 1),
            (2, "交流电路中视在功率 S 的单位是？", "W（瓦特）", "VAR（乏）", "VA（伏安）", "J（焦耳）", "C", "视在功率 S = VI，单位为 VA（伏安）；有功功率 P = VI cos φ 单位为 W。", 2),
            (2, "功率因数 (Power Factor) 的定义是？", "P/S", "S/P", "Q/P", "P×S", "A", "功率因数 PF = P/S = cos φ，表示有功功率占视在功率的比例。", 2),
            (2, "纯电感在交流中的相角是？", "电压超前电流 90°", "电流超前电压 90°", "电压与电流同相", "电压滞后电流 45°", "A", "对于电感，电压超前电流 90°（ELI — E leads I in L）。", 3),

            # Analog Electronics (cat 3)
            (3, "理想运算放大器的输入阻抗是？", "0", "无穷大", "等于反馈电阻", "等于负载电阻", "B", "理想运放具有无穷大的输入阻抗，因此输入端几乎不吸取电流。", 1),
            (3, "硅二极管的正向导通电压约为？", "0.3V", "0.7V", "1.2V", "2.5V", "B", "硅二极管的典型正向压降约为 0.6-0.7V。锗二极管约为 0.3V。", 1),
            (3, "反相放大器的闭环增益公式是？", "Av = 1 + Rf/Rin", "Av = -Rf/Rin", "Av = Rf/Rin", "Av = -Rin/Rf", "B", "反相放大器增益 Av = -Rf/Rin，负号表示输出与输入相位差 180°。", 2),
            (3, "BJT 三个电极分别是？", "源极、漏极、栅极", "发射极、集电极、基极", "阳极、阴极、栅极", "漏极、源极、体极", "B", "BJT（双极结型晶体管）的三个端是发射极(Emitter)、集电极(Collector)、基极(Base)。", 1),
            (3, "共射极放大器的输入信号加在哪个极？", "集电极和发射极之间", "基极和发射极之间", "基极和集电极之间", "电源和地之间", "B", "共射极放大器中，发射极接地（公共端），输入信号加在基极和发射极之间。", 2),
            (3, "负反馈对放大器的带宽有什么影响？", "减小带宽", "增大带宽", "不影响带宽", "只影响低频", "B", "负反馈以降低增益为代价，扩展了放大器的带宽（增益带宽积恒定）。", 3),

            # Digital Circuits (cat 4)
            (4, "与门 (AND) 的输出为 1 的条件是？", "至少一个输入为 1", "所有输入为 1", "所有输入为 0", "输入不同", "B", "AND 门仅在所有输入都为高电平 (1) 时才输出高电平。", 1),
            (4, "D 触发器的功能是什么？", "计数", "延时/存储数据", "振荡", "放大", "B", "D 触发器在时钟边沿将 D 输入端的数据传送到 Q 输出端，实现一位数据存储。", 2),
            (4, "布尔表达式 A + A·B 简化后等于？", "A", "B", "A·B", "A+B", "A", "A + AB = A(1+B) = A（吸收律）。", 2),
            (4, "一个 4 位二进制计数器最多能计多少个状态？", "4 个", "8 个", "16 个", "32 个", "C", "n 位计数器可计 2^n 个状态，4 位 → 2^4 = 16 个状态。", 1),
            (4, "异或门 (XOR) 输出为 1 的条件是？", "两个输入相同", "两个输入不同", "两个输入都为 1", "两个输入都为 0", "B", "XOR 门在输入不同时输出 1，相同时输出 0。", 1),

            # Filters & Resonance (cat 5)
            (5, "RC 低通滤波器的截止频率公式是？", "fc = 1/(2πRC)", "fc = 2πRC", "fc = RC", "fc = 1/RC", "A", "截止频率（-3dB 点）fc = 1/(2πRC)。", 1),
            (5, "RLC 串联谐振时，电路呈什么性质？", "纯电容性", "纯电感性", "纯电阻性", "性质不确定", "C", "串联谐振时感抗和容抗抵消，电路呈现纯电阻性，阻抗最小，电流最大。", 2),
            (5, "品质因数 Q 越高，谐振电路的？", "带宽越宽", "选择性越差", "带宽越窄，选择性越好", "不影响选择性", "C", "Q 值越高，通频带越窄，频率选择性越好，谐振峰值越尖锐。", 2),
            (5, "一阶 RC 低通滤波器在截止频率处的增益是？", "0dB", "-3dB", "-6dB", "-20dB", "B", "截止频率处输出功率下降到 1/2，即 -3dB（约 0.707 倍电压增益）。", 2),
            (5, "带通滤波器的中心频率 f0 与截止频率 fL, fH 的关系是？", "f0 = fH + fL", "f0 = (fH - fL)/2", "f0 = √(fL·fH)", "f0 = fH·fL", "C", "中心频率 f0 = √(fL·fH)，是几何平均值而非算术平均值。", 3),
        ]
        for q in questions:
            db.execute("""
                INSERT INTO quiz_questions (category_id, question, option_a, option_b, option_c, option_d, correct_answer, explanation, difficulty)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, q)

seed_quiz_data()

# ============================================================
# QUIZ API
# ============================================================

class QuizSubmitRequest(BaseModel):
    category_id: int
    answers: List[Dict[str, str]]  # [{"question_id": 1, "answer": "A"}, ...]

@app.get("/api/quiz/categories")
def list_quiz_categories():
    with get_db() as db:
        cats = db.execute("""
            SELECT c.*, (SELECT COUNT(*) FROM quiz_questions WHERE category_id=c.id) as question_count
            FROM quiz_categories c ORDER BY c.id
        """).fetchall()
        return {"categories": [dict(c) for c in cats]}

@app.get("/api/quiz/questions")
def get_quiz_questions(
    category_id: int = Query(...),
    count: int = Query(10, ge=5, le=30),
    difficulty: Optional[int] = Query(None, ge=1, le=3)
):
    with get_db() as db:
        where = "WHERE category_id = ?"
        params = [category_id]
        if difficulty:
            where += " AND difficulty = ?"
            params.append(difficulty)

        questions = db.execute(f"""
            SELECT id, category_id, question, option_a, option_b, option_c, option_d, difficulty
            FROM quiz_questions {where}
            ORDER BY RANDOM() LIMIT ?
        """, params + [count]).fetchall()

        # Don't include correct_answer in the response
        return {
            "questions": [{
                "id": q["id"], "category_id": q["category_id"],
                "question": q["question"],
                "options": {"A": q["option_a"], "B": q["option_b"], "C": q["option_c"], "D": q["option_d"]},
                "difficulty": q["difficulty"]
            } for q in questions]
        }

@app.post("/api/quiz/submit")
def submit_quiz(req: QuizSubmitRequest, user: dict = Depends(require_user)):
    with get_db() as db:
        correct = 0
        total = len(req.answers)
        results = []

        for a in req.answers:
            q = db.execute(
                "SELECT correct_answer, explanation, question FROM quiz_questions WHERE id=?",
                (a["question_id"],)
            ).fetchone()
            is_correct = q["correct_answer"] == a["answer"] if q else False
            if is_correct: correct += 1
            results.append({
                "question_id": a["question_id"],
                "question": q["question"] if q else "",
                "your_answer": a["answer"],
                "correct_answer": q["correct_answer"] if q else "",
                "is_correct": is_correct,
                "explanation": q["explanation"] if q else ""
            })

        # Save result
        db.execute(
            "INSERT INTO quiz_results (user_id, category_id, score, total, answers_json) VALUES (?,?,?,?,?)",
            (user["id"], req.category_id, correct, total, json.dumps(results, ensure_ascii=False))
        )

        # Update reputation
        db.execute("UPDATE users SET reputation = reputation + ? WHERE id=?", (correct, user["id"]))

        return {
            "score": correct,
            "total": total,
            "percentage": round(correct / total * 100, 1) if total > 0 else 0,
            "results": results
        }

@app.get("/api/quiz/history")
def quiz_history(user: dict = Depends(require_user)):
    with get_db() as db:
        records = db.execute("""
            SELECT r.*, c.name as category_name, c.icon as category_icon
            FROM quiz_results r
            LEFT JOIN quiz_categories c ON r.category_id = c.id
            WHERE r.user_id = ?
            ORDER BY r.completed_at DESC LIMIT 20
        """, (user["id"],)).fetchall()
        return {"history": [dict(r) for r in records]}

@app.get("/api/quiz/leaderboard")
def quiz_leaderboard():
    with get_db() as db:
        leaders = db.execute("""
            SELECT u.username, u.avatar_url, SUM(r.score) as total_score, COUNT(r.id) as quizzes_taken,
                   ROUND(AVG(CAST(r.score AS FLOAT)/CAST(r.total AS FLOAT))*100, 1) as avg_pct
            FROM quiz_results r
            JOIN users u ON r.user_id = u.id
            GROUP BY r.user_id
            ORDER BY total_score DESC
            LIMIT 20
        """).fetchall()
        return {"leaderboard": [dict(l) for l in leaders]}

# ============================================================
# EXPORT API
# ============================================================

@app.post("/api/export/svg")
def export_svg(svg_content: str = None, components: List[Dict[str, Any]] = None, template_id: str = None):
    """Export circuit diagram as downloadable SVG"""
    from fastapi.responses import Response
    if svg_content:
        svg = svg_content
    elif template_id and template_id in CIRCUIT_TEMPLATES:
        comps = CIRCUIT_TEMPLATES[template_id]["components"]
        svg = generate_diagram_internal(comps)
    elif components:
        svg = generate_diagram_internal(components)
    else:
        raise HTTPException(status_code=400, detail="请提供 SVG 内容或 components")

    return Response(content=svg, media_type="image/svg+xml",
                    headers={"Content-Disposition": "attachment; filename=circuit.svg"})

@app.post("/api/export/report")
def export_report(data: Dict[str, Any] = None, user: dict = Depends(require_user)):
    """Generate a downloadable HTML report of simulation or quiz results"""
    report_type = data.get("type", "simulation") if data else "simulation"
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>Circuit AI Report</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0369a1; border-bottom: 3px solid #0369a1; padding-bottom: 10px; }}
  h2 {{ color: #334155; margin-top: 24px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
  th {{ background: #0369a1; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #e2e8f0; }}
  .meta {{ color: #64748b; font-size: 14px; }}
  .footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid #cbd5e1; color: #94a3b8; font-size: 12px; }}
  .correct {{ color: #16a34a; font-weight: bold; }}
  .wrong {{ color: #dc2626; }}
  pre {{ background: #f1f5f9; padding: 12px; border-radius: 6px; overflow-x: auto; }}
</style></head>
<body>
<h1>⚡ Circuit AI Report</h1>
<p class="meta">Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | User: {user['username']}</p>
"""
    if report_type == "simulation" and data:
        html += f"<h2>📊 Simulation Results — {data.get('analysis_type', 'DC').upper()} Analysis</h2>"
        result = data.get("result", data)
        if isinstance(result, dict):
            for key, val in result.items():
                if key == "node_voltages" and isinstance(val, dict):
                    html += "<h3>Node Voltages</h3><table><tr><th>Node</th><th>Voltage</th></tr>"
                    for node, v in val.items():
                        html += f"<tr><td>{node}</td><td>{v}</td></tr>"
                    html += "</table>"
                elif key == "branch_currents" and isinstance(val, list):
                    html += "<h3>Branch Currents</h3><table><tr><th>Component</th><th>Voltage</th><th>Current</th><th>Power</th></tr>"
                    for b in val:
                        html += f"<tr><td>{b.get('component','')}</td><td>{b.get('voltage','')}V</td><td>{b.get('current','')}A</td><td>{b.get('power','')}W</td></tr>"
                    html += "</table>"
                else:
                    html += f"<p><strong>{key}:</strong> <code>{val}</code></p>"
            if "circuit_svg" in data:
                html += f"<h3>Circuit Diagram</h3>{data['circuit_svg']}"

    elif report_type == "quiz" and data:
        html += f"<h2>📝 Quiz Results</h2>"
        html += f"<p><strong>Score:</strong> {data.get('score',0)}/{data.get('total',0)} ({data.get('percentage',0)}%)</p>"
        results = data.get("results", [])
        for r in results:
            cls = "correct" if r.get("is_correct") else "wrong"
            html += f"<div style='margin:12px 0;padding:12px;border-left:4px solid {'#16a34a' if r.get('is_correct') else '#dc2626'};background:#fff;'>"
            html += f"<p><strong>Q:</strong> {r.get('question','')}</p>"
            html += f"<p>Your answer: <span class='{cls}'>{r.get('your_answer','')}</span> | Correct: <span class='correct'>{r.get('correct_answer','')}</span></p>"
            html += f"<p style='color:#64748b;'>{r.get('explanation','')}</p></div>"

    html += f"""<div class="footer">Generated by Circuit AI — circuit learning platform</div></body></html>"""
    return Response(content=html, media_type="text/html",
                    headers={"Content-Disposition": "attachment; filename=circuit_report.html"})

def generate_diagram_internal(components: List[Dict]) -> str:
    """Internal SVG diagram generation helper"""
    svg_parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400" style="background:#0a0a0a;">']
    svg_parts.append('<defs><style>text{fill:#00ff41;font-family:monospace;font-size:14px} line{stroke:#00ff41;stroke-width:2} .comp{stroke:#00ffff;stroke-width:2;fill:none}</style></defs>')
    node_positions = {}
    nodes_seen = set()
    for c in components:
        for n in [c.get("node1", 0), c.get("node2", 0)]:
            nodes_seen.add(n)
    sorted_nodes = sorted(nodes_seen)
    cols = min(4, len(sorted_nodes)) or 1
    for i, n in enumerate(sorted_nodes):
        col = i % cols; row = i // cols
        x = 100 + col * 150; y = 80 + row * 120
        node_positions[n] = (x, y)

    for c in components:
        n1 = c.get("node1", 0); n2 = c.get("node2", 0)
        if n1 not in node_positions or n2 not in node_positions:
            continue
        x1, y1 = node_positions[n1]; x2, y2 = node_positions[n2]
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        name = c.get("name", ""); value = c.get("value", "")

        if c["type"] == "resistor":
            dx, dy = x2-x1, y2-y1; length = (dx*dx+dy*dy)**0.5 or 1
            nx, ny = -dy/length, dx/length
            zigzag = f'M {x1} {y1} '
            for i in range(1, 6):
                frac = i/6; px = x1+dx*frac+(nx*12 if i%2==0 else -nx*12)
                py = y1+dy*frac+(ny*12 if i%2==0 else -ny*12)
                zigzag += f'L {px} {py} '
            zigzag += f'L {x2} {y2}'
            svg_parts.append(f'<path d="{zigzag}" class="comp"/>')
        elif c["type"] == "capacitor":
            svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{mid_x-8}" y2="{y1}" class="comp"/>')
            svg_parts.append(f'<line x1="{mid_x-8}" y1="{y1-15}" x2="{mid_x-8}" y2="{y1+15}" class="comp"/>')
            svg_parts.append(f'<line x1="{mid_x+8}" y1="{y1-15}" x2="{mid_x+8}" y2="{y1+15}" class="comp"/>')
            svg_parts.append(f'<line x1="{mid_x+8}" y1="{y1}" x2="{x2}" y2="{y2}" class="comp"/>')
        elif "source" in c["type"]:
            svg_parts.append(f'<circle cx="{mid_x}" cy="{mid_y}" r="18" class="comp"/>')
            sign = "~" if "ac" in c["type"] else "+"
            svg_parts.append(f'<text x="{mid_x-3}" y="{mid_y-5}" text-anchor="end" fill="#00ff41">+</text>')
            svg_parts.append(f'<text x="{mid_x+3}" y="{mid_y+10}" text-anchor="start" fill="#00ff41">-</text>')
            svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{mid_x-18}" y2="{mid_y}" class="comp"/>')
            svg_parts.append(f'<line x1="{mid_x+18}" y1="{mid_y}" x2="{x2}" y2="{y2}" class="comp"/>')
        else:
            svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="comp"/>')
        if name:
            svg_parts.append(f'<text x="{mid_x}" y="{mid_y-25}" text-anchor="middle" fill="#00ffff" font-size="12">{name}={value}</text>')

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
    return "\n".join(svg_parts)

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
