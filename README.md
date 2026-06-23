# ⚡ Circuit AI — 电路学习平台

AI 驱动的电路学习网站，为电子工程学生打造。

## ✨ 功能

- **💬 AI 导师** — DeepSeek 驱动的电路问答
- **🌐 问答社区** — 提问/回答/采纳/图片上传
- **🔧 电路模拟器** — 可视化搭建 + SPICE 网表 + 电路模板（DC/AC/暂态）
- **🧮 工具箱** — 欧姆定律、分压器、RC 时间常数、色环电阻等 8 种计算器

## 🛠 技术栈

| 层 | 技术 |
|---|---|
| 后端 | Python FastAPI |
| 数据库 | SQLite |
| 前端 | HTML/CSS/JS（原生） |
| AI | DeepSeek API |
| 模拟 | NumPy MNA 求解器 |

## 🚀 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 设置 AI Key（可选）
# Windows: $env:DEEPSEEK_API_KEY="sk-..."
# Mac/Linux: export DEEPSEEK_API_KEY="sk-..."

# 启动
python app.py
```

打开 http://localhost:8000

## 📁 项目结构

```
circuit-helper/
├── app.py              # FastAPI 后端（所有路由 + MNA 求解器）
├── requirements.txt    # Python 依赖
└── static/
    ├── index.html      # 主页面
    ├── style.css       # OLED Dark 主题
    └── app.js          # 前端逻辑
```

## 🎨 设计系统

基于 UI/UX Pro Max Skill：
- Dark Mode (OLED) + Developer Tool palette
- Poppins + Open Sans + JetBrains Mono
- WCAG AAA 可访问性

## 📄 许可

MIT
