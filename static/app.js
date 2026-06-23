/**
 * Circuit AI - Frontend Application
 * Cyberpunk circuit learning platform
 */

// ============================================================
// STATE
// ============================================================
const state = {
  token: localStorage.getItem('token') || '',
  user: JSON.parse(localStorage.getItem('user') || 'null'),
  currentPanel: 'chat',
  currentChatId: null,
  chatMessages: [],
  currentQuestion: null,
  simulator: {
    mode: 'visual',
    components: [],
    wires: [],
    nextNode: 1,
  },
  community: { page: 1, tag: '', search: '', sort: 'newest' },
};

// ============================================================
// API HELPERS
// ============================================================
const API = {
  async request(method, url, body = null) {
    const opts = { method, headers: {} };
    if (state.token) opts.headers['Authorization'] = `Bearer ${state.token}`;
    if (body) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(url, opts);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '请求失败');
    return data;
  },
  get: (url) => API.request('GET', url),
  post: (url, body) => API.request('POST', url, body),
  put: (url, body) => API.request('PUT', url, body),
  delete: (url) => API.request('DELETE', url),
};

// ============================================================
// TOAST
// ============================================================
function toast(msg, type = 'info') {
  const container = document.getElementById('toast-container');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, 3000);
}

// ============================================================
// AUTH
// ============================================================
function initAuth() {
  const overlay = document.getElementById('auth-overlay');
  const app = document.getElementById('app');

  if (state.token && state.user) {
    overlay.classList.add('hidden');
    app.classList.remove('hidden');
    document.getElementById('user-name-display').textContent = state.user.username;
    loadApp();
  }

  // Tab switching
  document.querySelectorAll('.auth-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.auth-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.auth-form').forEach(f => f.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById(`${tab.dataset.tab}-form`).classList.add('active');
    });
  });

  // Login
  document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
      const data = await API.post('/api/auth/login', {
        email: document.getElementById('login-email').value,
        password: document.getElementById('login-password').value,
      });
      state.token = data.token;
      state.user = data.user;
      localStorage.setItem('token', data.token);
      localStorage.setItem('user', JSON.stringify(data.user));
      overlay.classList.add('hidden');
      app.classList.remove('hidden');
      document.getElementById('user-name-display').textContent = data.user.username;
      toast('登录成功 ⚡', 'success');
      loadApp();
    } catch (err) {
      toast(err.message, 'error');
    }
  });

  // Register
  document.getElementById('register-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
      const data = await API.post('/api/auth/register', {
        username: document.getElementById('reg-username').value,
        email: document.getElementById('reg-email').value,
        password: document.getElementById('reg-password').value,
      });
      state.token = data.token;
      state.user = data.user;
      localStorage.setItem('token', data.token);
      localStorage.setItem('user', JSON.stringify(data.user));
      overlay.classList.add('hidden');
      app.classList.remove('hidden');
      document.getElementById('user-name-display').textContent = data.user.username;
      toast('注册成功！欢迎 ⚡', 'success');
      loadApp();
    } catch (err) {
      toast(err.message, 'error');
    }
  });

  // Logout
  document.getElementById('btn-logout').addEventListener('click', () => {
    state.token = '';
    state.user = null;
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    app.classList.add('hidden');
    overlay.classList.remove('hidden');
    document.getElementById('login-email').value = '';
    document.getElementById('login-password').value = '';
  });
}

// ============================================================
// NAVIGATION
// ============================================================
function initNavigation() {
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      const panel = tab.dataset.panel;
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById(`panel-${panel}`).classList.add('active');
      state.currentPanel = panel;
      if (panel === 'community') loadQuestions();
      if (panel === 'simulator') initSimulator();
      if (panel === 'tools') { /* tools are static */ }
    });
  });
}

// ============================================================
// CHAT
// ============================================================
function initChat() {
  loadChatHistory();

  document.getElementById('btn-new-chat').addEventListener('click', () => {
    state.currentChatId = null;
    state.chatMessages = [];
    document.getElementById('chat-messages').innerHTML = `
      <div class="chat-welcome">
        <h2>👋 新对话</h2>
        <p>开始提问吧！</p>
        <div class="suggestions">
          <button class="suggestion" data-q="什么是基尔霍夫定律？">什么是基尔霍夫定律？</button>
          <button class="suggestion" data-q="RC电路的充放电过程是怎样的？">RC电路的充放电过程</button>
          <button class="suggestion" data-q="帮我设计一个截止频率1kHz的低通滤波器">设计1kHz低通滤波器</button>
          <button class="suggestion" data-q="戴维南定理和诺顿定理有什么区别？">戴维南定理vs诺顿定理</button>
        </div>
      </div>`;
    bindSuggestions();
    document.querySelectorAll('.chat-item').forEach(el => el.classList.remove('active'));
  });

  document.getElementById('btn-send').addEventListener('click', sendMessage);
  document.getElementById('chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  bindSuggestions();
}

function bindSuggestions() {
  document.querySelectorAll('.suggestion').forEach(btn => {
    btn.addEventListener('click', () => {
      document.getElementById('chat-input').value = btn.dataset.q;
      sendMessage();
    });
  });
}

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;

  input.value = '';
  const messagesDiv = document.getElementById('chat-messages');

  // Remove welcome
  const welcome = messagesDiv.querySelector('.chat-welcome');
  if (welcome) welcome.remove();

  // Add user message
  const userMsg = { role: 'user', content: msg };
  state.chatMessages.push(userMsg);
  messagesDiv.appendChild(createMsgEl(userMsg));

  // Add loading placeholder
  const loading = createMsgEl({ role: 'assistant', content: '▌' });
  loading.classList.add('loading');
  messagesDiv.appendChild(loading);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${state.token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, history: state.chatMessages.slice(0, -1) }),
    });

    loading.remove();

    if (res.status === 503) {
      const noKeyMsg = { role: 'assistant', content:
        '⚠️ **AI 服务未配置**\n\n需要设置 DeepSeek API Key 才能使用 AI 导师：\n\n' +
        '**步骤 1：** 去 [platform.deepseek.com](https://platform.deepseek.com) 注册获取 Key\n' +
        '**步骤 2：** 终端运行：\n```\n$env:DEEPSEEK_API_KEY="sk-..."\n```\n' +
        '**步骤 3：** 重启：\n```\ncd circuit-helper && python app.py\n```\n\n' +
        '> 💡 新用户有免费额度\n\n' +
        '没有 Key 也不要紧 — 问答社区、模拟器、工具箱都可以正常使用！'
      };
      state.chatMessages.push(noKeyMsg);
      messagesDiv.appendChild(createMsgEl(noKeyMsg));
    } else if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      const errMsg = { role: 'assistant', content: '❌ ' + (errData.detail || '请求失败') };
      state.chatMessages.push(errMsg);
      messagesDiv.appendChild(createMsgEl(errMsg));
    } else {
      const data = await res.json();
      const assistantMsg = { role: 'assistant', content: data.reply };
      state.chatMessages.push(assistantMsg);
      messagesDiv.appendChild(createMsgEl(assistantMsg));
      if (data.chat_id && !state.currentChatId) {
        state.currentChatId = data.chat_id;
        loadChatHistory();
      }
    }
  } catch (err) {
    loading.remove();
    const errMsg = { role: 'assistant', content: '❌ 网络错误: ' + err.message };
    state.chatMessages.push(errMsg);
    messagesDiv.appendChild(createMsgEl(errMsg));
  }

  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function createMsgEl(msg) {
  const div = document.createElement('div');
  div.className = `chat-msg ${msg.role}`;
  div.innerHTML = `<div class="role">${msg.role === 'user' ? '🧑 你' : '🤖 电路小AI'}</div>
    <div class="content">${formatContent(msg.content)}</div>`;
  return div;
}

function formatContent(text) {
  // Simple markdown-like formatting
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`{3}(\w*)\n([\s\S]*?)`{3}/g, '<pre><code>$2</code></pre>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>')
    .replace(/\$([^$]+)\$/g, '<code>$1</code>');
}

async function loadChatHistory() {
  if (!state.token) return;
  try {
    const data = await API.get('/api/chats');
    const list = document.getElementById('chat-list');
    list.innerHTML = data.chats.map(c => `
      <div class="chat-item ${c.id === state.currentChatId ? 'active' : ''}" data-id="${c.id}">
        ${escapeHtml(c.title)}
      </div>
    `).join('') || '<div style="color:var(--text-dim);font-size:12px;padding:8px;">暂无历史对话</div>';

    list.querySelectorAll('.chat-item').forEach(item => {
      item.addEventListener('click', async () => {
        const id = parseInt(item.dataset.id);
        try {
          const chatData = await API.get(`/api/chats/${id}`);
          const msgs = JSON.parse(chatData.chat.messages);
          state.currentChatId = id;
          state.chatMessages = msgs;
          const messagesDiv = document.getElementById('chat-messages');
          messagesDiv.innerHTML = '';
          msgs.forEach(m => messagesDiv.appendChild(createMsgEl(m)));
          messagesDiv.scrollTop = messagesDiv.scrollHeight;
          document.querySelectorAll('.chat-item').forEach(el => el.classList.remove('active'));
          item.classList.add('active');
        } catch (err) {
          toast('加载对话失败', 'error');
        }
      });
    });
  } catch (err) {
    // Chat history load failed - non-critical
  }
}

// ============================================================
// IMAGE UPLOAD MANAGER
// ============================================================
class ImageUploader {
  constructor(dropzoneId, fileInputId, previewsId) {
    this.dropzone = document.getElementById(dropzoneId);
    this.fileInput = document.getElementById(fileInputId);
    this.previews = document.getElementById(previewsId);
    this.urls = [];
    this._init();
  }

  _init() {
    if (!this.dropzone || !this.fileInput) return;

    // Click to select
    this.dropzone.addEventListener('click', () => this.fileInput.click());
    this.fileInput.addEventListener('change', (e) => this._handleFiles(e.target.files));

    // Drag & drop
    this.dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.dropzone.classList.add('drag-over');
    });
    this.dropzone.addEventListener('dragleave', () => {
      this.dropzone.classList.remove('drag-over');
    });
    this.dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      this.dropzone.classList.remove('drag-over');
      this._handleFiles(e.dataTransfer.files);
    });

    // Clipboard paste
    document.addEventListener('paste', (e) => {
      // Only handle paste when this uploader's modal/form is visible
      const modal = this.dropzone.closest('.modal');
      if (modal && modal.classList.contains('hidden')) return;
      // Also check if the panel is active for answer form
      const panel = this.dropzone.closest('#question-modal');
      if (panel && panel.classList.contains('hidden')) return;

      const items = e.clipboardData?.items;
      if (!items) return;
      const files = [];
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          files.push(item.getAsFile());
        }
      }
      if (files.length) this._handleFiles(files);
    });
  }

  async _handleFiles(files) {
    for (const file of files) {
      if (!file.type.startsWith('image/')) continue;
      if (file.size > 10 * 1024 * 1024) {
        toast(`${file.name} 超过10MB限制`, 'error');
        continue;
      }

      // Show loading
      const loadingEl = document.createElement('div');
      loadingEl.className = 'upload-loading';
      loadingEl.textContent = '⏳';
      this.previews.appendChild(loadingEl);

      try {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch('/api/upload', {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${state.token}` },
          body: formData,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail);

        loadingEl.remove();
        this.urls.push(data.url);
        this._addPreview(data.url);
      } catch (err) {
        loadingEl.remove();
        toast(`上传失败: ${err.message}`, 'error');
      }
    }
  }

  _addPreview(url) {
    const preview = document.createElement('div');
    preview.className = 'upload-preview';
    preview.innerHTML = `
      <img src="${url}" alt="preview">
      <button class="remove-preview" data-url="${url}">&times;</button>
    `;
    preview.querySelector('.remove-preview').addEventListener('click', (e) => {
      e.stopPropagation();
      this.urls = this.urls.filter(u => u !== url);
      preview.remove();
    });
    this.previews.appendChild(preview);
  }

  getUrls() {
    return [...this.urls];
  }

  reset() {
    this.urls = [];
    this.previews.innerHTML = '';
    if (this.fileInput) this.fileInput.value = '';
  }
}

// Render images inline (used in question/answer display)
function renderImages(imageUrlsStr) {
  try {
    const urls = JSON.parse(imageUrlsStr || '[]');
    if (!urls.length) return '';
    return `<div class="inline-images">${urls.map(url =>
      `<img src="${url}" alt="uploaded image" onclick="event.stopPropagation(); showLightbox('${url}')" loading="lazy">`
    ).join('')}</div>`;
  } catch { return ''; }
}

function showLightbox(url) {
  const lb = document.createElement('div');
  lb.className = 'lightbox';
  lb.innerHTML = `<img src="${url}" alt="lightbox">`;
  lb.addEventListener('click', () => lb.remove());
  document.body.appendChild(lb);
}

// ============================================================
// COMMUNITY
// ============================================================
async function loadQuestions() {
  const { page, tag, search, sort } = state.community;
  try {
    const params = new URLSearchParams({ page, per_page: 20, tag, search, sort });
    const data = await API.get(`/api/questions?${params}`);
    renderQuestions(data);
  } catch (err) {
    toast('加载问题失败', 'error');
  }
}

function renderQuestions(data) {
  const list = document.getElementById('question-list');
  if (!data.questions.length) {
    list.innerHTML = '<p style="color:var(--text-dim);text-align:center;padding:40px;">还没有问题，成为第一个提问的人吧 🚀</p>';
    return;
  }
  list.innerHTML = data.questions.map(q => {
    const hasImages = (() => { try { return JSON.parse(q.image_urls || '[]').length > 0; } catch { return false; } })();
    return `
    <div class="question-card" data-id="${q.id}">
      <h3>${escapeHtml(q.title)} ${hasImages ? '📷' : ''}</h3>
      <div class="q-meta">
        <span>👤 ${escapeHtml(q.username)}</span>
        <span>💬 ${q.answer_count} 回答</span>
        <span>👍 ${q.vote_score}</span>
        <span>👁 ${q.view_count}</span>
        <span>${timeAgo(q.created_at)}</span>
      </div>
    </div>
  `}).join('');

  list.querySelectorAll('.question-card').forEach(card => {
    card.addEventListener('click', () => viewQuestion(parseInt(card.dataset.id)));
  });

  // Pagination
  const pag = document.getElementById('pagination');
  const totalPages = Math.ceil(data.total / data.per_page);
  pag.innerHTML = '';
  for (let i = 1; i <= Math.min(totalPages, 10); i++) {
    const btn = document.createElement('button');
    btn.textContent = i;
    if (i === state.community.page) btn.classList.add('active');
    btn.addEventListener('click', () => { state.community.page = i; loadQuestions(); });
    pag.appendChild(btn);
  }
}

async function viewQuestion(id) {
  try {
    const data = await API.get(`/api/questions/${id}`);
    state.currentQuestion = data;
    const modal = document.getElementById('question-modal');
    const detail = document.getElementById('question-detail');
    detail.innerHTML = `
      <div class="question-detail">
        <h2>${escapeHtml(data.question.title)}</h2>
        <div class="q-content content-block">${formatContent(data.question.content)}</div>
        ${renderImages(data.question.image_urls)}
        <div class="q-meta" style="margin-bottom:16px;">
          <span>👤 ${escapeHtml(data.question.username)}</span>
          <span>👁 ${data.question.view_count} 浏览</span>
        </div>
        <h3 style="color:var(--neon);">${data.answers.length} 个回答</h3>
        ${data.answers.map(a => `
          <div class="answer-card ${a.is_accepted ? 'accepted' : ''}">
            <div class="a-content content-block">${formatContent(a.content)}</div>
            ${renderImages(a.image_urls)}
            <div class="a-meta">
              <span>👤 ${escapeHtml(a.username)}</span>
              <span>👍 ${a.vote_score}</span>
              <span>${timeAgo(a.created_at)}</span>
              ${a.is_accepted ? '<span style="color:var(--neon);">✅ 已采纳</span>' : ''}
              ${state.user && state.user.id === data.question.user_id && !a.is_accepted ?
                `<button class="btn-small btn-accept" data-answer-id="${a.id}">采纳</button>` : ''}
            </div>
          </div>
        `).join('')}
        ${state.user ? `
          <div class="answer-form" id="answer-form-area">
            <textarea id="answer-content" rows="4" placeholder="写下你的回答..."></textarea>
            <div class="upload-area" style="margin:8px 0;">
              <input type="file" id="answer-file-input" accept="image/*" multiple hidden>
              <div class="upload-dropzone" id="answer-dropzone">
                <span>📁 上传图片（可选）</span>
              </div>
              <div class="upload-previews" id="answer-previews"></div>
            </div>
            <button class="btn-glow" id="btn-submit-answer">提交回答</button>
          </div>
        ` : '<p style="color:var(--text-dim);">请登录后回答</p>'}
      </div>`;
    modal.classList.remove('hidden');

    // Close button
    modal.querySelector('.modal-close').addEventListener('click', () => modal.classList.add('hidden'));
    modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.add('hidden'); });

    // Accept answer
    detail.querySelectorAll('.btn-accept').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
          await API.put(`/api/answers/${btn.dataset.answerId}/accept`);
          toast('已采纳答案 ✅', 'success');
          viewQuestion(id);
        } catch (err) { toast(err.message, 'error'); }
      });
    });

    // Set up answer image uploader
    let answerUploader = null;
    const answerFormArea = document.getElementById('answer-form-area');
    if (answerFormArea) {
      answerUploader = new ImageUploader('answer-dropzone', 'answer-file-input', 'answer-previews');
    }

    // Submit answer
    const submitBtn = document.getElementById('btn-submit-answer');
    if (submitBtn) {
      submitBtn.addEventListener('click', async () => {
        const content = document.getElementById('answer-content').value.trim();
        if (!content) return toast('请输入回答内容', 'error');
        try {
          const imageUrls = JSON.stringify(answerUploader ? answerUploader.getUrls() : []);
          await API.post(`/api/questions/${id}/answers`, { content, image_urls: imageUrls });
          toast('回答发布成功！', 'success');
          viewQuestion(id);
        } catch (err) { toast(err.message, 'error'); }
      });
    }
  } catch (err) {
    toast('加载问题失败', 'error');
  }
}

function initCommunity() {
  // Image uploader for ask form
  const askUploader = new ImageUploader('ask-dropzone', 'ask-file-input', 'ask-previews');

  document.getElementById('btn-ask').addEventListener('click', () => {
    askUploader.reset();
    document.getElementById('ask-modal').classList.remove('hidden');
  });
  document.querySelectorAll('#ask-modal .modal-close').forEach(btn => {
    btn.addEventListener('click', () => document.getElementById('ask-modal').classList.add('hidden'));
  });
  document.getElementById('ask-modal').addEventListener('click', (e) => {
    if (e.target === document.getElementById('ask-modal')) document.getElementById('ask-modal').classList.add('hidden');
  });

  document.getElementById('ask-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
      const imageUrls = JSON.stringify(askUploader.getUrls());
      await API.post('/api/questions', {
        title: document.getElementById('ask-title').value,
        content: document.getElementById('ask-content').value,
        tags: document.getElementById('ask-tags').value,
        image_urls: imageUrls,
      });
      toast('问题发布成功！', 'success');
      document.getElementById('ask-modal').classList.add('hidden');
      document.getElementById('ask-title').value = '';
      document.getElementById('ask-content').value = '';
      document.getElementById('ask-tags').value = '';
      askUploader.reset();
      loadQuestions();
    } catch (err) { toast(err.message, 'error'); }
  });

  // Search
  document.getElementById('community-search').addEventListener('input', (e) => {
    state.community.search = e.target.value;
    state.community.page = 1;
    loadQuestions();
  });

  // Tags
  document.querySelectorAll('#tag-cloud .tag').forEach(tag => {
    tag.addEventListener('click', () => {
      document.querySelectorAll('#tag-cloud .tag').forEach(t => t.classList.remove('active'));
      tag.classList.add('active');
      state.community.tag = tag.dataset.tag;
      state.community.page = 1;
      loadQuestions();
    });
  });

  // Sort
  document.getElementById('sort-select').addEventListener('change', (e) => {
    state.community.sort = e.target.value;
    loadQuestions();
  });

  // Close modals
  document.querySelectorAll('.modal-close').forEach(btn => {
    btn.addEventListener('click', function() { this.closest('.modal').classList.add('hidden'); });
  });
  document.querySelectorAll('.modal').forEach(m => {
    m.addEventListener('click', function(e) { if (e.target === this) this.classList.add('hidden'); });
  });
}

// ============================================================
// SIMULATOR
// ============================================================
function initSimulator() {
  // Mode switching
  document.querySelectorAll('.sim-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.sim-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.sim-mode').forEach(m => m.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById(`sim-${tab.dataset.mode}`).classList.add('active');
      state.simulator.mode = tab.dataset.mode;
      if (tab.dataset.mode === 'template') loadTemplates();
    });
  });

  // Analysis type toggle
  const analysisType = document.getElementById('analysis-type');
  analysisType.addEventListener('change', () => {
    document.getElementById('ac-params').classList.toggle('hidden', analysisType.value !== 'ac');
    document.getElementById('transient-params').classList.toggle('hidden', analysisType.value !== 'transient');
  });

  // Visual builder
  initVisualBuilder();

  // SPICE
  document.getElementById('btn-spice-run').addEventListener('click', runSpiceSimulation);

  // Template
  loadTemplates();
}

// ---- Visual Builder ----
function initVisualBuilder() {
  const canvas = document.getElementById('sim-canvas');
  let dragType = null;
  let nextNodeId = 1;

  // Drag from palette
  document.querySelectorAll('.comp-item').forEach(item => {
    item.addEventListener('dragstart', (e) => {
      dragType = item.dataset.type;
      e.dataTransfer.setData('text/plain', dragType);
    });
  });

  canvas.addEventListener('dragover', (e) => e.preventDefault());
  canvas.addEventListener('drop', (e) => {
    e.preventDefault();
    if (!dragType && !e.dataTransfer.getData('text/plain')) return;
    const type = dragType || e.dataTransfer.getData('text/plain');
    const rect = canvas.getBoundingClientRect();
    const scaleX = 800 / rect.width;
    const scaleY = 600 / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    if (type === 'ground') {
      addComponentToCanvas(canvas, type, 0, x, y);
    } else {
      const n1 = nextNodeId++;
      const n2 = nextNodeId++;
      addComponentToCanvas(canvas, type, n1, n2, x, y);
    }
    updatePlacedCount();
    dragType = null;
  });

  // Clear canvas
  document.getElementById('btn-clear-canvas').addEventListener('click', () => {
    state.simulator.components = [];
    state.simulator.nextNode = 1;
    nextNodeId = 1;
    canvas.querySelectorAll('.placed-comp, .placed-node, .placed-label').forEach(el => el.remove());
    document.getElementById('sim-results-content').innerHTML = '<p class="placeholder">点击「运行模拟」查看结果</p>';
    updatePlacedCount();
  });

  // Simulate button
  document.getElementById('btn-simulate').addEventListener('click', runVisualSimulation);
}

function addComponentToCanvas(canvas, type, n1, n2, x, y) {
  const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  g.classList.add('placed-comp');
  g.setAttribute('transform', `translate(${x},${y})`);

  const colors = { resistor: '#00ffff', capacitor: '#00ffff', inductor: '#00ffff',
    voltage_source_dc: '#ff6600', voltage_source_ac: '#ff6600', current_source_dc: '#ffd600', ground: '#00ff41' };
  const color = colors[type] || '#00ffff';

  if (type === 'resistor') {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'M-40,0 L-30,0 L-25,-10 L-15,10 L-5,-10 L5,10 L15,-10 L25,10 L35,0 L40,0');
    path.setAttribute('stroke', color); path.setAttribute('fill', 'none'); path.setAttribute('stroke-width', '2');
    g.appendChild(path);
  } else if (type === 'capacitor') {
    const l1 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    l1.setAttribute('x1', '-40'); l1.setAttribute('y1', '0'); l1.setAttribute('x2', '-6'); l1.setAttribute('y2', '0');
    l1.setAttribute('stroke', color); l1.setAttribute('stroke-width', '2');
    const l2 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    l2.setAttribute('x1', '-6'); l2.setAttribute('y1', '-14'); l2.setAttribute('x2', '-6'); l2.setAttribute('y2', '14');
    l2.setAttribute('stroke', color); l2.setAttribute('stroke-width', '2');
    const l3 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    l3.setAttribute('x1', '6'); l3.setAttribute('y1', '-14'); l3.setAttribute('x2', '6'); l3.setAttribute('y2', '14');
    l3.setAttribute('stroke', color); l3.setAttribute('stroke-width', '2');
    const l4 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    l4.setAttribute('x1', '6'); l4.setAttribute('y1', '0'); l4.setAttribute('x2', '40'); l4.setAttribute('y2', '0');
    l4.setAttribute('stroke', color); l4.setAttribute('stroke-width', '2');
    g.append(l1, l2, l3, l4);
  } else if (type === 'inductor') {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'M-40,0 L-20,0 Q-17,-8 -14,0 Q-11,8 -8,0 Q-5,-8 -2,0 Q1,8 4,0 Q7,-8 10,0 Q13,8 16,0 Q19,-8 22,0 L40,0');
    path.setAttribute('stroke', color); path.setAttribute('fill', 'none'); path.setAttribute('stroke-width', '2');
    g.appendChild(path);
  } else if (type.startsWith('voltage_source')) {
    const l1 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    l1.setAttribute('x1', '-40'); l1.setAttribute('y1', '0'); l1.setAttribute('x2', '-14'); l1.setAttribute('y2', '0');
    l1.setAttribute('stroke', color); l1.setAttribute('stroke-width', '2');
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', '0'); circle.setAttribute('cy', '0'); circle.setAttribute('r', '14');
    circle.setAttribute('stroke', color); circle.setAttribute('fill', 'none'); circle.setAttribute('stroke-width', '2');
    const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    txt.setAttribute('x', '0'); txt.setAttribute('y', '5'); txt.setAttribute('text-anchor', 'middle');
    txt.setAttribute('fill', color); txt.setAttribute('font-size', '12');
    txt.textContent = type.includes('ac') ? '~' : 'DC';
    const l2 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    l2.setAttribute('x1', '14'); l2.setAttribute('y1', '0'); l2.setAttribute('x2', '40'); l2.setAttribute('y2', '0');
    l2.setAttribute('stroke', color); l2.setAttribute('stroke-width', '2');
    g.append(l1, circle, txt, l2);
  } else if (type === 'current_source_dc') {
    const l1 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    l1.setAttribute('x1', '-40'); l1.setAttribute('y1', '0'); l1.setAttribute('x2', '-14'); l1.setAttribute('y2', '0');
    l1.setAttribute('stroke', color); l1.setAttribute('stroke-width', '2');
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', '0'); circle.setAttribute('cy', '0'); circle.setAttribute('r', '14');
    circle.setAttribute('stroke', color); circle.setAttribute('fill', 'none'); circle.setAttribute('stroke-width', '2');
    const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    txt.setAttribute('x', '0'); txt.setAttribute('y', '5'); txt.setAttribute('text-anchor', 'middle');
    txt.setAttribute('fill', color); txt.setAttribute('font-size', '14'); txt.textContent = 'I';
    const l2 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    l2.setAttribute('x1', '14'); l2.setAttribute('y1', '0'); l2.setAttribute('x2', '40'); l2.setAttribute('y2', '0');
    l2.setAttribute('stroke', color); l2.setAttribute('stroke-width', '2');
    g.append(l1, circle, txt, l2);
  } else if (type === 'ground') {
    const l1 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    l1.setAttribute('x1', '0'); l1.setAttribute('y1', '-10'); l1.setAttribute('x2', '0'); l1.setAttribute('y2', '0');
    l1.setAttribute('stroke', color); l1.setAttribute('stroke-width', '2');
    const l2 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    l2.setAttribute('x1', '-14'); l2.setAttribute('y1', '0'); l2.setAttribute('x2', '14'); l2.setAttribute('y2', '0');
    l2.setAttribute('stroke', color); l2.setAttribute('stroke-width', '2');
    const l3 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    l3.setAttribute('x1', '-9'); l3.setAttribute('y1', '6'); l3.setAttribute('x2', '9'); l3.setAttribute('y2', '6');
    l3.setAttribute('stroke', color); l3.setAttribute('stroke-width', '2');
    const l4 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    l4.setAttribute('x1', '-5'); l4.setAttribute('y1', '12'); l4.setAttribute('x2', '5'); l4.setAttribute('y2', '12');
    l4.setAttribute('stroke', color); l4.setAttribute('stroke-width', '2');
    g.append(l1, l2, l3, l4);
  }

  // Node dots
  const n1Circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  n1Circle.setAttribute('cx', '-40'); n1Circle.setAttribute('cy', '0'); n1Circle.setAttribute('r', '4');
  n1Circle.setAttribute('fill', '#ff6600'); n1Circle.classList.add('placed-node');
  const n2Circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  n2Circle.setAttribute('cx', '40'); n2Circle.setAttribute('cy', '0'); n2Circle.setAttribute('r', '4');
  n2Circle.setAttribute('fill', '#ff6600'); n2Circle.classList.add('placed-node');
  g.append(n1Circle, n2Circle);

  // Label
  const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  label.setAttribute('x', '0'); label.setAttribute('y', '-18'); label.setAttribute('text-anchor', 'middle');
  label.setAttribute('fill', '#00ff41'); label.setAttribute('font-size', '11');
  label.classList.add('placed-label');
  const names = { resistor: 'R', capacitor: 'C', inductor: 'L', voltage_source_dc: 'Vdc', voltage_source_ac: 'Vac', current_source_dc: 'Idc', ground: 'GND' };
  const idx = state.simulator.components.filter(c => c.type === type).length + 1;
  label.textContent = `${names[type] || type}${idx}`;
  g.appendChild(label);

  // Make draggable
  g.style.cursor = 'move';
  let dragging = false, startX, startY, origX, origY;
  g.addEventListener('mousedown', (e) => {
    dragging = true;
    startX = e.clientX; startY = e.clientY;
    const t = g.getAttribute('transform');
    const match = t.match(/translate\(([^,]+),\s*([^)]+)\)/);
    origX = match ? parseFloat(match[1]) : x;
    origY = match ? parseFloat(match[2]) : y;
    e.stopPropagation();
  });
  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    g.setAttribute('transform', `translate(${origX + e.clientX - startX},${origY + e.clientY - startY})`);
  });
  document.addEventListener('mouseup', () => { dragging = false; });

  canvas.appendChild(g);
  state.simulator.components.push({ type, n1, n2, x, y, el: g });
}

function updatePlacedCount() {
  document.getElementById('placed-components').textContent = `已放置 ${state.simulator.components.length} 个元件`;
}

async function runVisualSimulation() {
  const comps = state.simulator.components.map(c => {
    const names = { resistor: 'R', capacitor: 'C', inductor: 'L', voltage_source_dc: 'Vdc', voltage_source_ac: 'Vac', current_source_dc: 'Idc', ground: 'GND' };
    const idx = state.simulator.components.filter(sc => sc.type === c.type).indexOf(c) + 1;
    return {
      type: c.type,
      name: `${names[c.type] || c.type}${idx}`,
      value: c.type.includes('resistor') ? 1000 : c.type.includes('source') ? 5 : 1e-6,
      node1: c.n1,
      node2: c.n2,
    };
  });

  const analysisType = document.getElementById('analysis-type').value;
  const body = { components: comps, analysis_type: analysisType };
  if (analysisType === 'ac') body.frequency = parseFloat(document.getElementById('ac-freq').value) || 1000;
  if (analysisType === 'transient') {
    body.time_start = parseFloat(document.getElementById('trans-start').value) || 0;
    body.time_stop = parseFloat(document.getElementById('trans-stop').value) || 0.01;
    body.time_step = parseFloat(document.getElementById('trans-step').value) || 0.0001;
  }

  try {
    const data = await API.post('/api/simulate', body);
    displaySimResults(data, 'sim-results-content');
  } catch (err) {
    document.getElementById('sim-results-content').innerHTML = `<p style="color:var(--red);">${err.message}</p>`;
  }
}

async function runSpiceSimulation() {
  const netlist = document.getElementById('spice-netlist').value.trim();
  if (!netlist) return toast('请输入 SPICE 网表', 'error');
  const analysisType = document.getElementById('spice-analysis-type').value;
  const body = { netlist, analysis_type: analysisType };
  if (analysisType === 'ac') body.frequency = parseFloat(document.getElementById('spice-freq').value) || 1000;
  try {
    const data = await API.post('/api/simulate/spice', body);
    displaySimResults(data, 'spice-results-content');
  } catch (err) {
    document.getElementById('spice-results-content').innerHTML = `<p style="color:var(--red);">${err.message}</p>`;
  }
}

function displaySimResults(data, containerId) {
  const container = document.getElementById(containerId);
  if (data.error) {
    container.innerHTML = `<p style="color:var(--red);">⚠ ${data.error}</p>`;
    return;
  }

  let html = '';
  if (data.type === 'dc') {
    html += '<h4 style="color:var(--neon);margin-bottom:8px;">📌 DC 工作点分析</h4>';
    html += '<table class="result-table"><tr><th>节点</th><th>电压 (V)</th></tr>';
    for (const [node, voltage] of Object.entries(data.node_voltages)) {
      html += `<tr><td>${node}</td><td style="color:var(--cyan);">${voltage}</td></tr>`;
    }
    html += '</table>';
    if (data.branch_currents && data.branch_currents.length) {
      html += '<h4 style="color:var(--neon);margin:12px 0 8px;">⚡ 支路电流</h4>';
      html += '<table class="result-table"><tr><th>元件</th><th>电压</th><th>电流</th><th>功率</th></tr>';
      for (const b of data.branch_currents) {
        html += `<tr><td>${b.component}</td><td>${b.voltage}V</td><td style="color:var(--cyan);">${b.current}A</td><td>${b.power}W</td></tr>`;
      }
      html += '</table>';
    }
  } else if (data.type === 'ac') {
    html += `<h4 style="color:var(--neon);margin-bottom:8px;">📌 AC 分析 @ ${data.frequency}Hz</h4>`;
    html += '<table class="result-table"><tr><th>节点</th><th>幅值</th><th>相位</th></tr>';
    for (const [node, v] of Object.entries(data.node_voltages)) {
      html += `<tr><td>${node}</td><td style="color:var(--cyan);">${v.magnitude}V</td><td>${v.phase_deg}°</td></tr>`;
    }
    html += '</table>';
  } else if (data.type === 'transient') {
    html += '<h4 style="color:var(--neon);margin-bottom:8px;">📌 暂态分析</h4>';
    html += `<p style="font-size:12px;color:var(--text-dim);">${data.points} 个时间点</p>`;
    // Simple text table of first and last values
    html += '<table class="result-table"><tr><th>节点</th><th>初始值</th><th>最终值</th></tr>';
    const t = data.data.time;
    for (const [node, vals] of Object.entries(data.data.node_voltages)) {
      html += `<tr><td>${node}</td><td style="color:var(--cyan);">${vals[0]}V</td><td style="color:var(--cyan);">${vals[vals.length-1]}V</td></tr>`;
    }
    html += '</table>';
    // Show data for plotting
    html += `<div class="sim-plot" id="plot-${containerId}"></div>`;
    setTimeout(() => drawTransientPlot(data.data, `plot-${containerId}`), 100);
  }
  container.innerHTML = html;
}

function drawTransientPlot(transData, plotId) {
  const plotEl = document.getElementById(plotId);
  if (!plotEl) return;
  const canvas = document.createElement('canvas');
  canvas.width = plotEl.clientWidth;
  canvas.height = 300;
  plotEl.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  const t = transData.time;
  const voltages = Object.values(transData.node_voltages);
  if (!voltages.length || !t.length) return;

  const allV = voltages.flat();
  const vMin = Math.min(...allV), vMax = Math.max(...allV);
  const vRange = vMax - vMin || 1;
  const padding = 40;
  const w = canvas.width - 2 * padding;
  const h = canvas.height - 2 * padding;

  // Grid
  ctx.strokeStyle = '#1a3a1a';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 5; i++) {
    const y = padding + h * i / 5;
    ctx.beginPath(); ctx.moveTo(padding, y); ctx.lineTo(padding + w, y); ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = '#00ff41';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(padding, padding); ctx.lineTo(padding, padding + h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(padding, padding + h); ctx.lineTo(padding + w, padding + h); ctx.stroke();

  // Plot each node voltage
  const colors = ['#00ffff', '#ff6600', '#ffd600', '#ff1744', '#00ff41'];
  voltages.forEach((vals, vi) => {
    ctx.strokeStyle = colors[vi % colors.length];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < t.length; i++) {
      const x = padding + (t[i] - t[0]) / (t[t.length-1] - t[0]) * w;
      const y = padding + h - (vals[i] - vMin) / vRange * h;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });

  // Labels
  ctx.fillStyle = '#00ff41';
  ctx.font = '11px monospace';
  ctx.fillText(`${vMax.toFixed(2)}V`, 4, padding + 10);
  ctx.fillText(`${vMin.toFixed(2)}V`, 4, padding + h);
  ctx.fillText(`${t[0].toFixed(4)}s`, padding, padding + h + 16);
  ctx.fillText(`${t[t.length-1].toFixed(4)}s`, padding + w - 60, padding + h + 16);
}

// ---- Templates ----
async function loadTemplates() {
  try {
    const data = await API.get('/api/templates');
    const grid = document.getElementById('template-grid');
    grid.innerHTML = data.templates.map(t => `
      <div class="template-card" data-id="${t.id}">
        <h3>${t.name}</h3>
        <p>${t.description}</p>
        <div class="template-params">
          ${Object.entries(t.params).map(([k, v]) =>
            `<label>${k}: <input type="number" class="tp-${k}" value="${v}" step="any"></label>`
          ).join('')}
        </div>
        <button class="btn-glow btn-template-run" style="margin-top:8px;">▶ 仿真</button>
        <div class="template-sim-result" style="margin-top:8px;"></div>
      </div>
    `).join('');

    grid.querySelectorAll('.btn-template-run').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const card = e.target.closest('.template-card');
        const templateId = card.dataset.id;
        const resultDiv = card.querySelector('.template-sim-result');
        resultDiv.innerHTML = '<span style="color:var(--text-dim);">计算中...</span>';

        // Build component list from params
        const inputs = card.querySelectorAll('input');
        const params = {};
        inputs.forEach(inp => {
          const key = inp.className.replace('tp-', '');
          params[key] = parseFloat(inp.value);
        });

        // Get template and update component values
        try {
          const template = await API.get(`/api/templates/${templateId}`);
          const comps = template.components.map(c => {
            const updated = { ...c };
            // Update value if it's in params
            for (const [k, v] of Object.entries(params)) {
              if (c.name === k) updated.value = v;
            }
            return updated;
          });

          const analysisType = document.getElementById('analysis-type')?.value || 'dc';
          const body = { components: comps, analysis_type: analysisType };
          if (analysisType === 'ac') body.frequency = parseFloat(document.getElementById('ac-freq')?.value) || 1000;

          const simData = await API.post('/api/simulate', body);
          let resultHtml = '';
          if (simData.type === 'dc' && simData.node_voltages) {
            resultHtml = '<table class="result-table"><tr><th>节点</th><th>电压</th></tr>';
            for (const [n, v] of Object.entries(simData.node_voltages)) {
              resultHtml += `<tr><td>${n}</td><td style="color:var(--cyan);">${v}V</td></tr>`;
            }
            resultHtml += '</table>';
          } else if (simData.type === 'ac' && simData.node_voltages) {
            resultHtml = '<table class="result-table"><tr><th>节点</th><th>幅值</th><th>相位</th></tr>';
            for (const [n, v] of Object.entries(simData.node_voltages)) {
              resultHtml += `<tr><td>${n}</td><td>${v.magnitude}V</td><td>${v.phase_deg}°</td></tr>`;
            }
            resultHtml += '</table>';
          } else {
            resultHtml = '<pre style="font-size:12px;">' + JSON.stringify(simData, null, 2) + '</pre>';
          }
          resultDiv.innerHTML = resultHtml;
        } catch (err) {
          resultDiv.innerHTML = `<span style="color:var(--red);">${err.message}</span>`;
        }
      });
    });
  } catch (err) {
    // Templates load failed - non-critical
  }
}

// ============================================================
// TOOLS
// ============================================================
function initTools() {
  document.querySelectorAll('.tool-card').forEach(card => {
    const btn = card.querySelector('.tool-calc');
    const tool = card.dataset.tool;
    if (!btn || !tool) return;

    btn.addEventListener('click', async () => {
      const inputs = card.querySelectorAll('input');
      const params = {};
      inputs.forEach(inp => {
        const val = inp.value.trim();
        if (val) {
          const cls = Array.from(inp.classList).find(c => c.startsWith('tool-'));
          if (cls) params[cls.replace('tool-', '')] = parseFloat(val);
        }
      });
      const resultDiv = card.querySelector('.tool-result');
      resultDiv.textContent = '计算中...';
      try {
        const data = await API.post('/api/tools', { tool, params });
        let txt = data.formula + '\n';
        for (const [k, v] of Object.entries(data.result)) {
          if (typeof v === 'number') txt += `${k} = ${v.toFixed(4)}\n`;
          else txt += `${k}: ${v}\n`;
        }
        resultDiv.textContent = txt;
        resultDiv.style.whiteSpace = 'pre-line';
      } catch (err) {
        resultDiv.textContent = '错误: ' + err.message;
      }
    });

    // Real-time resistor color code
    if (tool === 'resistor_color') {
      card.querySelectorAll('.band').forEach(select => {
        select.addEventListener('change', () => {
          const b1 = card.querySelector('[data-band="1"]').value;
          const b2 = card.querySelector('[data-band="2"]').value;
          const b3 = card.querySelector('[data-band="3"]').value;
          const b4 = card.querySelector('[data-band="4"]').value;
          if (b1 && b2 && b3) {
            const val = (parseInt(b1) * 10 + parseInt(b2)) * parseFloat(b3);
            const tol = b4 ? ` ±${b4}%` : '';
            card.querySelector('.resistor-result').textContent = `电阻值: ${val}Ω${tol}`;
          }
        });
      });
    }
  });
}

// ============================================================
// UTILS
// ============================================================
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function timeAgo(dateStr) {
  const diff = Date.now() - new Date(dateStr + 'Z').getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return '刚刚';
  if (mins < 60) return `${mins}分钟前`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}小时前`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}天前`;
  return new Date(dateStr).toLocaleDateString('zh-CN');
}

// ============================================================
// INIT
// ============================================================
function loadApp() {
  initNavigation();
  initChat();
  loadChatHistory();
  initCommunity();
  initSimulator();
  initTools();
}

// Start
initAuth();
