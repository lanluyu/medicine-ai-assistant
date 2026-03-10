/* ── 药品知识库 AI 助手 — 前端逻辑 ────────────────────────────────── */
"use strict";

// ── Marked 配置（Markdown 渲染）──────────────────────────────────────
if (typeof marked !== "undefined") {
  marked.setOptions({
    breaks: true,
    gfm: true,
    highlight: function (code, lang) {
      if (typeof hljs !== "undefined" && lang && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value;
      }
      return code;
    },
  });
}

function renderMarkdown(text) {
  if (typeof marked !== "undefined") {
    return marked.parse(text);
  }
  // 降级：换行转 <br>
  return text.replace(/\n/g, "<br>");
}

// ── 状态 ─────────────────────────────────────────────────────────────
const state = {
  threadId: "",        // 当前会话 ID
  streaming: false,    // 是否正在流式输出
  abortCtrl: null,     // AbortController（停止流）
  sessions: [],        // [{id, title, ts}] 历史会话（localStorage）
};

// ── DOM 引用 ──────────────────────────────────────────────────────────
const $messages      = document.getElementById("messages");
const $userInput     = document.getElementById("userInput");
const $sendBtn       = document.getElementById("sendBtn");
const $welcome       = document.getElementById("welcome");
const $sessionList   = document.getElementById("sessionList");
const $newChatBtn    = document.getElementById("newChatBtn");
const $sidebar       = document.getElementById("sidebar");
const $sidebarToggle = document.getElementById("sidebarToggle");
const $mobileSidebar = document.getElementById("mobileSidebarBtn");

// ── 初始化 ────────────────────────────────────────────────────────────
(function init() {
  loadSessions();
  renderSessionList();
  startNewChat();

  // 示例按钮
  document.querySelectorAll(".example-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const q = btn.dataset.q;
      if (q) {
        $userInput.value = q;
        autoResize($userInput);
        updateSendBtn();
        sendMessage();
      }
    });
  });

  // 输入事件
  $userInput.addEventListener("input", () => {
    autoResize($userInput);
    updateSendBtn();
  });

  $userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !state.streaming) {
      e.preventDefault();
      sendMessage();
    }
  });

  $sendBtn.addEventListener("click", () => {
    if (state.streaming) {
      stopStream();
    } else {
      sendMessage();
    }
  });

  $newChatBtn.addEventListener("click", startNewChat);
  $sidebarToggle.addEventListener("click", toggleSidebar);
  if ($mobileSidebar) $mobileSidebar.addEventListener("click", toggleSidebar);
})();

// ── 新建对话 ──────────────────────────────────────────────────────────
function startNewChat() {
  state.threadId = "";
  state.streaming = false;

  // 清空消息区，显示欢迎屏
  $messages.innerHTML = "";
  $welcome.style.display = "flex";
  $messages.appendChild($welcome);

  $userInput.value = "";
  autoResize($userInput);
  updateSendBtn();
  $userInput.focus();

  renderSessionList();
}

// ── 发送消息 ──────────────────────────────────────────────────────────
async function sendMessage() {
  const text = $userInput.value.trim();
  if (!text || state.streaming) return;

  // 隐藏欢迎屏
  if ($welcome.style.display !== "none") {
    $welcome.style.display = "none";
  }

  // 追加用户消息
  appendMessage("user", text);
  $userInput.value = "";
  autoResize($userInput);
  updateSendBtn();

  // 如果是新会话中第一条消息，保存标题
  const isNewSession = !state.threadId;

  // 创建 AI 消息占位
  const aiRow = createAssistantRow();
  $messages.appendChild(aiRow);
  scrollToBottom();

  // 开始流式请求
  state.streaming = true;
  showStopBtn(true);
  updateSendBtn();

  state.abortCtrl = new AbortController();
  let buffer = "";  // 累积 token 用于 Markdown 渲染

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, thread_id: state.threadId }),
      signal: state.abortCtrl.signal,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let remainder = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = remainder + decoder.decode(value, { stream: true });
      const lines = chunk.split("\n");
      remainder = lines.pop(); // 最后一行可能不完整

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6).trim();

        if (data === "[DONE]") break;

        let parsed;
        try { parsed = JSON.parse(data); } catch { continue; }

        if (parsed.type === "meta") {
          // 服务器返回 thread_id
          if (parsed.thread_id) {
            state.threadId = parsed.thread_id;
            if (isNewSession) {
              saveSession(state.threadId, text);
              renderSessionList();
            }
          }
        } else if (parsed.type === "token" && parsed.token) {
          buffer += parsed.token;
          updateAssistantContent(aiRow, buffer);
          scrollToBottom();
        } else if (parsed.type === "error") {
          buffer += `\n\n**错误：** ${parsed.error}`;
          updateAssistantContent(aiRow, buffer);
        }
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      buffer += `\n\n**网络错误：** ${err.message}`;
      updateAssistantContent(aiRow, buffer);
    }
  } finally {
    state.streaming = false;
    state.abortCtrl = null;
    showStopBtn(false);
    updateSendBtn();
    // 移除打字光标
    const cursor = aiRow.querySelector(".cursor");
    if (cursor) cursor.remove();
    scrollToBottom();
  }
}

// ── 停止流 ────────────────────────────────────────────────────────────
function stopStream() {
  if (state.abortCtrl) {
    state.abortCtrl.abort();
  }
}

// ── DOM 辅助：追加用户消息 ────────────────────────────────────────────
function appendMessage(role, text) {
  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.textContent = role === "user" ? "你" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";

  if (role === "user") {
    bubble.textContent = text;
  } else {
    bubble.innerHTML = renderMarkdown(text);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  $messages.appendChild(row);
  scrollToBottom();
  return row;
}

// ── DOM 辅助：创建 AI 消息行（带光标）────────────────────────────────
function createAssistantRow() {
  const row = document.createElement("div");
  row.className = "msg-row assistant";

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.textContent = "AI";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";

  const cursor = document.createElement("span");
  cursor.className = "cursor";
  bubble.appendChild(cursor);

  row.appendChild(avatar);
  row.appendChild(bubble);
  return row;
}

// ── DOM 辅助：更新 AI 消息内容 ────────────────────────────────────────
function updateAssistantContent(row, text) {
  const bubble = row.querySelector(".msg-bubble");
  if (!bubble) return;
  // 渲染 Markdown
  bubble.innerHTML = renderMarkdown(text);
  // 重新追加光标
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  bubble.appendChild(cursor);
}

// ── 会话管理（localStorage）──────────────────────────────────────────
const SESSION_KEY = "medicine_sessions";
const MAX_SESSIONS = 30;

function loadSessions() {
  try {
    state.sessions = JSON.parse(localStorage.getItem(SESSION_KEY) || "[]");
  } catch { state.sessions = []; }
}

function saveSession(threadId, firstMessage) {
  const title = firstMessage.length > 28
    ? firstMessage.slice(0, 28) + "…"
    : firstMessage;
  // 去重
  state.sessions = state.sessions.filter((s) => s.id !== threadId);
  state.sessions.unshift({ id: threadId, title, ts: Date.now() });
  if (state.sessions.length > MAX_SESSIONS) {
    state.sessions = state.sessions.slice(0, MAX_SESSIONS);
  }
  localStorage.setItem(SESSION_KEY, JSON.stringify(state.sessions));
}

function renderSessionList() {
  $sessionList.innerHTML = "";

  if (state.sessions.length === 0) {
    const empty = document.createElement("div");
    empty.style.cssText = "padding:20px 10px;text-align:center;font-size:12px;color:var(--text-small)";
    empty.textContent = "暂无历史对话";
    $sessionList.appendChild(empty);
    return;
  }

  const label = document.createElement("div");
  label.className = "session-list-label";
  label.textContent = "历史对话";
  $sessionList.appendChild(label);

  state.sessions.forEach((session) => {
    const item = document.createElement("div");
    item.className = "session-item" + (session.id === state.threadId ? " active" : "");
    item.title = session.title;

    const icon = document.createElement("span");
    icon.className = "session-icon";
    icon.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
    </svg>`;

    const text = document.createElement("span");
    text.textContent = session.title;
    text.style.overflow = "hidden";
    text.style.textOverflow = "ellipsis";

    item.appendChild(icon);
    item.appendChild(text);

    item.addEventListener("click", () => {
      // 切换到已有会话（仅记录 thread_id，实际历史由服务端 checkpointer 保存）
      state.threadId = session.id;
      startNewChat();
      state.threadId = session.id;  // startNewChat 会清空，重新赋值
      renderSessionList();
      // 更新 active 样式
      document.querySelectorAll(".session-item").forEach((el) => el.classList.remove("active"));
      item.classList.add("active");
    });

    $sessionList.appendChild(item);
  });
}

// ── 侧边栏开关 ───────────────────────────────────────────────────────
function toggleSidebar() {
  $sidebar.classList.toggle("collapsed");
}

// ── 停止按钮显示 ─────────────────────────────────────────────────────
let $stopBtn = null;
function showStopBtn(visible) {
  if (!$stopBtn) {
    $stopBtn = document.createElement("button");
    $stopBtn.className = "stop-btn";
    $stopBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <rect x="4" y="4" width="16" height="16" rx="2"/>
    </svg> 停止生成`;
    $stopBtn.addEventListener("click", stopStream);
    document.querySelector(".input-area").insertBefore($stopBtn, document.querySelector(".input-tip"));
  }
  $stopBtn.classList.toggle("visible", visible);
}

// ── 工具函数 ──────────────────────────────────────────────────────────
function scrollToBottom() {
  $messages.scrollTop = $messages.scrollHeight;
}

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 200) + "px";
}

function updateSendBtn() {
  const hasText = $userInput.value.trim().length > 0;
  $sendBtn.disabled = !hasText && !state.streaming;
  if (state.streaming) {
    $sendBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <rect x="4" y="4" width="16" height="16" rx="2"/>
    </svg>`;
    $sendBtn.disabled = false;
    $sendBtn.title = "停止生成";
  } else {
    $sendBtn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
    </svg>`;
    $sendBtn.title = "发送";
  }
}
