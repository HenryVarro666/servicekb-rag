/*
 * servicekb 网页问答（/ui）的页面逻辑。原生 JavaScript，不依赖任何框架和构建工具。
 *
 * 读代码的顺序：
 *   1. 工具函数：el()（安全地创建页面元素）、call()（调用 API + 统一处理错误）
 *   2. 设置：API Key / 用户组（存在浏览器里，随请求头发送）
 *   3. 提问：文字问答走 /v1/ask，带照片走 /v1/photo/ask
 *   4. 渲染回答：结构化事实、答案正文（[n] 变成可点击的引用）、引用来源、识别结果、反馈
 *
 * 安全：服务端返回的所有文字都用 textContent / 文本节点插入，**从不使用 innerHTML**——
 * 证据来自工单、聊天记录，里面可能夹带任何内容，不能让它被浏览器当成 HTML / 脚本执行。
 */
"use strict";

// ---------------------------------------------------------------- 1. 工具函数

// 页面地址是 /ui/，API 在它的上一级（/v1/ask）。用相对地址，部署在网关的子路径下也能用。
const BASE = new URL("../", window.location.href);
const apiUrl = (path) => new URL(path, BASE).toString();

const DOC_TYPES = [
  ["manual", "手册"], ["bulletin", "服务通告"], ["qa", "历史工单"],
  ["part", "物料主数据"], ["chat", "聊天记录提炼"], ["record", "维修记录"],
];
const EXAMPLES = [
  "AS-110K-HV 早上报 E037，中午自动恢复，怎么处理？",
  "310-04521-A 还能订到吗？",
  "2022 年出厂的 AS-110K-HV 换新风扇还需要什么配件？",
  "After replacing the control board on AS-110K-HV we get E077",
];
// 拒答 / 降级原因（服务端的 reason 字段）→ 给现场人员看的说明
const REASONS = {
  no_results: "没有检索到相关资料",
  low_rerank_score: "检索到的资料和问题相关度太低",
  low_keyword_coverage: "问题里的关键信息在资料里没有出现",
  no_valid_citations: "回答没有给出有效的引用",
};
const MAX_UPLOAD = 15 * 1024 * 1024;

/** 创建元素。children 里的字符串一律作为纯文本插入（安全）。 */
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : String(c));
  }
  return node;
}

class ApiError extends Error {
  constructor(status, message, retryAfter = null, requestId = null) {
    super(message);
    this.status = status;
    this.retryAfter = retryAfter;
    this.requestId = requestId;
  }
}

function describeError(status, body) {
  const detail = body && body.detail;
  const text = typeof detail === "string" ? detail : "";
  if (status === 401) return "API Key 无效或没有填写：请点右上角“设置”填写。";
  if (status === 413) return "照片太大（最大 15 MB）。";
  if (status === 422) {
    if (Array.isArray(detail)) {
      return "输入不符合要求：" + detail.map((d) => `${(d.loc || []).slice(-1)[0] ?? ""} ${d.msg || ""}`.trim()).join("；");
    }
    return "输入不符合要求" + (text ? "：" + text : "");
  }
  if (status === 503) return "服务暂时繁忙或不可用，请稍后重试" + (text ? `（${text}）` : "") + "。";
  return `服务出错（HTTP ${status}）` + (text ? "：" + text : "");
}

/** 调用 API：返回 {body, requestId}；失败时抛出 ApiError（带可读的中文说明）。 */
async function call(path, options = {}) {
  let resp;
  try {
    resp = await fetch(apiUrl(path), options);
  } catch {
    throw new ApiError(0, "无法连接服务器：请检查网络，或确认服务是否在运行。");
  }
  const requestId = resp.headers.get("X-Request-ID");
  const raw = await resp.text();
  let body = null;
  try { body = raw ? JSON.parse(raw) : null; } catch { body = null; }
  if (!resp.ok) throw new ApiError(resp.status, describeError(resp.status, body), resp.headers.get("Retry-After"), requestId);
  return { body, requestId };
}

// ---------------------------------------------------------------- 2. 设置（API Key / 用户组）

// API Key 只放 sessionStorage（关闭标签页即清除）；用户组放 localStorage（下次打开还在）。
// 浏览器禁用存储时（隐私模式等）读写会抛异常，这里一律兜底，不影响使用。
const prefs = {
  get apiKey() { try { return sessionStorage.getItem("servicekb.apiKey") || ""; } catch { return ""; } },
  set apiKey(v) { try { v ? sessionStorage.setItem("servicekb.apiKey", v) : sessionStorage.removeItem("servicekb.apiKey"); } catch { /* 忽略 */ } },
  get groups() {
    try { return localStorage.getItem("servicekb.groups") ?? "public,service"; } catch { return "public,service"; }
  },
  set groups(v) { try { localStorage.setItem("servicekb.groups", v); } catch { /* 忽略 */ } },
};

function authHeaders(json) {
  const h = {};
  if (json) h["Content-Type"] = "application/json";
  if (prefs.apiKey) h["X-API-Key"] = prefs.apiKey;
  const groups = prefs.groups.trim();
  if (groups) h["X-User-Groups"] = groups;
  return h;
}

const $ = (id) => document.getElementById(id);
const settings = $("settings");

function openSettings() {
  $("api-key").value = prefs.apiKey;
  $("user-groups").value = prefs.groups;
  settings.showModal();
}

$("open-settings").addEventListener("click", openSettings);
settings.addEventListener("close", () => {
  if (settings.returnValue !== "save") return;
  const groups = $("user-groups").value.trim();
  // 请求头只能是 ASCII：用户组名请用英文 / 数字
  if (/[^\x20-\x7e]/.test(groups)) {
    alert("用户组只能包含英文字母、数字和常见符号（例如 public,service）。");
    return;
  }
  prefs.apiKey = $("api-key").value.trim();
  prefs.groups = groups;
  refreshStatus();
});

// ---------------------------------------------------------------- 服务状态

async function refreshStatus() {
  const s = $("status");
  try {
    const { body } = await call("readyz");
    const broken = Object.entries(body.llm || {}).filter(([, state]) => state !== "closed");
    s.className = broken.length ? "status status-warn" : "status status-ok";
    $("status-text").textContent = "就绪";
    // 手机上只显示"就绪"，细节藏起来（见 style.css 的 .status-detail）
    $("status-detail").textContent = broken.length ? " · 大模型熔断中" : ` · ${body.chunks ?? "?"} 个知识块`;
    s.title = [`向量模型：${body.embedding_model || "?"}`,
      ...Object.entries(body.llm || {}).map(([m, state]) => `${m}：${state}`)].join("\n");
  } catch (e) {
    s.className = "status status-down";
    $("status-text").textContent = "服务不可用";
    $("status-detail").textContent = "";
    s.title = e.message;
  }
}

async function loadConfig() {
  try {
    const { body } = await call("ui/config");
    $("auth-hint").textContent = body.auth_required
      ? "这个服务开启了鉴权，需要填写 API Key。"
      : "这个服务没有开启鉴权（本地开发模式），可以不填。";
    if (body.auth_required && !prefs.apiKey) openSettings();
  } catch { /* 配置接口不可用时不影响提问 */ }
}

// ---------------------------------------------------------------- 3. 提问

const form = $("ask-form");
const question = $("question");
const askBtn = $("ask-btn");
const results = $("results");
let photo = null; // {file, url}
let cardSeq = 0;

// 资料类型复选框 + 示例问题
for (const [value, label] of DOC_TYPES) {
  $("filter-types").append(el("label", {}, el("input", { type: "checkbox", value }), label));
}
for (const q of EXAMPLES) {
  $("examples").append(el("button", { type: "button", class: "chip", onclick: () => { question.value = q; submit(); } }, q));
}

function setPhoto(file) {
  if (photo) URL.revokeObjectURL(photo.url);
  photo = null;
  $("photo-preview").hidden = true;
  $("photo-input").value = "";
  if (!file) return;
  if (!file.type.startsWith("image/")) { alert("请选择图片文件。"); return; }
  if (file.size > MAX_UPLOAD) { alert("照片太大（最大 15 MB）。"); return; }
  photo = { file, url: URL.createObjectURL(file) };
  $("photo-thumb").src = photo.url;
  $("photo-name").textContent = `${file.name || "粘贴的图片"}（${(file.size / 1024 / 1024).toFixed(1)} MB）`;
  $("photo-preview").hidden = false;
}

$("photo-input").addEventListener("change", (e) => setPhoto(e.target.files[0]));
$("photo-remove").addEventListener("click", () => setPhoto(null));
question.addEventListener("paste", (e) => {
  const img = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith("image/"));
  if (img) setPhoto(img);
});
form.addEventListener("dragover", (e) => e.preventDefault());
form.addEventListener("drop", (e) => {
  const img = [...(e.dataTransfer?.files || [])].find((f) => f.type.startsWith("image/"));
  if (img) { e.preventDefault(); setPhoto(img); }
});
// Enter 发送、Shift+Enter 换行；中文输入法正在选字（isComposing）时按 Enter 不发送
question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(); }
});
form.addEventListener("submit", (e) => { e.preventDefault(); submit(); });

function setBusy(busy) {
  askBtn.disabled = busy;
  askBtn.textContent = busy ? "检索中…" : "提问";
}

async function submit() {
  if (askBtn.disabled) return;
  const q = question.value.trim();
  const shot = photo;
  if (!shot && q.length < 2) { question.focus(); return; }

  $("welcome").hidden = true;
  const card = newCard(q || "（只上传了照片）", shot?.url);
  const t0 = performance.now();
  const loading = el("div", { class: "loading" }, el("span", { class: "spinner", "aria-hidden": "true" }), el("span", { text: "正在检索知识库…" }));
  const tick = setInterval(() => {
    loading.lastChild.textContent = `正在检索知识库… ${((performance.now() - t0) / 1000).toFixed(0)} 秒`;
  }, 1000);
  card.body.append(loading);
  setBusy(true);
  try {
    if (shot) {
      const fd = new FormData();
      fd.append("image", shot.file, shot.file.name || "photo.jpg");
      fd.append("question", q);
      const { body, requestId } = await call("v1/photo/ask", { method: "POST", headers: authHeaders(false), body: fd });
      renderIdentification(card, body.identification);
      renderAnswer(card, body.answer, requestId);
      photo = null; // 卡片里还在用这张图的地址，不要 revoke
      $("photo-preview").hidden = true;
      $("photo-input").value = "";
    } else {
      const payload = { question: q };
      const model = $("filter-model").value.trim();
      if (model) payload.product_models = [model];
      const types = [...$("filter-types").querySelectorAll("input:checked")].map((i) => i.value);
      if (types.length) payload.doc_types = types;
      const { body, requestId } = await call("v1/ask", { method: "POST", headers: authHeaders(true), body: JSON.stringify(payload) });
      renderAnswer(card, body, requestId);
    }
    question.value = "";
  } catch (e) {
    renderError(card, e);
    if (e.status === 401) openSettings();
  } finally {
    clearInterval(tick);
    loading.remove();
    setBusy(false);
    question.focus();
  }
}

// ---------------------------------------------------------------- 4. 渲染

function newCard(q, imageUrl) {
  const id = ++cardSeq;
  const body = el("div");
  const root = el("article", { class: "card result" },
    el("h3", { class: "result-question" }, imageUrl ? el("img", { src: imageUrl, alt: "提问时上传的照片" }) : null, el("span", { text: q })),
    body);
  results.prepend(root);
  return { id, root, body };
}

function renderError(card, e) {
  const extra = [];
  if (e.retryAfter) extra.push(`建议 ${e.retryAfter} 秒后重试。`);
  if (e.requestId) extra.push(`请求 ID：${e.requestId}`);
  card.body.append(el("div", { class: "notice notice-error", role: "alert" }, e.message, extra.length ? el("div", { class: "hint", text: extra.join(" ") }) : null));
}

/** 极简 Markdown：段落、列表、**加粗**、`代码`、[n] 引用。逐个创建文本节点，不解析任何 HTML。 */
function renderMarkdown(text, makeCite) {
  const root = el("div", { class: "answer" });
  let list = null;
  let listType = null;
  let para = [];
  const flushPara = () => {
    if (!para.length) return;
    const p = el("p");
    para.forEach((line, i) => { if (i) p.append(el("br")); appendInline(p, line, makeCite); });
    root.append(p);
    para = [];
  };
  for (const raw of String(text || "").replace(/\r\n/g, "\n").split("\n")) {
    const line = raw.trimEnd();
    if (!line.trim()) { flushPara(); list = null; continue; }
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const ordered = line.match(/^\s*(\d+)[.)、）]\s*(.*)$/);
    if (bullet || ordered) {
      flushPara();
      const type = bullet ? "ul" : "ol";
      if (!list || listType !== type) {
        list = el(type);
        if (ordered && ordered[1] !== "1") list.setAttribute("start", ordered[1]);
        listType = type;
        root.append(list);
      }
      const li = el("li");
      appendInline(li, bullet ? bullet[1] : ordered[2], makeCite);
      list.append(li);
      continue;
    }
    list = null;
    const heading = line.match(/^\s*#{1,6}\s+(.*)$/);
    if (heading) {
      flushPara();
      const strong = el("strong");
      appendInline(strong, heading[1], makeCite);
      root.append(el("p", {}, strong));
      continue;
    }
    para.push(line);
  }
  flushPara();
  return root;
}

function appendInline(parent, text, makeCite) {
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\[\d+\])/g;
  let last = 0;
  let m;
  while ((m = re.exec(text))) {
    if (m.index > last) parent.append(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) parent.append(el("strong", { text: tok.slice(2, -2) }));
    else if (tok.startsWith("`")) parent.append(el("code", { text: tok.slice(1, -1) }));
    else parent.append(makeCite(Number(tok.slice(1, -1))) || tok);
    last = re.lastIndex;
  }
  if (last < text.length) parent.append(text.slice(last));
}

/** 非 http 地址怎么显示：服务器本地文件只显示文件名（浏览器打不开，完整路径也不该暴露给用户）。 */
function sourceLabel(uri) {
  if (/^file:/i.test(uri) || /^([a-z]:[\\/]|\/)/i.test(uri)) {
    const name = uri.split(/[\\/]/).filter(Boolean).pop() || uri;
    try { return `文件：${decodeURIComponent(name)}`; } catch { return `文件：${name}`; }
  }
  return uri; // 例如 ticket://TK-2024-0103：工单号本身就有用
}

function renderAnswer(card, a, requestId) {
  const c = card.body;
  const sourceIds = new Map();

  if (a.escalate) {
    const why = REASONS[(a.reason || "").split(":")[0]] || "";
    c.append(el("div", { class: "notice notice-warn", role: "status" },
      "知识库里没有找到足够的依据，已标记为需要升级（转技术支持工程师）。", why ? el("div", { class: "hint", text: `原因：${why}` }) : null));
  } else if ((a.reason || "").startsWith("llm_error")) {
    c.append(el("div", { class: "notice notice-warn", text: "大模型暂时不可用，下面是从证据里摘录的原文。" }));
  }

  // 离线抽取模式的答案正文里已经逐条列出了事实，这里只显示正文里没有的，避免重复
  const facts = (a.facts || []).filter((f) => !(a.answer || "").includes(f));
  if (facts.length) {
    c.append(el("div", { class: "facts" },
      el("div", { class: "facts-title", text: "结构化事实（来自物料主数据，最权威）" }),
      el("ul", {}, facts.map((f) => el("li", { text: f })))));
  }

  const showSource = (n) => {
    const target = document.getElementById(sourceIds.get(n));
    if (!target) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    target.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "nearest" });
    target.querySelector(".source-snippet")?.classList.remove("clamped");
    target.classList.add("flash");
    setTimeout(() => target.classList.remove("flash"), 1400);
  };
  for (const cit of a.citations || []) sourceIds.set(cit.index, `src-${card.id}-${cit.index}`);
  const makeCite = (n) => (sourceIds.has(n)
    ? el("button", { type: "button", class: "cite", "aria-label": `查看来源 ${n}`, onclick: () => showSource(n) }, String(n))
    : null);

  c.append(renderMarkdown(a.answer, makeCite));

  if (a.citations?.length) {
    const box = el("div", { class: "sources" }, el("div", { class: "section-title", text: "引用来源" }));
    for (const cit of a.citations) {
      const loc = [cit.section, cit.page ? `第 ${cit.page} 页` : ""].filter(Boolean).join(" · ");
      const snippet = el("div", { class: "source-snippet clamped", title: "点击展开 / 收起", text: cit.snippet || "" });
      snippet.addEventListener("click", () => snippet.classList.toggle("clamped"));
      const link = /^https?:\/\//i.test(cit.uri || "")
        ? el("a", { href: cit.uri, target: "_blank", rel: "noopener noreferrer", text: "打开原文" })
        : (cit.uri ? el("div", { class: "source-loc", text: sourceLabel(cit.uri) }) : null);
      box.append(el("div", { class: "source", id: sourceIds.get(cit.index) },
        el("div", { class: "source-head" }, el("span", { class: "cite-demo", text: String(cit.index) }),
          el("span", { class: "source-title", text: cit.title || cit.doc_ref }), loc ? el("span", { class: "source-loc", text: loc }) : null),
        snippet, link));
    }
    c.append(box);
  }

  const meta = el("div", { class: "meta" });
  if (a.model && a.model !== "none") meta.append(el("span", { text: `模型：${a.model}` }));
  else if (a.escalate) meta.append(el("span", { text: "证据不足，未调用大模型" }));
  if (a.timings_ms?.total != null) meta.append(el("span", { text: `耗时 ${(a.timings_ms.total / 1000).toFixed(1)} 秒` }));
  if (a.usage?.input_tokens != null) meta.append(el("span", { text: `token ${a.usage.input_tokens} + ${a.usage.output_tokens}` }));
  for (const d of a.degraded || []) meta.append(el("span", { class: "degraded", title: "某个依赖失败，已自动降级", text: `降级：${d}` }));
  if (requestId) meta.append(el("span", { title: "报障时请提供这个 ID", text: `请求 ID：${requestId}` }));
  c.append(meta);

  if (a.query_id) c.append(renderFeedback(a.query_id));
}

function renderIdentification(card, ident) {
  const box = el("div", { class: "ident" }, el("div", { class: "section-title", text: "照片识别" }));
  if (!ident.identified) {
    box.append(el("div", { class: "notice notice-warn", text: "没有在图库里识别出这个零件。请拍清楚铭牌或标签再试，或者直接输入料号。" }));
  }
  if (ident.candidates?.length) {
    const statusBadge = (s) => (s === "active" ? el("span", { class: "badge badge-ok", text: "在产" })
      : s === "obsolete" ? el("span", { class: "badge badge-bad", text: "停产" }) : el("span", { class: "badge", text: s || "—" }));
    const match = (cand) => (cand.ocr_confirmed ? "标签确认 ✓"
      : cand.visual_score != null ? `外观 ${Math.round(cand.visual_score * 100)}%` : "—");
    box.append(el("table", {},
      el("thead", {}, el("tr", {}, ["料号", "名称", "状态", "匹配", "依据"].map((h) => el("th", { text: h })))),
      el("tbody", {}, ident.candidates.slice(0, 5).map((cand) => el("tr", {},
        el("td", {}, el("b", { text: cand.part_number })), el("td", { text: cand.name || "" }),
        el("td", {}, statusBadge(cand.status)), el("td", { text: match(cand) }),
        el("td", { text: (cand.evidence || []).join("；") })))),
    ));
  }
  for (const w of ident.warnings || []) box.append(el("div", { class: "notice notice-warn", text: w }));
  const extra = [];
  if (ident.serial_number) extra.push(`序列号 ${ident.serial_number}`);
  if (ident.manufacture_date) extra.push(`出厂 ${ident.manufacture_date}`);
  if (extra.length) box.append(el("div", { class: "ident-extra", text: extra.join(" · ") }));
  card.body.append(box);
}

function renderFeedback(queryId) {
  const wrap = el("div");
  const bar = el("div", { class: "feedback" });
  const done = (msg) => { wrap.replaceChildren(el("div", { class: "feedback-done", text: msg })); };

  async function send(rating, comment, corrected) {
    try {
      await call("v1/feedback", {
        method: "POST", headers: authHeaders(true),
        body: JSON.stringify({ query_id: queryId, rating, comment: comment || null, corrected_answer: corrected || null }),
      });
      done(corrected ? "已提交修正，谢谢！专家修正会用来改进知识库。" : "已提交反馈，谢谢！");
    } catch (e) {
      wrap.append(el("div", { class: "notice notice-error", text: e.message }));
    }
  }

  function showForm(focusCorrected) {
    const comment = el("textarea", { maxlength: "2000", placeholder: "哪里不对？（可选）例如：料号写错了 / 步骤不完整 / 引用的资料过期了" });
    const corrected = el("textarea", { maxlength: "10000", placeholder: "正确的答案是什么？（可选，专家填写——会沉淀为精选问答，并进入评估集）" });
    const f = el("form", { class: "feedback-form" },
      comment, corrected,
      el("div", { class: "feedback" },
        el("button", { type: "submit", class: "btn btn-primary", text: "提交" }),
        el("button", { type: "button", class: "btn btn-secondary", text: "取消", onclick: () => wrap.replaceChildren(bar) })));
    f.addEventListener("submit", (e) => { e.preventDefault(); send(-1, comment.value.trim(), corrected.value.trim()); });
    wrap.replaceChildren(f);
    (focusCorrected ? corrected : comment).focus();
  }

  bar.append(
    el("button", { type: "button", class: "btn btn-secondary", onclick: () => send(1) }, "👍 有用"),
    el("button", { type: "button", class: "btn btn-secondary", onclick: () => showForm(false) }, "👎 没用"),
    el("button", { type: "button", class: "btn btn-secondary", onclick: () => showForm(true) }, "✎ 纠正答案"),
  );
  wrap.append(bar);
  return wrap;
}

// ---------------------------------------------------------------- 启动

loadConfig();
refreshStatus();
setInterval(refreshStatus, 60_000);
question.focus();
