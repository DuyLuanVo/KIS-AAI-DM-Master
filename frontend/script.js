/* KIS Architect · giao diện tìm kiếm và nạp ảnh */
'use strict';

// Khi giao diện được backend phục vụ (cổng 8000) thì gọi cùng origin; nếu mở bằng http.server/file:// thì gọi localhost:8000
const API = window.KIS_API || (
  location.protocol === 'file:' ? 'http://localhost:8000'
    : (location.port && location.port !== '8000' ? `${location.protocol}//${location.hostname}:8000` : location.origin)
);
const PAGE = 24;
const MIN_CAPTION = 10;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const mediaUrl = (u) => (u && !/^https?:/.test(u) ? API + u : u);
const fmtNum = (n) => new Intl.NumberFormat('vi-VN').format(n);

// ============================================================ chung
async function api(path, opts = {}) {
  const res = await fetch(API + path, opts);
  let body = null;
  try { body = await res.json(); } catch { /* không phải JSON */ }
  if (!res.ok) {
    const msg = body && (body.error || body.detail);
    throw new Error(typeof msg === 'string' ? msg : `Máy chủ trả về lỗi ${res.status}`);
  }
  return body;
}

let toastTimer;
function toast(msg, bad = false) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.toggle('bad', bad);
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 4200);
}

// Tách từ giống backend (chữ thường, bỏ đuôi số nhiều) để tô vàng từ khớp trong caption
function stem(w) {
  w = w.toLowerCase();
  if (w.length > 4 && w.endsWith('ies')) return w.slice(0, -3) + 'y';
  if (w.length > 3 && w.endsWith('s') && !w.endsWith('ss')) return w.slice(0, -1);
  return w;
}
function highlight(text, terms) {
  const set = new Set((terms || []).map(stem));
  if (!set.size) return esc(text);
  return String(text || '').split(/([\p{L}\p{N}]+)/u)
    .map((part, i) => (i % 2 === 1 && set.has(stem(part)) ? `<mark>${esc(part)}</mark>` : esc(part)))
    .join('');
}

function setupDropzone(zone, input, onFiles) {
  zone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  ['dragenter', 'dragover'].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add('is-over'); }));
  ['dragleave', 'drop'].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove('is-over'); }));
  zone.addEventListener('drop', (e) => onFiles([...e.dataTransfer.files]));
  input.addEventListener('change', () => { onFiles([...input.files]); input.value = ''; });
}

// ============================================================ tabs
function showView(name) {
  const isSearch = name === 'search';
  $('viewSearch').hidden = !isSearch;
  $('viewIngest').hidden = isSearch;
  $('tabSearch').classList.toggle('is-active', isSearch);
  $('tabIngest').classList.toggle('is-active', !isSearch);
  $('tabSearch').setAttribute('aria-selected', isSearch);
  $('tabIngest').setAttribute('aria-selected', !isSearch);
}
$('tabSearch').addEventListener('click', () => showView('search'));
$('tabIngest').addEventListener('click', () => showView('ingest'));

// ============================================================ thống kê
let stats = null;
async function loadStats() {
  try {
    stats = await api('/api/v1/stats');
    const up = stats.by_source?.upload || 0;
    $('statsLine').textContent = `${fmtNum(stats.total_images)} ảnh có caption trong kho` + (up ? `, trong đó ${fmtNum(up)} ảnh tải lên` : '');
    renderLabelChips();
  } catch (e) {
    $('statsLine').textContent = `Không kết nối được máy chủ tại ${API}. Hãy khởi động backend.`;
  }
}

// ============================================================ TÌM KIẾM
const S = { mode: 'text', results: [], shown: 0, last: null, labels: new Set(), queryImage: null, queryImageUrl: null };

document.querySelectorAll('.mode').forEach((btn) => btn.addEventListener('click', () => setMode(btn.dataset.mode)));
function setMode(mode) {
  S.mode = mode;
  document.querySelectorAll('.mode').forEach((b) => {
    const on = b.dataset.mode === mode;
    b.classList.toggle('is-active', on);
    b.setAttribute('aria-checked', on);
  });
  $('textForm').hidden = mode !== 'text';
  $('textHints').hidden = mode !== 'text';
  $('imageForm').hidden = mode !== 'image';
  if (mode === 'text') $('queryInput').focus();
}

$('textForm').addEventListener('submit', (e) => { e.preventDefault(); runTextSearch(); });
document.querySelectorAll('.hint').forEach((h) => h.addEventListener('click', () => { $('queryInput').value = h.textContent; runTextSearch(); }));
$('sourceFilter').addEventListener('change', rerun);

function renderLabelChips() {
  const labels = stats?.object_labels || [];
  $('labelFilter').hidden = labels.length === 0;
  $('labelChips').innerHTML = labels.slice(0, 16).map((l) =>
    `<button class="chip${S.labels.has(l.label) ? ' is-on' : ''}" type="button" data-label="${esc(l.label)}" aria-pressed="${S.labels.has(l.label)}">${esc(l.label)} <small>${l.count}</small></button>`).join('');
  $('labelChips').querySelectorAll('.chip').forEach((c) => c.addEventListener('click', () => {
    const l = c.dataset.label;
    S.labels.has(l) ? S.labels.delete(l) : S.labels.add(l);
    renderLabelChips();
    rerun();
  }));
}

function filters() {
  return {
    source: $('sourceFilter').value || null,
    object_labels: S.labels.size ? [...S.labels] : null,
  };
}

function rerun() {
  if (!S.last) return;
  if (S.last.type === 'text') runTextSearch();
  else if (S.last.type === 'image') runImageSearch();
}

async function runTextSearch() {
  const q = $('queryInput').value.trim();
  if (!q) { $('queryInput').focus(); toast('Nhập mô tả cần tìm, ví dụ: brick tower with a stepped roof'); return; }
  S.last = { type: 'text', query: q };
  showSkeleton();
  try {
    const d = await api('/api/v1/search/text', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q, limit: 120, mode: $('textMode').value, ...filters() }),
    });
    renderResults(d);
  } catch (e) { renderError(e); }
}

async function runImageSearch() {
  if (!S.queryImage) { toast('Chọn hoặc dán một ảnh để tìm.'); return; }
  S.last = { type: 'image' };
  showSkeleton();
  try {
    const d = await api('/api/v1/search/image', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: S.queryImage, limit: 120, ...filters() }),
    });
    renderResults(d);
  } catch (e) { renderError(e); }
}

async function runSimilar(hit) {
  S.last = { type: 'similar', hit };
  showView('search');
  showSkeleton();
  try {
    const d = await api(`/api/v1/images/similar?key=${encodeURIComponent(hit.key)}&limit=120`);
    d.from = hit;
    renderResults(d);
  } catch (e) { renderError(e); }
}

// ---- ảnh truy vấn
function setQueryImage(file) {
  if (!file || !file.type.startsWith('image/')) { toast('Tệp này không phải ảnh.', true); return; }
  const reader = new FileReader();
  reader.onload = () => {
    S.queryImage = reader.result;
    S.queryImageUrl = reader.result;
    $('queryPreview').src = reader.result;
    $('queryPreview').hidden = false;
    $('queryDropText').innerHTML = `<small>${esc(file.name || 'Ảnh dán từ bộ nhớ tạm')}</small>`;
    $('imageSearchBtn').disabled = false;
    $('clearQueryImage').hidden = false;
    runImageSearch();
  };
  reader.readAsDataURL(file);
}
setupDropzone($('queryDrop'), $('queryFile'), (files) => setQueryImage(files.find((f) => f.type.startsWith('image/'))));
$('imageSearchBtn').addEventListener('click', runImageSearch);
$('clearQueryImage').addEventListener('click', () => {
  S.queryImage = null;
  $('queryPreview').hidden = true;
  $('queryDropText').innerHTML = '<strong>Kéo ảnh vào đây</strong>, bấm để chọn, hoặc dán bằng Ctrl+V';
  $('imageSearchBtn').disabled = true;
  $('clearQueryImage').hidden = true;
});
document.addEventListener('paste', (e) => {
  if ($('viewSearch').hidden || $('detail').open) return;
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
  if (!item) return;
  e.preventDefault();
  setMode('image');
  setQueryImage(item.getAsFile());
});

// ---- hiển thị kết quả
function showSkeleton() {
  $('resultsEmpty').hidden = true;
  $('loadMore').hidden = true;
  $('resultsMeta').innerHTML = '<span class="sub">Đang tìm…</span>';
  $('resultsGrid').innerHTML = Array.from({ length: 8 }, () =>
    '<div class="card skeleton"><div class="card-media"></div><div class="card-body"><div class="line"></div><div class="line" style="width:70%"></div></div></div>').join('');
}

function renderError(e) {
  $('resultsGrid').innerHTML = '';
  $('resultsMeta').innerHTML = '';
  $('resultsEmpty').hidden = false;
  $('resultsEmpty').innerHTML = `<p class="empty-title">Không tìm được</p><p>${esc(e.message)}</p>`;
}

function renderResults(d) {
  S.results = d.results || [];
  S.terms = d.query_terms || [];
  S.type = d.search_type;
  S.shown = 0;
  $('resultsGrid').innerHTML = '';

  const n = S.results.length;
  let head;
  if (d.search_type === 'text') {
    head = `<span class="count">${fmtNum(n)} ảnh có caption khớp</span><span class="sub">“${esc(d.query)}”</span>`;
  } else if (d.search_type === 'image') {
    head = `<span class="query-chip">${S.queryImageUrl ? `<img src="${S.queryImageUrl}" alt="">` : ''}<span class="count">${fmtNum(n)} ảnh, xếp theo độ tương đồng</span></span>`;
  } else if (d.search_type === 'browse') {
    head = `<span class="count">${fmtNum(n)} ảnh vừa nạp</span><span class="sub">${esc(d.batch_name || d.query)}</span>`;
  } else {
    head = `<span class="query-chip"><img src="${esc(mediaUrl(d.from.image_url))}" alt=""><span class="count">${fmtNum(n)} ảnh giống nhất với</span></span><span class="sub">${esc(d.from.title)}</span>`;
  }
  $('resultsMeta').innerHTML = `${head}<span class="sub">${Math.round(d.query_time_ms)} ms</span>`;

  if (!n) {
    $('resultsEmpty').hidden = false;
    $('resultsEmpty').innerHTML = d.search_type === 'text'
      ? '<p class="empty-title">Không có caption nào khớp</p><p>Caption trong kho viết bằng tiếng Anh. Thử từ khóa khác như “temple gate”, “tiled roof”, hoặc bỏ bớt bộ lọc.</p>'
      : '<p class="empty-title">Không có ảnh phù hợp</p><p>Thử bỏ bớt bộ lọc nguồn hoặc cấu kiện.</p>';
    return;
  }
  $('resultsEmpty').hidden = true;
  showMore();
}

function showMore() {
  const frag = document.createDocumentFragment();
  S.results.slice(S.shown, S.shown + PAGE).forEach((hit) => frag.appendChild(card(hit)));
  $('resultsGrid').appendChild(frag);
  S.shown = Math.min(S.shown + PAGE, S.results.length);
  $('loadMore').hidden = S.shown >= S.results.length;
  $('loadMore').textContent = `Hiện thêm kết quả (còn ${S.results.length - S.shown})`;
}
$('loadMore').addEventListener('click', showMore);

function scoreText(hit) {
  if (hit.similarity == null) return '';
  return `${Math.round(Math.max(0, hit.similarity) * 100)}%`;
}

function matchBadges(hit) {
  if (S.type !== 'text') return '';
  const out = [];
  if (hit.match_sources.includes('keyword')) out.push('<span class="kw">từ khóa</span>');
  if (hit.match_sources.includes('semantic')) out.push('<span class="sem">ngữ nghĩa</span>');
  return `<span class="match">${out.join('')}</span>`;
}

function card(hit) {
  const el = document.createElement('button');
  el.type = 'button';
  el.className = 'card';
  el.setAttribute('aria-label', `${hit.title}. ${hit.caption}`);
  el.innerHTML = `
    <div class="card-media">
      <img src="${esc(mediaUrl(hit.image_url))}" alt="" loading="lazy">
      <span class="card-rank">${hit.rank}</span>
    </div>
    <div class="card-body">
      <p class="card-title">${esc(hit.title)}</p>
      <p class="caption-text">${highlight(hit.caption, S.type === 'text' ? (hit.matched_terms.length ? hit.matched_terms : S.terms) : [])}</p>
      <div class="card-foot">${matchBadges(hit)}<span class="score" title="Độ tương đồng cosine">${scoreText(hit)}</span></div>
    </div>`;
  el.addEventListener('click', () => openDetail(hit));
  return el;
}

// ============================================================ chi tiết
function openDetail(hit) {
  const dlg = $('detail');
  $('detailImg').src = mediaUrl(hit.image_url);
  $('detailImg').alt = hit.caption;
  $('detailTitle').textContent = hit.title;
  const parts = [];
  if (hit.similarity != null) parts.push(`Độ tương đồng ${scoreText(hit)}`);
  if (hit.matched_terms?.length) parts.push(`khớp từ: ${hit.matched_terms.join(', ')}`);
  $('detailScore').textContent = parts.join('; ');
  $('detailCaption').innerHTML = highlight(hit.caption, S.type === 'text' ? (hit.matched_terms.length ? hit.matched_terms : S.terms) : []);

  const rows = [
    ['Tệp', hit.file_name],
    ['Nguồn', hit.source === 'upload' ? `Tải lên (${hit.batch_name || hit.batch_id})` : 'Bộ dữ liệu Wikimedia Commons'],
    ['Thời gian', hit.date],
    ['Tác giả', hit.author],
    ['Giấy phép', hit.license],
    ['Tọa độ', hit.latitude != null && hit.longitude != null ? `${hit.latitude.toFixed(5)}, ${hit.longitude.toFixed(5)}` : ''],
  ].filter(([, v]) => v);
  $('detailMeta').innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');

  const tags = [...(hit.object_labels || []), ...(hit.depicts || []), ...(hit.categories || []).slice(0, 6)];
  $('detailTags').innerHTML = tags.map((t) => `<span class="chip chip-static">${esc(t)}</span>`).join('');
  $('detailDesc').textContent = hit.description || '';
  $('detailDesc').hidden = !hit.description;

  const src = $('detailSource');
  src.href = hit.source_url || mediaUrl(hit.image_url);
  src.textContent = hit.source_url ? 'Xem nguồn gốc' : 'Mở ảnh gốc';
  $('detailSimilar').onclick = () => { dlg.close(); runSimilar(hit); };
  dlg.showModal();
}
$('detailClose').addEventListener('click', () => $('detail').close());
$('detail').addEventListener('click', (e) => { if (e.target === $('detail')) $('detail').close(); });

// ============================================================ NẠP ẢNH
const I = { items: [], captionMap: new Map(), onlyMissing: false, busy: false, lastBatch: null };
let uid = 0;

const baseName = (p) => String(p || '').split(/[\\/]/).pop().trim().toLowerCase();
const stemName = (p) => baseName(p).replace(/\.[^.]+$/, '');

function captionFor(file) {
  return I.captionMap.get(baseName(file.name)) || I.captionMap.get(stemName(file.name)) || '';
}

async function addFiles(files) {
  const txts = files.filter((f) => /\.txt$/i.test(f.name));
  for (const t of txts) {
    const text = (await t.text()).trim();
    if (text) I.captionMap.set(stemName(t.name), text);
  }
  const imgs = files.filter((f) => f.type.startsWith('image/'));
  const seen = new Set(I.items.map((it) => it.file.name + it.file.size));
  let added = 0;
  for (const f of imgs) {
    if (seen.has(f.name + f.size)) continue;
    I.items.push({ id: ++uid, file: f, url: URL.createObjectURL(f), caption: captionFor(f), status: null, reason: '' });
    added++;
  }
  // Áp caption từ .txt cho ảnh đã có mà chưa có caption
  if (txts.length) I.items.forEach((it) => { if (!it.caption) it.caption = captionFor(it.file); });
  if (imgs.length && !added) toast('Các ảnh này đã có trong danh sách.');
  $('ingestResult').hidden = true;
  renderReview();
}
setupDropzone($('ingestDrop'), $('ingestFiles'), addFiles);

// ---- đọc file caption
function parseCSV(text) {
  const rows = []; let row = []; let cell = ''; let q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (c === '"') q = false; else cell += c;
    } else if (c === '"') q = true;
    else if (c === ',') { row.push(cell); cell = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(cell); rows.push(row); row = []; cell = '';
    } else cell += c;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  return rows.filter((r) => r.some((x) => x.trim()));
}
const NAME_COLS = ['file', 'filename', 'file_name', 'filepath', 'path', 'image', 'image_path', 'image_file'];
const CAP_COLS = ['caption', 'text', 'description', 'mo_ta'];

function pairsFromRecords(records) {
  const out = [];
  for (const r of records) {
    if (!r || typeof r !== 'object') continue;
    const keys = Object.keys(r);
    const nk = keys.find((k) => NAME_COLS.includes(k.toLowerCase().trim()));
    const ck = keys.find((k) => CAP_COLS.includes(k.toLowerCase().trim()));
    if (nk && ck) out.push([r[nk], r[ck]]);
  }
  return out;
}

async function loadCaptionFile(file) {
  const text = (await file.text()).replace(/^﻿/, '');
  let pairs = [];
  try {
    if (/\.jsonl$/i.test(file.name)) {
      pairs = pairsFromRecords(text.split(/\r?\n/).filter(Boolean).map((l) => JSON.parse(l)));
    } else if (/\.json$/i.test(file.name)) {
      const data = JSON.parse(text);
      pairs = Array.isArray(data) ? pairsFromRecords(data) : Object.entries(data);
    } else {
      const rows = parseCSV(text);
      const head = rows.shift().map((h) => h.toLowerCase().trim());
      const ni = head.findIndex((h) => NAME_COLS.includes(h));
      const ci = head.findIndex((h) => CAP_COLS.includes(h));
      if (ni < 0 || ci < 0) throw new Error('Không thấy cột tên ảnh (file/filename/filepath) và cột caption.');
      pairs = rows.map((r) => [r[ni], r[ci]]);
    }
  } catch (e) {
    $('captionFileStatus').className = 'status-line warn';
    $('captionFileStatus').textContent = `Không đọc được ${file.name}: ${e.message}`;
    return;
  }
  let n = 0;
  for (const [name, cap] of pairs) {
    if (!name || !cap || !String(cap).trim()) continue;
    I.captionMap.set(baseName(name), String(cap).trim());
    I.captionMap.set(stemName(name), String(cap).trim());
    n++;
  }
  let applied = 0;
  I.items.forEach((it) => {
    const c = captionFor(it.file);
    if (c) { it.caption = c; applied++; }
  });
  $('captionFileStatus').className = `status-line ${n ? 'good' : 'warn'}`;
  $('captionFileStatus').textContent = n
    ? `Đọc ${fmtNum(n)} caption từ ${file.name}. Đã gắn cho ${applied}/${I.items.length} ảnh đang chọn; ảnh chọn sau cũng sẽ tự được gắn.`
    : `${file.name} không có dòng caption hợp lệ.`;
  renderReview();
}
$('captionFile').addEventListener('change', (e) => { if (e.target.files[0]) loadCaptionFile(e.target.files[0]); e.target.value = ''; });

// ---- danh sách kiểm tra
const ready = (it) => it.caption.trim().length >= MIN_CAPTION;

function itemState(it) {
  if (it.status === 'indexed') return ['is-done', 'Đã nạp vào kho'];
  if (it.status === 'skipped') return ['is-skipped', it.reason || 'Bị bỏ qua'];
  if (!ready(it)) return ['is-missing', it.caption.trim() ? `Caption quá ngắn (tối thiểu ${MIN_CAPTION} ký tự)` : 'Thiếu caption'];
  return ['is-ready', 'Sẵn sàng'];
}

function renderReview() {
  const list = $('reviewList');
  const has = I.items.length > 0;
  $('reviewEmpty').hidden = has;
  $('reviewHead').hidden = !has;
  list.innerHTML = '';
  I.items.forEach((it, idx) => {
    if (I.onlyMissing && ready(it)) return;
    const [cls, state] = itemState(it);
    const li = document.createElement('li');
    li.className = `review-item ${cls}`;
    li.dataset.id = it.id;
    const locked = it.status === 'indexed';
    li.innerHTML = `
      <img src="${it.url}" alt="">
      <div class="review-fields">
        <div class="review-name">${idx + 1}. ${esc(it.file.name)} <small>${(it.file.size / 1024).toFixed(0)} KB</small></div>
        <label class="sr-only" for="cap${it.id}">Caption cho ${esc(it.file.name)}</label>
        <textarea id="cap${it.id}" placeholder="Mô tả ảnh bằng tiếng Anh: công trình gì, vật liệu, mái, cột, cảnh quanh…" ${locked ? 'readonly' : ''}>${esc(it.caption)}</textarea>
        <div class="review-note"><span class="state">${esc(state)}</span><span class="len">${it.caption.trim().length} ký tự</span></div>
      </div>
      <button class="icon-btn" type="button" aria-label="Bỏ ${esc(it.file.name)}" ${locked ? 'hidden' : ''}>
        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/></svg>
      </button>`;
    const ta = li.querySelector('textarea');
    ta.addEventListener('input', () => {
      it.caption = ta.value;
      if (it.status === 'skipped') it.status = null;
      const [c2, s2] = itemState(it);
      li.className = `review-item ${c2}`;
      li.querySelector('.state').textContent = s2;
      li.querySelector('.len').textContent = `${it.caption.trim().length} ký tự`;
      updateSummary();
    });
    li.querySelector('.icon-btn').addEventListener('click', () => {
      URL.revokeObjectURL(it.url);
      I.items = I.items.filter((x) => x.id !== it.id);
      renderReview();
    });
    list.appendChild(li);
  });
  updateSummary();
}

function pending() { return I.items.filter((it) => it.status !== 'indexed'); }

function updateSummary() {
  const p = pending();
  const missing = p.filter((it) => !ready(it)).length;
  const s = $('ingestSummary');
  const btn = $('ingestSubmit');
  if (!p.length) {
    s.className = 'status-line';
    s.textContent = I.items.length ? 'Tất cả ảnh trong danh sách đã được nạp.' : 'Chưa có ảnh nào.';
    btn.disabled = true;
    btn.textContent = 'Nạp ảnh vào kho';
    return;
  }
  if (missing) {
    s.className = 'status-line warn';
    s.textContent = `${missing}/${p.length} ảnh còn thiếu caption. Điền caption hoặc bỏ ảnh đó ra để tiếp tục.`;
  } else {
    s.className = 'status-line good';
    s.textContent = `${p.length} ảnh đã có caption, sẵn sàng nạp.`;
  }
  btn.disabled = missing > 0 || I.busy;
  btn.textContent = I.busy ? 'Đang nạp…' : `Nạp ${p.length} ảnh vào kho`;
}

$('onlyMissing').addEventListener('click', (e) => {
  I.onlyMissing = !I.onlyMissing;
  e.currentTarget.setAttribute('aria-pressed', I.onlyMissing);
  e.currentTarget.textContent = I.onlyMissing ? 'Hiện tất cả ảnh' : 'Chỉ hiện ảnh thiếu caption';
  renderReview();
});
$('clearAll').addEventListener('click', () => {
  I.items.forEach((it) => URL.revokeObjectURL(it.url));
  I.items = [];
  $('ingestResult').hidden = true;
  renderReview();
});

$('ingestSubmit').addEventListener('click', async () => {
  const items = pending();
  if (!items.length || items.some((it) => !ready(it))) return;
  const fd = new FormData();
  items.forEach((it) => fd.append('files', it.file, it.file.name));
  fd.append('captions', JSON.stringify(items.map((it) => it.caption.trim())));
  fd.append('batch_name', $('batchName').value.trim());
  fd.append('categories', $('batchCats').value.trim());
  fd.append('description', $('batchDesc').value.trim());
  fd.append('detect_objects', $('detectObjects').checked ? 'true' : 'false');

  I.busy = true;
  updateSummary();
  try {
    const d = await api('/api/v1/ingest/upload', { method: 'POST', body: fd });
    d.files.forEach((f, i) => { items[i].status = f.status; items[i].reason = f.reason || ''; items[i].key = f.key; });
    I.lastBatch = d.batch_id;
    showIngestResult(d);
    loadStats();
  } catch (e) {
    showIngestResult({ success: false, message: e.message });
  } finally {
    I.busy = false;
    renderReview();
  }
});

function showIngestResult(d) {
  const b = $('ingestResult');
  b.hidden = false;
  b.className = `banner ${d.success ? 'good' : 'bad'}`;
  requestAnimationFrame(() => b.scrollIntoView({ behavior: 'smooth', block: 'start' }));
  b.innerHTML = `<p><strong>${esc(d.message)}</strong></p>` + (d.success ? `
    <div class="actions">
      <button class="btn btn-primary btn-sm" id="viewBatch">Xem ảnh vừa nạp</button>
      <button class="btn btn-quiet btn-sm" id="undoBatch">Hoàn tác lần nạp này</button>
    </div>` : '');
  if (!d.success) return;
  $('viewBatch').addEventListener('click', () => {
    showView('search');
    runBrowseBatch(d.batch_id, d.batch_name);
  });
  $('undoBatch').addEventListener('click', async () => {
    try {
      const r = await api(`/api/v1/ingest/batches/${encodeURIComponent(d.batch_id)}`, { method: 'DELETE' });
      I.items.forEach((it) => { if (it.status === 'indexed') { it.status = null; it.key = null; } });
      b.className = 'banner bad';
      b.innerHTML = `<p>Đã hoàn tác: xóa ${r.deleted} ảnh khỏi kho. Danh sách vẫn giữ để bạn sửa và nạp lại.</p>`;
      renderReview();
      loadStats();
    } catch (e) { toast(e.message, true); }
  });
}

async function runBrowseBatch(batchId, name) {
  S.last = { type: 'browse' };
  showSkeleton();
  try {
    const d = await api(`/api/v1/images/list?batch_id=${encodeURIComponent(batchId)}`);
    d.batch_name = name;
    renderResults(d);
  } catch (e) { renderError(e); }
}

// ============================================================ khởi động
setMode('text');
loadStats();
