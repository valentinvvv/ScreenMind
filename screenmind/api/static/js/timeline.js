//  TIMELINE VIEW
// ══════════════════════════════════════════════════════════
const TL_PAGE_SIZE = 50;
let _tlPage = 0;
let _tlStatusES = null;
let _tlStatusHideTimer = null;

async function renderTimeline(el) {
  el.innerHTML = `
    <div id="tl-status-panel" class="tl-status-panel" style="display:none">
      <img class="tl-status-thumb" id="tl-status-thumb" alt="">
      <div class="tl-status-body">
        <div class="tl-status-top">
          <span class="tl-status-stage" id="tl-status-stage"></span>
          <span class="day-num" id="tl-status-daynum"></span>
          <span class="tl-status-app" id="tl-status-app"></span>
          <span class="tl-status-queue" id="tl-status-queue"></span>
        </div>
        <div class="tl-status-response" id="tl-status-response"></div>
      </div>
    </div>
    <div class="date-nav" style="margin-bottom:20px">
      <button class="btn btn-ghost btn-sm" id="prev-day">\u25c0</button>
      <input type="date" id="timeline-date" value="${currentDate}">
      <button class="btn btn-ghost btn-sm" id="next-day">\u25b6</button>
      <span style="margin-left:12px;color:var(--text-muted);font-size:0.85rem" id="tl-count"></span>
      <button class="btn btn-ghost btn-sm" id="clear-timeline" style="margin-left:auto;color:#ef4444;font-size:0.8rem" title="Clear today's timeline">🗑 Clear Timeline</button>
    </div>
    <div class="timeline" id="timeline-list"><div class="spinner"></div></div>
    <div class="tl-pagination" id="tl-pagination" style="display:none">
      <button class="btn btn-ghost btn-sm" id="tl-prev-page">\u25c0 Prev</button>
      <span class="tl-page-info">Page</span>
      <input type="number" id="tl-page-input" class="tl-page-input" min="1" value="1">
      <span class="tl-page-info">/ <span id="tl-total-pages">1</span></span>
      <button class="btn btn-ghost btn-sm" id="tl-next-page">Next \u25b6</button>
    </div>`;
  $('#timeline-date').addEventListener('change', e => { currentDate = e.target.value; _tlPage = 0; loadTimeline(); });
  $('#prev-day').addEventListener('click', () => shiftDate(-1));
  $('#tl-prev-page').addEventListener('click', () => { if (_tlPage > 0) { _tlPage--; loadTimeline(); } });
  $('#tl-next-page').addEventListener('click', () => { _tlPage++; loadTimeline(); });
  $('#tl-page-input').addEventListener('change', e => {
    const val = parseInt(e.target.value) - 1;
    const totalPages = parseInt($('#tl-total-pages').textContent);
    if (val >= 0 && val < totalPages) { _tlPage = val; loadTimeline(); }
    else { e.target.value = _tlPage + 1; }
  });
  loadTimeline();
  _openStatusStream();
  // Inject Model Hub pill into header-actions (guard against duplicates)
  _injectTimelinePill();
  // Auto-refresh timeline every 30s
  clearInterval(window._tlRefresh);
  window._tlRefresh = setInterval(() => { if (currentView === 'timeline') loadTimeline(true); }, 30000);
}

// Called by core.js navigate() when leaving the timeline view
function onTimelineLeave() {
  _closeStatusStream();
}

function shiftDate(days) {
  const d = new Date(currentDate); d.setDate(d.getDate() + days);
  currentDate = d.toISOString().split('T')[0];
  $('#timeline-date').value = currentDate;
  _tlPage = 0;
  loadTimeline();
}

async function loadTimeline(silent = false) {
  const list = $('#timeline-list');
  if (!list) return;  // view destroyed mid-flight
  if (!silent) list.innerHTML = '<div class="spinner"></div>';
  try {
    const offset = _tlPage * TL_PAGE_SIZE;
    const data = await api(`/api/timeline?date=${currentDate}&limit=${TL_PAGE_SIZE}&offset=${offset}`);
    if (!document.getElementById('timeline-list')) return;  // navigated away
    const acts = data.activities || [];
    const total = data.total || acts.length;
    const totalPages = Math.max(1, Math.ceil(total / TL_PAGE_SIZE));
    if (_tlPage >= totalPages) { _tlPage = totalPages - 1; return loadTimeline(silent); }

    // Also fetch meetings for this date
    let meetings = [];
    try {
      const mData = await api(`/api/meetings?date=${currentDate}`);
      meetings = mData.meetings || [];
    } catch { /* meetings endpoint not available */ }
    const pager = $('#tl-pagination');
    if (total > TL_PAGE_SIZE) {
      pager.style.display = '';
      $('#tl-page-input').value = _tlPage + 1;
      $('#tl-total-pages').textContent = totalPages;
      $('#tl-prev-page').disabled = _tlPage === 0;
      $('#tl-next-page').disabled = _tlPage >= totalPages - 1;
    } else {
      pager.style.display = 'none';
    }

    if (!acts.length && !meetings.length) {
      list.innerHTML = `<div class="onboarding">
        <h2>🚀 ScreenMind is Running!</h2>
        <div class="subtitle">Your privacy-first AI memory is warming up. First capture coming soon...</div>
        <div class="feature-grid">
          <div class="feature-card"><div class="fc-icon">📋</div><div class="fc-title">Timeline</div><div class="fc-desc">Chronological activity feed with AI analysis</div></div>
          <div class="feature-card"><div class="fc-icon">🔍</div><div class="fc-title">Semantic Search</div><div class="fc-desc">Natural language queries powered by Gemma 4</div></div>
          <div class="feature-card"><div class="fc-icon">📊</div><div class="fc-title">Analytics</div><div class="fc-desc">Category breakdown, top apps, hours tracked</div></div>
          <div class="feature-card"><div class="fc-icon">⏪</div><div class="fc-title">Day Rewind</div><div class="fc-desc">Timelapse playback of your screen activity</div></div>
          <div class="feature-card"><div class="fc-icon">📝</div><div class="fc-title">Daily Summary</div><div class="fc-desc">AI-generated summaries and standup notes</div></div>
          <div class="feature-card"><div class="fc-icon">🛡️</div><div class="fc-title">100% Local</div><div class="fc-desc">No cloud, no telemetry — your data stays yours</div></div>
        </div>
        <div class="shortcut-row" style="margin-bottom:8px">
          <span class="shortcut-key">Ctrl+Shift+B</span> Bookmark from any app
          &nbsp;&nbsp;
          <span class="shortcut-key">Ctrl+Shift+P</span> Pause/Resume anywhere
        </div>
        <div class="waiting-pulse"><span class="pulse-dot"></span> Waiting for first capture...</div>
      </div>`;
      return;
    }
    // Render meeting cards first (page 1 only), then activity cards
    const meetingHtml = _tlPage === 0 ? meetings.map(m => meetingCard(m)).join('') : '';
    const activityHtml = acts.map((a, i) => timelineCard(a, i)).join('');
    list.innerHTML = meetingHtml + activityHtml;
  } catch (err) {
    list.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠️</div><div class="empty-title">Error</div><div>${err.message}</div></div>`;
  }
}

// ── Live status panel (SSE) ───────────────────────────────
const _TL_STAGE_LABELS = {
  processing: '⚙️ Processing…',
  ocr: '🔤 Extracting text…',
  analyzing: '🤖 Analyzing…',
  done: '✅ Done',
  failed: '⚠️ Failed',
  yielded: '⏸ Yielded to chat',
  requeued: '↻ Re-queued',
};

function _openStatusStream() {
  _closeStatusStream();
  try {
    _tlStatusES = new EventSource('/api/analysis/stream');
    _tlStatusES.onmessage = function(e) {
      let ev;
      try { ev = JSON.parse(e.data); } catch { return; }
      if (ev.type === 'status') _renderStatusSnapshot(ev);
      else if (ev.type === 'delta') _appendStatusDelta(ev.text || '');
    };
    _tlStatusES.onerror = function() {
      // Server gone / restart — EventSource retries automatically;
      // hide the panel so stale state doesn't linger.
      const panel = document.getElementById('tl-status-panel');
      if (panel) panel.style.display = 'none';
    };
  } catch { /* EventSource unavailable — panel simply stays hidden */ }
}

function _closeStatusStream() {
  if (_tlStatusES) { _tlStatusES.close(); _tlStatusES = null; }
  if (_tlStatusHideTimer) { clearTimeout(_tlStatusHideTimer); _tlStatusHideTimer = null; }
}

function _renderStatusSnapshot(s) {
  const panel = document.getElementById('tl-status-panel');
  if (!panel) return;
  if (_tlStatusHideTimer) { clearTimeout(_tlStatusHideTimer); _tlStatusHideTimer = null; }

  panel.style.display = '';
  $('#tl-status-stage').textContent = _TL_STAGE_LABELS[s.stage] || s.stage;
  $('#tl-status-daynum').textContent = s.day_number ? `${s.date}_${s.day_number}` : '';
  $('#tl-status-app').textContent = s.app_name || '';
  $('#tl-status-queue').textContent = s.queue_size > 0 ? `${s.queue_size} queued` : '';
  const thumb = $('#tl-status-thumb');
  const src = s.activity_id ? `/api/screenshot/${s.activity_id}` : '';
  if (thumb) { thumb.src = src; thumb.style.display = src ? '' : 'none'; }

  const resp = $('#tl-status-response');
  if (resp) {
    if (s.stage === 'done' && s.summary) {
      resp.textContent = s.summary;
    } else if (s.stage === 'failed' && s.summary) {
      resp.textContent = s.summary;
    } else {
      resp.textContent = s.response || '';
    }
    resp.scrollTop = resp.scrollHeight;
  }

  // Terminal states: keep visible briefly, then collapse
  if (s.stage === 'done' || s.stage === 'failed' || s.stage === 'requeued') {
    _tlStatusHideTimer = setTimeout(() => {
      const p = document.getElementById('tl-status-panel');
      if (p) p.style.display = 'none';
    }, 8000);
  }
}

function _appendStatusDelta(text) {
  const resp = document.getElementById('tl-status-response');
  if (!resp) return;
  resp.textContent += text;
  resp.scrollTop = resp.scrollHeight;
}

function meetingCard(m) {
  const startTime = formatTime(m.start_time);
  const endTime = m.end_time ? formatTime(m.end_time) : 'ongoing';
  const duration = m.duration_minutes ? `${Math.round(m.duration_minutes)} min` : '';
  const summaryText = (m.summary || 'No summary').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const transcriptText = (m.transcript || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  // First meaningful line of summary for preview
  const previewLine = summaryText.split('\n').find(l => l.trim() && !l.startsWith('⏳')) || 'Meeting recorded';
  const tid = `tl-mtg-transcript-${m.id}`;
  const sid = `tl-mtg-summary-${m.id}`;
  return `
    <div class="timeline-item tl-meeting-card" onclick="toggleTlMeeting(${m.id})">
      <div class="thumb" style="display:flex;align-items:center;justify-content:center;font-size:2rem;background:rgba(236,72,153,0.1);cursor:pointer">🎙️</div>
      <div class="info" style="flex:1;min-width:0">
        <div class="top">
          <span class="time">${startTime}</span>
          <span class="app-name">Meeting — ${(m.app_name || 'Unknown').replace(/</g, '&lt;')}</span>
          <span class="badge badge-other" style="background:rgba(236,72,153,0.15);color:#ec4899">meeting</span>
        </div>
        <div class="summary">${startTime} – ${endTime}${duration ? ' · ' + duration : ''} · ${previewLine.substring(0, 80)}${previewLine.length > 80 ? '...' : ''}</div>
        <div class="tl-meeting-detail" id="${sid}" style="display:none;margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.06)">
          <div style="white-space:pre-wrap;font-size:0.82rem;line-height:1.6;color:var(--text-secondary)">${summaryText}</div>
          ${transcriptText && transcriptText !== '(No speech detected)' ? `
            <button class="meeting-transcript-toggle" style="margin-top:8px" onclick="event.stopPropagation(); var t=document.getElementById('${tid}'); t.classList.toggle('open'); this.textContent = t.classList.contains('open') ? '▲ Hide transcript' : '▼ Show full transcript'">▼ Show full transcript</button>
            <div class="meeting-transcript" id="${tid}">${transcriptText}</div>
          ` : ''}
        </div>
      </div>
      <div style="color:var(--text-muted);font-size:0.78rem;white-space:nowrap;margin-left:8px">${duration}</div>
    </div>`;
}

window.toggleTlMeeting = function(id) {
  const el = document.getElementById('tl-mtg-summary-' + id);
  if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
};

function timelineCard(a, i) {
  const time = formatTime(a.timestamp);
  const cat = a.category || 'other';
  const bookmarkLabel = a.bookmarked ? '★ Bookmarked' : '☆ Bookmark';
  const bookmarkClass = a.bookmarked ? 'active' : '';
  const devCtx = a.repo_name ? `<div class="dev-ctx">🔀 ${a.repo_name}/${a.branch || 'main'} ${a.insertions ? `<span style="color:#10b981">+${a.insertions}</span>` : ''}${a.deletions ? ` <span style="color:#ef4444">-${a.deletions}</span>` : ''}</div>` : '';
  const thumb = a.screenshot_url ? `<img class="thumb" src="${a.screenshot_url}" loading="lazy" onclick="openModal('${a.screenshot_url}', ${a.id})" alt="">` : '<div class="thumb"></div>';

  // Day-scoped event number: YYYY-MM-DD_N (chronological within the day)
  const dayNum = a.day_number ? `<span class="day-num" title="Event number for the day">${(a.timestamp || '').substring(0, 10)}_${a.day_number}</span>` : '';

  // Analysis method badge — color-coded
  const method = a.analysis_method || '';
  const methodColors = {'full': '#a78bfa', 'cache:identical': '#34d399', 'cache:minor': '#fbbf24', 'skipped': '#6b7280', 'backfill:full': '#22d3ee', 'backfill:cache:identical': '#22d3ee', 'backfill:cache:minor': '#22d3ee', 'reanalyze': '#f97316'};
  const methodColor = methodColors[method] || '';
  const methodBadge = method && methodColor ? ` <span style="background:${methodColor}18;color:${methodColor};border:1px solid ${methodColor}33;font-size:0.65rem;padding:1px 6px;border-radius:8px;font-weight:500">${method}</span>` : '';

  return `
    <div class="timeline-item" style="animation-delay:${i * 0.06}s">
      ${thumb}
      <div class="info">
        <div class="top">
          <span class="time">${time}</span>
          ${dayNum}
          <span class="app-name">${a.app_name || 'Unknown'}</span>
          <span class="badge badge-${cat}">${cat}</span>${methodBadge}
        </div>
        <div class="summary">${a.summary || 'No analysis'}</div>
        ${a.active_url ? `<div style="margin-top:2px"><a href="${a.active_url}" target="_blank" rel="noopener" onclick="event.stopPropagation()" style="font-size:0.7rem;color:#60a5fa;text-decoration:none;opacity:0.8;word-break:break-all" title="${a.active_url}">🔗 ${(() => { try { return new URL(a.active_url).hostname } catch(e) { return a.active_url.substring(0, 40) } })()}</a></div>` : ''}
        ${devCtx}
      </div>
      <div class="card-menu-wrap">
        <button class="card-menu-trigger" onclick="event.stopPropagation(); toggleCardMenu(this)" title="Actions">⋮</button>
        <div class="card-menu">
          <button class="menu-item bookmark-item ${bookmarkClass}" onclick="event.stopPropagation(); toggleBookmark(${a.id}, this)">
            <span class="menu-icon">${a.bookmarked ? '★' : '☆'}</span> ${bookmarkLabel}
          </button>
          <button class="menu-item reanalyze-item" onclick="event.stopPropagation(); reanalyzeActivity(${a.id}, this)">
            <span class="menu-icon">↻</span> Re-analyze
          </button>
          <div class="menu-divider"></div>
          <button class="menu-item delete-item" onclick="event.stopPropagation(); deleteActivity(${a.id}, this)">
            <span class="menu-icon">✕</span> Delete
          </button>
        </div>
      </div>
    </div>`;
}

// ── Three-dot menu toggle ─────────────────────────────────
window.toggleCardMenu = function(trigger) {
  const menu = trigger.nextElementSibling;
  const isOpen = menu.classList.contains('open');
  // Close all other open menus first
  document.querySelectorAll('.card-menu.open').forEach(function(m) { m.classList.remove('open'); });
  document.querySelectorAll('.card-menu-trigger.active').forEach(function(t) { t.classList.remove('active'); });
  if (!isOpen) {
    menu.classList.add('open');
    trigger.classList.add('active');
  }
};

// Close menus when clicking outside
document.addEventListener('click', function() {
  document.querySelectorAll('.card-menu.open').forEach(function(m) { m.classList.remove('open'); });
  document.querySelectorAll('.card-menu-trigger.active').forEach(function(t) { t.classList.remove('active'); });
});

window.toggleBookmark = async function(id, el) {
  try {
    const r = await fetch(`/api/activities/${id}/bookmark`, { method: 'PUT' });
    const data = await r.json();
    el.closest('.card-menu').classList.remove('open');
    el.closest('.card-menu-wrap').querySelector('.card-menu-trigger').classList.remove('active');
    showToast(data.bookmarked ? '⭐ Bookmarked!' : 'Bookmark removed', data.bookmarked ? 'success' : 'info');
    loadTimeline(true);
  } catch {}
};

window.reanalyzeActivity = async function(id, el) {
  el.closest('.card-menu').classList.remove('open');
  el.closest('.card-menu-wrap').querySelector('.card-menu-trigger').classList.remove('active');
  showToast('↻ Re-analyzing screenshot...', 'info');
  try {
    const r = await fetch(`/api/activities/${id}/reanalyze`, { method: 'POST' });
    if (!r.ok) throw new Error('Failed');
    showToast('✓ Re-analysis complete!', 'success');
    loadTimeline(true);
  } catch (err) {
    showToast('Re-analysis failed: ' + err.message, 'warning');
  }
};

window.deleteActivity = async function(id, btn) {
  if (!btn.classList.contains('confirm-delete')) {
    btn.innerHTML = '<span class="menu-icon">⚠</span> Confirm Delete';
    btn.classList.add('confirm-delete');
    setTimeout(function() {
      btn.innerHTML = '<span class="menu-icon">✕</span> Delete';
      btn.classList.remove('confirm-delete');
    }, 3000);
    return;
  }
  const card = btn.closest('.timeline-item');
  card.style.transform = 'translateX(100px)';
  card.style.opacity = '0';
  card.style.transition = 'all 0.3s ease';
  try {
    const r = await fetch(`/api/activities/${id}`, { method: 'DELETE' });
    if (!r.ok) throw new Error('Failed');
    setTimeout(function() { card.remove(); }, 300);
    showToast('Activity deleted', 'success');
    loadTimeline(true);
  } catch (err) {
    card.style.transform = ''; card.style.opacity = '';
    showToast('Delete failed', 'warning');
  }
};

// ── Clear Timeline (with confirmation) ────────────────────
function confirmClearTimeline() {
  const container = $('#toast-container');
  // Remove any existing confirm toast
  const existing = document.querySelector('.confirm-toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = 'toast confirm-toast warning';
  toast.style.cssText = 'display:flex;flex-direction:column;gap:10px;max-width:320px;';
  toast.innerHTML = `
    <div style="font-weight:600">Clear timeline for ${currentDate}?</div>
    <div style="font-size:0.85rem;opacity:0.8">This will permanently delete all screenshots and activity data for this day.</div>
    <div style="display:flex;gap:8px;justify-content:flex-end">
      <button class="btn btn-ghost btn-sm" onclick="this.closest('.confirm-toast').remove()" style="color:var(--text-muted)">Cancel</button>
      <button class="btn btn-sm" onclick="executeClearTimeline()" style="background:#ef4444;color:#fff;border:none;padding:6px 16px;border-radius:6px;cursor:pointer">Yes, Delete</button>
    </div>`;
  container.appendChild(toast);
}

window.executeClearTimeline = async function() {
  const toast = document.querySelector('.confirm-toast');
  if (toast) toast.remove();
  try {
    const r = await fetch(`/api/timeline/clear?date=${currentDate}`, { method: 'DELETE' });
    const data = await r.json();
    showToast(`Cleared ${data.deleted} activities for ${data.date}`, 'success');
    _tlPage = 0;
    loadTimeline();
  } catch (err) {
    showToast('Failed to clear timeline', 'warning');
  }
};

// ══════════════════════════════════════════════════════════
