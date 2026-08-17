"""
name: Timesheet
schedule: daily
description: Builds copy-paste timesheet entries from helpdesk ticket screen activity
enabled: true
output: local
"""
# Timesheet agent — deterministic extraction, no LLM.
#
# Reconstructs time-per-ticket from screen captures: a ticket is "worked on"
# while its detail page is visible (ticket id in the helpdesk URL, or a
# "Ticket 98447"-style reference in window title / OCR). Time per ticket is
# the span of its captures with gaps capped at 2x the capture interval
# (same method ScreenMind's daily summary uses). Nothing is invented:
# tickets, customers and subjects come only from captured screen data;
# unknown fields are "—".
#
# Data access: reads the local SQLite DB directly (read-only). This bypasses
# the HTTP SDK, which returns 401 whenever a dashboard PIN is set.

import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

# ── Config ────────────────────────────────────────────────────────────
HELPDESK_DOMAINS = ["helpdesk.gits.lu"]   # domains that count as the helpdesk
OCR_CHARS = 1500        # OCR chars read per capture
ATTRIBUTE_LIST_PAGES = False  # True: captures showing several ticket ids count for all of them
INCLUDE_UNTICKETED = True     # also emit "—" entries for non-helpdesk activity
CUSTOMER_OVERRIDES = {
    # "98447": "OMEA ADVISORS",   # manual ticket → customer fixes
}

# Ticket ids: 5-6 digits in a helpdesk URL, or 4-6 digits after an explicit
# "ticket"/"#" marker in screen text. Bare numbers are never trusted.
_URL_ID_RE = re.compile(r"\b(\d{5,6})\b")
_TEXT_ID_RE = re.compile(r"(?:ticket|#|n°|no\.?)\s*[:.\-]?\s*(\d{4,6})\b", re.I)
_CUSTOMER_RE = re.compile(
    r"(?:customer|client|company|organisation|organization|soci[ée]t[ée])\s*[:\-–]\s*([^\n|;]{2,40})",
    re.I,
)


def _capture_interval(data_dir: Path) -> int:
    """Live capture interval from settings.json; falls back to the 40s default."""
    try:
        cfg = json.loads((data_dir / "settings.json").read_text(encoding="utf-8"))
        return max(10, int(cfg.get("capture_interval", 40)))
    except Exception:
        return 40


def _fetch_day(db_path: Path, date_str: str) -> list:
    """All analyzed activities for the day, with truncated OCR, read-only."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""SELECT timestamp, app_name, category, summary, details,
                       active_url, window_title,
                       SUBSTR(ocr_text, 1, {OCR_CHARS}) AS ocr_text
                FROM activities
                WHERE analyzed = 1 AND DATE(timestamp) = ?
                ORDER BY timestamp ASC""",
            (date_str,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _is_helpdesk(url: str, text: str) -> bool:
    hay_url = (url or "").lower()
    hay_txt = text.lower()
    return any(d in hay_url or d in hay_txt for d in HELPDESK_DOMAINS)


def _ticket_ids(act: dict) -> set:
    """Ticket ids a capture attributes work to. Empty = not ticket work."""
    url = act.get("active_url") or ""
    text = " ".join(
        str(x) for x in (
            act.get("window_title"), act.get("summary"),
            act.get("details"), act.get("ocr_text"),
        ) if x
    )
    if not _is_helpdesk(url, text):
        return set()

    # URL ids (ticket detail view) are the strongest signal.
    url_ids = set()
    if url:
        try:
            p = urlparse(url)
            url_ids = set(_URL_ID_RE.findall(f"{p.path}?{p.query}#{p.fragment}"))
        except Exception:
            url_ids = set()
    if len(url_ids) == 1:
        return url_ids

    text_ids = set(_TEXT_ID_RE.findall(text))
    if url_ids:                      # several ids in URL — ambiguous
        return url_ids if ATTRIBUTE_LIST_PAGES else set()
    if len(text_ids) == 1:
        return text_ids
    if len(text_ids) > 1:            # list page with several tickets
        return text_ids if ATTRIBUTE_LIST_PAGES else set()
    return set()


def _customer_for(captures: list, ticket: str) -> str:
    if ticket in CUSTOMER_OVERRIDES:
        return CUSTOMER_OVERRIDES[ticket]
    names = Counter()
    for act in captures:
        ocr = act.get("ocr_text") or ""
        for m in _CUSTOMER_RE.finditer(ocr):
            name = m.group(1).strip(" \t.:;|-–")
            if len(name) >= 2:
                names[name] += 1
    return names.most_common(1)[0][0] if names else "—"


def _subject_for(captures: list, max_words: int = 8) -> str:
    summaries = Counter(
        (a.get("summary") or "").strip() for a in captures if (a.get("summary") or "").strip()
    )
    if not summaries:
        return "—"
    words = summaries.most_common(1)[0][0].split()[:max_words]
    return " ".join(words)


def _span_seconds(timestamps: list, interval: int) -> float:
    """First capture counts one interval; each gap counts up to 2x interval."""
    ts = sorted(timestamps)
    total = float(interval)
    for prev, cur in zip(ts, ts[1:]):
        gap = (cur - prev).total_seconds()
        if gap > 0:
            total += min(gap, 2 * interval)
    return total


def _fmt(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60}:{minutes % 60:02d} h"


def _round5(seconds: float) -> int:
    return max(5, int(round(seconds / 60.0 / 5.0)) * 5)


def _parse_ts(value) -> datetime:
    return datetime.fromisoformat(str(value))


def _resolve_date(data_dir: Path, agent_name: str, ui_date: str = "") -> tuple:
    """Pick the target date. Priority: dashboard date picker (context["date"])
    > TIMESHEET_DATE env var > state file > today.

    The state file ("date" key) is consumed once and cleared, so scheduled
    runs fall back to today. Accepts YYYY-MM-DD or "yesterday".
    Returns (date_str, error_or_None).
    """
    raw = (ui_date or "").strip()
    source = "dashboard date picker"

    if not raw:
        raw = os.environ.get("TIMESHEET_DATE", "").strip()
        source = "TIMESHEET_DATE env var"

    if not raw:
        state_path = data_dir / "agents" / "state" / f"{agent_name}.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        raw = str(state.get("date", "")).strip()
        source = "state file"
        if raw:  # consume once: clear the key so the next run uses today
            state.pop("date", None)
            try:
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            except OSError:
                pass

    if not raw:
        return datetime.now().strftime("%Y-%m-%d"), None

    if raw.lower() == "yesterday":
        return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"), None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").strftime("%Y-%m-%d"), None
    except ValueError:
        return "", (
            f"Invalid date '{raw}' from {source}. "
            "Use YYYY-MM-DD or 'yesterday'."
        )


def run(context) -> str:
    data_dir = Path(context.get("data_dir", "~/.screenmind")).expanduser()
    db_path = data_dir / "screenmind.db"
    if not db_path.exists():
        return f"ScreenMind database not found at {db_path}."

    date_str, err = _resolve_date(
        data_dir, context.get("agent_name", "Timesheet"),
        context.get("date", ""),
    )
    if err:
        return err
    interval = _capture_interval(data_dir)
    activities = _fetch_day(db_path, date_str)
    if not activities:
        return f"No analyzed screen activity found for {date_str}."

    ticket_caps = defaultdict(list)   # ticket id -> [activity]
    ticket_times = defaultdict(list)  # ticket id -> [datetime]
    other_times = defaultdict(list)   # (app, category) -> [datetime]
    other_caps = defaultdict(list)

    for act in activities:
        try:
            ts = _parse_ts(act.get("timestamp"))
        except (TypeError, ValueError):
            continue
        ids = _ticket_ids(act)
        if ids:
            for tid in ids:
                ticket_caps[tid].append(act)
                ticket_times[tid].append(ts)
        elif INCLUDE_UNTICKETED:
            key = (act.get("app_name") or "?", act.get("category") or "?")
            other_times[key].append(ts)
            other_caps[key].append(act)

    if not ticket_caps and not other_times:
        return f"No helpdesk ticket activity found for {date_str}."

    # Build entries: (customer, ticket, subject, minutes)
    entries = []
    for tid, caps in ticket_caps.items():
        minutes = _round5(_span_seconds(ticket_times[tid], interval))
        entries.append((_customer_for(caps, tid), f"Ticket {tid}", _subject_for(caps), minutes))
    for key, ts_list in other_times.items():
        minutes = _round5(_span_seconds(ts_list, interval))
        entries.append(("—", "—", _subject_for(other_caps[key]), minutes))

    # Group by customer; ticketed customers by total time desc, "—" last.
    groups = defaultdict(list)
    for customer, ticket, subject, minutes in entries:
        groups[customer].append((ticket, subject, minutes))

    ordered = sorted(
        (c for c in groups if c != "—"),
        key=lambda c: sum(m for _, _, m in groups[c]),
        reverse=True,
    )
    if "—" in groups:
        ordered.append("—")

    lines = []
    for customer in ordered:
        rows = sorted(groups[customer], key=lambda r: r[2], reverse=True)
        lines.append(customer)
        for ticket, subject, minutes in rows:
            lines.append(f"{ticket} | {customer} | {subject} | {_fmt(minutes)}")
        lines.append(f"Subtotal: {_fmt(sum(m for _, _, m in rows))}")
        lines.append("")
    return "\n".join(lines).rstrip()
