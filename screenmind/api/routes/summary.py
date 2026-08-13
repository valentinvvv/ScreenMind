"""Summary & Standup routes — AI-generated daily summaries."""

import logging
import asyncio
from collections import Counter
from datetime import datetime as dt

from fastapi import APIRouter, Query

from screenmind.config import settings
from screenmind.api.dependencies import db
from screenmind.engine.llm_client import CHARS_PER_TOKEN, text_model_window

logger = logging.getLogger("screenmind.api.routes.summary")

router = APIRouter(prefix="/api", tags=["summary"])


# Categories that count as focused/productive work time.
# Browsing and communication are excluded — they can be doomscrolling/chat.
_PRODUCTIVE_CATEGORIES = {"coding", "writing", "terminal", "design", "meeting"}


def _compute_day_metrics(activities: list) -> dict:
    """Compute productive_hours, category_breakdown, and top_repos from activities.

    Uses actual timestamps with per-gap capping for accuracy instead of
    count × capture_interval which overcounts rapid-fire frames and
    undercounts long static work sessions.
    """
    analyzed = [a for a in activities if a.get("analyzed")]
    if not analyzed:
        return {"productive_hours": 0.0, "category_breakdown": {}, "top_repos": []}

    # Category breakdown
    category_counts = Counter(
        (a.get("category") or "other") for a in analyzed
    )

    # Productive hours from timestamps — sort chronologically, sum capped deltas
    max_gap = 2 * settings.capture_interval  # Cap per gap (e.g. 80s at default 40s)
    productive_seconds = 0.0
    productive_entries = sorted(
        [a for a in analyzed if (a.get("category") or "other").lower() in _PRODUCTIVE_CATEGORIES],
        key=lambda a: a.get("timestamp", ""),
    )
    for i, a in enumerate(productive_entries):
        if i == 0:
            # First entry: count one interval
            productive_seconds += settings.capture_interval
            continue
        try:
            prev_ts = dt.fromisoformat(productive_entries[i - 1]["timestamp"])
            curr_ts = dt.fromisoformat(a["timestamp"])
            delta = (curr_ts - prev_ts).total_seconds()
            productive_seconds += min(delta, max_gap)
        except (ValueError, KeyError, TypeError):
            productive_seconds += settings.capture_interval

    productive_hours = round(productive_seconds / 3600, 2)

    # Top repos from dev_context JOIN
    repo_counts = Counter(
        a["repo_name"] for a in analyzed
        if a.get("repo_name")
    )
    top_repos = [repo for repo, _ in repo_counts.most_common(5)]

    return {
        "productive_hours": productive_hours,
        "category_breakdown": dict(category_counts),
        "top_repos": top_repos,
    }


# Reserved for the prompt template + model output when budgeting activity text
# against the context window.
_PROMPT_OVERHEAD_TOKENS = 500


def _budget_chars(window: int, max_tokens: int) -> int:
    """Max chars of activity text for a window, at the worst-case token density."""
    return max(int((window - max_tokens - _PROMPT_OVERHEAD_TOKENS) * CHARS_PER_TOKEN), 0)


def _pick_window(full_block_chars: int, out_tokens: int) -> int:
    """Context window the summary prompt is budgeted against.

    With a configured text model: 'always' routing budgets against the text
    model's window outright; 'overflow' routing does so when the full day
    would exceed the primary Context Window. llm_client.chat() makes the same
    routing decision, so a prompt sized for the text window lands on the text
    model instead of being rejected with HTTP 400.
    """
    tw = text_model_window()
    if tw is None or settings.text_llm_routing == "off":
        return settings.context_window
    if settings.text_llm_routing == "always":
        return tw
    est_tokens = int(full_block_chars / CHARS_PER_TOKEN) + out_tokens + _PROMPT_OVERHEAD_TOKENS
    return tw if est_tokens > settings.context_window else settings.context_window


def _build_activity_block(
    activities: list, max_rich: int, rich_chars: int, budget_chars: int
) -> tuple:
    """Render analyzed activities as prompt lines, capped at budget_chars.

    The backend rejects oversized prompts with HTTP 400 (llama.cpp:
    "request (N tokens) exceeds the available context size"), so the block is
    capped and the OLDEST entries are dropped first — recent activity matters
    most in a day summary.

    Returns (block_text, entry_count).
    """
    # Build newest-first so the limited rich-content slots go to the most
    # recent activity (original behavior), then reverse to chronological order.
    entries = []
    rich_count = 0
    for a in sorted(activities, key=lambda x: x.get("timestamp", ""), reverse=True):
        if not a.get("analyzed"):
            continue
        entry = f"[{a.get('timestamp', '')}] {a.get('app_name', '?')} ({a.get('category', '?')}): {a.get('summary', '')}"
        if rich_count < max_rich:
            org_text = (a.get("organized_text") or "").strip()
            if org_text:
                if len(org_text) > rich_chars:
                    org_text = org_text[:rich_chars] + "..."
                entry += f"\n  Screen content: {org_text}"
                rich_count += 1
        entries.append(entry)
    entries.reverse()

    total = sum(len(e) for e in entries) + max(len(entries) - 1, 0)
    while entries and total > budget_chars:
        total -= len(entries.pop(0)) + 1

    return "\n".join(entries), len(entries)


@router.get("/summary")
async def get_summary(
    date: str = Query(default=None),
):
    target = date or str(__import__("datetime").date.today())
    summary = db.get_daily_summary(target)
    return {"date": target, "generated": summary is not None, "summary": summary, "standup": (summary or {}).get("standup", "")}


@router.post("/summary/generate")
async def generate_summary(
    date: str = Query(default=None),
):
    """Generate a daily summary using the configured LLM backend."""
    from screenmind.engine import llm_client
    from screenmind.storage.models import DailySummary

    target = date or str(__import__("datetime").date.today())
    activities = db.get_activities_by_date(target, limit=200)

    if not activities:
        return {"date": target, "summary": {"summary": "No activities recorded on this date."}}

    # Full render decides whether the day overflows to the text model
    full_block, _ = _build_activity_block(activities, 20, 300, budget_chars=10**9)
    window = _pick_window(len(full_block), 2048)
    out_tokens = min(2048, window // 2)
    acts_text, act_count = _build_activity_block(
        activities, max_rich=20, rich_chars=300,
        budget_chars=_budget_chars(window, out_tokens),
    )

    prompt = f"""Summarize this user's day based on their screen activities.

Rules:
- Be SPECIFIC: mention actual names, email subjects, chat contacts, repo names — not vague descriptions
- Scale your response to the data: {act_count} activities = {1 if act_count <= 5 else 2 if act_count <= 15 else 3}-{2 if act_count <= 5 else 3 if act_count <= 15 else 5} short paragraphs
- Don't pad with filler. If there's little data, write a short summary
- Use the "Screen content" fields for specific details (who messaged, what emails, etc.)

Activities:
{acts_text}

Write the summary:"""

    try:
        summary_text = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=out_tokens,
            ),
        )
    except Exception as e:
        summary_text = f"Summary generation failed: {e}"

    # Compute day metrics (productive hours, categories, top repos)
    metrics = _compute_day_metrics(activities)

    summary_obj = DailySummary(
        date=target,
        summary=summary_text,
        total_activities=len(activities),
        category_breakdown=metrics["category_breakdown"],
        productive_hours=metrics["productive_hours"],
        top_repos=metrics["top_repos"],
    )
    db.upsert_daily_summary(summary_obj)

    # Fire integrations
    _fire_summary_integrations(target, summary_text, "", act_count)

    return {"date": target, "summary": {"summary": summary_text}}


@router.post("/standup/generate")
async def generate_standup(
    date: str = Query(default=None),
):
    """Generate standup notes using the configured LLM backend."""
    from screenmind.engine import llm_client

    target = date or str(__import__("datetime").date.today())
    activities = db.get_activities_by_date(target, limit=200)

    if not activities:
        return {"date": target, "standup": "No activities to summarize."}

    full_block, _ = _build_activity_block(activities, 15, 200, budget_chars=10**9)
    window = _pick_window(len(full_block), 1024)
    out_tokens = min(1024, window // 2)
    acts_text, _ = _build_activity_block(
        activities, max_rich=15, rich_chars=200,
        budget_chars=_budget_chars(window, out_tokens),
    )

    prompt = f"""Generate standup notes from these screen activities.

Rules:
- Be SPECIFIC: use actual names, subjects, contacts from the "Content" fields
- Keep each bullet point to 1 line — no vague descriptions
- If few activities, keep it short (2-3 bullets per section max)
- "Blockers" should be real issues visible in the data, or say "None identified"

Format:
## Yesterday / Today
- Specific things done (e.g. "Replied to aachii on Discord", "Checked Gmail inbox — portfolio/main")
## Blockers
- Real issues or "None identified"
## Plan
- Concrete next steps based on what was seen

Activities:
{acts_text}"""

    try:
        standup = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=out_tokens,
            ),
        )
    except Exception as e:
        standup = f"Standup generation failed: {e}"

    # Save standup to DB alongside summary — include metrics so we don't
    # clobber values computed by generate_summary (upsert is unconditional
    # on numeric/JSON columns).
    from screenmind.storage.models import DailySummary
    metrics = _compute_day_metrics(activities)
    standup_summary = DailySummary(
        date=target,
        summary="",  # Don't overwrite existing summary
        total_activities=len(activities),
        category_breakdown=metrics["category_breakdown"],
        productive_hours=metrics["productive_hours"],
        top_repos=metrics["top_repos"],
    )
    db.upsert_daily_summary(standup_summary, standup=standup)

    # Fire integrations
    _fire_summary_integrations(target, "", standup, len(activities))

    return {"date": target, "standup": standup}


def _fire_summary_integrations(date_str: str, summary: str, standup: str, activity_count: int):
    """Fire all enabled integrations after summary/standup generation."""
    try:
        if settings.obsidian_enabled and settings.obsidian_vault_path:
            from screenmind.integrations.obsidian import export_summary
            export_summary(settings.obsidian_vault_path, date_str, summary, standup, activity_count)
    except Exception as e:
        logger.error(f"Obsidian error: {e}")

    try:
        if settings.notion_enabled and settings.notion_token:
            from screenmind.integrations.notion import export_summary
            export_summary(settings.notion_token, settings.notion_database_id, date_str, summary, standup, activity_count)
    except Exception as e:
        logger.error(f"Notion error: {e}")

    try:
        if settings.webhook_enabled and settings.webhook_url:
            from screenmind.integrations.webhooks import fire
            fire("daily_summary", {
                "date": date_str,
                "summary": summary,
                "standup": standup,
                "activity_count": activity_count,
            }, settings.webhook_url, settings.webhook_secret, settings.webhook_events, settings.webhook_headers)
    except Exception as e:
        logger.error(f"Webhook error: {e}")
