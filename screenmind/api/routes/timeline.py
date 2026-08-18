"""Timeline routes — list activities, activity detail, delete, reanalyze."""

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from starlette.responses import StreamingResponse

from screenmind.config import settings
from screenmind.api.dependencies import db

router = APIRouter(prefix="/api", tags=["timeline"])

# Serialize re-analyze requests — llama.cpp is single-slot, concurrent requests get 500
_reanalyze_lock = asyncio.Lock()


@router.get("/timeline")
async def get_timeline(
    date: str = Query(default=None),
    since: str = Query(default=None, description="ISO timestamp cutoff — only return activities after this time"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Get activities for a specific date, optionally filtered by since timestamp.

    Returns total (count for the whole day, ignoring limit/offset) so the
    dashboard can paginate. Each activity carries day_number — its
    chronological position within the day (e.g. 2026-08-17_42).
    """
    target_date = date or str(__import__("datetime").date.today())
    activities = db.get_activities_by_date(target_date, limit=limit, offset=offset)
    total = db.count_activities_by_date(target_date)
    if since:
        activities = [a for a in activities if a.get("timestamp", "") >= since]
    for a in activities:
        if a.get("screenshot_path"):
            a["screenshot_url"] = f"/api/screenshot/{a['id']}"
    return {"date": target_date, "activities": activities, "total": total}


@router.get("/analysis/stream")
async def analysis_stream():
    """SSE stream of live analysis progress for the Timeline status panel.

    Events:
      status — full snapshot of the item being processed (stage, day_number,
               accumulated model response, summary). Sent on connect and on
               every stage transition.
      delta  — incremental model output during the analyzing stage.

    A keepalive comment goes out every 15s so proxies don't drop the socket.
    """
    from screenmind.api import dependencies as deps
    if deps.analysis_worker is None:
        raise HTTPException(status_code=503, detail="Analysis worker not available")
    worker = deps.analysis_worker

    async def generate():
        q = worker.subscribe()
        try:
            snapshot = worker.current_status
            if snapshot:
                yield f"data: {json.dumps({'type': 'status', **snapshot})}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            worker.unsubscribe(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/activity/{activity_id}")
async def get_activity(activity_id: int):
    """Get a single activity with full details."""
    activity = db.get_activity_by_id(activity_id)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    if activity.get("screenshot_path"):
        activity["screenshot_url"] = f"/api/screenshot/{activity['id']}"
    return activity


@router.delete("/activities/{activity_id}")
async def delete_activity(activity_id: int):
    """Delete a single activity and its screenshot."""
    conn = db._get_conn()
    row = conn.execute(
        "SELECT screenshot_path FROM activities WHERE id = ?", (activity_id,)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Activity not found")

    # Delete screenshot file
    ss_path = row[0] if row else None
    if ss_path and Path(ss_path).exists():
        try:
            Path(ss_path).unlink()
        except OSError:
            pass

    # Delete from DB
    conn.execute("DELETE FROM activities WHERE id = ?", (activity_id,))
    conn.commit()
    return {"deleted": True, "activity_id": activity_id}


@router.post("/activities/{activity_id}/reanalyze")
async def reanalyze_activity(activity_id: int):
    """Re-run Gemma analysis + layout detection on a single activity."""
    conn = db._get_conn()
    row = conn.execute(
        "SELECT screenshot_path, ocr_text, ocr_boxes, detected_app, window_title FROM activities WHERE id = ?",
        (activity_id,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Activity not found")

    ss_path = row[0]
    if not ss_path or not Path(ss_path).exists():
        raise HTTPException(status_code=400, detail="Screenshot not found")

    from PIL import Image as _Image
    from screenmind.engine.analyzer import GemmaAnalyzer
    from screenmind.engine.layout_analyzer import organize_ocr_text, cluster_ocr_layout
    from screenmind.config import settings as app_settings

    async with _reanalyze_lock:
        try:
            from screenmind.privacy.encryption import open_image as _enc_open
            img = _enc_open(ss_path)

            # Re-run Gemma analysis (respects analysis_mode setting)
            ocr_text = row[1] or ""
            app_name = row[3] or None
            window_title = row[4] or None

            # Extract URLs from OCR text (same as analysis_worker)
            from screenmind.workers.analysis_worker import _extract_all_urls
            active_urls = _extract_all_urls(ocr_text) if ocr_text else []

            analyzer = GemmaAnalyzer()
            try:
                _MODE_MAP = {
                    "fast": analyzer.analyze_screenshot_fast,
                    "balanced": analyzer.analyze_screenshot_balanced,
                    "merged": analyzer.analyze_screenshot,
                }
                analyze_fn = _MODE_MAP.get(app_settings.analysis_mode, analyzer.analyze_screenshot_fast)
                record, layout_regions = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: analyze_fn(
                            image=img,
                            window_title=window_title,
                            app_name_hint=app_name,
                            ocr_text=ocr_text,
                            active_urls=active_urls,
                        ),
                    ),
                    timeout=300,
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Re-analysis timed out (server unresponsive)")

            # Organize text using layout regions (Gemma regions or OCR clustering)
            organized_text = ""
            ocr_boxes_raw = row[2]
            if ocr_boxes_raw:
                ocr_boxes = json.loads(ocr_boxes_raw)
                try:
                    screen_w, screen_h = img.size
                    regions = layout_regions if layout_regions else cluster_ocr_layout(ocr_boxes, screen_w, screen_h)
                    organized_text = organize_ocr_text(
                        ocr_boxes, regions, screen_w, screen_h
                    )
                except Exception as e:
                    organized_text = ""  # Non-fatal — skip layout on error

            # Scene description via the text-only model — from OCR text,
            # never the screenshot. Overwrites the vision model's scene
            # when it produces one; on failure the vision scene stays.
            try:
                text_scene = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: analyzer.generate_scene_from_text(
                            ocr_text=ocr_text,
                            organized_text=organized_text,
                            app_name=app_name,
                            window_title=window_title,
                        ),
                    ),
                    timeout=60,
                )
            except Exception:
                text_scene = None  # Non-fatal — vision scene (if any) stays
            if text_scene:
                record.scene_description = text_scene

            # Generate embedding for semantic search
            embedding = None
            try:
                from screenmind.api.dependencies import embedder as _emb
                if _emb:
                    embedding = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: _emb.embed_activity(
                            summary=record.activity_summary,
                            details=record.detailed_context,
                            visible_text=record.visible_text_snippets,
                            app_name=record.app_name,
                            category=record.activity_category,
                            scene_description=record.scene_description,
                        ),
                    )
            except Exception:
                pass  # Non-fatal — search still works via FTS

            # Update DB (handles FTS5 sync automatically)
            db.update_activity_analysis(
                activity_id=activity_id,
                analysis=record,
                embedding=embedding,
                ocr_text=ocr_text,
                organized_text=organized_text,
                analysis_method="reanalyze",
            )

            return {
                "reanalyzed": True,
                "activity_id": activity_id,
                "summary": record.activity_summary,
                "category": record.activity_category,
            }
        except HTTPException:
            raise  # Re-raise timeout 504
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Re-analysis failed: {str(e)[:200]}")


@router.post("/timeline/backfill")
async def backfill_timeline(limit: int = Query(default=100, ge=1, le=500)):
    """Manually backfill unanalyzed/skipped/failed activities.

    Unlike the idle-loop backfill: no 24h window, no cooldown, oldest
    rows first. Starts a background batch and returns immediately —
    progress is visible via GET /api/status (analysis.backfill).
    """
    from screenmind.api import dependencies as deps
    if deps.analysis_worker is None:
        raise HTTPException(status_code=503, detail="Analysis worker not available")
    return deps.analysis_worker.start_backfill_batch(limit=limit)


@router.post("/timeline/scenes/backfill")
async def backfill_scene_descriptions(limit: int = Query(default=100, ge=1, le=500)):
    """Generate scene descriptions for all analyzed activities missing one.

    Text-model only — narration is built from stored OCR/organized text,
    never the screenshot, so no vision re-analysis runs. Starts a background
    batch and returns immediately — progress is visible via GET /api/status
    (analysis.scenes). Serialized with /api/timeline/backfill.
    """
    from screenmind.api import dependencies as deps
    if deps.analysis_worker is None:
        raise HTTPException(status_code=503, detail="Analysis worker not available")
    return deps.analysis_worker.start_scene_backfill(limit=limit)
