"""Comprehensive tests for capture and analysis workers."""
import time

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from screenmind.workers.capture_worker import CaptureWorker, CaptureResult
from screenmind.engine.llm_client import InferenceCancelled


class TestCaptureWorker:
    """Tests for the capture worker."""

    def _make_worker(self):
        queue = asyncio.Queue(maxsize=100)
        return CaptureWorker(queue=queue), queue

    def test_starts_paused(self):
        worker, _ = self._make_worker()
        assert worker.is_paused is True
        assert worker._running is False

    def test_pause_resume(self):
        worker, _ = self._make_worker()
        worker.resume(source="test")
        assert worker.is_paused is False
        worker.pause(source="test")
        assert worker.is_paused is True

    def test_stats_keys(self):
        worker, _ = self._make_worker()
        stats = worker.stats
        assert "running" in stats
        assert "paused" in stats
        assert "captures" in stats
        assert "skipped" in stats

    def test_trigger_bookmark(self):
        worker, _ = self._make_worker()
        assert worker._pending_bookmark is False
        worker.trigger_bookmark()
        assert worker._pending_bookmark is True

    def test_stop_sets_running_false(self):
        worker, _ = self._make_worker()
        worker._running = True
        worker.stop()
        assert worker._running is False

    def test_pause_resets_dedup(self):
        """Pausing resets the dedup hash so next capture is always fresh."""
        worker, _ = self._make_worker()
        worker._paused = False  # Must be unpaused for pause() to run (idempotent guard)
        worker._dedup._last_hash = "something"
        worker.pause(source="test")
        assert worker._dedup._last_hash is None

    def test_initial_counts_zero(self):
        worker, _ = self._make_worker()
        assert worker._capture_count == 0
        assert worker._skip_count == 0
        assert worker._consecutive_skips == 0


class TestCaptureResult:
    """Tests for CaptureResult dataclass."""

    def test_create_basic(self, tmp_path):
        result = CaptureResult(
            filepath=tmp_path / "test.jpg",
            timestamp=datetime.now(),
            window_title="Test Window",
            app_name="TestApp",
        )
        assert result.app_name == "TestApp"
        assert result.bookmarked is False
        assert result.activity_id is None
        assert result.a11y_text is None
        assert result.phash is None

    def test_create_bookmarked(self, tmp_path):
        result = CaptureResult(
            filepath=tmp_path / "test.jpg",
            timestamp=datetime.now(),
            bookmarked=True,
        )
        assert result.bookmarked is True


class TestAnalysisWorkerStats:
    """Tests for analysis worker state management."""

    def test_flush_queue(self):
        """flush_queue drains all items."""
        from screenmind.workers.analysis_worker import AnalysisWorker

        queue = asyncio.Queue(maxsize=100)
        db = MagicMock()
        worker = AnalysisWorker(queue=queue, database=db)

        # Add some items
        for i in range(5):
            queue.put_nowait(MagicMock())

        assert queue.qsize() == 5
        worker.flush_queue()
        assert queue.qsize() == 0

    def test_stats_keys(self):
        from screenmind.workers.analysis_worker import AnalysisWorker

        queue = asyncio.Queue(maxsize=100)
        db = MagicMock()
        worker = AnalysisWorker(queue=queue, database=db)

        stats = worker.stats
        assert "running" in stats
        assert "processed" in stats
        assert "errors" in stats
        assert "queue_size" in stats
        assert "cache_hits" in stats
        assert "cache_size" in stats

    def test_initial_state(self):
        from screenmind.workers.analysis_worker import AnalysisWorker

        queue = asyncio.Queue(maxsize=100)
        db = MagicMock()
        worker = AnalysisWorker(queue=queue, database=db)

        assert worker._processed == 0
        assert worker._errors == 0
        assert worker._cache_hits == 0
        assert len(worker._app_cache) == 0
        assert len(worker._priority_items) == 0

    def test_stop(self):
        from screenmind.workers.analysis_worker import AnalysisWorker

        queue = asyncio.Queue(maxsize=100)
        db = MagicMock()
        worker = AnalysisWorker(queue=queue, database=db)
        worker._running = True
        worker.stop()
        assert worker._running is False


class TestURLExtraction:
    """Tests for URL extraction in analysis worker."""

    def test_extract_url_basic(self):
        from screenmind.workers.analysis_worker import _extract_url
        assert _extract_url("Visit https://github.com/user/repo today") == "https://github.com/user/repo"

    def test_extract_url_none_for_empty(self):
        from screenmind.workers.analysis_worker import _extract_url
        assert _extract_url("") is None
        assert _extract_url("no urls here") is None

    def test_extract_url_filters_noise(self):
        from screenmind.workers.analysis_worker import _extract_url
        # localhost and CDN URLs should be filtered
        assert _extract_url("http://localhost:3000/api") is None
        assert _extract_url("https://cdn.example.com/file.js") is None

    def test_extract_all_urls(self):
        from screenmind.workers.analysis_worker import _extract_all_urls
        text = "Check https://github.com and https://dev.to for updates"
        urls = _extract_all_urls(text)
        assert len(urls) == 2
        assert "https://github.com" in urls[0]

    def test_extract_url_strips_punctuation(self):
        from screenmind.workers.analysis_worker import _extract_all_urls
        urls = _extract_all_urls("See https://example.com/page.")
        assert urls[0] == "https://example.com/page"


class TestBackfillFailureLoop:
    """Regression: a permanently-failing row must not be retried every 2s.

    Bug: backfill picked the same 'Analysis failed' row, the identical-cache
    tier copied the failure placeholder back into the DB, and the query
    matched it again — an infinite loop spamming the log every 2 seconds.
    """

    def _worker(self, rows, summary_after):
        """AnalysisWorker whose DB returns `rows` for the backfill query and
        `(summary_after,)` for the post-processing summary check."""
        from screenmind.workers.analysis_worker import AnalysisWorker

        db = MagicMock()
        conn = MagicMock()

        def execute(sql, params=None):
            cur = MagicMock()
            if "screenshot_path" in sql:
                cur.fetchall.return_value = rows
            else:
                cur.fetchone.return_value = (summary_after,)
            return cur

        conn.execute.side_effect = execute
        db._get_conn.return_value = conn
        return AnalysisWorker(queue=asyncio.Queue(), database=db)

    def _patch_image_load(self):
        """Stub image decode + pHash so any file path passes the checks."""
        img = MagicMock()
        return (
            patch("screenmind.privacy.encryption.open_image", return_value=img),
            patch("imagehash.phash", return_value=MagicMock()),
        )

    async def test_failed_row_gets_cooldown(self):
        """Row still failing after backfill enters cooldown (no 2s retry loop)."""
        row = (450, __file__, "Jump List", "ShellExperienceHost", None, None, "2026-08-18 07:00:00")
        worker = self._worker([row], "Analysis failed")
        p1, p2 = self._patch_image_load()
        with p1, p2:
            worker._process = AsyncMock()
            await worker._backfill_skipped()
        assert 450 in worker._backfill_cooldown
        worker._process.assert_awaited_once()

    async def test_row_in_cooldown_is_skipped(self):
        """A cooling-down row is skipped; the next candidate is processed."""
        row1 = (450, __file__, "t1", "app1", None, None, "2026-08-18 07:00:00")
        row2 = (451, __file__, "t2", "app2", None, None, "2026-08-18 07:01:00")
        row2 = (451, __file__, "t2", "app2", None, None)
        worker = self._worker([row1, row2], "Analysis failed")
        worker._backfill_cooldown[450] = time.time()  # fresh cooldown
        p1, p2 = self._patch_image_load()
        with p1, p2:
            worker._process = AsyncMock()
            await worker._backfill_skipped()
        capture = worker._process.await_args[0][0]
        assert capture.activity_id == 451

    async def test_all_rows_cooling_down_is_noop(self):
        """When every candidate is cooling down, nothing is processed."""
        row = (450, __file__, "t", "app", None, None, "2026-08-18 07:00:00")
        worker = self._worker([row], "Analysis failed")
        worker._backfill_cooldown[450] = time.time()
        p1, p2 = self._patch_image_load()
        with p1, p2:
            worker._process = AsyncMock()
            await worker._backfill_skipped()
        worker._process.assert_not_awaited()

    async def test_successful_backfill_clears_cooldown(self):
        """A real summary after backfill clears the cooldown for that row."""
        row = (450, __file__, "t", "app", None, None, "2026-08-18 07:00:00")
        worker = self._worker([row], "Reading documentation on GitHub")
        worker._backfill_cooldown[450] = time.time() - 700  # expired cooldown
        p1, p2 = self._patch_image_load()
        with p1, p2:
            worker._process = AsyncMock()
            await worker._backfill_skipped()
        worker._process.assert_awaited_once()
        assert 450 not in worker._backfill_cooldown

    async def test_backfill_exception_sets_cooldown(self):
        """An exception during backfill also backs off instead of looping."""
        row = (450, __file__, "t", "app", None, None, "2026-08-18 07:00:00")
        worker = self._worker([row], "Analysis failed")
        p1, p2 = self._patch_image_load()
        with p1, p2:
            worker._process = AsyncMock(side_effect=RuntimeError("boom"))
            await worker._backfill_skipped()
        assert 450 in worker._backfill_cooldown



class TestManualBackfillBatch:
    """POST /api/timeline/backfill — batch re-analysis of pending rows."""

    def _worker(self, rows):
        """AnalysisWorker whose DB returns `rows` for the backfill query."""
        from screenmind.workers.analysis_worker import AnalysisWorker

        db = MagicMock()
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = rows
        db._get_conn.return_value = conn
        return AnalysisWorker(queue=asyncio.Queue(), database=db)

    async def test_batch_processes_rows_and_counts(self):
        """Each row's result lands in the right status bucket."""
        rows = [(1, "a.jpg", "t1", "app1", None, None),
                (2, "b.jpg", "t2", "app2", None, None),
                (3, "c.jpg", "t3", "app3", None, None)]
        worker = self._worker(rows)
        worker._backfill_row = AsyncMock(side_effect=["done", "failed", "skipped"])
        worker._backfill_status = {"running": True, "requested": 3,
                                   "analyzed": 0, "failed": 0, "skipped": 0}
        await worker._run_backfill_batch(rows, MagicMock())
        status = worker.backfill_status
        assert status["running"] is False
        assert (status["analyzed"], status["failed"], status["skipped"]) == (1, 1, 1)

    async def test_batch_preempts_on_fresh_capture(self):
        """A fresh capture in the queue stops the batch immediately."""
        rows = [(1, "a.jpg", "t", "app", None, None)]
        worker = self._worker(rows)
        worker._backfill_row = AsyncMock()
        worker._backfill_status = {"running": True, "requested": 1,
                                   "analyzed": 0, "failed": 0, "skipped": 0}
        await worker._queue.put(MagicMock())  # fresh capture arrived
        await worker._run_backfill_batch(rows, MagicMock())
        worker._backfill_row.assert_not_awaited()
        assert worker.backfill_status["running"] is False

    def test_start_gates_when_already_running(self):
        """A second start while a batch runs returns already_running."""
        worker = self._worker([(1, "a.jpg", "t", "app", None, None)])
        worker._backfill_status["running"] = True
        result = worker.start_backfill_batch(limit=10)
        assert result["already_running"] is True

    def test_start_empty_backlog(self):
        """No pending rows → zeros, no batch started."""
        worker = self._worker([])
        result = worker.start_backfill_batch(limit=10)
        assert result == {"requested": 0, "analyzed": 0, "failed": 0, "skipped": 0}
        assert worker.backfill_running() is False

    async def test_start_spawns_batch_and_completes(self):
        """start_backfill_batch returns immediately; the task drains the rows."""
        rows = [(1, "a.jpg", "t", "app", None, None)]
        worker = self._worker(rows)
        worker._backfill_row = AsyncMock(return_value="done")
        result = worker.start_backfill_batch(limit=10)
        assert result["running"] is True
        assert result["requested"] == 1
        for _ in range(10):
            await asyncio.sleep(0)
            if not worker.backfill_running():
                break
        assert worker.backfill_status["analyzed"] == 1
        worker._backfill_row.assert_awaited_once()

    async def test_idle_loop_stands_down_during_batch(self):
        """The 2s idle backfill doesn't race the manual batch."""
        worker = self._worker([(1, "a.jpg", "t", "app", None, None)])
        worker._backfill_status["running"] = True
        worker._backfill_one = AsyncMock()
        await worker._backfill_skipped()
        worker._backfill_one.assert_not_awaited()

class TestFailureSummaryHelper:
    """Tests for _is_failure_summary — guards cache writes and backfill."""

    def test_detects_bare_and_detailed_failures(self):
        from screenmind.workers.analysis_worker import _is_failure_summary
        assert _is_failure_summary("Analysis failed") is True
        assert _is_failure_summary("Analysis failed: HTTP 500") is True

    def test_rejects_real_and_skip_summaries(self):
        from screenmind.workers.analysis_worker import _is_failure_summary
        assert _is_failure_summary("Watching YouTube") is False
        assert _is_failure_summary("Skipped (analysis backlog)") is False
        assert _is_failure_summary("") is False
        assert _is_failure_summary(None) is False


class TestTextModelSceneWiring:
    """_process must generate scene_description via the text-only model
    (generate_scene_from_text) and prefer it over the vision model's field."""

    def _worker(self):
        from screenmind.workers.analysis_worker import AnalysisWorker
        return AnalysisWorker(queue=asyncio.Queue(), database=MagicMock())

    def _capture(self):
        img = MagicMock()
        img.size = (1920, 1080)
        return CaptureResult(
            filepath=Path(__file__),
            timestamp=datetime.now(),
            window_title="main.py - VS Code",
            app_name="Code",
            bookmarked=False,
            image=img,
            activity_id=7,
            a11y_text=None,
            phash=None,  # forces full tier (no cache comparison)
            is_backfill=False,
        )


    async def test_text_scene_overwrites_vision_scene(self):
        worker = self._worker()
        worker._ocr = MagicMock(is_available=True,
                                extract_text_with_boxes=MagicMock(return_value=("screen text " * 10, [])))
        worker._analyzer.analyze_screenshot_fast = MagicMock(
            return_value=(MagicMock(
                scene_description="vision scene",
                activity_summary="Editing main.py",
                activity_category="coding",
                visible_text_snippets=[], detailed_context="", app_name="VS Code",
            ), []))
        worker._analyzer.generate_scene_from_text = MagicMock(return_value="text scene")
        worker._dev_context.is_coding_activity = MagicMock(return_value=False)
        worker._embedder = None

        with patch("screenmind.workers.analysis_worker.settings",
                   sensitive_filter_enabled=False, auto_bookmark=False):
            await worker._process(self._capture())

        saved = worker._db.update_activity_analysis.call_args.kwargs["analysis"]
        assert saved.scene_description == "text scene"
        # Text source got the OCR text, not the screenshot
        worker._analyzer.generate_scene_from_text.assert_called_once()
        assert worker._analyzer.generate_scene_from_text.call_args.kwargs["ocr_text"]

    async def test_vision_scene_kept_when_text_scene_fails(self):
        worker = self._worker()
        worker._ocr = MagicMock(is_available=True,
                                extract_text_with_boxes=MagicMock(return_value=("screen text " * 10, [])))
        vision_record = MagicMock(
            scene_description="vision scene",
            activity_summary="Editing main.py",
            activity_category="coding",
            visible_text_snippets=[], detailed_context="", app_name="VS Code",
        )
        worker._analyzer.analyze_screenshot_fast = MagicMock(return_value=(vision_record, []))
        worker._analyzer.generate_scene_from_text = MagicMock(return_value=None)
        worker._dev_context.is_coding_activity = MagicMock(return_value=False)
        worker._embedder = None

        with patch("screenmind.workers.analysis_worker.settings",
                   sensitive_filter_enabled=False, auto_bookmark=False):
            await worker._process(self._capture())

        saved = worker._db.update_activity_analysis.call_args.kwargs["analysis"]
        assert saved.scene_description == "vision scene"


class TestSceneBackfillBatch:
    """POST /api/timeline/scenes/backfill — generate missed scene descriptions."""

    def _worker(self, rows):
        """AnalysisWorker whose DB returns `rows` for the scene-backfill query."""
        from screenmind.workers.analysis_worker import AnalysisWorker

        db = MagicMock()
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = rows
        db._get_conn.return_value = conn
        worker = AnalysisWorker(queue=asyncio.Queue(), database=db)
        worker._embedder = MagicMock()  # Skip model download in _ensure_embedder
        return worker

    def _row(self, ocr="plenty of screen text to narrate in detail"):
        return (1, ocr, None, "Code", "main.py - VS Code",
                "Editing main.py", "details", '["snippet"]', "coding")

    async def test_batch_counts_results(self):
        """Each row's result lands in the right status bucket."""
        rows = [self._row(), self._row(), self._row()]
        worker = self._worker(rows)
        worker._scene_backfill_row = AsyncMock(side_effect=["done", "failed", "skipped"])
        worker._scene_backfill_status = {"running": True, "requested": 3,
                                         "generated": 0, "failed": 0, "skipped": 0}
        await worker._run_scene_backfill_batch(rows)
        status = worker.scene_backfill_status
        assert status["running"] is False
        assert (status["generated"], status["failed"], status["skipped"]) == (1, 1, 1)

    async def test_batch_preempts_on_fresh_capture(self):
        """A fresh capture in the queue stops the batch immediately."""
        rows = [self._row()]
        worker = self._worker(rows)
        worker._scene_backfill_row = AsyncMock()
        worker._scene_backfill_status = {"running": True, "requested": 1,
                                         "generated": 0, "failed": 0, "skipped": 0}
        await worker._queue.put(MagicMock())  # fresh capture arrived
        await worker._run_scene_backfill_batch(rows)
        worker._scene_backfill_row.assert_not_awaited()
        assert worker.scene_backfill_status["running"] is False

    def test_start_gates_when_already_running(self):
        worker = self._worker([self._row()])
        worker._scene_backfill_status["running"] = True
        result = worker.start_scene_backfill(limit=10)
        assert result["already_running"] is True

    def test_start_gates_when_analysis_backfill_running(self):
        """Single-slot LLM — the two batches never run together."""
        worker = self._worker([self._row()])
        worker._backfill_status["running"] = True
        result = worker.start_scene_backfill(limit=10)
        assert "error" in result
        assert result["running"] is False

    def test_start_empty_backlog(self):
        """No rows missing scenes → zeros, no batch started."""
        worker = self._worker([])
        result = worker.start_scene_backfill(limit=10)
        assert result == {"requested": 0, "generated": 0, "failed": 0, "skipped": 0}
        assert worker.scene_backfill_status["running"] is False

    async def test_row_generates_scene_and_updates_db(self):
        """Generated scene + refreshed embedding land in update_scene_description."""
        worker = self._worker([])
        worker._analyzer.generate_scene_from_text = MagicMock(return_value="A code editor...")
        worker._embedder.embed_activity = MagicMock(return_value=[0.1] * 384)
        result = await worker._scene_backfill_row(self._row())
        assert result == "done"
        worker._db.update_scene_description.assert_called_once_with(
            1, "A code editor...", embedding=[0.1] * 384)
        # Text source got the stored OCR text + row context, never a screenshot
        kwargs = worker._analyzer.generate_scene_from_text.call_args.kwargs
        assert kwargs["ocr_text"]
        assert kwargs["app_name"] == "Code"
        assert kwargs["window_title"] == "main.py - VS Code"
        # Embedding refresh includes the new scene
        assert worker._embedder.embed_activity.call_args.kwargs["scene_description"] == "A code editor..."

    async def test_row_failed_when_text_model_returns_none(self):
        """Model unreachable / empty completion → failed, DB untouched."""
        worker = self._worker([])
        worker._analyzer.generate_scene_from_text = MagicMock(return_value=None)
        result = await worker._scene_backfill_row(self._row())
        assert result == "failed"
        worker._db.update_scene_description.assert_not_called()

    async def test_row_skipped_when_text_too_short(self):
        """Source text below the 40-char floor → skipped, no LLM call."""
        worker = self._worker([])
        worker._analyzer.generate_scene_from_text = MagicMock()
        result = await worker._scene_backfill_row(self._row(ocr="tiny"))
        assert result == "skipped"
        worker._analyzer.generate_scene_from_text.assert_not_called()
        worker._db.update_scene_description.assert_not_called()

    def test_fetch_query_selects_only_missing_scenes(self, db):
        """Query picks analyzed rows with a missing scene and usable text only."""
        from screenmind.workers.analysis_worker import AnalysisWorker
        from screenmind.storage.models import ScreenshotEntry, ActivityRecord

        worker = AnalysisWorker(queue=asyncio.Queue(), database=db)
        OCR = "plenty of screen text to narrate in detail"

        def _insert(hour, **rec_kwargs):
            aid = db.insert_activity(ScreenshotEntry(
                timestamp=datetime(2026, 8, 1, hour, 0, 0),
                screenshot_path=f"/tmp/{hour}.jpg", analyzed=True))
            ocr = rec_kwargs.pop("ocr", OCR)
            db.update_activity_analysis(
                aid, ActivityRecord(app_name="Code", activity_category="coding",
                                    **rec_kwargs),
                ocr_text=ocr)
            return aid

        aid_missing = _insert(10)                                    # selected
        _insert(11, scene_description="A VS Code window with main.py")  # has scene
        _insert(12, activity_summary="Analysis failed: timeout")     # failure placeholder
        _insert(13, activity_summary="Skipped (screenshot deleted)")  # skip placeholder
        aid_no_text = db.insert_activity(ScreenshotEntry(
            timestamp=datetime(2026, 8, 1, 14, 0, 0),
            screenshot_path="/tmp/14.jpg", analyzed=True))
        db.update_activity_analysis(
            aid_no_text, ActivityRecord(app_name="Code", activity_category="coding",
                                        activity_summary="Editing main.py"))  # no OCR text

        rows = worker._fetch_scene_backfill_rows(100)
        assert [r[0] for r in rows] == [aid_missing]

    def test_stats_expose_scene_backfill(self):
        worker = self._worker([])
        assert "scenes" in worker.stats
        assert worker.stats["scenes"]["running"] is False

    def test_backfill_running_includes_scene_batch(self):
        """The idle loop stands down for scene batches too."""
        worker = self._worker([])
        worker._scene_backfill_status["running"] = True
        assert worker.backfill_running() is True

    async def test_idle_loop_stands_down_during_scene_batch(self):
        worker = self._worker([])
        worker._scene_backfill_status["running"] = True
        worker._backfill_one = AsyncMock()
        await worker._backfill_skipped()
        worker._backfill_one.assert_not_awaited()


class TestQualityGateSkipsScene:
    """Regression: a missing scene_description must NOT trigger a second vision
    call — the text model owns that field now (step 3e). Gating on it halved
    throughput with split text/vision models and grew the queue."""

    def _worker(self):
        from screenmind.workers.analysis_worker import AnalysisWorker
        return AnalysisWorker(queue=asyncio.Queue(), database=MagicMock())

    def _capture(self):
        img = MagicMock()
        img.size = (1920, 1080)
        return CaptureResult(
            filepath=Path(__file__),
            timestamp=datetime.now(),
            window_title="main.py - VS Code",
            app_name="Code",
            bookmarked=False,
            image=img,
            activity_id=7,
            a11y_text=None,
            phash=None,
            is_backfill=False,
        )

    def _wire(self, worker, vision_record, analyze_fn):
        worker._ocr = MagicMock(is_available=True,
                                extract_text_with_boxes=MagicMock(return_value=("screen text " * 10, [])))
        worker._analyzer.analyze_screenshot_fast = analyze_fn
        worker._dev_context.is_coding_activity = MagicMock(return_value=False)
        worker._embedder = None

    async def test_missing_scene_does_not_retry_vision(self):
        worker = self._worker()
        vision_record = MagicMock(
            scene_description="",  # vision model returned no scene
            activity_summary="Editing main.py",
            activity_category="coding",
            visible_text_snippets=[], detailed_context="", app_name="VS Code",
        )
        analyze_fn = MagicMock(return_value=(vision_record, []))
        self._wire(worker, vision_record, analyze_fn)
        worker._analyzer.generate_scene_from_text = MagicMock(return_value="text scene")

        with patch("screenmind.workers.analysis_worker.settings",
                   sensitive_filter_enabled=False, auto_bookmark=False):
            await worker._process(self._capture())

        assert analyze_fn.call_count == 1  # no quality-gate retry
        saved = worker._db.update_activity_analysis.call_args.kwargs["analysis"]
        assert saved.scene_description == "text scene"

    async def test_missing_summary_still_retries(self):
        """The gate still fires for fields only the vision model can fill."""
        worker = self._worker()
        bad = MagicMock(scene_description="scene", activity_summary="",
                        activity_category="coding",
                        visible_text_snippets=[], detailed_context="", app_name="VS Code")
        good = MagicMock(scene_description="scene", activity_summary="Editing main.py",
                         activity_category="coding",
                         visible_text_snippets=[], detailed_context="", app_name="VS Code")
        analyze_fn = MagicMock(side_effect=[(bad, []), (good, [])])
        self._wire(worker, None, analyze_fn)
        worker._analyzer.generate_scene_from_text = MagicMock(return_value=None)

        with patch("screenmind.workers.analysis_worker.settings",
                   sensitive_filter_enabled=False, auto_bookmark=False):
            await worker._process(self._capture())

        assert analyze_fn.call_count == 2
        saved = worker._db.update_activity_analysis.call_args.kwargs["analysis"]
        assert saved.activity_summary == "Editing main.py"


class TestLiveStatus:
    """Live status: current-item snapshot + SSE event bus."""

    def _worker(self):
        from screenmind.workers.analysis_worker import AnalysisWorker
        db = MagicMock()
        db.get_day_number.return_value = 3
        worker = AnalysisWorker(queue=asyncio.Queue(), database=db)
        worker._loop = asyncio.get_event_loop()
        return worker

    def _capture(self):
        img = MagicMock()
        img.size = (1920, 1080)
        return CaptureResult(
            filepath=Path(__file__),
            timestamp=datetime(2026, 5, 16, 10, 0, 0),
            window_title="main.py - VS Code",
            app_name="Code",
            bookmarked=False,
            image=img,
            activity_id=7,
            a11y_text=None,
            phash=None,
            is_backfill=False,
        )

    async def test_begin_current_builds_snapshot(self):
        worker = self._worker()
        worker._begin_current(7, self._capture())
        snap = worker.current_status
        assert snap["activity_id"] == 7
        assert snap["day_number"] == 3
        assert snap["date"] == "2026-05-16"
        assert snap["stage"] == "processing"
        assert snap["response"] == ""

    async def test_stream_chunk_accumulates_and_publishes_delta(self):
        worker = self._worker()
        q = worker.subscribe()
        worker._begin_current(7, self._capture())
        worker._stream_chunk("Hel")
        worker._stream_chunk("lo")
        await asyncio.sleep(0)  # let call_soon_threadsafe callbacks deliver
        assert worker.current_status["response"] == "Hello"
        deltas = []
        while not q.empty():
            ev = q.get_nowait()
            if ev["type"] == "delta":
                deltas.append(ev["text"])
        assert deltas == ["Hel", "lo"]
        worker.unsubscribe(q)

    async def test_set_stage_resets_response(self):
        """Each stage streams fresh model output — stale text must not leak."""
        worker = self._worker()
        worker._begin_current(7, self._capture())
        worker._stream_chunk("scene text")
        worker._set_stage("analyzing")
        assert worker.current_status["stage"] == "analyzing"
        assert worker.current_status["response"] == ""

    async def test_finish_current_sets_summary(self):
        worker = self._worker()
        worker._begin_current(7, self._capture())
        worker._finish_current("done", summary="Editing code")
        snap = worker.current_status
        assert snap["stage"] == "done"
        assert snap["summary"] == "Editing code"

    async def test_stats_expose_current(self):
        worker = self._worker()
        assert worker.stats["current"] is None
        worker._begin_current(7, self._capture())
        assert worker.stats["current"]["activity_id"] == 7

    async def test_slow_consumer_drops_oldest_not_newest(self):
        """A stalled SSE client keeps receiving the freshest events."""
        worker = self._worker()
        q = asyncio.Queue(maxsize=2)
        worker._subscribers.add(q)
        for i in range(5):
            worker._enqueue(q, {"type": "delta", "text": str(i)})
        events = [q.get_nowait()["text"] for _ in range(q.qsize())]
        assert events[-1] == "4"  # newest survives

    async def test_process_emits_full_lifecycle(self):
        """_process drives processing → ocr → analyzing → done, streaming
        the model response and finishing with the summary."""
        worker = self._worker()
        worker._ocr = MagicMock(is_available=True,
                                extract_text_with_boxes=MagicMock(return_value=("screen text " * 10, [])))

        def _analyze(**kwargs):
            # The analyzer must receive the stream callback and invoke it
            cb = kwargs.get("stream_callback")
            assert cb is not None
            cb("model ")
            cb("answer")
            rec = MagicMock(
                scene_description=None,
                activity_summary="Editing main.py",
                activity_category="coding",
                visible_text_snippets=[], detailed_context="", app_name="VS Code",
            )
            return rec, []

        worker._analyzer.analyze_screenshot_fast = MagicMock(side_effect=_analyze)
        worker._analyzer.generate_scene_from_text = MagicMock(return_value=None)
        worker._dev_context.is_coding_activity = MagicMock(return_value=False)
        worker._embedder = None

        with patch("screenmind.workers.analysis_worker.settings",
                   sensitive_filter_enabled=False, auto_bookmark=False):
            await worker._process(self._capture())

        snap = worker.current_status
        assert snap["stage"] == "done"
        assert snap["summary"] == "Editing main.py"
        assert snap["response"] == "model answer"

    async def test_process_failure_marks_failed(self):
        worker = self._worker()
        worker._ocr = MagicMock(is_available=True,
                                extract_text_with_boxes=MagicMock(return_value=("text " * 20, [])))
        worker._analyzer.analyze_screenshot_fast = MagicMock(side_effect=RuntimeError("inference exploded"))
        worker._analyzer.generate_scene_from_text = MagicMock(return_value=None)
        worker._dev_context.is_coding_activity = MagicMock(return_value=False)
        worker._embedder = None

        with patch("screenmind.workers.analysis_worker.settings",
                   sensitive_filter_enabled=False, auto_bookmark=False):
            await worker._process(self._capture())

        snap = worker.current_status
        assert snap["stage"] == "failed"
        assert "inference exploded" in snap["summary"]

    async def test_process_cancellation_marks_yielded(self):
        worker = self._worker()
        worker._ocr = MagicMock(is_available=True,
                                extract_text_with_boxes=MagicMock(return_value=("text " * 20, [])))
        worker._analyzer.analyze_screenshot_fast = MagicMock(side_effect=InferenceCancelled("chat"))
        worker._embedder = None

        with patch("screenmind.workers.analysis_worker.settings",
                   sensitive_filter_enabled=False, auto_bookmark=False):
            await worker._process(self._capture())

        assert worker.current_status["stage"] == "yielded"
        # Item re-queued at front for resumption
        assert len(worker._priority_items) == 1
