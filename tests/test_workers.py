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
        row = (450, __file__, "Jump List", "ShellExperienceHost", "2026-08-18 07:00:00")
        worker = self._worker([row], "Analysis failed")
        p1, p2 = self._patch_image_load()
        with p1, p2:
            worker._process = AsyncMock()
            await worker._backfill_skipped()
        assert 450 in worker._backfill_cooldown
        worker._process.assert_awaited_once()

    async def test_row_in_cooldown_is_skipped(self):
        """A cooling-down row is skipped; the next candidate is processed."""
        row1 = (450, __file__, "t1", "app1", "2026-08-18 07:00:00")
        row2 = (451, __file__, "t2", "app2", "2026-08-18 07:01:00")
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
        row = (450, __file__, "t", "app", "2026-08-18 07:00:00")
        worker = self._worker([row], "Analysis failed")
        worker._backfill_cooldown[450] = time.time()
        p1, p2 = self._patch_image_load()
        with p1, p2:
            worker._process = AsyncMock()
            await worker._backfill_skipped()
        worker._process.assert_not_awaited()

    async def test_successful_backfill_clears_cooldown(self):
        """A real summary after backfill clears the cooldown for that row."""
        row = (450, __file__, "t", "app", "2026-08-18 07:00:00")
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
        row = (450, __file__, "t", "app", "2026-08-18 07:00:00")
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

    def _stub_stages(self, worker, outcomes=None, prepare=None):
        """Replace both pipeline stages so no disk, OCR or model is touched.

        Returns the analyze-stage mock; each prepared row carries the id from
        its source tuple so tests can assert on ordering.
        """
        async def _prepare(row, conn):
            return "ready", CaptureResult(
                filepath=Path(row[1]), timestamp=datetime.now(),
                window_title=row[2], app_name=row[3],
                activity_id=row[0], is_backfill=True,
            )

        worker._prepare_backfill_row = AsyncMock(side_effect=prepare or _prepare)
        worker._analyze_prepared_row = AsyncMock(
            side_effect=outcomes if outcomes is not None else None,
            return_value="done" if outcomes is None else None,
        )
        return worker._analyze_prepared_row

    async def test_batch_processes_rows_and_counts(self):
        """Each row's result lands in the right status bucket."""
        rows = [(1, "a.jpg", "t1", "app1", None),
                (2, "b.jpg", "t2", "app2", None),
                (3, "c.jpg", "t3", "app3", None)]
        worker = self._worker(rows)
        self._stub_stages(worker, outcomes=["done", "failed", "skipped"])
        worker._backfill_status = {"running": True, "requested": 3,
                                   "analyzed": 0, "failed": 0, "skipped": 0}
        await worker._run_backfill_batch(rows, MagicMock())
        status = worker.backfill_status
        assert status["running"] is False
        assert (status["analyzed"], status["failed"], status["skipped"]) == (1, 1, 1)

    async def test_settled_rows_never_reach_the_model(self):
        """A deleted/corrupt screenshot is counted by prepare, not analyzed."""
        rows = [(1, "gone.jpg", "t", "app", None)]
        worker = self._worker(rows)

        async def _prepare(row, conn):
            return "skipped", None

        analyze = self._stub_stages(worker, prepare=_prepare)
        worker._backfill_status = {"running": True, "requested": 1,
                                   "analyzed": 0, "failed": 0, "skipped": 0,
                                   "state": "running"}
        await worker._run_backfill_batch(rows, MagicMock())
        analyze.assert_not_awaited()
        assert worker.backfill_status["skipped"] == 1

    async def test_prepare_runs_ahead_of_analyze(self):
        """The next row is prepared while the current one is in the model."""
        rows = [(1, "a.jpg", "t", "app", None), (2, "b.jpg", "t", "app", None)]
        worker = self._worker(rows)
        worker._backfill_status = {"running": True, "requested": 2,
                                   "analyzed": 0, "failed": 0, "skipped": 0,
                                   "state": "running"}
        prepared = []
        release = asyncio.Event()

        async def _prepare(row, conn):
            prepared.append(row[0])
            return "ready", CaptureResult(
                filepath=Path(row[1]), timestamp=datetime.now(),
                activity_id=row[0], is_backfill=True,
            )

        async def _analyze(capture, conn):
            if capture.activity_id == 1:
                await release.wait()  # row 1 sits in the model
            return "done"

        worker._prepare_backfill_row = AsyncMock(side_effect=_prepare)
        worker._analyze_prepared_row = AsyncMock(side_effect=_analyze)
        task = asyncio.create_task(worker._run_backfill_batch(rows, MagicMock()))
        for _ in range(10):
            await asyncio.sleep(0)

        # Row 2's CPU stage finished without waiting for row 1's model call
        assert prepared == [1, 2]
        release.set()
        await asyncio.wait_for(task, timeout=5)
        assert worker.backfill_status["analyzed"] == 2

    async def test_concurrency_keeps_several_rows_in_the_model(self):
        """backfill_concurrency > 1 analyzes that many rows at once."""
        from screenmind.config import settings

        rows = [(i, f"{i}.jpg", "t", "app", None) for i in range(1, 4)]
        worker = self._worker(rows)
        worker._backfill_status = {"running": True, "requested": 3,
                                   "analyzed": 0, "failed": 0, "skipped": 0,
                                   "state": "running"}
        self._stub_stages(worker)
        in_flight = 0
        peak = 0
        release = asyncio.Event()

        async def _analyze(capture, conn):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await release.wait()
            in_flight -= 1
            return "done"

        worker._analyze_prepared_row = AsyncMock(side_effect=_analyze)
        with patch.object(settings, "backfill_concurrency", 3):
            task = asyncio.create_task(worker._run_backfill_batch(rows, MagicMock()))
            for _ in range(20):
                await asyncio.sleep(0)
            assert peak == 3
            release.set()
            await asyncio.wait_for(task, timeout=5)
        assert worker.backfill_status["analyzed"] == 3
        # Per-item status is suppressed only for the duration of the batch
        assert worker._parallel_backfill is False

    async def test_batch_waits_for_fresh_capture_then_resumes(self):
        """A queued live capture parks the batch; it resumes once drained."""
        rows = [(1, "a.jpg", "t", "app", None)]
        worker = self._worker(rows)
        analyze = self._stub_stages(worker, outcomes=["done"])
        worker._backfill_status = {"running": True, "requested": 1,
                                   "analyzed": 0, "failed": 0, "skipped": 0,
                                   "state": "running"}
        await worker._queue.put(MagicMock())  # fresh capture arrived
        task = asyncio.create_task(worker._run_backfill_batch(rows, MagicMock()))
        for _ in range(5):
            await asyncio.sleep(0)
        analyze.assert_not_awaited()
        assert worker.backfill_status["state"] == "waiting"

        worker._queue.get_nowait()  # main loop drained the live capture
        await asyncio.wait_for(task, timeout=5)
        analyze.assert_awaited_once()
        assert worker.backfill_status["analyzed"] == 1
        assert worker.backfill_status["state"] == "finished"

    async def test_batch_stops_when_cancelled(self):
        """stop_backfill_batch ends the run without analyzing more rows."""
        rows = [(1, "a.jpg", "t", "app", None), (2, "b.jpg", "t", "app", None)]
        worker = self._worker(rows)
        worker._backfill_status = {"running": True, "requested": 2,
                                   "analyzed": 0, "failed": 0, "skipped": 0,
                                   "state": "running"}

        async def _row(*_args):
            worker.stop_backfill_batch()  # user hit Stop mid-row
            return "done"

        analyze = self._stub_stages(worker)
        worker._analyze_prepared_row = AsyncMock(side_effect=_row)
        await asyncio.wait_for(
            worker._run_backfill_batch(rows, MagicMock()), timeout=5
        )
        assert worker._analyze_prepared_row.await_count == 1
        assert worker.backfill_status["state"] == "cancelled"
        assert worker.backfill_status["running"] is False

    def test_stop_without_running_batch_is_noop(self):
        """Stop on an idle worker reports nothing was stopped."""
        worker = self._worker([])
        assert worker.stop_backfill_batch()["stopped"] is False

    async def test_crashing_row_does_not_stall_the_batch(self):
        """A worker that blows up must not strand the prefetcher."""
        rows = [(i, f"{i}.jpg", "t", "app", None) for i in range(1, 4)]
        worker = self._worker(rows)
        worker._backfill_status = {"running": True, "requested": 3,
                                   "analyzed": 0, "failed": 0, "skipped": 0,
                                   "state": "running"}
        self._stub_stages(worker)

        async def _analyze(capture, conn):
            if capture.activity_id == 1:
                raise RuntimeError("model exploded")
            return "done"

        worker._analyze_prepared_row = AsyncMock(side_effect=_analyze)
        await asyncio.wait_for(
            worker._run_backfill_batch(rows, MagicMock()), timeout=5
        )
        status = worker.backfill_status
        assert (status["analyzed"], status["failed"]) == (2, 1)
        assert status["state"] == "finished"

    async def test_batch_publishes_progress(self):
        """Each finished row broadcasts a backfill event to SSE subscribers."""
        rows = [(1, "a.jpg", "t", "app", None)]
        worker = self._worker(rows)
        worker._loop = asyncio.get_event_loop()
        q = worker.subscribe()
        self._stub_stages(worker, outcomes=["done"])
        worker._backfill_status = {"running": True, "requested": 1,
                                   "analyzed": 0, "failed": 0, "skipped": 0,
                                   "state": "running"}
        await worker._run_backfill_batch(rows, MagicMock())
        await asyncio.sleep(0)  # call_soon_threadsafe delivery
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        assert [e["type"] for e in events] == ["backfill", "backfill"]
        assert events[-1]["state"] == "finished"
        assert events[-1]["analyzed"] == 1

    def test_start_gates_when_already_running(self):
        """A second start while a batch runs returns already_running."""
        worker = self._worker([(1, "a.jpg", "t", "app", None)])
        worker._backfill_status["running"] = True
        result = worker.start_backfill_batch(limit=10)
        assert result["already_running"] is True

    def test_start_empty_backlog(self):
        """No pending rows → zeros, no batch started."""
        worker = self._worker([])
        result = worker.start_backfill_batch(limit=10)
        assert result == {"requested": 0, "analyzed": 0, "failed": 0,
                          "skipped": 0, "state": "idle"}
        assert worker.backfill_running() is False

    async def test_start_spawns_batch_and_completes(self):
        """start_backfill_batch returns immediately; the task drains the rows."""
        rows = [(1, "a.jpg", "t", "app", None)]
        worker = self._worker(rows)
        analyze = self._stub_stages(worker, outcomes=["done"])
        result = worker.start_backfill_batch(limit=10)
        assert result["running"] is True
        assert result["requested"] == 1
        for _ in range(20):
            await asyncio.sleep(0)
            if not worker.backfill_running():
                break
        assert worker.backfill_status["analyzed"] == 1
        analyze.assert_awaited_once()

    async def test_idle_loop_stands_down_during_batch(self):
        """The 2s idle backfill doesn't race the manual batch."""
        worker = self._worker([(1, "a.jpg", "t", "app", None)])
        worker._backfill_status["running"] = True
        worker._backfill_one = AsyncMock()
        await worker._backfill_skipped()
        worker._backfill_one.assert_not_awaited()


class TestQualityGate:
    """The one-shot re-analysis gate — what counts as worth a second call."""

    def _record(self, summary="Reviewing a pull request", category="coding"):
        from screenmind.storage.models import ActivityRecord
        return ActivityRecord(
            app_name="app", activity_category=category,
            activity_summary=summary, confidence=0.9,
        )

    def test_other_category_is_a_real_answer(self):
        """'other' must not trigger a retry — _normalize coerces into it."""
        from screenmind.workers.analysis_worker import _missing_quality_fields
        assert _missing_quality_fields(self._record(category="other")) == []

    def test_empty_summary_is_worth_a_retry(self):
        from screenmind.workers.analysis_worker import _missing_quality_fields
        assert _missing_quality_fields(self._record(summary="")) == ["summary"]

    def test_complete_record_needs_nothing(self):
        from screenmind.workers.analysis_worker import _missing_quality_fields
        assert _missing_quality_fields(self._record()) == []

    def _worker_for_process(self):
        from screenmind.workers.analysis_worker import AnalysisWorker
        w = AnalysisWorker(queue=asyncio.Queue(), database=MagicMock())
        w._ocr = MagicMock()
        w._ocr.is_available = False
        w._analyzer = MagicMock()
        w._analyzer.generate_scene_from_text.return_value = None
        w._embedder = None
        w._embedder_available = False
        w._dev_context = MagicMock()
        w._dev_context.is_coding_activity.return_value = False
        return w

    def _capture(self):
        return CaptureResult(
            filepath=Path(__file__), timestamp=datetime.now(),
            window_title="t", app_name="app", activity_id=11,
            image=MagicMock(), is_backfill=True,
        )

    async def _run(self, record):
        from screenmind.config import settings
        worker = self._worker_for_process()
        worker._analyzer.analyze_screenshot_fast.return_value = (record, [])
        with patch.object(settings, "analysis_mode", "fast"), \
             patch.object(settings, "auto_bookmark", False):
            await worker._process(self._capture())
        return worker._analyzer.analyze_screenshot_fast.call_count

    async def test_other_category_costs_one_vision_call(self):
        """The regression this gate caused: 'other' doubled ~half of all rows."""
        assert await self._run(self._record(category="other")) == 1

    async def test_empty_summary_costs_two_vision_calls(self):
        """A genuinely empty analysis is still retried once."""
        assert await self._run(self._record(summary="")) == 2


class TestBackfillPrefetch:
    """The CPU stage that runs ahead of the model — decode, pHash, OCR."""

    def _worker(self):
        from screenmind.workers.analysis_worker import AnalysisWorker
        return AnalysisWorker(queue=asyncio.Queue(), database=MagicMock())

    def _cached(self, worker, phash, summary="Reviewing a pull request"):
        from screenmind.storage.models import ActivityRecord
        worker._app_cache[("app", "title")] = {
            "phash": phash,
            "analysis": ActivityRecord(
                app_name="app", activity_category="coding",
                activity_summary=summary, confidence=0.9,
            ),
        }

    def _capture(self, phash, bookmarked=False):
        return CaptureResult(
            filepath=Path("a.jpg"), timestamp=datetime.now(),
            window_title="title", app_name="app",
            bookmarked=bookmarked, phash=phash, is_backfill=True,
        )

    def test_identical_screen_skips_ocr_prefetch(self):
        """Cache hits reuse everything, so prefetching OCR is wasted work."""
        import imagehash
        worker = self._worker()
        phash = imagehash.hex_to_hash("f0f0f0f0f0f0f0f0")
        self._cached(worker, phash)
        assert worker._would_hit_identical_cache(self._capture(phash)) is True

    def test_changed_screen_prefetches_ocr(self):
        """A different screen runs the full pipeline — OCR is worth prefetching."""
        import imagehash
        worker = self._worker()
        self._cached(worker, imagehash.hex_to_hash("0000000000000000"))
        assert worker._would_hit_identical_cache(
            self._capture(imagehash.hex_to_hash("ffffffffffffffff"))
        ) is False

    def test_cached_failure_never_counts_as_identical(self):
        """A cached failure placeholder must not suppress real work."""
        import imagehash
        worker = self._worker()
        phash = imagehash.hex_to_hash("f0f0f0f0f0f0f0f0")
        self._cached(worker, phash, summary="Analysis failed: boom")
        assert worker._would_hit_identical_cache(self._capture(phash)) is False

    def test_bookmarked_row_always_prefetches(self):
        """Bookmarks bypass the cache in _process, so they need OCR."""
        import imagehash
        worker = self._worker()
        phash = imagehash.hex_to_hash("f0f0f0f0f0f0f0f0")
        self._cached(worker, phash)
        assert worker._would_hit_identical_cache(
            self._capture(phash, bookmarked=True)
        ) is False

    def test_no_phash_prefetches(self):
        """Without a pHash there is nothing to compare — run OCR."""
        worker = self._worker()
        assert worker._would_hit_identical_cache(self._capture(None)) is False

    async def test_process_uses_prefetched_ocr(self):
        """_process consumes the prefetched result instead of re-running OCR."""
        worker = self._worker()
        worker._ocr = MagicMock()
        worker._ocr.is_available = True
        worker._ocr.extract_text_with_boxes = MagicMock()
        worker._db.get_activity_by_id = MagicMock(return_value=None)

        capture = CaptureResult(
            filepath=Path(__file__), timestamp=datetime.now(),
            window_title="t", app_name="app", activity_id=7,
            image=MagicMock(), is_backfill=True,
            prefetched_ocr=("prefetched screen text", [{"text": "hi"}]),
        )
        worker._analyzer = MagicMock()
        worker._analyzer.analyze_screenshot_fast.side_effect = RuntimeError("stop here")
        await worker._process(capture)
        # The executor OCR path was never taken
        worker._ocr.extract_text_with_boxes.assert_not_called()

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
