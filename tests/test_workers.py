"""Comprehensive tests for capture and analysis workers."""
import time

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from screenmind.workers.capture_worker import CaptureWorker, CaptureResult


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
        row = (450, __file__, "Jump List", "ShellExperienceHost", None, None)
        worker = self._worker([row], "Analysis failed")
        p1, p2 = self._patch_image_load()
        with p1, p2:
            worker._process = AsyncMock()
            await worker._backfill_skipped()
        assert 450 in worker._backfill_cooldown
        worker._process.assert_awaited_once()

    async def test_row_in_cooldown_is_skipped(self):
        """A cooling-down row is skipped; the next candidate is processed."""
        row1 = (450, __file__, "t1", "app1", None, None)
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
        row = (450, __file__, "t", "app", None, None)
        worker = self._worker([row], "Analysis failed")
        worker._backfill_cooldown[450] = time.time()
        p1, p2 = self._patch_image_load()
        with p1, p2:
            worker._process = AsyncMock()
            await worker._backfill_skipped()
        worker._process.assert_not_awaited()

    async def test_successful_backfill_clears_cooldown(self):
        """A real summary after backfill clears the cooldown for that row."""
        row = (450, __file__, "t", "app", None, None)
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
        row = (450, __file__, "t", "app", None, None)
        worker = self._worker([row], "Analysis failed")
        p1, p2 = self._patch_image_load()
        with p1, p2:
            worker._process = AsyncMock(side_effect=RuntimeError("boom"))
            await worker._backfill_skipped()
        assert 450 in worker._backfill_cooldown


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
