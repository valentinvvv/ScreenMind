"""Tests for API endpoints — uses httpx AsyncClient with the FastAPI app."""

import pytest
from datetime import datetime
from unittest.mock import patch

from screenmind.storage.models import ScreenshotEntry, ActivityRecord
from screenmind.config import settings
import screenmind.api.dependencies as deps


@pytest.mark.asyncio
async def test_status_endpoint(client):
    resp = await client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"


@pytest.mark.asyncio
async def test_timeline_empty(client):
    resp = await client.get("/api/timeline?date=2099-01-01")
    assert resp.status_code == 200
    data = resp.json()
    assert data["activities"] == []


@pytest.mark.asyncio
async def test_timeline_with_data(client, db):
    entry = ScreenshotEntry(
        timestamp=datetime(2026, 5, 16, 10, 0, 0),
        screenshot_path="/tmp/test.jpg",
        window_title="Test Window",
        analyzed=False,
    )
    db.insert_activity(entry)

    resp = await client.get("/api/timeline?date=2026-05-16")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["activities"]) >= 1
    assert any(a["window_title"] == "Test Window" for a in data["activities"])


@pytest.mark.asyncio
async def test_bookmarks_empty(client):
    resp = await client.get("/api/bookmarks")
    assert resp.status_code == 200
    assert resp.json()["bookmarks"] == [] or isinstance(resp.json()["bookmarks"], list)


@pytest.mark.asyncio
async def test_toggle_bookmark(client, db):
    entry = ScreenshotEntry(
        timestamp=datetime(2026, 5, 16, 11, 0, 0),
        screenshot_path="/tmp/bm.jpg",
        bookmarked=False,
        analyzed=False,
    )
    aid = db.insert_activity(entry)

    resp = await client.put(f"/api/activities/{aid}/bookmark")
    assert resp.status_code == 200
    data = resp.json()
    assert data["bookmarked"] is True


@pytest.mark.asyncio
async def test_stats_endpoint(client, db):
    entry = ScreenshotEntry(
        timestamp=datetime(2026, 5, 16, 9, 0, 0),
        screenshot_path="/tmp/s.jpg",
        analyzed=False,
    )
    aid = db.insert_activity(entry)
    db.update_activity_analysis(aid, ActivityRecord(
        app_name="Code", activity_category="coding", activity_summary="test"
    ))

    resp = await client.get("/api/stats?date_from=2026-05-16&date_to=2026-05-16")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_activities" in data


@pytest.mark.asyncio
async def test_settings_get(client):
    resp = await client.get("/api/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert "capture_interval" in data
    assert "performance_mode" in data


@pytest.mark.asyncio
async def test_llm_test_endpoint_ok(client):
    """/api/llm/test returns discovered models from a reachable endpoint."""
    with patch("screenmind.engine.llm_client.list_remote_models", return_value=["model-a", "model-b"]):
        resp = await client.post("/api/llm/test", json={"base_url": "http://api.test/v1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["models"] == ["model-a", "model-b"]


@pytest.mark.asyncio
async def test_llm_test_endpoint_failure(client):
    """/api/llm/test reports errors without raising."""
    with patch("screenmind.engine.llm_client.list_remote_models", side_effect=Exception("refused")):
        resp = await client.post("/api/llm/test", json={"base_url": "http://down.test/v1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert "refused" in data["error"]

@pytest.mark.asyncio
async def test_capture_pause_resume(client):
    resp = await client.post("/api/capture/pause")
    assert resp.status_code == 200
    assert resp.json()["paused"] is True

    resp = await client.post("/api/capture/resume")
    assert resp.status_code == 200
    assert resp.json()["paused"] is False


@pytest.mark.asyncio
async def test_rewind_empty(client):
    resp = await client.get("/api/rewind?date=2099-01-01")
    assert resp.status_code == 200
    assert resp.json()["frames"] == []


@pytest.mark.asyncio
async def test_summary_not_generated(client):
    resp = await client.get("/api/summary?date=2099-01-01")
    assert resp.status_code == 200
    data = resp.json()
    assert data["generated"] is False


@pytest.mark.asyncio
async def test_auth_status_no_pin(client):
    resp = await client.get("/api/auth/status")
    assert resp.status_code == 200
    data = resp.json()
    # PIN is cleared in test fixture
    assert data["has_pin"] is False
    assert data["authenticated"] is True


@pytest.mark.asyncio
async def test_activity_not_found(client):
    resp = await client.get("/api/activity/99999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_clear_timeline(client, db):
    entry = ScreenshotEntry(
        timestamp=datetime(2026, 5, 20, 10, 0, 0),
        screenshot_path="/tmp/clear.jpg",
        analyzed=False,
    )
    db.insert_activity(entry)

    resp = await client.delete("/api/timeline/clear?date=2026-05-20")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"] >= 1


@pytest.mark.asyncio
async def test_meetings_empty(client):
    resp = await client.get("/api/meetings?date=2099-01-01")
    assert resp.status_code == 200
    assert resp.json()["meetings"] == []


@pytest.mark.asyncio
async def test_models_list(client):
    resp = await client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) >= 1


def _seed_busy_day(db, n=200):
    """Insert n analyzed activities with large organized_text for 2026-05-16."""
    from datetime import timedelta
    base = datetime(2026, 5, 16, 8, 0, 0)
    for i in range(n):
        entry = ScreenshotEntry(
            timestamp=base + timedelta(seconds=40 * i),
            screenshot_path=f"/tmp/{i}.jpg",
            window_title=f"Window {i}",
            analyzed=True,
        )
        aid = db.insert_activity(entry)
        db.update_activity_analysis(
            aid,
            ActivityRecord(
                app_name="Code",
                activity_category="coding",
                activity_summary=f"Editing file_{i}.py in repo src/auth/jwt.ts const token = req.headers",
            ),
            organized_text=f"src/auth/jwt.ts | const token = req.headers.authorization?.split(' ')[1] | line {i} " * 4,
        )


class TestSummaryPromptBudget:
    """Regression: busy-day summary prompts overflowed the context window and
    the backend rejected them with HTTP 400 (exceed_context_size_error)."""

    def test_block_fits_context_window(self, db):
        """200 rich activities are trimmed to the configured context budget."""
        from screenmind.api.routes.summary import _build_activity_block, _budget_chars
        _seed_busy_day(db, n=200)
        activities = db.get_activities_by_date("2026-05-16", limit=200)
        assert len(activities) == 200

        budget = _budget_chars(settings.context_window, 2048)
        block, count = _build_activity_block(
            activities, max_rich=20, rich_chars=300, budget_chars=budget
        )
        assert len(block) <= budget
        assert 0 < count < 200  # trimmed, not emptied
        # Oldest dropped first: newest entry survives, oldest doesn't
        assert "file_199.py" in block
        assert "file_0.py" not in block

    def test_small_day_keeps_everything(self, db):
        """A short day fits whole — no trimming."""
        from screenmind.api.routes.summary import _build_activity_block, _budget_chars
        _seed_busy_day(db, n=5)
        activities = db.get_activities_by_date("2026-05-16", limit=200)
        block, count = _build_activity_block(
            activities, max_rich=20, rich_chars=300,
            budget_chars=_budget_chars(settings.context_window, 2048),
        )
        assert count == 5
        for i in range(5):
            assert f"file_{i}.py" in block

    def test_rich_slots_go_to_newest(self, db):
        """Limited Screen-content slots attach to the most recent activity."""
        from screenmind.api.routes.summary import _build_activity_block
        _seed_busy_day(db, n=30)
        activities = db.get_activities_by_date("2026-05-16", limit=200)
        block, _ = _build_activity_block(
            activities, max_rich=5, rich_chars=300, budget_chars=10**9
        )
        # newest entry has Screen content, oldest does not
        newest_chunk = block.split("[2026-05-16T")[-1]
        assert "Screen content" in newest_chunk
        oldest_chunk = block.split("\n[")[0]
        assert "Screen content" not in oldest_chunk

    @pytest.mark.asyncio
    async def test_generate_summary_prompt_within_budget(self, client, db):
        """POST /api/summary/generate sends a prompt that fits the window."""
        _seed_busy_day(db, n=200)
        captured = {}

        def fake_generate(prompt, temperature=0.3, max_tokens=1024, timeout=None):
            captured["prompt"] = prompt
            captured["max_tokens"] = max_tokens
            return "A productive day of coding."

        with patch("screenmind.engine.llm_client.generate", side_effect=fake_generate):
            resp = await client.post("/api/summary/generate?date=2026-05-16")

        assert resp.status_code == 200
        assert resp.json()["summary"]["summary"] == "A productive day of coding."
        # At the measured worst-case density (2 chars/token) the whole prompt
        # plus the requested output must fit the context window.
        prompt_tokens = len(captured["prompt"]) / 2
        assert prompt_tokens + captured["max_tokens"] <= settings.context_window


class TestTextModelWindowPick:
    """Summary budget follows the text-model routing switch."""

    def _pick(self, full_chars, out_tokens, routing, text_window=32768):
        from screenmind.api.routes import summary as sm
        with patch.object(sm, "text_model_window", return_value=text_window), \
             patch.object(sm.settings, "text_llm_routing", routing):
            return sm._pick_window(full_chars, out_tokens)

    def test_no_text_model_uses_primary_window(self):
        from screenmind.api.routes import summary as sm
        with patch.object(sm, "text_model_window", return_value=None):
            assert sm._pick_window(100000, 2048) == settings.context_window

    def test_always_uses_text_window(self):
        assert self._pick(10, 2048, "always") == 32768

    def test_overflow_large_day_uses_text_window(self):
        # 20k chars ≈ 10k tokens > 6144 primary window
        assert self._pick(20000, 2048, "overflow") == 32768

    def test_overflow_small_day_stays_primary(self):
        # 2k chars ≈ 1k tokens + output fits the primary window
        assert self._pick(2000, 2048, "overflow") == settings.context_window

    def test_off_uses_primary_window(self):
        assert self._pick(100000, 2048, "off") == settings.context_window

    @pytest.mark.asyncio
    async def test_generate_summary_routes_busy_day_to_text_window(self, client, db):
        """Busy day + overflow routing → prompt sized for the text model."""
        _seed_busy_day(db, n=200)
        captured = {}

        def fake_generate(prompt, temperature=0.3, max_tokens=1024, timeout=None):
            captured["prompt"] = prompt
            captured["max_tokens"] = max_tokens
            return "ok"

        from screenmind.api.routes import summary as sm
        with patch("screenmind.engine.llm_client.generate", side_effect=fake_generate), \
             patch.object(sm, "text_model_window", return_value=32768), \
             patch.object(sm.settings, "text_llm_routing", "overflow"):
            resp = await client.post("/api/summary/generate?date=2026-05-16")

        assert resp.status_code == 200
        # Prompt must fit the TEXT window now — and use far more of the day
        # than the primary-window budget allowed.
        prompt_tokens = len(captured["prompt"]) / 2
        assert prompt_tokens + captured["max_tokens"] <= 32768
        assert prompt_tokens > settings.context_window  # overflowed the primary


class TestBackfillEndpoint:
    """POST /api/timeline/backfill — manual batch re-analysis."""

    @pytest.mark.asyncio
    async def test_backfill_without_worker_returns_503(self, client):
        """No analysis worker (test app) → 503."""
        deps.analysis_worker = None
        resp = await client.post("/api/timeline/backfill")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_backfill_starts_batch(self, client):
        """With a worker, the endpoint forwards to start_backfill_batch."""
        from unittest.mock import MagicMock
        worker = MagicMock()
        worker.start_backfill_batch.return_value = {
            "running": True, "requested": 3, "analyzed": 0, "failed": 0, "skipped": 0,
        }
        deps.analysis_worker = worker
        try:
            resp = await client.post("/api/timeline/backfill?limit=50")
            assert resp.status_code == 200
            assert resp.json()["requested"] == 3
            worker.start_backfill_batch.assert_called_once_with(limit=50)
        finally:
            deps.analysis_worker = None

    @pytest.mark.asyncio
    async def test_backfill_limit_validation(self, client):
        """limit outside 1..500 is rejected."""
        resp = await client.post("/api/timeline/backfill?limit=0")
        assert resp.status_code == 422
        resp = await client.post("/api/timeline/backfill?limit=501")
        assert resp.status_code == 422


class TestSceneBackfillEndpoint:
    """POST /api/timeline/scenes/backfill — generate missed scene descriptions."""

    @pytest.mark.asyncio
    async def test_scene_backfill_without_worker_returns_503(self, client):
        """No analysis worker (test app) → 503."""
        deps.analysis_worker = None
        resp = await client.post("/api/timeline/scenes/backfill")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_scene_backfill_starts_batch(self, client):
        """With a worker, the endpoint forwards to start_scene_backfill."""
        from unittest.mock import MagicMock
        worker = MagicMock()
        worker.start_scene_backfill.return_value = {
            "running": True, "requested": 4, "generated": 0, "failed": 0, "skipped": 0,
        }
        deps.analysis_worker = worker
        try:
            resp = await client.post("/api/timeline/scenes/backfill?limit=25")
            assert resp.status_code == 200
            assert resp.json()["requested"] == 4
            worker.start_scene_backfill.assert_called_once_with(limit=25)
        finally:
            deps.analysis_worker = None

    @pytest.mark.asyncio
    async def test_scene_backfill_limit_validation(self, client):
        """limit outside 1..500 is rejected."""
        resp = await client.post("/api/timeline/scenes/backfill?limit=0")
        assert resp.status_code == 422
        resp = await client.post("/api/timeline/scenes/backfill?limit=501")
        assert resp.status_code == 422
