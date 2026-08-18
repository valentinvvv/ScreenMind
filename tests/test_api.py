"""Test API endpoints using FastAPI TestClient."""
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def client():
    """Create a test client with mocked dependencies."""
    from screenmind.api.server import create_app
    from screenmind.storage.database import Database
    from fastapi.testclient import TestClient

    db = MagicMock(spec=Database)
    db.get_activities_by_date.return_value = []
    db.count_activities_by_date.return_value = 0
    db.get_bookmarks.return_value = []
    db.get_stats.return_value = {"total_activities": 0, "category_breakdown": {}, "top_apps": {}, "top_repos": {}, "meetings_count": 0, "meetings_minutes": 0}
    db.get_daily_summary.return_value = None
    db.get_rewind_data.return_value = []

    app = create_app(database=db, capture_worker=MagicMock(), analysis_worker=MagicMock(), audio_worker=MagicMock())

    # Bypass PIN lock middleware — tests shouldn't need auth
    with patch("screenmind.api.server.settings") as mock_settings:
        mock_settings.dashboard_pin_hash = None
        yield TestClient(app)


def test_root_returns_html(client):
    """Dashboard HTML loads."""
    r = client.get("/")
    assert r.status_code == 200
    assert "ScreenMind" in r.text


def test_status_endpoint(client):
    """Status endpoint returns valid JSON."""
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data


def test_auth_status(client):
    """Auth status returns first_run and has_pin fields."""
    r = client.get("/api/auth/status")
    assert r.status_code == 200
    data = r.json()
    assert "has_pin" in data
    assert "authenticated" in data


def test_timeline_returns_list(client):
    """Timeline endpoint returns activities list."""
    r = client.get("/api/timeline?date=2026-01-01")
    assert r.status_code == 200
    data = r.json()
    assert "activities" in data


def test_search_requires_query(client):
    """Search endpoint requires q parameter."""
    r = client.get("/api/search")
    assert r.status_code == 422  # validation error
