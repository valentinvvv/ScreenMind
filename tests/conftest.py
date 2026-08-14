"""Shared pytest fixtures for ScreenMind tests."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest



from screenmind.config import settings


@pytest.fixture(autouse=True)
def _pin_local_backend():
    """Tests assume the default local backend with no text-model routing.
    The developer's real ~/.screenmind/settings.json may set gemma_mode=custom
    or configure a text model — don't let it leak into tests."""
    original = {
        k: getattr(settings, k)
        for k in (
            "gemma_mode", "text_llm_routing", "text_llm_model_name",
            "text_llm_api_base_url", "text_llm_api_key", "text_llm_context_window",
        )
    }
    settings.gemma_mode = "local"
    settings.text_llm_routing = "off"
    settings.text_llm_model_name = None
    settings.text_llm_api_base_url = None
    settings.text_llm_api_key = None
    settings.text_llm_context_window = 32768
    yield
    for k, v in original.items():
        setattr(settings, k, v)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provide a temporary data directory for tests."""
    return tmp_path


@pytest.fixture
def mock_settings(tmp_path):
    """Patch settings to use temp directories."""
    with patch("screenmind.config.settings") as mock:
        mock.data_path = tmp_path
        mock.screenshots_dir = tmp_path / "screenshots"
        mock.db_path = tmp_path / "test.db"
        mock.screenshots_dir.mkdir(parents=True, exist_ok=True)
        mock.capture_interval = 30
        mock.screenshot_quality = 70
        mock.ollama_model = "gemma4:e2b"
        mock.ollama_host = "http://localhost:11434"
        mock.blocked_apps_list = []
        mock.heavy_apps_list = []
        mock.auto_pause_heavy_apps = False
        mock.bookmark_hotkey = "ctrl+shift+b"
        mock.pause_hotkey = "ctrl+shift+p"
        mock.sensitive_filter_enabled = False
        mock.retention_days = 7
        mock.agents_enabled = False
        mock.webhook_enabled = False
        mock.webhook_url = ""
        mock.webhook_secret = ""
        mock.webhook_events = ""
        mock.webhook_headers = ""
        mock.dashboard_pin_hash = ""
        mock.dashboard_lock_timeout = 30
        mock.encryption_enabled = False
        mock.api_host = "127.0.0.1"
        mock.api_port = 7777
        yield mock


@pytest.fixture
def db(tmp_path):
    """Create a fresh test database."""
    from screenmind.storage.database import Database
    test_db = Database(db_path=tmp_path / "test.db")
    yield test_db
    test_db.close()


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    from PIL import Image
    img = Image.new("RGB", (1920, 1080), color=(50, 50, 80))
    return img


@pytest.fixture
def app(db):
    """Create a test FastAPI app with PIN auth disabled."""
    import screenmind.api.dependencies as deps
    from screenmind.api.server import create_app

    # Directly clear PIN so auth middleware passes all requests during tests
    original_pin = settings.dashboard_pin_hash
    settings.dashboard_pin_hash = ''

    application = create_app(database=db)

    # Also patch the module-level db references in route modules
    # since they may have been imported with a previous db value
    import screenmind.api.routes.timeline as _tl
    import screenmind.api.routes.bookmarks as _bm
    import screenmind.api.routes.stats as _st
    import screenmind.api.routes.data as _dt
    import screenmind.api.routes.screenshots as _ss
    import screenmind.api.routes.rewind as _rw
    import screenmind.api.routes.summary as _sm
    import screenmind.api.routes.meetings as _mt
    import screenmind.api.routes.agents as _ag
    import screenmind.api.routes.search as _sr
    import screenmind.api.routes.chat as _ch

    for mod in [_tl, _bm, _st, _dt, _ss, _rw, _sm, _mt, _ag, _sr, _ch]:
        mod.db = db

    yield application

    # Restore
    settings.dashboard_pin_hash = original_pin


@pytest.fixture
async def client(app):
    """Create a test HTTP client."""
    from httpx import AsyncClient, ASGITransport
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
