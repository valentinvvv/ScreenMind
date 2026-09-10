"""Tests for config.py — settings parsing and properties."""

from screenmind.config import Settings


def test_default_settings():
    s = Settings(data_dir="/tmp/screenmind_test")
    assert s.capture_interval == 40
    assert s.screenshot_quality == 70
    assert s.ollama_model == "gemma4:e2b"
    assert s.api_port == 7777


def test_blocked_apps_list_empty():
    s = Settings(data_dir="/tmp/test", blocked_apps="")
    assert s.blocked_apps_list == []


def test_blocked_apps_list_parsing():
    s = Settings(data_dir="/tmp/test", blocked_apps="1password, banking, keychain")
    assert s.blocked_apps_list == ["1password", "banking", "keychain"]


def test_workspace_dirs_list():
    s = Settings(data_dir="/tmp/test", workspace_dirs="~/Projects, ~/Code")
    dirs = s.workspace_dirs_list
    assert len(dirs) == 2


def test_heavy_apps_list():
    s = Settings(data_dir="/tmp/test", heavy_apps="game,valorant,blender")
    assert "game" in s.heavy_apps_list
    assert "valorant" in s.heavy_apps_list
    assert len(s.heavy_apps_list) == 3


def test_meeting_apps_list():
    s = Settings(data_dir="/tmp/test", meeting_apps="zoom,teams,meet")
    assert "zoom" in s.meeting_apps_list
    assert len(s.meeting_apps_list) == 3


def test_num_gpu_layers():
    s = Settings(data_dir="/tmp/test", performance_mode="minimal")
    assert s.num_gpu_layers == 0

    s = Settings(data_dir="/tmp/test", performance_mode="balanced")
    assert s.num_gpu_layers == 15

    s = Settings(data_dir="/tmp/test", performance_mode="maximum")
    assert s.num_gpu_layers == 99


def test_data_path_resolution():
    s = Settings(data_dir="~/.screenmind")
    assert s.data_path.is_absolute()
    assert "~" not in str(s.data_path)


class TestRuntimeOverrideMerge:
    """save_runtime_overrides must never touch keys it wasn't given.

    The dashboard omits retention_days when its radio group has no selection.
    Rewriting it there once turned "Forever" into 7 days, and the startup
    cleanup then permanently deleted everything older than that.
    """

    def _settings_with_json(self, tmp_path, initial):
        import json
        from screenmind.config import Settings
        s = Settings()
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(initial))
        type(s).settings_json_path = property(lambda self, _p=path: _p)
        return s, path

    def test_omitted_key_is_preserved(self, tmp_path):
        import json
        s, path = self._settings_with_json(tmp_path, {"retention_days": 0})
        s.save_runtime_overrides({"capture_interval": 45})
        saved = json.loads(path.read_text())
        assert saved["retention_days"] == 0
        assert saved["capture_interval"] == 45

    def test_present_key_is_written(self, tmp_path):
        import json
        s, path = self._settings_with_json(tmp_path, {"retention_days": 0})
        s.save_runtime_overrides({"retention_days": 30})
        assert json.loads(path.read_text())["retention_days"] == 30


class TestSuiteIsolation:
    """The suite must not inherit the developer's ~/.screenmind."""

    def test_settings_come_from_defaults_not_the_users_file(self):
        """CI has no settings.json; a local run must behave identically.

        A locally-passing concurrency test once failed in CI for exactly this
        reason — the developer's file set backfill_concurrency=4. conftest.py
        redirects DATA_DIR before screenmind is imported; this asserts it stuck.
        """
        from screenmind.config import Settings, settings
        defaults = Settings.model_fields
        for key in ("backfill_concurrency", "analysis_mode", "gemma_mode",
                    "capture_interval", "retention_days"):
            assert getattr(settings, key) == defaults[key].default, (
                f"{key} is {getattr(settings, key)!r}, not the default "
                f"{defaults[key].default!r} — the developer's settings.json leaked "
                f"into the test run, so this suite and CI disagree."
            )

    def test_logging_does_not_write_to_the_real_data_dir(self):
        """Fixture noise must not land in the log used to debug real runs."""
        import logging
        import os
        real = os.path.join(os.path.expanduser("~"), ".screenmind")
        for h in logging.getLogger("screenmind").handlers:
            path = getattr(h, "baseFilename", None)
            if path:
                assert not os.path.abspath(path).startswith(os.path.abspath(real)), (
                    f"tests are appending to the production log at {path}"
                )
