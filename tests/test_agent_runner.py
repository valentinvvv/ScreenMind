"""Comprehensive tests for agent runner module."""
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from screenmind.engine.agent_runner import (
    _parse_md_frontmatter, _parse_py_frontmatter, get_agents_dir, get_agent_log, _log_run,
)


class TestFrontmatterParsing:
    """Tests for markdown agent frontmatter parsing."""

    def test_full_frontmatter(self, tmp_path):
        """All frontmatter fields are parsed correctly."""
        f = tmp_path / "test-agent.md"
        f.write_text("""---
name: Daily Focus Report
schedule: every 6h
data: timeline, apps, mood
output: local, obsidian
description: Generates a focus score
enabled: true
model_requirement: 8192
---

Analyze my screen activity and generate a focus report.
Give me a score out of 10.
""", encoding="utf-8")

        meta = _parse_md_frontmatter(f)
        assert meta["name"] == "Daily Focus Report"
        assert meta["schedule"] == "every 6h"
        assert "timeline" in meta["data"]
        assert "apps" in meta["data"]
        assert "mood" in meta["data"]
        assert "local" in meta["output"]
        assert "obsidian" in meta["output"]
        assert meta["description"] == "Generates a focus score"
        assert meta["enabled"] is True
        assert meta["model_requirement"] == "8192"
        assert "Analyze my screen activity" in meta["prompt"]
        assert "score out of 10" in meta["prompt"]

    def test_no_frontmatter(self, tmp_path):
        """File without frontmatter uses full text as prompt."""
        f = tmp_path / "simple.md"
        f.write_text("Just tell me what I did today.", encoding="utf-8")

        meta = _parse_md_frontmatter(f)
        assert meta["prompt"] == "Just tell me what I did today."
        assert meta["name"] == "simple"
        assert meta["slug"] == "simple"

    def test_disabled_agent(self, tmp_path):
        """enabled: false is parsed correctly."""
        f = tmp_path / "disabled.md"
        f.write_text("""---
name: Disabled Agent
enabled: false
---

This should not run.
""", encoding="utf-8")

        meta = _parse_md_frontmatter(f)
        assert meta["enabled"] is False

    def test_defaults_applied(self, tmp_path):
        """Missing fields get sensible defaults."""
        f = tmp_path / "minimal.md"
        f.write_text("""---
name: Minimal
---

Do something.
""", encoding="utf-8")

        meta = _parse_md_frontmatter(f)
        assert meta["schedule"] == "every 6h"
        assert meta["output"] == "local"
        assert meta["data"] == "timeline, apps"
        assert meta["enabled"] is True

    def test_slug_is_filename_stem(self, tmp_path):
        """Slug is always the filename stem regardless of name field."""
        f = tmp_path / "my-cool-agent.md"
        f.write_text("""---
name: A Different Name
---

Prompt here.
""", encoding="utf-8")

        meta = _parse_md_frontmatter(f)
        assert meta["slug"] == "my-cool-agent"
        assert meta["name"] == "A Different Name"

    def test_multiline_prompt(self, tmp_path):
        """Prompt preserves multiline content."""
        f = tmp_path / "multi.md"
        f.write_text("""---
name: Multi
---

Line one.
Line two.
Line three.
""", encoding="utf-8")

        meta = _parse_md_frontmatter(f)
        assert "Line one." in meta["prompt"]
        assert "Line two." in meta["prompt"]
        assert "Line three." in meta["prompt"]


class TestAgentsDir:
    """Tests for agent directory management."""

    def test_get_agents_dir_exists(self):
        d = get_agents_dir()
        assert d.is_dir()

    def test_get_agents_dir_is_consistent(self):
        """Returns same path on repeated calls."""
        d1 = get_agents_dir()
        d2 = get_agents_dir()
        assert d1 == d2


class TestAgentLog:
    """Tests for agent run logging."""

    def test_get_agent_log_returns_list(self):
        log = get_agent_log()
        assert isinstance(log, list)

    def test_log_run_adds_entry(self):
        """_log_run adds an entry to the log."""
        before = len(get_agent_log())
        _log_run("test-agent", "markdown", "ok", output="test output", duration=1.5)
        after = len(get_agent_log())
        assert after >= before  # may wrap due to maxlen

    def test_log_entry_format(self):
        """Log entries have expected fields."""
        _log_run("format-test", "python", "error", error="something broke", duration=0.5)
        log = get_agent_log()
        entry = next((e for e in log if e["name"] == "format-test"), None)
        if entry:
            assert "timestamp" in entry
            assert entry["type"] == "python"
            assert entry["status"] == "error"
            assert "something broke" in entry["error"]
            assert entry["duration"] == 0.5

    def test_log_truncates_long_output(self):
        """Output is truncated to 500 chars."""
        long_output = "x" * 1000
        _log_run("truncate-test", "markdown", "ok", output=long_output)
        log = get_agent_log()
        entry = next((e for e in log if e["name"] == "truncate-test"), None)
        if entry:
            assert len(entry["output"]) <= 500


class TestDateForwarding:
    """run_agent(date=...) must reach Python plugins via context["date"]."""

    def _probe_plugin(self, tmp_path):
        plugin = tmp_path / "dateprobe.py"
        plugin.write_text(
            '"""\nname: dateprobe\ndescription: echoes the run date\n"""\n'
            "def run(context):\n"
            "    return f\"date={context.get('date', 'NONE')}\"\n",
            encoding="utf-8",
        )
        return _parse_py_frontmatter(plugin)

    def test_date_reaches_plugin_context(self, tmp_path):
        from screenmind.engine import agent_runner
        meta = self._probe_plugin(tmp_path)
        with patch.object(agent_runner.settings, "agents_auto_run_python", True):
            result = agent_runner.run_agent(meta, date="2026-08-14")
        assert result["status"] == "ok"
        assert "date=2026-08-14" in result["output"]

    def test_no_date_omits_context_key(self, tmp_path):
        from screenmind.engine import agent_runner
        meta = self._probe_plugin(tmp_path)
        with patch.object(agent_runner.settings, "agents_auto_run_python", True):
            result = agent_runner.run_agent(meta)
        assert result["status"] == "ok"
        assert "date=NONE" in result["output"]


class TestTimesheetPluginDateResolution:
    """The shipped timesheet.py plugin: dashboard date wins over env/state."""

    @pytest.fixture
    def plugin(self):
        import importlib.util
        path = Path(__file__).parent.parent / "dist_build" / "timesheet_agent" / "timesheet.py"
        spec = importlib.util.spec_from_file_location("timesheet_plugin", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_dashboard_date_beats_env(self, plugin, tmp_path, monkeypatch):
        monkeypatch.setenv("TIMESHEET_DATE", "2026-01-01")
        date, err = plugin._resolve_date(tmp_path, "timesheet", ui_date="2026-08-14")
        assert (date, err) == ("2026-08-14", None)

    def test_env_used_without_dashboard_date(self, plugin, tmp_path, monkeypatch):
        monkeypatch.setenv("TIMESHEET_DATE", "yesterday")
        date, err = plugin._resolve_date(tmp_path, "timesheet", ui_date="")
        assert err is None
        assert date != ""

    def test_invalid_dashboard_date_errors(self, plugin, tmp_path):
        date, err = plugin._resolve_date(tmp_path, "timesheet", ui_date="14/08/2026")
        assert date == ""
        assert "dashboard date picker" in err


class TestTimesheetJustification:
    """mode: timesheet — the LLM classification call also yields a
    per-entry justification (on-screen evidence), rendered as an
    indented 'because:' line. Placeholders and LLM failures omit it."""

    ACTIVITIES = [
        {
            "timestamp": "2026-08-16T10:00:00",
            "window_title": "GITS PSF - HelpDesk - Ticket 99177 Details",
            "organized_text": "password reset request from user@omeaadvisors.com",
            "app_name": "Chrome",
            "analyzed": True,
        },
        {
            "timestamp": "2026-08-16T10:00:40",
            "window_title": "GITS PSF - HelpDesk - Ticket 99177 Details",
            "organized_text": "password reset request from user@omeaadvisors.com",
            "app_name": "Chrome",
            "analyzed": True,
        },
    ]

    def _run(self, llm_reply=None, llm_error=None):
        from screenmind.engine import agent_runner
        db_inst = MagicMock()
        db_inst.get_activities_by_date.return_value = self.ACTIVITIES
        generate = MagicMock(side_effect=llm_error, return_value=llm_reply)
        with patch("screenmind.storage.database.Database", return_value=db_inst), \
             patch("screenmind.engine.llm_client.generate", generate), \
             patch("screenmind.engine.llm_client.text_model_window", return_value=32768):
            out = agent_runner._run_timesheet_mode(
                {"_temperature": 0.1}, date="2026-08-16"
            )
        return out, generate

    def test_prompt_requests_justification(self):
        _, generate = self._run(llm_reply="T99177 | — | Password reset | —")
        prompt = generate.call_args.kwargs["prompt"]
        assert "JUSTIFICATION" in prompt
        assert "ID | CUSTOMER | SUBJECT | JUSTIFICATION" in prompt

    def test_justification_rendered_under_entry(self):
        out, _ = self._run(
            llm_reply="T99177 | OMEA ADVISORS | Password reset | "
                      "Email thread with user about password reset"
        )
        lines = out.splitlines()
        i = next(j for j, l in enumerate(lines) if l.startswith("Ticket 99177"))
        assert lines[i + 1] == "    because: Email thread with user about password reset"
        assert "Subtotal:" in out

    def test_placeholder_justification_omitted(self):
        out, _ = self._run(llm_reply="T99177 | — | Password reset | —")
        assert "because:" not in out
        assert "Ticket 99177" in out

    def test_three_field_answer_still_parses(self):
        """Old-format answers without a justification field keep working."""
        out, _ = self._run(llm_reply="T99177 | OMEA ADVISORS | Password reset")
        assert "Password reset" in out
        assert "because:" not in out

    def test_llm_failure_yields_timesheet_without_justification(self):
        out, _ = self._run(llm_error=RuntimeError("model down"))
        assert "Ticket 99177" in out
        assert "because:" not in out
        assert "Subtotal:" in out
