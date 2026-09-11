"""Tests for engine/analyzer.py — response parsing logic (no Ollama needed)."""

import json
from unittest.mock import patch

from screenmind.engine.analyzer import (
    GemmaAnalyzer, UNPARSED_SUMMARY, is_remote_session,
)


def test_parse_clean_json():
    analyzer = GemmaAnalyzer()
    raw = '{"app_name": "Chrome", "activity_category": "browsing", "activity_summary": "Reading docs", "detailed_context": "", "visible_text_snippets": [], "mood": "learning", "confidence": 0.9, "scene_description": ""}'
    record = analyzer._parse_response(raw)
    assert record.app_name == "Chrome"
    assert record.activity_category == "browsing"
    assert record.mood == "learning"


def test_parse_json_in_code_block():
    analyzer = GemmaAnalyzer()
    raw = '```json\n{"app_name": "VS Code", "activity_category": "coding", "activity_summary": "Editing main.py", "detailed_context": "", "visible_text_snippets": [], "mood": "productive", "confidence": 0.85, "scene_description": ""}\n```'
    record = analyzer._parse_response(raw)
    assert record.app_name == "VS Code"
    assert record.activity_category == "coding"


def test_parse_with_thinking_tags():
    analyzer = GemmaAnalyzer()
    raw = '<think>Let me analyze this screenshot...</think>{"app_name": "Slack", "activity_category": "communication", "activity_summary": "Chatting", "detailed_context": "", "visible_text_snippets": [], "mood": "collaborative", "confidence": 0.8, "scene_description": ""}'
    record = analyzer._parse_response(raw)
    assert record.app_name == "Slack"
    assert record.activity_category == "communication"


def test_parse_regex_fallback():
    analyzer = GemmaAnalyzer()
    raw = 'Here is the analysis: "app_name": "Terminal", "activity_category": "terminal", "activity_summary": "running tests"'
    record = analyzer._parse_response(raw)
    # Regex fallback should extract what it can
    assert record.confidence == 0.3  # Low confidence for regex


def test_normalize_category():
    analyzer = GemmaAnalyzer()
    from screenmind.storage.models import ActivityRecord
    # The normalize function checks if a valid category is a substring
    record = ActivityRecord(activity_category="browsing", mood="productive")
    normalized = analyzer._normalize(record)
    assert normalized.activity_category == "browsing"
    assert normalized.mood == "productive"


def test_normalize_invalid_category():
    analyzer = GemmaAnalyzer()
    from screenmind.storage.models import ActivityRecord
    record = ActivityRecord(activity_category="invalid_thing", mood="unknown_mood")
    normalized = analyzer._normalize(record)
    assert normalized.activity_category == "other"
    assert normalized.mood == "neutral"


def test_normalize_confidence_clamping():
    analyzer = GemmaAnalyzer()
    from screenmind.storage.models import ActivityRecord
    # Pydantic enforces 0-1 range, so test that normalize handles edge values
    record = ActivityRecord(confidence=1.0)
    normalized = analyzer._normalize(record)
    assert normalized.confidence == 1.0

    record = ActivityRecord(confidence=0.0)
    normalized = analyzer._normalize(record)
    assert normalized.confidence == 0.0


class TestSceneFromText:
    """generate_scene_from_text — scene narration via the text-only model."""

    @patch("screenmind.engine.analyzer.llm_client")
    @patch("screenmind.engine.analyzer.settings")
    def test_returns_stripped_scene_with_context(self, mock_settings, mock_llm):
        mock_settings.context_window = 8192
        mock_llm.text_model_window.return_value = None
        mock_llm.chat.return_value = "  A terminal running pytest  "
        scene = GemmaAnalyzer().generate_scene_from_text(
            ocr_text="collected 12 items " * 10,
            app_name="Windows Terminal",
            window_title="pytest",
        )
        assert scene == "A terminal running pytest"
        messages = mock_llm.chat.call_args.kwargs["messages"]
        # Text-only payload — llm_client routes it per text_llm_routing
        assert all(isinstance(m["content"], str) for m in messages)
        assert "OS-detected app: Windows Terminal" in messages[0]["content"]
        assert "Window title: pytest" in messages[0]["content"]

    @patch("screenmind.engine.analyzer.llm_client")
    @patch("screenmind.engine.analyzer.settings")
    def test_organized_text_preferred_over_raw_ocr(self, mock_settings, mock_llm):
        mock_settings.context_window = 8192
        mock_llm.text_model_window.return_value = 16384
        mock_llm.chat.return_value = "scene"
        GemmaAnalyzer().generate_scene_from_text(
            ocr_text="raw ocr words " * 20,
            organized_text="[main_content] organized body " * 20,
        )
        prompt = mock_llm.chat.call_args.kwargs["messages"][0]["content"]
        assert "[main_content] organized body" in prompt
        assert "raw ocr words" not in prompt

    @patch("screenmind.engine.analyzer.llm_client")
    @patch("screenmind.engine.analyzer.settings")
    def test_short_text_skips_llm(self, mock_settings, mock_llm):
        assert GemmaAnalyzer().generate_scene_from_text(ocr_text="too short") is None
        assert GemmaAnalyzer().generate_scene_from_text(ocr_text=None) is None
        mock_llm.chat.assert_not_called()

    @patch("screenmind.engine.analyzer.llm_client")
    @patch("screenmind.engine.analyzer.settings")
    def test_llm_error_returns_none(self, mock_settings, mock_llm):
        mock_settings.context_window = 8192
        mock_llm.text_model_window.return_value = None
        mock_llm.chat.side_effect = RuntimeError("server down")
        assert GemmaAnalyzer().generate_scene_from_text(ocr_text="x" * 100) is None

    @patch("screenmind.engine.analyzer.llm_client")
    @patch("screenmind.engine.analyzer.settings")
    def test_empty_response_returns_none(self, mock_settings, mock_llm):
        mock_settings.context_window = 8192
        mock_llm.text_model_window.return_value = None
        mock_llm.chat.return_value = "   "
        assert GemmaAnalyzer().generate_scene_from_text(ocr_text="y" * 100) is None

    @patch("screenmind.engine.analyzer.llm_client")
    @patch("screenmind.engine.analyzer.settings")
    def test_prompt_budgeted_to_window(self, mock_settings, mock_llm):
        mock_settings.context_window = 2048
        mock_llm.text_model_window.return_value = None
        mock_llm.chat.return_value = "scene"
        GemmaAnalyzer().generate_scene_from_text(ocr_text="z" * 50000)
        prompt = mock_llm.chat.call_args.kwargs["messages"][0]["content"]
        budget = (2048 - 700) * 2  # window minus prompt+output, at 2 chars/token
        assert "z" * budget in prompt
        assert "z" * (budget + 1) not in prompt


class TestRemoteSessionDetection:
    """is_remote_session — remote-desktop clients must be recognized from
    the OS app name or window title so scene narration targets the session
    content, not the client chrome."""

    def test_citrix_desktop_viewer_title(self):
        assert is_remote_session(
            "Citrix.DesktopViewer.App", "GITSMGMT-XA FSlogix - Desktop Viewer")

    def test_citrix_app_name_alone(self):
        assert is_remote_session("Citrix.DesktopViewer.App", "")
        assert is_remote_session("Citrix Workspace", "Some session")

    def test_mstsc_and_vnc(self):
        assert is_remote_session("mstsc.exe", "SRV01 - Remote Desktop Connection")
        assert is_remote_session("vncviewer", "host - VNC Viewer")

    def test_regular_windows_not_remote(self):
        assert not is_remote_session("chrome.exe", "Inbox - Outlook")
        assert not is_remote_session(None, None)
        assert not is_remote_session("code.exe", "main.py - Visual Studio Code")


class TestSceneFromTextRemoteSession:
    """Remote-desktop captures: the scene prompt must point the model at
    the content INSIDE the session, not the client's toolbar/title."""

    @patch("screenmind.engine.analyzer.llm_client")
    @patch("screenmind.engine.analyzer.settings")
    def test_remote_prompt_targets_session_content(self, mock_settings, mock_llm):
        mock_settings.context_window = 8192
        mock_llm.text_model_window.return_value = None
        mock_llm.chat.return_value = "scene"
        GemmaAnalyzer().generate_scene_from_text(
            ocr_text="Backup job GITSLAB_VCA7 credentials management " * 5,
            app_name="Citrix.DesktopViewer.App",
            window_title="GITSMGMT-XA FSlogix - Desktop Viewer",
        )
        prompt = mock_llm.chat.call_args.kwargs["messages"][0]["content"]
        assert "REMOTE-DESKTOP session" in prompt
        assert "INSIDE the remote machine" in prompt
        # The generic inventory instruction must not be used for remotes
        assert "plain inventory of everything visible" not in prompt

    @patch("screenmind.engine.analyzer.llm_client")
    @patch("screenmind.engine.analyzer.settings")
    def test_regular_prompt_unchanged(self, mock_settings, mock_llm):
        mock_settings.context_window = 8192
        mock_llm.text_model_window.return_value = None
        mock_llm.chat.return_value = "scene"
        GemmaAnalyzer().generate_scene_from_text(
            ocr_text="editing auth_middleware.py in VS Code " * 5,
            app_name="code.exe",
            window_title="auth_middleware.py - Visual Studio Code",
        )
        prompt = mock_llm.chat.call_args.kwargs["messages"][0]["content"]
        assert "plain inventory of everything visible" in prompt
        assert "REMOTE-DESKTOP" not in prompt


# Shapes taken from real merged-mode responses, with the screen content
# replaced. The failing one is what the model emits when it loops inside
# visible_text_snippets until max_tokens cuts it off: fenced, nested two
# levels deep, and stopping mid-array with no closing fence.
_TRUNCATED = """```json
{
  "layout": [
    {
      "name": "main_content",
      "x_start": 0.0,
      "x_end": 0.75,
      "y_start": 0.0,
      "y_end": 0.96,
      "content_type": "code"
    },
    {
      "name": "toolbars",
      "x_start": 0.0,
      "x_end": 1.0,
      "y_start": 0.0,
      "y_end": 0.05,
      "content_type": "toolbar"
    }
  ],
  "analysis": {
    "app_name": "Terminal",
    "activity_category": "terminal",
    "activity_summary": "The user is running a deployment script.",
    "detailed_context": "A shell session is executing a deploy.",
    "visible_text_snippets": [
      "status error read tcp",
      "status error read tcp",
      "status error read tcp",
      "status error"""

_COMPLETE = """```json
{
  "layout": [
    {
      "name": "main_content",
      "x_start": 0.0,
      "x_end": 1.0,
      "y_start": 0.0,
      "y_end": 1.0,
      "content_type": "code"
    }
  ],
  "analysis": {
    "app_name": "Terminal",
    "activity_category": "terminal",
    "activity_summary": "The user is running a deployment script.",
    "detailed_context": "A shell session is executing a deploy.",
    "visible_text_snippets": ["deploy.sh"],
    "mood": "productive",
    "confidence": 0.9,
    "scene_description": "A terminal window fills the screen."
  }
}
```"""


class TestExtractJson:
    """Extraction must not hand back a valid-but-wrong fragment."""

    def setup_method(self):
        self.a = GemmaAnalyzer()

    def test_complete_response_yields_the_whole_object(self):
        body = self.a._extract_json(_COMPLETE)
        assert body is not None
        data = json.loads(body)
        assert set(data) == {"layout", "analysis"}

    def test_truncated_response_yields_nothing(self):
        """The bug: this used to return the first layout region.

        ~150 chars of valid JSON with no analysis in it, which callers then
        accepted as a successful parse and turned into an empty record.
        """
        assert self.a._extract_json(_TRUNCATED) is None

    def test_braces_inside_strings_do_not_close_the_object(self):
        raw = '{"activity_summary": "user typed a { brace", "confidence": 0.9}'
        data = json.loads(self.a._extract_json(raw))
        assert data["activity_summary"] == "user typed a { brace"

    def test_escaped_quote_inside_a_string_is_not_a_terminator(self):
        raw = r'{"activity_summary": "he said \"hi\" then {", "confidence": 0.5}'
        data = json.loads(self.a._extract_json(raw))
        assert data["confidence"] == 0.5

    def test_prose_before_the_object_is_skipped(self):
        raw = 'Here is the analysis you asked for:\n{"activity_summary": "coding"}'
        assert json.loads(self.a._extract_json(raw))["activity_summary"] == "coding"

    def test_no_object_at_all(self):
        assert self.a._extract_json("I cannot analyze this image.") is None


class TestMergedParseRecovery:
    """A truncated response still carries the analysis — recover it."""

    def setup_method(self):
        self.a = GemmaAnalyzer()

    def test_truncated_response_recovers_the_summary(self):
        record, regions = self.a._parse_merged_response(_TRUNCATED, "Terminal", "bash")
        assert record.activity_summary == "The user is running a deployment script."
        assert record.activity_category == "terminal"
        assert regions == []          # layout was in the part that got cut

    def test_complete_response_is_unaffected(self):
        record, regions = self.a._parse_merged_response(_COMPLETE, "Terminal", "bash")
        assert record.activity_summary == "The user is running a deployment script."
        assert len(regions) == 1

    def test_a_balanced_but_wrong_object_is_not_mistaken_for_analysis(self):
        """Defence in depth: ActivityRecord(**x) ignores unknown keys.

        A stray object — a layout region, say — must not be accepted as the
        analysis payload. There is nothing to recover from this text, so an
        empty record is the honest outcome (the worker's quality gate then
        retries); what matters is that no region field leaks in and no
        content is invented.
        """
        raw = ('{"name": "main_content", "x_start": 0.0, "x_end": 0.75, '
               '"y_start": 0.0, "y_end": 0.96, "content_type": "code"}')
        record, regions = self.a._parse_merged_response(raw, "Terminal", "bash")
        # Nothing in this text is recoverable, so the salvage marker is the
        # honest answer. The worker's quality gate treats it as a missing
        # summary and retries — see test_unparsed_marker_counts_as_missing.
        assert record.activity_summary == UNPARSED_SUMMARY
        assert record.detailed_context == ""
        assert regions == []
        assert record.app_name == "Terminal"      # from the hint, not the object

    def test_stray_object_before_the_analysis_does_not_win(self):
        """Extraction takes the first balanced object — which may be junk.

        Here a coordinate blob precedes the real payload. Without the
        structural check that blob is accepted and the analysis below it is
        never looked at; with it, the regex fallback still finds the summary.
        """
        raw = (
            'Detected regions: {"x_start": 0.0, "y_start": 0.0}\n'
            'Analysis:\n'
            '{"app_name": "Terminal", "activity_category": "terminal", '
            '"activity_summary": "The user is tailing a log file."}'
        )
        record, _regions = self.a._parse_merged_response(raw, "Terminal", "bash")
        assert record.activity_summary == "The user is tailing a log file."

    def test_looks_like_analysis(self):
        from screenmind.engine.analyzer import _looks_like_analysis
        assert _looks_like_analysis({"activity_summary": "x"}) is True
        assert _looks_like_analysis({"confidence": 0.5}) is True
        assert _looks_like_analysis({"name": "main_content", "x_start": 0.0}) is False
        assert _looks_like_analysis([]) is False
