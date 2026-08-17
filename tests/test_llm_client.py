"""Comprehensive tests for LLM client module."""
import pytest
import threading
from unittest.mock import patch, MagicMock, PropertyMock
import httpx

from screenmind.engine.llm_client import (
    InferenceCancelled, cancel_current_inference, is_inference_active,
    chat, chat_with_images, transcribe_audio, generate, is_available,
    get_server_status, list_remote_models, _cancel_event, _client_lock,
)


class TestInferenceCancellation:
    """Tests for the GPU priority / cancellation system."""

    def test_inference_cancelled_is_exception(self):
        with pytest.raises(InferenceCancelled):
            raise InferenceCancelled("test")

    def test_is_inference_active_default_false(self):
        assert is_inference_active() is False

    def test_cancel_no_active_client(self):
        """Cancel when nothing is running doesn't crash."""
        cancel_current_inference()

    def test_cancel_sets_event(self):
        """Cancel sets the cancel event flag."""
        _cancel_event.clear()
        cancel_current_inference()
        assert _cancel_event.is_set()
        _cancel_event.clear()

    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_chat_clears_cancel_event_on_start(self, mock_client_cls):
        """Each chat() call clears stale cancel flags."""
        _cancel_event.set()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_client_cls.return_value.post.return_value = mock_resp
        chat([{"role": "user", "content": "test"}])
        # Flag should be cleared at start of chat()
        assert not _cancel_event.is_set()

    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_chat_raises_cancelled_when_flag_set(self, mock_client_cls):
        """If cancel flag is set during request, InferenceCancelled is raised."""
        def side_effect(*args, **kwargs):
            _cancel_event.set()
            raise httpx.ConnectError("closed")
        mock_client_cls.return_value.post.side_effect = side_effect
        with pytest.raises(InferenceCancelled):
            chat([{"role": "user", "content": "test"}])
        _cancel_event.clear()

    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_active_client_set_during_request(self, mock_client_cls):
        """_active_client is set during request and cleared after."""
        active_during = []

        def capture_post(*args, **kwargs):
            active_during.append(is_inference_active())
            resp = MagicMock()
            resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
            resp.raise_for_status = MagicMock()
            return resp

        mock_client_cls.return_value.post.side_effect = capture_post
        chat([{"role": "user", "content": "test"}])
        assert active_during[0] is True
        assert is_inference_active() is False


class TestChat:
    """Tests for the chat() function."""

    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_chat_returns_content(self, mock_client_cls):
        """chat() returns the assistant message content."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "Hello!"}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_client_cls.return_value.post.return_value = mock_resp
        result = chat([{"role": "user", "content": "hi"}])
        assert result == "Hello!"

    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_chat_sends_correct_payload(self, mock_client_cls):
        """chat() sends messages, temperature, max_tokens in payload."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_client_cls.return_value.post.return_value = mock_resp

        messages = [{"role": "user", "content": "test"}]
        chat(messages, temperature=0.5, max_tokens=512)

        call_args = mock_client_cls.return_value.post.call_args
        payload = call_args[1]["json"]
        assert payload["messages"] == messages
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 512

    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_chat_raises_on_http_error(self, mock_client_cls):
        """chat() raises on non-cancelled HTTP errors."""
        mock_client_cls.return_value.post.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        with pytest.raises(httpx.HTTPStatusError):
            chat([{"role": "user", "content": "test"}])

    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_chat_null_content_returns_empty_string(self, mock_client_cls):
        """content: null (empty/filtered completion) returns "" — never None.

        Regression: a None return crashed callers on answer.strip() with
        'NoneType' object has no attribute 'strip'.
        """
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": None}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_client_cls.return_value.post.return_value = mock_resp
        result = chat([{"role": "user", "content": "hi"}])
        assert result == ""

    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_chat_malformed_response_raises_value_error(self, mock_client_cls):
        """A response without choices/message raises a descriptive ValueError."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": "chatcmpl-x"}
        mock_resp.raise_for_status = MagicMock()
        mock_client_cls.return_value.post.return_value = mock_resp
        with pytest.raises(ValueError, match="Malformed LLM response"):
            chat([{"role": "user", "content": "hi"}])


class TestChatWithImages:
    """Tests for chat_with_images()."""

    @patch("screenmind.engine.llm_client.chat")
    def test_encodes_images_as_base64(self, mock_chat):
        """Images are base64 encoded in the message."""
        mock_chat.return_value = "I see an image"
        result = chat_with_images("describe this", [b"\xff\xd8\xff\xe0test"])
        assert result == "I see an image"

        call_args = mock_chat.call_args[1] if mock_chat.call_args[1] else {}
        messages = mock_chat.call_args[0][0]
        user_msg = messages[-1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][1]["type"] == "image_url"

    @patch("screenmind.engine.llm_client.chat")
    def test_includes_system_message(self, mock_chat):
        """System message is prepended when provided."""
        mock_chat.return_value = "ok"
        chat_with_images("describe", [b"img"], system="You are helpful")
        messages = mock_chat.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"

    @patch("screenmind.engine.llm_client.chat")
    def test_multiple_images(self, mock_chat):
        """Multiple images are all included."""
        mock_chat.return_value = "ok"
        chat_with_images("compare", [b"img1", b"img2", b"img3"])
        messages = mock_chat.call_args[0][0]
        user_content = messages[-1]["content"]
        image_parts = [p for p in user_content if p["type"] == "image_url"]
        assert len(image_parts) == 3


class TestTranscribeAudio:
    """Tests for transcribe_audio()."""

    @patch("screenmind.engine.llm_client.chat")
    @patch("screenmind.engine.model_manager.is_audio_capable", return_value=True)
    def test_sends_audio_as_input_audio(self, _cap, mock_chat):
        """Audio bytes are sent as input_audio type."""
        mock_chat.return_value = "Hello world"
        result = transcribe_audio(b"fake wav bytes")
        assert result == "Hello world"
        messages = mock_chat.call_args[0][0]
        user_content = messages[0]["content"]
        audio_part = [p for p in user_content if p["type"] == "input_audio"]
        assert len(audio_part) == 1
        assert audio_part[0]["input_audio"]["format"] == "wav"

    @patch("screenmind.engine.model_manager.is_audio_capable", return_value=False)
    @patch("screenmind.engine.model_manager.get_active_model", return_value="non-audio-model")
    def test_raises_on_non_audio_model(self, _get, _cap):
        """transcribe_audio raises ValueError when model doesn't support audio."""
        with pytest.raises(ValueError, match="does not support audio"):
            transcribe_audio(b"fake wav bytes")

    @patch("screenmind.engine.llm_client.chat")
    @patch("screenmind.engine.model_manager.is_audio_capable", return_value=True)
    def test_strips_unused_tokens(self, _cap, mock_chat):
        """<unusedN> garbage tokens from audio encoder are stripped."""
        mock_chat.return_value = "<unused0><unused1>Hello world<unused99>"
        result = transcribe_audio(b"fake wav bytes")
        assert result == "Hello world"

    @patch("screenmind.engine.llm_client.chat")
    @patch("screenmind.engine.model_manager.is_audio_capable", return_value=True)
    def test_strips_only_unused_tokens(self, _cap, mock_chat):
        """Normal text with angle brackets is preserved."""
        mock_chat.return_value = "The value is <100 and >50"
        result = transcribe_audio(b"fake wav bytes")
        assert result == "The value is <100 and >50"

    @patch("screenmind.engine.llm_client.chat")
    @patch("screenmind.engine.model_manager.is_audio_capable", return_value=True)
    def test_all_unused_returns_empty(self, _cap, mock_chat):
        """If result is entirely garbage tokens, returns empty string."""
        mock_chat.return_value = "<unused0><unused1><unused2>"
        result = transcribe_audio(b"fake wav bytes")
        assert result == ""


class TestGenerate:
    """Tests for generate()."""

    @patch("screenmind.engine.llm_client.chat")
    def test_generate_wraps_as_user_message(self, mock_chat):
        """generate() wraps prompt as a single user message."""
        mock_chat.return_value = "response"
        result = generate("tell me a joke")
        assert result == "response"
        messages = mock_chat.call_args[0][0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "tell me a joke"


class TestHealthCheck:
    """Tests for is_available() and get_server_status()."""

    @patch("screenmind.engine.llm_client.httpx.get")
    def test_is_available_true(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        assert is_available() is True

    @patch("screenmind.engine.llm_client.httpx.get")
    def test_is_available_false_on_error(self, mock_get):
        mock_get.side_effect = Exception("refused")
        assert is_available() is False

    @patch("screenmind.engine.llm_client.httpx.get")
    def test_get_server_status_ok(self, mock_get):
        mock_resp = MagicMock(status_code=200, text='{"status":"ok"}')
        mock_resp.json.return_value = {"status": "ok"}
        mock_get.return_value = mock_resp
        status = get_server_status()
        assert status["status"] == "ok"

    @patch("screenmind.engine.llm_client.httpx.get")
    def test_get_server_status_unreachable(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("refused")
        status = get_server_status()
        assert status["status"] == "unreachable"

    @patch("screenmind.engine.llm_client.httpx.get")
    def test_get_server_status_http_error(self, mock_get):
        mock_get.return_value = MagicMock(status_code=503)
        status = get_server_status()
        assert status["status"] == "error"


class TestCustomBackend:
    """Tests for gemma_mode=custom (user-supplied OpenAI-compatible endpoint)."""

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_chat_uses_custom_url_model_and_auth(self, mock_client_cls, mock_settings):
        """Custom mode: /chat/completions (no /v1), model in payload, Bearer key."""
        mock_settings.gemma_mode = "custom"
        mock_settings.llm_api_base_url = "http://api.test/v1/"
        mock_settings.llm_api_key = "sk-test"
        mock_settings.llm_model_name = "test-model"
        mock_settings.text_llm_model_name = None
        mock_settings.text_llm_routing = "off"
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_client_cls.return_value.post.return_value = mock_resp

        chat([{"role": "user", "content": "hi"}])

        call_args = mock_client_cls.return_value.post.call_args
        assert call_args[0][0] == "http://api.test/v1/chat/completions"
        assert call_args[1]["json"]["model"] == "test-model"
        assert call_args[1]["headers"] == {"Authorization": "Bearer sk-test"}

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_chat_custom_no_key_sends_no_auth(self, mock_client_cls, mock_settings):
        """Custom mode without API key sends empty headers (local Ollama, vLLM)."""
        mock_settings.gemma_mode = "custom"
        mock_settings.llm_api_base_url = "http://localhost:11434/v1"
        mock_settings.llm_api_key = None
        mock_settings.llm_model_name = "gemma4:e2b"
        mock_settings.text_llm_model_name = None
        mock_settings.text_llm_routing = "off"
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_client_cls.return_value.post.return_value = mock_resp

        chat([{"role": "user", "content": "hi"}])

        assert mock_client_cls.return_value.post.call_args[1]["headers"] == {}

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_chat_local_mode_unchanged(self, mock_client_cls, mock_settings):
        """Local mode keeps /v1/chat/completions and omits the model field."""
        mock_settings.gemma_mode = "local"
        mock_settings.llama_server_host = "http://127.0.0.1:5809"
        mock_settings.text_llm_routing = "off"
        mock_settings.text_llm_model_name = None
        mock_settings.text_llm_api_base_url = None
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_client_cls.return_value.post.return_value = mock_resp

        chat([{"role": "user", "content": "hi"}])

        call_args = mock_client_cls.return_value.post.call_args
        assert call_args[0][0].endswith("/v1/chat/completions")
        assert "model" not in call_args[1]["json"]

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.get")
    def test_is_available_probes_models_endpoint(self, mock_get, mock_settings):
        """Custom mode health check hits /models with auth header."""
        mock_settings.gemma_mode = "custom"
        mock_settings.llm_api_base_url = "http://api.test/v1"
        mock_settings.llm_api_key = "sk-test"
        mock_get.return_value = MagicMock(status_code=200)

        assert is_available() is True
        call_args = mock_get.call_args
        assert call_args[0][0] == "http://api.test/v1/models"
        assert call_args[1]["headers"] == {"Authorization": "Bearer sk-test"}

    @patch("screenmind.engine.model_manager.settings")
    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.chat")
    def test_transcribe_audio_allowed_on_custom_gemma4(self, mock_chat, mock_lc_settings, mock_mm_settings):
        """Custom endpoint serving Gemma 4 transcribes audio (native encoder)."""
        mock_lc_settings.gemma_mode = "custom"
        mock_lc_settings.llm_model_name = "gemma4:e2b"
        mock_mm_settings.gemma_mode = "custom"
        mock_mm_settings.llm_model_name = "gemma4:e2b"
        mock_chat.return_value = "Hello world"
        assert transcribe_audio(b"fake wav bytes") == "Hello world"
        assert mock_chat.called

    @patch("screenmind.engine.model_manager.settings")
    @patch("screenmind.engine.llm_client.settings")
    def test_transcribe_audio_rejected_on_custom_non_audio_model(self, mock_lc_settings, mock_mm_settings):
        """Custom endpoint with a non-audio model still raises ValueError."""
        mock_lc_settings.gemma_mode = "custom"
        mock_lc_settings.llm_model_name = "llama3:8b"
        mock_mm_settings.gemma_mode = "custom"
        mock_mm_settings.llm_model_name = "llama3:8b"
        with pytest.raises(ValueError, match="does not support audio"):
            transcribe_audio(b"fake wav bytes")

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.get")
    def test_server_status_unreachable_mentions_api(self, mock_get, mock_settings):
        """Unreachable custom endpoint reports 'LLM API endpoint' in detail."""
        mock_settings.gemma_mode = "custom"
        mock_settings.llm_api_base_url = "http://api.test/v1"
        mock_settings.llm_api_key = None
        mock_get.side_effect = httpx.ConnectError("refused")
        status = get_server_status()
        assert status["status"] == "unreachable"
        assert "LLM API endpoint" in status["detail"]


class TestListRemoteModels:
    """Tests for list_remote_models() — GET /models model discovery."""

    @patch("screenmind.engine.llm_client.httpx.get")
    def test_returns_sorted_ids(self, mock_get):
        """Model ids are extracted from OpenAI /models response and sorted."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [
            {"id": "zeta-model"}, {"id": "alpha-model"}, {"id": "mid-model"},
        ]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        models = list_remote_models("http://api.test/v1", None)
        assert models == ["alpha-model", "mid-model", "zeta-model"]
        assert mock_get.call_args[0][0] == "http://api.test/v1/models"

    @patch("screenmind.engine.llm_client.httpx.get")
    def test_sends_bearer_key(self, mock_get):
        """API key is sent as Bearer auth when provided."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        list_remote_models("http://api.test/v1/", "sk-test")
        assert mock_get.call_args[1]["headers"] == {"Authorization": "Bearer sk-test"}

    @patch("screenmind.engine.llm_client.httpx.get")
    def test_strips_trailing_slash(self, mock_get):
        """Trailing slash on base URL doesn't produce //models."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        list_remote_models("http://api.test/v1/", None)
        assert mock_get.call_args[0][0] == "http://api.test/v1/models"

    @patch("screenmind.engine.llm_client.httpx.get")
    def test_skips_malformed_entries(self, mock_get):
        """Entries without a string id are ignored, not crashed on."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [
            {"id": "good"}, {"name": "no-id"}, "not-a-dict", {"id": 42},
        ]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        assert list_remote_models("http://api.test/v1", None) == ["good"]

    @patch("screenmind.engine.llm_client.httpx.get")
    def test_raises_on_http_error(self, mock_get):
        """HTTP errors propagate so the caller can report them."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=MagicMock()
        )
        mock_get.return_value = mock_resp
        with pytest.raises(httpx.HTTPStatusError):
            list_remote_models("http://api.test/v1", "bad-key")


class TestErrorDetailSurfacing:
    """Server error bodies must reach the user — a bare '400 Bad Request' hid
    the real cause of summary-generation failures (context overflow)."""

    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_chat_error_includes_json_error_message(self, mock_client_cls):
        """llama.cpp-style {"error": {"message": ...}} detail lands in the exception."""
        resp = httpx.Response(
            400,
            request=httpx.Request("POST", "http://api.test/v1/chat/completions"),
            json={"error": {
                "message": "request (10476 tokens) exceeds the available context size (6144 tokens), try increasing it",
                "type": "exceed_context_size_error",
            }},
        )
        mock_client_cls.return_value.post.return_value = resp
        with pytest.raises(httpx.HTTPStatusError, match="exceeds the available context size"):
            chat([{"role": "user", "content": "test"}])

    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_chat_error_includes_plain_text_body(self, mock_client_cls):
        """Non-JSON error bodies are surfaced too."""
        resp = httpx.Response(
            400,
            request=httpx.Request("POST", "http://api.test/v1/chat/completions"),
            text="model not loaded",
        )
        mock_client_cls.return_value.post.return_value = resp
        with pytest.raises(httpx.HTTPStatusError, match="model not loaded"):
            chat([{"role": "user", "content": "test"}])


class TestTextModelRouting:
    """Dedicated text model handles text-only operations on its own endpoint."""

    def _mock_settings(self, mock_settings, routing="overflow", text_model="big-model",
                       text_window=32768, context_window=6144,
                       text_url="", text_key=None):
        mock_settings.gemma_mode = "custom"
        mock_settings.llm_api_base_url = "http://api.test/v1"
        mock_settings.llm_api_key = None
        mock_settings.llm_model_name = "primary-model"
        mock_settings.text_llm_api_base_url = text_url
        mock_settings.text_llm_api_key = text_key
        mock_settings.text_llm_model_name = text_model
        mock_settings.text_llm_routing = routing
        mock_settings.text_llm_context_window = text_window
        mock_settings.context_window = context_window
        mock_settings.vision_llm_enabled = False
        mock_settings.vision_llm_model_name = None
        mock_settings.vision_llm_api_base_url = ""
        mock_settings.vision_llm_api_key = None
        mock_settings.vision_llm_context_window = 32768

    def _chat_ok(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_client_cls.return_value.post.return_value = mock_resp

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_always_routes_text_to_text_model(self, mock_client_cls, mock_settings):
        self._mock_settings(mock_settings, routing="always")
        self._chat_ok(mock_client_cls)
        chat([{"role": "user", "content": "short"}], max_tokens=64)
        payload = mock_client_cls.return_value.post.call_args[1]["json"]
        assert payload["model"] == "big-model"

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_overflow_routes_large_text_to_text_model(self, mock_client_cls, mock_settings):
        self._mock_settings(mock_settings, routing="overflow")
        self._chat_ok(mock_client_cls)
        chat([{"role": "user", "content": "x" * 20000}], max_tokens=2048)
        payload = mock_client_cls.return_value.post.call_args[1]["json"]
        assert payload["model"] == "big-model"

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_overflow_keeps_small_text_on_primary(self, mock_client_cls, mock_settings):
        self._mock_settings(mock_settings, routing="overflow")
        self._chat_ok(mock_client_cls)
        chat([{"role": "user", "content": "x" * 1000}], max_tokens=256)
        payload = mock_client_cls.return_value.post.call_args[1]["json"]
        assert payload["model"] == "primary-model"

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_off_never_routes(self, mock_client_cls, mock_settings):
        self._mock_settings(mock_settings, routing="off")
        self._chat_ok(mock_client_cls)
        chat([{"role": "user", "content": "x" * 20000}], max_tokens=2048)
        payload = mock_client_cls.return_value.post.call_args[1]["json"]
        assert payload["model"] == "primary-model"

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_multimodal_never_routes_even_on_always(self, mock_client_cls, mock_settings):
        """Vision/audio requests always stay on the primary model."""
        self._mock_settings(mock_settings, routing="always")
        self._chat_ok(mock_client_cls)
        chat([{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}},
        ]}], max_tokens=64)
        payload = mock_client_cls.return_value.post.call_args[1]["json"]
        assert payload["model"] == "primary-model"

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_no_text_model_configured_stays_primary(self, mock_client_cls, mock_settings):
        self._mock_settings(mock_settings, routing="always", text_model="")
        self._chat_ok(mock_client_cls)
        chat([{"role": "user", "content": "short"}], max_tokens=64)
        payload = mock_client_cls.return_value.post.call_args[1]["json"]
        assert payload["model"] == "primary-model"

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_text_model_uses_own_endpoint_url_and_key(self, mock_client_cls, mock_settings):
        """A configured text endpoint gets its own URL and its own key."""
        self._mock_settings(mock_settings, routing="always",
                            text_url="http://text.test/v1/", text_key="sk-text")
        self._chat_ok(mock_client_cls)
        chat([{"role": "user", "content": "short"}], max_tokens=64)
        call_args = mock_client_cls.return_value.post.call_args
        assert call_args[0][0] == "http://text.test/v1/chat/completions"
        assert call_args[1]["json"]["model"] == "big-model"
        assert call_args[1]["headers"] == {"Authorization": "Bearer sk-text"}

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_text_model_on_shared_endpoint_reuses_primary_key(self, mock_client_cls, mock_settings):
        """Empty text URL rides the primary endpoint — primary key applies."""
        self._mock_settings(mock_settings, routing="always")
        mock_settings.llm_api_key = "sk-primary"
        self._chat_ok(mock_client_cls)
        chat([{"role": "user", "content": "short"}], max_tokens=64)
        call_args = mock_client_cls.return_value.post.call_args
        assert call_args[0][0] == "http://api.test/v1/chat/completions"
        assert call_args[1]["headers"] == {"Authorization": "Bearer sk-primary"}

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_text_model_own_endpoint_without_key_sends_no_auth(self, mock_client_cls, mock_settings):
        """The primary key is never forwarded to a different endpoint."""
        self._mock_settings(mock_settings, routing="always", text_url="http://text.test/v1")
        mock_settings.llm_api_key = "sk-primary"
        self._chat_ok(mock_client_cls)
        chat([{"role": "user", "content": "short"}], max_tokens=64)
        call_args = mock_client_cls.return_value.post.call_args
        assert call_args[1]["headers"] == {}

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_local_primary_with_own_text_url_routes(self, mock_client_cls, mock_settings):
        """Local llama-server primary + dedicated text endpoint → routing works."""
        self._mock_settings(mock_settings, routing="always", text_url="http://text.test/v1")
        mock_settings.gemma_mode = "local"
        mock_settings.llama_server_host = "http://127.0.0.1:5809"
        self._chat_ok(mock_client_cls)
        chat([{"role": "user", "content": "short"}], max_tokens=64)
        call_args = mock_client_cls.return_value.post.call_args
        assert call_args[0][0] == "http://text.test/v1/chat/completions"
        assert call_args[1]["json"]["model"] == "big-model"

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_text_model_request_disables_thinking(self, mock_client_cls, mock_settings):
        """Reasoning models burn max_tokens on hidden thinking and return
        content: null — text-model requests must send enable_thinking: false.

        Regression: qwen3.5-9b on vLLM produced empty agent output and
        'model returned an empty response' in chat until thinking was off.
        """
        self._mock_settings(mock_settings, routing="always")
        self._chat_ok(mock_client_cls)
        chat([{"role": "user", "content": "short"}], max_tokens=64)
        payload = mock_client_cls.return_value.post.call_args[1]["json"]
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_primary_request_omits_thinking_knob(self, mock_client_cls, mock_settings):
        """Primary-backend requests never carry chat_template_kwargs."""
        self._mock_settings(mock_settings, routing="off")
        self._chat_ok(mock_client_cls)
        chat([{"role": "user", "content": "short"}], max_tokens=64)
        payload = mock_client_cls.return_value.post.call_args[1]["json"]
        assert "chat_template_kwargs" not in payload

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_unreachable_text_model_falls_back_to_primary(self, mock_client_cls, mock_settings):
        """Text endpoint down → the same request is retried on the primary
        backend with the primary model, thinking knob dropped."""
        import httpx as _httpx
        self._mock_settings(mock_settings, routing="always", text_url="http://text.test/v1")
        ok_resp = MagicMock()
        ok_resp.json.return_value = {"choices": [{"message": {"content": "from primary"}}]}
        ok_resp.raise_for_status = MagicMock()
        # Capture a snapshot of each attempt's payload (the dict is mutated in place)
        captured = []
        def _post(url, json=None, headers=None):
            captured.append((url, dict(json)))
            if len(captured) == 1:
                raise _httpx.ConnectError("text endpoint down")
            return ok_resp
        mock_client_cls.return_value.post.side_effect = _post
        result = chat([{"role": "user", "content": "short"}], max_tokens=64)
        assert result == "from primary"
        assert len(captured) == 2
        first_url, first_payload = captured[0]
        second_url, second_payload = captured[1]
        assert first_url == "http://text.test/v1/chat/completions"
        assert first_payload["model"] == "big-model"
        assert second_url == "http://api.test/v1/chat/completions"
        assert second_payload["model"] == "primary-model"
        assert "chat_template_kwargs" not in second_payload

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_both_endpoints_down_raises(self, mock_client_cls, mock_settings):
        """When the fallback also fails, the error surfaces to the caller."""
        import httpx as _httpx
        self._mock_settings(mock_settings, routing="always", text_url="http://text.test/v1")
        mock_client_cls.return_value.post.side_effect = [
            _httpx.ConnectError("text down"), _httpx.ConnectError("primary down"),
        ]
        with pytest.raises(_httpx.ConnectError):
            chat([{"role": "user", "content": "short"}], max_tokens=64)
        assert mock_client_cls.return_value.post.call_count == 2

    @patch("screenmind.engine.llm_client.settings")
    def test_local_primary_without_text_url_never_routes(self, mock_settings):
        """llama-server serves one model — without a text URL there is nowhere
        to route to."""
        mock_settings.gemma_mode = "local"
        mock_settings.text_llm_api_base_url = ""
        mock_settings.text_llm_api_key = None
        mock_settings.text_llm_model_name = "big-model"
        mock_settings.text_llm_routing = "always"
        mock_settings.text_llm_context_window = 32768
        from screenmind.engine.llm_client import _route_to_text_model
        assert _route_to_text_model([{"role": "user", "content": "x" * 20000}], 2048) is False


class TestVisionModelRouting:
    """Dedicated vision model handles image requests on its own endpoint."""

    def _mock_settings(self, mock_settings, enabled=True, vision_model="vision-model",
                       vision_url="", vision_key=None):
        mock_settings.gemma_mode = "custom"
        mock_settings.llm_api_base_url = "http://api.test/v1"
        mock_settings.llm_api_key = None
        mock_settings.llm_model_name = "primary-model"
        mock_settings.text_llm_model_name = None
        mock_settings.text_llm_routing = "off"
        mock_settings.text_llm_api_base_url = ""
        mock_settings.text_llm_context_window = 32768
        mock_settings.context_window = 6144
        mock_settings.vision_llm_enabled = enabled
        mock_settings.vision_llm_model_name = vision_model
        mock_settings.vision_llm_api_base_url = vision_url
        mock_settings.vision_llm_api_key = vision_key
        mock_settings.vision_llm_context_window = 32768

    def _chat_ok(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_client_cls.return_value.post.return_value = mock_resp

    def _image_messages(self):
        return [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}},
        ]}]

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_image_routes_to_vision_model(self, mock_client_cls, mock_settings):
        self._mock_settings(mock_settings)
        self._chat_ok(mock_client_cls)
        chat(self._image_messages(), max_tokens=64)
        payload = mock_client_cls.return_value.post.call_args[1]["json"]
        assert payload["model"] == "vision-model"

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_text_stays_on_primary_when_vision_enabled(self, mock_client_cls, mock_settings):
        self._mock_settings(mock_settings)
        self._chat_ok(mock_client_cls)
        chat([{"role": "user", "content": "short"}], max_tokens=64)
        payload = mock_client_cls.return_value.post.call_args[1]["json"]
        assert payload["model"] == "primary-model"

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_toggle_off_keeps_images_on_primary(self, mock_client_cls, mock_settings):
        self._mock_settings(mock_settings, enabled=False)
        self._chat_ok(mock_client_cls)
        chat(self._image_messages(), max_tokens=64)
        payload = mock_client_cls.return_value.post.call_args[1]["json"]
        assert payload["model"] == "primary-model"

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_empty_model_name_keeps_images_on_primary(self, mock_client_cls, mock_settings):
        self._mock_settings(mock_settings, vision_model="")
        self._chat_ok(mock_client_cls)
        chat(self._image_messages(), max_tokens=64)
        payload = mock_client_cls.return_value.post.call_args[1]["json"]
        assert payload["model"] == "primary-model"

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_vision_model_uses_own_endpoint_url_and_key(self, mock_client_cls, mock_settings):
        """A configured vision endpoint gets its own URL and its own key."""
        self._mock_settings(mock_settings, vision_url="http://vision.test/v1/",
                            vision_key="sk-vision")
        self._chat_ok(mock_client_cls)
        chat(self._image_messages(), max_tokens=64)
        call_args = mock_client_cls.return_value.post.call_args
        assert call_args[0][0] == "http://vision.test/v1/chat/completions"
        assert call_args[1]["json"]["model"] == "vision-model"
        assert call_args[1]["headers"] == {"Authorization": "Bearer sk-vision"}

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_vision_model_on_shared_endpoint_reuses_primary_key(self, mock_client_cls, mock_settings):
        """Empty vision URL rides the primary endpoint — primary key applies."""
        self._mock_settings(mock_settings)
        mock_settings.llm_api_key = "sk-primary"
        self._chat_ok(mock_client_cls)
        chat(self._image_messages(), max_tokens=64)
        call_args = mock_client_cls.return_value.post.call_args
        assert call_args[0][0] == "http://api.test/v1/chat/completions"
        assert call_args[1]["headers"] == {"Authorization": "Bearer sk-primary"}

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_vision_own_endpoint_without_key_sends_no_auth(self, mock_client_cls, mock_settings):
        """The primary key is never forwarded to a different endpoint."""
        self._mock_settings(mock_settings, vision_url="http://vision.test/v1")
        mock_settings.llm_api_key = "sk-primary"
        self._chat_ok(mock_client_cls)
        chat(self._image_messages(), max_tokens=64)
        call_args = mock_client_cls.return_value.post.call_args
        assert call_args[1]["headers"] == {}

    @patch("screenmind.engine.llm_client.settings")
    @patch("screenmind.engine.llm_client.httpx.Client")
    def test_local_primary_with_own_vision_url_routes(self, mock_client_cls, mock_settings):
        """Local llama-server primary + dedicated vision endpoint → routing works."""
        self._mock_settings(mock_settings, vision_url="http://vision.test/v1")
        mock_settings.gemma_mode = "local"
        mock_settings.llama_server_host = "http://127.0.0.1:5809"
        self._chat_ok(mock_client_cls)
        chat(self._image_messages(), max_tokens=64)
        call_args = mock_client_cls.return_value.post.call_args
        assert call_args[0][0] == "http://vision.test/v1/chat/completions"
        assert call_args[1]["json"]["model"] == "vision-model"

    @patch("screenmind.engine.llm_client.settings")
    def test_local_primary_without_vision_url_never_routes(self, mock_settings):
        """llama-server serves one model — without a vision URL there is
        nowhere to route to."""
        mock_settings.gemma_mode = "local"
        mock_settings.vision_llm_enabled = True
        mock_settings.vision_llm_api_base_url = ""
        mock_settings.vision_llm_api_key = None
        mock_settings.vision_llm_model_name = "vision-model"
        mock_settings.vision_llm_context_window = 32768
        from screenmind.engine.llm_client import _route_to_vision_model
        assert _route_to_vision_model(self._image_messages()) is False

    @patch("screenmind.engine.llm_client.settings")
    def test_vision_model_window_reflects_toggle(self, mock_settings):
        """vision_model_window() is None while disabled, the configured
        window while enabled."""
        from screenmind.engine.llm_client import vision_model_window
        mock_settings.gemma_mode = "custom"
        mock_settings.vision_llm_model_name = "vision-model"
        mock_settings.vision_llm_api_base_url = "http://vision.test/v1"
        mock_settings.vision_llm_context_window = 65536
        mock_settings.vision_llm_enabled = True
        assert vision_model_window() == 65536
        mock_settings.vision_llm_enabled = False
        assert vision_model_window() is None
