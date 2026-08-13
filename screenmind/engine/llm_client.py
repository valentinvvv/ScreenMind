"""
Unified LLM Client for ScreenMind
Communicates with the active inference backend via OpenAI-compatible API:
  - gemma_mode=local  → llama-server (llama.cpp) managed by model_manager
  - gemma_mode=custom → any OpenAI-compatible endpoint (llm_api_base_url)
Supports text, vision (images), and audio input (any backend whose model has
an audio encoder — Gemma 4 in local or custom mode).

Inference priority: chat can cancel in-flight analysis via cancel_current_inference().
llama-server frees the GPU slot when the HTTP client disconnects.
"""

import base64
import json
import logging
import threading
import time
from typing import Optional, List

import httpx

from screenmind.config import settings

logger = logging.getLogger("screenmind.engine.llm_client")


# Timeout for inference calls (screenshots can take 30-60s on slow hardware)
INFERENCE_TIMEOUT = 300.0
HEALTH_TIMEOUT = 5.0

# Gemma tokenizes dense screen/code text at ~2 chars/token (measured against
# llama.cpp: 8600 chars of activity lines = 3015 tokens). Used to budget
# prompts against settings.context_window without a tokenizer dependency.
CHARS_PER_TOKEN = 2.0


def _raise_with_detail(response: httpx.Response) -> None:
    """raise_for_status() with the server's error body surfaced in the message.

    llama.cpp/Ollama-style bodies: {"error": {"message": "...", "type": "..."}}.
    A bare "400 Bad Request" hides the actual cause (e.g. context overflow).
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        detail = response.text.strip()
        try:
            parsed = response.json()
            err = parsed.get("error") if isinstance(parsed, dict) else None
            if isinstance(err, dict) and err.get("message"):
                detail = str(err["message"])
            elif isinstance(err, str) and err:
                detail = err
        except (ValueError, AttributeError):
            pass
        raise httpx.HTTPStatusError(
            f"{e} — {detail[:500]}", request=e.request, response=e.response
        ) from None


def _extract_content(data) -> str:
    """Extract the assistant text from an OpenAI-style completion response.

    Returns "" when content is null/absent — some endpoints signal an empty or
    filtered completion that way (frequent with text-only models). Callers
    already handle empty answers (retry / vision fallback), so this preserves
    chat()'s `-> str` contract instead of leaking None into downstream
    `.strip()` calls. Raises ValueError on a structurally malformed response so
    the real problem surfaces instead of a NoneType crash.
    """
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(
            f"Malformed LLM response (missing choices/message): {str(data)[:200]}"
        ) from e
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        logger.debug("LLM response had no string content (%s) — treating as empty", type(content).__name__)
        return ""
    return content


class InferenceCancelled(Exception):
    """Raised when an in-flight inference is cancelled (e.g., chat pre-emption)."""
    pass


# ── Cancellation state ──────────────────────────────────────────────────────
# _cancel_event: set by cancel_current_inference(), cleared at start of chat()
# _active_client: the httpx.Client for the in-flight request (closed to abort)
# _client_lock: protects only _active_client reference, never blocks requests
_cancel_event = threading.Event()
_active_client: Optional[httpx.Client] = None
_client_lock = threading.Lock()


def cancel_current_inference():
    """
    Cancel any in-flight inference request. Safe to call anytime.

    Sets the cancel flag and closes the active HTTP client, which causes
    llama-server to free the GPU slot immediately. The caller (analysis worker)
    will receive an InferenceCancelled or httpx exception and should re-queue.

    Mainly beneficial for merged/accurate mode (~76s). In fast mode (~12s),
    the analysis may finish before cancellation propagates.
    """
    _cancel_event.set()
    with _client_lock:
        if _active_client:
            try:
                _active_client.close()
            except Exception:
                pass  # Already closed or errored — fine
    logger.info("Inference cancelled (chat priority)")


def is_inference_active() -> bool:
    """Check if an inference request is currently in-flight."""
    with _client_lock:
        return _active_client is not None


def _is_custom_backend() -> bool:
    """True when inference runs against a user-supplied OpenAI-compatible endpoint."""
    return settings.gemma_mode == "custom"


def _base_url() -> str:
    if _is_custom_backend():
        return settings.llm_api_base_url.rstrip("/")
    return settings.llama_server_host.rstrip("/")


def _auth_headers() -> dict:
    """Authorization header for the custom endpoint (empty when no key configured)."""
    if _is_custom_backend() and settings.llm_api_key:
        return {"Authorization": f"Bearer {settings.llm_api_key}"}
    return {}

def _is_text_only(messages: list) -> bool:
    """True when every message content is a plain string (no images/audio parts)."""
    return bool(messages) and all(isinstance(m.get("content"), str) for m in messages)


def _text_model_configured() -> bool:
    """A text model needs a name plus a reachable endpoint: either its own
    TEXT_LLM_API_BASE_URL, or the primary custom endpoint to ride on."""
    if not (settings.text_llm_model_name or "").strip():
        return False
    return bool((settings.text_llm_api_base_url or "").strip()) or _is_custom_backend()


def _text_base_url() -> str:
    """Base URL of the text-model endpoint (falls back to the primary endpoint)."""
    url = (settings.text_llm_api_base_url or "").strip()
    return url.rstrip("/") if url else _base_url()


def _text_auth_headers() -> dict:
    """Authorization header for the text endpoint.

    Own key when set; otherwise the primary key, but only while the text model
    shares the primary endpoint (empty text URL). A distinct endpoint with no
    key gets no header — the primary key is never sent to a different server.
    """
    key = (settings.text_llm_api_key or "").strip()
    if key:
        return {"Authorization": f"Bearer {key}"}
    if not (settings.text_llm_api_base_url or "").strip():
        return _auth_headers()
    return {}


def text_model_window() -> Optional[int]:
    """Context window of the configured text model, or None when routing is
    unavailable (model unset or routing 'off')."""
    if not _text_model_configured() or settings.text_llm_routing == "off":
        return None
    return settings.text_llm_context_window


def _route_to_text_model(messages: list, max_tokens: int) -> bool:
    """Decide whether this request goes to the secondary text model.

    - 'always':   every text-only request (vision/audio always stay on primary)
    - 'overflow': text-only requests whose estimated prompt + max_tokens exceed
                  the primary Context Window
    Token estimate uses the same conservative CHARS_PER_TOKEN as prompt budgeting.
    """
    window = text_model_window()
    if window is None or not _is_text_only(messages):
        return False
    if settings.text_llm_routing == "always":
        return True
    est_tokens = sum(int(len(m.get("content", "")) / CHARS_PER_TOKEN) for m in messages)
    return est_tokens + max_tokens > settings.context_window


def chat(
    messages: list,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout: float = INFERENCE_TIMEOUT,
) -> str:
    """
    Send a chat completion request to the active inference backend.

    Messages follow OpenAI format:
    [{"role": "user", "content": "text"}]
    or multimodal:
    [{"role": "user", "content": [
        {"type": "text", "text": "..."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    ]}]

    Raises InferenceCancelled if cancel_current_inference() is called during request.
    Returns the assistant's response text.
    """
    global _active_client

    # Clear cancel flag at start of every request — prevents stale cancellation
    # from a previous cancel_current_inference() call that had nothing to cancel
    _cancel_event.clear()

    use_text_model = _route_to_text_model(messages, max_tokens)
    if use_text_model:
        # The text model is always addressed as an OpenAI-compatible endpoint,
        # even when the primary backend is the local llama-server.
        url = f"{_text_base_url()}/chat/completions"
        headers = _text_auth_headers()
    else:
        url = (
            f"{_base_url()}/chat/completions"
            if _is_custom_backend()
            else f"{_base_url()}/v1/chat/completions"
        )
        headers = _auth_headers()
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if use_text_model or _is_custom_backend():
        # OpenAI-compatible APIs require the model identifier in the payload
        payload["model"] = (
            settings.text_llm_model_name if use_text_model else settings.llm_model_name
        )

    # Create a dedicated client for this request so it can be closed independently
    client = httpx.Client(timeout=timeout)
    with _client_lock:
        _active_client = client

    try:
        response = client.post(url, json=payload, headers=headers)
        _raise_with_detail(response)
        data = response.json()
        return _extract_content(data)
    except Exception as e:
        # Check if this was a cancellation (flag set + connection error)
        if _cancel_event.is_set():
            raise InferenceCancelled("Inference cancelled for chat priority") from e
        raise
    finally:
        with _client_lock:
            if _active_client is client:
                _active_client = None
        try:
            client.close()
        except Exception:
            pass


def chat_with_images(
    prompt: str,
    images: List[bytes],
    system: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout: float = INFERENCE_TIMEOUT,
) -> str:
    """
    Chat with image inputs. Convenience wrapper for vision calls.

    Args:
        prompt: User text prompt
        images: List of JPEG image bytes
        system: Optional system message
        temperature: Sampling temperature
        max_tokens: Max response tokens
    """
    content = [{"type": "text", "text": prompt}]
    for img_bytes in images:
        b64 = base64.b64encode(img_bytes).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    return chat(messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout)


def transcribe_audio(
    audio_bytes: bytes,
    prompt: str = "Transcribe this audio accurately. Output only the transcription, nothing else.",
    audio_format: str = "wav",
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout: float = INFERENCE_TIMEOUT,
) -> str:
    """
    Transcribe audio using the active model's native audio encoder.

    Args:
        audio_bytes: Raw audio file bytes (WAV format recommended)
        prompt: Instruction for the model
        audio_format: Audio format (wav, mp3, etc.)
        temperature: Sampling temperature
        max_tokens: Max response tokens

    Raises:
        ValueError: If the active backend/model doesn't support audio input.
    """
    # Guard: check if active model supports audio (local registry lookup, or
    # model-name inference for custom endpoints)
    from screenmind.engine import model_manager
    if not model_manager.is_audio_capable():
        active = (settings.llm_model_name if _is_custom_backend()
                  else (model_manager.get_active_model() or "unknown"))
        raise ValueError(
            f"Model '{active}' does not support audio input. "
            f"Switch to Gemma 4 E2B or E4B for voice memo and meeting transcription."
        )

    b64_audio = base64.b64encode(audio_bytes).decode()

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {"data": b64_audio, "format": audio_format}},
        ],
    }]

    result = chat(messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout)

    # Strip Gemma's <unusedN> garbage tokens — these appear when the audio
    # encoder can't parse the input (too short, silence, noise)
    import re
    result = re.sub(r'<unused\d+>', '', result).strip()

    return result


def generate(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    timeout: float = INFERENCE_TIMEOUT,
) -> str:
    """
    Simple text generation (no conversation history).
    Replaces ollama client.generate().
    """
    messages = [{"role": "user", "content": prompt}]
    return chat(messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout)


def is_available() -> bool:
    """Check if the inference backend is reachable and healthy."""
    try:
        if _is_custom_backend():
            url = f"{_base_url()}/models"
        else:
            url = f"{_base_url()}/health"
        response = httpx.get(url, timeout=HEALTH_TIMEOUT, headers=_auth_headers())
        return response.status_code == 200
    except Exception:
        return False


def get_server_status(timeout: float = HEALTH_TIMEOUT) -> dict:
    """Get detailed server status."""
    try:
        if _is_custom_backend():
            url = f"{_base_url()}/models"
        else:
            url = f"{_base_url()}/health"
        response = httpx.get(url, timeout=timeout, headers=_auth_headers())
        if response.status_code == 200:
            return {"status": "ok", "detail": response.json() if response.text else {}}
        return {"status": "error", "detail": f"HTTP {response.status_code}"}
    except httpx.ConnectError:
        target = "LLM API endpoint" if _is_custom_backend() else "llama-server"
        return {"status": "unreachable", "detail": f"Cannot connect to {target}"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def list_remote_models(base_url: Optional[str] = None, api_key: Optional[str] = None) -> list:
    """
    Fetch the model list from an OpenAI-compatible endpoint (GET /models).

    Args:
        base_url: Endpoint base URL (defaults to settings.llm_api_base_url)
        api_key: Bearer key (defaults to settings.llm_api_key)

    Returns a list of model id strings, sorted. Raises on connection/HTTP errors.
    """
    url = (base_url or settings.llm_api_base_url).rstrip("/") + "/models"
    headers = {}
    key = api_key if api_key is not None else settings.llm_api_key
    if key:
        headers["Authorization"] = f"Bearer {key}"
    response = httpx.get(url, timeout=HEALTH_TIMEOUT, headers=headers)
    response.raise_for_status()
    data = response.json()
    items = data.get("data", []) if isinstance(data, dict) else []
    return sorted(
        m["id"] for m in items
        if isinstance(m, dict) and isinstance(m.get("id"), str)
    )
