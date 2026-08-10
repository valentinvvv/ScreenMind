"""
Unified LLM Client for ScreenMind
Communicates with the active inference backend via OpenAI-compatible API:
  - gemma_mode=local  → llama-server (llama.cpp) managed by model_manager
  - gemma_mode=custom → any OpenAI-compatible endpoint (llm_api_base_url)
Supports text, vision (images), and audio input (local mode only).

Inference priority: chat can cancel in-flight analysis via cancel_current_inference().
llama-server frees the GPU slot when the HTTP client disconnects.
"""

import base64
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

    if _is_custom_backend():
        url = f"{_base_url()}/chat/completions"
    else:
        url = f"{_base_url()}/v1/chat/completions"
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if _is_custom_backend():
        # OpenAI-compatible APIs require the model identifier in the payload
        payload["model"] = settings.llm_model_name

    # Create a dedicated client for this request so it can be closed independently
    client = httpx.Client(timeout=timeout)
    with _client_lock:
        _active_client = client

    try:
        response = client.post(url, json=payload, headers=_auth_headers())
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
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
    # Guard: custom endpoints rarely accept input_audio content parts
    if _is_custom_backend():
        raise ValueError(
            "Audio transcription is not supported with custom LLM endpoints "
            "(gemma_mode=custom). Use gemma_mode=local with an audio-capable "
            "model (Gemma 4 E2B/E4B) for voice memos and meeting transcription."
        )

    # Guard: check if active model supports audio
    from screenmind.engine import model_manager
    if not model_manager.is_audio_capable():
        active = model_manager.get_active_model() or "unknown"
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


def get_server_status() -> dict:
    """Get detailed server status."""
    try:
        if _is_custom_backend():
            url = f"{_base_url()}/models"
        else:
            url = f"{_base_url()}/health"
        response = httpx.get(url, timeout=HEALTH_TIMEOUT, headers=_auth_headers())
        if response.status_code == 200:
            return {"status": "ok", "detail": response.json() if response.text else {}}
        return {"status": "error", "detail": f"HTTP {response.status_code}"}
    except httpx.ConnectError:
        target = "LLM API endpoint" if _is_custom_backend() else "llama-server"
        return {"status": "unreachable", "detail": f"Cannot connect to {target}"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
