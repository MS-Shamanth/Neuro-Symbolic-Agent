"""Tests for the Ollama local-model backend (additive to Task 5.1).

These exercise :class:`OllamaBackend` and :func:`ollama_available` WITHOUT a running
server by monkeypatching ``urllib.request.urlopen`` to return canned bytes or raise.
No real network calls are made. They confirm: a successful chat returns the message
content; the JSON format flag is included for ``schema.format == "json"``; a socket
timeout maps to :class:`BackendTimeout`; connection refusal / URLError maps to
:class:`BackendUnavailable` with a helpful message; and the preflight helper reports
reachability correctly.
"""

from __future__ import annotations

import io
import json
import socket
import urllib.error

import pytest

from nsr import llm_component
from nsr.llm_component import (
    BackendTimeout,
    BackendUnavailable,
    OllamaBackend,
    OutputSchema,
    build_ollama_backend,
    ollama_available,
)


class _FakeResponse(io.BytesIO):
    """A minimal context-manager stand-in for an ``http.client.HTTPResponse``."""

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _canned_urlopen(captured, body_bytes):
    """Return a fake ``urlopen`` that records the request and returns ``body_bytes``."""

    def _urlopen(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(body_bytes)

    return _urlopen


def _chat_body(content: str) -> bytes:
    return json.dumps({"message": {"role": "assistant", "content": content}}).encode()


def test_successful_chat_returns_message_content(monkeypatch):
    captured = {}
    body = _chat_body("hello from llama")
    monkeypatch.setattr(
        llm_component.urllib.request, "urlopen", _canned_urlopen(captured, body)
    )

    backend = OllamaBackend("llama3.1")
    result = backend.generate("hi", OutputSchema(format="text"), timeout_s=5.0)

    assert result == "hello from llama"
    # The per-attempt timeout is forwarded to urlopen.
    assert captured["timeout"] == 5.0
    # POST to the resolved host's /api/chat endpoint.
    assert captured["request"].full_url == "http://localhost:11434/api/chat"
    assert captured["request"].method == "POST"


def test_falls_back_to_response_field(monkeypatch):
    captured = {}
    body = json.dumps({"response": "fallback text"}).encode()
    monkeypatch.setattr(
        llm_component.urllib.request, "urlopen", _canned_urlopen(captured, body)
    )

    backend = OllamaBackend("mistral")
    result = backend.generate("hi", OutputSchema(format="text"), timeout_s=3.0)

    assert result == "fallback text"


def test_json_format_flag_included_when_schema_is_json(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        llm_component.urllib.request,
        "urlopen",
        _canned_urlopen(captured, _chat_body("{}")),
    )

    backend = OllamaBackend("qwen2.5")
    backend.generate("give me json", OutputSchema(format="json"), timeout_s=2.0)

    sent = json.loads(captured["request"].data.decode("utf-8"))
    assert sent["format"] == "json"
    assert sent["model"] == "qwen2.5"
    assert sent["stream"] is False
    assert sent["messages"] == [{"role": "user", "content": "give me json"}]


def test_json_format_flag_absent_for_non_json_schema(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        llm_component.urllib.request,
        "urlopen",
        _canned_urlopen(captured, _chat_body("plain")),
    )

    backend = OllamaBackend("phi3")
    backend.generate("hi", OutputSchema(format="text"), timeout_s=2.0)

    sent = json.loads(captured["request"].data.decode("utf-8"))
    assert "format" not in sent


def test_socket_timeout_raises_backend_timeout(monkeypatch):
    def _raise_timeout(request, timeout=None):
        raise socket.timeout("timed out")

    monkeypatch.setattr(llm_component.urllib.request, "urlopen", _raise_timeout)

    backend = OllamaBackend("gemma2")
    with pytest.raises(BackendTimeout):
        backend.generate("hi", OutputSchema(format="text"), timeout_s=0.5)


def test_urlerror_with_timeout_reason_raises_backend_timeout(monkeypatch):
    def _raise(request, timeout=None):
        raise urllib.error.URLError(socket.timeout("timed out"))

    monkeypatch.setattr(llm_component.urllib.request, "urlopen", _raise)

    backend = OllamaBackend("llama3.1")
    with pytest.raises(BackendTimeout):
        backend.generate("hi", OutputSchema(format="text"), timeout_s=0.5)


def test_connection_refused_raises_backend_unavailable(monkeypatch):
    def _raise_refused(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError("Connection refused"))

    monkeypatch.setattr(llm_component.urllib.request, "urlopen", _raise_refused)

    backend = OllamaBackend("llama3.1", host="http://localhost:11434")
    with pytest.raises(BackendUnavailable) as excinfo:
        backend.generate("hi", OutputSchema(format="text"), timeout_s=2.0)

    message = str(excinfo.value)
    assert "localhost:11434" in message
    assert "ollama serve" in message.lower()


def test_http_error_raises_backend_unavailable_with_pull_hint(monkeypatch):
    def _raise_http(request, timeout=None):
        raise urllib.error.HTTPError(
            url="http://localhost:11434/api/chat",
            code=404,
            msg="not found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(llm_component.urllib.request, "urlopen", _raise_http)

    backend = OllamaBackend("nonexistent-model")
    with pytest.raises(BackendUnavailable) as excinfo:
        backend.generate("hi", OutputSchema(format="text"), timeout_s=2.0)

    assert "nonexistent-model" in str(excinfo.value)


def test_generate_never_leaks_raw_exception(monkeypatch):
    def _raise_weird(request, timeout=None):
        raise RuntimeError("something unexpected")

    monkeypatch.setattr(llm_component.urllib.request, "urlopen", _raise_weird)

    backend = OllamaBackend("llama3.1")
    with pytest.raises(BackendUnavailable):
        backend.generate("hi", OutputSchema(format="text"), timeout_s=2.0)


def test_build_ollama_backend_resolves_host_from_env(monkeypatch):
    monkeypatch.setenv("NSR_OLLAMA_HOST", "http://remote-box:11434/")
    backend = build_ollama_backend("llama3.1")
    # Trailing slash stripped during resolution.
    assert backend.host == "http://remote-box:11434"
    assert backend.model == "llama3.1"


def test_build_ollama_backend_explicit_host_wins(monkeypatch):
    monkeypatch.setenv("NSR_OLLAMA_HOST", "http://env-host:11434")
    backend = build_ollama_backend("mistral", host="http://arg-host:9999")
    assert backend.host == "http://arg-host:9999"


def test_ollama_available_returns_false_when_unreachable(monkeypatch):
    def _raise_refused(url, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError("Connection refused"))

    monkeypatch.setattr(llm_component.urllib.request, "urlopen", _raise_refused)

    ok, reason = ollama_available(host="http://localhost:11434", timeout_s=1.0)
    assert ok is False
    assert "localhost:11434" in reason
    assert "ollama serve" in reason.lower()


def test_ollama_available_returns_true_with_model_list(monkeypatch):
    body = json.dumps(
        {"models": [{"name": "llama3.1:8b"}, {"name": "mistral:latest"}]}
    ).encode()

    def _urlopen(url, timeout=None):
        return _FakeResponse(body)

    monkeypatch.setattr(llm_component.urllib.request, "urlopen", _urlopen)

    ok, reason = ollama_available(timeout_s=1.0)
    assert ok is True
    assert "2 models available" in reason
    assert "llama3.1:8b" in reason
    assert "mistral:latest" in reason


def test_ollama_available_true_but_no_models_pulled(monkeypatch):
    body = json.dumps({"models": []}).encode()

    def _urlopen(url, timeout=None):
        return _FakeResponse(body)

    monkeypatch.setattr(llm_component.urllib.request, "urlopen", _urlopen)

    ok, reason = ollama_available(timeout_s=1.0)
    assert ok is True
    assert "0 models available" in reason
