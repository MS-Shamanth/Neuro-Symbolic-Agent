"""LLM Component (System 1) with a pluggable, config-selected backend.

This module implements the neural "System 1" generator described in the design's
*LLM Component* section (Task 5.1). Its job is narrow and deliberate:

- Generate **exactly one** candidate :class:`~nsr.models.CandidateStep` per request for
  the active sub-goal, using the symbolic-state context supplied by the
  Translation_Layer as a :class:`~nsr.models.PromptContext` (Req 2.1, 2.2).
- Select a **hosted-API** backend (endpoint and credentials read from configuration,
  never from source) or a **local-runtime** backend, chosen by configuration
  (Req 2.3, 2.4).
- Enforce the configured generation timeout with bounded retries; after the retry count
  is exhausted, record the failure with its reason and raise an error that names the LLM
  component (:class:`LLMTimeout` / :class:`LLMUnavailable`) (Req 2.5, 2.6).

The backend is abstracted behind :class:`LLMBackend` so the same component can run under
different runtimes. :class:`MockBackend` is provided for testing without a network or a
local model. Constrained decoding (forcing the structured output format) is a separate
concern handled in Task 5.2; here the configured format is passed to the backend as an
:class:`OutputSchema` and a best-effort structured parse is attached to the candidate.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from .models import CandidateStep, PromptContext, SystemConfig

#: The component name recorded in error records so failures identify the LLM (Req 2.6).
LLM_COMPONENT_NAME = "LLM"

#: Selections whose identifier carries this prefix are served by the local runtime.
LOCAL_SELECTION_PREFIX = "local-"

#: Default base URL of a locally running Ollama server.
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

#: Environment variable that overrides the Ollama host when no host is passed.
ENV_OLLAMA_HOST = "NSR_OLLAMA_HOST"


# ---------------------------------------------------------------------------
# Public exceptions (raised by the component after retries are exhausted)
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base class for all LLM component failures."""


class LLMUnavailable(LLMError):
    """Raised when the LLM cannot be reached / served after the retry count (Req 2.6)."""


class LLMTimeout(LLMError):
    """Raised when generation keeps timing out after the retry count (Req 2.5, 2.6)."""


class BackendConfigError(LLMError):
    """Raised when a backend cannot be built because required config is missing."""


# ---------------------------------------------------------------------------
# Backend-internal signals (caught by the component and turned into retries)
# ---------------------------------------------------------------------------


class BackendTimeout(Exception):
    """A backend signals that a single generation attempt timed out."""


class BackendUnavailable(Exception):
    """A backend signals that it could not be reached for a single attempt."""


# ---------------------------------------------------------------------------
# Output schema and backend settings (read from configuration, never source)
# ---------------------------------------------------------------------------


@dataclass
class OutputSchema:
    """The configured structured output format passed to the backend.

    ``format`` mirrors :attr:`SystemConfig.output_format` (for example ``"json"``).
    ``schema`` optionally carries a concrete schema/grammar the constrained decoder
    (Task 5.2) will enforce; the LLM component only forwards it to the backend.
    """

    format: str
    schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: SystemConfig) -> "OutputSchema":
        """Derive the output schema from the system configuration."""
        return cls(format=config.output_format)


@dataclass
class BackendSettings:
    """Connection settings for a backend, sourced from configuration or environment.

    Endpoint and credentials are **never** hardcoded in source; they are supplied
    through :func:`load_backend_settings` from a configuration mapping or environment
    variables (Req 2.3). Local-runtime settings (``local_runtime`` and ``model_path``)
    point at the configured local runtime (Req 2.4).
    """

    model_id: str
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    local_runtime: Optional[str] = None
    model_path: Optional[str] = None


#: Environment variable names used when settings are not supplied programmatically.
ENV_ENDPOINT = "NSR_LLM_ENDPOINT"
ENV_API_KEY = "NSR_LLM_API_KEY"
ENV_MODEL_ID = "NSR_LLM_MODEL_ID"
ENV_LOCAL_RUNTIME = "NSR_LLM_LOCAL_RUNTIME"
ENV_MODEL_PATH = "NSR_LLM_MODEL_PATH"


def load_backend_settings(
    config: SystemConfig,
    mapping: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> BackendSettings:
    """Build :class:`BackendSettings` from configuration / environment, not source.

    Values are resolved in priority order: an explicit ``mapping`` wins, then the
    process ``env`` (defaults to :data:`os.environ`). ``model_id`` falls back to the
    configured :attr:`SystemConfig.llm_selection` so a model is always identified.
    """
    mapping = mapping or {}
    env = os.environ if env is None else env

    def pick(map_key: str, env_key: str) -> Optional[str]:
        if map_key in mapping and mapping[map_key] is not None:
            return str(mapping[map_key])
        if env_key in env and env[env_key]:
            return str(env[env_key])
        return None

    model_id = pick("model_id", ENV_MODEL_ID) or config.llm_selection
    return BackendSettings(
        model_id=model_id,
        endpoint=pick("endpoint", ENV_ENDPOINT),
        api_key=pick("api_key", ENV_API_KEY),
        local_runtime=pick("local_runtime", ENV_LOCAL_RUNTIME),
        model_path=pick("model_path", ENV_MODEL_PATH),
    )


# ---------------------------------------------------------------------------
# Pluggable backend abstraction
# ---------------------------------------------------------------------------


class LLMBackend(ABC):
    """A pluggable text-generation backend.

    A backend turns a rendered prompt into a single raw completion string. It must
    honour ``timeout_s`` for one attempt and signal failures with :class:`BackendTimeout`
    or :class:`BackendUnavailable`; the :class:`LLMComponent` handles retry policy.
    """

    @abstractmethod
    def generate(self, prompt: str, schema: OutputSchema, timeout_s: float) -> str:
        """Produce one raw completion for ``prompt`` within ``timeout_s`` seconds."""
        raise NotImplementedError


class HostedAPIBackend(LLMBackend):
    """Hosted-API backend whose endpoint and credentials come from config (Req 2.3)."""

    def __init__(self, settings: BackendSettings) -> None:
        if not settings.endpoint:
            raise BackendConfigError(
                "hosted-API backend requires an endpoint from configuration"
            )
        if not settings.api_key:
            raise BackendConfigError(
                "hosted-API backend requires credentials from configuration"
            )
        self._settings = settings

    def generate(self, prompt: str, schema: OutputSchema, timeout_s: float) -> str:
        try:
            from openai import OpenAI  # imported lazily so tests need no SDK/network
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise BackendUnavailable("openai SDK is not installed") from exc

        client = OpenAI(
            base_url=self._settings.endpoint,
            api_key=self._settings.api_key,
            timeout=timeout_s,
        )
        try:  # pragma: no cover - exercised only against a live endpoint
            response = client.chat.completions.create(
                model=self._settings.model_id,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout_s,
            )
        except Exception as exc:  # classify into retryable backend signals
            if "timeout" in type(exc).__name__.lower():
                raise BackendTimeout(str(exc)) from exc
            raise BackendUnavailable(str(exc)) from exc
        return response.choices[0].message.content or ""


class LocalRuntimeBackend(LLMBackend):
    """Local-runtime backend that loads the model through the configured runtime (Req 2.4)."""

    def __init__(self, settings: BackendSettings) -> None:
        if not settings.local_runtime:
            raise BackendConfigError(
                "local-runtime backend requires a configured local runtime"
            )
        self._settings = settings

    def generate(self, prompt: str, schema: OutputSchema, timeout_s: float) -> str:  # pragma: no cover - needs a real local model
        # The reference implementation does not bundle a local model; concrete runtime
        # loading is environment-specific. Until a runtime is wired in, signal that the
        # configured local backend is unavailable so the component's retry/timeout
        # policy and error reporting still apply.
        raise BackendUnavailable(
            f"local runtime {self._settings.local_runtime!r} is not available in this "
            "environment"
        )


# ---------------------------------------------------------------------------
# Ollama local-server backend (stdlib only; no new pip dependency)
# ---------------------------------------------------------------------------


def _resolve_ollama_host(
    host: Optional[str] = None, env: Optional[Mapping[str, str]] = None
) -> str:
    """Resolve the Ollama base URL from arg, then env, else the default.

    Priority: explicit ``host`` argument, then ``NSR_OLLAMA_HOST`` in ``env`` (which
    defaults to :data:`os.environ`), else :data:`DEFAULT_OLLAMA_HOST`. A trailing slash
    is stripped so URL joins are predictable.
    """
    env = os.environ if env is None else env
    resolved = host or env.get(ENV_OLLAMA_HOST) or DEFAULT_OLLAMA_HOST
    return resolved.rstrip("/")


class OllamaBackend(LLMBackend):
    """Local-model backend that talks to a running `Ollama <https://ollama.com>`_ server.

    Uses the Python standard library only (``urllib.request`` / ``json``) so no extra
    pip dependency is introduced and the hosted-API ``openai`` SDK is never imported on
    this path. The backend posts to Ollama's ``/api/chat`` endpoint and honours the
    component's per-attempt ``timeout_s``. Failures are converted into the retryable
    :class:`BackendTimeout` / :class:`BackendUnavailable` signals so the
    :class:`LLMComponent` retry and error-reporting policy applies unchanged.

    The backend is selected by callers constructing it directly and injecting it into an
    :class:`LLMComponent`; it requires no change to the configuration enum.
    """

    def __init__(self, model: str, *, host: Optional[str] = None) -> None:
        self.model = model
        self.host = _resolve_ollama_host(host)

    def generate(self, prompt: str, schema: OutputSchema, timeout_s: float) -> str:
        """Produce one completion via Ollama's ``/api/chat`` within ``timeout_s``.

        When ``schema.format == "json"`` the request enables Ollama's JSON mode by
        sending ``"format": "json"``. Any failure is mapped to a backend signal: a socket
        timeout becomes :class:`BackendTimeout`; connection refusal, URL errors, HTTP
        errors, or malformed responses become :class:`BackendUnavailable` with a message
        that names the host and hints that Ollama may not be running or the model may not
        be pulled. A raw exception never escapes this method.
        """
        url = f"{self.host}/api/chat"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if schema is not None and getattr(schema, "format", None) == "json":
            body["format"] = "json"

        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read()
            data = json.loads(raw.decode("utf-8"))
        except socket.timeout as exc:  # per-attempt timeout -> retryable timeout
            raise BackendTimeout(
                f"Ollama request to {url} timed out after {timeout_s:.1f}s"
            ) from exc
        except urllib.error.HTTPError as exc:
            raise BackendUnavailable(
                f"Ollama at {self.host} returned HTTP {exc.code} for model "
                f"{self.model!r}; the model may not be pulled (try "
                f"`ollama pull {self.model}`)"
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, socket.timeout):
                raise BackendTimeout(
                    f"Ollama request to {url} timed out after {timeout_s:.1f}s"
                ) from exc
            raise BackendUnavailable(
                f"could not reach Ollama at {self.host} ({reason}); is the Ollama "
                "server running? (start it with `ollama serve`)"
            ) from exc
        except (ValueError, TypeError) as exc:  # malformed / non-JSON response body
            raise BackendUnavailable(
                f"Ollama at {self.host} returned an unparseable response: {exc}"
            ) from exc
        except Exception as exc:  # never let a raw exception escape generate()
            raise BackendUnavailable(
                f"unexpected error talking to Ollama at {self.host}: {exc}"
            ) from exc

        message = data.get("message") if isinstance(data, dict) else None
        if isinstance(message, dict) and message.get("content") is not None:
            return message["content"]
        if isinstance(data, dict):
            return data.get("response", "")
        return ""


def build_ollama_backend(
    model: str, host: Optional[str] = None
) -> OllamaBackend:
    """Construct an :class:`OllamaBackend` for ``model`` against an optional ``host``.

    Mirrors the other backend factories. The host is resolved from the argument, then
    the ``NSR_OLLAMA_HOST`` environment variable, then the local default.
    """
    return OllamaBackend(model, host=host)


def ollama_available(
    host: Optional[str] = None, *, timeout_s: float = 2.0
) -> tuple[bool, str]:
    """Preflight check for a reachable Ollama server (stdlib only).

    Performs a ``GET {host}/api/tags`` and returns ``(True, message)`` listing the number
    of available models when the server responds, or ``(False, reason)`` with a friendly
    explanation when it cannot be reached or returns an unexpected response. This lets
    callers print a helpful message before a run instead of failing mid-generation.
    """
    resolved = _resolve_ollama_host(host)
    url = f"{resolved}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            raw = response.read()
        data = json.loads(raw.decode("utf-8"))
    except socket.timeout:
        return False, (
            f"Ollama at {resolved} did not respond within {timeout_s:.1f}s; "
            "is the server running? (start it with `ollama serve`)"
        )
    except urllib.error.HTTPError as exc:
        return False, (
            f"Ollama at {resolved} returned HTTP {exc.code} for /api/tags"
        )
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        return False, (
            f"could not reach Ollama at {resolved} ({reason}); is the server "
            "running? (start it with `ollama serve`)"
        )
    except (ValueError, TypeError) as exc:
        return False, (
            f"Ollama at {resolved} returned an unparseable /api/tags response: {exc}"
        )
    except Exception as exc:  # defensive: always return a tuple, never raise
        return False, f"unexpected error checking Ollama at {resolved}: {exc}"

    models = data.get("models", []) if isinstance(data, dict) else []
    names = [
        m.get("name", "?")
        for m in models
        if isinstance(m, dict)
    ]
    if names:
        listed = ", ".join(names)
        return True, f"{len(names)} models available: {listed}"
    return True, "0 models available: none pulled yet (try `ollama pull llama3.1`)"


# ---------------------------------------------------------------------------
# Test/fake backend
# ---------------------------------------------------------------------------

#: A scripted backend item: a string to return, or an exception to raise.
ScriptedItem = Union[str, BaseException]


class MockBackend(LLMBackend):
    """A scriptable in-memory backend for tests (no network, no local model).

    Provide a sequence of scripted items: a ``str`` is returned as the completion, an
    exception instance is raised (use :class:`BackendTimeout` / :class:`BackendUnavailable`
    to drive the retry policy). When the script is exhausted, the last item repeats.
    Every call's prompt and schema are recorded on :attr:`calls` for assertions.
    """

    def __init__(self, script: Optional[Sequence[ScriptedItem]] = None) -> None:
        self._script: list[ScriptedItem] = list(script) if script else [""]
        self._index = 0
        self.calls: list[tuple[str, OutputSchema, float]] = []

    def generate(self, prompt: str, schema: OutputSchema, timeout_s: float) -> str:
        self.calls.append((prompt, schema, timeout_s))
        item = self._script[min(self._index, len(self._script) - 1)]
        self._index += 1
        if isinstance(item, BaseException):
            raise item
        return item

    @property
    def call_count(self) -> int:
        """How many times :meth:`generate` has been invoked."""
        return len(self.calls)


# ---------------------------------------------------------------------------
# Backend factory (selection by configuration)
# ---------------------------------------------------------------------------


def is_local_selection(llm_selection: str) -> bool:
    """Return whether ``llm_selection`` denotes a local-runtime backend (Req 2.4)."""
    return llm_selection.startswith(LOCAL_SELECTION_PREFIX)


def build_backend(
    config: SystemConfig, settings: Optional[BackendSettings] = None
) -> LLMBackend:
    """Construct the backend selected by configuration (Req 2.3, 2.4).

    A ``local-*`` selection yields a :class:`LocalRuntimeBackend`; any other selection
    yields a :class:`HostedAPIBackend`. Connection settings are resolved from
    configuration/environment when not supplied, never from hardcoded source values.
    """
    if settings is None:
        settings = load_backend_settings(config)
    if is_local_selection(config.llm_selection):
        return LocalRuntimeBackend(settings)
    return HostedAPIBackend(settings)


# ---------------------------------------------------------------------------
# LLM Component
# ---------------------------------------------------------------------------


class LLMComponent:
    """System 1 generator: one validated candidate step per request (Req 2.1, 2.2).

    The component is backend-agnostic: pass any :class:`LLMBackend` (or use
    :func:`build_backend` to select one from configuration). The configured generation
    timeout and retry count drive a bounded retry loop; persistent failure raises
    :class:`LLMTimeout` or :class:`LLMUnavailable` naming the LLM component (Req 2.5, 2.6).
    """

    def __init__(
        self,
        backend: LLMBackend,
        config: SystemConfig,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if config.retry_count < 0:
            raise ValueError("retry_count must be non-negative")
        if config.generation_timeout_ms <= 0:
            raise ValueError("generation_timeout_ms must be positive")
        self._backend = backend
        self._retry_count = config.retry_count
        self._timeout_ms = config.generation_timeout_ms
        self._output_schema = OutputSchema.from_config(config)
        self._clock = clock

    @property
    def output_schema(self) -> OutputSchema:
        """The output schema derived from configuration and sent to the backend."""
        return self._output_schema

    def generate_step(
        self,
        context: PromptContext,
        constraint: Optional[OutputSchema] = None,
        *,
        trace: Any = None,
    ) -> CandidateStep:
        """Generate exactly one candidate step for the active sub-goal (Req 2.1, 2.2).

        ``context`` carries the symbolic state rendered by the Translation_Layer, which
        is included in the generation prompt. ``constraint`` defaults to the configured
        output schema. On success a single :class:`CandidateStep` is returned. On
        repeated timeout/unavailability, the failure is recorded with its reason (on the
        optional ``trace`` builder, if provided) and :class:`LLMTimeout` /
        :class:`LLMUnavailable` is raised, naming the LLM component (Req 2.5, 2.6).
        """
        schema = constraint if constraint is not None else self._output_schema
        timeout_s = self._timeout_ms / 1000.0
        attempts = self._retry_count + 1  # initial attempt + configured retries

        last_kind = "unavailable"
        last_reason = "no generation attempt was made"

        for _ in range(attempts):
            try:
                start = self._clock()
                raw = self._backend.generate(context.prompt_text, schema, timeout_s)
                elapsed_ms = (self._clock() - start) * 1000.0
                if elapsed_ms > self._timeout_ms:
                    last_kind = "timeout"
                    last_reason = (
                        f"generation took {elapsed_ms:.1f}ms, exceeding the configured "
                        f"timeout of {self._timeout_ms}ms"
                    )
                    continue
                return self._to_candidate(raw, context)
            except (BackendTimeout, TimeoutError) as exc:
                last_kind = "timeout"
                last_reason = str(exc) or "generation timed out"
            except (BackendUnavailable, ConnectionError) as exc:
                last_kind = "unavailable"
                last_reason = str(exc) or "backend unavailable"

        reason = (
            f"LLM generation failed after {attempts} attempt(s) "
            f"({last_kind}): {last_reason}"
        )
        if trace is not None and hasattr(trace, "set_error_record"):
            trace.set_error_record(LLM_COMPONENT_NAME, reason)
        if last_kind == "timeout":
            raise LLMTimeout(reason)
        raise LLMUnavailable(reason)

    def _to_candidate(self, raw: str, context: PromptContext) -> CandidateStep:
        """Wrap a raw completion into a single candidate step for the active sub-goal.

        A best-effort structured parse is attached when the configured format is JSON
        and the completion is a JSON object; full constraint enforcement is the
        Constrained Decoder's responsibility (Task 5.2).
        """
        structured: dict[str, Any] = {}
        if self._output_schema.format == "json":
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                structured = parsed
        return CandidateStep(
            raw_text=raw,
            structured=structured,
            sub_goal=context.active_sub_goal,
        )
