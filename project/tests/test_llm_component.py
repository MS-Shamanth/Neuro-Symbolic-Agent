"""Tests for the LLM Component (System 1) and its pluggable backend (Task 5.1).

Covers single-step generation including symbolic-state context (Req 2.1, 2.2),
config-driven backend selection with endpoint/credentials sourced from configuration
(Req 2.3, 2.4), and the bounded retry / timeout policy with error reporting that names
the LLM component (Req 2.5, 2.6).
"""

from __future__ import annotations

import json

import pytest

from nsr.llm_component import (
    BackendConfigError,
    BackendTimeout,
    BackendUnavailable,
    HostedAPIBackend,
    LLMComponent,
    LLMTimeout,
    LLMUnavailable,
    LocalRuntimeBackend,
    MockBackend,
    OutputSchema,
    build_backend,
    is_local_selection,
    load_backend_settings,
)
from nsr.models import CandidateStep, PromptContext, SystemConfig
from nsr.proof_trace import ProofTraceBuilder


def make_config(**overrides):
    base = dict(
        max_cycle_limit=10,
        repair_attempt_limit=3,
        retry_count=2,
        llm_selection="gpt-4o-mini",
        output_format="json",
        conflict_resolution_policy="priority",
        generation_timeout_ms=30000,
    )
    base.update(overrides)
    return SystemConfig(**base)


def make_context(sub_goal="prove lemma"):
    return PromptContext(
        goal_description="solve the problem",
        active_sub_goal=sub_goal,
        established_conclusions=["fact(1)"],
        prompt_text="Goal: solve the problem\nCurrent sub-goal: prove lemma",
    )


# ---------------------------------------------------------------------------
# Single-step generation (Req 2.1, 2.2)
# ---------------------------------------------------------------------------


def test_generate_step_returns_single_candidate_for_active_sub_goal():
    backend = MockBackend(['{"logic_form": "add(2,2)=4"}'])
    component = LLMComponent(backend, make_config())

    step = component.generate_step(make_context(sub_goal="compute sum"))

    assert isinstance(step, CandidateStep)
    assert step.sub_goal == "compute sum"
    assert step.structured == {"logic_form": "add(2,2)=4"}
    assert backend.call_count == 1  # exactly one generation per request


def test_generate_step_includes_symbolic_state_context_in_prompt():
    backend = MockBackend(["ok"])
    component = LLMComponent(backend, make_config())
    context = make_context()

    component.generate_step(context)

    sent_prompt, sent_schema, _timeout = backend.calls[0]
    assert sent_prompt == context.prompt_text
    assert sent_schema.format == "json"


def test_non_json_output_leaves_structured_empty_but_keeps_raw_text():
    backend = MockBackend(["a plain text step"])
    component = LLMComponent(backend, make_config())

    step = component.generate_step(make_context())

    assert step.raw_text == "a plain text step"
    assert step.structured == {}


def test_non_json_format_does_not_attempt_json_parse():
    backend = MockBackend(['{"logic_form": "x"}'])
    component = LLMComponent(backend, make_config(output_format="logic-form"))

    step = component.generate_step(make_context())

    # format is not JSON, so structured is left to the constrained decoder (Task 5.2)
    assert step.structured == {}
    assert json.loads(step.raw_text) == {"logic_form": "x"}


# ---------------------------------------------------------------------------
# Retry / timeout policy (Req 2.5, 2.6)
# ---------------------------------------------------------------------------


def test_retries_then_succeeds_within_retry_count():
    backend = MockBackend([BackendTimeout("slow"), BackendTimeout("slow"), "recovered"])
    component = LLMComponent(backend, make_config(retry_count=2))

    step = component.generate_step(make_context())

    assert step.raw_text == "recovered"
    assert backend.call_count == 3  # initial + 2 retries


@pytest.mark.parametrize("retry_count", [0, 1, 3, 5])
def test_total_attempts_equals_retry_count_plus_one_on_persistent_timeout(retry_count):
    backend = MockBackend([BackendTimeout("always slow")])
    component = LLMComponent(backend, make_config(retry_count=retry_count))

    with pytest.raises(LLMTimeout):
        component.generate_step(make_context())

    assert backend.call_count == retry_count + 1


def test_timeout_exhaustion_records_failure_naming_llm():
    backend = MockBackend([BackendTimeout("deadline exceeded")])
    component = LLMComponent(backend, make_config(retry_count=1))
    builder = ProofTraceBuilder()

    with pytest.raises(LLMTimeout):
        component.generate_step(make_context(), trace=builder)

    err = builder.trace.error_record
    assert err is not None
    assert err.failed_component == "LLM"
    assert "deadline exceeded" in err.reason


def test_unavailable_exhaustion_raises_llm_unavailable_and_records_reason():
    backend = MockBackend([BackendUnavailable("connection refused")])
    component = LLMComponent(backend, make_config(retry_count=0))
    builder = ProofTraceBuilder()

    with pytest.raises(LLMUnavailable):
        component.generate_step(make_context(), trace=builder)

    assert backend.call_count == 1
    assert builder.trace.error_record.failed_component == "LLM"


def test_slow_response_exceeding_timeout_is_treated_as_timeout():
    # A controllable clock makes the first (and only) "successful" return look slow.
    ticks = iter([0.0, 100.0])  # start=0s, end=100s -> 100000ms elapsed

    def clock():
        return next(ticks)

    backend = MockBackend(["late answer"])
    component = LLMComponent(
        backend, make_config(retry_count=0, generation_timeout_ms=1000), clock=clock
    )

    with pytest.raises(LLMTimeout):
        component.generate_step(make_context())


# ---------------------------------------------------------------------------
# Backend selection and settings sourcing (Req 2.3, 2.4)
# ---------------------------------------------------------------------------


def test_is_local_selection():
    assert is_local_selection("local-llama3") is True
    assert is_local_selection("gpt-4o") is False


def test_build_backend_selects_hosted_for_hosted_selection():
    config = make_config(llm_selection="gpt-4o")
    settings = load_backend_settings(
        config, mapping={"endpoint": "https://api.example.com", "api_key": "secret"}
    )
    backend = build_backend(config, settings)
    assert isinstance(backend, HostedAPIBackend)


def test_build_backend_selects_local_for_local_selection():
    config = make_config(llm_selection="local-llama3")
    settings = load_backend_settings(config, mapping={"local_runtime": "llama.cpp"})
    backend = build_backend(config, settings)
    assert isinstance(backend, LocalRuntimeBackend)


def test_hosted_backend_requires_endpoint_and_credentials_from_config():
    config = make_config(llm_selection="gpt-4o")
    # No endpoint/credentials supplied anywhere -> cannot be built from source.
    settings = load_backend_settings(config, mapping={}, env={})
    with pytest.raises(BackendConfigError):
        build_backend(config, settings)


def test_settings_are_read_from_mapping_then_env_not_source():
    config = make_config(llm_selection="gpt-4o")
    env = {"NSR_LLM_ENDPOINT": "https://env.example.com", "NSR_LLM_API_KEY": "env-key"}
    settings = load_backend_settings(config, env=env)
    assert settings.endpoint == "https://env.example.com"
    assert settings.api_key == "env-key"

    # An explicit mapping overrides the environment.
    settings2 = load_backend_settings(
        config, mapping={"endpoint": "https://map.example.com"}, env=env
    )
    assert settings2.endpoint == "https://map.example.com"
    assert settings2.api_key == "env-key"  # falls back to env when not in mapping
    assert settings2.model_id == "gpt-4o"  # defaults to the configured selection


def test_local_backend_requires_configured_runtime():
    config = make_config(llm_selection="local-mistral")
    settings = load_backend_settings(config, mapping={}, env={})
    with pytest.raises(BackendConfigError):
        build_backend(config, settings)
