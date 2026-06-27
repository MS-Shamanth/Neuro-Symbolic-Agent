"""Unit tests for the retry and timeout paths of System 1 (Task 5.3).

This module focuses specifically on the failure/recovery paths shared by the
LLM Component and the Constrained Decoder:

- **Timeout-then-retry** -- a generation attempt that times out (either because the
  backend signals a timeout or because the elapsed time exceeds the configured
  generation timeout) is retried, and a later attempt within the retry count succeeds
  (Req 2.5).
- **Retry exhaustion** -- once the configured retry count is exhausted, the failure is
  recorded with its reason and an error record naming the LLM component is produced
  (Req 2.6).
- **Constraint-unsatisfied termination** -- when the LLM keeps producing non-conforming
  output, the Constrained Decoder marks and journals each attempt and, on exhaustion,
  terminates the query with ``constraint-unsatisfied`` (Req 3.3, 3.4).

These complement the broader coverage in ``test_llm_component.py`` and
``test_constrained_decoder.py`` by concentrating on the precise retry/timeout boundaries
and the error-record / termination-reason contracts. The real
:class:`~nsr.llm_component.LLMComponent` and :class:`~nsr.constrained_decoder.ConstrainedDecoder`
are driven by a scriptable :class:`~nsr.llm_component.MockBackend`, so no network or
local model is required.
"""

from __future__ import annotations

import pytest

from nsr.constrained_decoder import (
    NON_CONFORMING_KEY,
    ConstrainedDecoder,
    ConstraintUnsatisfied,
)
from nsr.llm_component import (
    LLM_COMPONENT_NAME,
    BackendTimeout,
    BackendUnavailable,
    LLMComponent,
    LLMTimeout,
    LLMUnavailable,
    MockBackend,
)
from nsr.models import (
    Goal,
    PromptContext,
    SubGoal,
    SymbolicRepresentation,
    SystemConfig,
    TerminationReason,
    ValidationStatus,
    WorkingMemoryState,
)
from nsr.proof_trace import ProofTraceBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(**overrides) -> SystemConfig:
    base = dict(
        max_cycle_limit=10,
        repair_attempt_limit=3,
        retry_count=2,
        llm_selection="gpt-4o-mini",
        output_format="json",
        conflict_resolution_policy="priority",
        generation_timeout_ms=1000,
    )
    base.update(overrides)
    return SystemConfig(**base)


def make_context(sub_goal: str = "prove lemma") -> PromptContext:
    return PromptContext(
        goal_description="solve the problem",
        active_sub_goal=sub_goal,
        established_conclusions=["fact(1)"],
        prompt_text="Goal: solve the problem\nCurrent sub-goal: " + sub_goal,
    )


def make_state() -> WorkingMemoryState:
    return WorkingMemoryState(
        goal_buffer=Goal(
            description="solve the equation",
            sub_goals=[SubGoal(description="isolate x")],
        ),
        declarative_memory=[SymbolicRepresentation(logic_form="eq(2x,6)")],
        procedural_memory=[],
        imaginal_buffer=None,
    )


class _SlowThenFastClock:
    """A controllable clock that returns scripted (start, end) pairs per attempt.

    Each call to :meth:`generate` on the backend consumes a ``start`` then an ``end``
    timestamp here, so the component computes ``end - start`` as the attempt's elapsed
    time. This lets a test make specific attempts look slow (over the timeout) and
    others look fast (under the timeout) without real sleeping.
    """

    def __init__(self, ticks):
        self._ticks = iter(ticks)

    def __call__(self) -> float:
        return next(self._ticks)


# ---------------------------------------------------------------------------
# Req 2.5 -- timeout on an attempt, then a successful retry within retry_count
# ---------------------------------------------------------------------------


def test_slow_attempt_then_fast_retry_succeeds_within_retry_count():
    # Attempt 1: start=0.0s end=5.0s -> 5000ms elapsed > 1000ms timeout (slow).
    # Attempt 2: start=5.0s end=5.1s -> 100ms elapsed <= 1000ms timeout (fast, ok).
    clock = _SlowThenFastClock([0.0, 5.0, 5.0, 5.1])
    backend = MockBackend(['{"logic_form": "slow"}', '{"logic_form": "x=3"}'])
    component = LLMComponent(
        backend, make_config(retry_count=2, generation_timeout_ms=1000), clock=clock
    )

    step = component.generate_step(make_context())

    # The slow attempt was discarded and the fast retry's result was returned.
    assert step.structured == {"logic_form": "x=3"}
    assert backend.call_count == 2  # initial slow attempt + one successful retry


def test_backend_timeout_signal_then_retry_succeeds():
    # The backend signals a timeout on the first attempt, then recovers on the retry.
    backend = MockBackend([BackendTimeout("deadline"), '{"logic_form": "recovered"}'])
    component = LLMComponent(backend, make_config(retry_count=2))

    step = component.generate_step(make_context())

    assert step.structured == {"logic_form": "recovered"}
    assert backend.call_count == 2


def test_unavailable_then_retry_succeeds_within_retry_count():
    backend = MockBackend([BackendUnavailable("connection reset"), "plain recovery"])
    component = LLMComponent(backend, make_config(retry_count=1))

    step = component.generate_step(make_context())

    assert step.raw_text == "plain recovery"
    assert backend.call_count == 2


def test_retry_succeeds_on_final_allowed_attempt():
    # retry_count=2 -> 3 total attempts; only the last one succeeds.
    backend = MockBackend(
        [BackendTimeout("1"), BackendTimeout("2"), '{"logic_form": "ok"}']
    )
    component = LLMComponent(backend, make_config(retry_count=2))

    step = component.generate_step(make_context())

    assert step.structured == {"logic_form": "ok"}
    assert backend.call_count == 3


# ---------------------------------------------------------------------------
# Req 2.6 -- retry exhaustion records the failure and names the LLM component
# ---------------------------------------------------------------------------


def test_timeout_exhaustion_records_error_record_naming_llm():
    backend = MockBackend([BackendTimeout("deadline exceeded")])
    component = LLMComponent(backend, make_config(retry_count=2))
    builder = ProofTraceBuilder()

    with pytest.raises(LLMTimeout) as excinfo:
        component.generate_step(make_context(), trace=builder)

    # Every attempt was made before failing (initial + 2 retries).
    assert backend.call_count == 3
    # The error record names the LLM and carries the underlying failure reason.
    err = builder.trace.error_record
    assert err is not None
    assert err.failed_component == LLM_COMPONENT_NAME
    assert "deadline exceeded" in err.reason
    assert "deadline exceeded" in str(excinfo.value)


def test_unavailable_exhaustion_raises_llm_unavailable_naming_llm():
    backend = MockBackend([BackendUnavailable("connection refused")])
    component = LLMComponent(backend, make_config(retry_count=1))
    builder = ProofTraceBuilder()

    with pytest.raises(LLMUnavailable) as excinfo:
        component.generate_step(make_context(), trace=builder)

    assert backend.call_count == 2
    err = builder.trace.error_record
    assert err is not None
    assert err.failed_component == LLM_COMPONENT_NAME
    assert "connection refused" in err.reason
    assert "connection refused" in str(excinfo.value)


def test_exhaustion_preserves_existing_proof_trace_contents():
    # A trace that already holds a recorded step must be preserved when the LLM fails
    # after exhausting its retries (Req 2.6: "preserve the existing Proof_Trace").
    builder = ProofTraceBuilder()
    builder.append_step(
        "earlier accepted step",
        status=ValidationStatus.ACCEPTED,
        applied_rule_id="R1",
    )
    backend = MockBackend([BackendUnavailable("down")])
    component = LLMComponent(backend, make_config(retry_count=0))

    with pytest.raises(LLMUnavailable):
        component.generate_step(make_context(), trace=builder)

    # The pre-existing step is intact and an error record was added alongside it.
    assert len(builder.trace.steps) == 1
    assert builder.trace.steps[0].step_text == "earlier accepted step"
    assert builder.trace.error_record is not None
    assert builder.trace.error_record.failed_component == LLM_COMPONENT_NAME


@pytest.mark.parametrize("retry_count", [0, 1, 4])
def test_attempts_equal_retry_count_plus_one_before_error(retry_count):
    backend = MockBackend([BackendTimeout("always slow")])
    component = LLMComponent(backend, make_config(retry_count=retry_count))

    with pytest.raises(LLMTimeout):
        component.generate_step(make_context())

    assert backend.call_count == retry_count + 1


def test_mixed_failures_report_last_kind_as_timeout():
    # The last failing attempt is a timeout, so LLMTimeout (not LLMUnavailable) is raised.
    backend = MockBackend(
        [BackendUnavailable("flaky"), BackendTimeout("then slow")]
    )
    component = LLMComponent(backend, make_config(retry_count=1))
    builder = ProofTraceBuilder()

    with pytest.raises(LLMTimeout):
        component.generate_step(make_context(), trace=builder)

    assert builder.trace.error_record.failed_component == LLM_COMPONENT_NAME
    assert "then slow" in builder.trace.error_record.reason


# ---------------------------------------------------------------------------
# Req 3.3, 3.4 -- repeated non-conforming output terminates constraint-unsatisfied
# ---------------------------------------------------------------------------


def _decoder(script, *, output_format="json", retry_count=2):
    config = make_config(output_format=output_format, retry_count=retry_count)
    backend = MockBackend(script=script)
    llm = LLMComponent(backend, config)
    return ConstrainedDecoder(llm, config), backend


def test_repeated_non_conforming_terminates_constraint_unsatisfied():
    builder = ProofTraceBuilder()
    decoder, backend = _decoder(["not json"], retry_count=2)

    with pytest.raises(ConstraintUnsatisfied) as excinfo:
        decoder.decode(make_context(), make_state(), builder=builder)

    # retry_count=2 -> initial attempt + 2 regenerations = 3 attempts.
    assert excinfo.value.attempts == 3
    assert backend.call_count == 3
    assert (
        excinfo.value.termination_reason == TerminationReason.CONSTRAINT_UNSATISFIED
    )
    # Each non-conforming attempt was marked and journalled in execution order.
    assert len(builder.trace.steps) == 3
    for index, step in enumerate(builder.trace.steps):
        assert step.status == ValidationStatus.REJECTED
        outcome = step.translation_outcomes[0]
        assert outcome[NON_CONFORMING_KEY] is True
        assert outcome["attempt"] == index
    # The trace records the constraint-unsatisfied termination (Req 3.4).
    assert (
        builder.trace.termination_reason == TerminationReason.CONSTRAINT_UNSATISFIED
    )


def test_non_conforming_then_conforming_recovers_before_exhaustion():
    builder = ProofTraceBuilder()
    decoder, backend = _decoder(
        ["bad", "still bad", '{"logic_form": "x=3"}'], retry_count=2
    )

    candidate = decoder.decode(make_context(), make_state(), builder=builder)

    assert candidate.structured["logic_form"] == "x=3"
    assert backend.call_count == 3
    # Only the two non-conforming attempts were journalled; no termination set.
    assert len(builder.trace.steps) == 2
    assert builder.trace.termination_reason is None


def test_constraint_unsatisfied_with_zero_retries_makes_single_attempt():
    builder = ProofTraceBuilder()
    decoder, backend = _decoder(["nope"], retry_count=0)

    with pytest.raises(ConstraintUnsatisfied) as excinfo:
        decoder.decode(make_context(), make_state(), builder=builder)

    assert excinfo.value.attempts == 1
    assert backend.call_count == 1
    assert len(builder.trace.steps) == 1
    assert (
        builder.trace.termination_reason == TerminationReason.CONSTRAINT_UNSATISFIED
    )


def test_constraint_unsatisfied_for_logic_form_empty_output():
    # An empty logic-form output never conforms, so retries exhaust and terminate.
    builder = ProofTraceBuilder()
    decoder, backend = _decoder(
        ["   ", "  ", " "], output_format="logic-form", retry_count=2
    )

    with pytest.raises(ConstraintUnsatisfied) as excinfo:
        decoder.decode(make_context(), make_state(), builder=builder)

    assert excinfo.value.attempts == 3
    assert backend.call_count == 3
    assert (
        builder.trace.termination_reason == TerminationReason.CONSTRAINT_UNSATISFIED
    )
