"""Property-based test for the repair attempt bound (Task 6.3).

**Property 8: Repair attempts never exceed the configured limit.**

For any repair limit ``N`` (0..bound) and any sequence of always-failing regenerated
steps, the recorded repair attempt count -- both ``len(proof_step.repair_attempts)`` and
``outcome.attempts_used`` -- is at most ``N``. When the regenerated steps never become
acceptable, the loop runs to exhaustion: the outcome reports a ``repair-exhausted``
termination with ``attempts_used == N`` and the same count of recorded attempts.

**Validates: Requirements 6.4, 6.6**
"""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from nsr.llm_component import LLMComponent, MockBackend
from nsr.models import (
    Goal,
    ProductionRule,
    ProofStep,
    SymbolicRepresentation,
    SystemConfig,
    TerminationReason,
    ValidationStatus,
    WorkingMemoryState,
)
from nsr.proof_trace import ProofTraceBuilder
from nsr.repair_coordinator import (
    RepairContext,
    RepairCoordinator,
    RepairTrigger,
)
from nsr.translation_layer import TranslationLayer
from nsr.validation_engine import ValidationEngine


# A rule that is always applicable (empty condition) but demands the token "ok" in the
# step; any step lacking "ok" is therefore rejected. Reused from the unit-test setup.
REQUIRE_OK = ProductionRule(rule_id="R-ok", condition="", action="THEN ok")


def _make_config() -> SystemConfig:
    return SystemConfig(
        max_cycle_limit=10,
        repair_attempt_limit=3,
        retry_count=0,
        llm_selection="hosted",
        output_format="json",
        conflict_resolution_policy="priority",
        generation_timeout_ms=30000,
    )


def _make_llm(scripted_logic_forms: list[str]) -> LLMComponent:
    """Build an LLMComponent over a MockBackend scripted with JSON logic-form steps."""
    script = [json.dumps({"logic_form": lf}) for lf in scripted_logic_forms]
    return LLMComponent(MockBackend(script), _make_config())


def _make_state() -> WorkingMemoryState:
    return WorkingMemoryState(
        goal_buffer=Goal(description="solve the problem"),
        declarative_memory=[],
        procedural_memory=[REQUIRE_OK],
        imaginal_buffer=None,
    )


def _new_proof_step(builder: ProofTraceBuilder) -> ProofStep:
    return builder.append_step(
        "offending step",
        representation=SymbolicRepresentation(logic_form="bad"),
        status=ValidationStatus.REJECTED,
    )


# Arbitrary repair limit N in 0..20 and an arbitrary-length sequence of regenerated
# steps. Each regenerated step is a logic form that never contains the required token
# "ok" (drawn from a pool of failing tokens), so every attempt is rejected -- driving the
# loop toward exhaustion regardless of the concrete strings produced.
_failing_tokens = st.sampled_from(["bad", "still bad", "nope", "wrong", "fail"])


@settings(max_examples=200, deadline=None)
@given(
    repair_limit=st.integers(min_value=0, max_value=20),
    rejections=st.lists(_failing_tokens, min_size=0, max_size=25),
)
def test_repair_attempts_never_exceed_limit(repair_limit: int, rejections: list[str]):
    """Property 8: recorded repair attempts are bounded by the configured limit.

    Validates: Requirements 6.4, 6.6
    """
    # Always script strictly more failing steps than the limit could ever consume, so the
    # backend never runs dry before the coordinator stops on its own bound. None of these
    # logic forms satisfy REQUIRE_OK, so every attempt is rejected.
    script = list(rejections) + ["bad"] * (repair_limit + 1)
    llm = _make_llm(script)
    coordinator = RepairCoordinator(
        llm=llm,
        translation=TranslationLayer(),
        validation=ValidationEngine(),
        repair_attempt_limit=repair_limit,
    )

    builder = ProofTraceBuilder()
    step = _new_proof_step(builder)
    context = RepairContext(
        trigger=RepairTrigger.REJECTION,
        state=_make_state(),
        proof_step=step,
        rejected_representation=SymbolicRepresentation(logic_form="bad"),
        violated_rules=[REQUIRE_OK],
    )

    outcome = coordinator.repair(context, builder=builder)

    # Bound: neither the recorded attempts nor the reported count may exceed the limit.
    assert outcome.attempts_used <= repair_limit
    assert len(step.repair_attempts) <= repair_limit
    assert len(step.repair_attempts) == outcome.attempts_used

    # On a never-accepting sequence the loop runs to exhaustion at exactly the limit.
    assert outcome.succeeded is False
    assert outcome.attempts_used == repair_limit
    assert len(step.repair_attempts) == repair_limit
    assert outcome.termination_reason == TerminationReason.REPAIR_EXHAUSTED
    assert builder.trace.termination_reason == TerminationReason.REPAIR_EXHAUSTED
