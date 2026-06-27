"""Unit tests for the Repair Coordinator (Task 6.2).

These cover the shared repair sub-loop driven for the three repair-triggering
outcomes -- rejection, untranslatable, and no-rule-matched -- exercising successful
repair, repair-attempt counting, prompt construction that references the offending
constraints, and ``repair-exhausted`` termination when the configured limit is reached
without acceptance (Req 6.4, 6.5, 6.6).

The dedicated property test for the repair attempt bound is Task 6.3; these are
example-based unit tests.
"""

from __future__ import annotations

import json

import pytest

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


# --------------------------------------------------------------------------- helpers


def make_config(repair_attempt_limit: int = 3) -> SystemConfig:
    """A minimal valid config; only the repair limit and LLM fields matter here."""
    return SystemConfig(
        max_cycle_limit=10,
        repair_attempt_limit=repair_attempt_limit,
        retry_count=0,
        llm_selection="hosted",
        output_format="json",
        conflict_resolution_policy="priority",
        generation_timeout_ms=30000,
    )


def make_llm(scripted_logic_forms):
    """Build an LLMComponent over a MockBackend scripted with JSON logic-form steps.

    Each item may be a string logic form (wrapped into a ``{"logic_form": ...}`` JSON
    object), a raw string (used verbatim), or an exception instance to raise.
    """
    script = []
    for item in scripted_logic_forms:
        if isinstance(item, BaseException):
            script.append(item)
        elif isinstance(item, dict):
            script.append(json.dumps(item))
        else:
            script.append(json.dumps({"logic_form": item}))
    backend = MockBackend(script)
    component = LLMComponent(backend, make_config())
    return component, backend


def make_state(procedural_memory=None) -> WorkingMemoryState:
    return WorkingMemoryState(
        goal_buffer=Goal(description="solve the problem"),
        declarative_memory=[],
        procedural_memory=list(procedural_memory) if procedural_memory else [],
        imaginal_buffer=None,
    )


def new_proof_step(builder: ProofTraceBuilder) -> ProofStep:
    return builder.append_step(
        "offending step",
        representation=SymbolicRepresentation(logic_form="bad"),
        status=ValidationStatus.REJECTED,
    )


def make_coordinator(llm, repair_attempt_limit):
    return RepairCoordinator(
        llm=llm,
        translation=TranslationLayer(),
        validation=ValidationEngine(),
        repair_attempt_limit=repair_attempt_limit,
    )


# A rule that is always applicable (empty condition) but demands the token "ok" in the
# step; any step lacking "ok" is therefore rejected.
REQUIRE_OK = ProductionRule(rule_id="R-ok", condition="", action="THEN ok")


# --------------------------------------------------------------------------- tests


def test_negative_repair_limit_rejected():
    llm, _ = make_llm(["ok"])
    with pytest.raises(ValueError):
        make_coordinator(llm, repair_attempt_limit=-1)


def test_repair_succeeds_on_first_attempt_for_rejection():
    # First regenerated step already satisfies the rule -> accepted on attempt 1.
    llm, backend = make_llm(["result ok"])
    coordinator = make_coordinator(llm, repair_attempt_limit=3)

    builder = ProofTraceBuilder()
    step = new_proof_step(builder)
    context = RepairContext(
        trigger=RepairTrigger.REJECTION,
        state=make_state([REQUIRE_OK]),
        proof_step=step,
        rejected_representation=SymbolicRepresentation(logic_form="result bad"),
        violated_rules=[REQUIRE_OK],
    )

    outcome = coordinator.repair(context, builder=builder)

    assert outcome.succeeded is True
    assert outcome.attempts_used == 1
    assert outcome.termination_reason is None
    assert outcome.accepted_representation is not None
    assert "ok" in outcome.accepted_representation.logic_form
    # Exactly one repair attempt recorded, carrying the repaired step (Req 8.3).
    assert len(step.repair_attempts) == 1
    assert step.repair_attempts[0].repaired_step is not None
    assert step.repair_attempts[0].violated_rule_ids == ["R-ok"]
    assert backend.call_count == 1


def test_repair_prompt_references_violated_rules():
    llm, backend = make_llm(["result ok"])
    coordinator = make_coordinator(llm, repair_attempt_limit=2)

    builder = ProofTraceBuilder()
    step = new_proof_step(builder)
    context = RepairContext(
        trigger=RepairTrigger.REJECTION,
        state=make_state([REQUIRE_OK]),
        proof_step=step,
        rejected_representation=SymbolicRepresentation(logic_form="result bad"),
        violated_rules=[REQUIRE_OK],
    )

    coordinator.repair(context, builder=builder)

    prompt = backend.calls[0][0]
    assert "REPAIR REQUIRED" in prompt
    assert "R-ok" in prompt  # the offending rule id is referenced (Req 6.4)


def test_repair_exhausted_when_limit_reached():
    # Every regenerated step keeps failing the rule -> never accepted.
    llm, backend = make_llm(["still bad", "still bad", "still bad", "still bad"])
    coordinator = make_coordinator(llm, repair_attempt_limit=2)

    builder = ProofTraceBuilder()
    step = new_proof_step(builder)
    context = RepairContext(
        trigger=RepairTrigger.REJECTION,
        state=make_state([REQUIRE_OK]),
        proof_step=step,
        rejected_representation=SymbolicRepresentation(logic_form="bad"),
        violated_rules=[REQUIRE_OK],
    )

    outcome = coordinator.repair(context, builder=builder)

    assert outcome.succeeded is False
    assert outcome.attempts_used == 2  # exactly the configured limit (Req 6.4)
    assert outcome.termination_reason == TerminationReason.REPAIR_EXHAUSTED
    assert builder.trace.termination_reason == TerminationReason.REPAIR_EXHAUSTED
    assert len(step.repair_attempts) == 2
    # The LLM was asked exactly twice, once per attempt.
    assert backend.call_count == 2


def test_zero_repair_limit_exhausts_immediately():
    llm, backend = make_llm(["ok"])
    coordinator = make_coordinator(llm, repair_attempt_limit=0)

    builder = ProofTraceBuilder()
    step = new_proof_step(builder)
    context = RepairContext(
        trigger=RepairTrigger.REJECTION,
        state=make_state([REQUIRE_OK]),
        proof_step=step,
        rejected_representation=SymbolicRepresentation(logic_form="bad"),
        violated_rules=[REQUIRE_OK],
    )

    outcome = coordinator.repair(context, builder=builder)

    assert outcome.succeeded is False
    assert outcome.attempts_used == 0
    assert outcome.termination_reason == TerminationReason.REPAIR_EXHAUSTED
    assert step.repair_attempts == []
    assert backend.call_count == 0  # no regeneration attempted


def test_repair_succeeds_after_one_rejected_attempt():
    # First regenerated step still fails, second satisfies the rule -> accepted on 2.
    llm, backend = make_llm(["still bad", "now ok"])
    coordinator = make_coordinator(llm, repair_attempt_limit=4)

    builder = ProofTraceBuilder()
    step = new_proof_step(builder)
    context = RepairContext(
        trigger=RepairTrigger.REJECTION,
        state=make_state([REQUIRE_OK]),
        proof_step=step,
        rejected_representation=SymbolicRepresentation(logic_form="bad"),
        violated_rules=[REQUIRE_OK],
    )

    outcome = coordinator.repair(context, builder=builder)

    assert outcome.succeeded is True
    assert outcome.attempts_used == 2
    assert len(step.repair_attempts) == 2
    # First attempt rejected (no repaired-then-accepted), second produced the accepted.
    assert step.repair_attempts[1].repaired_step is not None
    assert "ok" in outcome.accepted_representation.logic_form


def test_untranslatable_trigger_repairs_to_accepted():
    # No applicable rules -> any translatable step is accepted. The first regenerated
    # step has no logic_form (untranslatable), the second translates and is accepted.
    llm, backend = make_llm([{"no_logic": "x"}, "fine"])
    coordinator = make_coordinator(llm, repair_attempt_limit=4)

    builder = ProofTraceBuilder()
    step = new_proof_step(builder)
    context = RepairContext(
        trigger=RepairTrigger.UNTRANSLATABLE,
        state=make_state([]),  # no rules -> translatable step accepted
        proof_step=step,
        reason="candidate step has no logic_form",
    )

    outcome = coordinator.repair(context, builder=builder)

    assert outcome.succeeded is True
    assert outcome.attempts_used == 2
    # The first attempt recorded no repaired step (still untranslatable).
    assert step.repair_attempts[0].repaired_step is None
    # The second attempt produced the accepted, translatable step.
    assert step.repair_attempts[1].repaired_step is not None


def test_untranslatable_persists_until_exhausted():
    # Every regenerated step is untranslatable -> exhaustion with no repaired steps.
    llm, backend = make_llm([{"no_logic": 1}, {"no_logic": 2}, {"no_logic": 3}])
    coordinator = make_coordinator(llm, repair_attempt_limit=2)

    builder = ProofTraceBuilder()
    step = new_proof_step(builder)
    context = RepairContext(
        trigger=RepairTrigger.UNTRANSLATABLE,
        state=make_state([]),
        proof_step=step,
        reason="no logic form",
    )

    outcome = coordinator.repair(context, builder=builder)

    assert outcome.succeeded is False
    assert outcome.attempts_used == 2
    assert outcome.termination_reason == TerminationReason.REPAIR_EXHAUSTED
    assert all(a.repaired_step is None for a in step.repair_attempts)


def test_no_rule_matched_trigger_builds_prompt_and_repairs():
    llm, backend = make_llm(["anything"])
    coordinator = make_coordinator(llm, repair_attempt_limit=3)

    builder = ProofTraceBuilder()
    step = new_proof_step(builder)
    context = RepairContext(
        trigger=RepairTrigger.NO_RULE_MATCHED,
        state=make_state([]),  # no rules -> translatable step accepted
        proof_step=step,
        reason="no production rule matched the current state",
    )

    outcome = coordinator.repair(context, builder=builder)

    assert outcome.succeeded is True
    prompt = backend.calls[0][0]
    assert "No production rule applied" in prompt
    assert "no production rule matched the current state" in prompt


def test_recorded_attempts_never_exceed_limit():
    # Defensive check on the attempt bound (Req 6.4): with a large rejection streak the
    # number of recorded attempts equals the configured limit, no more.
    llm, _ = make_llm(["bad"] * 10)
    coordinator = make_coordinator(llm, repair_attempt_limit=5)

    builder = ProofTraceBuilder()
    step = new_proof_step(builder)
    context = RepairContext(
        trigger=RepairTrigger.REJECTION,
        state=make_state([REQUIRE_OK]),
        proof_step=step,
        rejected_representation=SymbolicRepresentation(logic_form="bad"),
        violated_rules=[REQUIRE_OK],
    )

    outcome = coordinator.repair(context, builder=builder)

    assert outcome.attempts_used == 5
    assert len(step.repair_attempts) == 5
