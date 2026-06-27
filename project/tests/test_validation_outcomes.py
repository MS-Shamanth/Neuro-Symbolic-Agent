"""Unit tests for validation outcomes (Task 6.4).

These complement ``test_validation_engine.py`` (Task 6.1, which already covers the
core accept/reject semantics and ``test_repair_coordinator.py`` (Task 6.2). The focus
here is the three outcomes named by Task 6.4:

1. **All-rules-satisfied acceptance** -- a step that satisfies *every* applicable
   production rule (with inapplicable rules present alongside) is accepted (Req 6.2).
2. **Partial-violation rejection** -- a step that satisfies some applicable rules but
   violates others is rejected, and *every* violated rule is recorded while the
   satisfied rules are not (Req 6.3).
3. **Re-validation of a repaired step** -- a corrected step, re-validated against
   *every* applicable production rule, yields the correct accepted/rejected outcome.
   This is exercised both directly through the :class:`ValidationEngine` and end-to-end
   through the :class:`RepairCoordinator` re-validation path (Req 6.5).
"""

from __future__ import annotations

import json

from nsr import ValidationEngine
from nsr.llm_component import LLMComponent, MockBackend
from nsr.models import (
    Goal,
    ProductionRule,
    SymbolicRepresentation,
    SystemConfig,
    ValidationStatus,
    WorkingMemoryState,
)
from nsr.proof_trace import ProofTraceBuilder
from nsr.repair_coordinator import RepairContext, RepairCoordinator, RepairTrigger
from nsr.translation_layer import TranslationLayer


# --------------------------------------------------------------------------- helpers


def _rep(logic_form: str = "", source_text: str = "", predicates=None):
    return SymbolicRepresentation(
        logic_form=logic_form,
        source_text=source_text,
        predicates=predicates or {},
    )


def _config(repair_attempt_limit: int = 3) -> SystemConfig:
    return SystemConfig(
        max_cycle_limit=10,
        repair_attempt_limit=repair_attempt_limit,
        retry_count=0,
        llm_selection="hosted",
        output_format="json",
        conflict_resolution_policy="priority",
        generation_timeout_ms=30000,
    )


def _make_llm(scripted_logic_forms):
    """An LLMComponent over a MockBackend scripted with JSON logic-form steps.

    Each string item is wrapped into ``{"logic_form": item}``; dict items are emitted
    verbatim as JSON.
    """
    script = []
    for item in scripted_logic_forms:
        if isinstance(item, dict):
            script.append(json.dumps(item))
        else:
            script.append(json.dumps({"logic_form": item}))
    backend = MockBackend(script)
    return LLMComponent(backend, _config()), backend


def _state(procedural_memory=None) -> WorkingMemoryState:
    return WorkingMemoryState(
        goal_buffer=Goal(description="solve the problem"),
        declarative_memory=[],
        procedural_memory=list(procedural_memory) if procedural_memory else [],
        imaginal_buffer=None,
    )


def _coordinator(llm, repair_attempt_limit):
    return RepairCoordinator(
        llm=llm,
        translation=TranslationLayer(),
        validation=ValidationEngine(),
        repair_attempt_limit=repair_attempt_limit,
    )


# ---------------------------------------------- 1. all-rules-satisfied acceptance (6.2)


def test_acceptance_requires_all_applicable_rules_satisfied():
    """A step is accepted only when it satisfies EVERY applicable rule (Req 6.2).

    Two rules are applicable (their conditions match) and both their actions hold, while
    a third rule is inapplicable. The applicable rules are all satisfied, so the step is
    accepted and records no violations.
    """
    engine = ValidationEngine()
    rules = [
        ProductionRule(rule_id="add", condition="IF sum", action="THEN total"),
        ProductionRule(rule_id="sign", condition="IF total", action="THEN positive"),
        # inapplicable: condition term absent from the step
        ProductionRule(rule_id="div", condition="IF quotient", action="THEN nonzero"),
    ]
    rep = _rep(
        logic_form="sum total positive",
        source_text="the running total is positive",
    )

    outcome = engine.validate(rep, rules)

    assert outcome.accepted is True
    assert outcome.status is ValidationStatus.ACCEPTED
    # Only the two matching rules are applicable; the inapplicable one is excluded.
    assert outcome.applicable_rule_ids == ["add", "sign"]
    assert outcome.violated_rule_ids == []
    assert outcome.violated_rules == []


def test_acceptance_holds_when_only_inapplicable_rules_present():
    """When no rule's condition matches, acceptance holds vacuously (Req 6.2)."""
    engine = ValidationEngine()
    rules = [
        ProductionRule(rule_id="div", condition="IF quotient", action="THEN nonzero"),
        ProductionRule(rule_id="mul", condition="IF product", action="THEN factored"),
    ]
    rep = _rep(logic_form="addition step", source_text="2 + 3 = 5")

    outcome = engine.validate(rep, rules)

    assert outcome.accepted is True
    assert outcome.applicable_rule_ids == []
    assert outcome.violated_rule_ids == []


# --------------------------------- 2. partial-violation rejection with recorded rules (6.3)


def test_partial_violation_records_only_the_violated_rules():
    """A mix of satisfied and violated applicable rules rejects, recording each
    violated rule and excluding the satisfied ones (Req 6.3)."""
    engine = ValidationEngine()
    rules = [
        # applicable + satisfied
        ProductionRule(rule_id="ok1", condition="IF sum", action="THEN total"),
        # applicable + violated (action token absent)
        ProductionRule(rule_id="bad1", condition="IF sum", action="THEN normalized"),
        # applicable + satisfied
        ProductionRule(rule_id="ok2", condition="IF total", action="THEN recorded"),
        # applicable + violated (action token absent)
        ProductionRule(rule_id="bad2", condition="IF total", action="THEN rounded"),
        # inapplicable
        ProductionRule(rule_id="na", condition="IF integral", action="THEN bounded"),
    ]
    rep = _rep(
        logic_form="sum total recorded",
        source_text="the sum total is recorded",
    )

    outcome = engine.validate(rep, rules)

    assert outcome.rejected is True
    assert outcome.status is ValidationStatus.REJECTED
    assert outcome.applicable_rule_ids == ["ok1", "bad1", "ok2", "bad2"]
    # EVERY violated applicable rule is recorded; satisfied rules are not (Req 6.3).
    assert outcome.violated_rule_ids == ["bad1", "bad2"]
    assert [r.rule_id for r in outcome.violated_rules] == ["bad1", "bad2"]
    # The violated rule objects are the actual rules (so repair can reference them).
    assert outcome.violated_rules[0].action == "THEN normalized"
    assert outcome.violated_rules[1].action == "THEN rounded"


def test_single_violation_among_many_satisfied_rejects():
    """One violated rule is enough to reject, and it is the only one recorded (Req 6.3)."""
    engine = ValidationEngine()
    rules = [
        ProductionRule(rule_id="a", condition="IF step", action="THEN grounded"),
        ProductionRule(rule_id="b", condition="IF step", action="THEN justified"),
        ProductionRule(rule_id="c", condition="IF step", action="THEN cited"),
    ]
    # "cited" is missing -> only rule "c" is violated.
    rep = _rep(logic_form="step grounded justified")

    outcome = engine.validate(rep, rules)

    assert outcome.rejected is True
    assert outcome.violated_rule_ids == ["c"]
    assert len(outcome.violated_rules) == 1


# ----------------------------------------- 3. re-validation of a repaired step (6.5)


def test_revalidation_of_repaired_step_directly_yields_accepted():
    """Re-validating a corrected step against ALL applicable rules accepts it (Req 6.5).

    The original step violates a rule; the repaired representation supplies the missing
    action token, so re-validation against the same applicable rules accepts it.
    """
    engine = ValidationEngine()
    rules = [
        ProductionRule(rule_id="r1", condition="IF sum", action="THEN total"),
        ProductionRule(rule_id="r2", condition="IF sum", action="THEN normalized"),
    ]

    original = _rep(logic_form="sum total")  # missing "normalized" -> rejected
    first = engine.validate(original, rules)
    assert first.rejected is True
    assert first.violated_rule_ids == ["r2"]

    # The repaired step now contains every required action token.
    repaired = _rep(logic_form="sum total normalized")
    revalidated = engine.validate(repaired, rules)

    assert revalidated.accepted is True
    assert revalidated.applicable_rule_ids == ["r1", "r2"]
    assert revalidated.violated_rule_ids == []


def test_revalidation_of_partially_repaired_step_still_rejected():
    """A repaired step that fixes only one violation is still rejected on re-validation,
    recording the remaining violated rule (Req 6.5, 6.3)."""
    engine = ValidationEngine()
    rules = [
        ProductionRule(rule_id="r1", condition="IF sum", action="THEN total"),
        ProductionRule(rule_id="r2", condition="IF sum", action="THEN normalized"),
        ProductionRule(rule_id="r3", condition="IF sum", action="THEN rounded"),
    ]

    original = _rep(logic_form="sum")  # both r2/r3 (and r1) action tokens missing
    first = engine.validate(original, rules)
    assert first.violated_rule_ids == ["r1", "r2", "r3"]

    # Repair fixes r1 and r2 but not r3 -> still rejected, only r3 remains violated.
    repaired = _rep(logic_form="sum total normalized")
    revalidated = engine.validate(repaired, rules)

    assert revalidated.rejected is True
    assert revalidated.violated_rule_ids == ["r3"]


def test_revalidation_through_repair_coordinator_accepts_against_all_rules():
    """The Repair Coordinator re-validates the regenerated step against every applicable
    rule and accepts only when all are satisfied (Req 6.5).

    Two always-applicable rules each demand a distinct token. The first regenerated step
    supplies only one token (still rejected on re-validation); the second supplies both
    and is accepted.
    """
    require_alpha = ProductionRule(rule_id="R-alpha", condition="", action="THEN alpha")
    require_beta = ProductionRule(rule_id="R-beta", condition="", action="THEN beta")

    # Attempt 1: "alpha only" satisfies R-alpha but violates R-beta -> re-validated reject.
    # Attempt 2: "alpha beta" satisfies both -> re-validated accept.
    llm, backend = _make_llm(["alpha only", "alpha beta done"])
    coordinator = _coordinator(llm, repair_attempt_limit=4)

    builder = ProofTraceBuilder()
    step = builder.append_step(
        "offending step",
        representation=_rep(logic_form="neither token"),
        status=ValidationStatus.REJECTED,
    )
    context = RepairContext(
        trigger=RepairTrigger.REJECTION,
        state=_state([require_alpha, require_beta]),
        proof_step=step,
        rejected_representation=_rep(logic_form="neither token"),
        violated_rules=[require_alpha, require_beta],
    )

    outcome = coordinator.repair(context, builder=builder)

    assert outcome.succeeded is True
    # Re-validation rejected attempt 1 and accepted attempt 2.
    assert outcome.attempts_used == 2
    assert backend.call_count == 2
    assert "alpha" in outcome.accepted_representation.logic_form
    assert "beta" in outcome.accepted_representation.logic_form
    # Both attempts recorded; the second carries the accepted repaired step (Req 8.3).
    assert len(step.repair_attempts) == 2
    assert step.repair_attempts[1].repaired_step is not None


def test_revalidation_confirms_repaired_step_satisfies_previously_violated_rule():
    """An independent re-validation of the accepted repaired step confirms the
    previously violated rule is now satisfied (Req 6.5)."""
    require_alpha = ProductionRule(rule_id="R-alpha", condition="", action="THEN alpha")
    require_beta = ProductionRule(rule_id="R-beta", condition="", action="THEN beta")

    llm, _ = _make_llm(["alpha beta combined"])
    coordinator = _coordinator(llm, repair_attempt_limit=2)

    builder = ProofTraceBuilder()
    step = builder.append_step(
        "offending step",
        representation=_rep(logic_form="alpha"),  # beta missing -> was rejected
        status=ValidationStatus.REJECTED,
    )
    context = RepairContext(
        trigger=RepairTrigger.REJECTION,
        state=_state([require_alpha, require_beta]),
        proof_step=step,
        rejected_representation=_rep(logic_form="alpha"),
        violated_rules=[require_beta],
    )

    outcome = coordinator.repair(context, builder=builder)
    assert outcome.succeeded is True

    # Re-validate the accepted representation directly against all applicable rules.
    engine = ValidationEngine()
    confirm = engine.validate(
        outcome.accepted_representation, [require_alpha, require_beta]
    )
    assert confirm.accepted is True
    assert confirm.applicable_rule_ids == ["R-alpha", "R-beta"]
    assert confirm.violated_rule_ids == []
