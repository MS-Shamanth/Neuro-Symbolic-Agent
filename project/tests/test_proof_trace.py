"""Unit tests for the append-only Proof_Trace builder (Task 8.1).

These cover execution-order sequencing, the explicit no-rule-applied indicator,
per-attempt repair recording, and the latency breakdown including the
latency-budget-exceeded flag (Req 8.1, 8.2, 8.3, 11.1, 11.2, 11.4).
"""

from __future__ import annotations

import pytest

from nsr.models import (
    ProofStep,
    ProofTrace,
    SymbolicRepresentation,
    TerminationReason,
    ValidationStatus,
)
from nsr.proof_trace import (
    NO_RULE_APPLIED,
    ProofTraceBuilder,
    applied_rule_label,
)


def _rep(text: str = "x") -> SymbolicRepresentation:
    return SymbolicRepresentation(logic_form=text, source_text=text)


def test_new_builder_produces_empty_trace():
    builder = ProofTraceBuilder()
    assert isinstance(builder.trace, ProofTrace)
    assert builder.trace.steps == []
    assert builder.trace.latency is None
    assert builder.trace.termination_reason is None


def test_append_assigns_sequence_in_execution_order():
    builder = ProofTraceBuilder()
    s0 = builder.append_step("first", status=ValidationStatus.ACCEPTED)
    s1 = builder.append_step("second", status=ValidationStatus.REJECTED)
    s2 = builder.append_step("third", status=ValidationStatus.REPAIRED)

    assert [s.sequence for s in (s0, s1, s2)] == [0, 1, 2]
    # invariant: step at index i has sequence i, in execution order
    for i, step in enumerate(builder.trace.steps):
        assert step.sequence == i
    assert [s.step_text for s in builder.trace.steps] == ["first", "second", "third"]


def test_append_records_outcome_and_applied_rule():
    builder = ProofTraceBuilder()
    step = builder.append_step(
        "use rule R1",
        representation=_rep(),
        status=ValidationStatus.ACCEPTED,
        applied_rule_id="R1",
    )
    assert step.status is ValidationStatus.ACCEPTED
    assert step.applied_rule_id == "R1"
    assert applied_rule_label(step) == "R1"


def test_no_rule_applied_indicator():
    builder = ProofTraceBuilder()
    step = builder.append_step("no rule", status=ValidationStatus.ACCEPTED)
    assert step.applied_rule_id is None
    assert applied_rule_label(step) == NO_RULE_APPLIED


def test_violated_rule_ids_recorded_on_rejection():
    builder = ProofTraceBuilder()
    step = builder.append_step(
        "bad step",
        status=ValidationStatus.REJECTED,
        violated_rule_ids=["R2", "R3"],
    )
    assert step.violated_rule_ids == ["R2", "R3"]


def test_repair_attempts_recorded_in_execution_order():
    builder = ProofTraceBuilder()
    step = builder.append_step("step", status=ValidationStatus.REPAIRED)

    a0 = builder.record_repair_attempt(
        step,
        rejected_step=_rep("bad0"),
        violated_rule_ids=["R1"],
        repaired_step=None,
    )
    a1 = builder.record_repair_attempt(
        step,
        rejected_step=_rep("bad1"),
        violated_rule_ids=["R1"],
        repaired_step=_rep("good"),
    )

    assert [a.attempt_index for a in (a0, a1)] == [0, 1]
    assert step.repair_attempts[0].rejected_step.logic_form == "bad0"
    assert step.repair_attempts[1].repaired_step.logic_form == "good"
    assert step.repair_attempts[1].violated_rule_ids == ["R1"]


def test_record_repair_attempt_rejects_foreign_step():
    builder = ProofTraceBuilder()
    foreign = ProofStep(
        sequence=0,
        step_text="foreign",
        representation=None,
        status=ValidationStatus.REJECTED,
    )
    with pytest.raises(ValueError):
        builder.record_repair_attempt(foreign, rejected_step=_rep())


def test_record_latency_explicit_values():
    builder = ProofTraceBuilder()
    rec = builder.record_latency(120.0, system2_ms=40.0, llm_ms=70.0)
    assert rec.pipeline_ms == 120.0
    assert rec.system2_ms == 40.0
    assert rec.llm_ms == 70.0
    assert rec.latency_budget_exceeded is False
    assert builder.trace.latency is rec


def test_record_latency_uses_accumulated_values():
    builder = ProofTraceBuilder()
    builder.add_system2_latency(10.0)
    builder.add_system2_latency(15.0)
    builder.add_llm_latency(50.0)
    rec = builder.record_latency(100.0)
    assert rec.system2_ms == 25.0
    assert rec.llm_ms == 50.0


def test_latency_budget_exceeded_flag_set_when_over_budget():
    builder = ProofTraceBuilder(latency_budget_ms=30)
    builder.add_system2_latency(31.0)
    rec = builder.record_latency(200.0)
    assert rec.system2_ms == 31.0
    assert rec.latency_budget_exceeded is True


def test_latency_budget_not_exceeded_at_or_under_budget():
    builder = ProofTraceBuilder(latency_budget_ms=30)
    rec = builder.record_latency(200.0, system2_ms=30.0)
    assert rec.latency_budget_exceeded is False


def test_no_budget_means_never_exceeded():
    builder = ProofTraceBuilder()
    rec = builder.record_latency(200.0, system2_ms=999999.0)
    assert rec.latency_budget_exceeded is False


def test_negative_latency_rejected():
    builder = ProofTraceBuilder()
    with pytest.raises(ValueError):
        builder.add_system2_latency(-1.0)
    with pytest.raises(ValueError):
        builder.record_latency(-5.0)


def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        ProofTraceBuilder(latency_budget_ms=-1)


def test_set_termination_reason_and_error_record():
    builder = ProofTraceBuilder()
    builder.set_termination_reason(TerminationReason.GOAL_SATISFIED)
    assert builder.trace.termination_reason is TerminationReason.GOAL_SATISFIED

    err = builder.set_error_record("LLM", "unavailable")
    assert builder.trace.error_record is err
    assert err.failed_component == "LLM"
    assert err.reason == "unavailable"


def test_translation_outcome_recorded():
    builder = ProofTraceBuilder()
    step = builder.append_step("s", status=ValidationStatus.ACCEPTED)
    builder.add_translation_outcome(step, {"direction": "forward", "untranslatable": False})
    assert step.translation_outcomes == [
        {"direction": "forward", "untranslatable": False}
    ]
