"""Unit tests for the Proof_Trace exporters (Task 8.2).

Covers the machine-readable serializer/parser round-trip (Req 8.4) and the
human-readable rendering presenting each step, outcome, and applied rule in execution
order (Req 8.5).
"""

from __future__ import annotations

from nsr.models.enums import TerminationReason, ValidationStatus
from nsr.models.learning import RuleOrigin
from nsr.models.reasoning import SymbolicRepresentation
from nsr.models.trace import (
    ErrorRecord,
    LatencyRecord,
    ProofStep,
    ProofTrace,
    RepairAttempt,
)
from nsr.proof_trace import NO_RULE_APPLIED
from nsr.proof_trace_export import (
    render_trace,
    trace_from_dict,
    trace_from_json,
    trace_to_dict,
    trace_to_json,
)


def _sample_trace() -> ProofTrace:
    """Build a trace exercising every recorded field, including nested records."""

    rep1 = SymbolicRepresentation(
        logic_form="add(2,2)=4",
        predicates={"op": "add", "args": [2, 2], "result": 4},
        source_text="Two plus two is four.",
    )
    rejected = SymbolicRepresentation(
        logic_form="add(2,2)=5",
        predicates={"op": "add", "result": 5},
        source_text="Two plus two is five.",
    )
    repaired = SymbolicRepresentation(logic_form="add(2,2)=4")

    accepted_step = ProofStep(
        sequence=0,
        step_text="2 + 2 = 4",
        representation=rep1,
        status=ValidationStatus.ACCEPTED,
        applied_rule_id="R1",
        applied_rule_origin=RuleOrigin.SEEDED,
        translation_outcomes=[{"direction": "forward", "untranslatable": False}],
    )
    repaired_step = ProofStep(
        sequence=1,
        step_text="2 + 2 = 5 (later corrected)",
        representation=repaired,
        status=ValidationStatus.REPAIRED,
        applied_rule_id=None,  # renders as no-rule-applied
        applied_rule_origin=RuleOrigin.LEARNED,
        violated_rule_ids=["R2"],
        repair_attempts=[
            RepairAttempt(
                attempt_index=0,
                rejected_step=rejected,
                violated_rule_ids=["R2"],
                repaired_step=repaired,
            )
        ],
    )

    return ProofTrace(
        steps=[accepted_step, repaired_step],
        termination_reason=TerminationReason.GOAL_SATISFIED,
        latency=LatencyRecord(
            pipeline_ms=120.5,
            system2_ms=40.25,
            llm_ms=80.25,
            latency_budget_exceeded=True,
        ),
        error_record=ErrorRecord(failed_component="LLM", reason="timeout"),
    )


def test_dict_round_trip_is_lossless():
    trace = _sample_trace()
    assert trace_from_dict(trace_to_dict(trace)) == trace


def test_json_round_trip_is_lossless():
    trace = _sample_trace()
    assert trace_from_json(trace_to_json(trace)) == trace


def test_json_pretty_printed_round_trip():
    trace = _sample_trace()
    text = trace_to_json(trace, indent=2)
    assert "\n" in text  # indentation produces multi-line output
    assert trace_from_json(text) == trace


def test_empty_trace_round_trip():
    trace = ProofTrace()
    assert trace_from_dict(trace_to_dict(trace)) == trace


def test_dict_uses_string_enum_values():
    trace = _sample_trace()
    data = trace_to_dict(trace)
    assert data["termination_reason"] == "goal-satisfied"
    assert data["steps"][0]["status"] == "accepted"
    assert data["steps"][1]["status"] == "repaired"
    # None applied_rule_id is preserved as null, not the indicator string.
    assert data["steps"][1]["applied_rule_id"] is None


def test_nested_representation_predicates_preserved():
    trace = _sample_trace()
    restored = trace_from_dict(trace_to_dict(trace))
    rep = restored.steps[0].representation
    assert rep is not None
    assert rep.predicates == {"op": "add", "args": [2, 2], "result": 4}


def test_dict_serializes_applied_rule_origin_as_enum_value():
    trace = _sample_trace()
    data = trace_to_dict(trace)
    # The learned-vs-seeded marker is serialized as the plain enum value (Req 14.5).
    assert data["steps"][0]["applied_rule_origin"] == "seeded"
    assert data["steps"][1]["applied_rule_origin"] == "learned"


def test_applied_rule_origin_round_trips_losslessly():
    trace = _sample_trace()
    restored = trace_from_dict(trace_to_dict(trace))
    # Both SEEDED and LEARNED markers survive the round-trip (Req 14.5).
    assert restored.steps[0].applied_rule_origin is RuleOrigin.SEEDED
    assert restored.steps[1].applied_rule_origin is RuleOrigin.LEARNED
    assert restored == trace


def test_applied_rule_origin_round_trips_via_json():
    trace = _sample_trace()
    restored = trace_from_json(trace_to_json(trace))
    assert restored.steps[0].applied_rule_origin is RuleOrigin.SEEDED
    assert restored.steps[1].applied_rule_origin is RuleOrigin.LEARNED


def test_none_applied_rule_origin_round_trips():
    step = ProofStep(
        sequence=0,
        step_text="no marker",
        representation=None,
        status=ValidationStatus.ACCEPTED,
    )
    trace = ProofTrace(steps=[step])
    data = trace_to_dict(trace)
    assert data["steps"][0]["applied_rule_origin"] is None
    restored = trace_from_dict(data)
    assert restored.steps[0].applied_rule_origin is None
    assert restored == trace


def test_older_dict_missing_applied_rule_origin_parses_to_none():
    # Simulate an artifact produced before the marker key existed: the step dict
    # has no "applied_rule_origin" key at all. Parsing must tolerate the absence
    # and reconstruct the marker as None (backward compatibility, Req 14.5).
    legacy_step = {
        "sequence": 0,
        "step_text": "legacy step",
        "representation": None,
        "status": "accepted",
        "applied_rule_id": "R1",
        "violated_rule_ids": [],
        "repair_attempts": [],
        "translation_outcomes": [],
    }
    legacy_data = {
        "schema_version": 1,
        "steps": [legacy_step],
        "termination_reason": None,
        "latency": None,
        "error_record": None,
    }
    restored = trace_from_dict(legacy_data)
    assert restored.steps[0].applied_rule_origin is None
    assert restored.steps[0].applied_rule_id == "R1"


def test_repair_attempt_round_trip():
    trace = _sample_trace()
    restored = trace_from_dict(trace_to_dict(trace))
    attempt = restored.steps[1].repair_attempts[0]
    assert attempt.attempt_index == 0
    assert attempt.violated_rule_ids == ["R2"]
    assert attempt.rejected_step.logic_form == "add(2,2)=5"
    assert attempt.repaired_step is not None
    assert attempt.repaired_step.logic_form == "add(2,2)=4"


def test_render_presents_steps_in_execution_order():
    trace = _sample_trace()
    rendered = render_trace(trace)

    # Steps appear in sequence order.
    idx0 = rendered.index("Step 0:")
    idx1 = rendered.index("Step 1:")
    assert idx0 < idx1

    # Outcomes and applied rule ids are presented.
    assert "accepted" in rendered
    assert "[rule: R1]" in rendered
    # A step with no applied rule renders the explicit indicator.
    assert f"[rule: {NO_RULE_APPLIED}]" in rendered


def test_render_includes_repair_outcome_and_termination():
    trace = _sample_trace()
    rendered = render_trace(trace)
    assert "repair 0" in rendered
    assert "Termination: goal-satisfied" in rendered
    assert "budget exceeded" in rendered
    assert "Error: LLM - timeout" in rendered


def test_render_empty_trace():
    rendered = render_trace(ProofTrace())
    assert "no reasoning steps recorded" in rendered
