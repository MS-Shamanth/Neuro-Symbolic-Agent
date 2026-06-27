"""Smoke tests for the core data models defined in Task 1.

These verify the package is importable and that every enum and dataclass from the
design's Data Models section instantiates with the documented fields and defaults.
"""

from __future__ import annotations

import dataclasses

import nsr
from nsr.models import (
    CandidateRule,
    DatasetItem,
    DiscardedCandidate,
    Domain,
    ErrorRecord,
    Goal,
    LatencyRecord,
    LearnedRule,
    LearnedRuleStore,
    MethodMetrics,
    ProductionRule,
    ProofStep,
    ProofTrace,
    PromotionDecision,
    PromotionResult,
    QueryMetrics,
    RepairAttempt,
    RuleOrigin,
    RuleProvenance,
    RunRecord,
    SubGoal,
    SymbolicRepresentation,
    SystemConfig,
    TerminationReason,
    ValidationStatus,
    VerifiedOutput,
    WorkingMemoryState,
)


def test_package_has_version():
    assert isinstance(nsr.__version__, str)
    assert nsr.__version__


def test_enum_string_values():
    # str-Enum members serialize to their documented string form.
    assert ValidationStatus.ACCEPTED.value == "accepted"
    assert ValidationStatus.REJECTED.value == "rejected"
    assert ValidationStatus.REPAIRED.value == "repaired"

    assert TerminationReason.GOAL_SATISFIED.value == "goal-satisfied"
    assert TerminationReason.CYCLE_LIMIT_REACHED.value == "cycle-limit-reached"
    assert TerminationReason.CONSTRAINT_UNSATISFIED.value == "constraint-unsatisfied"
    assert TerminationReason.REPAIR_EXHAUSTED.value == "repair-exhausted"
    assert TerminationReason.COMPONENT_ERROR.value == "component-error"

    assert Domain.MATH.value == "mathematical-reasoning"
    assert {d.value for d in Domain} == {
        "mathematical-reasoning",
        "commonsense-reasoning",
        "multi-hop-reasoning",
        "science-reasoning",
        "logical-puzzles",
        "legal-question-answering",
    }


def test_goal_and_subgoal_defaults():
    sub = SubGoal(description="prove lemma")
    assert sub.satisfied is False

    goal = Goal(description="solve problem")
    assert goal.sub_goals == []
    assert goal.satisfied is False

    goal2 = Goal(description="solve", sub_goals=[sub])
    assert goal2.sub_goals == [sub]


def test_symbolic_representation_and_rule():
    rep = SymbolicRepresentation(logic_form="add(2,2)=4")
    assert rep.predicates == {}
    assert rep.source_text == ""

    rule = ProductionRule(rule_id="R1", condition="IF a", action="THEN b")
    assert rule.rule_id == "R1"


def test_working_memory_state_defaults():
    state = WorkingMemoryState(goal_buffer=Goal(description="g"))
    assert state.declarative_memory == []
    assert state.procedural_memory == []
    assert state.imaginal_buffer is None


def test_proof_trace_structure():
    rep = SymbolicRepresentation(logic_form="x")
    attempt = RepairAttempt(attempt_index=0, rejected_step=rep)
    assert attempt.violated_rule_ids == []
    assert attempt.repaired_step is None

    step = ProofStep(
        sequence=1,
        step_text="2+2=4",
        representation=rep,
        status=ValidationStatus.ACCEPTED,
    )
    assert step.applied_rule_id is None
    assert step.applied_rule_origin is None
    assert step.violated_rule_ids == []
    assert step.repair_attempts == []
    assert step.translation_outcomes == []

    trace = ProofTrace(steps=[step])
    assert trace.termination_reason is None
    assert trace.latency is None
    assert trace.error_record is None


def test_latency_and_error_and_output():
    latency = LatencyRecord(pipeline_ms=10.0, system2_ms=4.0, llm_ms=6.0)
    assert latency.latency_budget_exceeded is False

    err = ErrorRecord(failed_component="LLM", reason="timeout")
    assert err.failed_component == "LLM"

    output = VerifiedOutput(
        final_answer="4",
        proof_trace=ProofTrace(),
        faithfulness_score=1.0,
    )
    assert output.final_answer == "4"


def test_metrics_models():
    qm = QueryMetrics(faithfulness_score=0.5, step_hallucination_rate=0.5)
    assert qm.reasoning_consistency is None

    mm = MethodMetrics(
        method="nsr",
        final_answer_accuracy=0.9,
        step_hallucination_rate=0.1,
        faithfulness_score=0.9,
        latency_overhead_ms=50.0,
        mean_latency_ms=120.0,
        p95_latency_ms=200.0,
    )
    assert mm.reasoning_consistency is None


def test_dataset_item():
    item = DatasetItem(
        item_id="m-001",
        query="What is 2+2?",
        ground_truth="4",
        domain=Domain.MATH,
    )
    assert item.domain is Domain.MATH


def test_system_config_defaults_and_run_record():
    cfg = SystemConfig(
        max_cycle_limit=10,
        repair_attempt_limit=3,
        retry_count=2,
        llm_selection="hosted",
        output_format="json",
        conflict_resolution_policy="priority",
        generation_timeout_ms=30000,
    )
    assert cfg.repeated_run_count == 1
    assert cfg.latency_budget_ms is None
    assert cfg.random_seed is None
    # Adaptive rule-learning fields default to backward-compatible values (Req 14.8).
    assert cfg.rule_learning_enabled is False
    assert cfg.corroboration_threshold == 2
    assert cfg.max_learned_rules == 64

    record = RunRecord(
        config=cfg,
        dataset_ids=["math-v1"],
        model_id="gpt-x",
        seed=42,
    )
    assert record.applied_defaults == {}
    # New rule-learning run-record fields default to empty/None (Req 14.6).
    assert record.learned_rules == []
    assert record.induction_seed is None
    assert record.corroboration_threshold is None
    assert record.promotion_decisions == []
    # Every dataclass is a real dataclass instance.
    assert dataclasses.is_dataclass(record)


def test_rule_origin_enum_values():
    assert RuleOrigin.SEEDED.value == "seeded"
    assert RuleOrigin.LEARNED.value == "learned"
    # str-Enum compares equal to its string value.
    assert RuleOrigin.LEARNED == "learned"


def test_rule_provenance_and_candidate_defaults():
    prov = RuleProvenance(trace_ids=["t1"], step_ids=[1])
    assert prov.induction_seed is None

    rule = ProductionRule(rule_id="L1", condition="IF a", action="THEN b")
    candidate = CandidateRule(rule=rule, provenance=prov)
    assert candidate.corroboration_count == 1
    assert candidate.normalized_key == ""


def test_learned_rule_and_discarded_defaults():
    prov = RuleProvenance(trace_ids=["t1"], step_ids=[1], induction_seed=7)
    rule = ProductionRule(rule_id="L1", condition="IF a", action="THEN b")

    learned = LearnedRule(rule=rule, provenance=prov)
    assert learned.origin is RuleOrigin.LEARNED

    candidate = CandidateRule(rule=rule, provenance=prov)
    discarded = DiscardedCandidate(candidate=candidate, conflicting_rule_id="R9")
    assert discarded.conflicting_rule_id == "R9"


def test_promotion_decision_and_result_defaults():
    decision = PromotionDecision(
        normalized_key="k", promoted=True, reason="corroborated"
    )
    assert decision.conflicting_rule_id is None

    result = PromotionResult()
    assert result.promoted == []
    assert result.discarded == []
    assert result.decisions == []
    assert result.cap_reached is False


def test_learned_rule_store_defaults():
    store = LearnedRuleStore()
    assert store.version == 1
    assert store.candidates == {}
    assert store.learned_rules == []


def test_proof_step_records_learned_origin():
    rep = SymbolicRepresentation(logic_form="x")
    step = ProofStep(
        sequence=1,
        step_text="2+2=4",
        representation=rep,
        status=ValidationStatus.ACCEPTED,
        applied_rule_id="L1",
        applied_rule_origin=RuleOrigin.LEARNED,
    )
    assert step.applied_rule_origin is RuleOrigin.LEARNED
