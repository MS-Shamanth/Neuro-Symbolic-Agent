"""Unit tests for RuleLearner.induce (Task 17.1, Req 14.1, 14.2).

These verify induction over goal-satisfied traces:

- candidates are generalized only from accepted (or accepted-after-repair) steps,
- each candidate records its provenance (trace id + accepted step sequences),
- non-goal-satisfied traces and traces with no accepted steps yield ``[]``,
- the induced IF/THEN candidate is evaluable by the unchanged ValidationEngine and
  accepts the step it was induced from, and
- structurally equivalent accepted steps collapse to a single candidate.
"""

from __future__ import annotations

from nsr import RuleLearner, ValidationEngine
from nsr.models import LearnedRuleStore, SymbolicRepresentation
from nsr.models.enums import TerminationReason, ValidationStatus
from nsr.models.trace import ProofStep, ProofTrace


def _rep(logic_form="", source_text="", predicates=None):
    return SymbolicRepresentation(
        logic_form=logic_form,
        source_text=source_text,
        predicates=predicates or {},
    )


def _step(sequence, *, status=ValidationStatus.ACCEPTED, rep=None):
    return ProofStep(
        sequence=sequence,
        step_text=f"step-{sequence}",
        representation=rep if rep is not None else _rep(logic_form=f"sum({sequence})"),
        status=status,
    )


def _learner():
    return RuleLearner(LearnedRuleStore(), ValidationEngine(), seed=7)


def _trace(steps, reason=TerminationReason.GOAL_SATISFIED):
    return ProofTrace(steps=list(steps), termination_reason=reason)


def test_induce_returns_empty_for_non_goal_satisfied_trace():
    learner = _learner()
    trace = _trace([_step(0)], reason=TerminationReason.CYCLE_LIMIT_REACHED)

    assert learner.induce(trace, trace_id="t1") == []


def test_induce_returns_empty_when_no_accepted_steps():
    learner = _learner()
    trace = _trace([_step(0, status=ValidationStatus.REJECTED)])

    assert learner.induce(trace, trace_id="t1") == []


def test_induce_records_provenance_for_accepted_step():
    learner = _learner()
    rep = _rep(logic_form="sum(2,3)=5", predicates={"operation": "add", "operands": [2, 3]})
    trace = _trace([_step(4, rep=rep)])

    candidates = learner.induce(trace, trace_id="trace-abc")

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.provenance.trace_ids == ["trace-abc"]
    assert candidate.provenance.step_ids == [4]
    assert candidate.provenance.induction_seed == 7
    assert candidate.corroboration_count == 1
    assert candidate.normalized_key


def test_induced_candidate_is_evaluable_and_accepts_its_source_step():
    """The candidate, used as the sole rule, accepts the step it generalized (Req 14.2)."""
    learner = _learner()
    engine = ValidationEngine()
    rep = _rep(logic_form="sum(2,3)=5", predicates={"operation": "add"})
    trace = _trace([_step(0, rep=rep)])

    candidate = learner.induce(trace, trace_id="t1")[0]
    outcome = engine.validate(rep, [candidate.rule])

    assert outcome.accepted


def test_induce_includes_accepted_after_repair_steps():
    learner = _learner()
    rep = _rep(logic_form="rel(x)", predicates={"k": 1})
    trace = _trace([_step(2, status=ValidationStatus.REPAIRED, rep=rep)])

    candidates = learner.induce(trace, trace_id="t1")

    assert len(candidates) == 1
    assert candidates[0].provenance.step_ids == [2]


def test_structurally_equivalent_steps_collapse_to_one_candidate():
    """Two accepted steps with the same generalization share one candidate (Req 14.3)."""
    learner = _learner()
    rep_a = _rep(logic_form="sum(2,3)=5", predicates={"operation": "add"})
    rep_b = _rep(logic_form="sum(4,7)=11", predicates={"operation": "add"})
    trace = _trace([_step(0, rep=rep_a), _step(1, rep=rep_b)])

    candidates = learner.induce(trace, trace_id="t1")

    assert len(candidates) == 1
    assert candidates[0].provenance.step_ids == [0, 1]


def test_rejected_steps_excluded_from_induction():
    learner = _learner()
    accepted_rep = _rep(logic_form="ok(a)", predicates={"p": 1})
    rejected_rep = _rep(logic_form="bad(b)", predicates={"q": 2})
    trace = _trace(
        [
            _step(0, rep=accepted_rep),
            _step(1, status=ValidationStatus.REJECTED, rep=rejected_rep),
        ]
    )

    candidates = learner.induce(trace, trace_id="t1")

    assert len(candidates) == 1
    assert candidates[0].provenance.step_ids == [0]
