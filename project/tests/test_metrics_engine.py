"""Unit tests for the Metrics Engine (Task 9.1).

Covers Faithfulness_Score, Step_Level_Hallucination_Rate, and Reasoning_Consistency
against specific examples and edge cases described in Requirements 7.1-7.5.
"""

from __future__ import annotations

from nsr.metrics_engine import (
    compute_faithfulness_score,
    compute_query_metrics,
    compute_reasoning_consistency,
    compute_step_hallucination_rate,
)
from nsr.models import ProofStep, ProofTrace, ValidationStatus


def _step(sequence: int, status: ValidationStatus) -> ProofStep:
    return ProofStep(
        sequence=sequence,
        step_text=f"step {sequence}",
        representation=None,
        status=status,
    )


def _trace(*statuses: ValidationStatus) -> ProofTrace:
    return ProofTrace(
        steps=[_step(i, status) for i, status in enumerate(statuses)],
    )


# --- Faithfulness_Score (Req 7.1, 7.2) ---------------------------------------


def test_faithfulness_empty_trace_is_zero():
    assert compute_faithfulness_score(ProofTrace()) == 0.0


def test_faithfulness_all_accepted_is_one():
    trace = _trace(ValidationStatus.ACCEPTED, ValidationStatus.ACCEPTED)
    assert compute_faithfulness_score(trace) == 1.0


def test_faithfulness_is_accepted_over_total():
    trace = _trace(
        ValidationStatus.ACCEPTED,
        ValidationStatus.REJECTED,
        ValidationStatus.ACCEPTED,
        ValidationStatus.REPAIRED,
    )
    assert compute_faithfulness_score(trace) == 0.5


def test_faithfulness_repaired_does_not_count_as_accepted():
    trace = _trace(ValidationStatus.REPAIRED, ValidationStatus.REPAIRED)
    assert compute_faithfulness_score(trace) == 0.0


# --- Step_Level_Hallucination_Rate (Req 7.3) ---------------------------------


def test_hallucination_empty_trace_is_zero():
    assert compute_step_hallucination_rate(ProofTrace()) == 0.0


def test_hallucination_is_rejected_over_total():
    trace = _trace(
        ValidationStatus.ACCEPTED,
        ValidationStatus.REJECTED,
        ValidationStatus.REJECTED,
        ValidationStatus.REPAIRED,
    )
    assert compute_step_hallucination_rate(trace) == 0.5


def test_faithfulness_and_hallucination_sum_excludes_repaired():
    trace = _trace(
        ValidationStatus.ACCEPTED,
        ValidationStatus.REJECTED,
        ValidationStatus.REPAIRED,
    )
    # 1 accepted, 1 rejected, 1 repaired out of 3
    assert compute_faithfulness_score(trace) == 1 / 3
    assert compute_step_hallucination_rate(trace) == 1 / 3


# --- Reasoning_Consistency (Req 7.4, 7.5) ------------------------------------


def test_consistency_unset_when_run_count_below_two():
    assert compute_reasoning_consistency(["a"], repeated_run_count=1) is None
    assert compute_reasoning_consistency(["a", "b"], repeated_run_count=0) is None


def test_consistency_all_agree_is_one():
    answers = ["42", "42", "42"]
    assert compute_reasoning_consistency(answers, repeated_run_count=3) == 1.0


def test_consistency_is_modal_fraction():
    answers = ["42", "42", "7"]
    assert compute_reasoning_consistency(answers, repeated_run_count=3) == 2 / 3


def test_consistency_empty_answers_is_none():
    assert compute_reasoning_consistency([], repeated_run_count=3) is None


# --- compute_query_metrics bundle --------------------------------------------


def test_query_metrics_without_repeated_runs_leaves_consistency_unset():
    trace = _trace(ValidationStatus.ACCEPTED, ValidationStatus.REJECTED)
    metrics = compute_query_metrics(trace)
    assert metrics.faithfulness_score == 0.5
    assert metrics.step_hallucination_rate == 0.5
    assert metrics.reasoning_consistency is None


def test_query_metrics_with_repeated_runs_sets_consistency():
    trace = _trace(ValidationStatus.ACCEPTED)
    metrics = compute_query_metrics(
        trace,
        run_answers=["x", "x", "y"],
        repeated_run_count=3,
    )
    assert metrics.faithfulness_score == 1.0
    assert metrics.step_hallucination_rate == 0.0
    assert metrics.reasoning_consistency == 2 / 3
