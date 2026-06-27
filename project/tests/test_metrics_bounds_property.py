"""Property-based tests for faithfulness and hallucination bounds (Task 9.2).

**Property 1: Faithfulness_Score equals accepted/total and lies in [0,1]**
**Property 2: Step_Level_Hallucination_Rate equals rejected/total and lies in [0,1]**

For any generated :class:`~nsr.models.trace.ProofTrace`, both per-query metrics fall in
the closed interval ``[0.0, 1.0]``; an empty trace yields a Faithfulness_Score of
exactly ``0.0``.

**Validates: Requirements 7.1, 7.2, 7.3**

Requirement 7.1 defines the Faithfulness_Score as ``accepted / total`` in ``[0.0, 1.0]``
for a non-empty trace; Requirement 7.2 fixes it to exactly ``0.0`` for an empty trace;
Requirement 7.3 defines the Step_Level_Hallucination_Rate as ``rejected / total`` in
``[0.0, 1.0]``. These tests generate arbitrary traces whose steps carry arbitrary
:class:`~nsr.models.enums.ValidationStatus` values (including empty traces) and assert
both the exact-ratio identities and the closed-interval bounds.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from nsr.metrics_engine import (
    compute_faithfulness_score,
    compute_step_hallucination_rate,
)
from nsr.models import ProofStep, ProofTrace, ValidationStatus


# --------------------------------------------------------------------- generators

# Every possible validation status is in play so the generated traces span the full
# input space the metrics operate over: ACCEPTED feeds faithfulness, REJECTED feeds the
# hallucination rate, and REPAIRED counts toward neither numerator.
_validation_statuses = st.sampled_from(list(ValidationStatus))


@st.composite
def proof_steps(draw: st.DrawFn, sequence: int) -> ProofStep:
    """Generate a single :class:`ProofStep` with an arbitrary validation status."""
    status = draw(_validation_statuses)
    return ProofStep(
        sequence=sequence,
        step_text=draw(st.text(max_size=20)),
        representation=None,
        status=status,
    )


@st.composite
def proof_traces(draw: st.DrawFn) -> ProofTrace:
    """Generate an arbitrary :class:`ProofTrace`, including empty traces.

    ``min_size=0`` ensures the empty-trace edge case (Req 7.2) is exercised; the upper
    bound keeps each example fast while still covering multi-step traces.
    """
    count = draw(st.integers(min_value=0, max_value=40))
    steps = [draw(proof_steps(sequence=i)) for i in range(count)]
    return ProofTrace(steps=steps)


# ---------------------------------------------------------------------- properties


@settings(max_examples=300)
@given(trace=proof_traces())
def test_faithfulness_equals_accepted_over_total_and_in_bounds(trace):
    """Property 1: Faithfulness_Score == accepted/total, within [0.0, 1.0].

    Validates: Requirements 7.1, 7.2
    """
    score = compute_faithfulness_score(trace)
    total = len(trace.steps)

    # Bound: the score always lies in the closed interval [0.0, 1.0] (Req 7.1).
    assert 0.0 <= score <= 1.0

    if total == 0:
        # Empty trace yields exactly 0.0 (Req 7.2).
        assert score == 0.0
    else:
        # Exact ratio identity: accepted / total (Req 7.1).
        accepted = sum(
            1 for step in trace.steps if step.status == ValidationStatus.ACCEPTED
        )
        assert score == accepted / total


@settings(max_examples=300)
@given(trace=proof_traces())
def test_hallucination_equals_rejected_over_total_and_in_bounds(trace):
    """Property 2: Step_Level_Hallucination_Rate == rejected/total, within [0.0, 1.0].

    Validates: Requirements 7.3
    """
    rate = compute_step_hallucination_rate(trace)
    total = len(trace.steps)

    # Bound: the rate always lies in the closed interval [0.0, 1.0] (Req 7.3).
    assert 0.0 <= rate <= 1.0

    if total == 0:
        # An empty trace has no rejected steps, so the rate is 0.0.
        assert rate == 0.0
    else:
        # Exact ratio identity: rejected / total (Req 7.3).
        rejected = sum(
            1 for step in trace.steps if step.status == ValidationStatus.REJECTED
        )
        assert rate == rejected / total


@settings(max_examples=300)
@given(trace=proof_traces())
def test_faithfulness_and_hallucination_never_exceed_one_together(trace):
    """Accepted and rejected fractions are disjoint, so their sum stays within [0,1].

    Validates: Requirements 7.1, 7.3
    """
    score = compute_faithfulness_score(trace)
    rate = compute_step_hallucination_rate(trace)
    # ACCEPTED and REJECTED are disjoint status partitions, so the two fractions can
    # never overlap; their sum is therefore bounded above by 1.0.
    assert 0.0 <= score + rate <= 1.0
