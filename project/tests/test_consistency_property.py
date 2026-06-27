"""Property-based tests for Reasoning_Consistency (Task 9.3).

Property 10: Reasoning_Consistency is the modal-answer fraction in [0, 1].

For any multiset of run answers with a repeated-run count of 2 or greater,
:func:`nsr.metrics_engine.compute_reasoning_consistency` returns the modal-answer
fraction ``modal_count / len(run_answers)``, which always lies in the closed interval
``[0.0, 1.0]``. When the repeated-run count is less than 2, the metric is left unset
and ``None`` is returned.

**Validates: Requirements 7.4, 7.5**
"""

from __future__ import annotations

from collections import Counter

from hypothesis import given
from hypothesis import strategies as st

from nsr.metrics_engine import compute_reasoning_consistency

# Run answers are drawn from a small alphabet so that ties and clear majorities both
# occur frequently, exercising the modal-answer logic across genuine multisets.
_ANSWERS = st.lists(
    st.sampled_from(["A", "B", "C", "D", "yes", "no"]),
    min_size=1,
    max_size=20,
)


@given(run_answers=_ANSWERS, repeated_run_count=st.integers(min_value=2, max_value=50))
def test_consistency_equals_modal_fraction_in_unit_interval(
    run_answers: list[str], repeated_run_count: int
) -> None:
    """With count >= 2 and non-empty answers, result is modal_count/len in [0, 1]."""
    result = compute_reasoning_consistency(run_answers, repeated_run_count)

    assert result is not None
    modal_count = Counter(run_answers).most_common(1)[0][1]
    expected = modal_count / len(run_answers)

    assert result == expected
    assert 0.0 <= result <= 1.0


@given(run_answers=_ANSWERS, repeated_run_count=st.integers(max_value=1))
def test_consistency_unset_when_count_below_two(
    run_answers: list[str], repeated_run_count: int
) -> None:
    """A repeated-run count below 2 always leaves consistency unset (None)."""
    assert compute_reasoning_consistency(run_answers, repeated_run_count) is None


@given(repeated_run_count=st.integers(min_value=2, max_value=50))
def test_consistency_unset_for_empty_answers(repeated_run_count: int) -> None:
    """Even with count >= 2, an empty answer multiset yields None (nothing to count)."""
    assert compute_reasoning_consistency([], repeated_run_count) is None


@given(
    answer=st.sampled_from(["A", "B", "C", "yes", "no"]),
    n=st.integers(min_value=2, max_value=20),
    repeated_run_count=st.integers(min_value=2, max_value=50),
)
def test_unanimous_answers_give_full_consistency(
    answer: str, n: int, repeated_run_count: int
) -> None:
    """When every run agrees, the modal fraction is exactly 1.0."""
    result = compute_reasoning_consistency([answer] * n, repeated_run_count)
    assert result == 1.0
