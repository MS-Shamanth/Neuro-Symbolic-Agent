"""Property-based test for latency overhead computation (Task 14.3).

**Property 12: Latency overhead is the mean per-query latency difference**

For any per-query latency sets, the reported overhead for a non-LLM-only method equals
the mean of ``(method latency - LLM-only latency)`` taken over the *shared* query set --
the items both that method and the LLM-only baseline successfully evaluated.

**Validates: Requirements 9.5**

Requirement 9.5 states the Evaluation_Harness SHALL compute latency overhead for a method
as the mean per-query difference between that method's wall-clock latency and the
LLM_Only_Baseline latency over the same query set. The harness implements this in
:meth:`nsr.evaluation_harness.EvaluationHarness._latency_overhead`, which averages the
per-item differences over the items present in both the method's outcomes and the
LLM-only latency map. These tests generate arbitrary per-query latency sets for a method
and the LLM-only baseline over an overlapping query set and assert the reported overhead
equals the mean of the per-query differences over the shared items (and is ``0.0`` for
the LLM-only baseline itself and when the shared set is empty).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from nsr.evaluation_harness import (
    LLM_ONLY_METHOD_NAME,
    EvaluationHarness,
    ItemOutcome,
)


# --------------------------------------------------------------------- generators

# Finite, bounded latencies keep the floating-point arithmetic well-behaved while still
# spanning negative, zero, and positive per-query differences (a faster method yields a
# negative overhead, which the mean-of-differences definition must preserve).
_latencies = st.floats(
    min_value=0.0,
    max_value=1.0e6,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def latency_sets(draw: st.DrawFn) -> "tuple[list[ItemOutcome], dict[str, float]]":
    """Generate a method's per-item outcomes and an LLM-only latency map.

    A pool of unique query ids is drawn first; each id is then independently assigned to
    the method's outcome list, to the LLM-only latency map, or both, each with its own
    latency. This intelligently covers the full input space: items only the method
    evaluated, items only the baseline evaluated, and the shared items that actually feed
    the overhead average -- including the empty-overlap edge case.
    """
    ids = draw(
        st.lists(st.integers(min_value=0, max_value=999), min_size=0, max_size=30, unique=True)
    )

    outcomes: list[ItemOutcome] = []
    llm_latency_by_item: dict[str, float] = {}
    for i in ids:
        item_id = f"q{i}"
        in_method = draw(st.booleans())
        in_llm = draw(st.booleans())
        if in_method:
            outcomes.append(
                ItemOutcome(
                    item_id=item_id,
                    final_answer="",
                    correct=False,
                    latency_ms=draw(_latencies),
                )
            )
        if in_llm:
            llm_latency_by_item[item_id] = draw(_latencies)

    return outcomes, llm_latency_by_item


# ---------------------------------------------------------------------- properties


@settings(max_examples=400)
@given(data=latency_sets())
def test_overhead_is_mean_per_query_difference_over_shared_set(data):
    """Property 12: overhead == mean of (method - llm) over the shared query set.

    Validates: Requirements 9.5
    """
    outcomes, llm_latency_by_item = data
    method = "neuro-symbolic"

    overhead = EvaluationHarness._latency_overhead(
        method, outcomes, llm_latency_by_item
    )

    # The shared query set is exactly the method's outcomes whose item also appears in
    # the LLM-only latency map; the difference is averaged over precisely those items.
    diffs = [
        o.latency_ms - llm_latency_by_item[o.item_id]
        for o in outcomes
        if o.item_id in llm_latency_by_item
    ]

    if not diffs:
        # No shared items => no defined per-query difference => overhead is 0.0.
        assert overhead == 0.0
    else:
        expected = sum(diffs) / len(diffs)
        assert overhead == expected


@settings(max_examples=400)
@given(data=latency_sets())
def test_llm_only_baseline_has_zero_overhead(data):
    """The LLM-only baseline is the reference, so its own overhead is exactly 0.0.

    Validates: Requirements 9.5
    """
    outcomes, llm_latency_by_item = data

    overhead = EvaluationHarness._latency_overhead(
        LLM_ONLY_METHOD_NAME, outcomes, llm_latency_by_item
    )

    assert overhead == 0.0


@settings(max_examples=200)
@given(
    pairs=st.lists(
        st.tuples(_latencies, _latencies), min_size=1, max_size=30
    )
)
def test_overhead_matches_direct_mean_on_fully_shared_set(pairs):
    """On a fully shared query set, overhead equals the plain mean of the differences.

    Validates: Requirements 9.5
    """
    # Every query id is shared, so the shared set is the entire query set.
    outcomes = [
        ItemOutcome(
            item_id=f"q{i}",
            final_answer="",
            correct=False,
            latency_ms=method_lat,
        )
        for i, (method_lat, _llm_lat) in enumerate(pairs)
    ]
    llm_latency_by_item = {f"q{i}": llm_lat for i, (_m, llm_lat) in enumerate(pairs)}

    overhead = EvaluationHarness._latency_overhead(
        "tree-of-thoughts", outcomes, llm_latency_by_item
    )

    diffs = [m - llm for (m, llm) in pairs]
    assert overhead == sum(diffs) / len(diffs)
