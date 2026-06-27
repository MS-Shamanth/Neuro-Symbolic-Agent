"""Property test for the learned-rule cap (Task 17.9).

**Property 9: Promoted learned rules never exceed the cap**

For any set of promotable (corroborated, count >= threshold), mutually
non-contradicting Candidate_Rules and any ``max_learned_rules`` cap ``m``, a single
:meth:`RuleLearner.promote` pass promotes at most ``m`` Learned_Rules; and whenever the
candidate set would exceed ``m`` a ``cap-reached`` :class:`PromotionDecision` is recorded
and :attr:`PromotionResult.cap_reached` is set.

**Validates: Requirements 14.9**
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from nsr import RuleLearner, ValidationEngine
from nsr.models import (
    CandidateRule,
    LearnedRuleStore,
    ProductionRule,
    RuleProvenance,
    SymbolicRepresentation,
)


def _token(prefix: str, index: int) -> str:
    """A fixed-width token so no token is ever a substring of another.

    The Validation Engine matches IF/THEN terms by plain substring containment, so
    distinct candidates must use tokens where none contains another (otherwise an
    unrelated candidate's rule could spuriously apply to — and contradict over — a
    different candidate's witness). Zero-padding to a fixed width guarantees that two
    distinct indices never overlap as substrings, leaving the cap as the only promotion
    gate.
    """
    return f"{prefix}{index:04d}"


def _make_candidate(index: int, *, threshold: int) -> CandidateRule:
    """A promotable, self-accepting candidate over tokens disjoint from all others.

    The witness satisfies the candidate's own IF/THEN rule (both terms present), so the
    candidate accepts it; because every other candidate uses disjoint fixed-width tokens,
    no other rule is applicable to this witness and hence none can contradict it. The
    corroboration count is set at the threshold so the candidate clears the
    corroboration gate and the cap is the sole remaining constraint.
    """
    cond = _token("cond", index)
    act = _token("act", index)
    key = f"k{index:04d}"
    witness = SymbolicRepresentation(logic_form=f"{cond} {act}")
    return CandidateRule(
        rule=ProductionRule(
            rule_id=f"learned::{key}",
            condition=f"IF {cond}",
            action=f"THEN {act}",
        ),
        provenance=RuleProvenance(trace_ids=[f"t{index}"], step_ids=[0]),
        corroboration_count=threshold,
        normalized_key=key,
        witnesses=[witness],
    )


@given(
    n=st.integers(min_value=1, max_value=12),
    cap=st.integers(min_value=0, max_value=12),
)
def test_promoted_learned_rules_never_exceed_cap(n: int, cap: int) -> None:
    """Property 9: a promotion pass never promotes more than the cap, and records a
    cap-reached decision exactly when the candidate set would exceed it (Req 14.9)."""
    threshold = 1
    store = LearnedRuleStore()
    for i in range(n):
        candidate = _make_candidate(i, threshold=threshold)
        store.candidates[candidate.normalized_key] = candidate

    learner = RuleLearner(
        store,
        ValidationEngine(),
        corroboration_threshold=threshold,
        max_learned_rules=cap,
        seed=7,
    )

    result = learner.promote(procedural_memory=[])

    # Core property: promote at most the cap, and exactly min(n, cap) here since every
    # candidate is corroborated and mutually non-contradicting.
    assert len(result.promoted) == min(n, cap)
    assert len(result.promoted) <= cap
    assert len(learner.store.learned_rules) == min(n, cap)

    cap_reached_decisions = [d for d in result.decisions if d.reason == "cap-reached"]
    if n > cap:
        # The set exceeds the cap: cap-reached must be flagged and logged.
        assert result.cap_reached is True
        assert len(cap_reached_decisions) >= 1
        assert len(cap_reached_decisions) == n - cap
    else:
        # The whole set fits: nothing is capped.
        assert result.cap_reached is False
        assert cap_reached_decisions == []
