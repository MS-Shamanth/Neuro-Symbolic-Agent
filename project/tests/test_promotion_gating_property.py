"""Property-based test for promotion gating (Task 17.5).

Property 3: Promotion requires corroboration and no contradiction.

For any candidate corroborated across ``k`` independent successful traces under a
corroboration threshold ``t``, with an existing rule set ``R``, the candidate is
promoted to a Learned_Rule **if and only if** ``k >= t`` *and* it does not contradict
any rule in ``R``.

The test drives the *real* :meth:`RuleLearner.promote` and uses the *real*
:meth:`RuleLearner.contradicts` semantics (over the candidate's witness set, decided by
:class:`ValidationEngine`) as the ground-truth oracle for the contradiction half of the
gate. A single candidate is placed in the store so its promotion depends only on its
corroboration count and the supplied Procedural_Memory ``R`` -- isolating the gate from
the learned-rule cap (held far above 1) and from cross-candidate interactions.

**Validates: Requirements 14.3**
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

# A small token alphabet keeps conditions/actions and witness text overlapping enough
# that rules are frequently *applicable* and frequently *(un)satisfied*, so both the
# "contradicts" and "does-not-contradict" branches of the gate are exercised often.
_TOKENS = ["foo", "bar", "baz", "qux", "req", "alpha", "beta", "gamma"]


def _clause(keyword: str, terms: list[str]) -> str:
    """Compose an ``IF``/``THEN`` clause from conjunctive terms (bare keyword if empty)."""
    if not terms:
        return keyword
    return f"{keyword} " + " AND ".join(terms)


@st.composite
def _representations(draw: st.DrawFn) -> SymbolicRepresentation:
    """A witness representation whose searchable text is a subset of the token alphabet."""
    tokens = draw(st.lists(st.sampled_from(_TOKENS), max_size=5))
    return SymbolicRepresentation(
        logic_form=" ".join(tokens),
        predicates={},
        source_text=draw(st.text(max_size=8)),
    )


@st.composite
def _production_rules(draw: st.DrawFn, *, rule_id: str) -> ProductionRule:
    """An IF/THEN rule whose halves are drawn from the shared token alphabet."""
    condition = draw(st.lists(st.sampled_from(_TOKENS), max_size=3))
    action = draw(st.lists(st.sampled_from(_TOKENS), max_size=3))
    return ProductionRule(
        rule_id=rule_id,
        condition=_clause("IF", condition),
        action=_clause("THEN", action),
    )


@st.composite
def _candidates(draw: st.DrawFn, *, k: int) -> CandidateRule:
    """A single store candidate with corroboration count ``k`` and one witness."""
    condition = draw(st.lists(st.sampled_from(_TOKENS), max_size=3))
    action = draw(st.lists(st.sampled_from(_TOKENS), max_size=3))
    witness = draw(_representations())
    key = "cand-key"
    # Provenance trace ids mirror the corroboration count of distinct traces; promote
    # gates on corroboration_count directly, so we set it explicitly to ``k``.
    return CandidateRule(
        rule=ProductionRule(
            rule_id=f"learned::{key}",
            condition=_clause("IF", condition),
            action=_clause("THEN", action),
        ),
        provenance=RuleProvenance(
            trace_ids=[f"t{i}" for i in range(k)], step_ids=[0]
        ),
        corroboration_count=k,
        normalized_key=key,
        witnesses=[witness],
    )


@st.composite
def _scenarios(draw: st.DrawFn):
    """Draw (candidate, existing_rules R, threshold t) with k and t overlapping."""
    t = draw(st.integers(min_value=1, max_value=5))
    k = draw(st.integers(min_value=0, max_value=6))
    candidate = draw(_candidates(k=k))
    n_rules = draw(st.integers(min_value=0, max_value=4))
    existing = [
        draw(_production_rules(rule_id=f"seed-{i}")) for i in range(n_rules)
    ]
    return candidate, existing, t, k


@given(scenario=_scenarios())
def test_promotion_iff_corroborated_and_no_contradiction(scenario) -> None:
    """A single candidate is promoted iff (k >= t) AND it contradicts no rule in R."""
    candidate, existing_rules, threshold, k = scenario

    store = LearnedRuleStore()
    store.candidates[candidate.normalized_key] = candidate
    # Hold the cap far above one so the gate under test is corroboration + contradiction,
    # never the learned-rule cap (Req 14.9 is covered by a separate property).
    learner = RuleLearner(
        store,
        ValidationEngine(),
        corroboration_threshold=threshold,
        max_learned_rules=1000,
        seed=7,
    )

    # Ground-truth oracle for the contradiction half, using the real contradicts().
    contradicts_some_rule = any(
        learner.contradicts(candidate, rule) for rule in existing_rules
    )
    expected_promoted = (k >= threshold) and not contradicts_some_rule

    result = learner.promote(procedural_memory=existing_rules)

    was_promoted = any(
        lr.rule.rule_id == candidate.rule.rule_id for lr in result.promoted
    )
    assert was_promoted == expected_promoted

    # The single recorded decision must agree with the gate and name the reason.
    assert len(result.decisions) == 1
    decision = result.decisions[0]
    assert decision.promoted == expected_promoted
    if expected_promoted:
        assert decision.reason == "corroborated"
        # A promoted candidate becomes an active LearnedRule in the store.
        assert store.learned_rules and store.learned_rules[-1].rule == candidate.rule
    elif k < threshold:
        # Corroboration failed: gate stops at the threshold check regardless of R.
        assert decision.reason == "below-threshold"
    else:
        # Corroborated but contradicting: discarded with the conflicting rule logged.
        assert decision.reason == "contradiction"
        assert len(result.discarded) == 1
        assert result.discarded[0].conflicting_rule_id is not None


@given(scenario=_scenarios())
def test_below_threshold_never_promotes_regardless_of_contradiction(scenario) -> None:
    """When k < t the candidate is never promoted, independent of the rule set R."""
    candidate, existing_rules, threshold, k = scenario

    store = LearnedRuleStore()
    store.candidates[candidate.normalized_key] = candidate
    # Force the under-threshold regime by setting the threshold above k.
    learner = RuleLearner(
        store,
        ValidationEngine(),
        corroboration_threshold=k + 1,
        max_learned_rules=1000,
        seed=7,
    )

    result = learner.promote(procedural_memory=existing_rules)

    assert result.promoted == []
    assert store.learned_rules == []
    assert [d.reason for d in result.decisions] == ["below-threshold"]
