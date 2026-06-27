"""Property-based test for contradiction discard-and-log (Task 17.6).

**Property 4: Contradicting candidates are discarded and logged.**

For any candidate ``c`` and existing rule ``r`` such that *some* witness representation is
*accepted by ``c`` alone* but *rejected by ``r`` alone* (per
:meth:`~nsr.validation_engine.ValidationEngine.validate` semantics),
:meth:`~nsr.rule_learner.RuleLearner.promote` never promotes ``c`` and instead produces a
discard record naming ``r``'s ``rule_id`` plus a ``"contradiction"`` promotion decision
(Req 14.4).

Construction strategy
---------------------
Witnesses, candidates, and existing rules are built from a fixed pool of words none of
which is a substring of another, so the :class:`ValidationEngine`'s substring matching is
unambiguous. For each example:

- the witness's searchable text is exactly a non-empty set of pool words;
- the candidate's ``IF``/``THEN`` terms are drawn only from those witness words, so the
  candidate *accepts* the witness (every term is present);
- the existing rule's ``IF`` terms are drawn from the witness words (so it is
  *applicable*) while its ``THEN`` includes a ``missing`` pool word absent from the
  witness, so the rule is *violated* and *rejects* the witness.

The precondition (accepted-by-candidate-alone, rejected-by-existing-alone) is *verified*
inside the test via :class:`ValidationEngine` before the promotion assertions, so the
property is only asserted over inputs that genuinely satisfy it.

**Validates: Requirements 14.4**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from nsr import RuleLearner, ValidationEngine
from nsr.models import (
    CandidateRule,
    LearnedRuleStore,
    ProductionRule,
    RuleProvenance,
    SymbolicRepresentation,
)

# A pool of tokens, none of which is a substring of another, so ValidationEngine's
# substring-based term matching is exact and free of accidental overlaps.
_POOL = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
]


def _clause(keyword: str, terms: list[str]) -> str:
    """Compose an ``IF``/``THEN`` clause exactly as the controller/validator parse it."""
    if not terms:
        return keyword
    return f"{keyword} " + " AND ".join(terms)


@st.composite
def contradiction_cases(draw):
    """Draw a (witness, candidate, existing_rule) triple engineered to contradict.

    The candidate accepts the witness (its terms are a subset of the witness words) and
    the existing rule rejects it (applicable via witness words, but its action requires a
    word the witness lacks).
    """

    # Witness words: a non-empty proper subset of the pool, leaving at least one word
    # available to act as the "missing" token the existing rule will require.
    witness_tokens = draw(
        st.lists(
            st.sampled_from(_POOL),
            min_size=1,
            max_size=len(_POOL) - 1,
            unique=True,
        )
    )
    leftover = [w for w in _POOL if w not in witness_tokens]
    missing = draw(st.sampled_from(leftover))

    # Subsets of the witness words for the various clauses (any of them may be empty).
    def _subset():
        return draw(
            st.lists(st.sampled_from(witness_tokens), unique=True, max_size=len(witness_tokens))
        )

    cand_condition_terms = _subset()
    cand_action_terms = _subset()
    existing_condition_terms = _subset()
    # The existing rule's action requires the missing word (absent from the witness) plus
    # an optional subset of present words; the missing word forces a violation.
    existing_extra = _subset()
    existing_action_terms = existing_extra + [missing]

    witness = SymbolicRepresentation(
        logic_form=" ".join(witness_tokens),
        source_text="",
        predicates={},
    )

    candidate_rule = ProductionRule(
        rule_id="learned::candidate",
        condition=_clause("IF", cand_condition_terms),
        action=_clause("THEN", cand_action_terms),
    )
    existing_rule = ProductionRule(
        rule_id=draw(st.sampled_from(["seed-1", "seed-2", "learned::prior", "r-99"])),
        condition=_clause("IF", existing_condition_terms),
        action=_clause("THEN", existing_action_terms),
    )

    candidate = CandidateRule(
        rule=candidate_rule,
        provenance=RuleProvenance(trace_ids=["t1"], step_ids=[0]),
        corroboration_count=1,
        normalized_key="contradiction-key",
        witnesses=[witness],
    )

    return witness, candidate, existing_rule


@settings(max_examples=300)
@given(case=contradiction_cases())
def test_contradicting_candidate_is_discarded_and_logged(case) -> None:
    """A candidate contradicting an existing rule is never promoted, and the conflict is
    recorded as a :class:`DiscardedCandidate` plus a ``"contradiction"`` decision.

    **Validates: Requirements 14.4**
    """

    witness, candidate, existing = case
    validation = ValidationEngine()

    # Verify the contradiction precondition holds for this example: the witness is
    # accepted by the candidate alone and rejected by the existing rule alone.
    assert validation.validate(witness, [candidate.rule]).accepted
    assert validation.validate(witness, [existing]).rejected

    # Promote with a threshold of 1 so corroboration never masks the contradiction path.
    store = LearnedRuleStore()
    store.candidates[candidate.normalized_key] = candidate
    learner = RuleLearner(store, validation, corroboration_threshold=1, seed=7)

    result = learner.promote(procedural_memory=[existing])

    # The candidate is never promoted.
    assert result.promoted == []
    assert store.learned_rules == []
    promoted_keys = {lr.rule.rule_id for lr in result.promoted}
    assert candidate.rule.rule_id not in promoted_keys

    # A discard record names the conflicting existing rule's id.
    assert len(result.discarded) == 1
    assert result.discarded[0].conflicting_rule_id == existing.rule_id
    assert result.discarded[0].candidate.normalized_key == candidate.normalized_key

    # A "contradiction" decision is logged for this candidate, naming the conflict.
    contradiction_decisions = [
        d for d in result.decisions if d.reason == "contradiction"
    ]
    assert len(contradiction_decisions) == 1
    decision = contradiction_decisions[0]
    assert decision.promoted is False
    assert decision.normalized_key == candidate.normalized_key
    assert decision.conflicting_rule_id == existing.rule_id
