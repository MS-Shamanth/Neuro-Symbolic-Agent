"""Property-based test for the lossless Learned_Rule_Store round-trip.

Task 17.8 / **Property 7: Learned rule store serialization round-trips losslessly.**

For any generated :class:`~nsr.models.learning.LearnedRuleStore`, serializing it through
``store_to_dict`` and re-parsing through ``store_from_dict`` yields a store equal to the
original in every recorded field: the version identifier, every candidate (its
``ProductionRule`` IF/THEN form, ``RuleProvenance`` trace_ids/step_ids/induction_seed,
corroboration count, normalized key, and witness ``SymbolicRepresentation`` list), and
every learned rule with its ``RuleOrigin`` marker. The serialized dict is also asserted
to be JSON-safe (``json.dumps``-able), matching how the store is persisted durably.

**Validates: Requirements 14.7**
"""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from nsr.models.learning import (
    CandidateRule,
    LearnedRule,
    LearnedRuleStore,
    RuleOrigin,
    RuleProvenance,
    store_from_dict,
    store_to_dict,
)
from nsr.models.reasoning import ProductionRule, SymbolicRepresentation

# --------------------------------------------------------------------------- #
# Leaf strategies
# --------------------------------------------------------------------------- #

# Bounded, finite-only text keeps the JSON form losslessly comparable: the dict is
# json.dumps-able and survives a round-trip with equality intact.
_text = st.text(max_size=24)

# JSON-compatible scalar values that survive json.dumps/json.loads and compare equal
# afterward. Floats are finite to avoid NaN/inf equality pitfalls.
_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**6), max_value=10**6),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    st.text(max_size=16),
)


def _json_values(max_leaves: int = 6):
    """Recursive JSON-compatible values (scalars, lists, string-keyed dicts).

    Dict keys are strings because JSON only supports string keys; this mirrors the
    structured ``predicates`` schema the real pipeline emits, so generated values are
    representative and round-trip cleanly.
    """

    return st.recursive(
        _json_scalars,
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.dictionaries(st.text(max_size=8), children, max_size=4),
        ),
        max_leaves=max_leaves,
    )


_predicates = st.dictionaries(st.text(max_size=8), _json_values(), max_size=4)


# --------------------------------------------------------------------------- #
# Composite strategies for the nested learned-store records
# --------------------------------------------------------------------------- #


@st.composite
def production_rules(draw) -> ProductionRule:
    """A :class:`ProductionRule` in the shared IF/THEN string form."""

    return ProductionRule(
        rule_id=draw(_text),
        condition=draw(_text),
        action=draw(_text),
    )


@st.composite
def representations(draw) -> SymbolicRepresentation:
    """A witness :class:`SymbolicRepresentation` with arbitrary structured predicates."""

    return SymbolicRepresentation(
        logic_form=draw(_text),
        predicates=draw(_predicates),
        source_text=draw(_text),
    )


@st.composite
def provenances(draw) -> RuleProvenance:
    """A :class:`RuleProvenance` with random trace ids, step ids, and induction seed."""

    return RuleProvenance(
        trace_ids=draw(st.lists(st.text(min_size=1, max_size=8), max_size=4)),
        step_ids=draw(st.lists(st.integers(min_value=0, max_value=10_000), max_size=4)),
        induction_seed=draw(
            st.one_of(st.none(), st.integers(min_value=0, max_value=2**31 - 1))
        ),
    )


@st.composite
def candidate_rules(draw) -> CandidateRule:
    """A :class:`CandidateRule` with provenance, corroboration, key, and witnesses."""

    return CandidateRule(
        rule=draw(production_rules()),
        provenance=draw(provenances()),
        corroboration_count=draw(st.integers(min_value=1, max_value=1000)),
        normalized_key=draw(_text),
        witnesses=draw(st.lists(representations(), max_size=4)),
    )


@st.composite
def learned_rules(draw) -> LearnedRule:
    """A :class:`LearnedRule` spanning both ``RuleOrigin`` markers."""

    return LearnedRule(
        rule=draw(production_rules()),
        provenance=draw(provenances()),
        origin=draw(st.sampled_from(list(RuleOrigin))),
    )


@st.composite
def learned_rule_stores(draw) -> LearnedRuleStore:
    """An arbitrary :class:`LearnedRuleStore` exercising every recorded field.

    A random version, a dict of candidates keyed by string, and a list of learned rules
    covering both seeded and learned origins.
    """

    candidates = draw(
        st.dictionaries(
            st.text(min_size=1, max_size=12),
            candidate_rules(),
            max_size=5,
        )
    )
    return LearnedRuleStore(
        version=draw(st.integers(min_value=0, max_value=1000)),
        candidates=candidates,
        learned_rules=draw(st.lists(learned_rules(), max_size=5)),
    )


# --------------------------------------------------------------------------- #
# Property 7: lossless serialization round-trip
# --------------------------------------------------------------------------- #


@settings(max_examples=200)
@given(store=learned_rule_stores())
def test_store_round_trip_is_lossless(store: LearnedRuleStore) -> None:
    """Property 7: ``store_from_dict(store_to_dict(store)) == store``.

    Preserves the version identifier, all candidates, corroboration counts,
    learned/seeded markers, and every provenance and witness field.

    **Validates: Requirements 14.7**
    """

    assert store_from_dict(store_to_dict(store)) == store


@settings(max_examples=200)
@given(store=learned_rule_stores())
def test_store_dict_is_json_safe(store: LearnedRuleStore) -> None:
    """Property 7 (JSON safety): the serialized dict is ``json.dumps``-able and the
    store survives a full JSON round-trip equal to the original.

    **Validates: Requirements 14.7**
    """

    encoded = json.dumps(store_to_dict(store))
    assert store_from_dict(json.loads(encoded)) == store
