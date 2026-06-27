"""Property-based test for the learned-vs-seeded marker trace round-trip.

Task 17.11 / **Property 5: Learned-vs-seeded marker is preserved across trace
round-trip.**

For any :class:`~nsr.models.trace.ProofTrace` whose steps carry an
``applied_rule_origin`` drawn from ``{SEEDED, LEARNED, None}`` (Req 14.5), exporting it
through the machine-readable serializers and re-parsing preserves the learned-vs-seeded
marker of every step. This exercises both the dict form
(``trace_from_dict(trace_to_dict(t))``) and the JSON form
(``trace_from_json(trace_to_json(t))``), and asserts full trace equality so no other
recorded field is disturbed by the marker.

**Validates: Requirements 14.5**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from nsr.models.enums import TerminationReason, ValidationStatus
from nsr.models.learning import RuleOrigin
from nsr.models.reasoning import SymbolicRepresentation
from nsr.models.trace import ProofStep, ProofTrace
from nsr.proof_trace_export import (
    trace_from_dict,
    trace_from_json,
    trace_to_dict,
    trace_to_json,
)

# --------------------------------------------------------------------------- #
# Leaf strategies
# --------------------------------------------------------------------------- #

# Bounded, finite-only text keeps the JSON form losslessly comparable.
_text = st.text(max_size=24)
_rule_ids = st.lists(st.text(min_size=1, max_size=8), max_size=4)

# The learned-vs-seeded marker: SEEDED, LEARNED, or None (unknown / no rule applied).
_rule_origins = st.one_of(st.none(), st.sampled_from(list(RuleOrigin)))


@st.composite
def representations(draw) -> SymbolicRepresentation:
    """A nested :class:`SymbolicRepresentation` (kept simple; marker is the focus)."""

    return SymbolicRepresentation(
        logic_form=draw(_text),
        predicates={},
        source_text=draw(_text),
    )


@st.composite
def proof_steps(draw) -> ProofStep:
    """A :class:`ProofStep` whose ``applied_rule_origin`` is a random SEEDED/LEARNED/None."""

    return ProofStep(
        sequence=draw(st.integers(min_value=0, max_value=10_000)),
        step_text=draw(_text),
        representation=draw(st.one_of(st.none(), representations())),
        status=draw(st.sampled_from(list(ValidationStatus))),
        applied_rule_id=draw(st.one_of(st.none(), st.text(min_size=1, max_size=8))),
        # The field under test: the learned-vs-seeded marker (Req 14.5).
        applied_rule_origin=draw(_rule_origins),
        violated_rule_ids=draw(_rule_ids),
        repair_attempts=[],
        translation_outcomes=[],
    )


@st.composite
def proof_traces(draw) -> ProofTrace:
    """A :class:`ProofTrace` whose every step carries a random learned-vs-seeded marker.

    At least one step is generated so the property exercises the per-step marker on a
    non-empty trace; the upper bound keeps examples small and fast.
    """

    return ProofTrace(
        steps=draw(st.lists(proof_steps(), min_size=1, max_size=6)),
        termination_reason=draw(
            st.one_of(st.none(), st.sampled_from(list(TerminationReason)))
        ),
        latency=None,
        error_record=None,
    )


# --------------------------------------------------------------------------- #
# Property 5: learned-vs-seeded marker round-trip
# --------------------------------------------------------------------------- #


@settings(max_examples=200)
@given(trace=proof_traces())
def test_marker_present_for_every_step(trace: ProofTrace) -> None:
    """Every step exposes an ``applied_rule_origin`` marker (SEEDED/LEARNED/None).

    **Validates: Requirements 14.5**
    """

    for step in trace.steps:
        assert step.applied_rule_origin is None or isinstance(
            step.applied_rule_origin, RuleOrigin
        )


@settings(max_examples=200)
@given(trace=proof_traces())
def test_dict_round_trip_preserves_marker(trace: ProofTrace) -> None:
    """Dict form preserves each step's learned-vs-seeded marker (and full equality).

    **Validates: Requirements 14.5**
    """

    restored = trace_from_dict(trace_to_dict(trace))

    assert [s.applied_rule_origin for s in restored.steps] == [
        s.applied_rule_origin for s in trace.steps
    ]
    assert restored == trace


@settings(max_examples=200)
@given(trace=proof_traces())
def test_json_round_trip_preserves_marker(trace: ProofTrace) -> None:
    """JSON form preserves each step's learned-vs-seeded marker (and full equality).

    **Validates: Requirements 14.5**
    """

    restored = trace_from_json(trace_to_json(trace))

    assert [s.applied_rule_origin for s in restored.steps] == [
        s.applied_rule_origin for s in trace.steps
    ]
    assert restored == trace
