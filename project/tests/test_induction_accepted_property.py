"""Property-based test for induction over accepted steps only (Task 17.2).

**Property 1: Induction generalizes only accepted steps.**

For any goal-satisfied :class:`~nsr.models.trace.ProofTrace`, every
:class:`~nsr.models.learning.CandidateRule` produced by
:meth:`~nsr.rule_learner.RuleLearner.induce` is generalized *solely* from accepted (or
accepted-after-repair) :class:`~nsr.models.trace.ProofStep`\\s -- i.e. every provenance
``step_ids`` entry references only an accepted/repaired step. Conversely, induction over
a trace with no accepted steps, or one whose ``termination_reason`` is not
``goal-satisfied``, yields ``[]`` (Req 14.1).

The generators below build traces with a mix of step statuses (accepted, rejected,
repaired), optionally-``None`` representations, and every possible termination reason, so
the property is exercised across the full input space.

**Validates: Requirements 14.1**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from nsr import RuleLearner, ValidationEngine
from nsr.models import LearnedRuleStore, SymbolicRepresentation
from nsr.models.enums import TerminationReason, ValidationStatus
from nsr.models.trace import ProofStep, ProofTrace

# --------------------------------------------------------------------------- #
# Leaf strategies
# --------------------------------------------------------------------------- #

_text = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=16
)
_pred_keys = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=6
)
_pred_values = st.one_of(st.integers(-50, 50), st.text(max_size=6), st.booleans())

# A step that contributes to induction must be ACCEPTED or REPAIRED *and* carry a
# representation; the other statuses must never feed a candidate's provenance.
_ACCEPTING = (ValidationStatus.ACCEPTED, ValidationStatus.REPAIRED)


@st.composite
def representations(draw) -> SymbolicRepresentation:
    """A small :class:`SymbolicRepresentation` with random logic form and predicates."""

    return SymbolicRepresentation(
        logic_form=draw(_text),
        predicates=draw(
            st.dictionaries(_pred_keys, _pred_values, max_size=4)
        ),
        source_text=draw(_text),
    )


@st.composite
def proof_steps(draw, *, sequence: int) -> ProofStep:
    """A :class:`ProofStep` with a random status and an optionally-``None`` representation.

    The ``sequence`` is supplied by the trace generator so every step in a trace has a
    unique id; this lets the property map each provenance ``step_id`` back to exactly one
    step and check its status unambiguously.
    """

    return ProofStep(
        sequence=sequence,
        step_text=draw(_text),
        representation=draw(st.one_of(st.none(), representations())),
        status=draw(st.sampled_from(list(ValidationStatus))),
    )


@st.composite
def proof_traces(draw, *, reason=None) -> ProofTrace:
    """A :class:`ProofTrace` with uniquely-sequenced, mixed-status steps.

    ``reason`` pins the termination reason when provided; otherwise a random reason
    (including ``None``) is drawn so both the goal-satisfied and the non-goal-satisfied
    branches of induction are covered.
    """

    count = draw(st.integers(min_value=0, max_value=6))
    steps = [draw(proof_steps(sequence=i)) for i in range(count)]
    termination = (
        reason
        if reason is not None
        else draw(st.one_of(st.none(), st.sampled_from(list(TerminationReason))))
    )
    return ProofTrace(steps=steps, termination_reason=termination)


def _learner() -> RuleLearner:
    return RuleLearner(LearnedRuleStore(), ValidationEngine(), seed=7)


def _accepting_sequences(trace: ProofTrace) -> set[int]:
    """Sequences of steps eligible to feed induction (accepted/repaired, has a rep)."""

    return {
        step.sequence
        for step in trace.steps
        if step.status in _ACCEPTING and step.representation is not None
    }


# --------------------------------------------------------------------------- #
# Property 1: Induction generalizes only accepted steps
# --------------------------------------------------------------------------- #


@settings(max_examples=300)
@given(trace=proof_traces(reason=TerminationReason.GOAL_SATISFIED))
def test_candidates_generalize_only_accepted_steps(trace: ProofTrace) -> None:
    """Every induced candidate's provenance references only accepted/repaired steps.

    **Validates: Requirements 14.1**
    """

    learner = _learner()
    eligible = _accepting_sequences(trace)

    candidates = learner.induce(trace, trace_id="trace-1")

    referenced: set[int] = set()
    for candidate in candidates:
        # Provenance must name the originating trace and a non-empty set of steps.
        assert candidate.provenance.trace_ids == ["trace-1"]
        assert candidate.provenance.step_ids
        referenced.update(candidate.provenance.step_ids)

    # The core property: induction draws *solely* from accepted/repaired steps.
    assert referenced <= eligible

    # A trace with no accepted (usable) steps must yield no candidates at all.
    if not eligible:
        assert candidates == []
    else:
        # When eligible steps exist, induction must produce at least one candidate and
        # account for every eligible step in some candidate's provenance.
        assert candidates
        assert referenced == eligible


@settings(max_examples=300)
@given(
    trace=proof_traces(),
    reason=st.sampled_from(
        [r for r in TerminationReason if r != TerminationReason.GOAL_SATISFIED]
    ),
)
def test_non_goal_satisfied_traces_yield_no_candidates(
    trace: ProofTrace, reason: TerminationReason
) -> None:
    """Any non-goal-satisfied termination reason yields ``[]`` regardless of steps.

    **Validates: Requirements 14.1**
    """

    trace.termination_reason = reason

    assert _learner().induce(trace, trace_id="trace-1") == []


@settings(max_examples=300)
@given(
    statuses=st.lists(
        st.sampled_from(
            [ValidationStatus.REJECTED]
        ),
        max_size=6,
    )
)
def test_goal_satisfied_without_accepted_steps_yields_no_candidates(
    statuses: list[ValidationStatus],
) -> None:
    """A goal-satisfied trace with no accepted/repaired steps induces nothing.

    **Validates: Requirements 14.1**
    """

    steps = [
        ProofStep(
            sequence=i,
            step_text=f"step-{i}",
            representation=SymbolicRepresentation(logic_form=f"f({i})"),
            status=status,
        )
        for i, status in enumerate(statuses)
    ]
    trace = ProofTrace(
        steps=steps, termination_reason=TerminationReason.GOAL_SATISFIED
    )

    assert _learner().induce(trace, trace_id="trace-1") == []
