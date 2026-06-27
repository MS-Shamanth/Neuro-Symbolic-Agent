"""Property-based test for lossless, pure-function trace visualization.

Task 18.3 / **Property 12: Visualization is a lossless pure function of the trace.**

*For any* :class:`~nsr.models.trace.ProofTrace`, :func:`~nsr.trace_visualizer.to_mermaid`
and :func:`~nsr.trace_visualizer.to_dot` do **not** mutate the trace, and the emitted
text contains every step's sequence position, its validation outcome, its applied
production rule id (or the explicit ``no-rule-applied`` indicator), and the termination
reason string.

**Validates: Requirements 15.2**
"""

from __future__ import annotations

import copy

from hypothesis import given, settings
from hypothesis import strategies as st

from nsr.models.enums import TerminationReason, ValidationStatus
from nsr.models.learning import RuleOrigin
from nsr.models.reasoning import SymbolicRepresentation
from nsr.models.trace import ProofStep, ProofTrace, RepairAttempt
from nsr.proof_trace import applied_rule_label
from nsr.trace_visualizer import _terminal_label, to_dot, to_mermaid

# --------------------------------------------------------------------------- #
# Leaf strategies
# --------------------------------------------------------------------------- #
#
# Rule ids and logic forms are drawn from an identifier-safe alphabet (no quotes or
# newlines) so the verbatim token-presence checks below stay reliable: the renderers
# escape ``"`` and newlines for Mermaid/DOT, which would otherwise mask a raw token
# without that ever being a real-world rule identifier.
_ident_alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
_rule_id = st.text(alphabet=_ident_alphabet, min_size=1, max_size=8)
_rule_ids = st.lists(_rule_id, max_size=3)
_text = st.text(max_size=16)


@st.composite
def representations(draw) -> SymbolicRepresentation:
    """A minimal :class:`SymbolicRepresentation` (content is irrelevant to rendering)."""

    return SymbolicRepresentation(
        logic_form=draw(_text),
        source_text=draw(_text),
    )


@st.composite
def repair_attempts(draw) -> RepairAttempt:
    """One :class:`RepairAttempt`; ``rejected_step`` is required on the dataclass."""

    return RepairAttempt(
        attempt_index=draw(st.integers(min_value=0, max_value=20)),
        rejected_step=draw(representations()),
        violated_rule_ids=draw(_rule_ids),
        repaired_step=draw(st.one_of(st.none(), representations())),
    )


@st.composite
def proof_steps(draw) -> ProofStep:
    """A :class:`ProofStep` spanning every status, present/absent rule id and origin."""

    return ProofStep(
        sequence=draw(st.integers(min_value=0, max_value=50)),
        step_text=draw(_text),
        representation=draw(st.one_of(st.none(), representations())),
        status=draw(st.sampled_from(list(ValidationStatus))),
        # applied_rule_id None -> the explicit no-rule-applied indicator must render.
        applied_rule_id=draw(st.one_of(st.none(), _rule_id)),
        applied_rule_origin=draw(
            st.one_of(st.none(), st.sampled_from(list(RuleOrigin)))
        ),
        violated_rule_ids=draw(_rule_ids),
        repair_attempts=draw(st.lists(repair_attempts(), max_size=3)),
    )


@st.composite
def proof_traces(draw) -> ProofTrace:
    """An arbitrary :class:`ProofTrace`: mixed statuses, rule ids, and termination reasons.

    Steps may be empty (the minimal-diagram case) up to several, with present/None
    applied rule ids and any of the termination reasons (or none recorded yet).
    """

    return ProofTrace(
        steps=draw(st.lists(proof_steps(), max_size=5)),
        termination_reason=draw(
            st.one_of(st.none(), st.sampled_from(list(TerminationReason)))
        ),
    )


# --------------------------------------------------------------------------- #
# Property 12: lossless pure-function visualization
# --------------------------------------------------------------------------- #


def _assert_lossless(rendered: str, trace: ProofTrace) -> None:
    """Assert ``rendered`` carries every losslessly-required token for ``trace``."""

    for step in trace.steps:
        # Sequence position (execution order).
        assert f"Step {step.sequence}" in rendered
        # Validation outcome.
        assert step.status.value in rendered
        # Applied rule id, or the explicit no-rule-applied indicator.
        assert applied_rule_label(step) in rendered

    # Termination reason string (rendered via the shared terminal label).
    assert _terminal_label(trace) in rendered


@settings(max_examples=200)
@given(trace=proof_traces())
def test_to_mermaid_is_lossless_and_pure(trace: ProofTrace) -> None:
    """``to_mermaid`` does not mutate the trace and emits every required token.

    **Validates: Requirements 15.2**
    """

    before = copy.deepcopy(trace)
    rendered = to_mermaid(trace)
    # Pure: the input trace is unchanged.
    assert trace == before
    _assert_lossless(rendered, trace)


@settings(max_examples=200)
@given(trace=proof_traces())
def test_to_dot_is_lossless_and_pure(trace: ProofTrace) -> None:
    """``to_dot`` does not mutate the trace and emits every required token.

    **Validates: Requirements 15.2**
    """

    before = copy.deepcopy(trace)
    rendered = to_dot(trace)
    # Pure: the input trace is unchanged.
    assert trace == before
    _assert_lossless(rendered, trace)
