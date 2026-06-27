"""Property-based test for outcome-distinguished, rule-annotated trace nodes.

Task 18.4 / **Property 13: Nodes are outcome-distinguished and rule-annotated.**

For any :class:`~nsr.models.trace.ProofTrace`, each step node in the rendered Mermaid
and DOT output:

- carries the style class/shape corresponding to its
  :class:`~nsr.models.enums.ValidationStatus`, with accepted/rejected/repaired mutually
  distinct (Mermaid ``classDef`` class names; DOT node ``fillcolor``);
- has a label containing either the applied production rule id or the explicit
  ``no-rule-applied`` indicator;
- and, where a rule validated the step, the learned-vs-seeded marker derived from
  :attr:`ProofStep.applied_rule_origin`.

The test generates random traces spanning all three statuses, ``applied_rule_id``
present/absent, and ``applied_rule_origin`` in ``{SEEDED, LEARNED, None}``, then parses
the emitted Mermaid/DOT text and asserts the per-node styling and annotation contract.

**Validates: Requirements 15.4**
"""

from __future__ import annotations

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from nsr.models.enums import TerminationReason, ValidationStatus
from nsr.models.learning import RuleOrigin
from nsr.models.reasoning import SymbolicRepresentation
from nsr.models.trace import ProofStep, ProofTrace, RepairAttempt
from nsr.proof_trace import NO_RULE_APPLIED
from nsr.trace_visualizer import (
    _DOT_STATUS_STYLE,
    _MERMAID_CLASSDEF,
    _STATUS_CLASS,
    to_dot,
    to_mermaid,
)

# --------------------------------------------------------------------------- #
# Leaf strategies
# --------------------------------------------------------------------------- #

# Rule ids are constrained to an identifier-safe alphabet so neither the Mermaid nor the
# DOT escaping rewrites them. This keeps the rendered rule id a verbatim substring of the
# node label, which is exactly what the annotation assertions check.
_SAFE_ID = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_",
    min_size=1,
    max_size=8,
)

_ORIGINS = st.sampled_from([None, RuleOrigin.SEEDED, RuleOrigin.LEARNED])
_STATUSES = st.sampled_from(list(ValidationStatus))


@st.composite
def _representations(draw) -> SymbolicRepresentation:
    return SymbolicRepresentation(logic_form=draw(st.text(max_size=16)))


@st.composite
def _repair_attempts(draw) -> RepairAttempt:
    return RepairAttempt(
        attempt_index=0,  # reassigned by trace order below
        rejected_step=draw(_representations()),
        violated_rule_ids=draw(st.lists(_SAFE_ID, max_size=2)),
        repaired_step=draw(st.one_of(st.none(), _representations())),
    )


@st.composite
def _proof_steps(draw) -> ProofStep:
    """A step spanning every status, rule-id present/absent, and every origin."""

    return ProofStep(
        sequence=0,  # reassigned to its execution index in _proof_traces
        step_text=draw(st.text(max_size=16)),
        representation=draw(st.one_of(st.none(), _representations())),
        status=draw(_STATUSES),
        applied_rule_id=draw(st.one_of(st.none(), _SAFE_ID)),
        applied_rule_origin=draw(_ORIGINS),
        violated_rule_ids=draw(st.lists(_SAFE_ID, max_size=2)),
        repair_attempts=draw(st.lists(_repair_attempts(), max_size=2)),
    )


@st.composite
def _proof_traces(draw) -> ProofTrace:
    """A trace whose steps carry unique, execution-ordered sequence numbers.

    Sequences are reassigned to the zero-based execution index (the builder's
    ``steps[i].sequence == i`` invariant) so each step maps to a unique ``S{sequence}``
    node id, letting the test parse per-node styling unambiguously. Repair-attempt
    indices are likewise normalized to execution order.
    """

    steps = draw(st.lists(_proof_steps(), max_size=5))
    for i, step in enumerate(steps):
        step.sequence = i
        for j, attempt in enumerate(step.repair_attempts):
            attempt.attempt_index = j
    return ProofTrace(
        steps=steps,
        termination_reason=draw(
            st.one_of(st.none(), st.sampled_from(list(TerminationReason)))
        ),
    )


# --------------------------------------------------------------------------- #
# Parsers for step nodes in each rendering
# --------------------------------------------------------------------------- #

# Mermaid step node:  `    S0["<label>"]`
_MERMAID_STEP = re.compile(r'^\s*(S\d+)\["(.*)"\]$')
# Mermaid class assignment:  `    class S0 accepted;`
_MERMAID_CLASS = re.compile(r"^\s*class (S\d+) (\w+);$")
# DOT step node:  `    S0 [label="<label>", shape=box, style=filled, fillcolor="#..."];`
_DOT_STEP = re.compile(
    r'^\s*(S\d+) \[label="(.*)", shape=(\w+), style=filled, fillcolor="(#[0-9a-fA-F]+)"\];$'
)


def _mermaid_step_nodes(mermaid: str) -> dict[str, str]:
    """Map ``S{seq}`` -> rendered label for every Mermaid step node."""

    out: dict[str, str] = {}
    for line in mermaid.splitlines():
        m = _MERMAID_STEP.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _mermaid_step_classes(mermaid: str) -> dict[str, str]:
    """Map ``S{seq}`` -> assigned classDef class for every Mermaid step node."""

    out: dict[str, str] = {}
    for line in mermaid.splitlines():
        m = _MERMAID_CLASS.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _dot_step_nodes(dot: str) -> dict[str, tuple[str, str, str]]:
    """Map ``S{seq}`` -> (label, shape, fillcolor) for every DOT step node."""

    out: dict[str, tuple[str, str, str]] = {}
    for line in dot.splitlines():
        m = _DOT_STEP.match(line)
        if m:
            out[m.group(1)] = (m.group(2), m.group(3), m.group(4))
    return out


def _expected_annotation_checks(step: ProofStep, label: str) -> None:
    """Assert ``label`` carries the rule-id / no-rule indicator and origin marker."""

    # Rule id or the explicit no-rule indicator (Req 15.4).
    if step.applied_rule_id is not None:
        assert step.applied_rule_id in label
    else:
        assert NO_RULE_APPLIED in label

    # Learned-vs-seeded marker where a rule validated the step (origin recorded).
    if step.applied_rule_origin is not None:
        assert f"({step.applied_rule_origin.value})" in label
    else:
        assert "(seeded)" not in label
        assert "(learned)" not in label


# --------------------------------------------------------------------------- #
# Static mutual-distinctness of the status -> style mappings (Req 15.4)
# --------------------------------------------------------------------------- #


def test_status_style_classes_are_mutually_distinct() -> None:
    """accepted/rejected/repaired map to distinct Mermaid classes and DOT fills."""

    statuses = list(ValidationStatus)
    # Mermaid: distinct classDef class names, each with a declared style.
    classes = [_STATUS_CLASS[s] for s in statuses]
    assert len(set(classes)) == len(statuses)
    for cls in classes:
        assert cls in _MERMAID_CLASSDEF
    # DOT: distinct fillcolors per status.
    fills = [_DOT_STATUS_STYLE[_STATUS_CLASS[s]][1] for s in statuses]
    assert len(set(fills)) == len(statuses)


# --------------------------------------------------------------------------- #
# Property 13: nodes are outcome-distinguished and rule-annotated
# --------------------------------------------------------------------------- #


@settings(max_examples=200)
@given(trace=_proof_traces())
def test_mermaid_nodes_are_outcome_distinguished_and_rule_annotated(
    trace: ProofTrace,
) -> None:
    """Property 13 (Mermaid): per-node class matches status; label carries annotations.

    **Validates: Requirements 15.4**
    """

    mermaid = to_mermaid(trace)
    labels = _mermaid_step_nodes(mermaid)
    classes = _mermaid_step_classes(mermaid)

    for step in trace.steps:
        node_id = f"S{step.sequence}"
        assert node_id in labels
        assert node_id in classes
        # Outcome-distinguished styling: the class matches the step's status.
        assert classes[node_id] == _STATUS_CLASS[step.status]
        # Rule-id / no-rule indicator and learned/seeded marker on this node's label.
        _expected_annotation_checks(step, labels[node_id])


@settings(max_examples=200)
@given(trace=_proof_traces())
def test_dot_nodes_are_outcome_distinguished_and_rule_annotated(
    trace: ProofTrace,
) -> None:
    """Property 13 (DOT): per-node fillcolor matches status; label carries annotations.

    **Validates: Requirements 15.4**
    """

    dot = to_dot(trace)
    nodes = _dot_step_nodes(dot)

    for step in trace.steps:
        node_id = f"S{step.sequence}"
        assert node_id in nodes
        label, _shape, fill = nodes[node_id]
        # Outcome-distinguished styling: the fillcolor matches the step's status.
        expected_fill = _DOT_STATUS_STYLE[_STATUS_CLASS[step.status]][1]
        assert fill == expected_fill
        # Rule-id / no-rule indicator and learned/seeded marker on this node's label.
        _expected_annotation_checks(step, label)
