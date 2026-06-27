"""Property-based test for structurally complete trace visualization.

Task 18.2 / **Property 11: Visualization is structurally complete.**

*For any* :class:`~nsr.models.trace.ProofTrace`, the diagram emitted by both
:func:`~nsr.trace_visualizer.to_mermaid` and :func:`~nsr.trace_visualizer.to_dot`
contains:

- exactly one **step node** per :class:`~nsr.models.trace.ProofStep`,
- exactly one **goal node** and exactly one **terminal node**,
- exactly one **branch node** per :class:`~nsr.models.trace.RepairAttempt`,
- an **edge from each step node to its successor** in the shared flow order, and
- a **terminal node** whose label reflects the Verified_Output (on a
  ``goal-satisfied`` termination) or the ``termination_reason``.

The two renderings are parsed back out of their emitted text and the node/edge counts
are checked structurally against the source trace.

**Validates: Requirements 15.1**
"""

from __future__ import annotations

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from nsr.models.enums import TerminationReason, ValidationStatus
from nsr.models.learning import RuleOrigin
from nsr.models.reasoning import SymbolicRepresentation
from nsr.models.trace import ProofStep, ProofTrace, RepairAttempt
from nsr.trace_visualizer import _terminal_label, to_dot, to_mermaid

# --------------------------------------------------------------------------- #
# Leaf strategies
# --------------------------------------------------------------------------- #
#
# Identifier-safe text avoids quotes/newlines so escaping never perturbs the structural
# counts; the structural property is about node/edge cardinality, not label content.
_ident_alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
_rule_id = st.text(alphabet=_ident_alphabet, min_size=1, max_size=8)
_rule_ids = st.lists(_rule_id, max_size=3)
_text = st.text(max_size=16)


@st.composite
def representations(draw) -> SymbolicRepresentation:
    """A minimal :class:`SymbolicRepresentation` (content is irrelevant to structure)."""

    return SymbolicRepresentation(
        logic_form=draw(_text),
        source_text=draw(_text),
    )


@st.composite
def proof_steps(draw, sequence: int) -> ProofStep:
    """A :class:`ProofStep` with the given ``sequence`` and a varying repair-attempt set.

    Repair attempts get distinct ``attempt_index`` values ``0..k-1`` so that, like real
    execution order, no two branch nodes for one step collide on their node id. ``k``
    varies (including zero) to exercise steps with and without repairs.
    """

    n_repairs = draw(st.integers(min_value=0, max_value=3))
    repair_attempts = [
        RepairAttempt(
            attempt_index=i,
            rejected_step=draw(representations()),
            violated_rule_ids=draw(_rule_ids),
            repaired_step=draw(st.one_of(st.none(), representations())),
        )
        for i in range(n_repairs)
    ]
    return ProofStep(
        sequence=sequence,
        step_text=draw(_text),
        representation=draw(st.one_of(st.none(), representations())),
        status=draw(st.sampled_from(list(ValidationStatus))),
        applied_rule_id=draw(st.one_of(st.none(), _rule_id)),
        applied_rule_origin=draw(st.one_of(st.none(), st.sampled_from(list(RuleOrigin)))),
        violated_rule_ids=draw(_rule_ids),
        repair_attempts=repair_attempts,
    )


@st.composite
def proof_traces(draw) -> ProofTrace:
    """An arbitrary :class:`ProofTrace` with unique step sequences and varying repairs.

    Step ``sequence`` values are the distinct ``0..n-1`` (mirroring execution order) so
    step node ids never collide; ``n`` ranges from 0 (the minimal-diagram case) up to
    several. Termination reason spans every reason and ``None`` (in-progress).
    """

    n_steps = draw(st.integers(min_value=0, max_value=5))
    steps = [draw(proof_steps(sequence=i)) for i in range(n_steps)]
    return ProofTrace(
        steps=steps,
        termination_reason=draw(
            st.one_of(st.none(), st.sampled_from(list(TerminationReason)))
        ),
    )


# --------------------------------------------------------------------------- #
# Parsers: recover node/edge structure from emitted diagram text
# --------------------------------------------------------------------------- #

# Mermaid declaration lines (one per node):
#   GB(["Goal Buffer"])      goal
#   T(["..."])               terminal
#   S<seq>["..."]            step
#   R<seq>_<idx>{{"..."}}    repair branch
_MM_GOAL = re.compile(r"^GB\(\[")
_MM_TERMINAL = re.compile(r"^T\(\[")
_MM_STEP = re.compile(r"^S(\d+)\[")
_MM_REPAIR = re.compile(r"^R(\d+)_(\d+)\{\{")
_MM_EDGE = re.compile(r"^(\S+)\s*-->\s*(\S+)$")

# DOT declaration lines (one per node) end every statement with ';':
#   GB [label=...];          goal
#   T [label=...];           terminal
#   S<seq> [label=...];      step
#   R<seq>_<idx> [label=...]; repair branch
_DOT_GOAL = re.compile(r"^GB\s+\[label=")
_DOT_TERMINAL = re.compile(r"^T\s+\[label=")
_DOT_STEP = re.compile(r"^S(\d+)\s+\[label=")
_DOT_REPAIR = re.compile(r"^R(\d+)_(\d+)\s+\[label=")
_DOT_EDGE = re.compile(r"^(\S+)\s*->\s*(\S+);$")


def _parse_mermaid(text: str):
    goal = terminal = 0
    steps: list[str] = []
    repairs: list[str] = []
    edges: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if _MM_GOAL.match(line):
            goal += 1
        elif _MM_TERMINAL.match(line):
            terminal += 1
        elif _MM_STEP.match(line):
            steps.append(f"S{_MM_STEP.match(line).group(1)}")
        elif _MM_REPAIR.match(line):
            m = _MM_REPAIR.match(line)
            repairs.append(f"R{m.group(1)}_{m.group(2)}")
        else:
            m = _MM_EDGE.match(line)
            if m:
                edges.append((m.group(1), m.group(2)))
    return goal, terminal, steps, repairs, edges


def _parse_dot(text: str):
    goal = terminal = 0
    steps: list[str] = []
    repairs: list[str] = []
    edges: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if _DOT_GOAL.match(line):
            goal += 1
        elif _DOT_TERMINAL.match(line):
            terminal += 1
        elif _DOT_STEP.match(line):
            steps.append(f"S{_DOT_STEP.match(line).group(1)}")
        elif _DOT_REPAIR.match(line):
            m = _DOT_REPAIR.match(line)
            repairs.append(f"R{m.group(1)}_{m.group(2)}")
        else:
            m = _DOT_EDGE.match(line)
            if m:
                edges.append((m.group(1), m.group(2)))
    return goal, terminal, steps, repairs, edges


# --------------------------------------------------------------------------- #
# Structural assertions shared by both formats
# --------------------------------------------------------------------------- #


def _assert_structure(parsed, trace: ProofTrace) -> None:
    goal, terminal, steps, repairs, edges = parsed

    # Exactly one goal node and exactly one terminal node frame the diagram.
    assert goal == 1, f"expected exactly one goal node, got {goal}"
    assert terminal == 1, f"expected exactly one terminal node, got {terminal}"

    # Exactly one step node per ProofStep, with ids matching the step sequences.
    expected_steps = {f"S{s.sequence}" for s in trace.steps}
    assert len(steps) == len(trace.steps), (
        f"expected {len(trace.steps)} step nodes, got {len(steps)}"
    )
    assert set(steps) == expected_steps, (
        f"step node ids {set(steps)} != step sequences {expected_steps}"
    )

    # Exactly one branch node per RepairAttempt, with ids matching step/attempt indices.
    expected_repairs = {
        f"R{s.sequence}_{a.attempt_index}"
        for s in trace.steps
        for a in s.repair_attempts
    }
    total_repairs = sum(len(s.repair_attempts) for s in trace.steps)
    assert len(repairs) == total_repairs, (
        f"expected {total_repairs} repair branch nodes, got {len(repairs)}"
    )
    assert set(repairs) == expected_repairs

    # The flow is a single chain over (goal, step+repairs..., terminal): one fewer edge
    # than nodes, and every step node has an outgoing edge to its successor.
    n_nodes = goal + terminal + len(steps) + len(repairs)
    assert len(edges) == n_nodes - 1, (
        f"expected {n_nodes - 1} edges for {n_nodes} nodes, got {len(edges)}"
    )
    edge_sources = {src for src, _ in edges}
    for step_id in expected_steps:
        assert step_id in edge_sources, f"step node {step_id} has no successor edge"


def _assert_terminal_label(text: str, trace: ProofTrace) -> None:
    """The terminal node label reflects the Verified_Output or the termination reason."""

    # _terminal_label spans the three cases: "Verified Output" on goal-satisfied, the
    # reason value otherwise, and the in-progress placeholder. The chosen alphabet has
    # no quotes/newlines, so the label appears verbatim in the rendered text.
    assert _terminal_label(trace) in text


# --------------------------------------------------------------------------- #
# Property 11: structural completeness
# --------------------------------------------------------------------------- #


@settings(max_examples=200)
@given(trace=proof_traces())
def test_mermaid_is_structurally_complete(trace: ProofTrace) -> None:
    """Mermaid output has one node per step/repair plus goal and terminal nodes.

    **Validates: Requirements 15.1**
    """

    text = to_mermaid(trace)
    _assert_structure(_parse_mermaid(text), trace)
    _assert_terminal_label(text, trace)


@settings(max_examples=200)
@given(trace=proof_traces())
def test_dot_is_structurally_complete(trace: ProofTrace) -> None:
    """DOT output has one node per step/repair plus goal and terminal nodes.

    **Validates: Requirements 15.1**
    """

    text = to_dot(trace)
    _assert_structure(_parse_dot(text), trace)
    _assert_terminal_label(text, trace)
