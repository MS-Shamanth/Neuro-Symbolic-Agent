"""Trace Visualizer: pure Mermaid and Graphviz DOT exporters for a Proof_Trace.

This module implements Task 18.1 of the design's *Trace Visualizer (Reasoning
Visualization)* component (Req 15.1-15.5). It is a pure, read-only exporter over the
append-only :class:`~nsr.models.trace.ProofTrace`: it introduces no new state and never
mutates the trace it is given. Two renderings are produced from one shared traversal so
they carry identical content and styling contracts:

- :func:`to_mermaid` -- a Mermaid ``flowchart TD`` diagram.
- :func:`to_dot` -- a Graphviz ``digraph { ... }`` diagram.

Both renderings show the same flow::

    Goal Buffer -> Reasoning Step 1 -> validation outcome -> Repair branch (when
    present) -> ... -> Verified Output / termination reason

and uphold the following contract:

- **Lossless (Req 15.2):** each step's execution-order sequence, its validation
  outcome, its applied production rule id (or the explicit ``no-rule-applied``
  indicator reused from :func:`~nsr.proof_trace.applied_rule_label`), and the
  termination reason all appear verbatim in the rendered text.
- **Outcome-distinguished (Req 15.4):** step nodes are styled by
  :class:`~nsr.models.enums.ValidationStatus` so accepted, rejected, and repaired steps
  are visually distinct -- Mermaid ``classDef`` classes, DOT node ``shape``/``fillcolor``
  -- via the internal :data:`_STATUS_CLASS` mapping.
- **Rule-annotated (Req 15.4):** each step node carries its applied rule id (or the
  no-rule indicator) plus a learned/seeded marker derived from
  :attr:`ProofStep.applied_rule_origin` when present.
- **Repair branches (Req 15.1):** one branch node is emitted per
  :class:`~nsr.models.trace.RepairAttempt`, ordered by ``attempt_index``.
- **Empty trace (Req 15.5):** a trace with no steps still renders a well-formed minimal
  diagram (a goal/placeholder node and a single placeholder terminal) rather than
  failing.
"""

from __future__ import annotations

from .models.enums import TerminationReason, ValidationStatus
from .models.trace import ProofStep, ProofTrace, RepairAttempt
from .proof_trace import NO_RULE_APPLIED, applied_rule_label

#: Stable identifiers for the fixed framing nodes.
_GOAL_NODE_ID = "GB"
_TERMINAL_NODE_ID = "T"

#: Maps a step's :class:`ValidationStatus` to the style class shared by both renderings.
#: The class name is used directly as the Mermaid ``classDef`` name and drives the DOT
#: node ``shape``/``fillcolor`` lookup, so accepted/rejected/repaired stay visually
#: distinct in both formats (Req 15.4).
_STATUS_CLASS: dict[ValidationStatus, str] = {
    ValidationStatus.ACCEPTED: "accepted",
    ValidationStatus.REJECTED: "rejected",
    ValidationStatus.REPAIRED: "repaired",
}

#: Mermaid ``classDef`` declarations, keyed by the style class name.
_MERMAID_CLASSDEF: dict[str, str] = {
    "accepted": "fill:#d4f7d4,stroke:#2e7d32,color:#000",
    "rejected": "fill:#f7d4d4,stroke:#c62828,color:#000",
    "repaired": "fill:#fff3c4,stroke:#f9a825,color:#000",
}

#: DOT node attributes (shape + fillcolor), keyed by the style class name (Req 15.4).
_DOT_STATUS_STYLE: dict[str, tuple[str, str]] = {
    "accepted": ("box", "#d4f7d4"),
    "rejected": ("box", "#f7d4d4"),
    "repaired": ("box", "#fff3c4"),
}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _origin_marker(step: ProofStep) -> str:
    """Return a ``" (learned)"``/``" (seeded)"`` marker, or ``""`` when unknown.

    Derived from :attr:`ProofStep.applied_rule_origin` (ties Req 14.5 into Req 15.4).
    ``None`` -- no rule applied or origin unknown -- yields no marker.
    """

    if step.applied_rule_origin is None:
        return ""
    return f" ({step.applied_rule_origin.value})"


def _step_label(step: ProofStep) -> str:
    """Human-readable, lossless label for a step node (Req 15.2, 15.4).

    Carries the execution-order sequence, the validation outcome, the applied rule id
    (or the explicit ``no-rule-applied`` indicator), and the learned/seeded marker.
    """

    return (
        f"Step {step.sequence} ({step.status.value}): "
        f"{applied_rule_label(step)}{_origin_marker(step)}"
    )


def _repair_label(step: ProofStep, attempt: RepairAttempt) -> str:
    """Label for a repair-branch node (Req 15.1)."""

    if attempt.violated_rule_ids:
        violated = ", ".join(attempt.violated_rule_ids)
    else:
        violated = NO_RULE_APPLIED
    return f"Repair {attempt.attempt_index} (step {step.sequence}): violated [{violated}]"


def _terminal_label(trace: ProofTrace) -> str:
    """Label for the terminal node (Req 15.1, 15.5).

    Renders the Verified_Output on a ``goal-satisfied`` termination, the
    ``termination_reason`` value for any other recorded reason, and a placeholder when
    no reason has been recorded yet (the empty/in-progress case).
    """

    reason = trace.termination_reason
    if reason is TerminationReason.GOAL_SATISFIED:
        return "Verified Output"
    if reason is None:
        return "(no termination reason)"
    return reason.value


def _flow(trace: ProofTrace) -> list[tuple[str, str, ProofStep | None, RepairAttempt | None]]:
    """Build the ordered flow of nodes shared by both renderings.

    Returns a list of ``(node_id, kind, step, attempt)`` tuples in execution order:
    the goal node, then for each step its node followed by its repair-branch nodes
    (ordered by ``attempt_index``), then the terminal node. ``kind`` is one of
    ``"goal"``, ``"step"``, ``"repair"``, ``"terminal"``. The same ordered list drives
    node declaration and the sequential edges, guaranteeing both formats agree.
    """

    flow: list[tuple[str, str, ProofStep | None, RepairAttempt | None]] = [
        (_GOAL_NODE_ID, "goal", None, None)
    ]
    for step in trace.steps:
        flow.append((f"S{step.sequence}", "step", step, None))
        for attempt in sorted(step.repair_attempts, key=lambda a: a.attempt_index):
            flow.append(
                (f"R{step.sequence}_{attempt.attempt_index}", "repair", step, attempt)
            )
    flow.append((_TERMINAL_NODE_ID, "terminal", None, None))
    return flow


# --------------------------------------------------------------------------- #
# Mermaid rendering (Req 15.1, 15.3, 15.4, 15.5)
# --------------------------------------------------------------------------- #


def _mermaid_escape(text: str) -> str:
    """Escape a label for use inside a Mermaid ``"..."`` node body.

    Mermaid uses HTML-style numeric/character entities; double quotes are replaced with
    ``#quot;`` and newlines with ``<br/>`` so the label stays on one logical node.
    """

    return text.replace('"', "#quot;").replace("\n", "<br/>")


def to_mermaid(trace: ProofTrace) -> str:
    """Render ``trace`` as a Mermaid ``flowchart TD`` (Req 15.1-15.5).

    Pure: never mutates ``trace``. Step nodes are styled by outcome via ``classDef``
    classes (accepted/rejected/repaired) and annotated with the applied rule id (or the
    explicit no-rule indicator) plus a learned/seeded marker; one branch node is emitted
    per repair attempt; the terminal node renders the Verified_Output or the termination
    reason; an empty trace still yields a well-formed minimal diagram.
    """

    flow = _flow(trace)
    lines: list[str] = ["flowchart TD"]
    classed: list[tuple[str, str]] = []  # (node_id, class) assignments

    for node_id, kind, step, attempt in flow:
        if kind == "goal":
            lines.append(f'    {node_id}(["Goal Buffer"])')
        elif kind == "terminal":
            lines.append(f'    {node_id}(["{_mermaid_escape(_terminal_label(trace))}"])')
        elif kind == "step":
            assert step is not None
            lines.append(f'    {node_id}["{_mermaid_escape(_step_label(step))}"]')
            classed.append((node_id, _STATUS_CLASS[step.status]))
        elif kind == "repair":
            assert step is not None and attempt is not None
            label = _mermaid_escape(_repair_label(step, attempt))
            lines.append(f'    {node_id}{{{{"{label}"}}}}')
            classed.append((node_id, "repaired"))

    # Sequential edges following the shared flow order.
    for (src, _, _, _), (dst, _, _, _) in zip(flow, flow[1:]):
        lines.append(f"    {src} --> {dst}")

    # classDef declarations (Req 15.4) then per-node class assignments.
    for name, style in _MERMAID_CLASSDEF.items():
        lines.append(f"    classDef {name} {style};")
    for node_id, class_name in classed:
        lines.append(f"    class {node_id} {class_name};")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Graphviz DOT rendering (Req 15.1, 15.3, 15.4, 15.5)
# --------------------------------------------------------------------------- #


def _dot_escape(text: str) -> str:
    """Escape a label for use inside a DOT double-quoted string."""

    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def to_dot(trace: ProofTrace) -> str:
    """Render ``trace`` as a Graphviz ``digraph`` (Req 15.3).

    Carries the same content and styling contract as :func:`to_mermaid`: step nodes are
    styled by outcome (DOT ``shape``/``fillcolor`` from :data:`_DOT_STATUS_STYLE`) and
    annotated with the applied rule id (or no-rule indicator) plus the learned/seeded
    marker; one branch node per repair attempt; the terminal renders the Verified_Output
    or termination reason; an empty trace yields a well-formed minimal digraph. Pure;
    never mutates ``trace``.
    """

    flow = _flow(trace)
    lines: list[str] = ["digraph ProofTrace {", "    rankdir=TD;"]

    for node_id, kind, step, attempt in flow:
        if kind == "goal":
            lines.append(
                f'    {node_id} [label="Goal Buffer", shape=stadium, '
                f'style=filled, fillcolor="#e0e0e0"];'
            )
        elif kind == "terminal":
            label = _dot_escape(_terminal_label(trace))
            lines.append(
                f'    {node_id} [label="{label}", shape=stadium, '
                f'style=filled, fillcolor="#e0e0e0"];'
            )
        elif kind == "step":
            assert step is not None
            shape, fill = _DOT_STATUS_STYLE[_STATUS_CLASS[step.status]]
            label = _dot_escape(_step_label(step))
            lines.append(
                f'    {node_id} [label="{label}", shape={shape}, '
                f'style=filled, fillcolor="{fill}"];'
            )
        elif kind == "repair":
            assert step is not None and attempt is not None
            _shape, fill = _DOT_STATUS_STYLE["repaired"]
            label = _dot_escape(_repair_label(step, attempt))
            lines.append(
                f'    {node_id} [label="{label}", shape=diamond, '
                f'style=filled, fillcolor="{fill}"];'
            )

    for (src, _, _, _), (dst, _, _, _) in zip(flow, flow[1:]):
        lines.append(f"    {src} -> {dst};")

    lines.append("}")
    return "\n".join(lines)
