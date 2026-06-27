"""Example/edge unit tests for the Trace Visualizer output formats (Task 18.5).

These are concrete example/edge checks (not property-based) that complement the
focused exporter unit tests in ``test_trace_visualizer.py`` without duplicating them.
They pin down the two format-shape guarantees on representative and edge traces:

- ``to_mermaid`` output begins with a ``flowchart`` header and ``to_dot`` output is a
  ``digraph { ... }`` block on a representative trace (Req 15.3);
- both exporters return a well-formed minimal diagram for an EMPTY ``ProofTrace``
  without raising (Req 15.5).

In addition they strengthen coverage with edge cases the base unit tests do not
exercise: a trace that terminated for a non-``goal-satisfied`` reason (so the terminal
renders the reason rather than a Verified_Output), and a trace whose single step
carries repair attempts.
"""

from __future__ import annotations

from nsr.models import RuleOrigin
from nsr.models.enums import TerminationReason, ValidationStatus
from nsr.models.reasoning import SymbolicRepresentation
from nsr.models.trace import ProofStep, ProofTrace, RepairAttempt
from nsr.trace_visualizer import to_dot, to_mermaid


def _representative_trace() -> ProofTrace:
    """A small but representative accepted-then-repaired trace."""

    rep = SymbolicRepresentation(logic_form="add(2,2)=4")
    rejected = SymbolicRepresentation(logic_form="add(2,2)=5")

    accepted_step = ProofStep(
        sequence=0,
        step_text="2 + 2 = 4",
        representation=rep,
        status=ValidationStatus.ACCEPTED,
        applied_rule_id="R1",
        applied_rule_origin=RuleOrigin.SEEDED,
    )
    repaired_step = ProofStep(
        sequence=1,
        step_text="2 + 2 = 5 (later corrected)",
        representation=rep,
        status=ValidationStatus.REPAIRED,
        applied_rule_id="R3",
        applied_rule_origin=RuleOrigin.LEARNED,
        violated_rule_ids=["R2"],
        repair_attempts=[
            RepairAttempt(
                attempt_index=0,
                rejected_step=rejected,
                violated_rule_ids=["R2"],
                repaired_step=rep,
            )
        ],
    )

    return ProofTrace(
        steps=[accepted_step, repaired_step],
        termination_reason=TerminationReason.GOAL_SATISFIED,
    )


# --- Req 15.3: format-shape on a representative trace ------------------------


def test_mermaid_output_begins_with_flowchart_header():
    """``to_mermaid`` opens with a ``flowchart`` header line (Req 15.3)."""

    mermaid = to_mermaid(_representative_trace())
    first_line = mermaid.splitlines()[0]
    assert first_line.startswith("flowchart")


def test_dot_output_is_a_digraph_block():
    """``to_dot`` is a complete ``digraph { ... }`` block (Req 15.3)."""

    dot = to_dot(_representative_trace())
    assert dot.lstrip().startswith("digraph")
    assert "{" in dot
    assert dot.rstrip().endswith("}")


# --- Req 15.5: empty trace renders a well-formed minimal diagram ------------


def test_empty_trace_mermaid_is_minimal_and_well_formed():
    """An empty ``ProofTrace`` still yields a well-formed Mermaid flowchart (Req 15.5)."""

    mermaid = to_mermaid(ProofTrace())
    first_line = mermaid.splitlines()[0]
    assert first_line.startswith("flowchart")
    # Minimal diagram: a framing goal node and at least one connecting edge.
    assert "Goal Buffer" in mermaid
    assert "-->" in mermaid


def test_empty_trace_dot_is_minimal_and_well_formed():
    """An empty ``ProofTrace`` still yields a well-formed DOT digraph (Req 15.5)."""

    dot = to_dot(ProofTrace())
    assert dot.lstrip().startswith("digraph")
    assert dot.rstrip().endswith("}")
    assert "Goal Buffer" in dot
    assert "->" in dot


def test_both_exporters_do_not_raise_on_empty_trace():
    """Neither exporter raises on an empty trace (Req 15.5)."""

    empty = ProofTrace()
    # Should simply return non-empty strings rather than raising.
    assert to_mermaid(empty)
    assert to_dot(empty)


# --- Edge: non-goal-satisfied termination has no Verified Output ------------


def test_terminal_renders_reason_when_not_goal_satisfied():
    """A non-``goal-satisfied`` reason renders that reason, not a Verified Output."""

    rep = SymbolicRepresentation(logic_form="step(1)")
    step = ProofStep(
        sequence=0,
        step_text="partial step",
        representation=rep,
        status=ValidationStatus.ACCEPTED,
        applied_rule_id="R1",
        applied_rule_origin=RuleOrigin.SEEDED,
    )
    trace = ProofTrace(
        steps=[step],
        termination_reason=TerminationReason.CYCLE_LIMIT_REACHED,
    )

    mermaid = to_mermaid(trace)
    dot = to_dot(trace)

    assert TerminationReason.CYCLE_LIMIT_REACHED.value in mermaid
    assert TerminationReason.CYCLE_LIMIT_REACHED.value in dot
    assert "Verified Output" not in mermaid
    assert "Verified Output" not in dot


# --- Edge: a trace whose only step carries repair attempts ------------------


def test_single_repaired_step_with_repair_attempts_renders_branch():
    """A lone repaired step still emits its repair-branch node in both formats."""

    rep = SymbolicRepresentation(logic_form="fixed")
    rejected = SymbolicRepresentation(logic_form="broken")
    step = ProofStep(
        sequence=0,
        step_text="needed repair",
        representation=rep,
        status=ValidationStatus.REPAIRED,
        applied_rule_id="R7",
        applied_rule_origin=RuleOrigin.LEARNED,
        violated_rule_ids=["R5"],
        repair_attempts=[
            RepairAttempt(
                attempt_index=0,
                rejected_step=rejected,
                violated_rule_ids=["R5"],
                repaired_step=rep,
            )
        ],
    )
    trace = ProofTrace(
        steps=[step],
        termination_reason=TerminationReason.REPAIR_EXHAUSTED,
    )

    mermaid = to_mermaid(trace)
    dot = to_dot(trace)

    # Both formats are well-formed and include the step and its repair branch.
    assert mermaid.splitlines()[0].startswith("flowchart")
    assert dot.rstrip().endswith("}")
    assert "Step 0" in mermaid and "Step 0" in dot
    assert "Repair 0" in mermaid and "Repair 0" in dot
    assert TerminationReason.REPAIR_EXHAUSTED.value in mermaid
    assert TerminationReason.REPAIR_EXHAUSTED.value in dot
