"""Unit tests for the Trace Visualizer Mermaid/DOT exporters (Task 18.1).

These cover, on a representative mixed trace and on the empty-trace placeholder:

- format headers: ``to_mermaid`` starts with ``flowchart`` and ``to_dot`` is a
  ``digraph { ... }`` (Req 15.3);
- a goal node, one node per step in execution order, repair-branch nodes ordered by
  ``attempt_index``, and a terminal node (Req 15.1);
- outcome-distinguished styling for accepted/rejected/repaired steps (Req 15.4);
- applied rule ids / the explicit no-rule indicator plus the learned/seeded marker
  (Req 15.4);
- the empty trace renders a well-formed minimal diagram without raising (Req 15.5);
- purity: the exporters never mutate the trace.

The dedicated property tests (18.2-18.4) and broader example tests (18.5) live in
separate task files; these are the focused unit checks for the exporter itself.
"""

from __future__ import annotations

import copy

from nsr.models import RuleOrigin
from nsr.models.enums import TerminationReason, ValidationStatus
from nsr.models.reasoning import SymbolicRepresentation
from nsr.models.trace import ProofStep, ProofTrace, RepairAttempt
from nsr.proof_trace import NO_RULE_APPLIED
from nsr.trace_visualizer import to_dot, to_mermaid


def _mixed_trace() -> ProofTrace:
    """A trace exercising accepted/repaired outcomes, a repair branch, and origins."""

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
        applied_rule_id=None,
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


# --- Format headers (Req 15.3) ----------------------------------------------


def test_mermaid_starts_with_flowchart():
    assert to_mermaid(_mixed_trace()).startswith("flowchart")


def test_dot_is_a_digraph_block():
    dot = to_dot(_mixed_trace())
    assert dot.startswith("digraph")
    assert dot.rstrip().endswith("}")


# --- Structure: goal, step, repair, terminal nodes (Req 15.1) ---------------


def test_mermaid_has_goal_step_repair_and_terminal_nodes():
    mermaid = to_mermaid(_mixed_trace())
    assert "Goal Buffer" in mermaid
    assert "Step 0" in mermaid
    assert "Step 1" in mermaid
    assert "Repair 0" in mermaid
    assert "Verified Output" in mermaid


def test_dot_has_goal_step_repair_and_terminal_nodes():
    dot = to_dot(_mixed_trace())
    assert "Goal Buffer" in dot
    assert "Step 0" in dot
    assert "Step 1" in dot
    assert "Repair 0" in dot
    assert "Verified Output" in dot


def test_steps_appear_in_execution_order():
    mermaid = to_mermaid(_mixed_trace())
    assert mermaid.index("Step 0") < mermaid.index("Step 1")


# --- Outcome-distinguished styling (Req 15.4) -------------------------------


def test_mermaid_assigns_outcome_classes():
    mermaid = to_mermaid(_mixed_trace())
    assert "classDef accepted" in mermaid
    assert "classDef repaired" in mermaid
    assert "class S0 accepted;" in mermaid
    assert "class S1 repaired;" in mermaid


def test_dot_styles_steps_by_outcome():
    dot = to_dot(_mixed_trace())
    # Accepted and repaired steps use distinct fill colours.
    assert "#d4f7d4" in dot  # accepted
    assert "#fff3c4" in dot  # repaired


# --- Rule ids, no-rule indicator, learned/seeded markers (Req 15.4) ---------


def test_applied_rule_ids_and_markers_present():
    mermaid = to_mermaid(_mixed_trace())
    assert "R1" in mermaid
    assert "(seeded)" in mermaid
    assert "(learned)" in mermaid
    # The repaired step has no applied rule id, so the no-rule indicator appears.
    assert NO_RULE_APPLIED in mermaid


# --- Empty trace renders a minimal diagram (Req 15.5) -----------------------


def test_empty_trace_mermaid_is_well_formed():
    mermaid = to_mermaid(ProofTrace())
    assert mermaid.startswith("flowchart")
    assert "Goal Buffer" in mermaid


def test_empty_trace_dot_is_well_formed():
    dot = to_dot(ProofTrace())
    assert dot.startswith("digraph")
    assert dot.rstrip().endswith("}")


# --- Purity: exporters never mutate the trace -------------------------------


def test_to_mermaid_does_not_mutate_trace():
    trace = _mixed_trace()
    before = copy.deepcopy(trace)
    to_mermaid(trace)
    assert trace == before


def test_to_dot_does_not_mutate_trace():
    trace = _mixed_trace()
    before = copy.deepcopy(trace)
    to_dot(trace)
    assert trace == before
