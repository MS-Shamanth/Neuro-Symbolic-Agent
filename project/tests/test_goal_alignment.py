"""Tests for goal-aligned semantic step-validation (demo, Phase 2+).

These confirm that :class:`GoalAlignmentValidationEngine` catches a step that is
*arithmetically correct but computes the wrong QUANTITY for the goal* — the headline being
the real profit failure: goal "What is the profit?" answered with "40 * 13 = 520" (the
COST). They also confirm the engine never weakens arithmetic checking, abstains when the
goal operation cannot be inferred, accepts goal-aligned and compound expressions, and that
goal validation + the bounded repair sub-loop FIXES the profit case end to end — all fully
offline (the safe evaluator uses no ``eval``/``exec``; the orchestrator runs over a
:class:`~nsr.llm_component.MockBackend`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from arithmetic_validation import ArithmeticValidationEngine  # noqa: E402
from goal_alignment import (  # noqa: E402
    GOAL_ALIGNMENT_RULE_ID,
    GoalAlignmentValidationEngine,
    expression_operations,
    infer_goal_operation,
)
from run_benchmark import numeric_answer_match  # noqa: E402
from scenarios import build_orchestrator_with_backend, make_config  # noqa: E402

from nsr.llm_component import MockBackend  # noqa: E402
from nsr.models import (  # noqa: E402
    ProductionRule,
    SymbolicRepresentation,
    TerminationReason,
    ValidationStatus,
    VerifiedOutput,
)

#: The motivating profit goal (ground truth 380 = 900 - 520).
PROFIT_GOAL = (
    "A store buys items for 520 dollars and sells them for 900 dollars. "
    "What is the profit in dollars?"
)


def _rep(logic_form="", predicates=None):
    return SymbolicRepresentation(
        logic_form=logic_form, predicates=predicates or {}, source_text=logic_form
    )


# --------------------------------------------------------------------------- #
# infer_goal_operation — keyword heuristic
# --------------------------------------------------------------------------- #


def test_infer_profit_is_subtract():
    assert infer_goal_operation("What is the profit?") == "subtract"


def test_infer_how_many_left_is_subtract():
    assert infer_goal_operation("How many are left?") == "subtract"


def test_infer_total_and_altogether_is_add():
    assert infer_goal_operation("How many in total?") == "add"
    assert infer_goal_operation("How many altogether?") == "add"


def test_infer_each_is_divide():
    assert infer_goal_operation("How many does each get?") == "divide"


def test_infer_ambiguous_is_none():
    assert infer_goal_operation("What is the answer?") is None
    assert infer_goal_operation("") is None


# --------------------------------------------------------------------------- #
# expression_operations — AST operation extraction
# --------------------------------------------------------------------------- #


def test_expression_operations_multiply():
    assert expression_operations("40 * 13") == {"multiply"}


def test_expression_operations_compound_subtract_multiply():
    assert expression_operations("900 - 40*13") == {"subtract", "multiply"}


def test_expression_operations_divide():
    assert expression_operations("480 / 6") == {"divide"}


def test_expression_operations_unparseable_is_empty():
    assert expression_operations("@@@") == set()
    assert expression_operations("") == set()


def test_expression_operations_normalises_x_glyph():
    # The same x/×/÷ normalisation as the arithmetic checker.
    assert expression_operations("7 x 8") == {"multiply"}


# --------------------------------------------------------------------------- #
# GoalAlignmentValidationEngine — the headline profit failure and friends
# --------------------------------------------------------------------------- #


def test_profit_failure_is_rejected_by_goal_alignment():
    """REPRODUCE THE PROFIT FAILURE: '40 * 13 = 520' for a profit goal is rejected.

    The arithmetic is CORRECT (40*13 really is 520), so arithmetic validation alone accepts
    it — but the goal asks for PROFIT (a subtraction) and the step contains only a
    multiplication, so goal-alignment REJECTS it with the synthetic ``goal-alignment`` rule.
    """
    eng = GoalAlignmentValidationEngine(goal_text=PROFIT_GOAL)
    outcome = eng.validate(_rep("40 * 13 = 520"), [])

    assert outcome.status is ValidationStatus.REJECTED
    assert GOAL_ALIGNMENT_RULE_ID in outcome.violated_rule_ids
    assert any(r.rule_id == GOAL_ALIGNMENT_RULE_ID for r in outcome.violated_rules)
    # It is a goal mismatch, NOT an arithmetic error.
    assert "arithmetic-correctness" not in outcome.violated_rule_ids


def test_goal_aligned_final_step_is_accepted():
    """The correct goal-aligned step '900 - 520 = 380' (a subtraction) is accepted."""
    eng = GoalAlignmentValidationEngine(goal_text=PROFIT_GOAL)
    assert eng.validate(_rep("900 - 520 = 380"), []).status is ValidationStatus.ACCEPTED


def test_compound_expression_containing_subtract_is_accepted():
    """A compound '900 - 40 * 13 = 380' passes: it CONTAINS the expected subtraction."""
    eng = GoalAlignmentValidationEngine(goal_text=PROFIT_GOAL)
    assert (
        eng.validate(_rep("900 - 40 * 13 = 380"), []).status
        is ValidationStatus.ACCEPTED
    )


def test_wrong_arithmetic_still_rejected_regardless_of_goal():
    """Wrong arithmetic is rejected via arithmetic-correctness, not goal-alignment."""
    eng = GoalAlignmentValidationEngine(goal_text=PROFIT_GOAL)
    outcome = eng.validate(_rep("40 * 13 = 999"), [])
    assert outcome.status is ValidationStatus.REJECTED
    assert "arithmetic-correctness" in outcome.violated_rule_ids
    assert GOAL_ALIGNMENT_RULE_ID not in outcome.violated_rule_ids


def test_uninferable_goal_behaves_like_arithmetic_engine():
    """When the goal op can't be inferred, the engine matches the arithmetic engine."""
    arithmetic = ArithmeticValidationEngine()
    goal = GoalAlignmentValidationEngine(goal_text="What is the answer?")

    for logic_form in ("40 * 13 = 520", "900 - 520 = 380", "cats_are_animals"):
        rep = _rep(logic_form)
        a_outcome = arithmetic.validate(rep, [])
        g_outcome = goal.validate(rep, [])
        assert g_outcome.status is a_outcome.status
        assert GOAL_ALIGNMENT_RULE_ID not in g_outcome.violated_rule_ids


def test_non_arithmetic_step_is_not_goal_rejected():
    """A step with no checkable equation is never rejected for goal-alignment."""
    eng = GoalAlignmentValidationEngine(goal_text=PROFIT_GOAL)
    assert eng.validate(_rep("the_profit_is_large"), []).status is ValidationStatus.ACCEPTED


# --------------------------------------------------------------------------- #
# End-to-end: goal validation + repair FIXES the profit case, fully offline
# --------------------------------------------------------------------------- #


def _equation_step(logic_form, lhs, op, rhs, result):
    """One constrained-decoder-shaped JSON completion asserting an arithmetic equation."""
    return json.dumps(
        {
            "logic_form": logic_form,
            "predicates": {"lhs": lhs, "op": op, "rhs": rhs, "result": result},
        }
    )


def test_orchestrator_catches_goal_mismatch_and_repairs_to_profit():
    """The misaligned cost step is rejected (goal-alignment), then repaired to 380.

    A scripted backend emits the misaligned ``40 * 13 = 520`` (the COST) first, then the
    goal-aligned ``900 - 520 = 380`` (the PROFIT) on repair. With the
    :class:`GoalAlignmentValidationEngine` injected (and shared with the Repair Coordinator),
    the first step is rejected against the synthetic ``goal-alignment`` rule and routed into
    the bounded repair sub-loop; the regenerated subtraction is accepted, the goal is
    satisfied, and the final answer is 380 — fully offline.
    """
    # A single-clause query => one sub-goal, so the 2-item script (misaligned, then
    # aligned-on-repair) drives a clean reject -> repair -> accept path.
    query = "What is the profit in dollars?"
    any_rule = ProductionRule(rule_id="R-any", condition="", action="")
    backend = MockBackend(
        [
            _equation_step("40 * 13 = 520", 40, "*", 13, 520),   # COST -> goal mismatch
            _equation_step("900 - 520 = 380", 900, "-", 520, 380),  # PROFIT -> accepted
        ]
    )
    orchestrator = build_orchestrator_with_backend(
        backend=backend,
        procedural_memory=[any_rule],
        config=make_config(),
        with_repair=True,
        validation=GoalAlignmentValidationEngine(goal_text=query),
    )

    result = orchestrator.run(query)

    assert isinstance(result, VerifiedOutput)
    trace = result.proof_trace
    assert trace.termination_reason == TerminationReason.GOAL_SATISFIED

    # The misaligned step was caught: recorded as repaired and citing the goal-alignment rule.
    repaired_steps = [s for s in trace.steps if s.status == ValidationStatus.REPAIRED]
    assert repaired_steps, "the misaligned cost step should be marked REPAIRED"
    assert any(
        GOAL_ALIGNMENT_RULE_ID in s.violated_rule_ids for s in trace.steps
    ), "the rejected step should cite the goal-alignment constraint"
    assert any(s.repair_attempts for s in trace.steps)

    # The final answer is the PROFIT (380), not the cost (520 alone). The aligned
    # subtraction "900 - 520 = 380" legitimately mentions 520 as the subtrahend, so we
    # compare the reduced NUMERIC answer rather than substring-excluding 520.
    assert numeric_answer_match(result.final_answer, "380")
    assert not numeric_answer_match(result.final_answer, "520")
