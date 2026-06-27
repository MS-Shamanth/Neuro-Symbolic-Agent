"""Tests for the arithmetic step-validation engine (demo Phase 2).

These confirm that :class:`ArithmeticValidationEngine` genuinely *computes* and verifies an
asserted arithmetic step — accepting correct equations, rejecting wrong ones (with the
synthetic ``arithmetic-correctness`` violated rule that drives repair), leaving
non-arithmetic steps to the base IF/THEN rules, and never crashing on garbage input. It
runs fully offline; the safe evaluator uses no ``eval``/``exec``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

import json  # noqa: E402

from arithmetic_validation import (  # noqa: E402
    ARITHMETIC_RULE,
    ArithmeticValidationEngine,
    extract_equation_for_test,
)
from scenarios import build_orchestrator_with_backend, make_config  # noqa: E402

from nsr.llm_component import MockBackend  # noqa: E402
from nsr.models import (  # noqa: E402
    ProductionRule,
    SymbolicRepresentation,
    TerminationReason,
    ValidationStatus,
    VerifiedOutput,
)


def _rep(logic_form="", predicates=None):
    return SymbolicRepresentation(
        logic_form=logic_form, predicates=predicates or {}, source_text=logic_form
    )


def test_correct_equation_from_predicates_is_accepted():
    eng = ArithmeticValidationEngine()
    rep = _rep("7 * 8 = 56", {"lhs": 7, "op": "*", "rhs": 8, "result": 56})
    assert eng.validate(rep, []).status is ValidationStatus.ACCEPTED


def test_wrong_equation_from_predicates_is_rejected_with_arithmetic_rule():
    eng = ArithmeticValidationEngine()
    rep = _rep("7 * 8 = 54", {"lhs": 7, "op": "*", "rhs": 8, "result": 54})
    outcome = eng.validate(rep, [])
    assert outcome.status is ValidationStatus.REJECTED
    assert ARITHMETIC_RULE.rule_id in outcome.violated_rule_ids
    assert any(r.rule_id == ARITHMETIC_RULE.rule_id for r in outcome.violated_rules)


def test_correct_equation_from_logic_form_is_accepted():
    eng = ArithmeticValidationEngine()
    assert eng.validate(_rep("24 - 9 = 15"), []).status is ValidationStatus.ACCEPTED
    assert eng.validate(_rep("144 / 12 = 12"), []).status is ValidationStatus.ACCEPTED
    assert eng.validate(_rep("60 * 3 = 180"), []).status is ValidationStatus.ACCEPTED


def test_wrong_equation_from_logic_form_is_rejected():
    eng = ArithmeticValidationEngine()
    assert eng.validate(_rep("24 - 9 = 20"), []).status is ValidationStatus.REJECTED


def test_x_as_multiplication_is_supported():
    eng = ArithmeticValidationEngine()
    assert eng.validate(_rep("7 x 8 = 56"), []).status is ValidationStatus.ACCEPTED
    assert eng.validate(_rep("7 x 8 = 55"), []).status is ValidationStatus.REJECTED


def test_non_arithmetic_step_is_unaffected_and_uses_base_rules():
    eng = ArithmeticValidationEngine()
    # No equation -> base IF/THEN rules decide. Empty rule set -> vacuously accepted.
    assert eng.validate(_rep("cats_are_animals"), []).status is ValidationStatus.ACCEPTED


def test_non_arithmetic_step_can_still_be_rejected_by_a_base_rule():
    eng = ArithmeticValidationEngine()
    rep = _rep("draft conclusion", {"status": "draft"})
    rule = ProductionRule(rule_id="must-verify", condition="", action="THEN verified")
    outcome = eng.validate(rep, [rule])
    assert outcome.status is ValidationStatus.REJECTED
    assert "must-verify" in outcome.violated_rule_ids


def test_garbage_equation_does_not_crash():
    eng = ArithmeticValidationEngine()
    # A bogus "equation" whose LHS is not evaluable is treated as non-arithmetic.
    assert eng.validate(_rep("foo = bar"), []).status is ValidationStatus.ACCEPTED


def test_division_by_zero_is_safe():
    eng = ArithmeticValidationEngine()
    # 5 / 0 is undefined; the LHS is unevaluable, so it falls through to base (accepted).
    assert eng.validate(_rep("5 / 0 = 0"), []).status is ValidationStatus.ACCEPTED


def test_no_eval_used_names_are_not_evaluated():
    eng = ArithmeticValidationEngine()
    # If the evaluator naively used eval(), "__import__('os')" style content would be a
    # security risk; here any name makes the expression non-numeric -> non-arithmetic.
    rep = _rep("__import__ = 1")
    assert eng.validate(rep, []).status is ValidationStatus.ACCEPTED


def test_extract_equation_prefers_predicates():
    rep = _rep("ignored", {"expression": "10 + 5", "result": 15})
    eq = extract_equation_for_test(rep)
    assert eq is not None
    lhs, result = eq
    assert result == 15.0


# --------------------------------------------------------------------------- #
# Explicit "7*8" equation checks driven purely from the logic form
# --------------------------------------------------------------------------- #


def test_accepts_correct_logic_form_seven_times_eight():
    eng = ArithmeticValidationEngine()
    assert eng.validate(_rep("7*8=56"), []).status is ValidationStatus.ACCEPTED


def test_rejects_wrong_logic_form_seven_times_eight_with_arithmetic_rule():
    eng = ArithmeticValidationEngine()
    outcome = eng.validate(_rep("7*8=54"), [])
    assert outcome.status is ValidationStatus.REJECTED
    assert ARITHMETIC_RULE.rule_id == "arithmetic-correctness"
    assert "arithmetic-correctness" in outcome.violated_rule_ids
    assert any(
        r.rule_id == "arithmetic-correctness" for r in outcome.violated_rules
    )


# --------------------------------------------------------------------------- #
# End-to-end: a wrong arithmetic step is caught and repaired, fully offline
# --------------------------------------------------------------------------- #


def _equation_step(logic_form, lhs, op, rhs, result):
    """One constrained-decoder-shaped JSON completion asserting an arithmetic equation."""
    return json.dumps(
        {
            "logic_form": logic_form,
            "predicates": {"lhs": lhs, "op": op, "rhs": rhs, "result": result},
        }
    )


def test_orchestrator_catches_wrong_arithmetic_and_repairs_to_correct_answer():
    """A scripted backend emits a WRONG equation, then a CORRECT one on repair.

    With the :class:`ArithmeticValidationEngine` injected (and shared with the Repair
    Coordinator via :func:`build_orchestrator_with_backend`), the wrong intermediate step
    ``7*8 = 54`` is rejected against the synthetic ``arithmetic-correctness`` rule and
    routed into the bounded repair sub-loop. The regenerated ``7*8 = 56`` is accepted, the
    goal is satisfied, and the final answer carries the correct value. Fully offline.
    """
    # An always-applicable, always-satisfied base rule, so arithmetic correctness is the
    # only thing that can reject a step.
    any_rule = ProductionRule(rule_id="R-any", condition="", action="")
    backend = MockBackend(
        [
            _equation_step("7*8 = 54", 7, "*", 8, 54),  # WRONG -> rejected
            _equation_step("7*8 = 56", 7, "*", 8, 56),  # CORRECT -> accepted on repair
        ]
    )
    orchestrator = build_orchestrator_with_backend(
        backend=backend,
        procedural_memory=[any_rule],
        config=make_config(),
        with_repair=True,
        validation=ArithmeticValidationEngine(),
    )

    result = orchestrator.run("what is 7 times 8")

    # The run reached goal-satisfied with a Verified_Output.
    assert isinstance(result, VerifiedOutput)
    trace = result.proof_trace
    assert trace.termination_reason == TerminationReason.GOAL_SATISFIED

    # The wrong step was caught: it is recorded as repaired and cited the arithmetic rule.
    repaired_steps = [
        s for s in trace.steps if s.status == ValidationStatus.REPAIRED
    ]
    assert repaired_steps, "the wrong arithmetic step should be marked REPAIRED"
    assert any(
        "arithmetic-correctness" in s.violated_rule_ids for s in trace.steps
    ), "the rejected step should cite the arithmetic-correctness constraint"

    # At least one repair attempt was journaled into the trace.
    assert any(s.repair_attempts for s in trace.steps)

    # The final answer reflects the corrected computation (7 * 8 = 56), not the wrong 54.
    assert "56" in result.final_answer
    assert "54" not in result.final_answer


def test_orchestrator_exhausts_repair_when_arithmetic_never_corrected():
    """If every regenerated step stays wrong, repair is bounded and exhausts cleanly.

    This proves the rejection genuinely routes to repair (rather than silently passing):
    with only wrong equations scripted, the bounded sub-loop terminates with
    ``repair-exhausted`` instead of accepting a wrong answer.
    """
    any_rule = ProductionRule(rule_id="R-any", condition="", action="")
    backend = MockBackend([_equation_step("7*8 = 54", 7, "*", 8, 54)])
    orchestrator = build_orchestrator_with_backend(
        backend=backend,
        procedural_memory=[any_rule],
        config=make_config(repair_attempt_limit=2),
        with_repair=True,
        validation=ArithmeticValidationEngine(),
    )

    result = orchestrator.run("what is 7 times 8")

    assert result.proof_trace.termination_reason == TerminationReason.REPAIR_EXHAUSTED
