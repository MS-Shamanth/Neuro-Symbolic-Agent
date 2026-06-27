"""Arithmetic step-validation for the GSM8K experiment (demo, Phase 2).

The base :class:`~nsr.validation_engine.ValidationEngine` only does IF/THEN *substring*
matching -- it cannot tell that ``7 * 8 = 54`` is wrong. :class:`ArithmeticValidationEngine`
adds genuine arithmetic checking on top: when a reasoning step asserts an arithmetic
equation, it evaluates the left-hand side with the SAFE evaluator from
:mod:`demo.arithmetic` (no ``eval``/``exec``) and **rejects** the step when the asserted
result is wrong, attaching a synthetic ``arithmetic-correctness`` violated rule so the
orchestrator routes the step into the bounded repair sub-loop. Correct equations and
non-arithmetic steps keep the base outcome unchanged, so every existing IF/THEN rule still
applies.

This is what turns step-level validation from "is the output well-formed?" into "is the
intermediate arithmetic actually correct?" -- letting the System catch and repair a wrong
intermediate step that a plain Chain-of-Thought trace would carry through to a wrong final
answer.

An equation is detected from a step's :class:`~nsr.models.SymbolicRepresentation` in these
forms (predicates preferred, then the logic form):

- predicates ``{"lhs": a, "op": "*", "rhs": b, "result": r}``;
- predicates ``{"expression": "7*8", "result": 56}``;
- predicates carrying an ``"equation"`` or ``"logic_form"`` string (``"7*8 = 56"``);
- the representation's ``logic_form`` itself, parsed as ``"<expr> = <result>"``.
"""

from __future__ import annotations

from typing import Optional

from arithmetic import parse_equation, safe_eval_arithmetic

from nsr.models import ProductionRule, SymbolicRepresentation, ValidationStatus
from nsr.validation_engine import RuleEvaluation, ValidationEngine, ValidationOutcome

#: The synthetic rule recorded when an arithmetic step asserts a wrong result, so the
#: orchestrator's Repair Coordinator receives a meaningful offending constraint.
ARITHMETIC_RULE = ProductionRule(
    rule_id="arithmetic-correctness",
    condition="IF equation",
    action="THEN compute correctly",
)

#: Tolerance for floating-point comparison of computed vs asserted values.
_EPS = 1e-6

#: Operator tokens accepted in structured ``{"lhs", "op", "rhs", "result"}`` predicates.
_OP_SYMBOL = {
    "+": "+",
    "-": "-",
    "*": "*",
    "x": "*",
    "×": "*",
    "·": "*",
    "/": "/",
    "÷": "/",
    "//": "//",
    "%": "%",
    "**": "**",
}


def _extract_equation(rep: SymbolicRepresentation) -> Optional[tuple[str, float]]:
    """Return ``(lhs_expression, asserted_result)`` for an arithmetic step, else ``None``.

    Prefers structured predicates; falls back to parsing ``"<expr> = <number>"`` out of an
    ``equation``/``logic_form`` predicate or the representation's own ``logic_form`` via
    :func:`~demo.arithmetic.parse_equation`. Returns ``None`` when the step asserts no
    checkable arithmetic equation.
    """
    preds = rep.predicates if isinstance(rep.predicates, dict) else {}

    # Form A: explicit lhs/op/rhs/result.
    if {"lhs", "op", "rhs", "result"} <= set(preds):
        op = _OP_SYMBOL.get(str(preds["op"]).strip().lower())
        if op is not None:
            try:
                lhs = float(str(preds["lhs"]).replace(",", ""))
                rhs = float(str(preds["rhs"]).replace(",", ""))
                result = float(str(preds["result"]).replace(",", ""))
            except (ValueError, TypeError):
                return None
            return f"{lhs}{op}{rhs}", result

    # Form B: expression + result.
    if "expression" in preds and "result" in preds:
        try:
            result = float(str(preds["result"]).replace(",", ""))
        except (ValueError, TypeError):
            return None
        return str(preds["expression"]), result

    # Form C: an explicit equation/logic_form string in the predicates.
    for key in ("equation", "logic_form"):
        if key in preds and preds[key] is not None:
            parsed = parse_equation(str(preds[key]))
            if parsed is not None:
                return parsed

    # Form D: parse "<expr> = <number>" from the representation's logic form.
    return parse_equation(rep.logic_form or "")


class ArithmeticValidationEngine(ValidationEngine):
    """A Validation Engine that also verifies asserted arithmetic (demo, Phase 2).

    :meth:`validate` first checks whether the step asserts a checkable arithmetic equation.
    If it does, it evaluates the expression with the safe evaluator and compares it to the
    asserted result: a correct equation yields an ``ACCEPTED`` outcome (also honouring any
    matching base production rules); a wrong equation yields a ``REJECTED`` outcome carrying
    :data:`ARITHMETIC_RULE` as a violated rule, routing the step to repair. Steps with no
    detectable arithmetic defer entirely to the base IF/THEN engine, so non-arithmetic
    behaviour is unchanged. Pure function; no ``eval``/``exec``.
    """

    def validate(
        self,
        rep: SymbolicRepresentation,
        rules: list[ProductionRule],
    ) -> ValidationOutcome:
        equation = _extract_equation(rep)
        if equation is None:
            # No checkable arithmetic -> defer entirely to the base substring rules.
            return super().validate(rep, rules)

        lhs_expr, asserted = equation
        computed = safe_eval_arithmetic(lhs_expr)
        if computed is None:
            # The asserted "equation" has an unevaluable LHS -> not genuinely arithmetic;
            # defer to the base rules rather than crashing or falsely rejecting.
            return super().validate(rep, rules)

        base = super().validate(rep, rules)

        if abs(computed - asserted) < _EPS:
            # Arithmetic is correct; accept (preserving any base rejection if a base rule
            # was also violated) and record a satisfied arithmetic evaluation.
            evaluations = list(base.evaluations) + [
                RuleEvaluation(
                    rule_id=ARITHMETIC_RULE.rule_id, applicable=True, satisfied=True
                )
            ]
            applicable = list(base.applicable_rule_ids) + [ARITHMETIC_RULE.rule_id]
            return ValidationOutcome(
                status=base.status,
                representation=rep,
                applicable_rule_ids=applicable,
                violated_rule_ids=list(base.violated_rule_ids),
                violated_rules=list(base.violated_rules),
                evaluations=evaluations,
            )

        # Arithmetic is wrong -> reject and route to repair (Req 6.3, 6.4 semantics).
        evaluations = list(base.evaluations) + [
            RuleEvaluation(
                rule_id=ARITHMETIC_RULE.rule_id, applicable=True, satisfied=False
            )
        ]
        applicable = list(base.applicable_rule_ids) + [ARITHMETIC_RULE.rule_id]
        violated_ids = list(base.violated_rule_ids) + [ARITHMETIC_RULE.rule_id]
        violated_rules = list(base.violated_rules) + [ARITHMETIC_RULE]
        return ValidationOutcome(
            status=ValidationStatus.REJECTED,
            representation=rep,
            applicable_rule_ids=applicable,
            violated_rule_ids=violated_ids,
            violated_rules=violated_rules,
            evaluations=evaluations,
        )


def extract_equation_for_test(rep: SymbolicRepresentation):
    """Expose equation extraction for unit tests."""
    return _extract_equation(rep)


__all__ = [
    "ArithmeticValidationEngine",
    "ARITHMETIC_RULE",
    "extract_equation_for_test",
]
