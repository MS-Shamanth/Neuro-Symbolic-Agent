"""Goal-aligned semantic step-validation for the GSM8K experiment (demo, Phase 2+).

Arithmetic validation (:class:`~demo.arithmetic_validation.ArithmeticValidationEngine`)
answers "is the computation *correct*?". It cannot answer the *different* question "is the
computation the one the GOAL actually asked for?". A real 30-item run surfaced exactly this
gap: for the goal *"What is the profit?"* (ground truth ``380 = 900 - 520``) the model
answered ``"40 * 13 = 520"`` — the COST. Arithmetic validation ACCEPTED it (``40*13`` really
is ``520``), yet it is the WRONG QUANTITY: profit is a *subtraction* (revenue − cost), but
the step is a bare *multiplication* with no subtraction anywhere.

:class:`GoalAlignmentValidationEngine` closes that gap. It composes the arithmetic engine
(never weakening it) and adds an *intent* check: it infers the arithmetic operation the
GOAL is asking for from the question's wording, and rejects an otherwise-correct
final-answer step whose computation does not contain that operation — routing it into the
bounded repair sub-loop instead of letting a confidently-wrong quantity through.

HONEST SCOPE — this v1 rule targets **final-answer** alignment and is most reliable for
single-equation answers (like the profit case). Limitations, stated plainly:

- It classifies the goal operation from **keywords only** (see
  :func:`infer_goal_operation`). It is deliberately CONSERVATIVE: when the wording is
  unclear, or when two different operation families both match, it returns ``None`` and the
  engine then behaves exactly like the arithmetic engine (no goal rejection). This trades
  recall for precision to avoid false rejections.
- It can over-reject a legitimate **intermediate** step in a fully decomposed multi-step
  trace whose final operation differs from an intermediate one (e.g. a trace that first
  computes the cost ``40 * 13`` as a genuine sub-step on the way to the profit). To reduce
  such false positives the engine ACCEPTS whenever the expected operation appears ANYWHERE
  in the expression (so a compound ``"900 - 40*13 = 380"`` passes), and only rejects when
  the expected operation is **entirely absent**. Per-sub-goal alignment (checking each
  sub-goal's own intended operation) is future work.

The check is layered on top of arithmetic correctness; it uses the same SAFE, ``ast``-based
parsing as :mod:`demo.arithmetic` (no ``eval``/``exec``), so it cannot execute code.
"""

from __future__ import annotations

import ast
import re
from typing import Optional

from arithmetic_validation import (
    ARITHMETIC_RULE,
    ArithmeticValidationEngine,
    extract_equation_for_test as _extract_equation,
)

from nsr.models import ProductionRule, SymbolicRepresentation, ValidationStatus
from nsr.validation_engine import RuleEvaluation, ValidationEngine, ValidationOutcome

#: The synthetic rule recorded when a step computes the wrong QUANTITY for the goal, so the
#: Repair Coordinator receives a meaningful offending constraint to regenerate against.
GOAL_ALIGNMENT_RULE_ID = "goal-alignment"

#: The four operation families this module reasons about.
_OPERATIONS = ("add", "subtract", "multiply", "divide")

# --------------------------------------------------------------------------- #
# Keyword -> goal-operation heuristic
# --------------------------------------------------------------------------- #
#
# Each family lists word-boundary keyword patterns. The mapping is intentionally small and
# transparent (it is a HEURISTIC, not natural-language understanding):
#
#   subtract : profit, gain, net, left, remaining, remain, difference, fewer, more than,
#              "how many ... left"
#   add      : total, altogether, in all, combined, sum, "how many ... in total"
#   divide   : each, per, evenly, apiece, per person, "how many ... per"
#   multiply : product, times, "how many in N groups"
#
# "how many ... left/per/in total" are covered by the bare keywords (left / per / total),
# so the trailing "how many ..." phrasings need no special case.
_OPERATION_KEYWORDS: dict[str, list[str]] = {
    "subtract": [
        r"\bprofit\b",
        r"\bgain\b",
        r"\bnet\b",
        r"\bleft\b",
        r"\bremaining\b",
        r"\bremain\b",
        r"\bdifference\b",
        r"\bfewer\b",
        r"\bmore than\b",
    ],
    "add": [
        r"\btotal\b",
        r"\baltogether\b",
        r"\bin all\b",
        r"\bcombined\b",
        r"\bsum\b",
    ],
    "divide": [
        r"\beach\b",
        r"\bper person\b",
        r"\bper\b",
        r"\bevenly\b",
        r"\bapiece\b",
    ],
    "multiply": [
        r"\bproduct\b",
        r"\btimes\b",
        r"how many in \d+ groups",
        r"\bin \d+ groups\b",
    ],
}


def infer_goal_operation(goal_text: str) -> Optional[str]:
    """Classify the arithmetic operation a question is asking for, from keywords.

    Returns one of ``"add"``, ``"subtract"``, ``"multiply"``, ``"divide"`` when the wording
    clearly indicates a single operation family, else ``None``.

    The classifier is a transparent keyword heuristic (NOT natural-language understanding),
    and is deliberately CONSERVATIVE so the goal-alignment engine errs toward *not*
    rejecting: it returns ``None`` both when no family matches and when *more than one*
    distinct family matches (an ambiguous goal). The keyword table is documented at module
    scope and in this function's source.

    Examples:
        ``"What is the profit?"`` -> ``"subtract"`` (profit = revenue − cost);
        ``"How many are left?"`` -> ``"subtract"``;
        ``"How many in total?"`` -> ``"add"``;
        ``"How many does each get?"`` -> ``"divide"``;
        ``"What is the answer?"`` -> ``None`` (no clear keyword).
    """
    if not goal_text:
        return None
    text = str(goal_text).lower()

    matched: set[str] = set()
    for operation, patterns in _OPERATION_KEYWORDS.items():
        if any(re.search(pattern, text) for pattern in patterns):
            matched.add(operation)

    # Conservative: classify only when exactly one family matches; otherwise abstain.
    if len(matched) == 1:
        return next(iter(matched))
    return None


def _normalise_expression(expr: str) -> str:
    """Lower-case and normalise multiplication/division glyphs and separators.

    Mirrors the normalisation used by :func:`demo.arithmetic.safe_eval_arithmetic` so the
    operation set is computed over exactly the expression the arithmetic checker evaluates:
    ``×``/``·`` and a standalone ``x`` between operands become ``*``, ``÷`` becomes ``/``,
    and ``$``/``,`` are stripped.
    """
    text = str(expr).strip().lower().replace("$", "").replace(",", "")
    text = text.replace("×", "*").replace("·", "*").replace("÷", "/")
    text = re.sub(r"(?<=[\d\)])\s*x\s*(?=[\d\(])", "*", text)
    return text


def expression_operations(expr: str) -> set[str]:
    """Return the SET of arithmetic operations present in ``expr``.

    Parses ``expr`` with the SAFE :mod:`ast` module (no ``eval``/``exec``) and maps each
    binary operator node to an operation family drawn from
    ``{"add", "subtract", "multiply", "divide"}`` (``ast.Add`` -> ``add``, ``ast.Sub`` ->
    ``subtract``, ``ast.Mult`` -> ``multiply``, ``ast.Div``/``ast.FloorDiv`` -> ``divide``).
    The same ``x``/``×``/``÷`` normalisation as the arithmetic checker is applied first.

    Returns an empty set when ``expr`` is empty, unparseable, or contains no recognised
    binary operation (a bare number, a name, etc.).
    """
    if not expr:
        return set()
    text = _normalise_expression(expr)
    if not text:
        return set()
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return set()

    ops: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            op = node.op
            if isinstance(op, ast.Add):
                ops.add("add")
            elif isinstance(op, ast.Sub):
                ops.add("subtract")
            elif isinstance(op, ast.Mult):
                ops.add("multiply")
            elif isinstance(op, (ast.Div, ast.FloorDiv)):
                ops.add("divide")
    return ops


def goal_mismatch_reason(expected: str, ops: set[str]) -> str:
    """Build the human-readable reason recorded for a goal-alignment rejection.

    Exposed as a helper so the reason is testable and reusable; the engine cites it when it
    rejects a step whose computation omits the goal's expected operation.
    """
    ops_text = ", ".join(sorted(ops)) if ops else "no recognised operation"
    return (
        f"goal mismatch: the goal asks for a result computed via {expected} "
        f"(e.g. profit = revenue - cost), but this step's computation uses only "
        f"{ops_text} and contains no {expected}"
    )


class GoalAlignmentValidationEngine(ValidationEngine):
    """A Validation Engine that checks *goal intent* on top of arithmetic correctness.

    Constructed per-query with the ``goal_text`` (the query IS the goal). :meth:`validate`:

    1. Runs an internal :class:`~demo.arithmetic_validation.ArithmeticValidationEngine`
       first. If that REJECTS the step (wrong arithmetic, or a violated base production
       rule), its outcome is returned UNCHANGED — goal alignment only ever *adds*
       rejections, it never overturns one.
    2. Otherwise (arithmetic accepted, or the step has no checkable equation) it applies
       goal alignment. It infers the goal's expected operation via
       :func:`infer_goal_operation`. When an operation is inferred AND the step carries a
       checkable equation AND the expression's operation set is non-empty AND the expected
       operation is **entirely absent** from that set, it REJECTS with a synthetic
       ``goal-alignment`` :class:`~nsr.models.ProductionRule` (so the bounded repair
       sub-loop regenerates a goal-aligned step). In every other case it ACCEPTS, deferring
       to the arithmetic-accepted outcome.

    Acceptance when the expected operation appears ANYWHERE in the expression keeps compound
    answers (``"900 - 40*13 = 380"``) valid and limits false rejections. Pure function of
    its inputs; no ``eval``/``exec``.
    """

    def __init__(self, goal_text: str) -> None:
        """Create an engine bound to ``goal_text`` (the query whose intent is enforced)."""
        self.goal_text = goal_text or ""
        self._arithmetic = ArithmeticValidationEngine()

    def validate(
        self,
        rep: SymbolicRepresentation,
        rules: list[ProductionRule],
    ) -> ValidationOutcome:
        # Step 1: arithmetic (and base-rule) check first. Never overturn its rejection.
        arithmetic_outcome = self._arithmetic.validate(rep, rules)
        if arithmetic_outcome.rejected:
            return arithmetic_outcome

        # Step 2: goal alignment. Abstain unless every precondition for a confident
        # final-answer mismatch holds.
        expected = infer_goal_operation(self.goal_text)
        if expected is None:
            return arithmetic_outcome

        equation = _extract_equation(rep)
        if equation is None:
            return arithmetic_outcome

        lhs_expr, _asserted = equation
        ops = expression_operations(lhs_expr)
        if not ops:
            return arithmetic_outcome

        if expected in ops:
            # The goal's operation is present (possibly inside a compound expression).
            return arithmetic_outcome

        # The expected operation is entirely absent -> wrong QUANTITY for the goal.
        goal_rule = ProductionRule(
            rule_id=GOAL_ALIGNMENT_RULE_ID,
            condition="IF final-answer",
            action=f"THEN compute the goal quantity using {expected}",
        )
        reason = goal_mismatch_reason(expected, ops)
        evaluations = list(arithmetic_outcome.evaluations) + [
            RuleEvaluation(
                rule_id=GOAL_ALIGNMENT_RULE_ID, applicable=True, satisfied=False
            )
        ]
        applicable = list(arithmetic_outcome.applicable_rule_ids) + [
            GOAL_ALIGNMENT_RULE_ID
        ]
        violated_ids = list(arithmetic_outcome.violated_rule_ids) + [
            GOAL_ALIGNMENT_RULE_ID
        ]
        violated_rules = list(arithmetic_outcome.violated_rules) + [goal_rule]
        # ``reason`` is carried in the rule's action text (which the Repair Coordinator
        # surfaces in its prompt) and is also available via :func:`goal_mismatch_reason`;
        # ValidationOutcome has no free-text field, so we annotate the rule, not the
        # frozen outcome.
        _ = reason
        return ValidationOutcome(
            status=ValidationStatus.REJECTED,
            representation=rep,
            applicable_rule_ids=applicable,
            violated_rule_ids=violated_ids,
            violated_rules=violated_rules,
            evaluations=evaluations,
        )


__all__ = [
    "infer_goal_operation",
    "expression_operations",
    "goal_mismatch_reason",
    "GoalAlignmentValidationEngine",
    "GOAL_ALIGNMENT_RULE_ID",
    "ARITHMETIC_RULE",
]
