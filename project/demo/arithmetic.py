"""Safe arithmetic evaluation and equation parsing for the demo (Phase 2).

The base :class:`~nsr.validation_engine.ValidationEngine` does substring matching only and
cannot tell that ``7 * 8 = 54`` is wrong. This module supplies the two primitives that let
:class:`~demo.arithmetic_validation.ArithmeticValidationEngine` actually *compute* and
*check* an asserted arithmetic step:

- :func:`safe_eval_arithmetic` -- evaluate a pure arithmetic expression using Python's
  :mod:`ast` module. It walks the parsed tree and permits ONLY numeric literals, the binary
  operators ``+ - * / // % **``, unary ``+``/``-`` and parentheses. Every other node type
  (names, calls, attribute access, subscripts, comprehensions, strings, ...) makes the
  expression invalid and yields ``None``. It NEVER calls the builtin ``eval``/``exec`` and
  never touches ``builtins`` or globals, so it cannot execute arbitrary code.
- :func:`parse_equation` -- pull an ``"<expr> = <result>"`` arithmetic claim out of a
  step's text, splitting on the *last* ``=`` (left = expression, right = a number).

Common "times"/"divide" glyphs are normalised: ``x`` / ``X`` / ``×`` / ``·`` become ``*``
and ``÷`` becomes ``/``. Currency ``$`` and thousands separators ``,`` are stripped.
"""

from __future__ import annotations

import ast
import re
from typing import Optional

#: Guard against pathological exponentiation (e.g. ``10 ** 10 ** 10``) that could hang or
#: exhaust memory even though it is "valid" arithmetic. Exponents beyond this magnitude
#: make the expression invalid (returns ``None``).
_MAX_EXPONENT = 1000

# Binary operators we permit, mapped to a callable. ``//``, ``%`` and ``/`` guard against
# division/modulo by zero by returning ``None`` rather than raising.
_BIN_OPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)
_UNARY_OPS = (ast.UAdd, ast.USub)


def _normalise(expr: str) -> str:
    """Lower-case and normalise multiplication/division glyphs and separators.

    ``×``/``·`` and a standalone ``x`` between operands become ``*``; ``÷`` becomes ``/``;
    ``$`` and ``,`` are removed. The standalone-``x`` rule only fires when ``x`` sits
    between two numeric/closing-paren and numeric/opening-paren contexts, so a bare name
    like ``x`` is left intact (and therefore later rejected as a non-numeric name).
    """
    text = expr.strip().lower().replace("$", "").replace(",", "")
    text = text.replace("×", "*").replace("·", "*").replace("÷", "/")
    # "7 x 8" -> "7*8" and ") x (" -> ")*(" but leave a lone "x" as a (rejected) name.
    text = re.sub(r"(?<=[\d\)])\s*x\s*(?=[\d\(])", "*", text)
    return text


def safe_eval_arithmetic(expr: str) -> Optional[float]:
    """Safely evaluate ``expr`` as a pure arithmetic expression, or return ``None``.

    Only numeric literals, the binary operators ``+ - * / // % **``, unary ``+``/``-`` and
    parentheses are allowed. Any name, call, attribute, subscript, comprehension, string,
    or other construct makes the expression invalid and returns ``None``. Division or
    modulo by zero returns ``None`` rather than raising. This uses :mod:`ast` only -- there
    is no ``eval``/``exec`` and no access to builtins or globals, so it cannot execute
    arbitrary code.

    Args:
        expr: The candidate arithmetic expression (glyphs ``x``/``×``/``÷`` are normalised).

    Returns:
        The numeric value as a ``float``, or ``None`` when ``expr`` is not a pure,
        evaluable arithmetic expression.
    """
    if expr is None:
        return None
    text = _normalise(expr)
    if not text:
        return None
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    return _eval_node(tree)


def _eval_node(node: ast.AST) -> Optional[float]:
    """Recursively evaluate a whitelisted AST node, returning ``None`` if disallowed."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        # bool is a subclass of int; reject it so "True"/"False" are not arithmetic.
        if isinstance(node.value, bool):
            return None
        if isinstance(node.value, (int, float)):
            return float(node.value)
        return None

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _UNARY_OPS):
        operand = _eval_node(node.operand)
        if operand is None:
            return None
        return +operand if isinstance(node.op, ast.UAdd) else -operand

    if isinstance(node, ast.BinOp) and isinstance(node.op, _BIN_OPS):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if left is None or right is None:
            return None
        return _apply_bin_op(node.op, left, right)

    # Any other node type (Name, Call, Attribute, Subscript, comprehension, ...) is
    # disallowed -- this is what makes the evaluator safe.
    return None


def _apply_bin_op(op: ast.operator, left: float, right: float) -> Optional[float]:
    """Apply a whitelisted binary operator, guarding division/modulo/pow edge cases."""
    try:
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return None if right == 0 else left / right
        if isinstance(op, ast.FloorDiv):
            return None if right == 0 else float(left // right)
        if isinstance(op, ast.Mod):
            return None if right == 0 else float(left % right)
        if isinstance(op, ast.Pow):
            if abs(right) > _MAX_EXPONENT:
                return None  # guard against pathological exponentiation
            return float(left**right)
    except (ValueError, OverflowError, ZeroDivisionError, MemoryError):
        return None
    return None


def parse_equation(text: str) -> Optional[tuple[str, float]]:
    """Extract an ``"<expr> = <result>"`` arithmetic claim from ``text``.

    Splits on the *last* ``=`` so a chained ``a = b = c`` keeps the final result on the
    right. The left side is returned verbatim (as the expression to evaluate) and the right
    side must itself reduce to a plain number via :func:`safe_eval_arithmetic`.

    Args:
        text: A step's text or logic form, for example ``"7 * 8 = 56"``.

    Returns:
        ``(expression, result)`` when a checkable equation is present, else ``None``. No
        equality sign, an empty left side, or a right side that is not a number all yield
        ``None``.
    """
    if not text or "=" not in text:
        return None
    lhs, _, rhs = text.rpartition("=")
    lhs = lhs.strip()
    if not lhs.strip():
        return None
    result = safe_eval_arithmetic(rhs)
    if result is None:
        return None
    return lhs, result


__all__ = ["safe_eval_arithmetic", "parse_equation"]
