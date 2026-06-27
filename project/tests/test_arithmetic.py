"""Tests for the safe arithmetic evaluator and equation parser (demo Phase 2).

These prove two things: (1) :func:`safe_eval_arithmetic` computes the right value for real
arithmetic, and (2) it is *safe* -- it never executes arbitrary code. Names, calls,
attribute access, subscripts, comprehensions, strings and ``__import__`` all return
``None`` instead of being evaluated. :func:`parse_equation` extracts ``(expr, result)``
from an equation string and returns ``None`` when there is nothing checkable. Fully
offline; no ``eval``/``exec`` anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from arithmetic import parse_equation, safe_eval_arithmetic  # noqa: E402


# --------------------------------------------------------------------------- #
# safe_eval_arithmetic -- correctness
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("7*8", 56.0),
        ("7 * 8", 56.0),
        ("24 - 9", 15.0),
        ("144 / 12", 12.0),
        ("2 + 3 * 4", 14.0),  # precedence
        ("(2 + 3) * 4", 20.0),  # parentheses
        ("-5 + 2", -3.0),  # unary minus
        ("+5", 5.0),  # unary plus
        ("2 ** 10", 1024.0),  # power
        ("17 // 5", 3.0),  # floor division
        ("17 % 5", 2.0),  # modulo
        ("3.5 * 2", 7.0),  # floats
        ("1,000 + 1", 1001.0),  # thousands separator stripped
        ("$20 - $5", 15.0),  # currency stripped
    ],
)
def test_safe_eval_computes_correct_value(expr, expected):
    assert safe_eval_arithmetic(expr) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("7 x 8", 56.0),
        ("7 × 8", 56.0),
        ("7 · 8", 56.0),
        ("(7) x (8)", 56.0),
        ("84 ÷ 12", 7.0),
    ],
)
def test_safe_eval_normalises_times_and_divide_glyphs(expr, expected):
    assert safe_eval_arithmetic(expr) == pytest.approx(expected)


@pytest.mark.parametrize("expr", ["5 / 0", "5 // 0", "5 % 0"])
def test_safe_eval_division_by_zero_is_none_not_raise(expr):
    assert safe_eval_arithmetic(expr) is None


def test_safe_eval_pathological_exponent_is_rejected():
    # A huge exponent must not hang or exhaust memory; it is simply invalid.
    assert safe_eval_arithmetic("10 ** 100000") is None


# --------------------------------------------------------------------------- #
# safe_eval_arithmetic -- SAFETY (no code execution)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "malicious",
    [
        "__import__('os')",
        "__import__('os').system('echo hi')",
        "os.system('echo hi')",
        "open('secret.txt').read()",
        "().__class__.__bases__",
        "(1).__class__",
        "[x for x in range(3)]",
        "{'a': 1}",
        "lambda: 1",
        "print(1)",
        "eval('1+1')",
        "x",
        "x + 1",
        "abs(-3)",
        "'string'",
        "True",
        "False",
        "None",
        "1; import os",
        "data[0]",
    ],
)
def test_safe_eval_rejects_non_arithmetic_and_never_executes(malicious):
    assert safe_eval_arithmetic(malicious) is None


def test_safe_eval_does_not_execute_side_effects(tmp_path):
    # If the evaluator ever executed code, this would create the sentinel file. It must
    # not: the call returns None and the file is never written.
    sentinel = tmp_path / "pwned.txt"
    payload = f"__import__('pathlib').Path({str(sentinel)!r}).write_text('x')"
    assert safe_eval_arithmetic(payload) is None
    assert not sentinel.exists()


@pytest.mark.parametrize("expr", ["", "   ", None, "=", "+", "* /"])
def test_safe_eval_handles_empty_and_garbage(expr):
    assert safe_eval_arithmetic(expr) is None


# --------------------------------------------------------------------------- #
# parse_equation
# --------------------------------------------------------------------------- #


def test_parse_equation_extracts_expr_and_result():
    parsed = parse_equation("7 * 8 = 56")
    assert parsed is not None
    expr, result = parsed
    assert expr == "7 * 8"
    assert result == pytest.approx(56.0)


def test_parse_equation_uses_last_equals_sign():
    # "a = b = c" -> expression is "a = b", result is c.
    parsed = parse_equation("3 + 1 = 4 = 4")
    assert parsed is not None
    expr, result = parsed
    assert expr == "3 + 1 = 4"
    assert result == pytest.approx(4.0)


@pytest.mark.parametrize(
    "text",
    [
        "no equals here",
        "",
        "= 5",  # empty left side
        "7 * 8 = fifty six",  # non-numeric right side
        "foo = bar",
    ],
)
def test_parse_equation_returns_none_when_not_checkable(text):
    assert parse_equation(text) is None


def test_parse_equation_result_may_be_numeric_expression():
    # The right side reduces to a number via the safe evaluator.
    parsed = parse_equation("28 + 28 = 7 * 8")
    assert parsed is not None
    expr, result = parsed
    assert result == pytest.approx(56.0)
