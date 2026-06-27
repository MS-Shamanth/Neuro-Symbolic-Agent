"""Tests for ``run_benchmark.numeric_answer_match`` (demo Phase 1).

Fully offline. The matcher extracts the LAST number from a (possibly verbose) prediction
and compares it numerically to the ground truth, tolerant of commas, ``$``, a trailing
period, and a trailing ``= N`` equation form. It is applied symmetrically to the System
and the baselines in the GSM8K run.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

run_benchmark = importlib.import_module("run_benchmark")


def test_matches_verbose_prediction():
    assert run_benchmark.numeric_answer_match("The answer is 72.", "72") is True


def test_matches_bare_number():
    assert run_benchmark.numeric_answer_match("72", "72") is True


def test_matches_thousands_separator():
    assert run_benchmark.numeric_answer_match("1,200", "1200") is True


def test_matches_dollar_prefix():
    assert run_benchmark.numeric_answer_match("$5", "5") is True


def test_matches_equation_form():
    assert run_benchmark.numeric_answer_match("x = 14", "14") is True


def test_rejects_wrong_number():
    assert run_benchmark.numeric_answer_match("The answer is 73.", "72") is False


def test_rejects_when_no_number():
    assert run_benchmark.numeric_answer_match("no number here", "72") is False


def test_decimal_equivalence():
    assert run_benchmark.numeric_answer_match("18.0", "18") is True
