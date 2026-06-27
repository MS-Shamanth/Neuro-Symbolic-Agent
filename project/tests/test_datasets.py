"""Tests for the standard-benchmark dataset loaders (demo Phase 1).

Fully offline: every loader reads a small temp JSONL written by the test (or the bundled
sample). They confirm the official GSM8K ``#### N`` format parses, multiple-choice and
StrategyQA shapes parse, ``limit`` caps the count, and malformed lines are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

import datasets  # noqa: E402
from nsr.models import Domain  # noqa: E402


# --------------------------------------------------------------------------- #
# load_gsm8k — official "#### N" format
# --------------------------------------------------------------------------- #


def test_load_gsm8k_parses_official_hash_marker(tmp_path):
    """The official GSM8K final-answer marker is parsed; commas are stripped."""
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"question": "Two and two?", "answer": "2 + 2 = 4\\n#### 4"}\n'
        '{"question": "A big total?", "answer": "adds up ...\\n#### 1,234"}\n',
        encoding="utf-8",
    )
    items = datasets.load_gsm8k(path)

    assert [i.ground_truth for i in items] == ["4", "1234"]
    assert all(i.domain is Domain.MATH for i in items)
    assert all(i.item_id.startswith("gsm8k-") for i in items)
    assert items[0].query == "Two and two?"


def test_load_gsm8k_strips_dollar_and_applies_limit(tmp_path):
    """A ``$``-prefixed final number is cleaned, and ``limit`` keeps the first N items."""
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"question": "q1", "answer": "#### $5"}\n'
        '{"question": "q2", "answer": "#### 6"}\n'
        '{"question": "q3", "answer": "#### 7"}\n',
        encoding="utf-8",
    )
    items = datasets.load_gsm8k(path, limit=2)

    assert len(items) == 2
    assert [i.ground_truth for i in items] == ["5", "6"]


def test_load_gsm8k_skips_malformed_and_unmarked_lines(tmp_path):
    """Malformed JSON lines and answers with no ``####`` marker are skipped."""
    path = tmp_path / "test.jsonl"
    path.write_text(
        "not valid json\n"
        "\n"
        '{"question": "no marker", "answer": "just prose"}\n'
        '{"question": "good", "answer": "#### 42"}\n',
        encoding="utf-8",
    )
    items = datasets.load_gsm8k(path)

    assert [i.ground_truth for i in items] == ["42"]


def test_bundled_sample_loads_math_items():
    """The bundled sample loads to >= 8 MATH items with numeric ground truths."""
    items = datasets.load_benchmark("gsm8k")  # path=None -> bundled sample

    assert len(items) >= 8
    for it in items:
        assert it.domain is Domain.MATH
        assert it.item_id and it.query and it.ground_truth
        assert float(it.ground_truth) == float(it.ground_truth)  # parses as a number


def test_bundled_sample_has_expanded_set_with_numeric_truths():
    """The expanded original sample now carries ~30 MATH items, all numeric."""
    items = datasets.load_benchmark("gsm8k")  # path=None -> bundled sample

    # The sample was expanded from 10 to ~30 original multi-step problems.
    assert len(items) >= 28
    for it in items:
        assert it.domain is Domain.MATH
        # Every ground truth is a non-empty, parseable number.
        assert it.ground_truth.strip()
        value = float(it.ground_truth)
        assert value == value  # not NaN



# --------------------------------------------------------------------------- #
# load_multiple_choice — ARC / CommonsenseQA shape variants
# --------------------------------------------------------------------------- #


def test_load_multiple_choice_nested_shape(tmp_path):
    """The nested ``question.stem`` + ``choices:[{label,text}]`` + ``answerKey`` shape."""
    path = tmp_path / "mc.jsonl"
    path.write_text(
        '{"id": "x1", "question": {"stem": "What color is the sky?", '
        '"choices": [{"label": "A", "text": "green"}, {"label": "B", "text": "blue"}]}, '
        '"answerKey": "B"}\n',
        encoding="utf-8",
    )
    items = datasets.load_multiple_choice(path)

    assert len(items) == 1
    item = items[0]
    assert item.ground_truth == "blue"
    assert item.domain is Domain.COMMONSENSE
    assert "What color is the sky?" in item.query
    assert "A) green" in item.query and "B) blue" in item.query


def test_load_multiple_choice_flat_shape_and_domain(tmp_path):
    """The flat ``{question, choices, answerKey}`` shape with a configurable domain."""
    path = tmp_path / "mc.jsonl"
    path.write_text(
        '{"question": "2+2?", "choices": ["3", "4", "5"], "answerKey": "B"}\n',
        encoding="utf-8",
    )
    items = datasets.load_multiple_choice(path, domain=Domain.SCIENCE)

    assert len(items) == 1
    assert items[0].ground_truth == "4"
    assert items[0].domain is Domain.SCIENCE


def test_load_multiple_choice_skips_unresolvable(tmp_path):
    """Lines with no resolvable answer key are skipped; valid ones are kept."""
    path = tmp_path / "mc.jsonl"
    path.write_text(
        '{"question": {"stem": "?", "choices": [{"label": "A", "text": "x"}]}, '
        '"answerKey": "Z"}\n'  # Z is not a label -> skipped
        '{"question": {"stem": "ok?", "choices": [{"label": "A", "text": "yes"}]}, '
        '"answerKey": "A"}\n',
        encoding="utf-8",
    )
    items = datasets.load_multiple_choice(path)

    assert [i.ground_truth for i in items] == ["yes"]


# --------------------------------------------------------------------------- #
# load_strategyqa — boolean answers
# --------------------------------------------------------------------------- #


def test_load_strategyqa_maps_booleans(tmp_path):
    """``answer: true/false`` maps to ``"yes"/"no"`` with the multi-hop domain."""
    path = tmp_path / "sqa.jsonl"
    path.write_text(
        '{"question": "Is the sky blue?", "answer": true}\n'
        '{"question": "Do fish fly?", "answer": false}\n'
        '{"question": "no answer here"}\n',  # skipped: no boolean answer
        encoding="utf-8",
    )
    items = datasets.load_strategyqa(path)

    assert [i.ground_truth for i in items] == ["yes", "no"]
    assert all(i.domain is Domain.MULTI_HOP for i in items)


# --------------------------------------------------------------------------- #
# load_benchmark dispatcher
# --------------------------------------------------------------------------- #


def test_load_benchmark_dispatches_and_limits(tmp_path):
    """The dispatcher routes by name and applies the limit."""
    path = tmp_path / "sqa.jsonl"
    path.write_text(
        '{"question": "q1", "answer": true}\n'
        '{"question": "q2", "answer": false}\n',
        encoding="utf-8",
    )
    items = datasets.load_benchmark("strategyqa", path=path, limit=1)
    assert len(items) == 1


def test_load_benchmark_unknown_name_raises():
    import pytest

    with pytest.raises(ValueError):
        datasets.load_benchmark("nope")


def test_load_benchmark_requires_path_for_non_gsm8k():
    import pytest

    with pytest.raises(ValueError):
        datasets.load_benchmark("strategyqa")  # path required
