"""Tests for the GSM8K dataset loader and numeric answer matching (demo Phase 1)."""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

import datasets  # noqa: E402
from nsr.models import Domain  # noqa: E402


def test_load_sample_returns_math_items():
    items = datasets.load_sample()
    assert len(items) >= 8
    for it in items:
        assert it.domain is Domain.MATH
        assert it.item_id and it.query and it.ground_truth
    # The first bundled problem's final answer is 15.
    assert items[0].ground_truth == "15"


def test_limit_caps_item_count():
    assert len(datasets.load_sample(limit=3)) == 3


def test_load_gsm8k_jsonl_parses_hash_marker(tmp_path):
    path = tmp_path / "mini.jsonl"
    path.write_text(
        '{"question": "2+2?", "answer": "two and two ... #### 4"}\n'
        '{"question": "big?", "answer": "computing ... #### 1,234"}\n'
        "\n",  # blank line tolerated
        encoding="utf-8",
    )
    items = datasets.load_gsm8k_jsonl(path)
    assert [i.ground_truth for i in items] == ["4", "1234"]
    assert items[0].domain is Domain.MATH


def test_items_missing_final_number_are_skipped(tmp_path):
    path = tmp_path / "mini.jsonl"
    path.write_text(
        '{"question": "ok?", "answer": "no marker here"}\n'
        '{"question": "good?", "answer": "#### 7"}\n',
        encoding="utf-8",
    )
    items = datasets.load_gsm8k_jsonl(path)
    assert [i.ground_truth for i in items] == ["7"]


def test_numeric_answer_match_handles_verbose_predictions():
    assert datasets.numeric_answer_match("The answer is 18.", "18") is True
    assert datasets.numeric_answer_match("#### 18", "18") is True
    assert datasets.numeric_answer_match("18.0", "18") is True
    assert datasets.numeric_answer_match("1,234 dollars", "1234") is True
    assert datasets.numeric_answer_match("Answer: 19", "18") is False
    assert datasets.numeric_answer_match("no number", "18") is False


def test_extract_final_number_prefers_marker_over_trailing_text():
    # The number after the last "####" wins even if other numbers follow.
    assert datasets.extract_final_number("steps 3 and 4 #### 12 done") == 12.0
