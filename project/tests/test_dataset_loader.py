"""Unit tests for the Dataset Loader (Task 12.1).

These cover the example behaviors and edge cases of dataset loading and validation
specified in Requirement 10: field validation, domain-label recognition, identifier
uniqueness, retention of valid items, exclusion logging, and per-domain statistics.

The dedicated property test for the retained/excluded partitioning is Task 12.2.
"""

from __future__ import annotations

from nsr.dataset_loader import (
    DatasetLoader,
    ExclusionRecord,
    load_dataset,
)
from nsr.models import DatasetItem, Domain


def _valid(item_id: str, domain: str = "mathematical-reasoning") -> dict:
    return {
        "item_id": item_id,
        "query": "What is 2 + 2?",
        "ground_truth": "4",
        "domain": domain,
    }


def test_all_valid_items_are_retained():
    raw = [
        _valid("m-001", "mathematical-reasoning"),
        _valid("c-001", "commonsense-reasoning"),
        _valid("l-001", "legal-question-answering"),
    ]

    result = load_dataset(raw)

    assert [i.item_id for i in result.items] == ["m-001", "c-001", "l-001"]
    assert all(isinstance(i, DatasetItem) for i in result.items)
    assert result.report.total_loaded == 3
    assert result.report.total_validated == 3
    assert result.report.total_excluded == 0
    assert result.report.exclusions == []


def test_each_retained_item_has_recognized_domain_enum():
    result = load_dataset([_valid("m-001", "science-reasoning")])
    assert result.items[0].domain is Domain.SCIENCE
    # Associated with exactly one of the six domains (Req 10.4).
    assert result.items[0].domain in set(Domain)


def test_missing_item_id_is_excluded_and_logged():
    raw = [{"query": "q", "ground_truth": "a", "domain": "mathematical-reasoning"}]
    result = load_dataset(raw)

    assert result.items == []
    assert result.report.total_excluded == 1
    rec = result.report.exclusions[0]
    assert isinstance(rec, ExclusionRecord)
    assert rec.missing_field == "item_id"


def test_empty_query_is_excluded_with_field_name():
    raw = [
        {
            "item_id": "m-001",
            "query": "   ",
            "ground_truth": "4",
            "domain": "mathematical-reasoning",
        }
    ]
    result = load_dataset(raw)

    assert result.items == []
    rec = result.report.exclusions[0]
    assert rec.item_id == "m-001"
    assert rec.missing_field == "query"


def test_missing_ground_truth_is_excluded():
    raw = [{"item_id": "m-001", "query": "q", "domain": "mathematical-reasoning"}]
    result = load_dataset(raw)

    assert result.items == []
    assert result.report.exclusions[0].missing_field == "ground_truth"


def test_missing_domain_is_excluded():
    raw = [{"item_id": "m-001", "query": "q", "ground_truth": "4"}]
    result = load_dataset(raw)

    assert result.items == []
    assert result.report.exclusions[0].missing_field == "domain"


def test_unrecognized_domain_label_is_excluded_and_logged():
    raw = [_valid("x-001", "astrology")]
    result = load_dataset(raw)

    assert result.items == []
    rec = result.report.exclusions[0]
    assert rec.item_id == "x-001"
    assert rec.bad_label == "astrology"
    assert rec.missing_field is None
    # Could not be attributed to a recognized domain.
    assert result.report.unassigned_loaded == 1
    assert result.report.unassigned_excluded == 1


def test_duplicate_item_id_is_excluded():
    raw = [_valid("dup"), _valid("dup")]
    result = load_dataset(raw)

    # First retained, second excluded as duplicate.
    assert [i.item_id for i in result.items] == ["dup"]
    assert result.report.total_excluded == 1
    rec = result.report.exclusions[0]
    assert "duplicate" in rec.reason
    assert rec.missing_field is None and rec.bad_label is None


def test_domain_enum_label_is_accepted():
    raw = [
        {
            "item_id": "m-001",
            "query": "q",
            "ground_truth": "4",
            "domain": Domain.LOGIC_PUZZLE,
        }
    ]
    result = load_dataset(raw)
    assert result.items[0].domain is Domain.LOGIC_PUZZLE


def test_per_domain_statistics_are_recorded():
    raw = [
        _valid("m-001", "mathematical-reasoning"),
        _valid("m-002", "mathematical-reasoning"),
        {  # excluded math item (missing ground truth) still counts as loaded for math
            "item_id": "m-003",
            "query": "q",
            "domain": "mathematical-reasoning",
        },
        _valid("s-001", "science-reasoning"),
        _valid("bad", "not-a-domain"),
    ]

    result = load_dataset(raw)
    report = result.report

    assert report.total_loaded == 5
    assert report.total_validated == 3
    assert report.total_excluded == 2

    math = report.per_domain[Domain.MATH]
    assert math.loaded == 3
    assert math.validated == 2
    assert math.excluded == 1

    science = report.per_domain[Domain.SCIENCE]
    assert science.loaded == 1
    assert science.validated == 1
    assert science.excluded == 0

    # The unrecognized-domain item is not attributed to any of the six domains.
    assert report.unassigned_loaded == 1
    assert report.unassigned_excluded == 1
    # Every one of the six domains has a stats entry.
    assert set(report.per_domain.keys()) == set(Domain)


def test_validation_order_reports_first_problem():
    # Missing id takes precedence over a bad domain label.
    raw = [{"query": "q", "ground_truth": "a", "domain": "astrology"}]
    result = load_dataset(raw)
    assert result.report.exclusions[0].missing_field == "item_id"


def test_retained_and_excluded_partition_the_input():
    raw = [
        _valid("ok-1"),
        _valid("bad", "nope"),
        _valid("ok-2", "legal-question-answering"),
        {"item_id": "", "query": "q", "ground_truth": "a", "domain": "science-reasoning"},
    ]
    result = load_dataset(raw)

    retained = len(result.items)
    excluded = len(result.report.exclusions)
    assert retained + excluded == len(raw)
    assert result.report.total_validated == retained
    assert result.report.total_excluded == excluded


def test_loader_instance_is_reusable_across_calls():
    loader = DatasetLoader()
    first = loader.load([_valid("dup")])
    second = loader.load([_valid("dup")])
    # Uniqueness state does not leak across separate load calls.
    assert [i.item_id for i in first.items] == ["dup"]
    assert [i.item_id for i in second.items] == ["dup"]
