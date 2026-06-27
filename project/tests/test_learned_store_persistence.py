"""Durable persistence tests for the versioned Learned_Rule_Store (Req 14.7).

These are concrete example/integration tests (the property-based round-trip of
``store_to_dict``/``store_from_dict`` is covered elsewhere). They exercise the durable
JSON persistence path end to end:

- A non-trivial store (two candidates with provenance + witnesses, plus a promoted
  learned rule) is written to a real file on disk via
  ``ReproducibilityManager.persist_learned_rule_store`` / ``RuleLearner.persist_store``,
  read back with ``json.load``, and reconstructed via ``store_from_dict`` so the
  reloaded store equals the original, *including the version identifier*.
- The failure path returns an :class:`ErrorRecord` naming the failed persistence
  operation/component rather than raising.

Validates: Requirements 14.7
"""

from __future__ import annotations

import json

import pytest

from nsr.models import ErrorRecord
from nsr.models.learning import (
    CandidateRule,
    LearnedRule,
    LearnedRuleStore,
    ProductionRule,
    RuleOrigin,
    RuleProvenance,
    store_from_dict,
    store_to_dict,
)
from nsr.models.reasoning import SymbolicRepresentation
from nsr.reproducibility import ReproducibilityManager
from nsr.rule_learner import RuleLearner
from nsr.validation_engine import ValidationEngine


def _build_nontrivial_store() -> LearnedRuleStore:
    """A store with a couple of candidates (provenance + witnesses) and a learned rule."""
    candidate_a = CandidateRule(
        rule=ProductionRule(
            rule_id="cand-mortal",
            condition="IF human(x)",
            action="THEN mortal(x)",
        ),
        provenance=RuleProvenance(
            trace_ids=["trace-1", "trace-2"],
            step_ids=[3, 7],
            induction_seed=4242,
        ),
        corroboration_count=2,
        normalized_key="human::mortal",
        witnesses=[
            SymbolicRepresentation(
                logic_form="human(socrates)",
                predicates={"subject": "socrates", "type": "human"},
                source_text="Socrates is a human.",
            ),
            SymbolicRepresentation(
                logic_form="human(plato)",
                predicates={"subject": "plato", "type": "human"},
                source_text="Plato is a human.",
            ),
        ],
    )
    candidate_b = CandidateRule(
        rule=ProductionRule(
            rule_id="cand-sky",
            condition="IF cloudless(sky)",
            action="THEN blue(sky)",
        ),
        provenance=RuleProvenance(
            trace_ids=["trace-9"],
            step_ids=[1],
            induction_seed=None,
        ),
        corroboration_count=1,
        normalized_key="cloudless::blue",
        witnesses=[
            SymbolicRepresentation(
                logic_form="cloudless(sky)",
                predicates={"state": "cloudless"},
                source_text="The sky is cloudless today.",
            ),
        ],
    )
    learned = LearnedRule(
        rule=ProductionRule(
            rule_id="learned-mortal",
            condition="IF human(x)",
            action="THEN mortal(x)",
        ),
        provenance=RuleProvenance(
            trace_ids=["trace-1", "trace-2", "trace-5"],
            step_ids=[3, 7, 11],
            induction_seed=4242,
        ),
        origin=RuleOrigin.LEARNED,
    )
    return LearnedRuleStore(
        version=7,
        candidates={
            candidate_a.normalized_key: candidate_a,
            candidate_b.normalized_key: candidate_b,
        },
        learned_rules=[learned],
    )


def _reload(path) -> LearnedRuleStore:
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    return store_from_dict(document)


def test_persist_learned_rule_store_round_trips_through_disk(tmp_path):
    """The versioned store writes to disk and reloads equal, including the version."""
    store = _build_nontrivial_store()
    output_path = tmp_path / "learned_store.json"

    manager = ReproducibilityManager()
    result = manager.persist_learned_rule_store(store, output_path)

    # Success returns None and never raises.
    assert result is None
    assert output_path.exists()

    reloaded = _reload(output_path)
    assert reloaded == store
    # The version identifier specifically survives the round trip.
    assert reloaded.version == store.version == 7


def test_persisted_document_carries_format_and_store_version(tmp_path):
    """The on-disk JSON is the versioned store_to_dict schema (Req 14.7)."""
    store = _build_nontrivial_store()
    output_path = tmp_path / "learned_store.json"

    assert ReproducibilityManager().persist_learned_rule_store(store, output_path) is None

    with open(output_path, "r", encoding="utf-8") as handle:
        document = json.load(handle)

    assert document == store_to_dict(store)
    assert document["version"] == 7
    assert "format_version" in document
    assert set(document["candidates"]) == {"human::mortal", "cloudless::blue"}


def test_rule_learner_persist_store_round_trips_through_disk(tmp_path):
    """RuleLearner.persist_store wraps the manager and persists its own store losslessly."""
    store = _build_nontrivial_store()
    learner = RuleLearner(store=store, validation=ValidationEngine())
    manager = ReproducibilityManager()
    output_path = tmp_path / "learner_store.json"

    result = learner.persist_store(manager, output_path)

    assert result is None
    assert _reload(output_path) == store


def test_write_failure_returns_error_record_naming_operation_without_raising(tmp_path):
    """Pointing output at a directory yields an ErrorRecord, not an exception (Req 13.5)."""
    store = _build_nontrivial_store()
    manager = ReproducibilityManager()

    # A directory cannot be opened for writing as a file; the open/write must fail.
    target_dir = tmp_path / "a_directory"
    target_dir.mkdir()

    result = manager.persist_learned_rule_store(store, target_dir)

    assert isinstance(result, ErrorRecord)
    assert result.failed_component == "ReproducibilityManager"
    # The reason names the failed persistence operation.
    assert "learned-rule-store-persistence" in result.reason
    # Nothing valid should have been written into the directory path itself.
    assert target_dir.is_dir()


def test_serialization_failure_returns_error_record_without_raising(tmp_path, monkeypatch):
    """A serialization failure is reported as an ErrorRecord, never raised."""
    store = _build_nontrivial_store()
    manager = ReproducibilityManager()
    output_path = tmp_path / "should_not_exist.json"

    def _boom(*args, **kwargs):
        raise TypeError("intentional serialization failure")

    # Force the JSON serialization step inside the manager to fail.
    monkeypatch.setattr("nsr.reproducibility.json.dumps", _boom)

    result = manager.persist_learned_rule_store(store, output_path)

    assert isinstance(result, ErrorRecord)
    assert result.failed_component == "ReproducibilityManager"
    assert "learned-rule-store-persistence" in result.reason
    assert "intentional serialization failure" in result.reason
    # The failed write must not leave a partial file behind.
    assert not output_path.exists()


def test_rule_learner_persist_store_failure_returns_error_record(tmp_path):
    """RuleLearner.persist_store also surfaces failures as an ErrorRecord, not a raise."""
    store = _build_nontrivial_store()
    learner = RuleLearner(store=store, validation=ValidationEngine())
    manager = ReproducibilityManager()

    target_dir = tmp_path / "dir_target"
    target_dir.mkdir()

    result = learner.persist_store(manager, target_dir)

    assert isinstance(result, ErrorRecord)
    assert "learned-rule-store-persistence" in result.reason
