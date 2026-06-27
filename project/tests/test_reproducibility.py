"""Unit tests for the Reproducibility Manager (Task 2.5).

Covers the three behaviours called out by the task:
  1. A seed is generated when ``config.random_seed`` is absent, and the resolved seed
     is applied and recorded (Req 13.3).
  2. A supplied seed is applied to stochastic operations, yielding deterministic
     results from Python's ``random`` after :meth:`apply_seed` (Req 13.2).
  3. A persistence failure returns an :class:`ErrorRecord` naming the failed
     persistence operation instead of raising (Req 13.5).
"""

from __future__ import annotations

import json
import random

import pytest

from nsr.models import ErrorRecord, RunRecord, SystemConfig
from nsr.reproducibility import ReproducibilityManager


def _make_config(random_seed=None) -> SystemConfig:
    return SystemConfig(
        max_cycle_limit=10,
        repair_attempt_limit=3,
        retry_count=2,
        llm_selection="hosted",
        output_format="json",
        conflict_resolution_policy="priority",
        generation_timeout_ms=30000,
        random_seed=random_seed,
    )


# --------------------------------------------------------------------- Req 13.3
# A seed is generated when none is supplied, then applied and recorded.


def test_resolve_seed_generates_when_absent():
    manager = ReproducibilityManager()
    cfg = _make_config(random_seed=None)

    seed = manager.resolve_seed(cfg)

    assert isinstance(seed, int)
    assert seed >= 0
    assert seed <= 2**63 - 1


def test_generated_seeds_are_non_constant():
    # A generated seed should come from a non-deterministic source, so two resolutions
    # without a supplied seed should (overwhelmingly likely) differ.
    manager = ReproducibilityManager()
    cfg = _make_config(random_seed=None)

    seeds = {manager.resolve_seed(cfg) for _ in range(20)}

    assert len(seeds) > 1


def test_seed_everything_generates_applies_and_records():
    manager = ReproducibilityManager()
    cfg = _make_config(random_seed=None)

    applied = manager.seed_everything(cfg)

    # The applied seed is recorded as the effective seed (Req 13.3).
    assert manager.effective_seed == applied
    assert isinstance(applied, int)


def test_generated_seed_is_recorded_in_run_record():
    manager = ReproducibilityManager()
    cfg = _make_config(random_seed=None)

    applied = manager.seed_everything(cfg)
    record = manager.build_run_record(
        config=cfg,
        dataset_ids=["math-v1"],
        model_id="gpt-x",
    )

    assert isinstance(record, RunRecord)
    assert record.seed == applied


# --------------------------------------------------------------------- Req 13.2
# A supplied seed is used as-is and applied to stochastic operations.


def test_resolve_seed_uses_supplied_value():
    manager = ReproducibilityManager()
    cfg = _make_config(random_seed=12345)

    assert manager.resolve_seed(cfg) == 12345


def test_apply_seed_makes_python_random_deterministic():
    manager = ReproducibilityManager()

    manager.apply_seed(777)
    first = [random.random() for _ in range(5)]

    manager.apply_seed(777)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_apply_seed_records_effective_seed_and_returns_it():
    manager = ReproducibilityManager()

    returned = manager.apply_seed(42)

    assert returned == 42
    assert manager.effective_seed == 42


def test_seed_everything_with_supplied_seed_applies_to_hooks():
    manager = ReproducibilityManager()
    received: list[int] = []
    manager.register_seed_hook(received.append)

    cfg = _make_config(random_seed=99)
    applied = manager.seed_everything(cfg)

    assert applied == 99
    # The hook is invoked with the effective seed so component RNGs are seeded too.
    assert received == [99]


def test_hook_registered_after_apply_is_seeded_immediately():
    manager = ReproducibilityManager()
    manager.apply_seed(55)

    received: list[int] = []
    manager.register_seed_hook(received.append)

    assert received == [55]


# --------------------------------------------------------------------- Req 13.5
# Persistence failure returns an ErrorRecord naming the failed operation.


def _build_record(manager: ReproducibilityManager) -> RunRecord:
    cfg = _make_config(random_seed=7)
    manager.seed_everything(cfg)
    return manager.build_run_record(
        config=cfg,
        dataset_ids=["math-v1"],
        model_id="gpt-x",
    )


def test_persist_returns_error_record_on_failure(tmp_path):
    manager = ReproducibilityManager()
    record = _build_record(manager)

    # A directory cannot be opened for writing as a file, forcing a write failure.
    bad_path = tmp_path / "a-directory"
    bad_path.mkdir()

    result = manager.persist(record, metrics={"accuracy": 1.0}, output_path=bad_path)

    assert isinstance(result, ErrorRecord)
    assert result.failed_component == "ReproducibilityManager"
    # The reason names the failed persistence operation.
    assert "persistence" in result.reason.lower()


def test_persist_failure_does_not_raise(monkeypatch, tmp_path):
    manager = ReproducibilityManager()
    record = _build_record(manager)

    # Simulate a serialization failure mid-persist; persist must catch and report it.
    def boom(*_args, **_kwargs):
        raise RuntimeError("serialization blew up")

    monkeypatch.setattr(json, "dumps", boom)

    result = manager.persist(
        record, metrics={"accuracy": 1.0}, output_path=tmp_path / "out.json"
    )

    assert isinstance(result, ErrorRecord)
    assert result.failed_component == "ReproducibilityManager"
    assert "serialization blew up" in result.reason


def test_persist_succeeds_and_returns_none(tmp_path):
    manager = ReproducibilityManager()
    record = _build_record(manager)
    out = tmp_path / "nested" / "run.json"

    result = manager.persist(record, metrics={"accuracy": 0.9}, output_path=out)

    assert result is None
    assert out.exists()
    document = json.loads(out.read_text(encoding="utf-8"))
    assert "run_record" in document
    assert "metrics" in document
    assert document["run_record"]["seed"] == record.seed
