"""Tests for the REASONING-STATISTICS collector + report (``demo/reasoning_stats.py``).

Two layers are covered, both fully offline (no Ollama, no network):

- :func:`reasoning_stats.aggregate_trace_stats` is exercised directly over hand-built
  synthetic :class:`~nsr.models.ProofTrace`s that hit every step category — a clean
  accepted-first-pass step, a plain rejected step, a repaired-successfully step, and a
  repair-failed (repair-exhausted) step — asserting every counter, the rule-utilization
  Counter, the repair-violation Counter, and the derived repair-success rate.
- :func:`reasoning_stats.generate_reasoning_stats` is run end-to-end with the backend
  factory monkeypatched to a canned wrong-then-correct equation (so a real repair is
  recorded) and the GSM8K dataset loader stubbed, asserting the HTML + JSON reports are
  written and carry the headcount and rule-utilization sections.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections import Counter
from pathlib import Path

# The demo modules live in ``project/demo`` (a sibling of ``tests``); add to the path.
_DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

reasoning_stats = importlib.import_module("reasoning_stats")
ablation = importlib.import_module("ablation")

from nsr.llm_component import MockBackend  # noqa: E402
from nsr.models import (  # noqa: E402
    DatasetItem,
    Domain,
    ProofStep,
    ProofTrace,
    RepairAttempt,
    SymbolicRepresentation,
    TerminationReason,
    ValidationStatus,
)


# --------------------------------------------------------------------------- #
# Synthetic-trace builders (one ProofStep per category)
# --------------------------------------------------------------------------- #


def _rep(logic_form: str) -> SymbolicRepresentation:
    return SymbolicRepresentation(logic_form=logic_form, source_text=logic_form)


def _accepted_first_pass_trace() -> ProofTrace:
    """A single ACCEPTED step with no repair attempts (goal satisfied)."""
    step = ProofStep(
        sequence=0,
        step_text="all good",
        representation=_rep("3 + 4 = 7"),
        status=ValidationStatus.ACCEPTED,
        applied_rule_id="well-formed-step",
    )
    return ProofTrace(steps=[step], termination_reason=TerminationReason.GOAL_SATISFIED)


def _plain_rejected_trace() -> ProofTrace:
    """A single REJECTED step with NO repair attempts (and no rule applied)."""
    step = ProofStep(
        sequence=0,
        step_text="bad step",
        representation=_rep("nope"),
        status=ValidationStatus.REJECTED,
        applied_rule_id=None,  # -> counted under "no-rule-applied"
        violated_rule_ids=["consistency-guard"],
    )
    return ProofTrace(
        steps=[step], termination_reason=TerminationReason.CYCLE_LIMIT_REACHED
    )


def _repaired_trace() -> ProofTrace:
    """A REPAIRED step (rejected-then-accepted) with one repair attempt."""
    step = ProofStep(
        sequence=0,
        step_text="repaired step",
        representation=_rep("2 + 2 = 4"),
        status=ValidationStatus.REPAIRED,
        applied_rule_id="well-formed-step",
        repair_attempts=[
            RepairAttempt(
                attempt_index=0,
                rejected_step=_rep("2 + 2 = 5"),
                violated_rule_ids=["arithmetic-correctness"],
                repaired_step=_rep("2 + 2 = 4"),
            )
        ],
    )
    return ProofTrace(steps=[step], termination_reason=TerminationReason.GOAL_SATISFIED)


def _repair_failed_trace() -> ProofTrace:
    """A REJECTED step whose repair attempts never succeeded (repair-exhausted)."""
    step = ProofStep(
        sequence=0,
        step_text="never fixed",
        representation=_rep("7 * 8 = 54"),
        status=ValidationStatus.REJECTED,
        applied_rule_id="well-formed-step",
        violated_rule_ids=["arithmetic-correctness"],
        repair_attempts=[
            RepairAttempt(
                attempt_index=0,
                rejected_step=_rep("7 * 8 = 54"),
                violated_rule_ids=["arithmetic-correctness"],
                repaired_step=None,
            ),
            RepairAttempt(
                attempt_index=1,
                rejected_step=_rep("7 * 8 = 55"),
                violated_rule_ids=["arithmetic-correctness"],
                repaired_step=None,
            ),
        ],
    )
    return ProofTrace(
        steps=[step], termination_reason=TerminationReason.REPAIR_EXHAUSTED
    )


def _goal_rejected_trace() -> ProofTrace:
    """A step rejected by goal-alignment.

    ``goal-alignment`` appears BOTH in the step's own ``violated_rule_ids`` AND in its
    repair attempt's ``violated_rule_ids`` — the per-step trigger count must still be 1
    (no double counting). The step is ultimately repaired into acceptance.
    """
    step = ProofStep(
        sequence=0,
        step_text="wrong quantity for the goal",
        representation=_rep("900 - 40 * 13 = 380"),
        status=ValidationStatus.REPAIRED,
        applied_rule_id="well-formed-step",
        violated_rule_ids=["goal-alignment"],
        repair_attempts=[
            RepairAttempt(
                attempt_index=0,
                rejected_step=_rep("40 * 13 = 520"),
                violated_rule_ids=["goal-alignment"],
                repaired_step=_rep("900 - 40 * 13 = 380"),
            )
        ],
    )
    return ProofTrace(steps=[step], termination_reason=TerminationReason.GOAL_SATISFIED)


def _arithmetic_rejected_trace() -> ProofTrace:
    """A step rejected by arithmetic-correctness (cited only in the step itself)."""
    step = ProofStep(
        sequence=0,
        step_text="wrong arithmetic",
        representation=_rep("7 * 8 = 54"),
        status=ValidationStatus.REJECTED,
        applied_rule_id="well-formed-step",
        violated_rule_ids=["arithmetic-correctness"],
    )
    return ProofTrace(
        steps=[step], termination_reason=TerminationReason.CYCLE_LIMIT_REACHED
    )


# --------------------------------------------------------------------------- #
# aggregate_trace_stats — every category
# --------------------------------------------------------------------------- #


def test_aggregate_trace_stats_counts_every_category():
    """Each step category, both Counters, and the repair-success rate are computed."""
    traces = [
        _accepted_first_pass_trace(),
        _plain_rejected_trace(),
        _repaired_trace(),
        _repair_failed_trace(),
    ]

    stats = reasoning_stats.aggregate_trace_stats(traces)

    # Headcount.
    assert stats.total_questions == 4
    assert stats.total_reasoning_steps == 4
    assert stats.accepted_first_pass == 1
    # Both the plain rejection and the repair-failed step are REJECTED at the end.
    assert stats.rejected == 2
    assert stats.repaired_successfully == 1
    # Only the repair-exhausted step is a repair failure (subset of rejected).
    assert stats.repair_failed == 1

    # Partition invariant: the three buckets sum to the total step count.
    assert (
        stats.accepted_first_pass
        + stats.repaired_successfully
        + stats.rejected
        == stats.total_reasoning_steps
    )
    # repair_failed is a sub-count of rejected, never added on top.
    assert stats.repair_failed <= stats.rejected

    # Rule utilization: three steps used well-formed-step, one had no rule applied.
    assert stats.rule_utilization == Counter(
        {"well-formed-step": 3, "no-rule-applied": 1}
    )

    # Repair-trigger breakdown: one attempt in the repaired trace + two in the failed one.
    assert stats.repair_violation_counts == Counter({"arithmetic-correctness": 3})

    # Termination-reason distribution.
    assert stats.termination_reason_counts == Counter(
        {"goal-satisfied": 2, "cycle-limit-reached": 1, "repair-exhausted": 1}
    )

    # Derived rate: 1 repaired / (1 repaired + 1 failed) = 0.5.
    assert stats.repair_success_rate == 0.5

    # Report-only validation triggers don't disturb the partition. No goal-alignment here;
    # arithmetic-correctness fired for the repaired step and the repair-failed step (once
    # each, per step), so 2 of 4 steps.
    assert stats.goal_validation_triggered == 0
    assert stats.arithmetic_validation_triggered == 2
    assert stats.goal_trigger_rate == 0.0
    assert stats.arithmetic_trigger_rate == 2 / 4
    # The partition still sums to total — the new counters are independent of it.
    assert (
        stats.accepted_first_pass
        + stats.repaired_successfully
        + stats.rejected
        == stats.total_reasoning_steps
    )


def test_repair_success_rate_is_none_without_repairs():
    """With no repaired or repair-failed steps the rate is undefined (None)."""
    stats = reasoning_stats.aggregate_trace_stats([_accepted_first_pass_trace()])

    assert stats.repaired_successfully == 0
    assert stats.repair_failed == 0
    assert stats.repair_success_rate is None


def test_aggregate_trace_stats_empty():
    """An empty trace list yields all-zero counts and an undefined repair rate."""
    stats = reasoning_stats.aggregate_trace_stats([])

    assert stats.total_questions == 0
    assert stats.total_reasoning_steps == 0
    assert stats.repair_success_rate is None
    assert stats.to_dict()["rule_utilization"] == {}


# --------------------------------------------------------------------------- #
# Validation-trigger metrics (goal-alignment / arithmetic-correctness)
# --------------------------------------------------------------------------- #


def test_goal_and_arithmetic_validation_triggers_counted_per_step():
    """A goal-rejected step and an arithmetic-rejected step each count once.

    The goal-rejected step cites ``goal-alignment`` in BOTH its own ``violated_rule_ids``
    and its repair attempt's — it must still count exactly once. Trigger rates equal the
    per-rule triggered count divided by the total step count.
    """
    traces = [_goal_rejected_trace(), _arithmetic_rejected_trace()]

    stats = reasoning_stats.aggregate_trace_stats(traces)

    assert stats.total_reasoning_steps == 2
    # Counted once even though "goal-alignment" appears in the step AND its repair attempt.
    assert stats.goal_validation_triggered == 1
    assert stats.arithmetic_validation_triggered == 1
    # Rates are triggered / total_steps.
    assert stats.goal_trigger_rate == 1 / 2
    assert stats.arithmetic_trigger_rate == 1 / 2

    # The to_dict view carries the new fields and matches the properties.
    document = stats.to_dict()
    assert document["goal_validation_triggered"] == 1
    assert document["goal_trigger_rate"] == 1 / 2
    assert document["arithmetic_validation_triggered"] == 1
    assert document["arithmetic_trigger_rate"] == 1 / 2


def test_goal_trigger_only_counted_once_when_in_step_and_repair():
    """A single step citing goal-alignment in both places contributes a count of 1."""
    stats = reasoning_stats.aggregate_trace_stats([_goal_rejected_trace()])

    assert stats.total_reasoning_steps == 1
    assert stats.goal_validation_triggered == 1
    assert stats.goal_trigger_rate == 1.0


def test_validation_trigger_rates_zero_when_no_steps():
    """With no steps the trigger rates are the float ``0.0`` (no division by zero)."""
    stats = reasoning_stats.aggregate_trace_stats([])

    assert stats.goal_validation_triggered == 0
    assert stats.arithmetic_validation_triggered == 0
    assert stats.goal_trigger_rate == 0.0
    assert stats.arithmetic_trigger_rate == 0.0
    assert isinstance(stats.goal_trigger_rate, float)


# --------------------------------------------------------------------------- #
# generate_reasoning_stats — offline end-to-end (stubbed backend + dataset)
# --------------------------------------------------------------------------- #

#: A wrong equation (2 + 2 = 5) the ArithmeticValidationEngine rejects, then the corrected
#: equation (2 + 2 = 4) it accepts — so the full system records a real repair.
_WRONG_EQ = json.dumps(
    {"logic_form": "2 + 2 = 5", "predicates": {"lhs": 2, "op": "+", "rhs": 2, "result": 5}}
)
_RIGHT_EQ = json.dumps(
    {"logic_form": "2 + 2 = 4", "predicates": {"lhs": 2, "op": "+", "rhs": 2, "result": 4}}
)


def _wrong_then_right_factory(*_args, **_kwargs) -> MockBackend:
    """Stand-in for ``build_ollama_backend``: a fresh wrong-then-correct backend per query.

    The MockBackend returns the wrong equation first (rejected by arithmetic validation),
    then repeats the corrected equation, so the bounded repair sub-loop turns a rejection
    into an acceptance and the step is recorded as REPAIRED.
    """
    return MockBackend([_WRONG_EQ, _RIGHT_EQ])


def _stub_dataset(*_args, **_kwargs):
    """Stand-in for ``_gsm8k_dataset`` returning a tiny in-memory MATH subset."""
    items = [
        DatasetItem(
            item_id="stub-1",
            query="Compute 2 plus 2.",
            ground_truth="4",
            domain=Domain.MATH,
        ),
        DatasetItem(
            item_id="stub-2",
            query="Add two and two.",
            ground_truth="4",
            domain=Domain.MATH,
        ),
    ]
    return items, "stubbed in-memory sample (2 items)"


def test_generate_reasoning_stats_writes_reports_offline(monkeypatch, tmp_path):
    """End-to-end: the full system runs offline, records a repair, and writes both reports."""
    # The full-System config builds its backend via ablation.build_ollama_backend.
    monkeypatch.setattr(ablation, "build_ollama_backend", _wrong_then_right_factory)
    # Stub the dataset loader so no file/network is touched.
    monkeypatch.setattr(reasoning_stats, "_gsm8k_dataset", _stub_dataset)

    paths = reasoning_stats.generate_reasoning_stats(
        model="qwen3:8b", host=None, dataset_path=None, limit=2, output_dir=tmp_path
    )

    assert set(paths) == {"html", "json"}
    for key, path in paths.items():
        assert path.exists(), f"{key} report was not created"
        assert path.stat().st_size > 0, f"{key} report is empty"
    assert paths["html"].name == "reasoning_stats.html"
    assert paths["json"].name == "reasoning_stats.json"

    html = paths["html"].read_text(encoding="utf-8")
    # The required report sections are present.
    assert "Headcount" in html
    assert "Rule utilization" in html
    assert "Repair triggers" in html
    assert "Termination reasons" in html
    # The new report-only validation-trigger rows render in the headcount table.
    assert "Goal validation triggered" in html
    assert "Goal trigger rate" in html
    assert "Arithmetic validation triggered" in html
    assert "Arithmetic trigger rate" in html
    # The honest framing about the starter rule set is included.
    assert "general-purpose starter rule set" in html

    document = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert document["report"] == "reasoning-stats"
    assert document["is_real"] is True
    assert document["real_model"] == "qwen3:8b"
    assert document["questions_evaluated"] == 2

    stats = document["stats"]
    assert stats["total_questions"] == 2
    # The new validation-trigger fields are present in the JSON view.
    assert "goal_validation_triggered" in stats
    assert "goal_trigger_rate" in stats
    assert "arithmetic_validation_triggered" in stats
    assert "arithmetic_trigger_rate" in stats
    # Each query's single step was rejected-then-repaired into acceptance.
    assert stats["repaired_successfully"] >= 1
    # The arithmetic-correctness rule fired the repair sub-loop.
    assert "arithmetic-correctness" in stats["repair_violation_counts"]
    # The well-formed-step starter rule dominates rule utilization.
    assert "well-formed-step" in stats["rule_utilization"]
    # The partition invariant holds in the live run too.
    assert (
        stats["accepted_first_pass"]
        + stats["repaired_successfully"]
        + stats["rejected"]
        == stats["total_reasoning_steps"]
    )
