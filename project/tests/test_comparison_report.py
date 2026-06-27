"""Unit tests for the comparison report (Task 14.2).

These cover the behaviour required by Req 9.4 (a comparison report listing, per metric,
the System value, each Baseline_Method value, and the numeric System-minus-baseline
difference), Req 9.5 (latency overhead, already computed per method in Task 14.1, is
surfaced as a compared metric), Req 9.6 / 7.4 / 7.5 (per-method Reasoning_Consistency
across repeated runs only when the repeated-run count is 2 or greater), and Req 13.4 /
13.5 (the report is persisted durably with the run record, with a persistence-failure
error path).

The evaluation results are built in-memory from :class:`ItemOutcome` /
:class:`MethodMetrics` so no model or network is involved.
"""

from __future__ import annotations

import json

import pytest

from nsr.comparison_report import (
    REASONING_CONSISTENCY_METRIC,
    REPORT_METRICS,
    ComparisonReport,
    MetricComparison,
    build_comparison_report,
    compute_method_consistency,
    persist_comparison_report,
)
from nsr.evaluation_harness import (
    LLM_ONLY_METHOD_NAME,
    SYSTEM_METHOD_NAME,
    EvaluationRunResult,
    ItemOutcome,
)
from nsr.models import MethodMetrics, RunRecord, SystemConfig
from nsr.reproducibility import ReproducibilityManager


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_config(repeated_run_count: int = 1) -> SystemConfig:
    return SystemConfig(
        max_cycle_limit=10,
        repair_attempt_limit=2,
        retry_count=1,
        llm_selection="mock",
        output_format="text",
        conflict_resolution_policy="priority",
        generation_timeout_ms=1000,
        repeated_run_count=repeated_run_count,
        random_seed=7,
    )


def make_run_record() -> RunRecord:
    return RunRecord(
        config=make_config(),
        dataset_ids=["a", "b"],
        model_id="mock-model",
        seed=7,
    )


def metrics(
    method: str,
    *,
    accuracy: float = 0.0,
    hallucination: float = 0.0,
    faithfulness: float = 0.0,
    overhead: float = 0.0,
    mean_latency: float = 0.0,
    p95_latency: float = 0.0,
) -> MethodMetrics:
    return MethodMetrics(
        method=method,
        final_answer_accuracy=accuracy,
        step_hallucination_rate=hallucination,
        faithfulness_score=faithfulness,
        latency_overhead_ms=overhead,
        mean_latency_ms=mean_latency,
        p95_latency_ms=p95_latency,
    )


def outcome(item_id: str, answer: str) -> ItemOutcome:
    return ItemOutcome(
        item_id=item_id,
        final_answer=answer,
        correct=True,
        latency_ms=1.0,
    )


def make_result(
    method_metrics: dict[str, MethodMetrics],
    per_item: dict[str, list[ItemOutcome]] | None = None,
) -> EvaluationRunResult:
    return EvaluationRunResult(
        run_record=make_run_record(),
        method_metrics=method_metrics,
        per_item_outcomes=per_item or {name: [] for name in method_metrics},
    )


# ---------------------------------------------------------------------------
# Report structure and per-metric differences (Req 9.4, 9.5)
# ---------------------------------------------------------------------------


def test_report_lists_all_metrics_in_order():
    result = make_result(
        {
            SYSTEM_METHOD_NAME: metrics(SYSTEM_METHOD_NAME),
            LLM_ONLY_METHOD_NAME: metrics(LLM_ONLY_METHOD_NAME),
        }
    )
    report = build_comparison_report(result)
    assert [m.metric for m in report.metrics] == list(REPORT_METRICS)
    # All seven metrics including latency_overhead and reasoning_consistency.
    assert "latency_overhead" in REPORT_METRICS
    assert REASONING_CONSISTENCY_METRIC in REPORT_METRICS


def test_per_metric_system_baseline_and_difference():
    result = make_result(
        {
            SYSTEM_METHOD_NAME: metrics(
                SYSTEM_METHOD_NAME,
                accuracy=0.9,
                hallucination=0.1,
                faithfulness=0.8,
                overhead=15.0,
                mean_latency=20.0,
                p95_latency=29.0,
            ),
            "chain-of-thought": metrics(
                "chain-of-thought",
                accuracy=0.6,
                hallucination=0.0,
                faithfulness=0.0,
                overhead=5.0,
                mean_latency=10.0,
                p95_latency=12.0,
            ),
        }
    )
    report = build_comparison_report(result)
    by_name = {m.metric: m for m in report.metrics}

    acc = by_name["final_answer_accuracy"]
    assert acc.system_value == pytest.approx(0.9)
    assert acc.baseline_values["chain-of-thought"] == pytest.approx(0.6)
    assert acc.differences["chain-of-thought"] == pytest.approx(0.3)

    # Latency overhead computed in Task 14.1 is surfaced and differenced (Req 9.5).
    overhead = by_name["latency_overhead"]
    assert overhead.system_value == pytest.approx(15.0)
    assert overhead.differences["chain-of-thought"] == pytest.approx(10.0)

    mean = by_name["mean_latency"]
    assert mean.differences["chain-of-thought"] == pytest.approx(10.0)


def test_report_identifies_system_and_baselines():
    result = make_result(
        {
            SYSTEM_METHOD_NAME: metrics(SYSTEM_METHOD_NAME),
            "react": metrics("react"),
            LLM_ONLY_METHOD_NAME: metrics(LLM_ONLY_METHOD_NAME),
        }
    )
    report = build_comparison_report(result)
    assert report.system_method == SYSTEM_METHOD_NAME
    assert report.baseline_methods == sorted([LLM_ONLY_METHOD_NAME, "react"])


def test_missing_system_method_raises():
    result = make_result({"react": metrics("react")})
    with pytest.raises(ValueError):
        build_comparison_report(result)


# ---------------------------------------------------------------------------
# Reasoning_Consistency across repeated runs (Req 9.6, 7.4, 7.5)
# ---------------------------------------------------------------------------


def test_consistency_unset_when_repeated_run_count_below_two():
    result = make_result(
        {
            SYSTEM_METHOD_NAME: metrics(SYSTEM_METHOD_NAME),
            LLM_ONLY_METHOD_NAME: metrics(LLM_ONLY_METHOD_NAME),
        }
    )
    report = build_comparison_report(result, repeated_run_count=1)
    assert report.reasoning_consistency[SYSTEM_METHOD_NAME] is None
    rc = next(m for m in report.metrics if m.metric == REASONING_CONSISTENCY_METRIC)
    assert rc.system_value is None
    # Difference is None when an operand is unset (not silently zero).
    assert rc.differences[LLM_ONLY_METHOD_NAME] is None


def test_consistency_unset_with_single_run_even_if_configured():
    result = make_result(
        {SYSTEM_METHOD_NAME: metrics(SYSTEM_METHOD_NAME)},
        {SYSTEM_METHOD_NAME: [outcome("a", "4"), outcome("b", "6")]},
    )
    report = build_comparison_report(result, repeated_run_count=3)
    assert report.reasoning_consistency[SYSTEM_METHOD_NAME] is None


def test_consistency_modal_fraction_across_runs():
    # Three repeated runs. System answers item "a" as 4,4,4 (consistency 1.0) and item
    # "b" as 6,6,5 (consistency 2/3). Method consistency is the mean: (1.0 + 2/3) / 2.
    def run(a_ans: str, b_ans: str) -> EvaluationRunResult:
        return make_result(
            {SYSTEM_METHOD_NAME: metrics(SYSTEM_METHOD_NAME)},
            {SYSTEM_METHOD_NAME: [outcome("a", a_ans), outcome("b", b_ans)]},
        )

    runs = [run("4", "6"), run("4", "6"), run("4", "5")]
    consistency = compute_method_consistency(runs, repeated_run_count=3)
    assert consistency[SYSTEM_METHOD_NAME] == pytest.approx((1.0 + 2 / 3) / 2)

    report = build_comparison_report(runs, repeated_run_count=3)
    rc = next(m for m in report.metrics if m.metric == REASONING_CONSISTENCY_METRIC)
    assert rc.system_value == pytest.approx((1.0 + 2 / 3) / 2)


def test_consistency_difference_between_system_and_baseline():
    def run(sys_b: str, base_b: str) -> EvaluationRunResult:
        return make_result(
            {
                SYSTEM_METHOD_NAME: metrics(SYSTEM_METHOD_NAME),
                LLM_ONLY_METHOD_NAME: metrics(LLM_ONLY_METHOD_NAME),
            },
            {
                SYSTEM_METHOD_NAME: [outcome("a", "4"), outcome("b", sys_b)],
                LLM_ONLY_METHOD_NAME: [outcome("a", "4"), outcome("b", base_b)],
            },
        )

    # System b: 6,6 -> 1.0 ; baseline b: 6,5 -> 0.5. With item a always 4,4 -> 1.0.
    runs = [run("6", "6"), run("6", "5")]
    report = build_comparison_report(runs, repeated_run_count=2)
    rc = next(m for m in report.metrics if m.metric == REASONING_CONSISTENCY_METRIC)
    # System mean (1.0 + 1.0)/2 = 1.0 ; baseline mean (1.0 + 0.5)/2 = 0.75.
    assert rc.system_value == pytest.approx(1.0)
    assert rc.baseline_values[LLM_ONLY_METHOD_NAME] == pytest.approx(0.75)
    assert rc.differences[LLM_ONLY_METHOD_NAME] == pytest.approx(0.25)


def test_consistency_none_for_method_without_items():
    runs = [
        make_result({SYSTEM_METHOD_NAME: metrics(SYSTEM_METHOD_NAME)}, {SYSTEM_METHOD_NAME: []}),
        make_result({SYSTEM_METHOD_NAME: metrics(SYSTEM_METHOD_NAME)}, {SYSTEM_METHOD_NAME: []}),
    ]
    consistency = compute_method_consistency(runs, repeated_run_count=2)
    assert consistency[SYSTEM_METHOD_NAME] is None


# ---------------------------------------------------------------------------
# Durable persistence with the run record (Req 13.4, 13.5)
# ---------------------------------------------------------------------------


def test_persist_writes_report_with_run_record(tmp_path):
    result = make_result(
        {
            SYSTEM_METHOD_NAME: metrics(SYSTEM_METHOD_NAME, accuracy=0.9, overhead=15.0),
            LLM_ONLY_METHOD_NAME: metrics(LLM_ONLY_METHOD_NAME, accuracy=0.6),
        }
    )
    report = build_comparison_report(result)
    out = tmp_path / "nested" / "report.json"

    error = persist_comparison_report(
        ReproducibilityManager(), result.run_record, report, out
    )
    assert error is None
    assert out.exists()

    document = json.loads(out.read_text(encoding="utf-8"))
    # Run record and report persisted together and remain associated (Req 13.4).
    assert "run_record" in document
    assert "metrics" in document
    assert document["run_record"]["model_id"] == "mock-model"
    persisted_metrics = {m["metric"] for m in document["metrics"]["metrics"]}
    assert persisted_metrics == set(REPORT_METRICS)


def test_persist_failure_returns_error_record(tmp_path):
    result = make_result(
        {
            SYSTEM_METHOD_NAME: metrics(SYSTEM_METHOD_NAME),
            LLM_ONLY_METHOD_NAME: metrics(LLM_ONLY_METHOD_NAME),
        }
    )
    report = build_comparison_report(result)

    # A directory at the target path makes the file write fail (Req 13.5).
    bad_path = tmp_path / "as_dir"
    bad_path.mkdir()

    error = persist_comparison_report(
        ReproducibilityManager(), result.run_record, report, bad_path
    )
    assert error is not None
    assert error.failed_component == "ReproducibilityManager"
