"""Comparison report: per-metric System-vs-baseline differences (Task 14.2).

This module implements the *comparison report* portion of the design's *Evaluation
Harness* component. Given the structured :class:`~nsr.evaluation_harness.EvaluationRunResult`
produced by Task 14.1 (which already carries per-method
:class:`~nsr.models.MethodMetrics`, including the latency overhead computed as the mean
per-query difference versus the LLM-only baseline -- Req 9.5), it:

- **Surfaces latency overhead** -- the mean per-query latency difference versus the
  LLM-only baseline, already computed per method in Task 14.1, is reported as one of the
  compared metrics (Req 9.5).
- **Computes per-method Reasoning_Consistency** -- across repeated evaluation runs, only
  when the configured ``repeated_run_count`` is 2 or greater; otherwise consistency is
  left unset (``None``). Consistency for a method is the mean over its evaluated items of
  the modal-answer fraction of that item's final answers across the repeated runs, using
  :func:`nsr.metrics_engine.compute_reasoning_consistency` (Req 9.6, 7.4, 7.5).
- **Produces a comparison report** -- for every computed metric
  (``final_answer_accuracy``, ``step_hallucination_rate``, ``faithfulness``,
  ``mean_latency``, ``p95_latency``, ``latency_overhead``, ``reasoning_consistency``) it
  lists the System value, each Baseline_Method value, and the numeric difference
  ``System - baseline`` per baseline (Req 9.4).
- **Persists the report durably** -- the report is written together with the run record
  via :meth:`nsr.reproducibility.ReproducibilityManager.persist`, which fsyncs the bytes
  so they survive process termination and returns an :class:`~nsr.models.ErrorRecord` on
  failure rather than raising (Req 13.4, 13.5).
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

from .evaluation_harness import SYSTEM_METHOD_NAME, EvaluationRunResult
from .metrics_engine import compute_reasoning_consistency
from .models import ErrorRecord, MethodMetrics, RunRecord
from .reproducibility import ReproducibilityManager

#: The metric label used for Reasoning_Consistency in the report and metric ordering.
REASONING_CONSISTENCY_METRIC = "reasoning_consistency"

#: Ordered ``report-metric-name -> MethodMetrics attribute`` for the static metrics that
#: come straight from the per-method metrics computed in Task 14.1. Reasoning_Consistency
#: is appended separately because it is computed here, across repeated runs.
_STATIC_METRICS: tuple[tuple[str, str], ...] = (
    ("final_answer_accuracy", "final_answer_accuracy"),
    ("step_hallucination_rate", "step_hallucination_rate"),
    ("faithfulness", "faithfulness_score"),
    ("mean_latency", "mean_latency_ms"),
    ("p95_latency", "p95_latency_ms"),
    ("latency_overhead", "latency_overhead_ms"),
)

#: The full ordered list of metric names the report compares (Req 9.4).
REPORT_METRICS: tuple[str, ...] = tuple(name for name, _ in _STATIC_METRICS) + (
    REASONING_CONSISTENCY_METRIC,
)


@dataclass
class MetricComparison:
    """One metric's System value, each baseline value, and the System-minus-baseline gap.

    ``differences`` maps each baseline name to ``system_value - baseline_value``. A
    difference is ``None`` when either operand is unset (e.g. Reasoning_Consistency when
    fewer than two repeated runs are configured), so absent data is never silently
    treated as zero.
    """

    metric: str
    system_value: Optional[float]
    baseline_values: dict[str, Optional[float]] = field(default_factory=dict)
    differences: dict[str, Optional[float]] = field(default_factory=dict)


@dataclass
class ComparisonReport:
    """A full comparison of the System against every baseline across all metrics.

    ``reasoning_consistency`` records the per-method Reasoning_Consistency that was
    computed across repeated runs (Req 9.6); every value is ``None`` when fewer than two
    repeated runs are configured. ``metrics`` lists one :class:`MetricComparison` per
    metric in :data:`REPORT_METRICS` order.
    """

    system_method: str
    baseline_methods: list[str]
    repeated_run_count: int
    metrics: list[MetricComparison]
    reasoning_consistency: dict[str, Optional[float]] = field(default_factory=dict)


def compute_method_consistency(
    runs: Sequence[EvaluationRunResult],
    repeated_run_count: int,
) -> dict[str, Optional[float]]:
    """Per-method Reasoning_Consistency across repeated runs (Req 9.6, 7.4, 7.5).

    For each method, the per-item final answers are gathered across all supplied
    ``runs`` (matched by item id), the modal-answer fraction is computed per item via
    :func:`~nsr.metrics_engine.compute_reasoning_consistency`, and the method's
    consistency is the mean of those per-item fractions over the items it produced an
    answer for. The result is ``None`` for every method when ``repeated_run_count`` is
    below 2 (Req 7.5) or when fewer than two runs are supplied (no repeated runs to
    compare). A method with no evaluated items maps to ``None``.
    """
    methods = _all_methods(runs)
    if repeated_run_count < 2 or len(runs) < 2:
        return {method: None for method in methods}

    consistency: dict[str, Optional[float]] = {}
    for method in methods:
        answers_by_item: dict[str, list[str]] = defaultdict(list)
        for run in runs:
            for outcome in run.per_item_outcomes.get(method, []):
                answers_by_item[outcome.item_id].append(outcome.final_answer)

        per_item = [
            value
            for answers in answers_by_item.values()
            if (value := compute_reasoning_consistency(answers, repeated_run_count))
            is not None
        ]
        consistency[method] = sum(per_item) / len(per_item) if per_item else None
    return consistency


def build_comparison_report(
    runs: Union[EvaluationRunResult, Sequence[EvaluationRunResult]],
    *,
    repeated_run_count: int = 1,
    system_method_name: str = SYSTEM_METHOD_NAME,
) -> ComparisonReport:
    """Build the comparison report from one or more evaluation runs (Req 9.4, 9.5, 9.6).

    ``runs`` is either a single :class:`~nsr.evaluation_harness.EvaluationRunResult` or a
    sequence of them (the repeated runs). The first run supplies the per-method metric
    values that are compared; all runs together drive Reasoning_Consistency. The System
    method (``system_method_name``) is compared against every other method present, and
    each metric row carries the System value, each baseline value, and the numeric
    difference ``System - baseline``.
    """
    run_list = _as_run_list(runs)
    primary = run_list[0]
    metrics_by_method = primary.method_metrics

    if system_method_name not in metrics_by_method:
        raise ValueError(
            f"system method {system_method_name!r} not present in evaluation results"
        )

    baseline_methods = sorted(
        name for name in metrics_by_method if name != system_method_name
    )

    consistency = compute_method_consistency(run_list, repeated_run_count)

    comparisons: list[MetricComparison] = []
    for metric_name, attr in _STATIC_METRICS:
        comparisons.append(
            _compare_metric(
                metric_name,
                system_method_name,
                baseline_methods,
                lambda method: float(getattr(metrics_by_method[method], attr)),
            )
        )
    comparisons.append(
        _compare_metric(
            REASONING_CONSISTENCY_METRIC,
            system_method_name,
            baseline_methods,
            lambda method: consistency.get(method),
        )
    )

    return ComparisonReport(
        system_method=system_method_name,
        baseline_methods=baseline_methods,
        repeated_run_count=repeated_run_count,
        metrics=comparisons,
        reasoning_consistency=consistency,
    )


def persist_comparison_report(
    reproducibility: ReproducibilityManager,
    run_record: RunRecord,
    report: ComparisonReport,
    output_path: Union[str, os.PathLike[str]],
) -> Optional[ErrorRecord]:
    """Persist ``report`` together with ``run_record`` durably (Req 13.4, 13.5).

    Delegates to :meth:`~nsr.reproducibility.ReproducibilityManager.persist`, which
    writes the run record and the report under a single document, fsyncs the bytes so
    they survive process termination, and returns an :class:`~nsr.models.ErrorRecord`
    (rather than raising) if persistence fails so the caller can avoid reporting the run
    as successful.
    """
    return reproducibility.persist(run_record, report, output_path)


# --------------------------------------------------------------------------- helpers


def _as_run_list(
    runs: Union[EvaluationRunResult, Sequence[EvaluationRunResult]],
) -> list[EvaluationRunResult]:
    """Coerce a single run or a sequence of runs into a non-empty list."""
    if isinstance(runs, EvaluationRunResult):
        return [runs]
    run_list = list(runs)
    if not run_list:
        raise ValueError("comparison report requires at least one evaluation run")
    return run_list


def _all_methods(runs: Sequence[EvaluationRunResult]) -> list[str]:
    """Return every method name appearing in any run, in first-seen order."""
    seen: dict[str, None] = {}
    for run in runs:
        for method in run.method_metrics:
            seen.setdefault(method, None)
        for method in run.per_item_outcomes:
            seen.setdefault(method, None)
    return list(seen)


def _compare_metric(metric_name, system_method, baseline_methods, value_of):
    """Build a :class:`MetricComparison` for one metric using ``value_of(method)``.

    ``value_of`` returns the metric value for a method (possibly ``None``). The numeric
    difference is computed only when both the System and the baseline have a value.
    """
    system_value = value_of(system_method)
    baseline_values: dict[str, Optional[float]] = {}
    differences: dict[str, Optional[float]] = {}
    for baseline in baseline_methods:
        baseline_value = value_of(baseline)
        baseline_values[baseline] = baseline_value
        if system_value is None or baseline_value is None:
            differences[baseline] = None
        else:
            differences[baseline] = system_value - baseline_value
    return MetricComparison(
        metric=metric_name,
        system_value=system_value,
        baseline_values=baseline_values,
        differences=differences,
    )


__all__ = [
    "REPORT_METRICS",
    "REASONING_CONSISTENCY_METRIC",
    "MetricComparison",
    "ComparisonReport",
    "compute_method_consistency",
    "build_comparison_report",
    "persist_comparison_report",
]
