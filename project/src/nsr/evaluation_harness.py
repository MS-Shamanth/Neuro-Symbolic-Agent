"""Evaluation Harness: execution and per-method metrics (Task 14.1).

This module implements the *execution* and *per-method metric* portion of the design's
*Evaluation Harness* component (Req 9.1, 9.2, 9.7, 11.3, 13.1). It runs the System under
test (the :class:`~nsr.orchestrator.PipelineOrchestrator`) and each configured baseline
method (:mod:`nsr.baselines`) over every item in a dataset, then aggregates per-method
metrics.

Scope of Task 14.1:

- **Execution** -- run the System and each configured baseline over every
  :class:`~nsr.models.DatasetItem`, and record the run record *before the first item* is
  evaluated via the :class:`~nsr.reproducibility.ReproducibilityManager` (Req 9.1, 13.1).
- **Per-method metrics** -- for each method, over the items it successfully evaluated,
  compute final-answer accuracy (vs ``ground_truth``), Step_Level_Hallucination_Rate and
  Faithfulness_Score (from the System's Proof_Trace; baselines produce no trace and so
  contribute ``0.0`` for these), the mean and 95th-percentile Pipeline latency (Req 11.3),
  and the latency overhead versus the LLM-only baseline computed as the mean per-query
  latency difference over the shared successfully-evaluated query set (Req 9.5).
- **Failure isolation** -- if a method fails to produce a result for an item (it raises,
  or the System returns an :class:`~nsr.models.ErrorRecord`), exclude that item's result
  for that method, log the failed method and item identifier, and continue (Req 9.7).

The comparison report, Reasoning_Consistency across repeated runs, and durable
persistence of the report are Task 14.2; the latency-overhead property test is Task 14.3;
and the end-to-end integration test is Task 14.4. The harness here returns a structured
:class:`EvaluationRunResult` carrying the run record, per-method
:class:`~nsr.models.MethodMetrics`, the per-item outcomes (so 14.2 can build the report
and consistency), and the run-log exclusions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Protocol, Sequence, Union, runtime_checkable

from .baselines import ReasoningMethod
from .metrics_engine import compute_step_hallucination_rate
from .models import (
    DatasetItem,
    ErrorRecord,
    MethodMetrics,
    RunRecord,
    SystemConfig,
    VerifiedOutput,
)
from .reproducibility import ReproducibilityManager

logger = logging.getLogger(__name__)

#: The method label used for the System under test in metrics and the run log.
SYSTEM_METHOD_NAME = "neuro-symbolic"

#: The reference baseline against which latency overhead is measured (Req 9.5).
LLM_ONLY_METHOD_NAME = "llm-only"

#: ``run`` of the System under test returns a VerifiedOutput on success or an ErrorRecord.
SystemResult = Union[VerifiedOutput, ErrorRecord]


@runtime_checkable
class SystemUnderTest(Protocol):
    """The System under test: maps a query to a VerifiedOutput or an ErrorRecord.

    :class:`~nsr.orchestrator.PipelineOrchestrator` conforms to this protocol via its
    :meth:`~nsr.orchestrator.PipelineOrchestrator.run` method.
    """

    def run(self, query: object) -> SystemResult:  # pragma: no cover - structural
        """Process ``query`` and return a VerifiedOutput or an ErrorRecord."""
        ...


def normalized_answer_match(predicted: str, ground_truth: str) -> bool:
    """Default final-answer matcher: case-insensitive, whitespace-normalized equality.

    Both answers are lowercased and have their internal whitespace collapsed to a single
    space before comparison, so trivial formatting differences do not count as errors.
    """
    return _normalize(predicted) == _normalize(ground_truth)


def _normalize(text: str) -> str:
    """Lowercase ``text`` and collapse all runs of whitespace to single spaces."""
    if text is None:
        return ""
    return " ".join(str(text).split()).lower()


def percentile(values: Sequence[float], pct: float) -> float:
    """Return the ``pct`` percentile of ``values`` by linear interpolation.

    ``pct`` is a percentage in ``[0, 100]``. An empty input yields ``0.0``; a single
    value yields that value. Interpolation matches the common "linear" method: the rank
    is ``(pct / 100) * (n - 1)`` and the result interpolates between the two nearest
    order statistics.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (n - 1)
    low = int(rank)
    high = min(low + 1, n - 1)
    frac = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


@dataclass
class ItemOutcome:
    """A single method's successful result for one dataset item.

    Baselines produce no Proof_Trace, so :attr:`faithfulness_score` and
    :attr:`step_hallucination_rate` are ``0.0`` and :attr:`has_trace` is ``False`` for
    them; only the System carries trace-derived values.
    """

    item_id: str
    final_answer: str
    correct: bool
    latency_ms: float
    faithfulness_score: float = 0.0
    step_hallucination_rate: float = 0.0
    has_trace: bool = False


@dataclass
class ExclusionEntry:
    """A run-log entry recording that a method failed to evaluate an item (Req 9.7)."""

    method: str
    item_id: str
    reason: str


@dataclass
class EvaluationRunResult:
    """The result of an evaluation run: run record, per-method metrics, and run log.

    ``per_item_outcomes`` maps each method name to the list of its successful
    :class:`ItemOutcome` entries (in dataset order), so Task 14.2 can build the
    comparison report and Reasoning_Consistency without re-running anything.
    """

    run_record: RunRecord
    method_metrics: dict[str, MethodMetrics]
    per_item_outcomes: dict[str, list[ItemOutcome]]
    exclusions: list[ExclusionEntry] = field(default_factory=list)


class EvaluationHarness:
    """Runs the System and each baseline over a dataset and aggregates metrics.

    The System under test and the baselines are supplied pre-constructed so the harness
    is agnostic to how they are wired (and fully testable with a
    :class:`~nsr.llm_component.MockBackend`). Latency is measured with an injectable
    ``clock`` (defaulting to :func:`time.perf_counter`); when the System's Proof_Trace
    already carries a pipeline-latency record (Req 11.1) that value is preferred over the
    harness's wall-clock measurement.
    """

    def __init__(
        self,
        system: SystemUnderTest,
        baselines: Union[Mapping[str, ReasoningMethod], Sequence[ReasoningMethod]],
        *,
        reproducibility: Optional[ReproducibilityManager] = None,
        answer_match: Callable[[str, str], bool] = normalized_answer_match,
        clock: Callable[[], float] = time.perf_counter,
        system_method_name: str = SYSTEM_METHOD_NAME,
    ) -> None:
        self._system = system
        self._baselines = self._normalize_baselines(baselines)
        self._reproducibility = reproducibility or ReproducibilityManager()
        self._answer_match = answer_match
        self._clock = clock
        self._system_method_name = system_method_name

    @staticmethod
    def _normalize_baselines(
        baselines: Union[Mapping[str, ReasoningMethod], Sequence[ReasoningMethod]],
    ) -> dict[str, ReasoningMethod]:
        """Coerce the supplied baselines into a ``name -> method`` mapping."""
        if isinstance(baselines, Mapping):
            return dict(baselines)
        return {method.name: method for method in baselines}

    def run(
        self,
        dataset: Sequence[DatasetItem],
        *,
        config: SystemConfig,
        model_id: str,
    ) -> EvaluationRunResult:
        """Execute the System and every baseline over ``dataset`` and aggregate metrics.

        The run record is built *before the first dataset item is evaluated* (Req 13.1):
        it carries the full configuration, the dataset identifiers, the model identifier,
        and the effective random seed. Each method is then run over every item; an item a
        method fails to evaluate is excluded for that method, logged, and skipped, while
        evaluation of the remaining items continues (Req 9.7). Finally per-method metrics
        are computed over each method's successfully-evaluated items (Req 9.2, 11.3).
        """
        if not dataset:
            raise ValueError("evaluation requires a non-empty dataset")

        dataset_ids = [item.item_id for item in dataset]

        # --- Req 13.1: record the run record BEFORE the first item is evaluated -----
        seed = self._reproducibility.seed_everything(config)
        run_record = self._reproducibility.build_run_record(
            config=config,
            dataset_ids=dataset_ids,
            model_id=model_id,
            seed=seed,
        )

        exclusions: list[ExclusionEntry] = []
        per_item_outcomes: dict[str, list[ItemOutcome]] = {}

        # --- Req 9.1: run the System and each configured baseline over every item ---
        per_item_outcomes[self._system_method_name] = self._evaluate_method(
            self._system_method_name, self._evaluate_system, dataset, exclusions
        )
        for name, method in self._baselines.items():
            per_item_outcomes[name] = self._evaluate_method(
                name, self._make_baseline_evaluator(method), dataset, exclusions
            )

        # --- Req 9.2, 11.3, 9.5: compute per-method metrics -------------------------
        method_metrics = self._aggregate_metrics(per_item_outcomes)

        return EvaluationRunResult(
            run_record=run_record,
            method_metrics=method_metrics,
            per_item_outcomes=per_item_outcomes,
            exclusions=exclusions,
        )

    # ----------------------------------------------------------------- execution

    def _evaluate_method(
        self,
        method_name: str,
        evaluator: Callable[[DatasetItem], "tuple[Optional[ItemOutcome], Optional[str]]"],
        dataset: Sequence[DatasetItem],
        exclusions: list[ExclusionEntry],
    ) -> list[ItemOutcome]:
        """Run ``evaluator`` over every item, excluding+logging failures (Req 9.7)."""
        outcomes: list[ItemOutcome] = []
        for item in dataset:
            outcome, reason = evaluator(item)
            if outcome is None:
                entry = ExclusionEntry(
                    method=method_name,
                    item_id=item.item_id,
                    reason=reason or "no result produced",
                )
                exclusions.append(entry)
                logger.warning(
                    "evaluation excluded item %r for method %r: %s",
                    item.item_id,
                    method_name,
                    entry.reason,
                )
                continue
            outcomes.append(outcome)
        return outcomes

    def _evaluate_system(
        self, item: DatasetItem
    ) -> "tuple[Optional[ItemOutcome], Optional[str]]":
        """Run the System over one item, returning an outcome or a failure reason."""
        try:
            start = self._clock()
            result = self._system.run(item.query)
            elapsed_ms = (self._clock() - start) * 1000.0
        except Exception as exc:  # noqa: BLE001 - failures are isolated per Req 9.7
            return None, f"system raised {exc!r}"

        if isinstance(result, ErrorRecord):
            return None, (
                f"system returned error from {result.failed_component}: {result.reason}"
            )

        # VerifiedOutput: prefer the trace's recorded pipeline latency (Req 11.1).
        trace = result.proof_trace
        if trace.latency is not None:
            latency_ms = float(trace.latency.pipeline_ms)
        else:
            latency_ms = elapsed_ms

        return (
            ItemOutcome(
                item_id=item.item_id,
                final_answer=result.final_answer,
                correct=self._answer_match(result.final_answer, item.ground_truth),
                latency_ms=latency_ms,
                faithfulness_score=result.faithfulness_score,
                step_hallucination_rate=compute_step_hallucination_rate(trace),
                has_trace=True,
            ),
            None,
        )

    def _make_baseline_evaluator(
        self, method: ReasoningMethod
    ) -> Callable[[DatasetItem], "tuple[Optional[ItemOutcome], Optional[str]]"]:
        """Build a per-item evaluator for a baseline method."""

        def evaluate(item: DatasetItem) -> "tuple[Optional[ItemOutcome], Optional[str]]":
            try:
                result = method.run(item.query)
            except Exception as exc:  # noqa: BLE001 - isolate per-item failures (Req 9.7)
                return None, f"baseline raised {exc!r}"
            # Baselines produce no Proof_Trace, so trace-derived metrics are 0.0.
            return (
                ItemOutcome(
                    item_id=item.item_id,
                    final_answer=result.final_answer,
                    correct=self._answer_match(result.final_answer, item.ground_truth),
                    latency_ms=float(result.latency_ms),
                    faithfulness_score=0.0,
                    step_hallucination_rate=0.0,
                    has_trace=False,
                ),
                None,
            )

        return evaluate

    # ----------------------------------------------------------------- metrics

    def _aggregate_metrics(
        self, per_item_outcomes: Mapping[str, list[ItemOutcome]]
    ) -> dict[str, MethodMetrics]:
        """Compute :class:`MethodMetrics` for every method (Req 9.2, 11.3, 9.5)."""
        # Map of item_id -> llm-only latency, used for latency overhead (Req 9.5).
        llm_latency_by_item = {
            outcome.item_id: outcome.latency_ms
            for outcome in per_item_outcomes.get(LLM_ONLY_METHOD_NAME, [])
        }

        metrics: dict[str, MethodMetrics] = {}
        for name, outcomes in per_item_outcomes.items():
            metrics[name] = self._method_metrics(name, outcomes, llm_latency_by_item)
        return metrics

    def _method_metrics(
        self,
        method: str,
        outcomes: list[ItemOutcome],
        llm_latency_by_item: Mapping[str, float],
    ) -> MethodMetrics:
        """Aggregate one method's per-item outcomes into :class:`MethodMetrics`."""
        n = len(outcomes)
        if n == 0:
            # No successfully-evaluated items: report zeroed metrics for the method.
            return MethodMetrics(
                method=method,
                final_answer_accuracy=0.0,
                step_hallucination_rate=0.0,
                faithfulness_score=0.0,
                latency_overhead_ms=0.0,
                mean_latency_ms=0.0,
                p95_latency_ms=0.0,
            )

        latencies = [o.latency_ms for o in outcomes]
        accuracy = sum(1 for o in outcomes if o.correct) / n
        hallucination = sum(o.step_hallucination_rate for o in outcomes) / n
        faithfulness = sum(o.faithfulness_score for o in outcomes) / n
        mean_latency = sum(latencies) / n
        p95_latency = percentile(latencies, 95.0)

        return MethodMetrics(
            method=method,
            final_answer_accuracy=accuracy,
            step_hallucination_rate=hallucination,
            faithfulness_score=faithfulness,
            latency_overhead_ms=self._latency_overhead(
                method, outcomes, llm_latency_by_item
            ),
            mean_latency_ms=mean_latency,
            p95_latency_ms=p95_latency,
        )

    @staticmethod
    def _latency_overhead(
        method: str,
        outcomes: list[ItemOutcome],
        llm_latency_by_item: Mapping[str, float],
    ) -> float:
        """Mean per-query latency difference vs the LLM-only baseline (Req 9.5).

        The difference is averaged over the items both this method and the LLM-only
        baseline successfully evaluated (the shared query set). The LLM-only baseline's
        own overhead is ``0.0`` by definition, and the overhead is ``0.0`` when there is
        no shared query set (e.g. the LLM-only baseline was not configured).
        """
        if method == LLM_ONLY_METHOD_NAME:
            return 0.0
        diffs = [
            o.latency_ms - llm_latency_by_item[o.item_id]
            for o in outcomes
            if o.item_id in llm_latency_by_item
        ]
        if not diffs:
            return 0.0
        return sum(diffs) / len(diffs)


__all__ = [
    "SYSTEM_METHOD_NAME",
    "LLM_ONLY_METHOD_NAME",
    "SystemUnderTest",
    "SystemResult",
    "normalized_answer_match",
    "percentile",
    "ItemOutcome",
    "ExclusionEntry",
    "EvaluationRunResult",
    "EvaluationHarness",
]
