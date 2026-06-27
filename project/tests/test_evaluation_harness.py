"""Unit tests for the Evaluation Harness execution + per-method metrics (Task 14.1).

These cover the behaviour required by Req 9.1 (run System + each baseline over every
item), 9.2 (per-method accuracy, hallucination, faithfulness), 11.3 (mean and p95
pipeline latency), 9.5 (latency overhead vs LLM-only), 9.7 (exclude+log failing items
and continue), and 13.1 (run record built before the first item).

The System under test and the baselines are stubbed in-memory (no network, no model):
the baselines reuse the real :mod:`nsr.baselines` strategies driven by a
:class:`~nsr.llm_component.MockBackend`, and the System is a small scriptable stub
returning a :class:`~nsr.models.VerifiedOutput` or :class:`~nsr.models.ErrorRecord`.
"""

from __future__ import annotations

import pytest

from nsr.baselines import LLMOnly, build_baseline
from nsr.evaluation_harness import (
    LLM_ONLY_METHOD_NAME,
    SYSTEM_METHOD_NAME,
    EvaluationHarness,
    ItemOutcome,
    normalized_answer_match,
    percentile,
)
from nsr.llm_component import MockBackend
from nsr.models import (
    DatasetItem,
    Domain,
    ErrorRecord,
    ProofStep,
    ProofTrace,
    SystemConfig,
    ValidationStatus,
    VerifiedOutput,
)


# ---------------------------------------------------------------------------
# Fixtures / stubs
# ---------------------------------------------------------------------------


def make_config(*, random_seed: int | None = 7) -> SystemConfig:
    return SystemConfig(
        max_cycle_limit=10,
        repair_attempt_limit=2,
        retry_count=1,
        llm_selection="mock",
        output_format="text",
        conflict_resolution_policy="priority",
        generation_timeout_ms=1000,
        repeated_run_count=1,
        random_seed=random_seed,
    )


def make_dataset() -> list[DatasetItem]:
    return [
        DatasetItem(item_id="m1", query="2+2?", ground_truth="4", domain=Domain.MATH),
        DatasetItem(item_id="m2", query="3+3?", ground_truth="6", domain=Domain.MATH),
        DatasetItem(
            item_id="c1", query="sky color?", ground_truth="blue", domain=Domain.COMMONSENSE
        ),
    ]


def make_trace(accepted: int, rejected: int) -> ProofTrace:
    """Build a Proof_Trace with the given counts of accepted/rejected steps."""
    steps: list[ProofStep] = []
    seq = 0
    for _ in range(accepted):
        steps.append(
            ProofStep(
                sequence=seq,
                step_text=f"step-{seq}",
                representation=None,
                status=ValidationStatus.ACCEPTED,
            )
        )
        seq += 1
    for _ in range(rejected):
        steps.append(
            ProofStep(
                sequence=seq,
                step_text=f"step-{seq}",
                representation=None,
                status=ValidationStatus.REJECTED,
            )
        )
        seq += 1
    return ProofTrace(steps=steps)


class StubSystem:
    """A scriptable System under test mapping query -> VerifiedOutput | ErrorRecord."""

    def __init__(self, responses: dict[str, object]) -> None:
        self._responses = responses
        self.queries: list[object] = []

    def run(self, query: object):
        self.queries.append(query)
        result = self._responses[query]
        if isinstance(result, Exception):
            raise result
        return result


def verified(answer: str, accepted: int, rejected: int) -> VerifiedOutput:
    trace = make_trace(accepted, rejected)
    faithfulness = accepted / (accepted + rejected) if (accepted + rejected) else 0.0
    return VerifiedOutput(
        final_answer=answer, proof_trace=trace, faithfulness_score=faithfulness
    )


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------


def test_normalized_answer_match():
    assert normalized_answer_match("  4 ", "4")
    assert normalized_answer_match("Blue", "blue")
    assert normalized_answer_match("the  cat", "the cat")
    assert not normalized_answer_match("5", "4")


def test_percentile_empty_and_single():
    assert percentile([], 95.0) == 0.0
    assert percentile([42.0], 95.0) == 42.0


def test_percentile_interpolation():
    # Linear interpolation, rank = 0.95*(n-1) for n=5 -> 3.8 -> between 40 and 50.
    assert percentile([10, 20, 30, 40, 50], 95.0) == pytest.approx(48.0)
    assert percentile([10, 20, 30, 40, 50], 50.0) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Run record built before the first item (Req 13.1)
# ---------------------------------------------------------------------------


def test_run_record_built_with_required_fields():
    dataset = make_dataset()
    system = StubSystem({item.query: verified(item.ground_truth, 2, 0) for item in dataset})
    harness = EvaluationHarness(system, baselines=[])

    result = harness.run(dataset, config=make_config(), model_id="mock-model-1")

    rr = result.run_record
    assert rr.model_id == "mock-model-1"
    assert rr.dataset_ids == ["m1", "m2", "c1"]
    assert rr.seed == 7
    assert rr.config is not None


def test_empty_dataset_rejected():
    system = StubSystem({})
    harness = EvaluationHarness(system, baselines=[])
    with pytest.raises(ValueError):
        harness.run([], config=make_config(), model_id="m")


# ---------------------------------------------------------------------------
# Execution over System + baselines (Req 9.1) and per-method metrics (Req 9.2)
# ---------------------------------------------------------------------------


def test_system_and_baseline_evaluated_over_every_item():
    dataset = make_dataset()
    system = StubSystem(
        {
            "2+2?": verified("4", 2, 0),  # correct
            "3+3?": verified("7", 1, 1),  # wrong
            "sky color?": verified("blue", 3, 0),  # correct
        }
    )
    # A baseline that always answers "blue" via the scripted backend.
    backend = MockBackend(["Answer: blue"])
    llm_only = LLMOnly(backend)
    harness = EvaluationHarness(system, baselines=[llm_only])

    result = harness.run(dataset, config=make_config(), model_id="m")

    # Both methods present.
    assert set(result.method_metrics) == {SYSTEM_METHOD_NAME, LLM_ONLY_METHOD_NAME}
    # System saw all three queries.
    assert system.queries == ["2+2?", "3+3?", "sky color?"]

    sys_metrics = result.method_metrics[SYSTEM_METHOD_NAME]
    # 2 of 3 correct.
    assert sys_metrics.final_answer_accuracy == pytest.approx(2 / 3)
    # Faithfulness mean over (1.0, 0.5, 1.0).
    assert sys_metrics.faithfulness_score == pytest.approx((1.0 + 0.5 + 1.0) / 3)
    # Hallucination mean over (0.0, 0.5, 0.0).
    assert sys_metrics.step_hallucination_rate == pytest.approx(0.5 / 3)

    llm_metrics = result.method_metrics[LLM_ONLY_METHOD_NAME]
    # llm-only answers "blue": correct only for the commonsense item.
    assert llm_metrics.final_answer_accuracy == pytest.approx(1 / 3)
    # Baselines have no trace -> trace metrics are 0.0.
    assert llm_metrics.faithfulness_score == 0.0
    assert llm_metrics.step_hallucination_rate == 0.0


# ---------------------------------------------------------------------------
# Failure isolation (Req 9.7)
# ---------------------------------------------------------------------------


def test_failing_item_excluded_and_logged_system_error_record():
    dataset = make_dataset()
    system = StubSystem(
        {
            "2+2?": verified("4", 1, 0),
            "3+3?": ErrorRecord(failed_component="Pipeline", reason="boom"),
            "sky color?": verified("blue", 1, 0),
        }
    )
    harness = EvaluationHarness(system, baselines=[])

    result = harness.run(dataset, config=make_config(), model_id="m")

    # The errored item is excluded for the System, the other two retained.
    sys_outcomes = result.per_item_outcomes[SYSTEM_METHOD_NAME]
    assert [o.item_id for o in sys_outcomes] == ["m1", "c1"]

    excl = [e for e in result.exclusions if e.method == SYSTEM_METHOD_NAME]
    assert len(excl) == 1
    assert excl[0].item_id == "m2"
    assert "boom" in excl[0].reason

    # Accuracy computed only over successfully evaluated items (2/2 here).
    assert result.method_metrics[SYSTEM_METHOD_NAME].final_answer_accuracy == 1.0


def test_failing_item_excluded_when_method_raises():
    dataset = make_dataset()
    system = StubSystem(
        {
            "2+2?": verified("4", 1, 0),
            "3+3?": RuntimeError("kaboom"),
            "sky color?": verified("blue", 1, 0),
        }
    )
    harness = EvaluationHarness(system, baselines=[])

    result = harness.run(dataset, config=make_config(), model_id="m")

    excl = [e for e in result.exclusions if e.method == SYSTEM_METHOD_NAME]
    assert len(excl) == 1
    assert excl[0].item_id == "m2"
    assert "kaboom" in excl[0].reason
    # Evaluation continued past the failure.
    assert len(result.per_item_outcomes[SYSTEM_METHOD_NAME]) == 2


# ---------------------------------------------------------------------------
# Latency metrics (Req 11.3) and latency overhead (Req 9.5)
# ---------------------------------------------------------------------------


class FixedLatencyMethod:
    """A baseline-like method returning a fixed latency and answer per query."""

    def __init__(self, name: str, latency_ms: float, answer: str = "x") -> None:
        self.name = name
        self._latency_ms = latency_ms
        self._answer = answer

    def run(self, query: str):
        from nsr.baselines import BaselineResult

        return BaselineResult(
            method=self.name, final_answer=self._answer, latency_ms=self._latency_ms
        )


def test_mean_and_p95_latency_and_overhead():
    dataset = make_dataset()
    # System uses a deterministic clock: +10ms per query (start/stop pair = +1 each).
    times = iter([0.0, 0.010, 0.010, 0.030, 0.030, 0.060])  # seconds: deltas 10,20,30 ms

    def clock() -> float:
        return next(times)

    system = StubSystem({item.query: verified(item.ground_truth, 1, 0) for item in dataset})

    llm_only = FixedLatencyMethod(LLM_ONLY_METHOD_NAME, latency_ms=5.0)
    slow = FixedLatencyMethod("slow", latency_ms=25.0)

    harness = EvaluationHarness(system, baselines=[llm_only, slow], clock=clock)
    result = harness.run(dataset, config=make_config(), model_id="m")

    sys_metrics = result.method_metrics[SYSTEM_METHOD_NAME]
    # Latencies 10, 20, 30 ms.
    assert sys_metrics.mean_latency_ms == pytest.approx(20.0)
    assert sys_metrics.p95_latency_ms == pytest.approx(29.0)
    # Overhead vs llm-only (5ms): mean of (10-5, 20-5, 30-5) = 15.
    assert sys_metrics.latency_overhead_ms == pytest.approx(15.0)

    # llm-only overhead is zero by definition.
    assert result.method_metrics[LLM_ONLY_METHOD_NAME].latency_overhead_ms == 0.0
    # slow baseline overhead: 25 - 5 = 20 per query.
    assert result.method_metrics["slow"].latency_overhead_ms == pytest.approx(20.0)


def test_overhead_zero_without_llm_only_baseline():
    dataset = make_dataset()
    system = StubSystem({item.query: verified(item.ground_truth, 1, 0) for item in dataset})
    other = FixedLatencyMethod("chain-of-thought", latency_ms=50.0)
    harness = EvaluationHarness(system, baselines=[other])

    result = harness.run(dataset, config=make_config(), model_id="m")
    assert result.method_metrics["chain-of-thought"].latency_overhead_ms == 0.0


def test_trace_pipeline_latency_preferred_over_wallclock():
    from nsr.models import LatencyRecord

    item = DatasetItem(item_id="x", query="q", ground_truth="a", domain=Domain.MATH)
    out = verified("a", 1, 0)
    out.proof_trace.latency = LatencyRecord(pipeline_ms=123.0, system2_ms=0.0, llm_ms=0.0)
    system = StubSystem({"q": out})
    harness = EvaluationHarness(system, baselines=[])

    result = harness.run([item], config=make_config(), model_id="m")
    assert result.method_metrics[SYSTEM_METHOD_NAME].mean_latency_ms == pytest.approx(123.0)
