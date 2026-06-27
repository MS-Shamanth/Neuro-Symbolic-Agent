"""End-to-end integration test for a full evaluation run (Task 14.4).

This test exercises the *entire* evaluation pipeline together, with no network and no
local model -- the only mocked seam is the LLM backend
(:class:`~nsr.llm_component.MockBackend`). A small multi-domain dataset is built through
the real :class:`~nsr.dataset_loader.DatasetLoader`, the **System under test** (the real
:class:`~nsr.orchestrator.PipelineOrchestrator` wired over every real reasoning
component) and all configured :mod:`nsr.baselines` are run over every item by the real
:class:`~nsr.evaluation_harness.EvaluationHarness`, a real comparison report is built,
and the run record is persisted durably together with the report and read back from
disk.

The flow asserts the three things Task 14.4 requires:

1. **All methods present** -- the report identifies the System and every configured
   Baseline_Method (Req 9.1, 9.4).
2. **All metrics with differences** -- every compared metric carries the System value,
   each baseline value, and the numeric ``System - baseline`` difference (Req 9.2, 9.4).
3. **Durable, associated run record** -- the run record and the report are persisted
   together; reading the file back shows the run record (with its dataset ids) and the
   metrics remain associated (Req 13.4).

**Validates: Requirements 9.1, 9.2, 9.4, 13.4**
"""

from __future__ import annotations

import json

from nsr.actr_controller import ACTRController
from nsr.baselines import build_baseline
from nsr.comparison_report import (
    REASONING_CONSISTENCY_METRIC,
    REPORT_METRICS,
    build_comparison_report,
    persist_comparison_report,
)
from nsr.constrained_decoder import ConstrainedDecoder
from nsr.dataset_loader import load_dataset
from nsr.evaluation_harness import (
    LLM_ONLY_METHOD_NAME,
    SYSTEM_METHOD_NAME,
    EvaluationHarness,
)
from nsr.llm_component import LLMComponent, MockBackend
from nsr.models import Domain, ProductionRule, SystemConfig, VerifiedOutput
from nsr.orchestrator import PipelineOrchestrator
from nsr.repair_coordinator import RepairCoordinator
from nsr.reproducibility import ReproducibilityManager
from nsr.translation_layer import TranslationLayer
from nsr.validation_engine import ValidationEngine

# A production rule that is always applicable (empty condition) and satisfied by any
# step whose logic form contains "ok", so the scripted backend drives goal satisfaction.
REQUIRE_OK = ProductionRule(rule_id="R-ok", condition="", action="THEN ok")

# The configured baselines for the run (the System is compared against each of these).
BASELINE_NAMES = (LLM_ONLY_METHOD_NAME, "chain-of-thought", "react")


def make_config() -> SystemConfig:
    """A System configuration with repeated runs so Reasoning_Consistency is computed."""
    return SystemConfig(
        max_cycle_limit=10,
        repair_attempt_limit=2,
        retry_count=0,
        llm_selection="mock",
        output_format="json",
        conflict_resolution_policy="priority",
        generation_timeout_ms=30000,
        repeated_run_count=2,
        random_seed=7,
    )


def json_step(logic_form: str) -> str:
    """A constrained-decoder-shaped JSON completion carrying a logic form."""
    return json.dumps({"logic_form": logic_form})


def build_system(config: SystemConfig) -> PipelineOrchestrator:
    """Assemble the real orchestrator over a scripted backend (no network/model).

    Every collaborator -- translation, constrained decoding, ACT-R control, validation,
    repair, and trace building -- is the real implementation; only the LLM backend is the
    in-memory :class:`MockBackend`. The scripted ``"derive ok"`` step is accepted by
    ``REQUIRE_OK`` every cycle, so each query drives the four-stage cycle to goal
    satisfaction and yields a :class:`VerifiedOutput` with a populated Proof_Trace.
    """
    backend = MockBackend([json_step("derive ok")])
    llm = LLMComponent(backend, config)
    translation = TranslationLayer()
    validation = ValidationEngine()
    repair = RepairCoordinator(llm, translation, validation, config.repair_attempt_limit)
    return PipelineOrchestrator(
        llm=llm,
        translation=translation,
        decoder=ConstrainedDecoder(llm, config),
        controller=ACTRController(config.conflict_resolution_policy),
        validation=validation,
        config=config,
        repair=repair,
        procedural_memory=[REQUIRE_OK],
    )


def build_baselines() -> dict[str, object]:
    """Construct each configured baseline over its own scripted backend."""
    baselines: dict[str, object] = {}
    for name in BASELINE_NAMES:
        baselines[name] = build_baseline(name, MockBackend(["Answer: ok"]))
    return baselines


def make_multi_domain_dataset():
    """Load a small dataset spanning four of the six benchmark domains."""
    raw_items = [
        {
            "item_id": "math-1",
            "query": "What is 2 + 2?",
            "ground_truth": "4",
            "domain": Domain.MATH.value,
        },
        {
            "item_id": "commonsense-1",
            "query": "What color is the clear daytime sky?",
            "ground_truth": "blue",
            "domain": Domain.COMMONSENSE.value,
        },
        {
            "item_id": "logic-1",
            "query": "All cats are mammals. then all mammals are animals. then are cats animals?",
            "ground_truth": "yes",
            "domain": Domain.LOGIC_PUZZLE.value,
        },
        {
            "item_id": "science-1",
            "query": "What gas do plants primarily absorb during photosynthesis?",
            "ground_truth": "carbon dioxide",
            "domain": Domain.SCIENCE.value,
        },
    ]
    load_result = load_dataset(raw_items)
    return load_result


def test_end_to_end_evaluation_run(tmp_path):
    """Run a multi-domain dataset through the harness and verify the persisted report.

    Validates: Requirements 9.1, 9.2, 9.4, 13.4
    """
    config = make_config()

    # --- Dataset: span multiple of the six domains via the real loader (Req 10) -----
    load_result = make_multi_domain_dataset()
    dataset = load_result.items
    assert load_result.report.total_validated == 4
    assert {item.domain for item in dataset} == {
        Domain.MATH,
        Domain.COMMONSENSE,
        Domain.LOGIC_PUZZLE,
        Domain.SCIENCE,
    }

    # --- Execution: run the System + every baseline over every item (Req 9.1) -------
    system = build_system(config)
    baselines = build_baselines()
    harness = EvaluationHarness(system, baselines=baselines)

    # Repeated runs so Reasoning_Consistency is computed across them (Req 9.6).
    runs = [
        harness.run(dataset, config=config, model_id="mock-model-1")
        for _ in range(config.repeated_run_count)
    ]

    # The System produced a real Verified_Output for every item, every run (Req 9.1).
    for run in runs:
        sys_outcomes = run.per_item_outcomes[SYSTEM_METHOD_NAME]
        assert {o.item_id for o in sys_outcomes} == {item.item_id for item in dataset}
        assert run.exclusions == []
        assert all(o.has_trace for o in sys_outcomes)
    # Sanity-check the System path actually runs the real orchestrator end-to-end.
    direct = system.run(dataset[0].query)
    assert isinstance(direct, VerifiedOutput)

    # --- Report: build the comparison report across the repeated runs (Req 9.4) -----
    report = build_comparison_report(runs, repeated_run_count=config.repeated_run_count)

    # (1) The report includes the System and every configured baseline method.
    assert report.system_method == SYSTEM_METHOD_NAME
    assert report.baseline_methods == sorted(BASELINE_NAMES)
    for name in (SYSTEM_METHOD_NAME, *BASELINE_NAMES):
        assert name in runs[0].method_metrics

    # (2) Every metric is present with a System value, baseline values, and numeric
    #     System-minus-baseline differences.
    assert [m.metric for m in report.metrics] == list(REPORT_METRICS)
    for comparison in report.metrics:
        assert comparison.system_value is not None, comparison.metric
        assert set(comparison.baseline_values) == set(BASELINE_NAMES), comparison.metric
        assert set(comparison.differences) == set(BASELINE_NAMES), comparison.metric
        for name in BASELINE_NAMES:
            assert comparison.baseline_values[name] is not None, (comparison.metric, name)
            diff = comparison.differences[name]
            assert diff is not None, (comparison.metric, name)
            assert isinstance(diff, float)
            assert diff == comparison.system_value - comparison.baseline_values[name]

    # Reasoning_Consistency was genuinely computed across the repeated runs (Req 9.6).
    rc = next(m for m in report.metrics if m.metric == REASONING_CONSISTENCY_METRIC)
    assert rc.system_value is not None

    # --- Persistence: write the run record + report durably and read it back (Req 13.4)
    out_path = tmp_path / "results" / "evaluation_run.json"
    error = persist_comparison_report(
        ReproducibilityManager(), runs[0].run_record, report, out_path
    )
    assert error is None
    assert out_path.exists()

    document = json.loads(out_path.read_text(encoding="utf-8"))

    # (3) The run record and the metrics are persisted together and remain associated.
    assert "run_record" in document
    assert "metrics" in document
    persisted_run_record = document["run_record"]
    assert persisted_run_record["model_id"] == "mock-model-1"
    # The run record is durably tied to the dataset it was evaluated over.
    assert persisted_run_record["dataset_ids"] == [item.item_id for item in dataset]

    persisted_report = document["metrics"]
    assert persisted_report["system_method"] == SYSTEM_METHOD_NAME
    assert persisted_report["baseline_methods"] == sorted(BASELINE_NAMES)
    persisted_metric_names = {m["metric"] for m in persisted_report["metrics"]}
    assert persisted_metric_names == set(REPORT_METRICS)
