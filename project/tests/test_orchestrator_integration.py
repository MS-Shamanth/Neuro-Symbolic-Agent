"""End-to-end integration tests for the Pipeline Orchestrator (Task 10.4).

Unlike the focused unit suites (``test_orchestrator_intake_cycle`` and
``test_orchestrator_termination``), these tests exercise the orchestrator *through every
real component of the reasoning pipeline* -- the real :class:`TranslationLayer`,
:class:`ConstrainedDecoder`, :class:`ACTRController`, :class:`ValidationEngine`,
:class:`RepairCoordinator`, :class:`ProofTraceBuilder`, and the real Metrics Engine
faithfulness computation. The *only* seam that is mocked is the LLM backend itself, via
:class:`~nsr.llm_component.MockBackend`, so no network or local model is needed while the
full neuro-symbolic cycle runs for real.

The three end-to-end flows required by Task 10.4 are covered:

1. **Goal-satisfied run** -- a valid query drives the full four-stage cycle through every
   component to goal satisfaction, emitting a :class:`VerifiedOutput` whose attached
   Faithfulness_Score is the real accepted/total computation and whose Proof_Trace is
   populated with the executed steps in order (Req 1.1, 1.3, 7.6).
2. **Empty / unparseable query rejection** -- intake rejects the query with an
   :class:`ErrorRecord` *before* the reasoning cycle is ever initialized, so no component
   runs and the trace stays empty (Req 1.7).
3. **LLM-unavailable error path** -- when the (mocked) LLM backend becomes unavailable
   mid-run, the failure is converted into a component-error :class:`ErrorRecord` while the
   Proof_Trace contents accumulated by the real components before the failure are
   preserved (Req 1.6).
"""

from __future__ import annotations

import json

import pytest

from nsr.actr_controller import ACTRController
from nsr.constrained_decoder import ConstrainedDecoder
from nsr.llm_component import BackendUnavailable, LLMComponent, MockBackend
from nsr.models import (
    ErrorRecord,
    ProductionRule,
    SystemConfig,
    TerminationReason,
    ValidationStatus,
    VerifiedOutput,
)
from nsr.orchestrator import STAGE_ORDER, PipelineOrchestrator
from nsr.repair_coordinator import RepairCoordinator
from nsr.translation_layer import TranslationLayer
from nsr.validation_engine import ValidationEngine


# --------------------------------------------------------------------------- helpers


def make_config(
    *,
    max_cycle_limit: int = 10,
    repair_attempt_limit: int = 2,
    retry_count: int = 0,
) -> SystemConfig:
    return SystemConfig(
        max_cycle_limit=max_cycle_limit,
        repair_attempt_limit=repair_attempt_limit,
        retry_count=retry_count,
        llm_selection="gpt-4o-mini",
        output_format="json",
        conflict_resolution_policy="priority",
        generation_timeout_ms=30000,
    )


def json_step(logic_form: str) -> str:
    return json.dumps({"logic_form": logic_form})


# Always applicable; satisfied by any step whose logic form contains "ok".
REQUIRE_OK = ProductionRule(rule_id="R-ok", condition="", action="THEN ok")


def build_full_pipeline(
    *,
    script,
    procedural_memory=None,
    config: SystemConfig | None = None,
    with_repair: bool = True,
) -> tuple[PipelineOrchestrator, MockBackend]:
    """Assemble an orchestrator wiring *all real components* over a scripted backend.

    Every collaborator is the real implementation; only the LLM backend is the in-memory
    :class:`MockBackend` so the four-stage cycle exercises real translation, decoding,
    control, validation, repair, and trace building end-to-end. The real
    :class:`RepairCoordinator` is wired in by default; set ``with_repair=False`` to let a
    rejected step simply advance the cycle (so the rejection is retained in the trace
    rather than being driven into the repair sub-loop).
    """
    cfg = config or make_config()
    backend = MockBackend(list(script))
    llm = LLMComponent(backend, cfg)
    translation = TranslationLayer()
    validation = ValidationEngine()
    repair = (
        RepairCoordinator(llm, translation, validation, cfg.repair_attempt_limit)
        if with_repair
        else None
    )
    orchestrator = PipelineOrchestrator(
        llm=llm,
        translation=translation,
        decoder=ConstrainedDecoder(llm, cfg),
        controller=ACTRController(cfg.conflict_resolution_policy),
        validation=validation,
        config=cfg,
        repair=repair,
        procedural_memory=list(procedural_memory) if procedural_memory else [],
    )
    return orchestrator, backend


# ----------------------------------------------- 1. goal-satisfied end-to-end run


def test_goal_satisfied_run_emits_verified_output_with_score_and_populated_trace():
    """A valid query runs end-to-end to goal satisfaction (Req 1.1, 1.3, 7.6).

    All real components drive the cycle; every scripted step is accepted, so the goal is
    satisfied and a :class:`VerifiedOutput` is emitted carrying the real
    Faithfulness_Score and a Proof_Trace populated with the executed steps.
    """
    orchestrator, backend = build_full_pipeline(
        script=[json_step("derive a ok"), json_step("derive b ok"), json_step("conclude c ok")],
        procedural_memory=[REQUIRE_OK],
    )

    result = orchestrator.run("establish a. then establish b. then conclude c")

    # A Verified_Output is emitted on goal satisfaction (Req 1.3).
    assert isinstance(result, VerifiedOutput)
    assert result.proof_trace.termination_reason == TerminationReason.GOAL_SATISFIED

    # The Faithfulness_Score is attached and equals the real accepted/total (Req 7.6).
    assert result.faithfulness_score == 1.0

    # The Proof_Trace is populated with every executed step, in order, all accepted
    # (Req 1.3 emission carries the trace; the metric is derived from it).
    assert len(result.proof_trace.steps) == 3
    assert [s.sequence for s in result.proof_trace.steps] == [0, 1, 2]
    assert all(
        s.status == ValidationStatus.ACCEPTED for s in result.proof_trace.steps
    )
    assert all(s.applied_rule_id == "R-ok" for s in result.proof_trace.steps)

    # The final answer was derived from the accepted reasoning state.
    assert result.final_answer != ""

    # The Goal_Buffer was initialized from the parsed query before any step (Req 1.1):
    # one cycle ran per sub-goal, each executing the full four-stage sequence in order.
    assert orchestrator.completed_cycles == 3
    expected = [(cycle, stage) for cycle in range(3) for stage in STAGE_ORDER]
    assert orchestrator.stage_log == expected
    # The full cycle actually invoked the (mocked) LLM once per cycle.
    assert backend.call_count == 3
    # The terminated trace is recoverable via last_trace.
    assert orchestrator.last_trace is result.proof_trace


def test_goal_satisfied_run_attaches_partial_faithfulness_score():
    """A rejected-then-accepted run attaches the real partial Faithfulness_Score (Req 7.6).

    The first step is rejected (no repair coordinator success on it because it is
    immediately followed by an accepted retry on the next cycle), exercising the real
    Metrics Engine on a mixed-outcome trace.
    """
    orchestrator, _ = build_full_pipeline(
        script=[json_step("wrong"), json_step("right ok")],
        procedural_memory=[REQUIRE_OK],
        # No repair coordinator, so the rejected step is not driven into the repair
        # sub-loop: it stays rejected in the trace and the next cycle's accepted step
        # advances the goal to satisfaction, leaving a mixed-outcome trace to score.
        with_repair=False,
    )

    result = orchestrator.run("only goal")

    assert isinstance(result, VerifiedOutput)
    assert result.proof_trace.termination_reason == TerminationReason.GOAL_SATISFIED
    # One rejected + one accepted step -> faithfulness 0.5 (accepted/total).
    assert result.faithfulness_score == 0.5
    assert len(result.proof_trace.steps) == 2


# ------------------------------------------ 2. empty / unparseable query rejection


@pytest.mark.parametrize(
    "bad_query",
    ["", "   ", "\n\t ", None, 42, ["not", "a", "string"]],
)
def test_empty_or_unparseable_query_rejected_before_cycle(bad_query):
    """Invalid queries are rejected with an ErrorRecord before any component runs (Req 1.7)."""
    orchestrator, backend = build_full_pipeline(
        script=[json_step("never reached ok")],
        procedural_memory=[REQUIRE_OK],
    )

    result = orchestrator.run(bad_query)

    # Rejected with an ErrorRecord identifying the invalid query (Req 1.7).
    assert isinstance(result, ErrorRecord)
    assert result.failed_component == "Pipeline"
    assert result.reason  # a non-empty explanation identifying the invalid query

    # The reasoning cycle was never initialized: no component ran, no LLM call occurred.
    assert orchestrator.completed_cycles == 0
    assert orchestrator.stage_log == []
    assert backend.call_count == 0

    # The (empty) trace carries the error record and has no steps or termination reason.
    trace = orchestrator.last_trace
    assert trace is not None
    assert trace.error_record is result
    assert trace.steps == []
    assert trace.termination_reason is None


# --------------------------------------------- 3. LLM-unavailable error path


def test_llm_unavailable_midrun_records_error_preserving_trace():
    """An LLM outage mid-run yields a component-error ErrorRecord with the trace kept (Req 1.6).

    The first cycle runs fully through every real component and accepts a step; the
    second cycle's generation hits an unavailable backend. The orchestrator halts the
    query, returns an error record naming the LLM, and preserves the Proof_Trace contents
    accumulated before the failure.
    """
    orchestrator, _ = build_full_pipeline(
        script=[json_step("first ok"), BackendUnavailable("backend down")],
        procedural_memory=[REQUIRE_OK],
        config=make_config(max_cycle_limit=5, retry_count=0),
    )

    result = orchestrator.run("step one. then step two")

    # The failure is surfaced as a component-error error record naming the LLM (Req 1.6).
    assert isinstance(result, ErrorRecord)
    assert result.failed_component == "LLM"

    trace = orchestrator.last_trace
    assert trace is not None
    assert trace.termination_reason == TerminationReason.COMPONENT_ERROR
    assert trace.error_record is result

    # The existing Proof_Trace contents (the step accepted before the outage) are
    # preserved (Req 1.6) -- the trace was not discarded by the failure.
    assert len(trace.steps) == 1
    assert trace.steps[0].status == ValidationStatus.ACCEPTED
    assert trace.steps[0].applied_rule_id == "R-ok"


def test_llm_unavailable_on_first_cycle_records_error_with_empty_trace():
    """An LLM outage on the very first generation still preserves the (empty) trace (Req 1.6)."""
    orchestrator, _ = build_full_pipeline(
        script=[BackendUnavailable("backend down")],
        procedural_memory=[REQUIRE_OK],
        config=make_config(retry_count=0),
    )

    result = orchestrator.run("only goal")

    assert isinstance(result, ErrorRecord)
    assert result.failed_component == "LLM"

    trace = orchestrator.last_trace
    assert trace is not None
    assert trace.termination_reason == TerminationReason.COMPONENT_ERROR
    assert trace.error_record is result
    # No step was accepted before the outage, so the preserved trace is empty.
    assert trace.steps == []
