"""Unit tests for the Pipeline Orchestrator termination, output, and error handling.

These exercise the Task 10.2 responsibilities layered on top of the four-stage cycle:

- **Goal satisfaction** -- terminate and emit a :class:`~nsr.models.VerifiedOutput` with
  the attached Faithfulness_Score on goal satisfaction (Req 1.3, 7.6).
- **Cycle-limit-reached** -- terminate with ``cycle-limit-reached`` at the cycle bound
  (Req 1.4).
- **Constraint-unsatisfied** -- surface the decoder's ``constraint-unsatisfied``
  termination (Req 3.4).
- **Repair-exhausted** -- drive the Repair Coordinator on repair-triggering outcomes and
  surface ``repair-exhausted`` when the attempt limit is reached (Req 6.4-6.6).
- **Component errors** -- convert an unavailable LLM and a back-translation failure into
  ``component-error`` :class:`~nsr.models.ErrorRecord`\\ s while preserving the
  Proof_Trace (Req 1.6).

Everything is driven by :class:`~nsr.llm_component.MockBackend`, so no network or local
model is needed.
"""

from __future__ import annotations

import json

from nsr.actr_controller import ACTRController
from nsr.constrained_decoder import ConstrainedDecoder
from nsr.llm_component import BackendUnavailable, LLMComponent, MockBackend
from nsr.models import (
    BackTranslationError,
    ErrorRecord,
    ProductionRule,
    SystemConfig,
    TerminationReason,
    ValidationStatus,
    VerifiedOutput,
)
from nsr.orchestrator import PipelineOrchestrator
from nsr.repair_coordinator import RepairCoordinator
from nsr.translation_layer import TRANSLATION_LAYER_COMPONENT, TranslationLayer
from nsr.validation_engine import ValidationEngine


# --------------------------------------------------------------------------- helpers


def make_config(
    *,
    max_cycle_limit: int = 10,
    repair_attempt_limit: int = 0,
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
# Always applicable, never satisfiable -> forces rejection on every step.
NEVER_SAT = ProductionRule(
    rule_id="R-never", condition="", action="THEN __unsatisfiable_token__"
)


def build_orchestrator(
    *,
    script,
    procedural_memory=None,
    config: SystemConfig | None = None,
    translation: TranslationLayer | None = None,
    with_repair: bool = False,
) -> tuple[PipelineOrchestrator, MockBackend]:
    """Assemble an orchestrator over a scripted :class:`MockBackend`.

    ``script`` items are raw completion strings handed verbatim to the LLM (use
    :func:`json_step` for conforming JSON steps, or a plain string to drive a
    non-conforming output). A :class:`RepairCoordinator` is wired in when
    ``with_repair`` is set.
    """
    cfg = config or make_config()
    backend = MockBackend(list(script))
    llm = LLMComponent(backend, cfg)
    trans = translation or TranslationLayer()
    validation = ValidationEngine()
    repair = (
        RepairCoordinator(llm, trans, validation, cfg.repair_attempt_limit)
        if with_repair
        else None
    )
    orchestrator = PipelineOrchestrator(
        llm=llm,
        translation=trans,
        decoder=ConstrainedDecoder(llm, cfg),
        controller=ACTRController(cfg.conflict_resolution_policy),
        validation=validation,
        config=cfg,
        repair=repair,
        procedural_memory=list(procedural_memory) if procedural_memory else [],
    )
    return orchestrator, backend


class FlakyBackTranslation(TranslationLayer):
    """A Translation_Layer whose backward translation fails on the Nth ``to_context``.

    Earlier calls delegate to the real implementation, so a query can accept one step
    before the back-translation failure on the next cycle, letting the test assert the
    prior Proof_Trace contents are preserved (Req 1.6/5.5).
    """

    def __init__(self, fail_on_call: int) -> None:
        super().__init__()
        self._calls = 0
        self._fail_on_call = fail_on_call

    def to_context(self, state, *, builder=None, proof_step=None):
        self._calls += 1
        if self._calls >= self._fail_on_call:
            reason = "symbolic state could not be converted into LLM context"
            record = ErrorRecord(
                failed_component=TRANSLATION_LAYER_COMPONENT, reason=reason
            )
            raise BackTranslationError(reason, record)
        return super().to_context(state, builder=builder, proof_step=proof_step)


# -------------------------------------------------------------- goal satisfaction


def test_goal_satisfied_emits_verified_output_with_score():
    orchestrator, _ = build_orchestrator(
        script=[json_step("x ok"), json_step("y ok")],
        procedural_memory=[REQUIRE_OK],
    )

    result = orchestrator.run("step one. step two")

    assert isinstance(result, VerifiedOutput)
    assert result.proof_trace.termination_reason == TerminationReason.GOAL_SATISFIED
    # Faithfulness_Score is attached (Req 7.6): both steps accepted -> 1.0.
    assert result.faithfulness_score == 1.0
    assert len(result.proof_trace.steps) == 2
    assert all(
        step.status == ValidationStatus.ACCEPTED for step in result.proof_trace.steps
    )
    # The terminated trace is also recoverable via last_trace.
    assert orchestrator.last_trace is result.proof_trace


def test_partial_faithfulness_score_attached_on_goal_satisfaction():
    # Reject the first step (no repair), then satisfy the goal on the second cycle.
    orchestrator, _ = build_orchestrator(
        script=[json_step("nope"), json_step("done ok")],
        procedural_memory=[REQUIRE_OK],
    )

    result = orchestrator.run("only goal")

    assert isinstance(result, VerifiedOutput)
    assert result.proof_trace.termination_reason == TerminationReason.GOAL_SATISFIED
    # One rejected + one accepted step -> faithfulness 0.5 (Req 7.1, 7.6).
    assert result.faithfulness_score == 0.5
    assert len(result.proof_trace.steps) == 2


# ------------------------------------------------------------- cycle-limit-reached


def test_cycle_limit_reached_emits_trace_with_reason():
    orchestrator, _ = build_orchestrator(
        script=[json_step("bad")],
        procedural_memory=[NEVER_SAT],
        config=make_config(max_cycle_limit=3),
    )

    result = orchestrator.run("a. b. c. d. e")

    assert isinstance(result, VerifiedOutput)
    assert orchestrator.completed_cycles == 3
    assert (
        result.proof_trace.termination_reason
        == TerminationReason.CYCLE_LIMIT_REACHED
    )
    # Every cycle journaled a (rejected) step (Req 1.5).
    assert len(result.proof_trace.steps) == 3


# --------------------------------------------------------- constraint-unsatisfied


def test_constraint_unsatisfied_surfaced_from_decoder():
    # Plain (non-JSON) output never conforms; with retry_count=0 the decoder exhausts
    # immediately and the orchestrator surfaces constraint-unsatisfied (Req 3.4).
    orchestrator, _ = build_orchestrator(
        script=["this is not json"],
        procedural_memory=[REQUIRE_OK],
        config=make_config(retry_count=0),
    )

    result = orchestrator.run("solve it")

    assert isinstance(result, VerifiedOutput)
    assert (
        result.proof_trace.termination_reason
        == TerminationReason.CONSTRAINT_UNSATISFIED
    )
    assert orchestrator.last_trace is result.proof_trace


# -------------------------------------------------------------- repair-exhausted


def test_repair_exhausted_surfaced_from_coordinator():
    # Every step is rejected and the regenerated steps never improve; the repair
    # coordinator reaches its attempt limit and surfaces repair-exhausted (Req 6.6).
    orchestrator, _ = build_orchestrator(
        script=[json_step("bad")],
        procedural_memory=[NEVER_SAT],
        config=make_config(max_cycle_limit=5, repair_attempt_limit=2),
        with_repair=True,
    )

    result = orchestrator.run("single goal")

    assert isinstance(result, VerifiedOutput)
    assert (
        result.proof_trace.termination_reason == TerminationReason.REPAIR_EXHAUSTED
    )
    # The first (rejected) step recorded its repair attempts, bounded by the limit.
    assert len(result.proof_trace.steps) >= 1
    first_step = result.proof_trace.steps[0]
    assert first_step.status == ValidationStatus.REJECTED
    assert len(first_step.repair_attempts) == 2


def test_repair_exhausted_with_zero_attempt_limit():
    # A limit of 0 permits no repair and yields an immediate repair-exhausted outcome.
    orchestrator, _ = build_orchestrator(
        script=[json_step("bad")],
        procedural_memory=[NEVER_SAT],
        config=make_config(max_cycle_limit=5, repair_attempt_limit=0),
        with_repair=True,
    )

    result = orchestrator.run("single goal")

    assert isinstance(result, VerifiedOutput)
    assert (
        result.proof_trace.termination_reason == TerminationReason.REPAIR_EXHAUSTED
    )
    assert result.proof_trace.steps[0].repair_attempts == []


# ----------------------------------------------------------------- component error


def test_llm_unavailable_converted_to_error_record_preserving_trace():
    # First cycle accepts a step; the second cycle's generation is unavailable. The
    # failure becomes a component-error error record naming the LLM, and the accepted
    # step recorded before the failure is preserved (Req 1.6, 2.6).
    orchestrator, _ = build_orchestrator(
        script=[json_step("a ok"), BackendUnavailable("backend down")],
        procedural_memory=[REQUIRE_OK],
        config=make_config(max_cycle_limit=5, retry_count=0),
    )

    result = orchestrator.run("step one. step two")

    assert isinstance(result, ErrorRecord)
    assert result.failed_component == "LLM"

    trace = orchestrator.last_trace
    assert trace is not None
    assert trace.termination_reason == TerminationReason.COMPONENT_ERROR
    assert trace.error_record is result
    # The Proof_Trace contents from before the failure are preserved (Req 1.6).
    assert len(trace.steps) == 1
    assert trace.steps[0].status == ValidationStatus.ACCEPTED


def test_back_translation_failure_converted_to_error_record_preserving_trace():
    # First cycle accepts a step; back-translation then fails on the second cycle. The
    # failure becomes a component-error naming the Translation_Layer with the prior
    # accepted step preserved (Req 1.6, 5.5).
    flaky = FlakyBackTranslation(fail_on_call=2)
    orchestrator, _ = build_orchestrator(
        script=[json_step("a ok"), json_step("b ok")],
        procedural_memory=[REQUIRE_OK],
        config=make_config(max_cycle_limit=5),
        translation=flaky,
    )

    result = orchestrator.run("step one. step two")

    assert isinstance(result, ErrorRecord)
    assert result.failed_component == TRANSLATION_LAYER_COMPONENT

    trace = orchestrator.last_trace
    assert trace is not None
    assert trace.termination_reason == TerminationReason.COMPONENT_ERROR
    assert trace.error_record is result
    # The accepted step from the first cycle survives the back-translation failure.
    assert len(trace.steps) == 1
    assert trace.steps[0].status == ValidationStatus.ACCEPTED


def test_back_translation_failure_on_first_cycle_returns_error_record():
    # Back-translation fails on the very first generation; an error record is returned
    # with an (empty) Proof_Trace preserved and a component-error termination.
    flaky = FlakyBackTranslation(fail_on_call=1)
    orchestrator, _ = build_orchestrator(
        script=[json_step("a ok")],
        procedural_memory=[REQUIRE_OK],
        translation=flaky,
    )

    result = orchestrator.run("only goal")

    assert isinstance(result, ErrorRecord)
    assert result.failed_component == TRANSLATION_LAYER_COMPONENT
    trace = orchestrator.last_trace
    assert trace is not None
    assert trace.termination_reason == TerminationReason.COMPONENT_ERROR
    assert trace.steps == []
