"""Unit tests for the Pipeline Orchestrator intake and four-stage cycle (Task 10.1).

These exercise the two responsibilities of Task 10.1:

- **Query intake** -- empty/unparseable queries are rejected with an
  :class:`~nsr.models.ErrorRecord` *before* the reasoning cycle is initialized, and a
  valid query initializes the Goal_Buffer with the parsed goal (Req 1.1, 1.7).
- **Cycle execution** -- cycles run in the fixed four-stage order (generate, translate,
  controller update, validate) bounded by the configured maximum cycle limit (Req 1.2).

The LLM is driven by :class:`~nsr.llm_component.MockBackend`, so no network or local
model is needed. Full termination/output/error semantics are Task 10.2; here we assert
only the intake and ordering/bound guarantees plus the minimal emission path.
"""

from __future__ import annotations

import json

import pytest

from nsr.actr_controller import ACTRController
from nsr.constrained_decoder import ConstrainedDecoder
from nsr.llm_component import LLMComponent, MockBackend
from nsr.models import (
    ErrorRecord,
    ProductionRule,
    SystemConfig,
    TerminationReason,
    VerifiedOutput,
)
from nsr.orchestrator import (
    STAGE_ORDER,
    CycleStage,
    PipelineOrchestrator,
    parse_query,
)
from nsr.translation_layer import TranslationLayer
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


def build_orchestrator(
    *,
    scripted_logic_forms,
    procedural_memory=None,
    config: SystemConfig | None = None,
    on_stage=None,
) -> tuple[PipelineOrchestrator, MockBackend]:
    """Assemble an orchestrator over a scripted MockBackend.

    Each scripted item is a logic-form string wrapped into a conforming JSON step.
    """
    cfg = config or make_config()
    backend = MockBackend([json_step(lf) for lf in scripted_logic_forms])
    llm = LLMComponent(backend, cfg)
    orchestrator = PipelineOrchestrator(
        llm=llm,
        translation=TranslationLayer(),
        decoder=ConstrainedDecoder(llm, cfg),
        controller=ACTRController(cfg.conflict_resolution_policy),
        validation=ValidationEngine(),
        config=cfg,
        procedural_memory=list(procedural_memory) if procedural_memory else [],
        on_stage=on_stage,
    )
    return orchestrator, backend


# A rule that is always applicable (empty condition) and satisfied by any step whose
# logic form contains "ok"; steps without "ok" are rejected.
REQUIRE_OK = ProductionRule(rule_id="R-ok", condition="", action="THEN ok")
# A rule that is always applicable but can never be satisfied, forcing rejection.
NEVER_SAT = ProductionRule(
    rule_id="R-never", condition="", action="THEN __unsatisfiable_token__"
)


# ------------------------------------------------------------------- query parsing


def test_parse_query_rejects_empty_and_non_string():
    assert parse_query("") is None
    assert parse_query("   ") is None
    assert parse_query("\n\t ") is None
    assert parse_query(None) is None
    assert parse_query(123) is None


def test_parse_query_splits_sub_goals():
    goal = parse_query("first step. second step; third step")
    assert goal is not None
    assert goal.description == "first step. second step; third step"
    assert [sg.description for sg in goal.sub_goals] == [
        "first step",
        "second step",
        "third step",
    ]
    assert all(not sg.satisfied for sg in goal.sub_goals)


def test_parse_query_single_clause_yields_one_sub_goal():
    goal = parse_query("solve the puzzle")
    assert goal is not None
    assert [sg.description for sg in goal.sub_goals] == ["solve the puzzle"]


# --------------------------------------------------------------- intake rejection


@pytest.mark.parametrize("bad_query", ["", "   ", "\n", None, 42, ["not", "a", "string"]])
def test_empty_or_unparseable_query_rejected_before_cycle(bad_query):
    orchestrator, backend = build_orchestrator(
        scripted_logic_forms=["step ok"], procedural_memory=[REQUIRE_OK]
    )

    result = orchestrator.run(bad_query)

    # Rejected with an ErrorRecord naming the Pipeline (Req 1.7).
    assert isinstance(result, ErrorRecord)
    assert result.failed_component == "Pipeline"

    # The reasoning cycle was never initialized / executed.
    assert orchestrator.completed_cycles == 0
    assert orchestrator.stage_log == []
    assert backend.call_count == 0  # no generation occurred

    # The (empty) trace carries the error record and no steps.
    trace = orchestrator.last_trace
    assert trace is not None
    assert trace.error_record is result
    assert trace.steps == []
    assert trace.termination_reason is None


def test_invalid_query_does_not_initialize_goal_buffer():
    cfg = make_config()
    backend = MockBackend([json_step("step ok")])
    llm = LLMComponent(backend, cfg)
    controller = ACTRController(cfg.conflict_resolution_policy)
    orchestrator = PipelineOrchestrator(
        llm=llm,
        translation=TranslationLayer(),
        decoder=ConstrainedDecoder(llm, cfg),
        controller=controller,
        validation=ValidationEngine(),
        config=cfg,
        procedural_memory=[REQUIRE_OK],
    )

    orchestrator.run("")

    # The controller was never initialized, so accessing its goal buffer raises.
    with pytest.raises(RuntimeError):
        _ = controller.goal_buffer


# --------------------------------------------------- goal-buffer initialization


def test_valid_query_initializes_goal_buffer_before_cycle():
    cfg = make_config()
    backend = MockBackend([json_step("step ok")])
    llm = LLMComponent(backend, cfg)
    controller = ACTRController(cfg.conflict_resolution_policy)
    orchestrator = PipelineOrchestrator(
        llm=llm,
        translation=TranslationLayer(),
        decoder=ConstrainedDecoder(llm, cfg),
        controller=controller,
        validation=ValidationEngine(),
        config=cfg,
        procedural_memory=[REQUIRE_OK],
    )

    orchestrator.run("only step")

    # Req 1.1: the Goal_Buffer holds the parsed query goal.
    assert controller.goal_buffer.description == "only step"
    assert [sg.description for sg in controller.goal_buffer.sub_goals] == ["only step"]


# ------------------------------------------------------- four-stage cycle order


def test_cycle_runs_four_stages_in_fixed_order():
    # One sub-goal, one accepted step -> exactly one cycle of four stages, in order.
    orchestrator, _ = build_orchestrator(
        scripted_logic_forms=["step ok"], procedural_memory=[REQUIRE_OK]
    )

    result = orchestrator.run("single goal")

    assert isinstance(result, VerifiedOutput)
    assert orchestrator.completed_cycles == 1
    # The fixed four-stage order within the single cycle (Req 1.2).
    assert orchestrator.stage_log == [(0, stage) for stage in STAGE_ORDER]


def test_multi_subgoal_query_runs_one_cycle_per_subgoal_in_order():
    # Three sub-goals, each step accepted -> three cycles, each four stages in order.
    orchestrator, _ = build_orchestrator(
        scripted_logic_forms=["a ok", "b ok", "c ok"],
        procedural_memory=[REQUIRE_OK],
    )

    result = orchestrator.run("do a. do b. do c")

    assert isinstance(result, VerifiedOutput)
    assert orchestrator.completed_cycles == 3
    expected = [(cycle, stage) for cycle in range(3) for stage in STAGE_ORDER]
    assert orchestrator.stage_log == expected
    # Each cycle's stages are exactly generate, translate, controller-update, validate.
    for cycle in range(3):
        stages = [s for (c, s) in orchestrator.stage_log if c == cycle]
        assert stages == list(STAGE_ORDER)


def test_on_stage_callback_observes_same_order():
    observed: list[tuple[int, CycleStage]] = []
    orchestrator, _ = build_orchestrator(
        scripted_logic_forms=["step ok"],
        procedural_memory=[REQUIRE_OK],
        on_stage=lambda cycle, stage: observed.append((cycle, stage)),
    )

    orchestrator.run("single goal")

    assert observed == [(0, stage) for stage in STAGE_ORDER]


# ------------------------------------------------------------ cycle limit bound


def test_cycles_bounded_by_max_cycle_limit():
    # Steps never satisfy the rule -> never accepted, goal never satisfied. The loop
    # must stop at exactly max_cycle_limit cycles (Req 1.2 bound; Task 10.3 Property 7).
    cfg = make_config(max_cycle_limit=4)
    orchestrator, backend = build_orchestrator(
        scripted_logic_forms=["bad"] * 20,
        procedural_memory=[NEVER_SAT],
        config=cfg,
    )

    result = orchestrator.run("a. b. c. d. e. f. g")

    assert orchestrator.completed_cycles == 4
    # Each of the four cycles executed the full four-stage sequence, in order.
    expected = [(cycle, stage) for cycle in range(4) for stage in STAGE_ORDER]
    assert orchestrator.stage_log == expected
    # Minimal termination path: reaching the bound yields cycle-limit-reached.
    assert isinstance(result, VerifiedOutput)
    assert result.proof_trace.termination_reason == TerminationReason.CYCLE_LIMIT_REACHED


def test_cycle_limit_of_one_runs_single_cycle():
    cfg = make_config(max_cycle_limit=1)
    orchestrator, _ = build_orchestrator(
        scripted_logic_forms=["bad"] * 5,
        procedural_memory=[NEVER_SAT],
        config=cfg,
    )

    orchestrator.run("a. b. c")

    assert orchestrator.completed_cycles == 1
    assert orchestrator.stage_log == [(0, stage) for stage in STAGE_ORDER]


def test_goal_satisfaction_terminates_before_cycle_limit():
    # Two sub-goals satisfied in two cycles, well under the limit of 10.
    orchestrator, _ = build_orchestrator(
        scripted_logic_forms=["x ok", "y ok"],
        procedural_memory=[REQUIRE_OK],
        config=make_config(max_cycle_limit=10),
    )

    result = orchestrator.run("step one. step two")

    assert isinstance(result, VerifiedOutput)
    assert orchestrator.completed_cycles == 2
    assert result.proof_trace.termination_reason == TerminationReason.GOAL_SATISFIED
    # Faithfulness is attached and both accepted steps are journaled.
    assert result.faithfulness_score == 1.0
    assert len(result.proof_trace.steps) == 2
