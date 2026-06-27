"""Property-based test for the cycle bound (Task 10.3).

**Property 7: Completed cycles never exceed the maximum cycle limit.**

For any maximum cycle limit ``M`` (1..bound) and any query that does not satisfy its
goal, the orchestrator runs at most ``M`` cycles and terminates for the
``cycle-limit-reached`` reason. The goal is forced never to be reached by validating
every generated step against a never-satisfiable production rule (``NEVER_SAT``) with no
repair coordinator wired in, so every cycle rejects its step and the active goal can
never be marked satisfied. The loop must therefore always run to the configured bound.

**Validates: Requirements 1.2, 1.4**
"""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from nsr.actr_controller import ACTRController
from nsr.constrained_decoder import ConstrainedDecoder
from nsr.llm_component import LLMComponent, MockBackend
from nsr.models import (
    ProductionRule,
    SystemConfig,
    TerminationReason,
    VerifiedOutput,
)
from nsr.orchestrator import PipelineOrchestrator
from nsr.translation_layer import TranslationLayer
from nsr.validation_engine import ValidationEngine


# A rule that is always applicable (empty condition) but can never be satisfied, so every
# generated step is rejected and the goal is never reached (reused from the unit setup).
NEVER_SAT = ProductionRule(
    rule_id="R-never", condition="", action="THEN __unsatisfiable_token__"
)


def _make_config(max_cycle_limit: int) -> SystemConfig:
    return SystemConfig(
        max_cycle_limit=max_cycle_limit,
        repair_attempt_limit=0,
        retry_count=0,
        llm_selection="gpt-4o-mini",
        output_format="json",
        conflict_resolution_policy="priority",
        generation_timeout_ms=30000,
    )


def _json_step(logic_form: str) -> str:
    return json.dumps({"logic_form": logic_form})


def _build_query(num_sub_goals: int) -> str:
    """Build a query whose parser yields ``num_sub_goals`` ordered sub-goals."""
    return ". ".join(f"goal{i}" for i in range(num_sub_goals))


# Arbitrary maximum cycle limit M within the valid configured range (1..10000); capped at
# a small bound here to keep each generated example fast while still exercising the
# guarantee across many distinct limits.
@settings(max_examples=200, deadline=None)
@given(
    max_cycle_limit=st.integers(min_value=1, max_value=40),
    num_sub_goals=st.integers(min_value=1, max_value=8),
)
def test_completed_cycles_never_exceed_max_cycle_limit(
    max_cycle_limit: int, num_sub_goals: int
):
    """Property 7: completed cycles are bounded by M and termination is cycle-limit.

    Validates: Requirements 1.2, 1.4
    """
    cfg = _make_config(max_cycle_limit)

    # Script strictly more failing steps than the loop could ever consume, so the backend
    # never runs dry before the orchestrator stops on its own cycle bound. None of these
    # logic forms can satisfy NEVER_SAT, so every cycle rejects its step.
    script = [_json_step("bad")] * (max_cycle_limit + 1)
    backend = MockBackend(script)
    llm = LLMComponent(backend, cfg)
    orchestrator = PipelineOrchestrator(
        llm=llm,
        translation=TranslationLayer(),
        decoder=ConstrainedDecoder(llm, cfg),
        controller=ACTRController(cfg.conflict_resolution_policy),
        validation=ValidationEngine(),
        config=cfg,
        # No repair coordinator: repair-triggering rejections simply advance the cycle.
        procedural_memory=[NEVER_SAT],
    )

    result = orchestrator.run(_build_query(num_sub_goals))

    # Bound: the number of completed cycles never exceeds the configured maximum (Req 1.2).
    assert orchestrator.completed_cycles <= max_cycle_limit

    # A goal that is never satisfied runs the loop to the bound (Req 1.4): exactly M
    # cycles complete and the run terminates for the cycle-limit-reached reason.
    assert orchestrator.completed_cycles == max_cycle_limit
    assert isinstance(result, VerifiedOutput)
    assert (
        result.proof_trace.termination_reason
        == TerminationReason.CYCLE_LIMIT_REACHED
    )
