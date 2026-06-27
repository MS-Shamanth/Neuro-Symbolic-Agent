"""Unit tests for the Constrained Decoder (Task 5.2).

These cover the decoder's three responsibilities:

- restricting LLM output to the configured structured format before return (Req 3.1);
- deriving the active decoding constraints from all four working-memory buffers
  (Req 3.5);
- marking and journalling non-conforming output, regenerating up to the retry count,
  and terminating with ``constraint-unsatisfied`` on exhaustion (Req 3.3, 3.4).

The decoder is exercised through the real :class:`~nsr.llm_component.LLMComponent`
driven by a scriptable :class:`~nsr.llm_component.MockBackend`, so no network or local
model is required.
"""

from __future__ import annotations

import pytest

from nsr.config_manager import load_config
from nsr.constrained_decoder import (
    NON_CONFORMING_KEY,
    ConstrainedDecoder,
    ConstraintUnsatisfied,
    DecodingConstraints,
    check_conformance,
    derive_constraints,
)
from nsr.llm_component import LLMComponent, MockBackend, OutputSchema
from nsr.models import (
    CandidateStep,
    Goal,
    PromptContext,
    ProductionRule,
    SubGoal,
    SymbolicRepresentation,
    TerminationReason,
    WorkingMemoryState,
)
from nsr.proof_trace import ProofTraceBuilder


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _config(output_format: str = "json", retry_count: int = 2):
    """Build a validated SystemConfig with the given format and retry count."""
    loaded = load_config(
        {
            "max_cycle_limit": 10,
            "repair_attempt_limit": 3,
            "retry_count": retry_count,
            "llm_selection": "gpt-4o-mini",
            "output_format": output_format,
            "conflict_resolution_policy": "priority",
            "generation_timeout_ms": 30000,
        }
    )
    return loaded.config


def _decoder(script, output_format: str = "json", retry_count: int = 2):
    """Build a decoder wrapping an LLMComponent over a scripted MockBackend."""
    config = _config(output_format=output_format, retry_count=retry_count)
    backend = MockBackend(script=script)
    llm = LLMComponent(backend, config)
    return ConstrainedDecoder(llm, config), backend


def _state() -> WorkingMemoryState:
    """A populated working-memory state spanning all four buffers."""
    return WorkingMemoryState(
        goal_buffer=Goal(
            description="solve the equation",
            sub_goals=[
                SubGoal(description="isolate x"),
                SubGoal(description="verify solution"),
            ],
        ),
        declarative_memory=[
            SymbolicRepresentation(logic_form="eq(2x+4,10)"),
            SymbolicRepresentation(logic_form="eq(2x,6)"),
        ],
        procedural_memory=[
            ProductionRule(rule_id="R1", condition="IF eq", action="THEN subtract"),
            ProductionRule(rule_id="R2", condition="IF isolate", action="THEN divide"),
        ],
        imaginal_buffer=SymbolicRepresentation(logic_form="eq(x,3)?"),
    )


def _context() -> PromptContext:
    return PromptContext(goal_description="solve the equation", prompt_text="prompt")


# ---------------------------------------------------------------------------
# Req 3.5 -- constraints derived from all four buffers
# ---------------------------------------------------------------------------


def test_derive_constraints_reads_all_four_buffers():
    constraints = derive_constraints(_state(), "json")

    # Goal_Buffer
    assert constraints.goal_terms == ["solve the equation"]
    assert constraints.sub_goal_terms == ["isolate x", "verify solution"]
    # Declarative_Memory
    assert constraints.established_conclusions == ["eq(2x+4,10)", "eq(2x,6)"]
    # Procedural_Memory
    assert constraints.rule_ids == ["R1", "R2"]
    assert constraints.rule_conditions == ["IF eq", "IF isolate"]
    # Imaginal_Buffer
    assert constraints.partial_representation == "eq(x,3)?"
    # Format requirement
    assert constraints.output_format == "json"
    assert "logic_form" in constraints.required_keys


def test_derive_constraints_rejects_unknown_format():
    with pytest.raises(ValueError):
        derive_constraints(_state(), "xml")


def test_to_schema_round_trips_constraint_fields():
    schema = derive_constraints(_state(), "logic-form").to_schema()
    assert schema["format"] == "logic-form"
    assert schema["rule_ids"] == ["R1", "R2"]
    assert schema["established_conclusions"] == ["eq(2x+4,10)", "eq(2x,6)"]
    assert schema["partial_representation"] == "eq(x,3)?"


# ---------------------------------------------------------------------------
# Req 3.1 -- conformance checking per format
# ---------------------------------------------------------------------------


def test_json_conforming_step_returns_normalised_candidate():
    constraints = derive_constraints(_state(), "json")
    step = CandidateStep(
        raw_text='{"logic_form": "sub(10,4)=6"}',
        structured={"logic_form": "sub(10,4)=6"},
    )
    result = check_conformance(step, constraints)
    assert result.conforming
    assert result.candidate.structured["logic_form"] == "sub(10,4)=6"
    assert result.candidate.structured["predicates"] == {}


def test_json_missing_logic_form_is_non_conforming():
    constraints = derive_constraints(_state(), "json")
    step = CandidateStep(raw_text='{"foo": 1}', structured={"foo": 1})
    result = check_conformance(step, constraints)
    assert not result.conforming
    assert "logic_form" in result.reason


def test_json_non_object_output_is_non_conforming():
    constraints = derive_constraints(_state(), "json")
    step = CandidateStep(raw_text="not json at all", structured={})
    result = check_conformance(step, constraints)
    assert not result.conforming


def test_logic_form_uses_raw_text_as_encoding():
    constraints = derive_constraints(_state(), "logic-form")
    step = CandidateStep(raw_text="  divide(6,2)=3  ", structured={})
    result = check_conformance(step, constraints)
    assert result.conforming
    assert result.candidate.structured["logic_form"] == "divide(6,2)=3"


def test_logic_form_empty_output_is_non_conforming():
    constraints = derive_constraints(_state(), "logic-form")
    step = CandidateStep(raw_text="   ", structured={})
    result = check_conformance(step, constraints)
    assert not result.conforming


def test_yaml_mapping_conforms_and_extracts_predicates():
    constraints = derive_constraints(_state(), "yaml")
    step = CandidateStep(
        raw_text="logic_form: eq(x,3)\nconfidence: high",
        structured={},
    )
    result = check_conformance(step, constraints)
    assert result.conforming
    assert result.candidate.structured["logic_form"] == "eq(x,3)"
    assert result.candidate.structured["predicates"] == {"confidence": "high"}


def test_yaml_without_logic_form_is_non_conforming():
    constraints = derive_constraints(_state(), "yaml")
    step = CandidateStep(raw_text="confidence: high", structured={})
    result = check_conformance(step, constraints)
    assert not result.conforming


# ---------------------------------------------------------------------------
# Req 3.1 -- decode passes the configured format to the LLM before return
# ---------------------------------------------------------------------------


def test_decode_constrains_output_to_configured_format():
    decoder, backend = _decoder(['{"logic_form": "x=3"}'], output_format="json")
    candidate = decoder.decode(_context(), _state())

    assert candidate.structured["logic_form"] == "x=3"
    # The schema handed to the backend carries the configured format (Req 3.1)
    assert backend.calls, "backend should have been invoked"
    _, schema, _ = backend.calls[0]
    assert isinstance(schema, OutputSchema)
    assert schema.format == "json"
    assert "logic_form" in schema.schema["required_keys"]


# ---------------------------------------------------------------------------
# Req 3.3 -- mark, journal, and regenerate non-conforming output
# ---------------------------------------------------------------------------


def test_decode_regenerates_until_conforming_and_journals_attempts():
    builder = ProofTraceBuilder()
    # First two attempts are non-conforming, the third conforms.
    decoder, backend = _decoder(
        ["garbage", "still bad", '{"logic_form": "x=3"}'],
        output_format="json",
        retry_count=2,
    )
    candidate = decoder.decode(_context(), _state(), builder=builder)

    assert candidate.structured["logic_form"] == "x=3"
    assert backend.call_count == 3  # two regenerations + the conforming attempt
    # Both non-conforming attempts were journalled into the trace (Req 3.3)
    assert len(builder.trace.steps) == 2
    for index, step in enumerate(builder.trace.steps):
        outcome = step.translation_outcomes[0]
        assert outcome[NON_CONFORMING_KEY] is True
        assert outcome["format"] == "json"
        assert outcome["attempt"] == index
    # A conforming step was found, so no termination reason is set.
    assert builder.trace.termination_reason is None


# ---------------------------------------------------------------------------
# Req 3.4 -- terminate with constraint-unsatisfied on exhaustion
# ---------------------------------------------------------------------------


def test_decode_exhausts_retries_and_terminates_constraint_unsatisfied():
    builder = ProofTraceBuilder()
    decoder, backend = _decoder(["nope"], output_format="json", retry_count=2)

    with pytest.raises(ConstraintUnsatisfied) as excinfo:
        decoder.decode(_context(), _state(), builder=builder)

    # retry_count=2 -> initial attempt + 2 regenerations = 3 attempts
    assert excinfo.value.attempts == 3
    assert backend.call_count == 3
    assert (
        excinfo.value.termination_reason == TerminationReason.CONSTRAINT_UNSATISFIED
    )
    # The trace records every non-conforming attempt and the termination reason.
    assert len(builder.trace.steps) == 3
    assert (
        builder.trace.termination_reason
        == TerminationReason.CONSTRAINT_UNSATISFIED
    )


def test_decode_zero_retries_makes_single_attempt():
    builder = ProofTraceBuilder()
    decoder, backend = _decoder(["nope"], output_format="json", retry_count=0)

    with pytest.raises(ConstraintUnsatisfied) as excinfo:
        decoder.decode(_context(), _state(), builder=builder)

    assert excinfo.value.attempts == 1
    assert backend.call_count == 1
    assert len(builder.trace.steps) == 1


def test_decode_succeeds_without_builder():
    decoder, _ = _decoder(['{"logic_form": "x=3"}'], output_format="json")
    candidate = decoder.decode(_context(), _state())
    assert candidate.structured["logic_form"] == "x=3"


def test_decoder_rejects_unknown_format_config():
    # SystemConfig is validated by the config manager, but guard the decoder directly.
    config = _config(output_format="json")
    config.output_format = "xml"
    backend = MockBackend(script=['{"logic_form": "x"}'])
    llm = LLMComponent(backend, _config())
    with pytest.raises(ValueError):
        ConstrainedDecoder(llm, config)
