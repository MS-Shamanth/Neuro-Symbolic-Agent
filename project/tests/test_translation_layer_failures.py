"""Tests for Translation Layer untranslatable and back-translation failure handling.

Covers Task 4.2:

- Untranslatable forward steps are flagged, routed to repair (returned as an
  ``Untranslatable`` outcome), and leave the working-memory buffers unchanged (Req 5.3).
- Back-translation failures are flagged, journaled, and surfaced as a
  ``BackTranslationError`` carrying an ``ErrorRecord`` naming the Translation_Layer
  (Req 5.5).
- Every translation outcome -- with its direction and untranslatable/failed flag -- is
  recorded into the Proof_Trace (Req 5.4).

The named buffer-invariance property (Task 4.3) is intentionally out of scope here.
"""

from __future__ import annotations

import copy

import pytest

from nsr.models import (
    BackTranslationError,
    CandidateStep,
    ErrorRecord,
    Goal,
    PromptContext,
    SubGoal,
    SymbolicRepresentation,
    Untranslatable,
    ValidationStatus,
    WorkingMemoryState,
)
from nsr.proof_trace import ProofTraceBuilder
from nsr.translation_layer import (
    BACKWARD,
    FORWARD,
    TRANSLATION_LAYER_COMPONENT,
    TranslationLayer,
)


# --------------------------------------------------------------------------- #
# Forward: untranslatable handling and journaling (Req 5.3, 5.4)
# --------------------------------------------------------------------------- #


def test_forward_records_successful_outcome_in_proof_trace():
    layer = TranslationLayer()
    builder = ProofTraceBuilder()
    proof_step = builder.append_step("s", status=ValidationStatus.ACCEPTED)
    step = CandidateStep(raw_text="x", structured={"logic_form": "p(x)"})

    result = layer.forward(step, builder=builder, proof_step=proof_step)

    assert isinstance(result, SymbolicRepresentation)
    assert proof_step.translation_outcomes == [
        {"direction": FORWARD, "untranslatable": False, "logic_form": "p(x)"}
    ]


def test_forward_untranslatable_is_flagged_and_routed_to_repair():
    layer = TranslationLayer()
    builder = ProofTraceBuilder()
    proof_step = builder.append_step("s", status=ValidationStatus.REJECTED)
    step = CandidateStep(raw_text="free prose", structured={})

    result = layer.forward(step, builder=builder, proof_step=proof_step)

    # The Untranslatable outcome is what the caller routes to the repair process.
    assert isinstance(result, Untranslatable)
    assert result.step is step
    # The outcome is flagged with the untranslatable flag and reason (Req 5.4).
    assert len(proof_step.translation_outcomes) == 1
    recorded = proof_step.translation_outcomes[0]
    assert recorded["direction"] == FORWARD
    assert recorded["untranslatable"] is True
    assert recorded["reason"]


def test_forward_untranslatable_leaves_working_memory_unchanged():
    layer = TranslationLayer()
    state = WorkingMemoryState(
        goal_buffer=Goal(description="g", sub_goals=[SubGoal(description="a")]),
        declarative_memory=[SymbolicRepresentation(logic_form="eq(x,1)")],
        imaginal_buffer=SymbolicRepresentation(logic_form="eq(x,1)"),
    )
    before = copy.deepcopy(state)
    step = CandidateStep(raw_text="prose", structured={"logic_form": "  "})

    result = layer.forward(step)

    assert isinstance(result, Untranslatable)
    # Buffers are untouched: the translation layer never mutates working memory.
    assert state == before


def test_forward_without_builder_does_not_raise():
    layer = TranslationLayer()
    step = CandidateStep(raw_text="x", structured={"logic_form": "p(x)"})

    # Journaling is optional; omitting the builder simply skips recording.
    result = layer.forward(step)

    assert isinstance(result, SymbolicRepresentation)


# --------------------------------------------------------------------------- #
# Backward: back-translation failure handling (Req 5.5) and journaling (Req 5.4)
# --------------------------------------------------------------------------- #


def test_to_context_records_successful_backward_outcome():
    layer = TranslationLayer()
    builder = ProofTraceBuilder()
    proof_step = builder.append_step("s", status=ValidationStatus.ACCEPTED)
    state = WorkingMemoryState(goal_buffer=Goal(description="solve"))

    ctx = layer.to_context(state, builder=builder, proof_step=proof_step)

    assert isinstance(ctx, PromptContext)
    assert proof_step.translation_outcomes == [
        {"direction": BACKWARD, "failed": False}
    ]


def test_to_context_raises_back_translation_error_for_blank_goal():
    layer = TranslationLayer()
    state = WorkingMemoryState(goal_buffer=Goal(description="   "))

    with pytest.raises(BackTranslationError) as exc_info:
        layer.to_context(state)

    err = exc_info.value
    # The error carries an ErrorRecord naming the Translation_Layer (Req 5.5).
    assert isinstance(err.error_record, ErrorRecord)
    assert err.error_record.failed_component == TRANSLATION_LAYER_COMPONENT
    assert err.error_record.reason


def test_to_context_failure_is_journaled_into_proof_trace():
    layer = TranslationLayer()
    builder = ProofTraceBuilder()
    proof_step = builder.append_step("s", status=ValidationStatus.ACCEPTED)
    state = WorkingMemoryState(goal_buffer=Goal(description=""))

    with pytest.raises(BackTranslationError):
        layer.to_context(state, builder=builder, proof_step=proof_step)

    # The failed backward outcome is journaled on the step (Req 5.4).
    assert len(proof_step.translation_outcomes) == 1
    recorded = proof_step.translation_outcomes[0]
    assert recorded["direction"] == BACKWARD
    assert recorded["failed"] is True
    assert recorded["reason"]

    # The error record is attached to the trace, preserving its contents (Req 5.5).
    assert builder.trace.error_record is not None
    assert builder.trace.error_record.failed_component == TRANSLATION_LAYER_COMPONENT


def test_to_context_failure_without_builder_still_names_translation_layer():
    layer = TranslationLayer()
    state = WorkingMemoryState(goal_buffer=Goal(description="\t\n"))

    with pytest.raises(BackTranslationError) as exc_info:
        layer.to_context(state)

    assert exc_info.value.error_record.failed_component == TRANSLATION_LAYER_COMPONENT


def test_to_context_succeeds_for_usable_state_without_journaling():
    layer = TranslationLayer()
    state = WorkingMemoryState(goal_buffer=Goal(description="g"))

    # No builder/proof_step: success path still returns a PromptContext.
    ctx = layer.to_context(state)

    assert isinstance(ctx, PromptContext)
    assert ctx.goal_description == "g"
