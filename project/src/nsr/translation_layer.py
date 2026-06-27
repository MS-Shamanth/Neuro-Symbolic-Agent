"""Translation Layer: the bidirectional bridge between neural text and symbolic state.

The Translation_Layer mediates between System 1 (the LLM) and System 2 (the ACT-R
controller):

- **Forward** (:meth:`TranslationLayer.to_symbolic`): convert a structured
  :class:`~nsr.models.CandidateStep` into a machine-checkable
  :class:`~nsr.models.SymbolicRepresentation` before the controller is updated
  (Requirement 5.1).
- **Backward** (:meth:`TranslationLayer.to_context`): convert the working-memory state
  (Goal_Buffer, Imaginal_Buffer, and Declarative_Memory) into a
  :class:`~nsr.models.PromptContext` for the next LLM generation (Requirement 5.2).

This module implements forward/backward translation (Task 4.1) together with the
Task 4.2 extensions: untranslatable-step routing to repair (Requirement 5.3),
back-translation failure handling that returns an error record naming the
Translation_Layer (Requirement 5.5), and journaling of every translation outcome and
direction into the Proof_Trace (Requirement 5.4). The return type of
:meth:`to_symbolic` admits an :class:`~nsr.models.Untranslatable` outcome and
:meth:`to_context` raises :class:`~nsr.models.BackTranslationError` on failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from nsr.models import (
    BackTranslationError,
    CandidateStep,
    PromptContext,
    SymbolicRepresentation,
    Untranslatable,
    WorkingMemoryState,
)

if TYPE_CHECKING:  # only needed for type hints; avoids a hard import dependency
    from nsr.proof_trace import ProofTraceBuilder
    from nsr.models import ProofStep

# The key under which the constrained decoder places the machine-checkable encoding.
LOGIC_FORM_KEY = "logic_form"
# The key under which structured predicate fields are placed.
PREDICATES_KEY = "predicates"

# Translation direction labels journaled into the Proof_Trace (Requirement 5.4).
FORWARD = "forward"
BACKWARD = "backward"

# The component name recorded on a back-translation error record (Requirement 5.5).
TRANSLATION_LAYER_COMPONENT = "Translation_Layer"

# Forward translation may yield a Symbolic_Representation or an Untranslatable outcome.
TranslationResult = Union[SymbolicRepresentation, Untranslatable]


class TranslationLayer:
    """Bidirectional translator between candidate steps and symbolic working memory."""

    def to_symbolic(self, step: CandidateStep) -> TranslationResult:
        """Convert a structured candidate step into a Symbolic_Representation.

        The machine-checkable encoding is taken from the candidate's ``structured``
        payload (the ``logic_form`` field produced by constrained decoding). Structured
        predicate fields, when present, are carried over verbatim, and the original step
        text is preserved as the representation's source.

        Returns an :class:`~nsr.models.Untranslatable` outcome when the candidate
        carries no usable machine-checkable encoding, so the caller (the Repair process,
        Task 4.2) can act on it without an exception.
        """
        logic_form = step.structured.get(LOGIC_FORM_KEY)

        if not isinstance(logic_form, str) or not logic_form.strip():
            return Untranslatable(
                step=step,
                reason=(
                    f"candidate step has no non-empty '{LOGIC_FORM_KEY}' field to "
                    "convert into the machine-checkable encoding"
                ),
            )

        predicates = step.structured.get(PREDICATES_KEY, {})
        if not isinstance(predicates, dict):
            predicates = {}

        return SymbolicRepresentation(
            logic_form=logic_form,
            predicates=dict(predicates),
            source_text=step.raw_text,
        )

    def forward(
        self,
        step: CandidateStep,
        *,
        builder: Optional["ProofTraceBuilder"] = None,
        proof_step: Optional["ProofStep"] = None,
    ) -> TranslationResult:
        """Forward-translate a candidate step and journal the outcome (Req 5.1/5.3/5.4).

        Calls :meth:`to_symbolic` and, when a ``builder`` and ``proof_step`` are
        supplied, records the forward translation outcome -- including its direction and
        the untranslatable flag -- into the Proof_Trace (Requirement 5.4).

        When the step is untranslatable, the outcome is flagged in the trace and the
        :class:`~nsr.models.Untranslatable` result is returned unchanged so the caller
        can route the step to the repair process (Requirement 5.3). This method never
        touches the working-memory buffers, so an untranslatable step leaves them
        unchanged by construction.
        """
        result = self.to_symbolic(step)
        if builder is not None and proof_step is not None:
            self.record_forward_outcome(builder, proof_step, result)
        return result

    def to_context(
        self,
        state: WorkingMemoryState,
        *,
        builder: Optional["ProofTraceBuilder"] = None,
        proof_step: Optional["ProofStep"] = None,
    ) -> PromptContext:
        """Convert working-memory state into LLM prompt context for the next step.

        Pulls the active goal from the Goal_Buffer, the current active sub-goal (the
        first unsatisfied sub-goal, if any), the partial problem representation from the
        Imaginal_Buffer, and the accepted intermediate conclusions from
        Declarative_Memory, and renders them into a prompt the LLM can consume.

        When the symbolic state cannot be converted into context (Requirement 5.5), the
        failure is flagged and, when a ``builder`` and ``proof_step`` are supplied,
        journaled into the Proof_Trace together with an :class:`~nsr.models.ErrorRecord`
        naming the Translation_Layer; a :class:`~nsr.models.BackTranslationError`
        carrying that error record is then raised so the caller can surface a
        ``component-error`` termination while preserving the trace.
        """
        failure_reason = self._back_translation_failure_reason(state)
        if failure_reason is not None:
            raise self._fail_back_translation(failure_reason, builder, proof_step)

        goal = state.goal_buffer
        active_sub_goal = self._active_sub_goal(state)

        partial_representation = (
            state.imaginal_buffer.logic_form
            if state.imaginal_buffer is not None
            else None
        )

        established_conclusions = [
            rep.logic_form for rep in state.declarative_memory
        ]

        prompt_text = self._render_prompt(
            goal_description=goal.description,
            active_sub_goal=active_sub_goal,
            partial_representation=partial_representation,
            established_conclusions=established_conclusions,
        )

        if builder is not None and proof_step is not None:
            self.record_backward_outcome(builder, proof_step, success=True)

        return PromptContext(
            goal_description=goal.description,
            active_sub_goal=active_sub_goal,
            partial_representation=partial_representation,
            established_conclusions=established_conclusions,
            prompt_text=prompt_text,
        )

    # -- Proof_Trace journaling (Requirement 5.4) --------------------------------

    @staticmethod
    def record_forward_outcome(
        builder: "ProofTraceBuilder",
        proof_step: "ProofStep",
        result: TranslationResult,
    ) -> dict[str, Any]:
        """Journal a forward translation outcome into the Proof_Trace (Req 5.4).

        Records the translation direction and the untranslatable flag; on a successful
        translation the resulting logic form is included, and on an untranslatable
        outcome the failure reason is included. Returns the recorded outcome dict.
        """
        untranslatable = isinstance(result, Untranslatable)
        outcome: dict[str, Any] = {
            "direction": FORWARD,
            "untranslatable": untranslatable,
        }
        if untranslatable:
            outcome["reason"] = result.reason
        else:
            outcome["logic_form"] = result.logic_form
        builder.add_translation_outcome(proof_step, outcome)
        return outcome

    @staticmethod
    def record_backward_outcome(
        builder: "ProofTraceBuilder",
        proof_step: "ProofStep",
        *,
        success: bool,
        reason: Optional[str] = None,
    ) -> dict[str, Any]:
        """Journal a backward translation outcome into the Proof_Trace (Req 5.4/5.5).

        Records the translation direction and a ``failed`` flag; on failure the reason
        is included. Returns the recorded outcome dict.
        """
        outcome: dict[str, Any] = {
            "direction": BACKWARD,
            "failed": not success,
        }
        if not success and reason is not None:
            outcome["reason"] = reason
        builder.add_translation_outcome(proof_step, outcome)
        return outcome

    # -- back-translation failure helpers (Requirement 5.5) ----------------------

    @staticmethod
    def _back_translation_failure_reason(
        state: WorkingMemoryState,
    ) -> Optional[str]:
        """Return why ``state`` cannot be back-translated, or ``None`` when it can.

        The symbolic state is unusable as prompt context when it carries no
        Goal_Buffer, or when the active goal has no non-empty description, since the
        LLM cannot be given a coherent goal to reason toward.
        """
        if state is None:
            return "working-memory state is missing"
        goal = state.goal_buffer
        if goal is None:
            return "working-memory state has no Goal_Buffer to back-translate"
        if not isinstance(goal.description, str) or not goal.description.strip():
            return "Goal_Buffer holds no non-empty active goal to back-translate"
        return None

    def _fail_back_translation(
        self,
        reason: str,
        builder: Optional["ProofTraceBuilder"],
        proof_step: Optional["ProofStep"],
    ) -> BackTranslationError:
        """Flag/journal a back-translation failure and build the error to raise.

        Records the failed backward outcome on ``proof_step`` (when supplied) and an
        :class:`~nsr.models.ErrorRecord` naming the Translation_Layer on the trace,
        then returns a :class:`~nsr.models.BackTranslationError` carrying that record.
        """
        from nsr.models import ErrorRecord  # local import avoids an import cycle

        if builder is not None:
            if proof_step is not None:
                self.record_backward_outcome(
                    builder, proof_step, success=False, reason=reason
                )
            error_record = builder.set_error_record(
                TRANSLATION_LAYER_COMPONENT, reason
            )
        else:
            error_record = ErrorRecord(
                failed_component=TRANSLATION_LAYER_COMPONENT, reason=reason
            )
        return BackTranslationError(reason, error_record)

    @staticmethod
    def _active_sub_goal(state: WorkingMemoryState) -> Union[str, None]:
        """Return the description of the first unsatisfied sub-goal, or ``None``."""
        for sub_goal in state.goal_buffer.sub_goals:
            if not sub_goal.satisfied:
                return sub_goal.description
        return None

    @staticmethod
    def _render_prompt(
        *,
        goal_description: str,
        active_sub_goal: Union[str, None],
        partial_representation: Union[str, None],
        established_conclusions: list[str],
    ) -> str:
        """Render a deterministic, human-readable prompt from the symbolic state."""
        lines: list[str] = [f"Goal: {goal_description}"]

        if active_sub_goal is not None:
            lines.append(f"Current sub-goal: {active_sub_goal}")

        if established_conclusions:
            lines.append("Established conclusions:")
            lines.extend(f"- {conclusion}" for conclusion in established_conclusions)
        else:
            lines.append("Established conclusions: (none)")

        if partial_representation is not None:
            lines.append(f"Partial representation: {partial_representation}")

        lines.append("Produce the next reasoning step.")
        return "\n".join(lines)
