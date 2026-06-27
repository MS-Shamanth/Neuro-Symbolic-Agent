"""Repair Coordinator (Task 6.2).

This module implements the shared repair sub-loop described in the design's *Repair
Coordinator* section. It is the single component that drives correction for the three
repair-triggering outcomes that arise during the dual-process reasoning cycle:

- **rejection** -- a :class:`~nsr.validation_engine.ValidationOutcome` whose status is
  ``REJECTED`` because one or more applicable production rules were violated (Req 6.3,
  6.4);
- **untranslatable** -- a forward-translation outcome
  (:class:`~nsr.models.Untranslatable`) where a candidate step could not be converted
  into the machine-checkable encoding (Req 5.3); and
- **no-rule-matched** -- a controller outcome
  (:class:`~nsr.actr_controller.NoRuleMatched`) where no production rule applied to the
  current working-memory state (Req 4.8).

For each trigger the coordinator runs the same bounded sub-loop (Req 6.4-6.6):

1. Build a repair prompt that references the *offending constraints* -- the violated
   production rules, the untranslatable reason, or the no-rule-matched reason -- on top
   of the symbolic-state context.
2. Request a regenerated step from the LLM component (System 1).
3. Re-translate the regenerated step through the Translation_Layer (Req 6.5).
4. Re-validate the re-translated step against every applicable production rule through
   the Validation_Engine (Req 6.5).
5. Record the attempt into the Proof_Trace via
   :meth:`~nsr.proof_trace.ProofTraceBuilder.record_repair_attempt` -- the rejected
   step, the violated rule ids, and the resulting repaired step (Req 8.3).

The repair attempt count is incremented on every iteration up to the configured
``repair_attempt_limit``; if the limit is reached without an accepted step, the
coordinator signals a :attr:`~nsr.models.TerminationReason.REPAIR_EXHAUSTED`
termination on the trace and returns it so the orchestrator (Task 10) can surface it
(Req 6.6). The number of recorded repair attempts never exceeds the configured limit,
which is the guarantee exercised by Property 8 (Task 6.3).

The coordinator owns no per-query mutable state of its own: a single instance can drive
repair for any number of steps. Component failures raised by the LLM
(:class:`~nsr.llm_component.LLMError`) and back-translation failures are intentionally
*not* swallowed here -- they are component errors the orchestrator converts into a
``component-error`` termination while preserving the trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

from .models import (
    CandidateStep,
    ProductionRule,
    ProofStep,
    SymbolicRepresentation,
    TerminationReason,
    Untranslatable,
    WorkingMemoryState,
)

if TYPE_CHECKING:  # imported lazily / only for type hints to avoid hard coupling
    from .llm_component import LLMComponent, OutputSchema
    from .models.trace import RepairAttempt
    from .proof_trace import ProofTraceBuilder
    from .translation_layer import TranslationLayer
    from .validation_engine import ValidationEngine


class RepairTrigger(str, Enum):
    """The outcome that initiated a repair sub-loop.

    The trigger determines how the repair prompt frames the *offending constraints* the
    regenerated step must satisfy.
    """

    REJECTION = "rejection"
    UNTRANSLATABLE = "untranslatable"
    NO_RULE_MATCHED = "no-rule-matched"


@dataclass
class RepairContext:
    """The inputs the orchestrator hands to :meth:`RepairCoordinator.repair`.

    Attributes:
        trigger: Which of the three repair-triggering outcomes initiated the loop.
        state: The current working-memory state, used both to build prompt context for
            regeneration and to supply the applicable production rules for re-validation.
        proof_step: The Proof_Trace step being repaired; each attempt is recorded against
            it via :meth:`ProofTraceBuilder.record_repair_attempt` (Req 8.3).
        rejected_representation: The symbolic representation that was rejected, when one
            exists (the ``REJECTION`` trigger). For the untranslatable and
            no-rule-matched triggers a placeholder is synthesised from ``reason``.
        violated_rules: The production rules the rejected step violated, used to
            reference the offending constraints in the repair prompt (Req 6.4). Empty for
            the untranslatable and no-rule-matched triggers.
        reason: A human-readable explanation of the offending outcome (the untranslatable
            reason or the no-rule-matched reason), referenced in the repair prompt.
    """

    trigger: RepairTrigger
    state: WorkingMemoryState
    proof_step: ProofStep
    rejected_representation: Optional[SymbolicRepresentation] = None
    violated_rules: list[ProductionRule] = field(default_factory=list)
    reason: str = ""


@dataclass
class RepairOutcome:
    """The result of driving a repair sub-loop.

    Attributes:
        succeeded: ``True`` when a regenerated step was accepted within the attempt
            limit; ``False`` when the limit was reached without acceptance.
        accepted_representation: The accepted :class:`SymbolicRepresentation` when
            ``succeeded`` is ``True``; otherwise ``None``.
        attempts_used: The number of repair attempts recorded into the Proof_Trace. This
            never exceeds the configured ``repair_attempt_limit`` (Property 8).
        termination_reason: :attr:`TerminationReason.REPAIR_EXHAUSTED` when the limit was
            reached without acceptance (Req 6.6); ``None`` on success.
        attempts: The :class:`RepairAttempt` records appended to the step, in order.
    """

    succeeded: bool
    accepted_representation: Optional[SymbolicRepresentation]
    attempts_used: int
    termination_reason: Optional[TerminationReason]
    attempts: list["RepairAttempt"] = field(default_factory=list)


class RepairCoordinator:
    """Drives the shared repair sub-loop for the three repair-triggering outcomes.

    The coordinator wires together the LLM component (regeneration), the
    Translation_Layer (re-translation), and the Validation_Engine (re-validation), and
    enforces the configured repair attempt limit. It is stateless across calls, so one
    instance can serve every step of every query.
    """

    def __init__(
        self,
        llm: "LLMComponent",
        translation: "TranslationLayer",
        validation: "ValidationEngine",
        repair_attempt_limit: int,
        *,
        constraint: Optional["OutputSchema"] = None,
    ) -> None:
        """Create a coordinator.

        Args:
            llm: The System 1 generator used to regenerate a candidate step.
            translation: The Translation_Layer used to re-translate a regenerated step.
            validation: The Validation_Engine used to re-validate a re-translated step.
            repair_attempt_limit: The maximum number of repair attempts (Req 6.4, 6.6).
                Must be a non-negative integer (documented range 0..1000); a limit of 0
                permits no repair attempts and yields an immediate
                ``repair-exhausted`` outcome.
            constraint: Optional output schema forwarded to the LLM on regeneration;
                defaults to the component's configured schema.

        Raises:
            ValueError: If ``repair_attempt_limit`` is negative.
        """
        if repair_attempt_limit < 0:
            raise ValueError("repair_attempt_limit must be non-negative")
        self._llm = llm
        self._translation = translation
        self._validation = validation
        self._repair_attempt_limit = repair_attempt_limit
        self._constraint = constraint

    @property
    def repair_attempt_limit(self) -> int:
        """The configured maximum number of repair attempts (Req 6.4, 6.6)."""
        return self._repair_attempt_limit

    def repair(
        self,
        context: RepairContext,
        *,
        builder: "ProofTraceBuilder",
    ) -> RepairOutcome:
        """Drive the repair sub-loop for ``context`` and return its outcome.

        Runs up to ``repair_attempt_limit`` iterations. Each iteration builds a repair
        prompt referencing the current offending constraints, regenerates a step,
        re-translates it (Req 6.5), re-validates it against every applicable production
        rule (Req 6.5), and records the attempt into the Proof_Trace (Req 8.3). The first
        regenerated step that translates and validates as accepted ends the loop
        successfully. If the limit is reached without acceptance, a ``repair-exhausted``
        termination is recorded on the trace and returned (Req 6.6).

        Args:
            context: The triggering outcome and the working-memory state to repair from.
            builder: The Proof_Trace builder the attempts are journaled into.

        Returns:
            A :class:`RepairOutcome` describing success/exhaustion, the accepted
            representation (when any), the number of attempts used, and the recorded
            attempts.
        """
        proof_step = context.proof_step
        applicable_rules = list(context.state.procedural_memory)

        # Mutable "offending" state, refreshed each iteration. It begins from the
        # triggering outcome and is updated whenever an attempt is itself rejected or
        # untranslatable, so the next prompt always references the latest constraints.
        trigger = context.trigger
        rejected_rep = self._initial_rejected_rep(context)
        violated_rules: list[ProductionRule] = list(context.violated_rules)
        violated_rule_ids = [rule.rule_id for rule in violated_rules]
        reason = context.reason or self._default_reason(context.trigger)

        attempts: list["RepairAttempt"] = []

        for _ in range(self._repair_attempt_limit):
            prompt_context = self._translation.to_context(context.state)
            prompt_context.prompt_text = self._augment_prompt(
                prompt_context.prompt_text,
                trigger=trigger,
                violated_rules=violated_rules,
                reason=reason,
            )

            candidate = self._llm.generate_step(
                prompt_context, self._constraint, trace=builder
            )

            translated = self._translation.to_symbolic(candidate)

            if isinstance(translated, Untranslatable):
                # The regenerated step still cannot be translated; record the attempt
                # with no repaired step and carry the untranslatable outcome forward as
                # the offending constraint for the next iteration (Req 5.3, 8.3).
                attempt = builder.record_repair_attempt(
                    proof_step,
                    rejected_step=rejected_rep,
                    violated_rule_ids=violated_rule_ids,
                    repaired_step=None,
                )
                attempts.append(attempt)

                trigger = RepairTrigger.UNTRANSLATABLE
                rejected_rep = self._rep_from_candidate(candidate)
                violated_rules = []
                violated_rule_ids = []
                reason = translated.reason
                continue

            # Re-validate the re-translated step against every applicable rule (Req 6.5).
            outcome = self._validation.validate(translated, applicable_rules)
            attempt = builder.record_repair_attempt(
                proof_step,
                rejected_step=rejected_rep,
                violated_rule_ids=violated_rule_ids,
                repaired_step=translated,
            )
            attempts.append(attempt)

            if outcome.accepted:
                return RepairOutcome(
                    succeeded=True,
                    accepted_representation=translated,
                    attempts_used=len(attempts),
                    termination_reason=None,
                    attempts=attempts,
                )

            # Still rejected: the regenerated step becomes the offending step whose
            # violated rules constrain the next regeneration (Req 6.4).
            trigger = RepairTrigger.REJECTION
            rejected_rep = translated
            violated_rules = list(outcome.violated_rules)
            violated_rule_ids = list(outcome.violated_rule_ids)
            reason = self._default_reason(RepairTrigger.REJECTION)

        # Req 6.6: the repair attempt limit was reached without an accepted step.
        builder.set_termination_reason(TerminationReason.REPAIR_EXHAUSTED)
        return RepairOutcome(
            succeeded=False,
            accepted_representation=None,
            attempts_used=len(attempts),
            termination_reason=TerminationReason.REPAIR_EXHAUSTED,
            attempts=attempts,
        )

    # ----------------------------------------------------- repair-prompt helpers

    @staticmethod
    def _augment_prompt(
        base_prompt: str,
        *,
        trigger: RepairTrigger,
        violated_rules: list[ProductionRule],
        reason: str,
    ) -> str:
        """Append repair directives referencing the offending constraints (Req 6.4).

        Builds on the backward-translated state prompt so the regenerated step keeps the
        full reasoning context, then names the specific constraints the previous step
        failed: the violated production rules (rejection), the untranslatable reason, or
        the no-rule-matched reason.
        """
        lines = [base_prompt, "", "REPAIR REQUIRED:"]
        lines.append(
            "The previous reasoning step was not accepted and must be regenerated."
        )

        if trigger == RepairTrigger.REJECTION:
            if violated_rules:
                lines.append(
                    "It violated the following production rule(s). Regenerate a step "
                    "that satisfies every one of them:"
                )
                for rule in violated_rules:
                    lines.append(
                        f"- {rule.rule_id}: IF {rule.condition} THEN {rule.action}"
                    )
            else:
                lines.append(
                    "It violated one or more production rules. Regenerate a satisfying "
                    "step."
                )
        elif trigger == RepairTrigger.UNTRANSLATABLE:
            lines.append(
                "It could not be translated into the required machine-checkable form: "
                f"{reason}"
            )
            lines.append(
                "Regenerate a step that conforms to the required structured format."
            )
        else:  # RepairTrigger.NO_RULE_MATCHED
            lines.append(f"No production rule applied to it: {reason}")
            lines.append(
                "Regenerate a step that matches an available production rule."
            )

        return "\n".join(lines)

    @staticmethod
    def _default_reason(trigger: RepairTrigger) -> str:
        """A fallback offending-constraint reason when the caller supplies none."""
        if trigger == RepairTrigger.REJECTION:
            return "the step violated one or more applicable production rules"
        if trigger == RepairTrigger.UNTRANSLATABLE:
            return "the step could not be converted into the machine-checkable encoding"
        return "no production rule matched the current working-memory state"

    @staticmethod
    def _initial_rejected_rep(context: RepairContext) -> SymbolicRepresentation:
        """The representation recorded as the first attempt's rejected step (Req 8.3).

        Uses the explicit rejected representation when supplied (the rejection trigger);
        otherwise synthesises a placeholder from the offending reason so the
        untranslatable and no-rule-matched triggers still record a coherent rejected
        step.
        """
        if context.rejected_representation is not None:
            return context.rejected_representation
        return SymbolicRepresentation(
            logic_form="",
            predicates={},
            source_text=context.reason
            or RepairCoordinator._default_reason(context.trigger),
        )

    @staticmethod
    def _rep_from_candidate(candidate: CandidateStep) -> SymbolicRepresentation:
        """Build a placeholder representation for an untranslatable regenerated step.

        An untranslatable step has no machine-checkable encoding, so the recorded
        rejected step carries an empty ``logic_form`` and preserves the candidate's raw
        text as its source.
        """
        return SymbolicRepresentation(
            logic_form="",
            predicates={},
            source_text=candidate.raw_text,
        )
