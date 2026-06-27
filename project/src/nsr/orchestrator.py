"""Pipeline Orchestrator: query intake and the four-stage reasoning cycle.

This module implements the *Pipeline Orchestrator* described in the design's
*Pipeline Orchestrator* section and *Dual-Process Reasoning Cycle*. The orchestrator
owns the reasoning cycle: it validates and accepts a query, initializes the Goal_Buffer
before any Reasoning_Step is generated, and drives a bounded loop in which each cycle
runs four stages **in a fixed order** -- LLM generation, translation, ACT-R Controller
update, and validation (Req 1.2).

Task 10.1 scope (query intake + cycle execution) is extended here by Task 10.2
(termination, output emission, and error handling):

- **Intake** -- validate the query and initialize the Goal_Buffer before any step;
  reject empty or unparseable queries with an :class:`~nsr.models.ErrorRecord` *before*
  starting the cycle, without initializing the reasoning cycle (Req 1.1, 1.7).
- **Cycle execution** -- run cycles in the fixed four-stage order, bounded by the
  configured maximum cycle limit (Req 1.2).
- **Termination + output** -- emit a :class:`~nsr.models.VerifiedOutput` with the
  attached Faithfulness_Score on goal satisfaction (Req 1.3, 7.6); terminate with
  ``cycle-limit-reached`` at the cycle bound (Req 1.4); surface ``constraint-unsatisfied``
  from the decoder (Req 3.4) and ``repair-exhausted`` from the Repair Coordinator
  (Req 6.6).
- **Error handling** -- drive the Repair Coordinator on repair-triggering outcomes
  (rejection, untranslatable, no-rule-matched), and convert component failures (LLM
  unavailable/timeout, back-translation failure) into ``component-error`` error records
  while preserving the Proof_Trace (Req 1.6).
- **Journaling** -- record each step, its outcome, and the applied rule into the
  Proof_Trace (Req 1.5).

``PipelineResult`` is ``VerifiedOutput | ErrorRecord``, and in every case the produced
:class:`~nsr.models.ProofTrace` is carried: a :class:`VerifiedOutput` embeds it, and a
returned :class:`ErrorRecord` is also attached to the builder's trace so the caller can
recover it (the orchestrator exposes the last built trace via :attr:`last_trace`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional, Union

from .actr_controller import ACTRController, NoRuleMatched
from .constrained_decoder import ConstrainedDecoder, ConstraintUnsatisfied
from .llm_component import LLMComponent, LLMError
from .metrics_engine import compute_faithfulness_score
from .models import (
    BackTranslationError,
    ErrorRecord,
    Goal,
    ProductionRule,
    ProofStep,
    ProofTrace,
    RuleOrigin,
    SubGoal,
    SymbolicRepresentation,
    SystemConfig,
    TerminationReason,
    Untranslatable,
    ValidationStatus,
    VerifiedOutput,
    WorkingMemoryState,
)
from .proof_trace import ProofTraceBuilder
from .repair_coordinator import (
    RepairContext,
    RepairCoordinator,
    RepairOutcome,
    RepairTrigger,
)
from .translation_layer import TranslationLayer
from .validation_engine import ValidationEngine

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .rule_learner import RuleLearner

logger = logging.getLogger(__name__)

#: The component named on the error record when the LLM fails and no record was attached.
LLM_COMPONENT = "LLM"

#: The component named on the error record when intake rejects an invalid query (Req 1.7).
PIPELINE_COMPONENT = "Pipeline"

#: ``run`` returns a VerifiedOutput on success or an ErrorRecord on rejection/failure.
PipelineResult = Union[VerifiedOutput, ErrorRecord]


class CycleStage(str, Enum):
    """The four stages executed, in this fixed order, by every reasoning cycle (Req 1.2)."""

    GENERATE = "generate"
    TRANSLATE = "translate"
    CONTROLLER_UPDATE = "controller-update"
    VALIDATE = "validate"


#: The canonical, fixed order of the four stages within a single cycle (Req 1.2).
STAGE_ORDER: tuple[CycleStage, ...] = (
    CycleStage.GENERATE,
    CycleStage.TRANSLATE,
    CycleStage.CONTROLLER_UPDATE,
    CycleStage.VALIDATE,
)


def parse_query(query: object) -> Optional[Goal]:
    """Parse a raw query string into a :class:`Goal` with ordered sub-goals (Req 1.1).

    The parser is intentionally simple (a fuller NLU pass is out of scope for this
    task): the query text becomes the goal description, and the query is split on
    sentence/clause delimiters (``.``, ``;``, ``\\n``, and the connectives ``then`` and
    ``and``) into ordered :class:`SubGoal` entries. A query with a single clause yields
    one sub-goal equal to the whole query.

    Returns ``None`` when the query cannot be parsed into a query goal -- it is not a
    string, or it is empty/whitespace-only -- so the orchestrator can reject it with an
    error record before initializing the reasoning cycle (Req 1.7).
    """
    if not isinstance(query, str):
        return None
    text = query.strip()
    if not text:
        return None

    sub_goal_texts = _split_sub_goals(text)
    sub_goals = [SubGoal(description=part) for part in sub_goal_texts]
    return Goal(description=text, sub_goals=sub_goals)


def _split_sub_goals(text: str) -> list[str]:
    """Split a non-empty query into ordered, non-empty sub-goal clauses."""
    import re

    # Split on sentence/clause delimiters and the connectives "then" / "and".
    parts = re.split(r"[.;\n]+|\bthen\b|\band\b", text, flags=re.IGNORECASE)
    clauses = [part.strip() for part in parts if part.strip()]
    # A query with no usable delimiters becomes a single sub-goal equal to the query.
    return clauses if clauses else [text]


@dataclass
class CycleOutcome:
    """The result of executing one reasoning cycle.

    Attributes:
        proof_step: The Proof_Trace step recorded for the cycle, when one was appended.
        accepted: ``True`` when the cycle's step was validated as accepted.
        goal_satisfied: ``True`` when accepting the step satisfied the active goal.
        needs_repair: ``True`` when the cycle produced a repair-triggering outcome
            (untranslatable, no-rule-matched, or rejection); the orchestrator drives the
            Repair Coordinator on these outcomes (Req 6.4-6.6).
        repair_trigger: Which repair-triggering outcome occurred, used to frame the
            repair prompt's offending constraints.
        state: The working-memory state snapshot the repair sub-loop regenerates from.
        rejected_representation: The rejected Symbolic_Representation handed to repair,
            when one exists (``None`` for an untranslatable step).
        violated_rules: The production rules the rejected step violated (rejection only).
        repair_reason: A human-readable explanation of the offending outcome.
    """

    proof_step: Optional[ProofStep] = None
    accepted: bool = False
    goal_satisfied: bool = False
    needs_repair: bool = False
    repair_trigger: Optional[RepairTrigger] = None
    state: Optional[WorkingMemoryState] = None
    rejected_representation: Optional[SymbolicRepresentation] = None
    violated_rules: list[ProductionRule] = field(default_factory=list)
    repair_reason: str = ""


class PipelineOrchestrator:
    """Drives query intake and the bounded four-stage reasoning cycle (Req 1.1, 1.2, 1.7).

    The orchestrator wires the existing components together: backward translation to
    build the LLM prompt context, the constrained decoder/LLM to generate one candidate
    step, forward translation into a symbolic representation, the ACT-R controller for
    rule selection and accepted-step integration, and the validation engine for the
    accept/reject decision. Every step is journaled into a :class:`ProofTraceBuilder`.

    Construction takes the assembled components plus the :class:`SystemConfig`. When a
    :class:`~nsr.repair_coordinator.RepairCoordinator` is supplied, repair-triggering
    outcomes (rejection, untranslatable, no-rule-matched) drive the shared repair
    sub-loop, and exhaustion surfaces a ``repair-exhausted`` termination (Req 6.4-6.6);
    when no coordinator is supplied, such outcomes simply advance the cycle.
    """

    def __init__(
        self,
        *,
        llm: LLMComponent,
        translation: TranslationLayer,
        decoder: ConstrainedDecoder,
        controller: ACTRController,
        validation: ValidationEngine,
        config: SystemConfig,
        repair: Optional[RepairCoordinator] = None,
        procedural_memory: Optional[list[ProductionRule]] = None,
        rule_learner: "Optional[RuleLearner]" = None,
        query_parser: Callable[[object], Optional[Goal]] = parse_query,
        on_stage: Optional[Callable[[int, CycleStage], None]] = None,
    ) -> None:
        if config.max_cycle_limit < 1:
            raise ValueError("max_cycle_limit must be at least 1")
        self._llm = llm
        self._translation = translation
        self._decoder = decoder
        self._controller = controller
        self._validation = validation
        self._config = config
        self._repair = repair
        self._procedural_memory = list(procedural_memory) if procedural_memory else []
        self._rule_learner = rule_learner
        self._parse_query = query_parser
        self._on_stage = on_stage

        # Ids of rules promoted into Procedural_Memory by the Rule Learner during this
        # run, so accepted steps applying them can be marked LEARNED rather than SEEDED
        # in the Proof_Trace (Req 14.5). Every other applied rule is a Seeded_Rule.
        self._learned_rule_ids: set[str] = set()
        # Monotonic per-run query counter used to derive a stable induction trace id for
        # each goal-satisfied trace handed to the Rule Learner (Req 14.2, 14.6).
        self._query_counter = 0

        # Observable execution log of (cycle_index, stage) tuples for the most recent
        # run, so the fixed four-stage ordering (Req 1.2) is directly inspectable.
        self._stage_log: list[tuple[int, CycleStage]] = []
        self._completed_cycles = 0
        self._last_trace: Optional[ProofTrace] = None

    # ------------------------------------------------------------------ accessors

    @property
    def stage_log(self) -> list[tuple[int, CycleStage]]:
        """The ``(cycle_index, stage)`` execution log from the most recent run."""
        return list(self._stage_log)

    @property
    def completed_cycles(self) -> int:
        """The number of cycles completed during the most recent run."""
        return self._completed_cycles

    @property
    def last_trace(self) -> Optional[ProofTrace]:
        """The Proof_Trace produced by the most recent run (carries any error record)."""
        return self._last_trace

    @property
    def procedural_memory(self) -> list[ProductionRule]:
        """The Procedural_Memory seeded into each query, including promoted Learned_Rules.

        Returns a copy. After a goal-satisfied run with rule learning enabled, any
        promoted ``Learned_Rule`` has been appended here so the next query is initialized
        with it (Req 14.1, 14.3); with rule learning disabled this holds only the
        originally supplied Seeded_Rules (Req 14.10).
        """
        return list(self._procedural_memory)

    @property
    def learned_rule_ids(self) -> set[str]:
        """Ids of rules promoted into Procedural_Memory during this run (Req 14.5)."""
        return set(self._learned_rule_ids)

    # ----------------------------------------------------------------------- run

    def run(self, query: object) -> PipelineResult:
        """Process ``query`` through intake and the bounded four-stage cycle.

        Intake validates and parses the query and initializes the Goal_Buffer *before*
        any Reasoning_Step is generated (Req 1.1). An empty or unparseable query is
        rejected with an :class:`ErrorRecord` *without* initializing the reasoning cycle
        (Req 1.7). Otherwise the cycle runs in the fixed four-stage order bounded by the
        configured maximum cycle limit (Req 1.2).

        Returns a :class:`VerifiedOutput` on goal satisfaction or an :class:`ErrorRecord`
        on rejection; in both cases the produced Proof_Trace is available via
        :attr:`last_trace`.
        """
        builder = ProofTraceBuilder(latency_budget_ms=self._config.latency_budget_ms)
        self._stage_log = []
        self._completed_cycles = 0
        self._last_trace = builder.trace

        # --- Intake: validate + parse before initializing the cycle (Req 1.7) -------
        goal = self._parse_query(query)
        if goal is None:
            return self._reject_invalid_query(query, builder)

        # --- Initialize the Goal_Buffer before generating any step (Req 1.1) --------
        self._controller.initialize(goal, self._procedural_memory)

        # --- Bounded four-stage reasoning cycle (Req 1.2) ---------------------------
        return self._run_cycles(builder)

    # -------------------------------------------------------------------- intake

    def _reject_invalid_query(
        self, query: object, builder: ProofTraceBuilder
    ) -> ErrorRecord:
        """Reject an empty/unparseable query with an error record (Req 1.7).

        The reasoning cycle is never initialized: the controller is left untouched and
        no Goal_Buffer is created. The error record names the Pipeline and is attached
        to the (empty) Proof_Trace so the caller can recover it via :attr:`last_trace`.
        """
        if not isinstance(query, str):
            reason = (
                "submitted query is not a string and cannot be parsed into a query goal"
            )
        elif not query.strip():
            reason = "submitted query is empty and cannot be parsed into a query goal"
        else:  # pragma: no cover - parser only returns None for the cases above
            reason = "submitted query could not be parsed into a query goal"
        return builder.set_error_record(PIPELINE_COMPONENT, reason)

    # ------------------------------------------------------------- cycle control

    def _run_cycles(self, builder: ProofTraceBuilder) -> PipelineResult:
        """Run the bounded four-stage cycle and apply the full termination semantics.

        Each iteration executes exactly one cycle (four stages in order) and counts as
        one completed cycle, so the number of completed cycles can never exceed the
        configured maximum (the guarantee Task 10.3's Property 7 exercises). Termination
        follows the design's *Termination Semantics* for exactly one reason:

        - **goal-satisfied** -- the active goal is satisfied; a :class:`VerifiedOutput`
          with the attached Faithfulness_Score is emitted (Req 1.3, 7.6).
        - **cycle-limit-reached** -- the loop reaches the configured maximum without the
          goal being satisfied (Req 1.4).
        - **constraint-unsatisfied** -- constrained decoding exhausted its retries
          without a conforming step (Req 3.4), surfaced from the decoder.
        - **repair-exhausted** -- the Repair Coordinator reached its attempt limit
          without an accepted step (Req 6.6), surfaced from repair.
        - **component-error** -- the LLM was unavailable or back-translation failed; an
          :class:`ErrorRecord` is returned while the Proof_Trace is preserved (Req 1.6).
        """
        # An already-satisfied goal (e.g. a goal with no sub-goals) needs no cycle.
        if self._controller.goal_buffer.satisfied:
            return self._emit_verified_output(builder)

        while self._completed_cycles < self._config.max_cycle_limit:
            cycle_index = self._completed_cycles
            try:
                outcome = self._execute_cycle(builder, cycle_index)
            except ConstraintUnsatisfied:
                # Req 3.4: constrained decoding exhausted its retries. The decoder has
                # already set the termination reason on the trace; finish on it.
                self._completed_cycles += 1
                return self._finish(builder, TerminationReason.CONSTRAINT_UNSATISFIED)
            except BackTranslationError as exc:
                # Req 1.6/5.5: back-translation failed -> component-error, trace kept.
                self._completed_cycles += 1
                return self._terminate_component_error(builder, exc.error_record)
            except LLMError as exc:
                # Req 1.6/2.6: the LLM is unavailable/timed out -> component-error.
                self._completed_cycles += 1
                return self._terminate_component_error(
                    builder, self._llm_error_record(builder, exc)
                )

            self._completed_cycles += 1

            if outcome.goal_satisfied:
                return self._emit_verified_output(builder)

            # Drive the Repair Coordinator on repair-triggering outcomes (Req 6.4-6.6).
            if outcome.needs_repair and self._repair is not None:
                repaired = self._drive_repair(builder, outcome)
                if not repaired.succeeded:
                    # Req 6.6: the repair attempt limit was reached without acceptance.
                    return self._finish(builder, TerminationReason.REPAIR_EXHAUSTED)
                if self._integrate_repaired(outcome, repaired):
                    return self._emit_verified_output(builder)

        # Req 1.4: the loop reached the configured maximum without goal satisfaction.
        return self._finish(builder, TerminationReason.CYCLE_LIMIT_REACHED)

    def _execute_cycle(
        self, builder: ProofTraceBuilder, cycle_index: int
    ) -> CycleOutcome:
        """Execute one cycle's four stages in the fixed order (Req 1.2).

        Stage order is always generate -> translate -> controller update -> validate.
        Each stage is announced via :meth:`_record_stage` so the ordering is observable.
        Acceptance integration into working memory and sub-goal advancement (Req 4.2,
        4.3, 4.7) follow from the validation outcome after the four stages complete.
        """
        # --- Stage 1: LLM generation (constrained) ---------------------------------
        self._record_stage(cycle_index, CycleStage.GENERATE)
        state = self._controller.state()
        context = self._translation.to_context(state)
        candidate = self._decoder.decode(context, state, builder=builder)

        # --- Stage 2: translation to symbolic form ---------------------------------
        self._record_stage(cycle_index, CycleStage.TRANSLATE)
        translated = self._translation.to_symbolic(candidate)
        if isinstance(translated, Untranslatable):
            # Req 5.3: flag untranslatable, leave buffers unchanged, route to repair.
            step = builder.append_step(
                candidate.raw_text,
                status=ValidationStatus.REJECTED,
                applied_rule_id=None,
                translation_outcomes=[
                    {"direction": "forward", "untranslatable": True,
                     "reason": translated.reason}
                ],
            )
            return CycleOutcome(
                proof_step=step,
                needs_repair=True,
                repair_trigger=RepairTrigger.UNTRANSLATABLE,
                state=state,
                rejected_representation=None,
                repair_reason=translated.reason,
            )

        # --- Stage 3: ACT-R Controller update (rule selection) ----------------------
        self._record_stage(cycle_index, CycleStage.CONTROLLER_UPDATE)
        rule_selection = self._controller.select_rule(state)
        if isinstance(rule_selection, NoRuleMatched):
            # Req 4.8: record no-rule-matched and route the state to repair.
            step = builder.append_step(
                candidate.raw_text,
                representation=translated,
                status=ValidationStatus.REJECTED,
                applied_rule_id=None,
            )
            return CycleOutcome(
                proof_step=step,
                needs_repair=True,
                repair_trigger=RepairTrigger.NO_RULE_MATCHED,
                state=state,
                rejected_representation=translated,
                repair_reason=rule_selection.reason,
            )
        applied_rule: ProductionRule = rule_selection

        # --- Stage 4: validation ----------------------------------------------------
        self._record_stage(cycle_index, CycleStage.VALIDATE)
        outcome = self._validation.validate(translated, state.procedural_memory)

        if outcome.accepted:
            step = builder.append_step(
                candidate.raw_text,
                representation=translated,
                status=ValidationStatus.ACCEPTED,
                applied_rule_id=applied_rule.rule_id,
                applied_rule_origin=self._rule_origin(applied_rule.rule_id),
            )
            goal_satisfied = self._integrate_accepted(translated)
            return CycleOutcome(
                proof_step=step, accepted=True, goal_satisfied=goal_satisfied
            )

        # Rejected: record every violated rule; the orchestrator drives repair (Req 6.4).
        step = builder.append_step(
            candidate.raw_text,
            representation=translated,
            status=ValidationStatus.REJECTED,
            applied_rule_id=applied_rule.rule_id,
            violated_rule_ids=outcome.violated_rule_ids,
        )
        return CycleOutcome(
            proof_step=step,
            needs_repair=True,
            repair_trigger=RepairTrigger.REJECTION,
            state=state,
            rejected_representation=translated,
            violated_rules=list(outcome.violated_rules),
        )

    def _integrate_accepted(self, representation: SymbolicRepresentation) -> bool:
        """Integrate an accepted step and advance the goal; return goal satisfaction.

        Stores the accepted conclusion in Declarative_Memory and replaces the
        Imaginal_Buffer (Req 4.2, 4.5), then advances the Goal_Buffer to the next unmet
        sub-goal, marking the active goal satisfied when none remain (Req 4.3, 4.7).
        """
        self._controller.integrate_accepted(representation)
        self._controller.advance_sub_goal()
        return self._controller.goal_buffer.satisfied

    def _rule_origin(self, rule_id: Optional[str]) -> Optional[RuleOrigin]:
        """Classify an applied rule id as ``LEARNED`` or ``SEEDED`` (Req 14.5).

        A rule promoted into Procedural_Memory by the Rule Learner during this run is a
        ``Learned_Rule``; every other applied rule is a ``Seeded_Rule``. When no rule was
        applied (``rule_id is None``) the origin is left unknown (``None``) so the marker
        renders exactly as before. The classification is purely additive metadata and
        does not depend on whether rule learning is enabled, so the disabled path's
        accepted steps are simply all marked ``SEEDED``.
        """
        if rule_id is None:
            return None
        return (
            RuleOrigin.LEARNED
            if rule_id in self._learned_rule_ids
            else RuleOrigin.SEEDED
        )

    # ---------------------------------------------------------------- repair loop

    def _drive_repair(
        self, builder: ProofTraceBuilder, outcome: CycleOutcome
    ) -> RepairOutcome:
        """Drive the Repair Coordinator for a repair-triggering cycle outcome (Req 6.4-6.6).

        Builds a :class:`RepairContext` from the offending outcome -- the trigger, the
        working-memory state to regenerate from, the rejected step being repaired, the
        violated rules, and the reason -- and hands it to the coordinator. The coordinator
        journals each attempt into the same Proof_Trace and, on exhaustion, records the
        ``repair-exhausted`` termination reason (Req 6.6).
        """
        assert self._repair is not None  # guarded by the caller
        assert outcome.proof_step is not None and outcome.state is not None
        context = RepairContext(
            trigger=outcome.repair_trigger or RepairTrigger.REJECTION,
            state=outcome.state,
            proof_step=outcome.proof_step,
            rejected_representation=outcome.rejected_representation,
            violated_rules=list(outcome.violated_rules),
            reason=outcome.repair_reason,
        )
        return self._repair.repair(context, builder=builder)

    def _integrate_repaired(
        self, outcome: CycleOutcome, repaired: RepairOutcome
    ) -> bool:
        """Integrate an accepted repaired step and return whether the goal is satisfied.

        Marks the original step ``REPAIRED`` in the Proof_Trace (Req 8.2), integrates the
        accepted repaired representation into working memory, and advances the goal as for
        any accepted step (Req 4.2, 4.3, 4.5, 4.7).
        """
        if outcome.proof_step is not None:
            outcome.proof_step.status = ValidationStatus.REPAIRED
        representation = repaired.accepted_representation
        if representation is None:  # defensive: success implies a representation
            return self._controller.goal_buffer.satisfied
        return self._integrate_accepted(representation)

    # ------------------------------------------------------------ component error

    def _terminate_component_error(
        self, builder: ProofTraceBuilder, error_record: ErrorRecord
    ) -> ErrorRecord:
        """Convert a component failure into a ``component-error`` termination (Req 1.6).

        Sets the ``component-error`` termination reason, ensures the error record naming
        the failed component is attached to the Proof_Trace, and returns it. The existing
        Proof_Trace contents (every step recorded before the failure) are preserved and
        remain available via :attr:`last_trace`.
        """
        builder.set_termination_reason(TerminationReason.COMPONENT_ERROR)
        record = builder.set_error_record(
            error_record.failed_component, error_record.reason
        )
        self._last_trace = builder.trace
        return record

    def _llm_error_record(
        self, builder: ProofTraceBuilder, exc: LLMError
    ) -> ErrorRecord:
        """Return the error record for an LLM failure, naming the LLM component (Req 2.6).

        The LLM component attaches an error record to the trace when it is given the
        builder; reuse it when present so the recorded failure reason is preserved,
        otherwise synthesise one from the exception.
        """
        existing = builder.trace.error_record
        if existing is not None:
            return existing
        return ErrorRecord(failed_component=LLM_COMPONENT, reason=str(exc))

    # ---------------------------------------------------------- stage observation

    def _record_stage(self, cycle_index: int, stage: CycleStage) -> None:
        """Record (and optionally notify) one stage execution for ordering visibility."""
        self._stage_log.append((cycle_index, stage))
        if self._on_stage is not None:
            self._on_stage(cycle_index, stage)

    # --------------------------------------------------------- termination / emit

    def _emit_verified_output(self, builder: ProofTraceBuilder) -> VerifiedOutput:
        """Emit a Verified_Output on goal satisfaction (Req 1.3, 7.6).

        Sets the ``goal-satisfied`` termination reason, attaches the Faithfulness_Score
        computed from the trace (Req 7.6), and derives the final answer from the current
        Imaginal_Buffer (or the latest accepted conclusion).
        """
        builder.set_termination_reason(TerminationReason.GOAL_SATISFIED)
        trace = builder.trace
        self._last_trace = trace
        output = VerifiedOutput(
            final_answer=self._final_answer(),
            proof_trace=trace,
            faithfulness_score=compute_faithfulness_score(trace),
        )
        # Adaptive rule learning runs only here, on the goal-satisfied path, strictly
        # after the Verified_Output has been produced (Req 14.1). It is best-effort and
        # never alters the emitted output or trace.
        self._maybe_learn_rules(trace)
        return output

    def _maybe_learn_rules(self, trace: ProofTrace) -> None:
        """Induce, corroborate, and promote learned rules from a goal-satisfied trace.

        Gated by :attr:`SystemConfig.rule_learning_enabled` and the presence of a Rule
        Learner. When enabled, the learner induces Candidate_Rules from the trace's
        accepted steps, corroborates them into its shared store, and promotes the
        well-corroborated, non-contradicting ones; each promoted ``Learned_Rule`` is
        appended to this orchestrator's Procedural_Memory and remembered as learned so
        subsequent queries in the run can apply it and mark it ``LEARNED`` (Req 14.1).

        The entire block is **best-effort**: any exception is caught and logged and can
        never corrupt or discard the already-emitted Verified_Output / Proof_Trace
        (Req 14.1). When rule learning is disabled the block is skipped entirely, so
        Procedural_Memory holds only Seeded_Rules and behavior is identical to
        Requirements 1-13 (Req 14.10).
        """
        if not self._config.rule_learning_enabled or self._rule_learner is None:
            return
        try:
            self._query_counter += 1
            trace_id = f"trace-{self._query_counter}"
            candidates = self._rule_learner.induce(trace, trace_id=trace_id)
            self._rule_learner.corroborate(candidates)
            result = self._rule_learner.promote(self._procedural_memory)
            for learned in result.promoted:
                # Extend Procedural_Memory for subsequent queries in the run; the next
                # query re-initializes the controller from this list (Req 14.3).
                self._procedural_memory.append(learned.rule)
                self._learned_rule_ids.add(learned.rule.rule_id)
        except Exception as exc:  # noqa: BLE001 - best-effort; output must survive
            logger.warning(
                "rule learning failed after goal-satisfied emission "
                "(best-effort; emitted output preserved): %r",
                exc,
            )

    def _finish(
        self, builder: ProofTraceBuilder, reason: TerminationReason
    ) -> VerifiedOutput:
        """Terminate with ``reason`` and emit the current Proof_Trace.

        Used for the non-error terminations that still emit the Proof_Trace --
        ``cycle-limit-reached`` (Req 1.4), ``constraint-unsatisfied`` (Req 3.4), and
        ``repair-exhausted`` (Req 6.6). The Faithfulness_Score computed from the trace is
        attached to the emitted output (Req 7.6). Component failures take the separate
        :meth:`_terminate_component_error` path and return an :class:`ErrorRecord`.
        """
        builder.set_termination_reason(reason)
        trace = builder.trace
        self._last_trace = trace
        return VerifiedOutput(
            final_answer=self._final_answer(),
            proof_trace=trace,
            faithfulness_score=compute_faithfulness_score(trace),
        )

    def _final_answer(self) -> str:
        """Best-effort final answer from the Imaginal_Buffer or latest conclusion."""
        imaginal = self._controller.imaginal_buffer
        if imaginal is not None and imaginal.logic_form:
            return imaginal.logic_form
        declarative = self._controller.declarative_memory
        if declarative:
            return declarative[-1].logic_form
        return ""
