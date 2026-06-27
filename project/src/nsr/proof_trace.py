"""Append-only Proof_Trace builder and latency recording.

This module provides :class:`ProofTraceBuilder`, an append-only manager that produces
the :class:`~nsr.models.trace.ProofTrace` dataclass defined in the design. It is the
single journal through which the orchestrator records:

- every Reasoning_Step in execution order, with an automatically assigned sequence
  position, its validation outcome, and the applied production rule id (or an explicit
  ``no-rule-applied`` indicator when no rule was applied) -- Req 8.1, 8.2;
- the per-attempt repair details for a rejected step, in execution order -- Req 8.3;
- the pipeline, System-2 (Validation_Engine + ACT-R Controller), and LLM latencies, and
  a ``latency_budget_exceeded`` flag when the cumulative System-2 latency exceeds the
  configured latency budget -- Req 11.1, 11.2, 11.4.

Exporters (machine-readable and human-readable renderings) are implemented separately
in Task 8.2; this module only builds and journals the trace.
"""

from __future__ import annotations

from typing import Any, Optional

from .models.enums import TerminationReason, ValidationStatus
from .models.learning import RuleOrigin
from .models.reasoning import SymbolicRepresentation
from .models.trace import (
    ErrorRecord,
    LatencyRecord,
    ProofStep,
    ProofTrace,
    RepairAttempt,
)

#: Explicit indicator recorded/rendered when a step has no applied production rule.
NO_RULE_APPLIED = "no-rule-applied"


def applied_rule_label(step: ProofStep) -> str:
    """Return the applied rule id of ``step`` or the explicit no-rule indicator.

    Req 8.2 requires that each step record the identifier of the applied production
    rule, or an explicit ``no-rule-applied`` indicator when no production rule was
    applied. The :class:`ProofStep` stores ``None`` for the latter; this helper
    resolves that ``None`` into :data:`NO_RULE_APPLIED`.
    """

    return step.applied_rule_id if step.applied_rule_id is not None else NO_RULE_APPLIED


class ProofTraceBuilder:
    """Append-only builder that wraps and produces a :class:`ProofTrace`.

    Steps are appended strictly in execution order; each receives a sequence position
    equal to its zero-based index in the trace, so ``trace.steps[i].sequence == i`` is
    an invariant. The builder never reorders or removes recorded entries.
    """

    def __init__(self, latency_budget_ms: Optional[int] = None) -> None:
        """Create an empty builder.

        :param latency_budget_ms: Optional configured System-2 latency budget in
            milliseconds. When set, :meth:`record_latency` flags a query whose
            cumulative System-2 latency exceeds this budget (Req 11.4).
        """

        if latency_budget_ms is not None and latency_budget_ms < 0:
            raise ValueError("latency_budget_ms must be non-negative")
        self._trace = ProofTrace()
        self._latency_budget_ms = latency_budget_ms
        self._system2_ms = 0.0
        self._llm_ms = 0.0

    # -- accessors ---------------------------------------------------------------

    @property
    def trace(self) -> ProofTrace:
        """The :class:`ProofTrace` being built (the live, append-only instance)."""

        return self._trace

    @property
    def latency_budget_ms(self) -> Optional[int]:
        """The configured System-2 latency budget, or ``None`` when unconfigured."""

        return self._latency_budget_ms

    # -- step recording ----------------------------------------------------------

    def append_step(
        self,
        step_text: str,
        *,
        representation: Optional[SymbolicRepresentation] = None,
        status: ValidationStatus,
        applied_rule_id: Optional[str] = None,
        applied_rule_origin: Optional[RuleOrigin] = None,
        violated_rule_ids: Optional[list[str]] = None,
        translation_outcomes: Optional[list[dict[str, Any]]] = None,
    ) -> ProofStep:
        """Append a Reasoning_Step in execution order and return it.

        The sequence position is assigned automatically as the next index, satisfying
        the execution-order requirement (Req 8.1, 8.2). Pass ``applied_rule_id=None``
        to record that no production rule was applied; :func:`applied_rule_label`
        resolves it to the explicit ``no-rule-applied`` indicator.

        ``applied_rule_origin`` records whether the applied rule was a ``SEEDED`` or a
        ``LEARNED`` rule (Req 14.5). It defaults to ``None`` (unknown / no rule applied),
        so existing callers and serialization round-trips are unaffected.
        """

        step = ProofStep(
            sequence=len(self._trace.steps),
            step_text=step_text,
            representation=representation,
            status=status,
            applied_rule_id=applied_rule_id,
            applied_rule_origin=applied_rule_origin,
            violated_rule_ids=list(violated_rule_ids) if violated_rule_ids else [],
            translation_outcomes=(
                list(translation_outcomes) if translation_outcomes else []
            ),
        )
        self._trace.steps.append(step)
        return step

    def record_repair_attempt(
        self,
        step: ProofStep,
        *,
        rejected_step: SymbolicRepresentation,
        violated_rule_ids: Optional[list[str]] = None,
        repaired_step: Optional[SymbolicRepresentation] = None,
    ) -> RepairAttempt:
        """Record one repair attempt for ``step``, in execution order (Req 8.3).

        The attempt index is assigned automatically as the next index within the
        step's repair history, preserving per-attempt order. Each attempt records the
        rejected step, the violated production rule ids, and the resulting repaired
        step (``None`` when the attempt produced no accepted step yet).
        """

        if step not in self._trace.steps:
            raise ValueError("step is not part of this trace")
        attempt = RepairAttempt(
            attempt_index=len(step.repair_attempts),
            rejected_step=rejected_step,
            violated_rule_ids=list(violated_rule_ids) if violated_rule_ids else [],
            repaired_step=repaired_step,
        )
        step.repair_attempts.append(attempt)
        return attempt

    def add_translation_outcome(
        self, step: ProofStep, outcome: dict[str, Any]
    ) -> None:
        """Append a translation outcome record to ``step`` (supports Req 5.4)."""

        if step not in self._trace.steps:
            raise ValueError("step is not part of this trace")
        step.translation_outcomes.append(outcome)

    # -- latency recording -------------------------------------------------------

    def add_system2_latency(self, ms: float) -> float:
        """Accumulate Validation_Engine + ACT-R Controller latency in ms (Req 11.2).

        Returns the running System-2 total.
        """

        if ms < 0:
            raise ValueError("latency must be non-negative")
        self._system2_ms += ms
        return self._system2_ms

    def add_llm_latency(self, ms: float) -> float:
        """Accumulate LLM generation latency in milliseconds (Req 11.2).

        Returns the running LLM total.
        """

        if ms < 0:
            raise ValueError("latency must be non-negative")
        self._llm_ms += ms
        return self._llm_ms

    def record_latency(
        self,
        pipeline_ms: float,
        *,
        system2_ms: Optional[float] = None,
        llm_ms: Optional[float] = None,
    ) -> LatencyRecord:
        """Record the query's latency breakdown into the trace (Req 11.1, 11.2, 11.4).

        ``pipeline_ms`` is the complete wall-clock pipeline latency. ``system2_ms`` and
        ``llm_ms`` default to the values accumulated via :meth:`add_system2_latency`
        and :meth:`add_llm_latency`; pass explicit values to override them.

        When a latency budget is configured and the cumulative System-2 latency exceeds
        it, the resulting record's ``latency_budget_exceeded`` flag is set (Req 11.4).
        """

        s2 = self._system2_ms if system2_ms is None else system2_ms
        llm = self._llm_ms if llm_ms is None else llm_ms
        if pipeline_ms < 0 or s2 < 0 or llm < 0:
            raise ValueError("latency must be non-negative")

        exceeded = (
            self._latency_budget_ms is not None and s2 > self._latency_budget_ms
        )
        record = LatencyRecord(
            pipeline_ms=pipeline_ms,
            system2_ms=s2,
            llm_ms=llm,
            latency_budget_exceeded=exceeded,
        )
        self._trace.latency = record
        return record

    # -- termination / error -----------------------------------------------------

    def set_termination_reason(self, reason: TerminationReason) -> None:
        """Record the single reason the reasoning cycle terminated (Req 8.1)."""

        self._trace.termination_reason = reason

    def set_error_record(self, failed_component: str, reason: str) -> ErrorRecord:
        """Attach an error record naming the failed component, preserving the trace."""

        record = ErrorRecord(failed_component=failed_component, reason=reason)
        self._trace.error_record = record
        return record
