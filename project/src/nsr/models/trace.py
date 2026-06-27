"""Proof trace and result types: repair attempts, steps, latency, output, errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .enums import TerminationReason, ValidationStatus
from .learning import RuleOrigin
from .reasoning import SymbolicRepresentation


@dataclass
class RepairAttempt:
    """One repair attempt recorded for a rejected Reasoning_Step."""

    attempt_index: int
    rejected_step: SymbolicRepresentation
    violated_rule_ids: list[str] = field(default_factory=list)
    repaired_step: Optional[SymbolicRepresentation] = None


@dataclass
class ProofStep:
    """A single entry in the Proof_Trace, recorded in execution order."""

    sequence: int
    step_text: str
    representation: Optional[SymbolicRepresentation]
    status: ValidationStatus
    applied_rule_id: Optional[str] = None
    """``None`` renders as the explicit ``no-rule-applied`` indicator."""

    applied_rule_origin: Optional[RuleOrigin] = None
    """Whether the applied rule was ``SEEDED`` or ``LEARNED`` (Req 14.5).

    ``None`` means unknown / no rule applied and renders as today; existing traces and
    serialization round-trips remain unchanged.
    """

    violated_rule_ids: list[str] = field(default_factory=list)
    repair_attempts: list[RepairAttempt] = field(default_factory=list)
    translation_outcomes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LatencyRecord:
    """Wall-clock latency breakdown for a processed query, in milliseconds."""

    pipeline_ms: float
    system2_ms: float
    """Validation_Engine + ACT-R Controller cumulative latency."""

    llm_ms: float
    latency_budget_exceeded: bool = False


@dataclass
class ErrorRecord:
    """Identifies a failed component and the reason for the failure."""

    failed_component: str
    reason: str


@dataclass
class ProofTrace:
    """The ordered record of reasoning steps and outcomes that justifies output."""

    steps: list[ProofStep] = field(default_factory=list)
    termination_reason: Optional[TerminationReason] = None
    latency: Optional[LatencyRecord] = None
    error_record: Optional[ErrorRecord] = None


@dataclass
class VerifiedOutput:
    """The final answer accompanied by its Proof_Trace and Faithfulness_Score."""

    final_answer: str
    proof_trace: ProofTrace
    faithfulness_score: float
