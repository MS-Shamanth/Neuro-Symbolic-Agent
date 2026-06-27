"""Proof_Trace exporters: lossless machine-readable form and human-readable rendering.

This module implements Task 8.2 of the design's *Proof Trace and Exporters* component:

- **Machine-readable (Req 8.4):** :func:`trace_to_dict` / :func:`trace_to_json` serialize
  a :class:`~nsr.models.trace.ProofTrace` into a plain, JSON-compatible structure, and the
  matching :func:`trace_from_dict` / :func:`trace_from_json` parse it back into an equal
  :class:`ProofTrace`. Every recorded field round-trips without loss -- including enums,
  nested :class:`~nsr.models.reasoning.SymbolicRepresentation` records, repair attempts,
  translation outcomes, the latency record, the termination reason, and the error record.
  Formally, ``trace_from_dict(trace_to_dict(t)) == t`` for any ``t`` (the guarantee
  Property 4 in Task 8.3 exercises).

- **Human-readable (Req 8.5):** :func:`render_trace` produces a text rendering that
  presents each Reasoning_Step in execution order with its sequence position, validation
  outcome, and the applied production rule id (or the explicit ``no-rule-applied``
  indicator from :func:`~nsr.proof_trace.applied_rule_label`).

The machine-readable form embeds a ``schema_version`` so future format changes remain
detectable; parsing tolerates its absence for forward/backward leniency.
"""

from __future__ import annotations

import json
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
from .proof_trace import NO_RULE_APPLIED, applied_rule_label

#: Version tag embedded in the machine-readable form so format changes stay detectable.
SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Machine-readable serialization (Req 8.4)
# --------------------------------------------------------------------------- #


def _representation_to_dict(
    rep: Optional[SymbolicRepresentation],
) -> Optional[dict[str, Any]]:
    """Serialize a :class:`SymbolicRepresentation` (or ``None``) losslessly."""

    if rep is None:
        return None
    return {
        "logic_form": rep.logic_form,
        # Copy the predicates mapping so the serialized form is decoupled from the
        # live object; values are assumed JSON-compatible per the structured schema.
        "predicates": dict(rep.predicates),
        "source_text": rep.source_text,
    }


def _representation_from_dict(
    data: Optional[dict[str, Any]],
) -> Optional[SymbolicRepresentation]:
    """Reconstruct a :class:`SymbolicRepresentation` from its serialized form."""

    if data is None:
        return None
    return SymbolicRepresentation(
        logic_form=data["logic_form"],
        predicates=dict(data.get("predicates", {})),
        source_text=data.get("source_text", ""),
    )


def _repair_attempt_to_dict(attempt: RepairAttempt) -> dict[str, Any]:
    """Serialize a :class:`RepairAttempt` losslessly (Req 8.3)."""

    return {
        "attempt_index": attempt.attempt_index,
        "rejected_step": _representation_to_dict(attempt.rejected_step),
        "violated_rule_ids": list(attempt.violated_rule_ids),
        "repaired_step": _representation_to_dict(attempt.repaired_step),
    }


def _repair_attempt_from_dict(data: dict[str, Any]) -> RepairAttempt:
    """Reconstruct a :class:`RepairAttempt` from its serialized form."""

    rejected = _representation_from_dict(data["rejected_step"])
    # rejected_step is non-optional on the dataclass; a well-formed trace always
    # records it, so a missing value indicates a corrupt artifact.
    if rejected is None:
        raise ValueError("repair attempt is missing its rejected_step")
    return RepairAttempt(
        attempt_index=data["attempt_index"],
        rejected_step=rejected,
        violated_rule_ids=list(data.get("violated_rule_ids", [])),
        repaired_step=_representation_from_dict(data.get("repaired_step")),
    )


def _step_to_dict(step: ProofStep) -> dict[str, Any]:
    """Serialize a :class:`ProofStep` losslessly (Req 8.2, 8.3)."""

    return {
        "sequence": step.sequence,
        "step_text": step.step_text,
        "representation": _representation_to_dict(step.representation),
        "status": step.status.value,
        "applied_rule_id": step.applied_rule_id,
        # Learned-vs-seeded marker (Req 14.5): serialize as the enum value, or None.
        "applied_rule_origin": (
            step.applied_rule_origin.value
            if step.applied_rule_origin is not None
            else None
        ),
        "violated_rule_ids": list(step.violated_rule_ids),
        "repair_attempts": [
            _repair_attempt_to_dict(a) for a in step.repair_attempts
        ],
        # translation_outcomes is a list of plain dicts; copy each for isolation.
        "translation_outcomes": [dict(o) for o in step.translation_outcomes],
    }


def _step_from_dict(data: dict[str, Any]) -> ProofStep:
    """Reconstruct a :class:`ProofStep` from its serialized form."""

    # Read the learned-vs-seeded marker tolerantly (Req 14.5): older artifacts
    # predate the key, so an absent or None value reconstructs as None.
    origin_value = data.get("applied_rule_origin")
    applied_rule_origin = (
        RuleOrigin(origin_value) if origin_value is not None else None
    )
    return ProofStep(
        sequence=data["sequence"],
        step_text=data["step_text"],
        representation=_representation_from_dict(data.get("representation")),
        status=ValidationStatus(data["status"]),
        applied_rule_id=data.get("applied_rule_id"),
        applied_rule_origin=applied_rule_origin,
        violated_rule_ids=list(data.get("violated_rule_ids", [])),
        repair_attempts=[
            _repair_attempt_from_dict(a) for a in data.get("repair_attempts", [])
        ],
        translation_outcomes=[dict(o) for o in data.get("translation_outcomes", [])],
    )


def _latency_to_dict(latency: Optional[LatencyRecord]) -> Optional[dict[str, Any]]:
    """Serialize a :class:`LatencyRecord` (or ``None``) losslessly (Req 11)."""

    if latency is None:
        return None
    return {
        "pipeline_ms": latency.pipeline_ms,
        "system2_ms": latency.system2_ms,
        "llm_ms": latency.llm_ms,
        "latency_budget_exceeded": latency.latency_budget_exceeded,
    }


def _latency_from_dict(data: Optional[dict[str, Any]]) -> Optional[LatencyRecord]:
    """Reconstruct a :class:`LatencyRecord` from its serialized form."""

    if data is None:
        return None
    return LatencyRecord(
        pipeline_ms=data["pipeline_ms"],
        system2_ms=data["system2_ms"],
        llm_ms=data["llm_ms"],
        latency_budget_exceeded=data.get("latency_budget_exceeded", False),
    )


def _error_to_dict(error: Optional[ErrorRecord]) -> Optional[dict[str, Any]]:
    """Serialize an :class:`ErrorRecord` (or ``None``) losslessly."""

    if error is None:
        return None
    return {
        "failed_component": error.failed_component,
        "reason": error.reason,
    }


def _error_from_dict(data: Optional[dict[str, Any]]) -> Optional[ErrorRecord]:
    """Reconstruct an :class:`ErrorRecord` from its serialized form."""

    if data is None:
        return None
    return ErrorRecord(
        failed_component=data["failed_component"],
        reason=data["reason"],
    )


def trace_to_dict(trace: ProofTrace) -> dict[str, Any]:
    """Serialize a :class:`ProofTrace` into a lossless, JSON-compatible dict (Req 8.4).

    The result records every field of the trace -- each step in execution order, its
    nested representation, repair attempts, and translation outcomes, plus the
    termination reason, latency record, and error record. :func:`trace_from_dict`
    parses the result back into a :class:`ProofTrace` equal to ``trace``.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "steps": [_step_to_dict(s) for s in trace.steps],
        "termination_reason": (
            trace.termination_reason.value
            if trace.termination_reason is not None
            else None
        ),
        "latency": _latency_to_dict(trace.latency),
        "error_record": _error_to_dict(trace.error_record),
    }


def trace_from_dict(data: dict[str, Any]) -> ProofTrace:
    """Parse the machine-readable dict produced by :func:`trace_to_dict` (Req 8.4).

    Reconstructs an equal :class:`ProofTrace`, restoring enum members, nested
    representations, repair attempts, translation outcomes, latency, and error record.
    """

    termination = data.get("termination_reason")
    return ProofTrace(
        steps=[_step_from_dict(s) for s in data.get("steps", [])],
        termination_reason=(
            TerminationReason(termination) if termination is not None else None
        ),
        latency=_latency_from_dict(data.get("latency")),
        error_record=_error_from_dict(data.get("error_record")),
    )


def trace_to_json(trace: ProofTrace, *, indent: Optional[int] = None) -> str:
    """Serialize a :class:`ProofTrace` to a JSON string (Req 8.4).

    A thin wrapper over :func:`trace_to_dict`; pass ``indent`` for a pretty-printed
    artifact. :func:`trace_from_json` parses the result back without loss.
    """

    return json.dumps(trace_to_dict(trace), indent=indent, ensure_ascii=False)


def trace_from_json(text: str) -> ProofTrace:
    """Parse a JSON string produced by :func:`trace_to_json` into a :class:`ProofTrace`."""

    return trace_from_dict(json.loads(text))


# --------------------------------------------------------------------------- #
# Human-readable rendering (Req 8.5)
# --------------------------------------------------------------------------- #


def render_step(step: ProofStep) -> str:
    """Render a single :class:`ProofStep` as one or more human-readable lines.

    Presents the sequence position, validation outcome, applied rule id (or the
    explicit ``no-rule-applied`` indicator), the step text, any violated rules, and
    each repair attempt in execution order.
    """

    lines: list[str] = []
    header = (
        f"Step {step.sequence}: {step.status.value} "
        f"[rule: {applied_rule_label(step)}]"
    )
    lines.append(header)
    if step.step_text:
        lines.append(f"    text: {step.step_text}")
    if step.violated_rule_ids:
        lines.append(f"    violated rules: {', '.join(step.violated_rule_ids)}")
    for attempt in step.repair_attempts:
        violated = (
            ", ".join(attempt.violated_rule_ids)
            if attempt.violated_rule_ids
            else NO_RULE_APPLIED
        )
        if attempt.repaired_step is not None:
            repaired = attempt.repaired_step.logic_form
        else:
            repaired = "(unrepaired)"
        lines.append(
            f"    repair {attempt.attempt_index}: "
            f"violated [{violated}] -> repaired: {repaired}"
        )
    return "\n".join(lines)


def render_trace(trace: ProofTrace) -> str:
    """Produce a human-readable rendering of a :class:`ProofTrace` (Req 8.5).

    Presents every Reasoning_Step in execution order with its sequence position,
    validation outcome, and applied production rule, followed by the termination
    reason, latency summary, and any error record. Mirrors the ordering of
    ``trace.steps`` so the rendering reflects the true execution order.
    """

    lines: list[str] = ["Proof Trace"]

    if not trace.steps:
        lines.append("  (no reasoning steps recorded)")
    else:
        for step in trace.steps:
            rendered = render_step(step)
            lines.extend(f"  {line}" for line in rendered.split("\n"))

    if trace.termination_reason is not None:
        lines.append(f"Termination: {trace.termination_reason.value}")

    if trace.latency is not None:
        lat = trace.latency
        budget = " (budget exceeded)" if lat.latency_budget_exceeded else ""
        lines.append(
            "Latency: "
            f"pipeline={lat.pipeline_ms}ms, "
            f"system2={lat.system2_ms}ms, "
            f"llm={lat.llm_ms}ms{budget}"
        )

    if trace.error_record is not None:
        lines.append(
            f"Error: {trace.error_record.failed_component} - "
            f"{trace.error_record.reason}"
        )

    return "\n".join(lines)
