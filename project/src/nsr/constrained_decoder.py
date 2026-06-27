"""Constrained Decoder (System 1).

This module implements the *Constrained Decoder* described in the design's
*Constrained Decoder* section (Task 5.2). Its job is to force every candidate
Reasoning_Step produced by the LLM into the configured structured output format
*before the step is returned* to the rest of the pipeline:

- Restrict LLM output to the configured structured format (``json``, ``logic-form``,
  or ``yaml``) before return (Req 3.1).
- Derive the active decoding constraints from the current contents of the Goal_Buffer,
  Declarative_Memory, Procedural_Memory, and Imaginal_Buffer (Req 3.5).
- Mark non-conforming output as non-conforming, journal the attempt into the
  Proof_Trace, and request regeneration up to the configured retry count (Req 3.3).
- Terminate the query with a ``constraint-unsatisfied`` termination reason when the
  retry count is exhausted without a conforming step (Req 3.4).

The decoder wraps the :class:`~nsr.llm_component.LLMComponent` (System 1). The LLM owns
backend selection, the generation timeout, and timeout/unavailability retries; the
decoder owns *format-conformance* regeneration. A conforming step is normalised so its
``structured`` payload always carries the machine-checkable ``logic_form`` field the
Translation_Layer's forward translation consumes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .llm_component import OutputSchema
from .models import (
    CandidateStep,
    PromptContext,
    SystemConfig,
    TerminationReason,
    ValidationStatus,
    WorkingMemoryState,
)

#: The structured key carrying the machine-checkable encoding (shared with the
#: Translation_Layer). A conforming step must resolve to a non-empty value here.
LOGIC_FORM_KEY = "logic_form"
#: The structured key carrying the parsed predicate fields.
PREDICATES_KEY = "predicates"

#: The configured structured output formats the decoder can enforce (Req 3.1).
ALLOWED_FORMATS: frozenset[str] = frozenset({"json", "logic-form", "yaml"})

#: Marker key placed in a journaled translation-outcome dict for a non-conforming
#: attempt so the Proof_Trace records *why* a generated step was rejected (Req 3.3).
NON_CONFORMING_KEY = "non_conforming"


# ---------------------------------------------------------------------------
# Public exception (raised when the retry count is exhausted)
# ---------------------------------------------------------------------------


class ConstraintUnsatisfied(Exception):
    """Raised when no conforming step is produced within the retry count (Req 3.4).

    Carries the ``reason`` of the final non-conforming attempt, the number of
    ``attempts`` made, and the ``termination_reason`` the orchestrator should surface
    (always :data:`TerminationReason.CONSTRAINT_UNSATISFIED`). When a Proof_Trace
    builder is supplied to :meth:`ConstrainedDecoder.decode`, that builder's termination
    reason is set before this exception is raised so the trace is preserved.
    """

    def __init__(self, reason: str, attempts: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.attempts = attempts
        self.termination_reason = TerminationReason.CONSTRAINT_UNSATISFIED


# ---------------------------------------------------------------------------
# Active decoding constraints derived from the working-memory buffers (Req 3.5)
# ---------------------------------------------------------------------------


@dataclass
class DecodingConstraints:
    """Active decoding constraints derived from the four ACT-R buffers (Req 3.5).

    The constraints couple the *format* requirement (the configured structured output
    format and the keys a conforming step must carry) with the *content* signals pulled
    from the current working-memory state: the active goal and sub-goals, the
    established conclusions in Declarative_Memory, the available production rules in
    Procedural_Memory, and the partial representation in the Imaginal_Buffer. They are
    rendered into a schema (:meth:`to_schema`) handed to the LLM so generation is biased
    toward a step that fits both the format and the current reasoning state.
    """

    output_format: str
    required_keys: tuple[str, ...]
    goal_terms: list[str] = field(default_factory=list)
    sub_goal_terms: list[str] = field(default_factory=list)
    established_conclusions: list[str] = field(default_factory=list)
    rule_ids: list[str] = field(default_factory=list)
    rule_conditions: list[str] = field(default_factory=list)
    partial_representation: Optional[str] = None

    def to_schema(self) -> dict[str, Any]:
        """Render the active constraints into a schema dict for the LLM backend."""
        return {
            "format": self.output_format,
            "required_keys": list(self.required_keys),
            "goal_terms": list(self.goal_terms),
            "sub_goal_terms": list(self.sub_goal_terms),
            "established_conclusions": list(self.established_conclusions),
            "rule_ids": list(self.rule_ids),
            "rule_conditions": list(self.rule_conditions),
            "partial_representation": self.partial_representation,
        }


def derive_constraints(
    state: WorkingMemoryState, output_format: str
) -> DecodingConstraints:
    """Derive the active decoding constraints from the buffer contents (Req 3.5).

    Reads every one of the four buffers held in ``state``:

    - **Goal_Buffer**: the active goal description and its sub-goal descriptions.
    - **Declarative_Memory**: the logic forms of the accepted intermediate conclusions.
    - **Procedural_Memory**: the available production-rule ids and their conditions.
    - **Imaginal_Buffer**: the partial problem representation under construction.

    The configured ``output_format`` selects the required structured keys; every format
    requires the machine-checkable ``logic_form`` field.
    """
    if output_format not in ALLOWED_FORMATS:
        permitted = ", ".join(sorted(ALLOWED_FORMATS))
        raise ValueError(
            f"unknown output_format {output_format!r}; expected one of {{{permitted}}}"
        )
    if state is None:
        raise ValueError("derive_constraints requires a non-None WorkingMemoryState")

    goal = state.goal_buffer
    goal_terms: list[str] = []
    sub_goal_terms: list[str] = []
    if goal is not None:
        if goal.description:
            goal_terms.append(goal.description)
        sub_goal_terms = [sg.description for sg in goal.sub_goals if sg.description]

    established_conclusions = [
        rep.logic_form for rep in state.declarative_memory if rep.logic_form
    ]

    rule_ids = [rule.rule_id for rule in state.procedural_memory]
    rule_conditions = [
        rule.condition for rule in state.procedural_memory if rule.condition
    ]

    partial_representation = (
        state.imaginal_buffer.logic_form if state.imaginal_buffer is not None else None
    )

    return DecodingConstraints(
        output_format=output_format,
        required_keys=(LOGIC_FORM_KEY,),
        goal_terms=goal_terms,
        sub_goal_terms=sub_goal_terms,
        established_conclusions=established_conclusions,
        rule_ids=rule_ids,
        rule_conditions=rule_conditions,
        partial_representation=partial_representation,
    )


# ---------------------------------------------------------------------------
# Conformance checking (Req 3.1)
# ---------------------------------------------------------------------------


@dataclass
class ConformanceResult:
    """The outcome of checking a candidate step against the configured format.

    ``conforming`` is ``True`` when the step fits the format and carries the required
    machine-checkable encoding; ``candidate`` then holds a normalised step whose
    ``structured`` payload always contains a non-empty ``logic_form``. When
    ``conforming`` is ``False``, ``reason`` explains the non-conformance and
    ``candidate`` is ``None``.
    """

    conforming: bool
    candidate: Optional[CandidateStep] = None
    reason: Optional[str] = None


def _parse_simple_yaml(text: str) -> Optional[dict[str, str]]:
    """Parse a flat ``key: value`` YAML mapping, or return ``None`` if not a mapping.

    Only the small subset of YAML the decoder needs is supported: a sequence of
    top-level ``key: value`` lines (blank lines and ``#`` comments are ignored). Any
    line without a ``:`` separator, or a missing key, makes the text a non-mapping and
    yields ``None``.
    """
    result: dict[str, str] = {}
    saw_line = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        saw_line = True
        if ":" not in line:
            return None
        key, _, value = line.partition(":")
        key = key.strip()
        if not key:
            return None
        result[key] = value.strip()
    return result if saw_line else None


def check_conformance(
    step: CandidateStep, constraints: DecodingConstraints
) -> ConformanceResult:
    """Check ``step`` against the configured structured format (Req 3.1).

    Dispatches on ``constraints.output_format``:

    - ``json``: the step's structured payload (or its raw text parsed as a JSON object)
      must be a JSON object containing a non-empty string ``logic_form``.
    - ``logic-form``: the raw text must be a non-empty logic form, taken verbatim as the
      machine-checkable encoding.
    - ``yaml``: the raw text must parse as a flat mapping containing a non-empty
      ``logic_form`` key.

    On success the returned candidate is normalised so ``structured[logic_form]`` is a
    non-empty string ready for forward translation.
    """
    fmt = constraints.output_format
    if fmt == "json":
        return _check_json(step)
    if fmt == "logic-form":
        return _check_logic_form(step)
    if fmt == "yaml":
        return _check_yaml(step)
    # Defensive: derive_constraints already rejects unknown formats.
    return ConformanceResult(
        conforming=False, reason=f"unsupported output format {fmt!r}"
    )


def _check_json(step: CandidateStep) -> ConformanceResult:
    structured = step.structured
    if not isinstance(structured, dict) or not structured:
        # The LLM component parses JSON objects into ``structured``; fall back to
        # parsing the raw text so the decoder is self-contained.
        try:
            parsed = json.loads(step.raw_text)
        except (ValueError, TypeError):
            parsed = None
        if not isinstance(parsed, dict):
            return ConformanceResult(
                conforming=False,
                reason="output is not a JSON object conforming to the configured format",
            )
        structured = parsed

    logic_form = structured.get(LOGIC_FORM_KEY)
    if not isinstance(logic_form, str) or not logic_form.strip():
        return ConformanceResult(
            conforming=False,
            reason=f"JSON object is missing a non-empty {LOGIC_FORM_KEY!r} field",
        )

    predicates = structured.get(PREDICATES_KEY, {})
    if not isinstance(predicates, dict):
        predicates = {}
    normalised = dict(structured)
    normalised[LOGIC_FORM_KEY] = logic_form
    normalised[PREDICATES_KEY] = dict(predicates)
    return ConformanceResult(
        conforming=True,
        candidate=CandidateStep(
            raw_text=step.raw_text,
            structured=normalised,
            sub_goal=step.sub_goal,
        ),
    )


def _check_logic_form(step: CandidateStep) -> ConformanceResult:
    text = (step.raw_text or "").strip()
    if not text:
        return ConformanceResult(
            conforming=False,
            reason="output is empty and cannot form a logic-form encoding",
        )
    existing = step.structured if isinstance(step.structured, dict) else {}
    predicates = existing.get(PREDICATES_KEY, {})
    if not isinstance(predicates, dict):
        predicates = {}
    return ConformanceResult(
        conforming=True,
        candidate=CandidateStep(
            raw_text=step.raw_text,
            structured={LOGIC_FORM_KEY: text, PREDICATES_KEY: dict(predicates)},
            sub_goal=step.sub_goal,
        ),
    )


def _check_yaml(step: CandidateStep) -> ConformanceResult:
    parsed = _parse_simple_yaml(step.raw_text or "")
    if parsed is None:
        return ConformanceResult(
            conforming=False,
            reason="output does not parse as a YAML mapping",
        )
    logic_form = parsed.get(LOGIC_FORM_KEY, "")
    if not isinstance(logic_form, str) or not logic_form.strip():
        return ConformanceResult(
            conforming=False,
            reason=f"YAML mapping is missing a non-empty {LOGIC_FORM_KEY!r} key",
        )
    predicates = {k: v for k, v in parsed.items() if k != LOGIC_FORM_KEY}
    return ConformanceResult(
        conforming=True,
        candidate=CandidateStep(
            raw_text=step.raw_text,
            structured={LOGIC_FORM_KEY: logic_form.strip(), PREDICATES_KEY: predicates},
            sub_goal=step.sub_goal,
        ),
    )


# ---------------------------------------------------------------------------
# Generator protocol (the System 1 LLM the decoder wraps)
# ---------------------------------------------------------------------------


class StepGenerator(Protocol):
    """The minimal generation interface the decoder depends on.

    Satisfied by :class:`~nsr.llm_component.LLMComponent`; declared structurally so the
    decoder can be exercised with any conforming generator (including test fakes).
    """

    def generate_step(
        self,
        context: PromptContext,
        constraint: Optional[OutputSchema] = ...,
        *,
        trace: Any = ...,
    ) -> CandidateStep: ...


# ---------------------------------------------------------------------------
# Constrained Decoder
# ---------------------------------------------------------------------------


class ConstrainedDecoder:
    """Forces LLM output into the configured structured format before return.

    The decoder derives the active constraints from the current working-memory state
    (Req 3.5), asks the wrapped generator for a candidate step under those constraints,
    and checks the result against the configured format (Req 3.1). A non-conforming
    result is marked and journalled, and regeneration is requested up to the configured
    retry count (Req 3.3). If the retry count is exhausted without a conforming step,
    the query is terminated with ``constraint-unsatisfied`` (Req 3.4).
    """

    def __init__(self, generator: StepGenerator, config: SystemConfig) -> None:
        if config.output_format not in ALLOWED_FORMATS:
            permitted = ", ".join(sorted(ALLOWED_FORMATS))
            raise ValueError(
                f"unknown output_format {config.output_format!r}; "
                f"expected one of {{{permitted}}}"
            )
        if config.retry_count < 0:
            raise ValueError("retry_count must be non-negative")
        self._generator = generator
        self._output_format = config.output_format
        self._retry_count = config.retry_count

    @property
    def output_format(self) -> str:
        """The configured structured output format the decoder enforces."""
        return self._output_format

    @property
    def retry_count(self) -> int:
        """The configured number of regenerations permitted on non-conformance."""
        return self._retry_count

    def derive_constraints(self, state: WorkingMemoryState) -> DecodingConstraints:
        """Derive the active decoding constraints from ``state`` (Req 3.5)."""
        return derive_constraints(state, self._output_format)

    def decode(
        self,
        context: PromptContext,
        state: WorkingMemoryState,
        *,
        builder: Any = None,
    ) -> CandidateStep:
        """Return a single conforming candidate step, regenerating as needed.

        ``context`` is the prompt context produced by backward translation, and
        ``state`` supplies the buffers the active constraints are derived from. The
        decoder makes at most ``retry_count + 1`` generation attempts: the initial
        attempt plus up to ``retry_count`` regenerations (Req 3.3).

        Each non-conforming attempt is marked non-conforming and journalled into the
        Proof_Trace via the optional ``builder`` (a
        :class:`~nsr.proof_trace.ProofTraceBuilder`). On exhaustion the builder's
        termination reason is set to ``constraint-unsatisfied`` and
        :class:`ConstraintUnsatisfied` is raised (Req 3.4).
        """
        constraints = self.derive_constraints(state)
        schema = OutputSchema(
            format=self._output_format, schema=constraints.to_schema()
        )
        attempts = self._retry_count + 1
        last_reason = "no generation attempt produced a conforming step"

        for attempt_index in range(attempts):
            candidate = self._generator.generate_step(
                context, schema, trace=builder
            )
            result = check_conformance(candidate, constraints)
            if result.conforming and result.candidate is not None:
                return result.candidate

            last_reason = result.reason or "output did not conform to the format"
            if builder is not None:
                self._journal_non_conforming(
                    builder, candidate, last_reason, attempt_index
                )

        reason = (
            f"no conforming {self._output_format} step after {attempts} attempt(s): "
            f"{last_reason}"
        )
        if builder is not None and hasattr(builder, "set_termination_reason"):
            builder.set_termination_reason(TerminationReason.CONSTRAINT_UNSATISFIED)
        raise ConstraintUnsatisfied(reason, attempts)

    def _journal_non_conforming(
        self,
        builder: Any,
        candidate: CandidateStep,
        reason: str,
        attempt_index: int,
    ) -> None:
        """Mark a non-conforming attempt and record it in the Proof_Trace (Req 3.3).

        The attempt is recorded as a rejected step carrying a translation-outcome dict
        that flags the non-conformance, names the violated format, and gives the
        attempt index, so the trace preserves every regeneration attempt.
        """
        builder.append_step(
            candidate.raw_text,
            status=ValidationStatus.REJECTED,
            applied_rule_id=None,
            translation_outcomes=[
                {
                    NON_CONFORMING_KEY: True,
                    "format": self._output_format,
                    "reason": reason,
                    "attempt": attempt_index,
                }
            ],
        )
