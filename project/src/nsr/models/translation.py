"""Translation-layer boundary types shared by System 1 and System 2.

These types form the contract between the neural LLM component and the symbolic
ACT-R controller, exchanged through the Translation_Layer:

- :class:`CandidateStep` is the structured output produced by the LLM / constrained
  decoder and consumed by forward translation (:meth:`TranslationLayer.to_symbolic`).
- :class:`PromptContext` is the LLM-facing context produced by backward translation
  (:meth:`TranslationLayer.to_context`) and consumed by the LLM component.
- :class:`Untranslatable` represents the outcome of a forward translation that could
  not be converted into a :class:`~nsr.models.reasoning.SymbolicRepresentation`.

They live in ``nsr.models`` so both the Translation_Layer and the LLM component can
depend on them without creating an import cycle between the two components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # avoid an import cycle; only needed for type hints
    from .trace import ErrorRecord


@dataclass
class CandidateStep:
    """A single structured Reasoning_Step candidate emitted by the LLM (System 1).

    The constrained decoder guarantees ``structured`` conforms to the configured
    output format; forward translation maps it into a machine-checkable
    :class:`~nsr.models.reasoning.SymbolicRepresentation`.
    """

    raw_text: str
    """The original, unparsed LLM step text."""

    structured: dict[str, Any] = field(default_factory=dict)
    """Parsed structured fields from constrained decoding (e.g. ``logic_form``)."""

    sub_goal: Optional[str] = None
    """The active sub-goal this candidate step addresses, when known."""


@dataclass
class PromptContext:
    """LLM-facing prompt context produced by backward translation.

    Carries the symbolic state (active goal, current sub-goal, partial problem
    representation, and established conclusions) the LLM needs to generate the next
    Reasoning_Step, along with a rendered ``prompt_text`` ready for the LLM.
    """

    goal_description: str
    active_sub_goal: Optional[str] = None
    partial_representation: Optional[str] = None
    established_conclusions: list[str] = field(default_factory=list)
    prompt_text: str = ""


@dataclass
class Untranslatable:
    """The outcome of a forward translation that produced no Symbolic_Representation.

    Returned by :meth:`TranslationLayer.to_symbolic` when a candidate step cannot be
    converted into the machine-checkable encoding. The Repair process consumes this
    outcome (see Requirement 5.3).
    """

    step: CandidateStep
    reason: str


class BackTranslationError(Exception):
    """Raised when symbolic state cannot be converted into LLM prompt context.

    Signals a failed backward translation (:meth:`TranslationLayer.to_context`). The
    exception carries the human-readable ``reason`` and an :class:`ErrorRecord` whose
    ``failed_component`` names the Translation_Layer, so the orchestrator can surface a
    ``component-error`` termination while preserving the Proof_Trace (Requirement 5.5).
    """

    def __init__(self, reason: str, error_record: "ErrorRecord") -> None:
        super().__init__(reason)
        self.reason = reason
        self.error_record = error_record
