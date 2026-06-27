"""Core reasoning types: goals, symbolic representations, and production rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubGoal:
    """A single decomposed step toward satisfying the active :class:`Goal`."""

    description: str
    satisfied: bool = False


@dataclass
class Goal:
    """The active goal held in the Goal_Buffer, with its ordered sub-goals."""

    description: str
    sub_goals: list["SubGoal"] = field(default_factory=list)
    satisfied: bool = False


@dataclass
class SymbolicRepresentation:
    """A structured, machine-checkable encoding of a single Reasoning_Step."""

    logic_form: str
    """Machine-checkable encoding (for example, a logical form)."""

    predicates: dict[str, Any] = field(default_factory=dict)
    """Structured fields parsed from the step."""

    source_text: str = ""
    """The original LLM step text the representation was derived from."""


@dataclass
class ProductionRule:
    """An IF-THEN production rule stored in Procedural_Memory."""

    rule_id: str
    condition: str
    """IF pattern over working-memory state."""

    action: str
    """THEN effect applied when the condition matches."""
