"""Enumerations for the Neuro-Symbolic System-2 Reasoning Architecture.

These mirror the enums defined in the design's Data Models section. Each value is a
``str`` subclass so that the enums serialize directly to their string form, which keeps
the Proof_Trace and run records losslessly machine-readable.
"""

from __future__ import annotations

from enum import Enum


class ValidationStatus(str, Enum):
    """Outcome of validating a single Reasoning_Step."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REPAIRED = "repaired"


class TerminationReason(str, Enum):
    """The single reason a reasoning cycle terminated, recorded in the Proof_Trace."""

    GOAL_SATISFIED = "goal-satisfied"
    CYCLE_LIMIT_REACHED = "cycle-limit-reached"
    CONSTRAINT_UNSATISFIED = "constraint-unsatisfied"
    REPAIR_EXHAUSTED = "repair-exhausted"
    COMPONENT_ERROR = "component-error"


class Domain(str, Enum):
    """The six benchmark domains an evaluated item may belong to."""

    MATH = "mathematical-reasoning"
    COMMONSENSE = "commonsense-reasoning"
    MULTI_HOP = "multi-hop-reasoning"
    SCIENCE = "science-reasoning"
    LOGIC_PUZZLE = "logical-puzzles"
    LEGAL_QA = "legal-question-answering"
