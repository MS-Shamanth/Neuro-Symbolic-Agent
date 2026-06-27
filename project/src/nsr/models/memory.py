"""ACT-R working-memory state (System 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .reasoning import Goal, ProductionRule, SymbolicRepresentation


@dataclass
class WorkingMemoryState:
    """Snapshot of the four ACT-R buffers maintained throughout a query.

    - ``goal_buffer``: the current active goal and sub-goal.
    - ``declarative_memory``: accepted intermediate conclusions, in order.
    - ``procedural_memory``: the IF-THEN production rules.
    - ``imaginal_buffer``: the partial problem representation under construction.
    """

    goal_buffer: Goal
    declarative_memory: list[SymbolicRepresentation] = field(default_factory=list)
    procedural_memory: list[ProductionRule] = field(default_factory=list)
    imaginal_buffer: Optional[SymbolicRepresentation] = None
