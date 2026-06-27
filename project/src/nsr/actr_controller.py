"""ACT-R Controller and working-memory buffers (System 2).

This module implements the deliberate "System 2" controller described in the design's
*ACT-R Controller and Working-Memory Buffers* section. It maintains the four ACT-R
buffers for the lifetime of a single query and integrates accepted reasoning steps into
working memory.

Task 3.1 scope (buffer maintenance and accepted-step integration):

- Maintain ``Goal_Buffer``, ``Declarative_Memory``, ``Procedural_Memory``, and
  ``Imaginal_Buffer`` for the lifetime of a query (Req 4.1).
- On acceptance, append the resulting conclusion as a *distinct* ``Declarative_Memory``
  entry and replace the ``Imaginal_Buffer`` with a representation reflecting the
  accepted step (Req 4.2, 4.5).
- Retain all previously accepted conclusions, in order, until the query terminates
  (Req 4.4).

Task 3.2 scope (sub-goal advancement and deterministic rule selection):

- Advance the Goal_Buffer to the next unmet sub-goal; mark the active goal satisfied
  when no unmet sub-goal remains (Req 4.3, 4.7).
- When multiple production rules match the current state, select exactly one rule
  deterministically using the configured conflict-resolution policy (Req 4.6).
- When no production rule matches the current state, return a ``NoRuleMatched`` outcome
  so the caller can record ``no-rule-matched`` in the Proof_Trace and route the state to
  the repair process (Req 4.8).

Rule selection is a pure function of the supplied :class:`WorkingMemoryState` and the
controller's configured policy: it never consults hidden mutable state and never uses
randomness, so the same state under the same policy (and any seed) always selects the
same rule id. This determinism is what backs Property 3 in Task 3.3.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Optional

from .config_manager import ALLOWED_CONFLICT_POLICIES
from .models import (
    Goal,
    ProductionRule,
    SubGoal,
    SymbolicRepresentation,
    WorkingMemoryState,
)


@dataclass
class NoRuleMatched:
    """Outcome of :meth:`ACTRController.select_rule` when no rule matches the state.

    Returned (rather than raised) so the orchestrator can record a ``no-rule-matched``
    outcome in the Proof_Trace and route the current working-memory state to the repair
    process (Req 4.8). ``route_to_repair`` is always ``True`` and is exposed explicitly
    so callers can branch on it without special-casing the type.
    """

    state: WorkingMemoryState
    reason: str = "no production rule matched the current working-memory state"
    route_to_repair: bool = True


@dataclass
class _MatchInfo:
    """Internal record describing a rule that matched the current state.

    Carries the signals the conflict-resolution policies rank on: the rule's
    ``position`` in Procedural_Memory (priority), the number of matched ``specificity``
    terms, and a ``recency`` score derived from Declarative_Memory ordering.
    """

    rule: ProductionRule
    position: int
    specificity: int
    recency: int


class ACTRController:
    """Symbolic cognitive controller maintaining the four ACT-R buffers.

    A controller instance is scoped to a single query: call :meth:`initialize` to seed
    the Goal_Buffer (and optionally the Procedural_Memory rules), then call
    :meth:`integrate_accepted` for each accepted Reasoning_Step. The buffers persist for
    the lifetime of the instance, which represents the lifetime of the query.
    """

    def __init__(
        self,
        conflict_resolution_policy: str = "priority",
        seed: Optional[int] = None,
    ) -> None:
        # Buffers are created lazily by ``initialize``. Until then the controller holds
        # no goal and accessing state is an error.
        self._goal_buffer: Optional[Goal] = None
        self._declarative_memory: list[SymbolicRepresentation] = []
        self._procedural_memory: list[ProductionRule] = []
        self._imaginal_buffer: Optional[SymbolicRepresentation] = None
        self._initialized: bool = False

        # Conflict-resolution policy governs deterministic rule selection (Req 4.6).
        if conflict_resolution_policy not in ALLOWED_CONFLICT_POLICIES:
            permitted = ", ".join(sorted(ALLOWED_CONFLICT_POLICIES))
            raise ValueError(
                f"unknown conflict_resolution_policy {conflict_resolution_policy!r}; "
                f"expected one of {{{permitted}}}"
            )
        self._policy = conflict_resolution_policy
        # The seed is retained for API/reproducibility symmetry with the rest of the
        # system. Rule selection is fully deterministic and does not consume randomness,
        # so the same state under the same policy selects the same rule for any seed.
        self._seed = seed

    # ------------------------------------------------------------------ lifecycle

    def initialize(
        self,
        goal: Goal,
        procedural_memory: Optional[list[ProductionRule]] = None,
    ) -> None:
        """Initialize the working-memory buffers for a new query (Req 4.1).

        Seeds the Goal_Buffer with ``goal`` and the Procedural_Memory with the supplied
        production rules. Declarative_Memory starts empty and the Imaginal_Buffer starts
        cleared; both are populated as steps are accepted.

        Args:
            goal: The active goal to hold in the Goal_Buffer.
            procedural_memory: The IF-THEN production rules available for this query.
                Defaults to an empty list. The controller stores a shallow copy so the
                caller's list is not mutated as state evolves.
        """
        if goal is None:  # defensive: a query must always carry a goal
            raise ValueError("ACTRController.initialize requires a non-None goal")

        self._goal_buffer = goal
        self._declarative_memory = []
        self._procedural_memory = list(procedural_memory) if procedural_memory else []
        self._imaginal_buffer = None
        self._initialized = True

    # ------------------------------------------------------------ accepted steps

    def integrate_accepted(self, step: SymbolicRepresentation) -> None:
        """Integrate an accepted Reasoning_Step into working memory.

        On acceptance the controller:

        - appends the resulting intermediate conclusion as a *distinct* entry in
          Declarative_Memory (Req 4.2), preserving the order of all prior conclusions
          (Req 4.4); and
        - replaces the Imaginal_Buffer with a partial problem representation that
          reflects the accepted step (Req 4.5).

        Each Declarative_Memory entry is a distinct object (a deep copy of ``step``) so
        that later mutation of the Imaginal_Buffer or of the caller's reference can never
        alter a previously recorded conclusion.

        Args:
            step: The symbolic representation of the accepted Reasoning_Step.

        Raises:
            RuntimeError: If the controller has not been initialized for a query.
            ValueError: If ``step`` is ``None``.
        """
        self._require_initialized()
        if step is None:
            raise ValueError("integrate_accepted requires a non-None step")

        # Store a distinct, independent entry so the Declarative_Memory record is immune
        # to later mutation of the Imaginal_Buffer or the caller's object.
        conclusion = copy.deepcopy(step)
        self._declarative_memory.append(conclusion)

        # Replace the Imaginal_Buffer with a representation reflecting the accepted step.
        # A separate copy keeps the buffer independent from the Declarative_Memory entry.
        self._imaginal_buffer = copy.deepcopy(step)

    # -------------------------------------------------------- sub-goal management

    def active_sub_goal(self) -> Optional[SubGoal]:
        """Return the current active sub-goal, or ``None`` when none is pending.

        The active sub-goal is the first sub-goal in the Goal_Buffer that is not yet
        satisfied. When every sub-goal is satisfied (or the goal has no sub-goals) there
        is no active sub-goal and ``None`` is returned. The returned object is a copy, so
        mutating it does not affect the controller's buffers.
        """
        self._require_initialized()
        assert self._goal_buffer is not None
        for sub_goal in self._goal_buffer.sub_goals:
            if not sub_goal.satisfied:
                return copy.deepcopy(sub_goal)
        return None

    def advance_sub_goal(self) -> Optional[SubGoal]:
        """Satisfy the active sub-goal and advance the Goal_Buffer.

        Marks the current active (first unmet) sub-goal as satisfied, then:

        - if at least one unmet sub-goal remains, advances the Goal_Buffer to that next
          unmet sub-goal and returns it (Req 4.3); or
        - if no unmet sub-goal remains, marks the active goal in the Goal_Buffer as
          satisfied and returns ``None`` (Req 4.7).

        A goal with no sub-goals, or whose sub-goals are already all satisfied, is
        treated as having no unmet sub-goal remaining and is marked satisfied.

        Returns:
            The next active sub-goal after advancement, or ``None`` when the goal is now
            satisfied.

        Raises:
            RuntimeError: If the controller has not been initialized for a query.
        """
        self._require_initialized()
        assert self._goal_buffer is not None

        # Satisfy the current active sub-goal (the first unmet one), if any.
        for sub_goal in self._goal_buffer.sub_goals:
            if not sub_goal.satisfied:
                sub_goal.satisfied = True
                break

        # Determine the next unmet sub-goal after this advancement.
        for sub_goal in self._goal_buffer.sub_goals:
            if not sub_goal.satisfied:
                # Req 4.3: an unmet sub-goal remains -> the Goal_Buffer advances to it.
                return copy.deepcopy(sub_goal)

        # Req 4.7: no unmet sub-goal remains -> mark the active goal satisfied.
        self._goal_buffer.satisfied = True
        return None

    # ----------------------------------------------------------- rule selection

    def select_rule(
        self, state: WorkingMemoryState
    ) -> "ProductionRule | NoRuleMatched":
        """Select exactly one production rule for ``state`` (Req 4.6, 4.8).

        Every rule in ``state.procedural_memory`` is tested against the current
        working-memory state. When one or more rules match, exactly one is chosen
        deterministically using the controller's configured conflict-resolution policy
        (``priority``, ``specificity`` or ``recency``). When no rule matches, a
        :class:`NoRuleMatched` outcome is returned so the caller can record
        ``no-rule-matched`` and route the state to repair (Req 4.8).

        Selection is a pure function of ``state`` and the configured policy: it uses no
        randomness and no hidden mutable state, so repeated calls with an equal state
        always return the same rule id (Property 3).
        """
        self._require_initialized()
        if state is None:
            raise ValueError("select_rule requires a non-None WorkingMemoryState")

        haystack, declarative_texts = self._build_haystack(state)

        matches: list[_MatchInfo] = []
        for position, rule in enumerate(state.procedural_memory):
            terms = self._condition_terms(rule.condition)
            if self._rule_matches(terms, haystack):
                matches.append(
                    _MatchInfo(
                        rule=rule,
                        position=position,
                        specificity=len(terms),
                        recency=self._recency_score(terms, declarative_texts),
                    )
                )

        if not matches:
            # Req 4.8: no rule matched -> route the current state to repair.
            return NoRuleMatched(state=copy.deepcopy(state))

        winner = min(matches, key=self._policy_sort_key)
        return winner.rule

    # ----------------------------------------------------- rule-matching helpers

    @staticmethod
    def _condition_terms(condition: str) -> list[str]:
        """Decompose a rule condition into its conjunctive match terms.

        A leading ``IF`` keyword is stripped, and the remainder is split on the ``AND``
        connective (case-insensitive). Each resulting clause is trimmed; empty clauses
        are dropped. A condition with no remaining clauses (for example an empty string
        or a bare ``IF``) yields an empty list, which matches unconditionally and acts as
        a fallback/default rule.
        """
        text = re.sub(r"^\s*IF\b\s*", "", condition or "", flags=re.IGNORECASE)
        clauses = re.split(r"\bAND\b", text, flags=re.IGNORECASE)
        return [clause.strip().lower() for clause in clauses if clause.strip()]

    @staticmethod
    def _rule_matches(terms: list[str], haystack: str) -> bool:
        """A rule matches when every one of its terms appears in the state haystack."""
        return all(term in haystack for term in terms)

    @staticmethod
    def _build_haystack(state: WorkingMemoryState) -> tuple[str, list[str]]:
        """Build the searchable, lower-cased text of the working-memory state.

        Returns a ``(global_haystack, declarative_texts)`` pair where ``global_haystack``
        spans the goal, sub-goals, declarative memory, and imaginal buffer, and
        ``declarative_texts`` holds the per-entry text of Declarative_Memory in order
        (used to score recency).
        """
        parts: list[str] = []
        goal = state.goal_buffer
        if goal is not None:
            parts.append(goal.description or "")
            for sub_goal in goal.sub_goals:
                parts.append(sub_goal.description or "")

        declarative_texts: list[str] = []
        for entry in state.declarative_memory:
            entry_text = " ".join(
                [entry.logic_form or "", entry.source_text or "", str(entry.predicates)]
            ).lower()
            declarative_texts.append(entry_text)
            parts.append(entry_text)

        imaginal = state.imaginal_buffer
        if imaginal is not None:
            parts.append(
                " ".join(
                    [
                        imaginal.logic_form or "",
                        imaginal.source_text or "",
                        str(imaginal.predicates),
                    ]
                )
            )

        return " ".join(parts).lower(), declarative_texts

    @staticmethod
    def _recency_score(terms: list[str], declarative_texts: list[str]) -> int:
        """Highest Declarative_Memory index whose entry contains any rule term.

        A higher score means the rule is supported by a more recently accepted
        conclusion (Declarative_Memory is ordered oldest-to-newest). A rule that matches
        only via the goal or imaginal buffer (no declarative support) scores ``-1``.
        """
        best = -1
        for index, entry_text in enumerate(declarative_texts):
            if any(term in entry_text for term in terms):
                best = index
        return best

    def _policy_sort_key(self, match: "_MatchInfo") -> tuple:
        """Deterministic ordering key; the minimum match is the selected rule.

        - ``priority``: earlier position in Procedural_Memory wins (highest priority).
        - ``specificity``: more match terms (more specific condition) wins.
        - ``recency``: support from a more recently accepted conclusion wins.

        Every policy applies a total tie-break on ``rule_id`` then ``position`` so the
        outcome is always uniquely determined.
        """
        if self._policy == "priority":
            return (match.position, match.rule.rule_id)
        if self._policy == "specificity":
            return (-match.specificity, match.rule.rule_id, match.position)
        # recency
        return (-match.recency, match.rule.rule_id, match.position)

    # ------------------------------------------------------------------- accessors

    def state(self) -> WorkingMemoryState:
        """Return a snapshot of the four working-memory buffers.

        The returned :class:`WorkingMemoryState` is a defensive copy: mutating it does
        not affect the controller's internal buffers, and subsequent controller updates
        do not retroactively change a previously returned snapshot.
        """
        self._require_initialized()
        assert self._goal_buffer is not None  # guaranteed by _require_initialized
        return WorkingMemoryState(
            goal_buffer=copy.deepcopy(self._goal_buffer),
            declarative_memory=copy.deepcopy(self._declarative_memory),
            procedural_memory=list(self._procedural_memory),
            imaginal_buffer=copy.deepcopy(self._imaginal_buffer),
        )

    @property
    def declarative_memory(self) -> list[SymbolicRepresentation]:
        """All accepted conclusions, in acceptance order (read-only copy)."""
        self._require_initialized()
        return copy.deepcopy(self._declarative_memory)

    @property
    def imaginal_buffer(self) -> Optional[SymbolicRepresentation]:
        """The current partial problem representation (read-only copy)."""
        self._require_initialized()
        return copy.deepcopy(self._imaginal_buffer)

    @property
    def goal_buffer(self) -> Goal:
        """The active goal held in the Goal_Buffer (read-only copy)."""
        self._require_initialized()
        assert self._goal_buffer is not None
        return copy.deepcopy(self._goal_buffer)

    @property
    def procedural_memory(self) -> list[ProductionRule]:
        """The production rules available for this query (read-only copy)."""
        self._require_initialized()
        return list(self._procedural_memory)

    @property
    def conflict_resolution_policy(self) -> str:
        """The configured deterministic conflict-resolution policy (Req 4.6)."""
        return self._policy

    # -------------------------------------------------------------------- helpers

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "ACTRController has not been initialized; call initialize(goal) first"
            )
