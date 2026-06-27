"""Validation Engine (Task 6.1).

This module implements the step-level Validation Engine described in the design's
*Validation Engine* section. It evaluates the :class:`SymbolicRepresentation` of a
single Reasoning_Step against the applicable production rules in Procedural_Memory and
records an *accepted* or *rejected* outcome **before** the next Reasoning_Step is
generated (Req 6.1).

Acceptance/rejection semantics:

- A step is **accepted** only when every *applicable* production rule is *satisfied*
  (Req 6.2). With no applicable rules, acceptance holds vacuously.
- A step is **rejected** when one or more applicable rules are violated, and the outcome
  records *every* violated rule (Req 6.3).

Rule-satisfaction mechanism (consistent with :class:`nsr.actr_controller.ACTRController`):

Each :class:`ProductionRule` is an ``IF condition THEN action`` pattern. The engine
interprets the two halves against the searchable text of the candidate representation
(its ``logic_form``, ``source_text`` and ``predicates``):

- The **condition** (``IF`` clause) determines *applicability*. Its conjunctive terms
  (split on ``AND``, leading ``IF`` stripped) are matched against the representation, the
  same way the ACT-R controller matches a rule against working-memory state. A rule whose
  condition does not match is *not applicable* and is neither satisfied nor violated. A
  rule with an empty condition matches unconditionally and is always applicable.
- The **action** (``THEN`` clause) determines *satisfaction*. For an applicable rule, the
  step satisfies it when every action term is present in the representation. An applicable
  rule with an empty action is satisfied vacuously.

The engine is a pure function of its inputs: it consults no hidden state and uses no
randomness, so the same representation and rules always yield the same outcome.

The :class:`ValidationOutcome` returned here is structured so the Repair Coordinator
(Task 6.2) can consume rejections directly: it exposes both the violated rule ids and the
violated :class:`ProductionRule` objects, letting the repair sub-loop build a prompt that
references the exact offending constraints (Req 6.4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import ProductionRule, SymbolicRepresentation, ValidationStatus


@dataclass(frozen=True)
class RuleEvaluation:
    """The per-rule result of validating a step against a single production rule.

    A rule is *applicable* when its ``IF`` condition matches the representation; an
    applicable rule is *satisfied* when its ``THEN`` action also holds. A non-applicable
    rule is reported with ``applicable=False`` and ``satisfied=True`` (it cannot be
    violated by a step it does not govern).
    """

    rule_id: str
    applicable: bool
    satisfied: bool


@dataclass(frozen=True)
class ValidationOutcome:
    """The accept/reject result of validating one Reasoning_Step (Req 6.1-6.3).

    Attributes:
        status: ``ACCEPTED`` when every applicable rule is satisfied, otherwise
            ``REJECTED`` (Req 6.2, 6.3).
        representation: The symbolic representation that was validated.
        applicable_rule_ids: Ids of the rules whose condition matched the step, in input
            order.
        violated_rule_ids: Ids of every applicable rule the step failed to satisfy, in
            input order. Empty when accepted (Req 6.3).
        violated_rules: The :class:`ProductionRule` objects for ``violated_rule_ids``, in
            input order. Supplied so the Repair Coordinator can reference the offending
            constraints when regenerating the step (Req 6.4).
        evaluations: The per-rule evaluation record for every rule supplied, in input
            order, suitable for journaling to the Proof_Trace (Req 6.7).
    """

    status: ValidationStatus
    representation: SymbolicRepresentation
    applicable_rule_ids: list[str] = field(default_factory=list)
    violated_rule_ids: list[str] = field(default_factory=list)
    violated_rules: list[ProductionRule] = field(default_factory=list)
    evaluations: list[RuleEvaluation] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        """True when the step was accepted (all applicable rules satisfied)."""
        return self.status == ValidationStatus.ACCEPTED

    @property
    def rejected(self) -> bool:
        """True when the step was rejected (at least one applicable rule violated)."""
        return self.status == ValidationStatus.REJECTED


class ValidationEngine:
    """Evaluates a symbolic step against every applicable production rule (Req 6.1).

    The engine holds no per-query state; a single instance can validate any number of
    steps. :meth:`validate` is a pure function of its arguments.
    """

    def validate(
        self,
        rep: SymbolicRepresentation,
        rules: list[ProductionRule],
    ) -> ValidationOutcome:
        """Validate ``rep`` against ``rules`` and return an accept/reject outcome.

        Evaluates the representation against *every* applicable production rule and marks
        the step accepted only when all applicable rules are satisfied (Req 6.1, 6.2); on
        rejection, records *every* violated rule (Req 6.3).

        Args:
            rep: The symbolic representation of the candidate Reasoning_Step.
            rules: The applicable-candidate production rules (typically Procedural_Memory).
                A rule becomes *applicable* only when its condition matches ``rep``.

        Returns:
            A :class:`ValidationOutcome` whose ``status`` is ``ACCEPTED`` or ``REJECTED``.

        Raises:
            ValueError: If ``rep`` is ``None``.
        """
        if rep is None:
            raise ValueError("validate requires a non-None SymbolicRepresentation")

        haystack = self._build_haystack(rep)
        rule_list = list(rules) if rules else []

        applicable_rule_ids: list[str] = []
        violated_rule_ids: list[str] = []
        violated_rules: list[ProductionRule] = []
        evaluations: list[RuleEvaluation] = []

        for rule in rule_list:
            condition_terms = self._clause_terms(rule.condition, leading_keyword="IF")
            applicable = self._all_present(condition_terms, haystack)

            if not applicable:
                # A rule the step does not govern is neither satisfied nor violated.
                evaluations.append(
                    RuleEvaluation(
                        rule_id=rule.rule_id, applicable=False, satisfied=True
                    )
                )
                continue

            applicable_rule_ids.append(rule.rule_id)

            action_terms = self._clause_terms(rule.action, leading_keyword="THEN")
            satisfied = self._all_present(action_terms, haystack)
            evaluations.append(
                RuleEvaluation(
                    rule_id=rule.rule_id, applicable=True, satisfied=satisfied
                )
            )
            if not satisfied:
                violated_rule_ids.append(rule.rule_id)
                violated_rules.append(rule)

        # Req 6.2: accept only when every applicable rule is satisfied (vacuously true
        # when there are no applicable rules). Req 6.3: otherwise reject with every
        # violated rule recorded.
        status = (
            ValidationStatus.REJECTED
            if violated_rule_ids
            else ValidationStatus.ACCEPTED
        )

        return ValidationOutcome(
            status=status,
            representation=rep,
            applicable_rule_ids=applicable_rule_ids,
            violated_rule_ids=violated_rule_ids,
            violated_rules=violated_rules,
            evaluations=evaluations,
        )

    # ----------------------------------------------------- rule-matching helpers

    @staticmethod
    def _clause_terms(clause: str, *, leading_keyword: str) -> list[str]:
        """Decompose an ``IF``/``THEN`` clause into its conjunctive match terms.

        Mirrors :meth:`ACTRController._condition_terms`: a leading keyword (``IF`` or
        ``THEN``) is stripped and the remainder is split on the ``AND`` connective
        (case-insensitive). Each clause is trimmed and lower-cased; empty clauses are
        dropped. A clause with no remaining terms (an empty string or a bare keyword)
        yields an empty list, which is treated as matching unconditionally.
        """
        text = re.sub(
            rf"^\s*{leading_keyword}\b\s*",
            "",
            clause or "",
            flags=re.IGNORECASE,
        )
        clauses = re.split(r"\bAND\b", text, flags=re.IGNORECASE)
        return [c.strip().lower() for c in clauses if c.strip()]

    @staticmethod
    def _all_present(terms: list[str], haystack: str) -> bool:
        """True when every term appears in the representation haystack.

        An empty term list is vacuously present, so an empty condition matches every step
        (always applicable) and an empty action is always satisfied.
        """
        return all(term in haystack for term in terms)

    @staticmethod
    def _build_haystack(rep: SymbolicRepresentation) -> str:
        """Build the searchable, lower-cased text of a symbolic representation.

        Spans the representation's ``logic_form``, ``source_text`` and ``predicates`` so
        that both condition and action terms can be matched against the step's content,
        consistent with how the ACT-R controller builds its state haystack.
        """
        return " ".join(
            [
                rep.logic_form or "",
                rep.source_text or "",
                str(rep.predicates),
            ]
        ).lower()
