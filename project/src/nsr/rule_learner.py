"""Rule Learner (Adaptive Rule Learning, Req 14).

This module implements the optional Rule Learner subsystem described in the design's
*Rule Learner (Adaptive Rule Learning)* section. The Rule Learner induces new symbolic
production rules from successful (``goal-satisfied``) reasoning traces and — in later
tasks — corroborates and promotes the well-supported, non-contradicting ones into
Procedural_Memory as ``Learned_Rules``.

It runs **after** a query terminates with ``goal-satisfied``, never on the per-step
critical path, so the disabled path remains identical to Requirements 1-13 (Req 14.10).

The induction implemented here (Task 17.1) deliberately reuses the *exact* IF/THEN
string term-decomposition machinery already used by the rest of the system —
:meth:`ACTRController._condition_terms` and :meth:`ValidationEngine._clause_terms` — so
that every induced :class:`CandidateRule` is immediately evaluable by the unchanged
``ACTRController`` and ``ValidationEngine`` (Req 14.1, 14.2).

Generalization (Req 14.1, 14.2):

- The **IF** (condition) pattern is built from the step representation's *stable
  predicate terms* — the predicate field names. These survive across structurally
  similar traces while the concrete, instance-specific predicate *values* are dropped.
- The **THEN** (action) pattern is built from the accepted conclusion's *stable terms* —
  the identifier-like tokens of the ``logic_form`` (predicate/relation/variable names),
  dropping bare numeric literals which are instance-specific.

Both halves are drawn directly from the representation's searchable text (its
``logic_form``, ``source_text`` and ``predicates``), so a candidate induced from a step
is guaranteed to *accept* that step when evaluated as the sole rule by
:class:`ValidationEngine` — the basis for the later corroboration and contradiction
checks (Tasks 17.4).
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

from .models import (
    CandidateRule,
    DiscardedCandidate,
    LearnedRule,
    LearnedRuleStore,
    ProductionRule,
    PromotionDecision,
    PromotionResult,
    RuleOrigin,
    RuleProvenance,
    SymbolicRepresentation,
)
from .models.enums import TerminationReason, ValidationStatus
from .models.trace import ProofStep, ProofTrace
from .validation_engine import ValidationEngine

if TYPE_CHECKING:  # pragma: no cover - typing only
    import os

    from .models import ErrorRecord, RunRecord
    from .reproducibility import ReproducibilityManager

# Identifier-like tokens (predicate/relation/variable names). Bare numeric literals are
# intentionally excluded: they are instance-specific and must be generalized away.
_IDENTIFIER_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class RuleLearner:
    """Induces, corroborates, and promotes learned production rules (Req 14).

    Only :meth:`induce` is implemented in this task; :meth:`corroborate`,
    :meth:`promote`, and :meth:`contradicts` are declared with their final signatures so
    later tasks (17.4) can fill them in without changing the constructor or call sites.

    The learner is a pure, ordering-stable function of its inputs and the supplied seed:
    induction consults no hidden mutable state and uses no randomness, so the same trace
    under the same seed always yields the same candidates (Req 14.6).
    """

    def __init__(
        self,
        store: LearnedRuleStore,
        validation: ValidationEngine,
        *,
        corroboration_threshold: int = 2,
        max_learned_rules: int = 64,
        seed: Optional[int] = None,
        reproducibility: "Optional[ReproducibilityManager]" = None,
    ) -> None:
        """Construct a Rule Learner.

        Args:
            store: The versioned, persistable store of candidates and learned rules.
            validation: The shared Validation Engine; its IF/THEN semantics decide both
                evaluability of induced candidates and (later) the contradiction check.
            corroboration_threshold: Minimum number of independent successful traces a
                candidate must appear in before promotion (Req 14.3, default 2).
            max_learned_rules: The cap on promoted learned rules (Req 14.9).
            seed: The effective seed governing deterministic tie-breaking (Req 14.6); it
                is recorded on every induced candidate's provenance.
            reproducibility: Optional Reproducibility Manager; when supplied, a seed hook
                is registered so the single effective seed governs the learner's
                deterministic tie-breaking (Req 14.6).
        """
        self._store = store
        self._validation = validation
        self._corroboration_threshold = corroboration_threshold
        self._max_learned_rules = max_learned_rules
        self._seed = seed
        if reproducibility is not None:
            self.register_seed_hook(reproducibility)

    # ---------------------------------------------------------------- seeding

    def register_seed_hook(self, reproducibility: "ReproducibilityManager") -> None:
        """Register a seed hook so the single effective seed governs determinism.

        The induction/corroboration/promotion pipeline uses no randomness — its ordering
        is fixed by the canonical IF/THEN key and provenance trace id — but the effective
        seed is still recorded on induced candidates and the run record (Req 14.6). The
        hook keeps the learner's recorded ``induction_seed`` in lock-step with the seed
        the Reproducibility Manager applies, even if it is applied after construction.
        """

        def _hook(seed: int) -> None:
            self._seed = seed

        reproducibility.register_seed_hook(_hook)

    # --------------------------------------------------------------- induction

    def induce(self, trace: ProofTrace, *, trace_id: str) -> list[CandidateRule]:
        """Generalize the accepted steps of a goal-satisfied trace into candidates.

        Implements Req 14.1 and 14.2. Each accepted (or accepted-after-repair)
        :class:`ProofStep` is generalized into a conjunctive IF/THEN
        :class:`ProductionRule` using the same term decomposition the controller and
        validator use, so the candidate is immediately evaluable by both. Steps that
        generalize to the same canonical IF/THEN term-set collapse into a single
        candidate whose provenance lists every contributing step.

        Args:
            trace: The terminated Proof_Trace to learn from.
            trace_id: The identifier of the successful trace, recorded in provenance.

        Returns:
            One :class:`CandidateRule` per distinct normalized IF/THEN key, in first-seen
            order. Returns ``[]`` for a trace with no accepted steps, or whose
            ``termination_reason`` is not ``goal-satisfied`` (Req 14.1).

        Raises:
            ValueError: If ``trace`` is ``None`` or ``trace_id`` is empty.
        """
        if trace is None:
            raise ValueError("induce requires a non-None ProofTrace")
        if not trace_id:
            raise ValueError("induce requires a non-empty trace_id")

        # Req 14.1: induction only fires on goal-satisfied termination.
        if trace.termination_reason != TerminationReason.GOAL_SATISFIED:
            return []

        accepted = [
            step
            for step in trace.steps
            if self._is_accepted(step) and step.representation is not None
        ]
        if not accepted:
            return []

        # Group accepted steps by their canonical IF/THEN key so a trace that induces the
        # same generalization more than once yields a single candidate (its provenance
        # records every contributing step). First-seen ordering keeps induction stable.
        grouped: "OrderedDict[str, _Generalization]" = OrderedDict()
        for step in accepted:
            condition_terms, action_terms = self._generalize(step.representation)
            key = self._normalized_key(condition_terms, action_terms)
            entry = grouped.get(key)
            if entry is None:
                grouped[key] = _Generalization(
                    condition_terms=condition_terms,
                    action_terms=action_terms,
                    step_ids=[step.sequence],
                    witnesses=[step.representation],
                )
            else:
                entry.step_ids.append(step.sequence)
                entry.witnesses.append(step.representation)

        candidates: list[CandidateRule] = []
        for key, entry in grouped.items():
            rule = ProductionRule(
                rule_id=self._candidate_rule_id(key),
                condition=self._build_clause("IF", entry.condition_terms),
                action=self._build_clause("THEN", entry.action_terms),
            )
            provenance = RuleProvenance(
                trace_ids=[trace_id],
                step_ids=list(entry.step_ids),
                induction_seed=self._seed,
            )
            candidates.append(
                CandidateRule(
                    rule=rule,
                    provenance=provenance,
                    corroboration_count=1,
                    normalized_key=key,
                    witnesses=list(entry.witnesses),
                )
            )
        return candidates

    # --------------------------------------------------- later-task extensions

    def corroborate(self, candidates: list[CandidateRule]) -> None:
        """Merge candidates into the store, incrementing corroboration counts at most
        once per distinct source trace id (Req 14.3).

        Equivalence is by ``normalized_key`` (the canonical IF/THEN term-set key), so two
        traces that produce the same generalization corroborate a single stored
        candidate. For an already-stored key, each *new* distinct provenance trace id
        bumps the corroboration count by one (a trace already counted never bumps it
        again); the merged candidate accumulates the union of step ids and the witness
        representations from every corroborating trace so the later contradiction check
        (Req 14.4) can be decided over all of them. The store is mutated in place.

        Args:
            candidates: Candidates (typically the output of :meth:`induce` for one trace)
                to merge into the store.
        """
        for candidate in candidates:
            key = candidate.normalized_key or self._normalized_key(
                self._validation._clause_terms(  # reuse the shared decomposition
                    candidate.rule.condition, leading_keyword="IF"
                ),
                self._validation._clause_terms(
                    candidate.rule.action, leading_keyword="THEN"
                ),
            )
            existing = self._store.candidates.get(key)
            if existing is None:
                self._store.candidates[key] = self._clone_candidate(candidate, key)
                continue

            # Merge a corroborating candidate: count each new distinct trace id once.
            # If the candidate brings no new trace id, corroboration is idempotent — the
            # trace has already been counted, so neither the count nor the accumulated
            # witnesses change.
            new_trace_ids = [
                trace_id
                for trace_id in candidate.provenance.trace_ids
                if trace_id not in existing.provenance.trace_ids
            ]
            if not new_trace_ids:
                continue
            existing.provenance.trace_ids.extend(new_trace_ids)
            for step_id in candidate.provenance.step_ids:
                if step_id not in existing.provenance.step_ids:
                    existing.provenance.step_ids.append(step_id)
            existing.witnesses.extend(candidate.witnesses)
            # Invariant: corroboration count == number of distinct corroborating traces.
            existing.corroboration_count = len(existing.provenance.trace_ids)

    def promote(self, procedural_memory: list[ProductionRule]) -> PromotionResult:
        """Promote corroborated, non-contradicting candidates up to ``max_learned_rules``
        (Req 14.3, 14.4, 14.9).

        Candidates are processed in a deterministic canonical order — by ``normalized_key``
        then provenance trace ids — so the same store and inputs always yield the same
        promotion-decision sequence (Req 14.6). For each candidate, in order:

        - below the corroboration threshold -> ``below-threshold`` (not promoted);
        - otherwise, if it contradicts any existing rule (seeded, already-learned, or a
          rule promoted earlier in this pass) -> discarded with the conflicting rule id
          recorded (``contradiction``, Req 14.4);
        - otherwise, if the learned-rule cap has been reached -> ``cap-reached`` and
          promotion stops for it (Req 14.9);
        - otherwise it is promoted to a :class:`LearnedRule` (``corroborated``) and made
          visible to the contradiction check for subsequent candidates.

        Args:
            procedural_memory: The existing rules (seeded + any prior learned) a promoted
                candidate must not contradict.

        Returns:
            A :class:`PromotionResult` carrying the promoted learned rules, discarded
            candidates, the full ordered decision log, and whether the cap was reached.
        """
        result = PromotionResult()

        # Existing rules a candidate must not contradict: the supplied Procedural_Memory
        # plus rules already promoted in the store, growing as this pass promotes more.
        existing_rules: list[ProductionRule] = list(procedural_memory or [])
        existing_rules.extend(lr.rule for lr in self._store.learned_rules)
        promoted_count = len(self._store.learned_rules)

        for candidate in self._canonical_order(self._store.candidates.values()):
            key = candidate.normalized_key

            if candidate.corroboration_count < self._corroboration_threshold:
                result.decisions.append(
                    PromotionDecision(
                        normalized_key=key, promoted=False, reason="below-threshold"
                    )
                )
                continue

            conflicting_rule_id = self._first_conflict(candidate, existing_rules)
            if conflicting_rule_id is not None:
                result.discarded.append(
                    DiscardedCandidate(
                        candidate=candidate,
                        conflicting_rule_id=conflicting_rule_id,
                    )
                )
                result.decisions.append(
                    PromotionDecision(
                        normalized_key=key,
                        promoted=False,
                        reason="contradiction",
                        conflicting_rule_id=conflicting_rule_id,
                    )
                )
                continue

            if promoted_count >= self._max_learned_rules:
                result.cap_reached = True
                result.decisions.append(
                    PromotionDecision(
                        normalized_key=key, promoted=False, reason="cap-reached"
                    )
                )
                continue

            learned = LearnedRule(
                rule=candidate.rule,
                provenance=candidate.provenance,
                origin=RuleOrigin.LEARNED,
            )
            self._store.learned_rules.append(learned)
            existing_rules.append(candidate.rule)
            promoted_count += 1
            result.promoted.append(learned)
            result.decisions.append(
                PromotionDecision(
                    normalized_key=key, promoted=True, reason="corroborated"
                )
            )

        return result

    def contradicts(self, candidate: CandidateRule, existing: ProductionRule) -> bool:
        """True iff some witness representation is accepted by ``candidate`` alone but
        rejected by ``existing`` alone, per ValidationEngine semantics (Req 14.4).

        The witness set is the candidate's accumulated witness representations (its own
        provenance steps plus every corroborating trace's accepted steps merged in by
        :meth:`corroborate`). A contradiction exists when any witness ``s`` satisfies
        ``validate(s, [candidate.rule]).accepted`` while ``validate(s, [existing]).rejected``.
        """
        for witness in candidate.witnesses:
            if witness is None:
                continue
            accepted_by_candidate = self._validation.validate(
                witness, [candidate.rule]
            ).accepted
            if not accepted_by_candidate:
                continue
            rejected_by_existing = self._validation.validate(
                witness, [existing]
            ).rejected
            if rejected_by_existing:
                return True
        return False

    # --------------------------------------------------- run-record & persist

    def record_run(self, run_record: "RunRecord", result: PromotionResult) -> None:
        """Record the learned-rule set, induction seed, threshold, and promotion
        decisions onto ``run_record`` for reproducibility (Req 14.6).
        """
        run_record.learned_rules = list(self._store.learned_rules)
        run_record.induction_seed = self._seed
        run_record.corroboration_threshold = self._corroboration_threshold
        run_record.promotion_decisions = list(result.decisions)

    def persist_store(
        self,
        reproducibility: "ReproducibilityManager",
        output_path: "str | os.PathLike[str]",
    ) -> "Optional[ErrorRecord]":
        """Persist the versioned Learned_Rule_Store durably via the Reproducibility
        Manager (Req 14.7), returning an :class:`ErrorRecord` on failure rather than
        raising.
        """
        return reproducibility.persist_learned_rule_store(self._store, output_path)

    @property
    def store(self) -> LearnedRuleStore:
        """The Learned_Rule_Store this learner reads and writes."""
        return self._store

    # ----------------------------------------------- corroboration/promotion helpers

    @classmethod
    def _clone_candidate(cls, candidate: CandidateRule, key: str) -> CandidateRule:
        """A fresh candidate copy owned by the store, so merging never mutates the input.

        The corroboration count is normalized to the number of distinct provenance trace
        ids, establishing the store invariant ``count == #distinct corroborating traces``.
        """
        distinct_trace_ids: list[str] = []
        for trace_id in candidate.provenance.trace_ids:
            if trace_id not in distinct_trace_ids:
                distinct_trace_ids.append(trace_id)
        provenance = RuleProvenance(
            trace_ids=distinct_trace_ids,
            step_ids=list(candidate.provenance.step_ids),
            induction_seed=candidate.provenance.induction_seed,
        )
        return CandidateRule(
            rule=candidate.rule,
            provenance=provenance,
            corroboration_count=len(distinct_trace_ids),
            normalized_key=key,
            witnesses=list(candidate.witnesses),
        )

    @staticmethod
    def _canonical_order(
        candidates: "object",
    ) -> list[CandidateRule]:
        """Order candidates deterministically by normalized key then provenance traces."""
        return sorted(
            candidates,
            key=lambda c: (
                c.normalized_key,
                tuple(sorted(c.provenance.trace_ids)),
            ),
        )

    def _first_conflict(
        self, candidate: CandidateRule, existing_rules: list[ProductionRule]
    ) -> Optional[str]:
        """The id of the first existing rule the candidate contradicts, else ``None``."""
        for rule in existing_rules:
            if self.contradicts(candidate, rule):
                return rule.rule_id
        return None

    # ------------------------------------------------- generalization helpers

    @staticmethod
    def _is_accepted(step: ProofStep) -> bool:
        """An accepted (or accepted-after-repair) step contributes to induction.

        A ``REPAIRED`` status denotes a step that was accepted only after repair; both it
        and a directly ``ACCEPTED`` step are valid sources for a learned generalization
        (Req 14.1). Rejected steps are excluded.
        """
        return step.status in (ValidationStatus.ACCEPTED, ValidationStatus.REPAIRED)

    @classmethod
    def _generalize(
        cls, rep: SymbolicRepresentation
    ) -> tuple[list[str], list[str]]:
        """Decompose a representation into (condition_terms, action_terms).

        The condition (IF) keeps the *stable predicate terms* — predicate field names —
        and drops their instance-specific values. The action (THEN) keeps the stable,
        identifier-like tokens of the conclusion ``logic_form`` and drops bare numeric
        literals. Both sets are drawn from text the :class:`ValidationEngine` searches,
        so the resulting candidate accepts the step it was induced from.
        """
        return cls._predicate_terms(rep), cls._conclusion_terms(rep)

    @staticmethod
    def _predicate_terms(rep: SymbolicRepresentation) -> list[str]:
        """Stable IF terms: the predicate field names, trimmed, lower-cased, sorted.

        Predicate *keys* are structural and recur across structurally similar traces,
        whereas their values are instance-specific; using only the keys is the
        generalization. They appear in the validator's haystack via ``str(predicates)``.
        """
        terms = {
            str(key).strip().lower()
            for key in (rep.predicates or {}).keys()
            if str(key).strip()
        }
        return sorted(terms)

    @staticmethod
    def _conclusion_terms(rep: SymbolicRepresentation) -> list[str]:
        """Stable THEN terms: identifier-like tokens of the conclusion ``logic_form``.

        Bare numeric literals are excluded because they are instance-specific; the
        remaining predicate/relation/variable names appear verbatim (lower-cased) in the
        validator's haystack, so the candidate is satisfied by its source step.
        """
        tokens = _IDENTIFIER_TOKEN.findall(rep.logic_form or "")
        return sorted({token.lower() for token in tokens})

    @staticmethod
    def _build_clause(keyword: str, terms: list[str]) -> str:
        """Compose an ``IF``/``THEN`` clause from conjunctive terms.

        Mirrors the form the controller and validator parse: a leading keyword followed
        by terms joined with ``AND``. With no terms the clause is the bare keyword, which
        both decompose to an empty term list (matching/satisfied unconditionally).
        """
        if not terms:
            return keyword
        return f"{keyword} " + " AND ".join(terms)

    @staticmethod
    def _normalized_key(condition_terms: list[str], action_terms: list[str]) -> str:
        """Canonical, order-independent IF/THEN term-set key for equivalence.

        Two traces that produce the same generalization (same condition and action term
        sets, regardless of order or duplication) yield the same key, so they corroborate
        a single candidate in later tasks (Req 14.3).
        """
        condition = ",".join(sorted(set(condition_terms)))
        action = ",".join(sorted(set(action_terms)))
        return f"IF[{condition}]=>THEN[{action}]"

    @staticmethod
    def _candidate_rule_id(normalized_key: str) -> str:
        """A deterministic candidate rule id derived from its canonical key."""
        return f"learned::{normalized_key}"


class _Generalization:
    """Mutable accumulator for steps sharing one normalized IF/THEN key during induction."""

    __slots__ = ("condition_terms", "action_terms", "step_ids", "witnesses")

    def __init__(
        self,
        condition_terms: list[str],
        action_terms: list[str],
        step_ids: list[int],
        witnesses: list[SymbolicRepresentation],
    ) -> None:
        self.condition_terms = condition_terms
        self.action_terms = action_terms
        self.step_ids = step_ids
        self.witnesses = witnesses
