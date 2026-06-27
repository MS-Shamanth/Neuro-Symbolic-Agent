"""Adaptive rule-learning types (Req 14).

These dataclasses reuse the existing :class:`ProductionRule` IF/THEN string form so
learned rules are evaluable by the unchanged ``ACTRController`` and ``ValidationEngine``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .reasoning import ProductionRule, SymbolicRepresentation


class RuleOrigin(str, Enum):
    """Marks whether an applied/stored rule was seeded or learned (Req 14.5)."""

    SEEDED = "seeded"
    LEARNED = "learned"


@dataclass
class RuleProvenance:
    """Where a Candidate_Rule came from (Req 14.2)."""

    trace_ids: list[str]
    """Successful traces that induced/corroborated the candidate."""

    step_ids: list[int]
    """Accepted ``ProofStep.sequence`` values the candidate generalizes."""

    induction_seed: Optional[int] = None
    """Effective seed under which the candidate was induced."""


@dataclass
class CandidateRule:
    """An induced, not-yet-promoted generalization of accepted reasoning steps."""

    rule: ProductionRule
    """IF/THEN in the same form as seeded rules."""

    provenance: RuleProvenance
    corroboration_count: int = 1
    """Distinct successful traces corroborating the candidate."""

    normalized_key: str = ""
    """Canonical IF/THEN term-set key for equivalence."""

    witnesses: list[SymbolicRepresentation] = field(default_factory=list)
    """The accepted-step representations the candidate generalizes, accumulated across
    every corroborating trace. These are the witness representations over which the
    contradiction check is decided (Req 14.4): a witness accepted by the candidate alone
    but rejected by an existing rule alone proves a contradiction. Kept in the store so
    the check is reproducible after persistence (Req 14.7)."""


@dataclass
class LearnedRule:
    """A promoted Candidate_Rule now active in Procedural_Memory (Req 14.3)."""

    rule: ProductionRule
    provenance: RuleProvenance
    origin: RuleOrigin = RuleOrigin.LEARNED


@dataclass
class DiscardedCandidate:
    """A candidate discarded for contradicting an existing rule (Req 14.4)."""

    candidate: CandidateRule
    conflicting_rule_id: str


@dataclass
class PromotionDecision:
    """One recorded promote/discard/cap-reached decision (Req 14.6)."""

    normalized_key: str
    promoted: bool
    reason: str
    """One of ``"corroborated"``, ``"below-threshold"``, ``"contradiction"``,
    ``"cap-reached"``."""

    conflicting_rule_id: Optional[str] = None


@dataclass
class PromotionResult:
    """Outcome of a promotion pass over the store (Req 14.3, 14.4, 14.9)."""

    promoted: list[LearnedRule] = field(default_factory=list)
    discarded: list[DiscardedCandidate] = field(default_factory=list)
    decisions: list[PromotionDecision] = field(default_factory=list)
    cap_reached: bool = False


@dataclass
class LearnedRuleStore:
    """Versioned, JSON-persistable store of candidates and promoted learned rules.

    Persisted durably (with provenance) via ``ReproducibilityManager`` (Req 14.7). A
    ``store_to_dict``/``store_from_dict`` pair round-trips losslessly, including the
    version identifier, mirroring ``proof_trace_export``.
    """

    version: int = 1
    candidates: dict[str, CandidateRule] = field(default_factory=dict)
    """Maps normalized key -> candidate."""

    learned_rules: list[LearnedRule] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Lossless, versioned (de)serialization for the Learned_Rule_Store (Req 14.7). #
# --------------------------------------------------------------------------- #
#
# ``store_to_dict`` / ``store_from_dict`` are a pure, lossless pair so the store can be
# persisted durably (e.g. as JSON via ``ReproducibilityManager``) and round-tripped back
# into equal dataclasses, preserving the version identifier, every candidate, its
# corroboration count, learned/seeded markers, and every provenance and witness field.
# ``store_from_dict(store_to_dict(store)) == store`` holds for any store.

_STORE_FORMAT_VERSION = 1


def _rule_to_dict(rule: ProductionRule) -> dict[str, Any]:
    return {
        "rule_id": rule.rule_id,
        "condition": rule.condition,
        "action": rule.action,
    }


def _rule_from_dict(data: dict[str, Any]) -> ProductionRule:
    return ProductionRule(
        rule_id=data["rule_id"],
        condition=data["condition"],
        action=data["action"],
    )


def _rep_to_dict(rep: Optional[SymbolicRepresentation]) -> Optional[dict[str, Any]]:
    if rep is None:
        return None
    return {
        "logic_form": rep.logic_form,
        "predicates": copy.deepcopy(rep.predicates),
        "source_text": rep.source_text,
    }


def _rep_from_dict(
    data: Optional[dict[str, Any]],
) -> Optional[SymbolicRepresentation]:
    if data is None:
        return None
    return SymbolicRepresentation(
        logic_form=data.get("logic_form", ""),
        predicates=copy.deepcopy(data.get("predicates") or {}),
        source_text=data.get("source_text", ""),
    )


def _provenance_to_dict(provenance: RuleProvenance) -> dict[str, Any]:
    return {
        "trace_ids": list(provenance.trace_ids),
        "step_ids": list(provenance.step_ids),
        "induction_seed": provenance.induction_seed,
    }


def _provenance_from_dict(data: dict[str, Any]) -> RuleProvenance:
    return RuleProvenance(
        trace_ids=list(data.get("trace_ids", [])),
        step_ids=list(data.get("step_ids", [])),
        induction_seed=data.get("induction_seed"),
    )


def _candidate_to_dict(candidate: CandidateRule) -> dict[str, Any]:
    return {
        "rule": _rule_to_dict(candidate.rule),
        "provenance": _provenance_to_dict(candidate.provenance),
        "corroboration_count": candidate.corroboration_count,
        "normalized_key": candidate.normalized_key,
        "witnesses": [_rep_to_dict(w) for w in candidate.witnesses],
    }


def _candidate_from_dict(data: dict[str, Any]) -> CandidateRule:
    return CandidateRule(
        rule=_rule_from_dict(data["rule"]),
        provenance=_provenance_from_dict(data["provenance"]),
        corroboration_count=data.get("corroboration_count", 1),
        normalized_key=data.get("normalized_key", ""),
        witnesses=[
            rep
            for rep in (_rep_from_dict(w) for w in data.get("witnesses", []))
            if rep is not None
        ],
    )


def _learned_to_dict(learned: LearnedRule) -> dict[str, Any]:
    return {
        "rule": _rule_to_dict(learned.rule),
        "provenance": _provenance_to_dict(learned.provenance),
        "origin": learned.origin.value,
    }


def _learned_from_dict(data: dict[str, Any]) -> LearnedRule:
    return LearnedRule(
        rule=_rule_from_dict(data["rule"]),
        provenance=_provenance_from_dict(data["provenance"]),
        origin=RuleOrigin(data.get("origin", RuleOrigin.LEARNED.value)),
    )


def store_to_dict(store: LearnedRuleStore) -> dict[str, Any]:
    """Serialize a :class:`LearnedRuleStore` into a plain, versioned dict (Req 14.7).

    The returned structure is JSON-friendly and lossless: feeding it to
    :func:`store_from_dict` reconstructs an equal store, including the store's own
    ``version`` and a ``format_version`` tag identifying the serialization schema.
    """
    return {
        "format_version": _STORE_FORMAT_VERSION,
        "version": store.version,
        "candidates": {
            key: _candidate_to_dict(candidate)
            for key, candidate in store.candidates.items()
        },
        "learned_rules": [_learned_to_dict(lr) for lr in store.learned_rules],
    }


def store_from_dict(data: dict[str, Any]) -> LearnedRuleStore:
    """Reconstruct a :class:`LearnedRuleStore` from :func:`store_to_dict` output.

    The inverse of :func:`store_to_dict`; ``store_from_dict(store_to_dict(s)) == s`` for
    any store ``s`` (Req 14.7).
    """
    return LearnedRuleStore(
        version=data.get("version", 1),
        candidates={
            key: _candidate_from_dict(candidate)
            for key, candidate in (data.get("candidates") or {}).items()
        },
        learned_rules=[
            _learned_from_dict(lr) for lr in (data.get("learned_rules") or [])
        ],
    )
