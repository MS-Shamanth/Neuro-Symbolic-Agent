"""Unit tests for RuleLearner corroborate/promote/contradicts/cap (Task 17.4).

Covers Req 14.3 (corroboration + promotion), Req 14.4 (contradiction discard + logged
conflict), Req 14.6 (deterministic order + run-record recording), Req 14.7 (versioned
store round-trip + durable persistence), and Req 14.9 (learned-rule cap).

These are example/edge-case unit tests; the dedicated property tests live in
Tasks 17.5-17.9.
"""

from __future__ import annotations

import json

from nsr import RuleLearner, ValidationEngine
from nsr.models import (
    CandidateRule,
    LearnedRuleStore,
    ProductionRule,
    RuleProvenance,
    SymbolicRepresentation,
    store_from_dict,
    store_to_dict,
)
from nsr.models.config import RunRecord, SystemConfig
from nsr.models.enums import TerminationReason, ValidationStatus
from nsr.models.learning import RuleOrigin
from nsr.models.trace import ProofStep, ProofTrace
from nsr.reproducibility import ReproducibilityManager


# --------------------------------------------------------------------------- helpers


def _rep(logic_form="", source_text="", predicates=None):
    return SymbolicRepresentation(
        logic_form=logic_form,
        source_text=source_text,
        predicates=predicates or {},
    )


def _step(sequence, *, status=ValidationStatus.ACCEPTED, rep=None):
    return ProofStep(
        sequence=sequence,
        step_text=f"step-{sequence}",
        representation=rep if rep is not None else _rep(logic_form=f"sum({sequence})"),
        status=status,
    )


def _trace(steps, reason=TerminationReason.GOAL_SATISFIED):
    return ProofTrace(steps=list(steps), termination_reason=reason)


def _learner(store=None, **kwargs):
    kwargs.setdefault("seed", 7)
    return RuleLearner(store or LearnedRuleStore(), ValidationEngine(), **kwargs)


def _witness_candidate(*, condition, action, witness, trace_ids, key, count=None):
    """Build a candidate with explicit IF/THEN rule and a single witness representation."""
    return CandidateRule(
        rule=ProductionRule(rule_id=f"learned::{key}", condition=condition, action=action),
        provenance=RuleProvenance(trace_ids=list(trace_ids), step_ids=[0]),
        corroboration_count=count if count is not None else len(trace_ids),
        normalized_key=key,
        witnesses=[witness],
    )


# ----------------------------------------------------------------- corroborate (14.3)


def test_corroborate_increments_once_per_distinct_trace():
    learner = _learner(corroboration_threshold=2)
    cands_a = learner.induce(_trace([_step(0, rep=_rep("sum(2,3)", predicates={"operation": "add"}))]), trace_id="t1")
    cands_b = learner.induce(_trace([_step(1, rep=_rep("sum(4,5)", predicates={"operation": "add"}))]), trace_id="t2")

    # Two distinct traces produce the same generalization (same normalized key).
    assert cands_a[0].normalized_key == cands_b[0].normalized_key

    learner.corroborate(cands_a)
    learner.corroborate(cands_b)
    learner.corroborate(cands_a)  # re-corroborating the same trace must be idempotent

    stored = learner.store.candidates[cands_a[0].normalized_key]
    assert stored.corroboration_count == 2
    assert stored.provenance.trace_ids == ["t1", "t2"]
    assert len(stored.witnesses) == 2  # one witness per distinct corroborating trace


def test_corroborate_does_not_mutate_input_candidate():
    learner = _learner()
    cands = learner.induce(_trace([_step(0, rep=_rep("sum(1)", predicates={"op": "x"}))]), trace_id="t1")
    learner.corroborate(cands)
    # Mutating the store's copy must not bleed back into the caller's candidate.
    learner.store.candidates[cands[0].normalized_key].provenance.trace_ids.append("t9")
    assert cands[0].provenance.trace_ids == ["t1"]


# ------------------------------------------------------------------ contradicts (14.4)


def test_contradicts_true_when_existing_rejects_a_candidate_accepted_witness():
    learner = _learner()
    witness = _rep(logic_form="foo bar")
    candidate = _witness_candidate(
        condition="IF foo", action="THEN bar", witness=witness, trace_ids=["t1"], key="k"
    )
    existing = ProductionRule(rule_id="seed-1", condition="IF foo", action="THEN required")

    # candidate accepts the witness (foo+bar present); existing rejects it (no "required").
    assert learner.contradicts(candidate, existing) is True


def test_contradicts_false_when_existing_does_not_apply():
    learner = _learner()
    witness = _rep(logic_form="foo bar")
    candidate = _witness_candidate(
        condition="IF foo", action="THEN bar", witness=witness, trace_ids=["t1"], key="k"
    )
    existing = ProductionRule(rule_id="seed-1", condition="IF other", action="THEN x")

    # existing's condition doesn't match the witness, so it cannot reject it.
    assert learner.contradicts(candidate, existing) is False


# -------------------------------------------------------------------- promote (14.3)


def test_promote_promotes_corroborated_non_contradicting_candidate():
    learner = _learner(corroboration_threshold=2)
    learner.corroborate(learner.induce(_trace([_step(0, rep=_rep("sum(2,3)", predicates={"operation": "add"}))]), trace_id="t1"))
    learner.corroborate(learner.induce(_trace([_step(1, rep=_rep("sum(4,5)", predicates={"operation": "add"}))]), trace_id="t2"))

    result = learner.promote(procedural_memory=[])

    assert len(result.promoted) == 1
    assert result.promoted[0].origin is RuleOrigin.LEARNED
    assert learner.store.learned_rules == result.promoted
    assert [d.reason for d in result.decisions] == ["corroborated"]


def test_promote_skips_below_threshold_candidate():
    learner = _learner(corroboration_threshold=2)
    # Only one corroborating trace -> below threshold.
    learner.corroborate(learner.induce(_trace([_step(0, rep=_rep("sum(2,3)", predicates={"operation": "add"}))]), trace_id="t1"))

    result = learner.promote(procedural_memory=[])

    assert result.promoted == []
    assert [d.reason for d in result.decisions] == ["below-threshold"]


def test_promote_discards_contradicting_candidate_and_logs_conflict():
    learner = _learner(corroboration_threshold=1)
    witness = _rep(logic_form="foo bar")
    candidate = _witness_candidate(
        condition="IF foo", action="THEN bar", witness=witness, trace_ids=["t1"], key="k"
    )
    learner.store.candidates[candidate.normalized_key] = candidate
    existing = ProductionRule(rule_id="seed-1", condition="IF foo", action="THEN required")

    result = learner.promote(procedural_memory=[existing])

    assert result.promoted == []
    assert len(result.discarded) == 1
    assert result.discarded[0].conflicting_rule_id == "seed-1"
    assert result.decisions[0].reason == "contradiction"
    assert result.decisions[0].conflicting_rule_id == "seed-1"


# ------------------------------------------------------------------------ cap (14.9)


def test_promote_stops_at_cap_and_records_cap_reached():
    learner = _learner(corroboration_threshold=1, max_learned_rules=1)
    # Two distinct, non-contradicting, corroborated candidates.
    c1 = _witness_candidate(condition="IF a", action="THEN aa", witness=_rep("a aa"), trace_ids=["t1"], key="k1")
    c2 = _witness_candidate(condition="IF b", action="THEN bb", witness=_rep("b bb"), trace_ids=["t2"], key="k2")
    learner.store.candidates[c1.normalized_key] = c1
    learner.store.candidates[c2.normalized_key] = c2

    result = learner.promote(procedural_memory=[])

    assert len(result.promoted) == 1  # cap of 1 honored
    assert result.cap_reached is True
    reasons = {d.normalized_key: d.reason for d in result.decisions}
    # Canonical order (by key) promotes k1, then k2 hits the cap.
    assert reasons == {"k1": "corroborated", "k2": "cap-reached"}


# ----------------------------------------------------------- determinism (14.6)


def test_promotion_order_is_deterministic_and_canonical():
    def run():
        learner = _learner(corroboration_threshold=1)
        # Insert in non-canonical order; promotion must still be by normalized key.
        for key in ("k3", "k1", "k2"):
            cand = _witness_candidate(
                condition=f"IF {key}", action=f"THEN {key}x", witness=_rep(f"{key} {key}x"), trace_ids=["t"], key=key
            )
            learner.store.candidates[key] = cand
        return [d.normalized_key for d in learner.promote(procedural_memory=[]).decisions]

    assert run() == ["k1", "k2", "k3"]
    assert run() == run()


def test_record_run_writes_learning_fields_onto_run_record():
    learner = _learner(corroboration_threshold=1)
    cand = _witness_candidate(condition="IF a", action="THEN aa", witness=_rep("a aa"), trace_ids=["t1"], key="k1")
    learner.store.candidates["k1"] = cand
    result = learner.promote(procedural_memory=[])

    run_record = RunRecord(
        config=SystemConfig(
            max_cycle_limit=10,
            repair_attempt_limit=1,
            retry_count=0,
            llm_selection="x",
            output_format="json",
            conflict_resolution_policy="specificity",
            generation_timeout_ms=1000,
        ),
        dataset_ids=["d1"],
        model_id="m1",
        seed=7,
    )
    learner.record_run(run_record, result)

    assert run_record.learned_rules == learner.store.learned_rules
    assert run_record.induction_seed == 7
    assert run_record.corroboration_threshold == 1
    assert run_record.promotion_decisions == result.decisions


# ---------------------------------------------------- store round-trip + persist (14.7)


def test_store_to_dict_round_trips_losslessly():
    learner = _learner(corroboration_threshold=1)
    learner.corroborate(learner.induce(_trace([_step(0, rep=_rep("sum(2,3)", predicates={"operation": "add"}))]), trace_id="t1"))
    learner.promote(procedural_memory=[])

    data = store_to_dict(learner.store)
    restored = store_from_dict(data)

    assert restored == learner.store
    assert data["version"] == learner.store.version


def test_persist_learned_rule_store_writes_versioned_json(tmp_path):
    learner = _learner(corroboration_threshold=1)
    learner.corroborate(learner.induce(_trace([_step(0, rep=_rep("sum(2,3)", predicates={"operation": "add"}))]), trace_id="t1"))
    learner.promote(procedural_memory=[])

    repro = ReproducibilityManager()
    out = tmp_path / "store.json"
    error = learner.persist_store(repro, out)

    assert error is None
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["version"] == learner.store.version
    assert "format_version" in document
    assert store_from_dict(document) == learner.store


def test_seed_hook_registration_keeps_induction_seed_in_lockstep():
    repro = ReproducibilityManager()
    learner = RuleLearner(LearnedRuleStore(), ValidationEngine(), reproducibility=repro)

    repro.apply_seed(123)

    cands = learner.induce(_trace([_step(0, rep=_rep("sum(1)", predicates={"op": "x"}))]), trace_id="t1")
    assert cands[0].provenance.induction_seed == 123
