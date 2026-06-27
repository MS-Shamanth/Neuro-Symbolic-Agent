"""Rule-Learner / orchestrator integration tests (Task 17.12).

These tests cover how the :class:`~nsr.orchestrator.PipelineOrchestrator` wires the
optional Rule Learner into the ``goal-satisfied`` termination path (Req 14.1) and how the
disabled path stays identical to Requirements 1-13 (Req 14.10):

1. **Enabled path** -- on goal satisfaction the learner is invoked
   (``induce -> corroborate -> promote``) *after* the Verified_Output is produced, and a
   promoted ``Learned_Rule`` extends Procedural_Memory so a *subsequent* query in the run
   can apply it and have its accepted step marked ``LEARNED`` (Req 14.1, 14.5).
2. **Best-effort isolation** -- any exception inside the rule-learning block is caught and
   logged and never corrupts or discards the already-emitted Verified_Output / Proof_Trace.
3. **Disabled path** -- with ``rule_learning_enabled = False`` the entire block is skipped:
   the learner is never touched and Procedural_Memory holds only the Seeded_Rules.

Only the LLM backend is mocked (via :class:`~nsr.llm_component.MockBackend`); every other
collaborator -- translation, decoding, control, validation, trace building, and (where a
real learner is used) the Rule Learner itself -- is the real implementation.

The dedicated disabled-path *equivalence* property test is Task 17.13 and is intentionally
not implemented here.
"""

from __future__ import annotations

import json

from nsr.actr_controller import ACTRController
from nsr.constrained_decoder import ConstrainedDecoder
from nsr.llm_component import LLMComponent, MockBackend
from nsr.models import (
    LearnedRule,
    LearnedRuleStore,
    ProductionRule,
    PromotionResult,
    RuleOrigin,
    RuleProvenance,
    SystemConfig,
    TerminationReason,
    ValidationStatus,
    VerifiedOutput,
)
from nsr.orchestrator import PipelineOrchestrator
from nsr.rule_learner import RuleLearner
from nsr.translation_layer import TranslationLayer
from nsr.validation_engine import ValidationEngine


# --------------------------------------------------------------------------- helpers


def make_config(*, rule_learning_enabled: bool = False) -> SystemConfig:
    return SystemConfig(
        max_cycle_limit=10,
        repair_attempt_limit=2,
        retry_count=0,
        llm_selection="gpt-4o-mini",
        output_format="json",
        conflict_resolution_policy="priority",
        generation_timeout_ms=30000,
        rule_learning_enabled=rule_learning_enabled,
    )


def json_step(logic_form: str) -> str:
    return json.dumps({"logic_form": logic_form})


def build_orchestrator(*, script, procedural_memory, config, rule_learner=None):
    """Assemble an orchestrator over real components and a scripted mock backend."""
    backend = MockBackend(list(script))
    llm = LLMComponent(backend, config)
    translation = TranslationLayer()
    validation = ValidationEngine()
    orchestrator = PipelineOrchestrator(
        llm=llm,
        translation=translation,
        decoder=ConstrainedDecoder(llm, config),
        controller=ACTRController(config.conflict_resolution_policy),
        validation=validation,
        config=config,
        procedural_memory=list(procedural_memory),
        rule_learner=rule_learner,
    )
    return orchestrator, backend


class _FakeLearner:
    """A controllable stand-in for the Rule Learner exercising the orchestrator wiring.

    It records its calls and promotes its configured rules exactly once, so the test can
    assert the orchestrator invokes ``induce -> corroborate -> promote`` on goal
    satisfaction and extends Procedural_Memory with the promoted Learned_Rules.
    """

    def __init__(self, *, promote_once=None, raise_on_induce=False):
        self.induce_calls = 0
        self.corroborate_calls = 0
        self.promote_calls = 0
        self._promote_once = promote_once or []
        self._raise_on_induce = raise_on_induce
        self._promoted = False

    def induce(self, trace, *, trace_id):
        self.induce_calls += 1
        if self._raise_on_induce:
            raise RuntimeError("boom: induction failed")
        return []

    def corroborate(self, candidates):
        self.corroborate_calls += 1

    def promote(self, procedural_memory):
        self.promote_calls += 1
        result = PromotionResult()
        if not self._promoted:
            self._promoted = True
            result.promoted.extend(self._promote_once)
        return result


# Seeded rule applicable only to steps mentioning "alpha"; satisfied when "ok" is present.
SEED_ALPHA = ProductionRule(rule_id="R-alpha", condition="IF alpha", action="THEN ok")
# A learned rule applicable only to "beta" steps; satisfied when "ok" is present.
LEARNED_BETA = LearnedRule(
    rule=ProductionRule(rule_id="learned::beta", condition="IF beta", action="THEN ok"),
    provenance=RuleProvenance(trace_ids=["t1"], step_ids=[0]),
    origin=RuleOrigin.LEARNED,
)


# ----------------------------------------------- 1. enabled path: promote + extend


def test_enabled_path_promotes_and_extends_procedural_memory_across_queries():
    """On goal satisfaction the learner runs and a promoted rule extends Procedural_Memory.

    Query 1 (only the seeded "alpha" rule matches) is accepted and satisfies the goal; the
    orchestrator then invokes the learner and appends the promoted ``LEARNED_BETA`` rule to
    Procedural_Memory. Query 2 produces a "beta" step that only the learned rule governs,
    so it is selected, accepted, and its step is marked ``LEARNED`` (Req 14.1, 14.5).
    """
    config = make_config(rule_learning_enabled=True)
    learner = _FakeLearner(promote_once=[LEARNED_BETA])
    orchestrator, _ = build_orchestrator(
        script=[json_step("derive alpha ok"), json_step("derive beta ok")],
        procedural_memory=[SEED_ALPHA],
        config=config,
        rule_learner=learner,
    )

    # --- Query 1: goal-satisfied; learner invoked AFTER emission --------------------
    result1 = orchestrator.run("establish alpha")
    assert isinstance(result1, VerifiedOutput)
    assert result1.proof_trace.termination_reason == TerminationReason.GOAL_SATISFIED
    # The accepted step applied the seeded rule and is marked SEEDED.
    assert result1.proof_trace.steps[-1].applied_rule_id == "R-alpha"
    assert result1.proof_trace.steps[-1].applied_rule_origin == RuleOrigin.SEEDED

    # induce -> corroborate -> promote were each invoked once on the satisfied path.
    assert (learner.induce_calls, learner.corroborate_calls, learner.promote_calls) == (1, 1, 1)

    # The promoted Learned_Rule extended Procedural_Memory for subsequent queries.
    pm_ids = [r.rule_id for r in orchestrator.procedural_memory]
    assert pm_ids == ["R-alpha", "learned::beta"]
    assert orchestrator.learned_rule_ids == {"learned::beta"}

    # --- Query 2: only the learned "beta" rule matches; it is applied + marked LEARNED
    result2 = orchestrator.run("establish beta")

    assert isinstance(result2, VerifiedOutput)
    assert result2.proof_trace.termination_reason == TerminationReason.GOAL_SATISFIED
    accepted = result2.proof_trace.steps[-1]
    assert accepted.status == ValidationStatus.ACCEPTED
    assert accepted.applied_rule_id == "learned::beta"
    assert accepted.applied_rule_origin == RuleOrigin.LEARNED


def test_enabled_path_with_real_learner_promotes_and_extends_memory():
    """End-to-end with the real Rule Learner: a goal-satisfied run grows Procedural_Memory.

    With ``corroboration_threshold = 1`` a single successful trace is enough to promote an
    induced, non-contradicting candidate into the store and into the orchestrator's
    Procedural_Memory (Req 14.1, 14.3).
    """
    config = make_config(rule_learning_enabled=True)
    store = LearnedRuleStore()
    learner = RuleLearner(
        store, ValidationEngine(), corroboration_threshold=1, seed=7
    )
    seed_ok = ProductionRule(rule_id="R-ok", condition="", action="THEN ok")
    orchestrator, _ = build_orchestrator(
        script=[json_step("derive value ok")],
        procedural_memory=[seed_ok],
        config=config,
        rule_learner=learner,
    )

    result = orchestrator.run("establish value")

    assert isinstance(result, VerifiedOutput)
    assert result.proof_trace.termination_reason == TerminationReason.GOAL_SATISFIED
    # A learned rule was genuinely induced, corroborated, and promoted into the store.
    assert len(store.learned_rules) >= 1
    promoted_id = store.learned_rules[0].rule.rule_id
    # ...and the orchestrator extended Procedural_Memory with it for later queries.
    assert promoted_id in [r.rule_id for r in orchestrator.procedural_memory]
    assert promoted_id in orchestrator.learned_rule_ids


# ----------------------------------------- 2. best-effort: failure preserves output


def test_rule_learning_failure_is_best_effort_and_preserves_output():
    """An exception inside the rule-learning block never corrupts the emitted output.

    The learner raises during induction; the orchestrator must still return the intact
    Verified_Output and Proof_Trace it had already produced, and Procedural_Memory must be
    left unchanged (no partial promotion).
    """
    config = make_config(rule_learning_enabled=True)
    learner = _FakeLearner(raise_on_induce=True)
    orchestrator, _ = build_orchestrator(
        script=[json_step("derive alpha ok")],
        procedural_memory=[SEED_ALPHA],
        config=config,
        rule_learner=learner,
    )

    result = orchestrator.run("establish alpha")

    # The goal-satisfied Verified_Output is emitted and intact despite the failure.
    assert isinstance(result, VerifiedOutput)
    assert result.proof_trace.termination_reason == TerminationReason.GOAL_SATISFIED
    assert len(result.proof_trace.steps) == 1
    assert result.proof_trace.steps[0].status == ValidationStatus.ACCEPTED
    assert result.proof_trace.steps[0].applied_rule_id == "R-alpha"
    assert result.faithfulness_score == 1.0

    # Induction was attempted (and failed); nothing was promoted into Procedural_Memory.
    assert learner.induce_calls == 1
    assert [r.rule_id for r in orchestrator.procedural_memory] == ["R-alpha"]
    assert orchestrator.learned_rule_ids == set()


# ------------------------------------------------- 3. disabled path: block skipped


def test_disabled_path_skips_learning_and_leaves_memory_unchanged():
    """With rule learning disabled the learner is never invoked and Procedural_Memory holds
    only the Seeded_Rules (Req 14.10)."""
    config = make_config(rule_learning_enabled=False)
    # A learner that would explode if ever touched, proving the block is fully skipped.
    learner = _FakeLearner(raise_on_induce=True, promote_once=[LEARNED_BETA])
    orchestrator, _ = build_orchestrator(
        script=[json_step("derive alpha ok")],
        procedural_memory=[SEED_ALPHA],
        config=config,
        rule_learner=learner,
    )

    result = orchestrator.run("establish alpha")

    assert isinstance(result, VerifiedOutput)
    assert result.proof_trace.termination_reason == TerminationReason.GOAL_SATISFIED
    # The learner was never called on the disabled path.
    assert (learner.induce_calls, learner.corroborate_calls, learner.promote_calls) == (0, 0, 0)
    # Procedural_Memory is unchanged: only the Seeded_Rule remains.
    assert [r.rule_id for r in orchestrator.procedural_memory] == ["R-alpha"]
    assert orchestrator.learned_rule_ids == set()
    # The accepted step still records the seeded marker (additive, backward-compatible).
    assert result.proof_trace.steps[-1].applied_rule_origin == RuleOrigin.SEEDED
