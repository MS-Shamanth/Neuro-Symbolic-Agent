"""Property test for disabled-path equivalence (Task 17.13).

**Property 10: Disabled rule learning is behaviorally identical to Req 1-13**

For any query and seeded rule set, processing a query with
``rule_learning_enabled = False`` produces a result (a :class:`VerifiedOutput` or an
:class:`ErrorRecord`) and a :class:`ProofTrace` that are *identical* to the
no-rule-learning baseline pipeline -- same result type, same final answer, same
termination reason, same per-step ``(status, applied_rule_id, applied_rule_origin)``
sequence, and same Faithfulness_Score -- and Procedural_Memory still contains only the
originally seeded rules (no learned rule ids).

This is a **model-based equivalence** test. For each generated scenario (a MockBackend
script + seeded Procedural_Memory + query) we run *two* orchestrators over identical
inputs:

* **(A)** ``rule_learning_enabled = False`` *with* a real :class:`RuleLearner` wired in,
  proving the disabled gate -- not the absence of a learner -- is what suppresses
  learning; and
* **(B)** a baseline orchestrator with *no* ``rule_learner`` at all (and rule learning
  disabled), i.e. the fixed-rule pipeline defined by Requirements 1-13.

Each orchestrator is assembled over the real Translation_Layer, Constrained Decoder,
ACT-R Controller, Validation Engine, and Proof_Trace builder; only the LLM backend is the
in-memory :class:`MockBackend`. A fresh :func:`deepcopy` of the backend script and the
seeded Procedural_Memory is handed to each orchestrator so the two runs never share
mutable state.

**Validates: Requirements 14.10**
"""

from __future__ import annotations

import json
from copy import deepcopy

from hypothesis import given
from hypothesis import strategies as st

from nsr.actr_controller import ACTRController
from nsr.constrained_decoder import ConstrainedDecoder
from nsr.llm_component import LLMComponent, MockBackend
from nsr.models import (
    ErrorRecord,
    LearnedRuleStore,
    ProductionRule,
    SystemConfig,
    VerifiedOutput,
)
from nsr.orchestrator import PipelineOrchestrator
from nsr.rule_learner import RuleLearner
from nsr.translation_layer import TranslationLayer
from nsr.validation_engine import ValidationEngine


# --------------------------------------------------------------------------- helpers


def make_config() -> SystemConfig:
    """A disabled-rule-learning config used by both the A and B orchestrators."""
    return SystemConfig(
        max_cycle_limit=10,
        repair_attempt_limit=2,
        retry_count=0,
        llm_selection="gpt-4o-mini",
        output_format="json",
        conflict_resolution_policy="priority",
        generation_timeout_ms=30000,
        rule_learning_enabled=False,
    )


def build_orchestrator(*, script, procedural_memory, config, rule_learner=None):
    """Assemble an orchestrator over real components and a scripted mock backend.

    Mirrors the assembly pattern used in ``test_orchestrator_rule_learning`` /
    ``test_orchestrator_integration``: every collaborator is the real implementation and
    only the LLM backend is the in-memory :class:`MockBackend`.
    """
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


def signature(result, orchestrator):
    """A fully comparable fingerprint of a run's observable behavior.

    Captures the result type, the final answer / error identity, the Faithfulness_Score,
    the termination reason, and the ordered per-step ``(status, applied_rule_id,
    applied_rule_origin)`` tuples from the produced Proof_Trace.
    """
    trace = orchestrator.last_trace
    steps = (
        tuple(
            (s.sequence, s.status, s.applied_rule_id, s.applied_rule_origin)
            for s in trace.steps
        )
        if trace is not None
        else ()
    )
    termination = trace.termination_reason if trace is not None else None
    if isinstance(result, VerifiedOutput):
        return (
            "VerifiedOutput",
            result.final_answer,
            result.faithfulness_score,
            termination,
            steps,
        )
    assert isinstance(result, ErrorRecord)
    return ("ErrorRecord", result.failed_component, result.reason, termination, steps)


# ------------------------------------------------------------------- strategies

#: A small token alphabet; "ok" is the term the always-applicable rule requires.
_TOKENS = ["alpha", "beta", "gamma", "value", "a", "b", "c"]

#: Pool of seeded production rules; "R-ok" is satisfied by any step containing "ok".
_RULE_POOL = [
    ProductionRule(rule_id="R-ok", condition="", action="THEN ok"),
    ProductionRule(rule_id="R-alpha", condition="IF alpha", action="THEN ok"),
    ProductionRule(rule_id="R-beta", condition="IF beta", action="THEN ok"),
]


@st.composite
def _logic_form(draw) -> str:
    """A JSON step whose logic form is a few tokens, with "ok" present iff accepted-bound."""
    tokens = draw(st.lists(st.sampled_from(_TOKENS), min_size=1, max_size=3))
    if draw(st.booleans()):
        tokens.append("ok")
    return json.dumps({"logic_form": " ".join(tokens)})


@st.composite
def _scenario(draw):
    """Generate (script, seeded procedural memory, query) for one equivalence trial."""
    script = draw(st.lists(_logic_form(), min_size=1, max_size=6))
    # Always include the always-applicable "R-ok" rule so some scenarios reach goal
    # satisfaction; optionally add the token-specific rules for variety. Sample by index
    # (ProductionRule is an unhashable dataclass, so it cannot be deduped directly).
    extra_indices = draw(
        st.lists(st.integers(min_value=1, max_value=len(_RULE_POOL) - 1),
                 max_size=2, unique=True)
    )
    procedural_memory = [_RULE_POOL[0], *(_RULE_POOL[i] for i in sorted(extra_indices))]
    num_subgoals = draw(st.integers(min_value=1, max_value=4))
    query = ". then ".join(f"establish goal {i}" for i in range(num_subgoals))
    return script, procedural_memory, query


# ----------------------------------------------------------------------- property


@given(_scenario())
def test_disabled_rule_learning_is_identical_to_baseline(scenario) -> None:
    """Property 10: the disabled rule-learning path is behaviorally identical to the
    no-rule-learning baseline, and Procedural_Memory holds only the seeded rules
    (Req 14.10)."""
    script, procedural_memory, query = scenario
    seeded_ids = [r.rule_id for r in procedural_memory]

    # (A) Rule learning DISABLED but a real learner is wired in: the disabled gate, not
    # the absence of a learner, must suppress every learning effect.
    learner = RuleLearner(
        LearnedRuleStore(),
        ValidationEngine(),
        corroboration_threshold=1,
        seed=7,
    )
    orch_a, _ = build_orchestrator(
        script=deepcopy(script),
        procedural_memory=deepcopy(procedural_memory),
        config=make_config(),
        rule_learner=learner,
    )

    # (B) Baseline: the fixed-rule pipeline with NO rule learner at all.
    orch_b, _ = build_orchestrator(
        script=deepcopy(script),
        procedural_memory=deepcopy(procedural_memory),
        config=make_config(),
        rule_learner=None,
    )

    result_a = orch_a.run(query)
    result_b = orch_b.run(query)

    # The two runs are observably identical: type, final answer / error identity,
    # Faithfulness_Score, termination reason, and the full per-step outcome sequence.
    assert signature(result_a, orch_a) == signature(result_b, orch_b)

    # Procedural_Memory still contains only the originally seeded rules in both runs:
    # no Learned_Rule was promoted on the disabled path.
    assert [r.rule_id for r in orch_a.procedural_memory] == seeded_ids
    assert [r.rule_id for r in orch_b.procedural_memory] == seeded_ids
    assert orch_a.learned_rule_ids == set()
    assert orch_b.learned_rule_ids == set()
