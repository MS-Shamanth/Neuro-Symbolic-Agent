"""Reusable, fully-offline demo scenarios for the Neuro-Symbolic reasoning pipeline.

Every helper here wires a **real** :class:`~nsr.orchestrator.PipelineOrchestrator` over a
scripted :class:`~nsr.llm_component.MockBackend`. The MockBackend returns deterministic,
in-memory completions, so the whole dual-process cycle runs for real with **no network and
no API key**. Plugging in a hosted-API or local-runtime backend is a configuration change
(`SystemConfig.llm_selection` + credentials from the environment) and requires no source
edits — see ``demo/README.md``.

The module exposes:

- :func:`build_orchestrator` — assemble a fully-wired orchestrator over a scripted backend.
- :func:`build_orchestrator_with_rule_learning` — the same, with the optional Rule Learner
  enabled (adaptive rule learning).
- :data:`SCENARIOS` and :func:`run_scenario` — three self-contained example scenarios:
  a multi-step syllogism (all steps accepted), an arithmetic problem that exercises a
  **rejection → repair → acceptance** path, and a multi-hop derivation.

Production rules use the same ``IF <terms> THEN <terms>`` string form the
:class:`~nsr.validation_engine.ValidationEngine` and
:class:`~nsr.actr_controller.ACTRController` interpret: a rule is *applicable* when every
``IF`` term appears in the step (an empty ``IF`` is always applicable) and *satisfied* when
every ``THEN`` term appears. The scripted steps carry the required action token in their
structured ``predicates`` so the step's machine-checkable ``logic_form`` can stay equal to
the human-meaningful conclusion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from nsr.actr_controller import ACTRController
from nsr.constrained_decoder import ConstrainedDecoder
from nsr.llm_component import LLMBackend, LLMComponent, MockBackend
from nsr.models import (
    ProductionRule,
    SystemConfig,
    VerifiedOutput,
)
from nsr.orchestrator import CycleStage, PipelineOrchestrator
from nsr.repair_coordinator import RepairCoordinator
from nsr.rule_learner import RuleLearner
from nsr.models import LearnedRuleStore
from nsr.translation_layer import TranslationLayer
from nsr.validation_engine import ValidationEngine


# --------------------------------------------------------------------------- #
# Configuration + scripted-step helpers
# --------------------------------------------------------------------------- #


def make_config(
    *,
    max_cycle_limit: int = 10,
    repair_attempt_limit: int = 2,
    retry_count: int = 0,
    conflict_resolution_policy: str = "priority",
    rule_learning_enabled: bool = False,
    random_seed: int = 7,
) -> SystemConfig:
    """A deterministic :class:`SystemConfig` for the offline demo.

    The fixed ``random_seed`` keeps every seeded operation deterministic; the output
    format is JSON so the Constrained Decoder enforces a ``logic_form`` field.
    """
    return SystemConfig(
        max_cycle_limit=max_cycle_limit,
        repair_attempt_limit=repair_attempt_limit,
        retry_count=retry_count,
        llm_selection="gpt-4o-mini",
        output_format="json",
        conflict_resolution_policy=conflict_resolution_policy,
        generation_timeout_ms=30000,
        latency_budget_ms=2000,
        random_seed=random_seed,
        rule_learning_enabled=rule_learning_enabled,
        corroboration_threshold=1,
    )


def scripted_step(logic_form: str, *, status: str = "verified", **predicates: str) -> str:
    """Build one constrained-decoder-shaped JSON completion for the MockBackend.

    ``logic_form`` is the machine-checkable conclusion (and becomes the step's final
    answer when it is the last accepted step). ``status`` (default ``"verified"``) and any
    extra ``predicates`` are placed in the structured ``predicates`` map, where the
    Validation Engine can match production-rule action terms — so the ``logic_form`` itself
    stays equal to the human-meaningful conclusion.
    """
    payload = {"status": status, **predicates}
    return json.dumps({"logic_form": logic_form, "predicates": payload})


def rejected_step(logic_form: str, **predicates: str) -> str:
    """A scripted completion that will be *rejected* by a ``THEN verified`` rule.

    It carries a ``status`` of ``"draft"`` (not ``"verified"``), so a production rule whose
    action requires the ``verified`` token is violated, triggering the repair sub-loop.
    """
    payload = {"status": "draft", **predicates}
    return json.dumps({"logic_form": logic_form, "predicates": payload})


# --------------------------------------------------------------------------- #
# Orchestrator wiring (all real components, scripted offline backend)
# --------------------------------------------------------------------------- #


class RealModelTranslationLayer(TranslationLayer):
    """A Translation Layer that tells a *real* model the exact output schema to produce.

    The base :class:`~nsr.translation_layer.TranslationLayer` renders a generic
    "produce the next reasoning step" prompt, which a scripted MockBackend satisfies but a
    real model cannot — it does not know the Constrained Decoder requires a JSON object
    carrying a non-empty ``logic_form`` string. This subclass appends an explicit,
    schema-precise instruction to the backward-translation prompt so a real model (served
    via Ollama) emits conforming steps, and so the final step's ``logic_form`` is the
    concise final answer the Evaluation Harness scores. It changes only the *prompt text*;
    the symbolic pipeline (translation, validation, repair) is unchanged.
    """

    _SCHEMA_INSTRUCTION = (
        "\n\nRespond with ONLY a single JSON object and nothing else, of exactly this "
        'form: {"logic_form": "<conclusion>"}. '
        "The \"logic_form\" value must be one short, self-contained statement that "
        "settles the current sub-goal. If this is the final (or only) sub-goal, the "
        "\"logic_form\" MUST be the final answer to the overall question, as concise as "
        "possible — a bare number or a single word/phrase with no explanation, units, or "
        "punctuation (for example \"4\", \"blue\", or \"yes\"). Do not include any text "
        "outside the JSON object."
    )

    def to_context(self, state, **kwargs):  # type: ignore[override]
        context = super().to_context(state, **kwargs)
        context.prompt_text = f"{context.prompt_text}{self._SCHEMA_INSTRUCTION}"
        return context


class MathReasoningTranslationLayer(RealModelTranslationLayer):
    """Prompts a real model to emit *checkable* arithmetic steps for the math benchmark.

    Each reasoning step should assert one arithmetic operation in a form the
    :class:`~demo.arithmetic_validation.ArithmeticValidationEngine` can verify, so a wrong
    intermediate computation is rejected and repaired rather than carried through to a
    wrong final answer. The final step is the bare numeric answer.
    """

    _SCHEMA_INSTRUCTION = (
        "\n\nSolve the problem ONE arithmetic step at a time. Respond with ONLY a single "
        "JSON object and nothing else. For an intermediate calculation use exactly: "
        '{"logic_form": "<a> <op> <b> = <result>", "predicates": {"lhs": <a>, "op": '
        '"<op>", "rhs": <b>, "result": <result>}} where <op> is one of + - * /. '
        "Use the result of a previous step as an operand when needed. For the FINAL step, "
        'when the answer is known, respond with {"logic_form": "<final number>"} where the '
        "logic_form is just the final numeric answer with no words, units, or punctuation."
    )


#: Alias for the math-reasoning translation layer. It emits a checkable equation whose
#: right-hand side is the computed value, so the ArithmeticValidationEngine can verify it.
MathTranslationLayer = MathReasoningTranslationLayer


def build_orchestrator_with_backend(
    *,
    backend: LLMBackend,
    procedural_memory: Optional[list[ProductionRule]] = None,
    config: Optional[SystemConfig] = None,
    with_repair: bool = True,
    translation: Optional[TranslationLayer] = None,
    validation: Optional[ValidationEngine] = None,
    on_stage: Optional[Callable[[int, CycleStage], None]] = None,
) -> PipelineOrchestrator:
    """Assemble a :class:`PipelineOrchestrator` over an *already-constructed* backend.

    This is the backend-agnostic core of :func:`build_orchestrator`: it wires *all real
    components* (Translation Layer, Constrained Decoder, ACT-R Controller, Validation
    Engine, Repair Coordinator, Proof Trace builder) around whatever
    :class:`~nsr.llm_component.LLMBackend` is injected. Pass a
    :class:`~nsr.llm_component.MockBackend` for a fully-offline run, or a real backend
    (for example :class:`~nsr.llm_component.OllamaBackend`) to drive the genuine pipeline
    over a live model — the four-stage cycle is identical either way.

    ``with_repair`` wires in the real Repair Coordinator (the default); ``on_stage`` is an
    optional per-stage hook. Returns the orchestrator only (the caller already holds the
    backend it injected).
    """
    cfg = config or make_config()
    llm = LLMComponent(backend, cfg)
    translation = translation if translation is not None else TranslationLayer()
    validation = validation if validation is not None else ValidationEngine()
    repair = (
        RepairCoordinator(llm, translation, validation, cfg.repair_attempt_limit)
        if with_repair
        else None
    )
    return PipelineOrchestrator(
        llm=llm,
        translation=translation,
        decoder=ConstrainedDecoder(llm, cfg),
        controller=ACTRController(cfg.conflict_resolution_policy),
        validation=validation,
        config=cfg,
        repair=repair,
        procedural_memory=list(procedural_memory) if procedural_memory else [],
        on_stage=on_stage,
    )


def build_orchestrator(
    *,
    script: list[str],
    procedural_memory: Optional[list[ProductionRule]] = None,
    config: Optional[SystemConfig] = None,
    with_repair: bool = True,
    on_stage: Optional[Callable[[int, CycleStage], None]] = None,
) -> tuple[PipelineOrchestrator, MockBackend]:
    """Assemble a :class:`PipelineOrchestrator` wiring *all real components* offline.

    Only the LLM backend is mocked (via :class:`MockBackend`), so the four-stage cycle
    exercises the real Translation Layer, Constrained Decoder, ACT-R Controller, Validation
    Engine, Repair Coordinator, and Proof Trace builder end-to-end. ``with_repair`` wires in
    the real Repair Coordinator (the default); ``on_stage`` is an optional per-stage hook.

    This is a thin convenience wrapper over :func:`build_orchestrator_with_backend`: it
    constructs a scripted :class:`MockBackend` from ``script`` and delegates, so the
    offline and real paths share identical wiring.
    """
    backend = MockBackend(list(script))
    orchestrator = build_orchestrator_with_backend(
        backend=backend,
        procedural_memory=procedural_memory,
        config=config,
        with_repair=with_repair,
        on_stage=on_stage,
    )
    return orchestrator, backend


def build_orchestrator_with_rule_learning(
    *,
    script: list[str],
    procedural_memory: Optional[list[ProductionRule]] = None,
    config: Optional[SystemConfig] = None,
    on_stage: Optional[Callable[[int, CycleStage], None]] = None,
) -> tuple[PipelineOrchestrator, MockBackend, RuleLearner]:
    """Like :func:`build_orchestrator`, but with the optional Rule Learner enabled.

    Adaptive rule learning runs only after a goal-satisfied run, off the per-step critical
    path. The returned :class:`~nsr.rule_learner.RuleLearner` shares its
    :class:`~nsr.models.LearnedRuleStore`, so promoted Learned_Rules persist across queries
    issued to the same orchestrator and are marked ``LEARNED`` when applied.
    """
    cfg = config or make_config(rule_learning_enabled=True)
    if not cfg.rule_learning_enabled:
        cfg = make_config(rule_learning_enabled=True)
    backend = MockBackend(list(script))
    llm = LLMComponent(backend, cfg)
    translation = TranslationLayer()
    validation = ValidationEngine()
    repair = RepairCoordinator(llm, translation, validation, cfg.repair_attempt_limit)
    learner = RuleLearner(
        LearnedRuleStore(),
        validation,
        corroboration_threshold=cfg.corroboration_threshold,
        max_learned_rules=cfg.max_learned_rules,
        seed=cfg.random_seed,
    )
    orchestrator = PipelineOrchestrator(
        llm=llm,
        translation=translation,
        decoder=ConstrainedDecoder(llm, cfg),
        controller=ACTRController(cfg.conflict_resolution_policy),
        validation=validation,
        config=cfg,
        repair=repair,
        procedural_memory=list(procedural_memory) if procedural_memory else [],
        rule_learner=learner,
        on_stage=on_stage,
    )
    return orchestrator, backend, learner


# --------------------------------------------------------------------------- #
# Example scenarios
# --------------------------------------------------------------------------- #


@dataclass
class Scenario:
    """A self-contained, runnable demo scenario."""

    name: str
    title: str
    description: str
    query: str
    procedural_memory: list[ProductionRule]
    script: list[str]
    with_repair: bool = True
    config: Optional[SystemConfig] = None
    extra: dict = field(default_factory=dict)


# A guard rule that is never applicable to the demo steps (its IF term never appears),
# included to show a richer Procedural Memory of "available" rules.
_CONSISTENCY_GUARD = ProductionRule(
    rule_id="consistency-guard",
    condition="IF contradiction",
    action="THEN flag",
)


def _syllogism() -> Scenario:
    """Three-step deductive syllogism; every step is accepted (goal-satisfied)."""
    modus_ponens = ProductionRule(
        rule_id="modus-ponens",
        condition="",  # always applicable
        action="THEN entailed",  # satisfied when the step is marked entailed
    )
    return Scenario(
        name="syllogism",
        title="Deductive syllogism (all steps accepted)",
        description=(
            "A classic multi-step deduction. Each step is translated to a symbolic form, "
            "matched to the modus-ponens production rule, and validated as entailed, so the "
            "goal is satisfied with a perfect faithfulness score."
        ),
        query=(
            "All cats are mammals. then all mammals are animals. "
            "then conclude cats are animals."
        ),
        procedural_memory=[modus_ponens, _CONSISTENCY_GUARD],
        script=[
            scripted_step("all_cats_are_mammals", status="entailed"),
            scripted_step("all_mammals_are_animals", status="entailed"),
            scripted_step("cats_are_animals", status="entailed"),
        ],
    )


def _arithmetic_repair() -> Scenario:
    """Two sub-goals; the second is rejected, repaired, then accepted."""
    calc = ProductionRule(
        rule_id="calc-verified",
        condition="",  # always applicable
        action="THEN verified",  # satisfied only when the step is marked verified
    )
    return Scenario(
        name="arithmetic-repair",
        title="Arithmetic with a rejection → repair → acceptance path",
        description=(
            "The first step (the subtotal) is verified and accepted. The second step (the "
            "taxed total) arrives as an unverified draft and is REJECTED by the "
            "calc-verified rule, triggering the bounded repair sub-loop. The repaired step "
            "is verified and accepted, so the visualization shows the "
            "Validation ✗ → Repair → Validation ✓ path."
        ),
        query="Compute the subtotal. then apply tax to reach the total.",
        procedural_memory=[calc],
        script=[
            scripted_step("subtotal_equals_100", status="verified"),
            rejected_step("total_equals_92_typo"),  # rejected: not verified
            scripted_step("total_equals_108_with_tax", status="verified"),  # repair
        ],
    )


def _multi_hop() -> Scenario:
    """Three-hop factual derivation; every hop is accepted (goal-satisfied)."""
    grounded = ProductionRule(
        rule_id="grounded-inference",
        condition="",
        action="THEN grounded",
    )
    return Scenario(
        name="multi-hop",
        title="Multi-hop derivation (all hops accepted)",
        description=(
            "A multi-hop chain where each hop builds on the accepted conclusions already in "
            "Declarative Memory. Every hop is grounded and accepted, growing the Imaginal "
            "Buffer toward the final summary."
        ),
        query=(
            "Identify the capital of France. then identify its country. "
            "then conclude the summary."
        ),
        procedural_memory=[grounded, _CONSISTENCY_GUARD],
        script=[
            scripted_step("capital_of_france_is_paris", status="grounded"),
            scripted_step("paris_is_in_france", status="grounded"),
            scripted_step("summary_paris_is_the_french_capital", status="grounded"),
        ],
    )


#: The registered example scenarios, keyed by name.
SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "syllogism": _syllogism,
    "arithmetic-repair": _arithmetic_repair,
    "multi-hop": _multi_hop,
}

#: The default scenario the CLI runs when none is named (exercises repair).
DEFAULT_SCENARIO = "arithmetic-repair"


def get_scenario(name: str) -> Scenario:
    """Return the :class:`Scenario` registered under ``name``."""
    try:
        factory = SCENARIOS[name]
    except KeyError:
        allowed = ", ".join(sorted(SCENARIOS))
        raise ValueError(
            f"unknown scenario {name!r}; expected one of: {allowed}"
        ) from None
    return factory()


@dataclass
class ScenarioRun:
    """The full result of running a scenario, for rendering and inspection."""

    scenario: Scenario
    orchestrator: PipelineOrchestrator
    result: object  # VerifiedOutput | ErrorRecord
    stage_snapshots: list[dict]


def run_scenario(name: str) -> ScenarioRun:
    """Build a fully-wired offline orchestrator for ``name`` and run it.

    Returns a :class:`ScenarioRun` carrying the scenario, the orchestrator (for its final
    buffer state and procedural memory), the run result (a
    :class:`~nsr.models.VerifiedOutput` on success), and a per-cycle snapshot list captured
    via the orchestrator's stage hook.
    """
    scenario = get_scenario(name)

    snapshots: list[dict] = []

    def _on_stage(cycle_index: int, stage: CycleStage) -> None:
        # Capture a lightweight working-memory snapshot at the start of each cycle.
        if stage is not CycleStage.GENERATE:
            return
        try:
            state = orchestrator._controller.state()  # read-only snapshot
        except Exception:
            return
        snapshots.append(
            {
                "cycle": cycle_index,
                "declarative": [r.logic_form for r in state.declarative_memory],
                "imaginal": state.imaginal_buffer.logic_form
                if state.imaginal_buffer is not None
                else None,
            }
        )

    orchestrator, _backend = build_orchestrator(
        script=scenario.script,
        procedural_memory=scenario.procedural_memory,
        config=scenario.config,
        with_repair=scenario.with_repair,
        on_stage=_on_stage,
    )
    result = orchestrator.run(scenario.query)
    return ScenarioRun(
        scenario=scenario,
        orchestrator=orchestrator,
        result=result,
        stage_snapshots=snapshots,
    )


__all__ = [
    "make_config",
    "scripted_step",
    "rejected_step",
    "build_orchestrator",
    "build_orchestrator_with_backend",
    "RealModelTranslationLayer",
    "MathReasoningTranslationLayer",
    "MathTranslationLayer",
    "build_orchestrator_with_rule_learning",
    "Scenario",
    "ScenarioRun",
    "SCENARIOS",
    "DEFAULT_SCENARIO",
    "get_scenario",
    "run_scenario",
    "VerifiedOutput",
]
