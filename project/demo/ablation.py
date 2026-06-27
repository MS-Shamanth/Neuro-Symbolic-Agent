"""Ablation study for the Neuro-Symbolic reasoning demo (GSM8K).

This module evaluates **four configurations** of the architecture on the *same* GSM8K
subset so the contribution of each component can be quantified in isolation:

- **A — ``plain-llm``**: the model answers directly (the ``llm-only`` baseline). NO
  constrained decoding, NO ACT-R buffers/rule logic, NO validation.
- **B — ``constrained-decoding``**: LLM + the Constrained Decoder only (structured JSON
  output). NO ACT-R rule logic, NO symbolic validation/repair.
- **C — ``actr-no-validation``**: the FULL orchestrator pipeline (constrained decoding +
  ACT-R Controller buffers + rule selection) but wired with a
  :class:`NoOpValidationEngine` that ALWAYS accepts — no rejection, no repair.
- **D — ``full-neuro-symbolic``**: the full pipeline with the
  :class:`~demo.arithmetic_validation.ArithmeticValidationEngine` and the bounded repair
  sub-loop (identical to ``RealMathSystem``).

Every configuration is a :class:`~nsr.evaluation_harness.SystemUnderTest` whose
``run(query)`` returns a :class:`~nsr.models.VerifiedOutput` (or an
:class:`~nsr.models.ErrorRecord`) carrying a :class:`~nsr.models.ProofTrace`, so the
Evaluation Harness measures all four uniformly. Each config is constructed over a real
:class:`~nsr.llm_component.OllamaBackend` (built per query, exactly like ``RealBaseline`` /
``RealMathSystem``) and uses the :class:`~scenarios.MathTranslationLayer` so the model emits
``{"logic_form": "<expr> = <result>"}`` where appropriate.

HONESTY NOTE — *faithfulness* and *step-hallucination rate* are only substantive when the
Validation Engine can actually reject a step. Configs A and B have no validation and config
C always-accepts, so for those three configs these two metrics are **not meaningful** and
are reported as ``n/a`` (``None`` in the JSON). They carry real values only for config D
(the full system). **Accuracy** and **latency** are reported for all four configs.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# --- Make ``nsr`` (in src/) and the sibling demo modules importable when run directly ---
_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _DEMO_DIR.parent
for _p in (str(_PROJECT_DIR / "src"), str(_DEMO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nsr.baselines import BaselineResult  # noqa: E402
from nsr.comparison_report import build_comparison_report  # noqa: E402
from nsr.constrained_decoder import ConstrainedDecoder, ConstraintUnsatisfied  # noqa: E402
from nsr.evaluation_harness import (  # noqa: E402
    LLM_ONLY_METHOD_NAME,
    EvaluationHarness,
)
from nsr.baselines import build_baseline  # noqa: E402
from nsr.llm_component import (  # noqa: E402
    LLMComponent,
    LLMError,
    build_ollama_backend,
)
from nsr.metrics_engine import (  # noqa: E402
    compute_faithfulness_score,
    compute_step_hallucination_rate,
)
from nsr.models import (  # noqa: E402
    ErrorRecord,
    Goal,
    ProductionRule,
    SubGoal,
    SymbolicRepresentation,
    TerminationReason,
    ValidationStatus,
    VerifiedOutput,
    WorkingMemoryState,
)
from nsr.proof_trace import ProofTraceBuilder  # noqa: E402
from nsr.validation_engine import (  # noqa: E402
    ValidationEngine,
    ValidationOutcome,
)

import scenarios  # noqa: E402
from arithmetic_validation import ArithmeticValidationEngine  # noqa: E402
from goal_alignment import GoalAlignmentValidationEngine  # noqa: E402

# Reuse the existing real-run scaffolding so the ablation shares the same config, dataset
# loader, numeric matcher, starter rule set, and output directory as the GSM8K benchmark.
import run_benchmark  # noqa: E402
from run_benchmark import (  # noqa: E402
    DEFAULT_GSM8K_MODEL,
    OUTPUT_DIR,
    STARTER_PRODUCTION_RULES,
    _gsm8k_dataset,
    make_real_config,
    numeric_answer_match,
)


# --------------------------------------------------------------------------- #
# Configuration names + display metadata
# --------------------------------------------------------------------------- #

#: Config A — plain LLM (the ``llm-only`` baseline behaviour).
PLAIN_LLM = "plain-llm"
#: Config B — LLM + Constrained Decoder only.
CONSTRAINED_DECODING = "constrained-decoding"
#: Config C — full pipeline with an always-accepting (no-op) Validation Engine.
ACTR_NO_VALIDATION = "actr-no-validation"
#: Config D — the full neuro-symbolic system (constrained decoding + ACT-R + arithmetic).
FULL_NEURO_SYMBOLIC = "full-neuro-symbolic"
#: Config E — the full system with GOAL-ALIGNED validation (arithmetic + intent).
ARITHMETIC_GOAL = "arithmetic+goal"

#: All five config names in presentation order (A, B, C, D, E).
ABLATION_CONFIGS: tuple[str, ...] = (
    PLAIN_LLM,
    CONSTRAINED_DECODING,
    ACTR_NO_VALIDATION,
    FULL_NEURO_SYMBOLIC,
    ARITHMETIC_GOAL,
)

#: The ablations compared against the full system (mapped as harness baselines). Config D
#: (full arithmetic) is the harness ``system``; A/B/C and E (goal-aligned) are baselines.
ABLATION_BASELINES: tuple[str, ...] = (
    PLAIN_LLM,
    CONSTRAINED_DECODING,
    ACTR_NO_VALIDATION,
    ARITHMETIC_GOAL,
)

#: Human-friendly labels for the report.
CONFIG_LABELS = {
    PLAIN_LLM: "A · Plain LLM",
    CONSTRAINED_DECODING: "B · Constrained decoding",
    ACTR_NO_VALIDATION: "C · ACT-R, no validation",
    FULL_NEURO_SYMBOLIC: "D · Full (arithmetic)",
    ARITHMETIC_GOAL: "E · Full (arithmetic+goal)",
}

#: One-line description of what each config does / does not include.
CONFIG_DESCRIPTIONS = {
    PLAIN_LLM: (
        "Model answers directly (llm-only baseline). No constrained decoding, no ACT-R "
        "buffers/rule logic, no validation."
    ),
    CONSTRAINED_DECODING: (
        "LLM + Constrained Decoder only (structured JSON output). No ACT-R rule logic, "
        "no symbolic validation or repair. Constrained decoding ALONE is INSUFFICIENT "
        "(not 'bad'): on its own it forces a single structured equation, which cannot "
        "represent iterative multi-step reasoning, so it underperforms on GSM8K-style "
        "problems — the ACT-R control loop is what makes structured multi-step reasoning "
        "work."
    ),
    ACTR_NO_VALIDATION: (
        "Full pipeline (constrained decoding + ACT-R Controller buffers + rule selection) "
        "but with a no-op Validation Engine that always accepts — no rejection, no repair."
    ),
    FULL_NEURO_SYMBOLIC: (
        "Full neuro-symbolic system: constrained decoding + ACT-R Controller + arithmetic "
        "validation with the bounded repair sub-loop (same as RealMathSystem). Checks that "
        "each equation is arithmetically CORRECT."
    ),
    ARITHMETIC_GOAL: (
        "Full neuro-symbolic system with GOAL-ALIGNED validation: arithmetic correctness "
        "PLUS a check that the final-answer step computes the QUANTITY the goal asked for "
        "(intent). Rejects a correct-but-wrong-quantity step (e.g. the COST when the goal "
        "asks for PROFIT) and routes it to the bounded repair sub-loop. The validator is "
        "built per query, since the query IS the goal."
    ),
}

#: The two metrics that are only meaningful when validation can actually reject a step.
#: They are reported as ``n/a`` (None) for configs A, B, and C; the two VALIDATING configs
#: (D arithmetic, E arithmetic+goal) carry real values.
_NA_FOR_ABLATIONS = ("faithfulness", "step_hallucination_rate")

#: Configs for which faithfulness / step-hallucination are NOT meaningful (no real
#: step-rejection is possible), so they are shown as ``n/a``.
_NA_CONFIGS = (PLAIN_LLM, CONSTRAINED_DECODING, ACTR_NO_VALIDATION)

#: Configs whose faithfulness / step-hallucination ARE meaningful (validation can reject).
_VALIDATING_CONFIGS = (FULL_NEURO_SYMBOLIC, ARITHMETIC_GOAL)

#: Pretty metric labels for the report table.
METRIC_LABELS = {
    "final_answer_accuracy": "Accuracy (↑)",
    "faithfulness": "Faithfulness (↑)",
    "step_hallucination_rate": "Step hallucination rate (↓)",
    "mean_latency": "Mean latency",
    "p95_latency": "p95 latency",
    "latency_overhead": "Latency overhead vs Plain LLM",
    "reasoning_consistency": "Reasoning consistency (↑)",
}

_LATENCY_METRICS = {"mean_latency", "p95_latency", "latency_overhead"}

#: Matches a signed integer or decimal, optionally with thousands separators.
_NUMERIC_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


# --------------------------------------------------------------------------- #
# No-op Validation Engine (config C)
# --------------------------------------------------------------------------- #


class NoOpValidationEngine(ValidationEngine):
    """A Validation Engine that ALWAYS accepts any step (config C).

    Overrides :meth:`~nsr.validation_engine.ValidationEngine.validate` to return an
    ``ACCEPTED`` :class:`~nsr.validation_engine.ValidationOutcome` for *any* representation,
    with empty applicable/violated lists and no evaluations. Because nothing is ever
    rejected, no step is routed to repair — so config C exercises the full ACT-R pipeline
    (constrained decoding + controller buffers + rule selection) *without* genuine
    content-level validation. This is why faithfulness / step-hallucination are reported as
    ``n/a`` for config C: an always-accepting engine cannot make those metrics meaningful.
    """

    def validate(
        self,
        rep: SymbolicRepresentation,
        rules: list[ProductionRule],
    ) -> ValidationOutcome:
        return ValidationOutcome(
            status=ValidationStatus.ACCEPTED,
            representation=rep,
            applicable_rule_ids=[],
            violated_rule_ids=[],
            violated_rules=[],
            evaluations=[],
        )


# --------------------------------------------------------------------------- #
# Numeric extraction helpers
# --------------------------------------------------------------------------- #


def _last_number(text: object) -> Optional[float]:
    """Return the LAST number mentioned in ``text`` (the ``= <result>`` RHS), or ``None``.

    Tolerant of thousands separators, a leading ``$`` and a trailing period. For an
    ``"<expr> = <result>"`` logic form the last number is the right-hand-side result.
    """
    if text is None:
        return None
    matches = _NUMERIC_RE.findall(str(text))
    if not matches:
        return None
    token = matches[-1].replace(",", "").strip().strip("$").rstrip(".")
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _render_number(value: float) -> str:
    """Render a float as a bare integer when it is integral, else as a plain float."""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return str(value)


# --------------------------------------------------------------------------- #
# The four configurations (each a SystemUnderTest)
# --------------------------------------------------------------------------- #


class PlainLLMConfig:
    """Config A — the model answers directly (reuses the ``llm-only`` baseline).

    No constrained decoding, no ACT-R, no validation. ``run`` asks the model once via the
    real ``llm-only`` baseline over an Ollama backend and wraps the answer in a minimal
    :class:`~nsr.models.VerifiedOutput`. The trace records the single produced step but
    carries NO genuine symbolic validation, so faithfulness/hallucination are not
    meaningful here (reported as ``n/a``).
    """

    name = PLAIN_LLM

    def __init__(self, model: str, host: Optional[str], config) -> None:
        self._model = model
        self._host = host
        self._config = config

    def run(self, query: object):
        backend = build_ollama_backend(self._model, host=self._host)
        method = build_baseline(LLM_ONLY_METHOD_NAME, backend)
        try:
            result = method.run(str(query))
        except LLMError as exc:
            return ErrorRecord(failed_component="LLM", reason=str(exc))

        builder = ProofTraceBuilder()
        raw = result.raw_outputs[0] if result.raw_outputs else result.final_answer
        builder.append_step(
            str(raw),
            representation=SymbolicRepresentation(
                logic_form=result.final_answer, source_text=str(raw)
            ),
            status=ValidationStatus.ACCEPTED,
            applied_rule_id=None,
        )
        builder.set_termination_reason(TerminationReason.GOAL_SATISFIED)
        builder.record_latency(float(result.latency_ms))
        trace = builder.trace
        return VerifiedOutput(
            final_answer=result.final_answer,
            proof_trace=trace,
            faithfulness_score=compute_faithfulness_score(trace),
        )


class ConstrainedDecodingConfig:
    """Config B — LLM + Constrained Decoder only (structured JSON), no ACT-R, no validation.

    ``run`` derives a prompt from the :class:`~scenarios.MathTranslationLayer`, runs the
    Constrained Decoder to obtain a conforming :class:`~nsr.models.CandidateStep` (forcing
    the JSON ``logic_form``), takes the ``"<expr> = <result>"`` RHS via numeric extraction
    as the final answer, and builds a 1-step :class:`~nsr.models.ProofTrace`. There is no
    ACT-R rule logic and no symbolic validation/repair, so faithfulness/hallucination are
    not meaningful here (reported as ``n/a``).
    """

    name = CONSTRAINED_DECODING

    def __init__(self, model: str, host: Optional[str], config) -> None:
        self._model = model
        self._host = host
        self._config = config

    def run(self, query: object):
        backend = build_ollama_backend(self._model, host=self._host)
        llm = LLMComponent(backend, self._config)
        decoder = ConstrainedDecoder(llm, self._config)
        translation = scenarios.MathTranslationLayer()

        goal = Goal(description=str(query), sub_goals=[SubGoal(description=str(query))])
        state = WorkingMemoryState(goal_buffer=goal)

        start = time.perf_counter()
        try:
            context = translation.to_context(state)
            candidate = decoder.decode(context, state)
        except (ConstraintUnsatisfied, LLMError) as exc:
            return ErrorRecord(failed_component="Constrained_Decoder", reason=str(exc))
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        logic_form = candidate.structured.get("logic_form") or candidate.raw_text or ""
        predicates = candidate.structured.get("predicates", {})
        if not isinstance(predicates, dict):
            predicates = {}

        rhs = _last_number(logic_form)
        final_answer = _render_number(rhs) if rhs is not None else str(logic_form)

        builder = ProofTraceBuilder()
        builder.append_step(
            candidate.raw_text,
            representation=SymbolicRepresentation(
                logic_form=str(logic_form),
                predicates=dict(predicates),
                source_text=candidate.raw_text,
            ),
            status=ValidationStatus.ACCEPTED,
            applied_rule_id=None,
        )
        builder.set_termination_reason(TerminationReason.GOAL_SATISFIED)
        builder.record_latency(elapsed_ms)
        trace = builder.trace
        return VerifiedOutput(
            final_answer=final_answer,
            proof_trace=trace,
            faithfulness_score=compute_faithfulness_score(trace),
        )


class ActrNoValidationConfig:
    """Config C — full ACT-R pipeline but with an always-accepting Validation Engine.

    Wires the real orchestrator (constrained decoding + ACT-R Controller buffers + rule
    selection) with a :class:`NoOpValidationEngine` and ``with_repair=False``, seeded with
    the general-purpose starter production-rule set and the
    :class:`~scenarios.MathTranslationLayer`. Because validation always accepts, no step is
    ever rejected or repaired, so faithfulness/hallucination are not meaningful (``n/a``).
    """

    name = ACTR_NO_VALIDATION

    def __init__(self, model: str, host: Optional[str], config) -> None:
        self._model = model
        self._host = host
        self._config = config

    def run(self, query: object):
        backend = build_ollama_backend(self._model, host=self._host)
        orchestrator = scenarios.build_orchestrator_with_backend(
            backend=backend,
            procedural_memory=STARTER_PRODUCTION_RULES,
            config=self._config,
            translation=scenarios.MathTranslationLayer(),
            validation=NoOpValidationEngine(),
            with_repair=False,
        )
        return orchestrator.run(query)


class FullNeuroSymbolicConfig:
    """Config D — the full neuro-symbolic system (same wiring as ``RealMathSystem``).

    Wires the real orchestrator with the
    :class:`~demo.arithmetic_validation.ArithmeticValidationEngine`, the
    :class:`~scenarios.MathTranslationLayer`, the starter production-rule set, and the
    bounded repair sub-loop. This is the only config whose faithfulness/hallucination are
    genuine, because its Validation Engine can actually reject (and repair) a wrong step.
    """

    name = FULL_NEURO_SYMBOLIC

    def __init__(self, model: str, host: Optional[str], config) -> None:
        self._model = model
        self._host = host
        self._config = config

    def run(self, query: object):
        backend = build_ollama_backend(self._model, host=self._host)
        orchestrator = scenarios.build_orchestrator_with_backend(
            backend=backend,
            procedural_memory=STARTER_PRODUCTION_RULES,
            config=self._config,
            translation=scenarios.MathTranslationLayer(),
            validation=ArithmeticValidationEngine(),
            with_repair=True,
        )
        return orchestrator.run(query)


class FullPlusGoalConfig:
    """Config E — the full system with GOAL-ALIGNED validation (arithmetic + intent).

    Identical wiring to :class:`FullNeuroSymbolicConfig`, but the Validation Engine is a
    :class:`~goal_alignment.GoalAlignmentValidationEngine` built **per query** (the query IS
    the goal). On top of arithmetic correctness it checks that the final-answer step
    computes the QUANTITY the goal asked for — rejecting a correct-but-wrong-quantity step
    (e.g. answering the COST when the goal asks for PROFIT) and routing it to the bounded
    repair sub-loop. Like config D, its faithfulness/hallucination are genuine because its
    Validation Engine can actually reject (and repair) a step.
    """

    name = ARITHMETIC_GOAL

    def __init__(self, model: str, host: Optional[str], config) -> None:
        self._model = model
        self._host = host
        self._config = config

    def run(self, query: object):
        backend = build_ollama_backend(self._model, host=self._host)
        orchestrator = scenarios.build_orchestrator_with_backend(
            backend=backend,
            procedural_memory=STARTER_PRODUCTION_RULES,
            config=self._config,
            translation=scenarios.MathTranslationLayer(),
            validation=GoalAlignmentValidationEngine(goal_text=str(query)),
            with_repair=True,
        )
        return orchestrator.run(query)


class _ConfigAsBaseline:
    """Adapt a :class:`SystemUnderTest` config into a harness ``ReasoningMethod`` baseline.

    The Evaluation Harness runs one ``system`` against several ``baselines``. To compare
    the full system (config D, mapped as the harness ``system``) against the three
    ablations (A/B/C, mapped as baselines), each ablation config — which natively returns a
    :class:`~nsr.models.VerifiedOutput` — is wrapped so it conforms to the baseline
    ``run(query) -> BaselineResult`` protocol. The wall-clock latency of the config's run is
    measured (preferring the trace's recorded pipeline latency when present), and an
    :class:`~nsr.models.ErrorRecord` result raises so the harness isolates and excludes the
    item (Req 9.7), exactly as for any other baseline failure.
    """

    def __init__(self, name: str, system, *, clock=time.perf_counter) -> None:
        self.name = name
        self._system = system
        self._clock = clock
        #: Per-item trace-derived metrics captured from any VerifiedOutput this config
        #: produces, so a *validating* ablation config (e.g. ``arithmetic+goal``) can report
        #: REAL faithfulness / step-hallucination even though it is mapped as a harness
        #: baseline (baselines otherwise contribute ``0.0`` for these). Non-validating
        #: configs simply produce uninteresting values here and are shown as ``n/a``.
        self.captured_faithfulness: list[float] = []
        self.captured_hallucination: list[float] = []

    def run(self, query: str) -> BaselineResult:
        start = self._clock()
        result = self._system.run(query)
        elapsed_ms = (self._clock() - start) * 1000.0
        if isinstance(result, ErrorRecord):
            raise RuntimeError(
                f"{self.name} returned error from {result.failed_component}: "
                f"{result.reason}"
            )
        trace = result.proof_trace
        if trace is not None and trace.latency is not None:
            latency_ms = float(trace.latency.pipeline_ms)
        else:
            latency_ms = elapsed_ms
        if trace is not None:
            self.captured_faithfulness.append(float(result.faithfulness_score))
            self.captured_hallucination.append(
                float(compute_step_hallucination_rate(trace))
            )
        return BaselineResult(
            method=self.name,
            final_answer=result.final_answer,
            latency_ms=latency_ms,
        )

    @property
    def mean_faithfulness(self) -> Optional[float]:
        """Mean captured faithfulness across all VerifiedOutputs, or ``None`` if none."""
        if not self.captured_faithfulness:
            return None
        return sum(self.captured_faithfulness) / len(self.captured_faithfulness)

    @property
    def mean_hallucination(self) -> Optional[float]:
        """Mean captured step-hallucination across all VerifiedOutputs, or ``None``."""
        if not self.captured_hallucination:
            return None
        return sum(self.captured_hallucination) / len(self.captured_hallucination)


def build_ablation_configs(model: str, host: Optional[str], config):
    """Construct the five configurations, returning ``(system, baselines)``.

    ``system`` is the full neuro-symbolic config (D, arithmetic validation); ``baselines``
    maps each other config name (A/B/C and E, the goal-aligned full system) to a
    :class:`_ConfigAsBaseline` wrapper so the comparison report shows the full system versus
    each. Config E is a *validating* config, so its wrapper captures real faithfulness /
    step-hallucination from each Proof Trace (see :class:`_ConfigAsBaseline`).
    """
    system = FullNeuroSymbolicConfig(model, host, config)
    baselines = {
        PLAIN_LLM: _ConfigAsBaseline(PLAIN_LLM, PlainLLMConfig(model, host, config)),
        CONSTRAINED_DECODING: _ConfigAsBaseline(
            CONSTRAINED_DECODING, ConstrainedDecodingConfig(model, host, config)
        ),
        ACTR_NO_VALIDATION: _ConfigAsBaseline(
            ACTR_NO_VALIDATION, ActrNoValidationConfig(model, host, config)
        ),
        ARITHMETIC_GOAL: _ConfigAsBaseline(
            ARITHMETIC_GOAL, FullPlusGoalConfig(model, host, config)
        ),
    }
    return system, baselines


# --------------------------------------------------------------------------- #
# Report assembly (per-config values with the honest n/a handling)
# --------------------------------------------------------------------------- #


def _overhead_vs_plain_llm(primary_run) -> dict[str, float]:
    """Per-config mean latency overhead vs the ``plain-llm`` config (ms).

    Computed as the mean per-query latency difference over the items both the config and
    ``plain-llm`` successfully evaluated (the shared query set). ``plain-llm`` is the
    reference, so its overhead is ``0.0`` by definition; a config with no shared items maps
    to ``0.0``. This recomputation is needed because the harness measures overhead against
    its built-in ``llm-only`` reference, which is not one of the ablation method names.
    """
    plain_latency = {
        o.item_id: o.latency_ms
        for o in primary_run.per_item_outcomes.get(PLAIN_LLM, [])
    }
    overheads: dict[str, float] = {}
    for config in ABLATION_CONFIGS:
        outcomes = primary_run.per_item_outcomes.get(config, [])
        diffs = [
            o.latency_ms - plain_latency[o.item_id]
            for o in outcomes
            if o.item_id in plain_latency
        ]
        overheads[config] = (sum(diffs) / len(diffs)) if diffs else 0.0
    return overheads


def _config_value(comparison, config: str):
    """Pull one config's value out of a :class:`MetricComparison`."""
    if config == FULL_NEURO_SYMBOLIC:
        return comparison.system_value
    return comparison.baseline_values.get(config)


def build_ablation_metrics(report, primary_run, extra_values=None) -> list[dict]:
    """Build the per-metric, per-config value table with the honest n/a handling.

    For every metric in the comparison report, records a ``{metric, label, values}`` entry
    whose ``values`` maps each of the five config names to its value. Faithfulness and
    step-hallucination rate are forced to ``None`` (``n/a``) for the NON-validating configs
    A/B/C — they are only substantive when a Validation Engine can actually reject a step.
    The two validating configs carry real values: config D (the harness system) from the
    comparison report, and config E (``arithmetic+goal``, mapped as a baseline) from
    ``extra_values`` — trace-derived faithfulness/hallucination captured during the run,
    since the harness only computes those for its single ``system`` method. The
    latency-overhead row is recomputed relative to ``plain-llm`` for all configs.

    Args:
        report: The comparison report.
        primary_run: The first :class:`EvaluationRunResult` (for latency overhead).
        extra_values: Optional ``{config: {metric: value}}`` overrides applied last, used to
            supply config E's real faithfulness / step-hallucination.
    """
    extra_values = extra_values or {}
    overheads = _overhead_vs_plain_llm(primary_run)
    rows: list[dict] = []
    for comparison in report.metrics:
        metric = comparison.metric
        values: dict[str, Optional[float]] = {
            config: _config_value(comparison, config) for config in ABLATION_CONFIGS
        }
        if metric in _NA_FOR_ABLATIONS:
            for config in _NA_CONFIGS:
                values[config] = None
        if metric == "latency_overhead":
            values = {config: overheads.get(config) for config in ABLATION_CONFIGS}
        # Apply explicit per-config overrides last (e.g. config E's real faithfulness).
        for config, overrides in extra_values.items():
            if metric in overrides:
                values[config] = overrides[metric]
        rows.append(
            {
                "metric": metric,
                "label": METRIC_LABELS.get(metric, metric),
                "values": values,
            }
        )
    return rows


_ABLATION_NOTE = (
    "This is an ABLATION STUDY: all four configurations are evaluated on the SAME GSM8K "
    "subset so each architectural component's contribution can be isolated. Accuracy and "
    "latency are reported for ALL four configs. Faithfulness and step-hallucination rate "
    "are only substantive when the Validation Engine can actually REJECT a step — configs "
    "A (plain-llm) and B (constrained-decoding) have no validation, and config C "
    "(actr-no-validation) always accepts — so for those three configs these two metrics "
    "are marked n/a and shown only for config D (full-neuro-symbolic). "
    "Constrained decoding ALONE is insufficient (not 'bad'): in this implementation it "
    "forces a single structured equation, which cannot represent iterative multi-step "
    "reasoning, so it underperforms on GSM8K-style problems; the ACT-R control loop is "
    "what makes structured multi-step reasoning work."
)


def _ablation_to_dict(report, primary_run, dataset, dataset_label, model, extra_values=None) -> dict:
    """A JSON-serializable view of the ablation comparison."""
    metrics = build_ablation_metrics(report, primary_run, extra_values)
    return {
        "mode": "ablation",
        "is_real": True,
        "study": "ablation",
        "real_model": model,
        "dataset_label": dataset_label,
        "note": _ABLATION_NOTE,
        "system_method": report.system_method,
        "configs": list(ABLATION_CONFIGS),
        "config_labels": dict(CONFIG_LABELS),
        "config_descriptions": dict(CONFIG_DESCRIPTIONS),
        "na_metrics_for_ablations": list(_NA_FOR_ABLATIONS),
        "baseline_methods": list(report.baseline_methods),
        "repeated_run_count": report.repeated_run_count,
        "model_id": primary_run.run_record.model_id,
        "seed": primary_run.run_record.seed,
        "dataset": [
            {"item_id": i.item_id, "domain": i.domain.value, "ground_truth": i.ground_truth}
            for i in dataset
        ],
        "metrics": metrics,
    }


def _format_value(metric: str, value) -> str:
    """Render a metric value for the HTML table; ``None`` becomes ``n/a``."""
    if value is None:
        return "n/a"
    if metric in _LATENCY_METRICS:
        return f"{value:.2f} ms"
    return f"{value:.3f}"


def _build_ablation_html(report, primary_run, dataset, dataset_label, model, extra_values=None) -> str:
    """Render the ablation comparison as a self-contained HTML table."""
    metrics = build_ablation_metrics(report, primary_run, extra_values)

    header_cells = "<th>Metric</th>" + "".join(
        f"<th>{html.escape(CONFIG_LABELS[config])}<br/>"
        f"<code>{html.escape(config)}</code></th>"
        for config in ABLATION_CONFIGS
    )

    rows = []
    for row in metrics:
        metric = row["metric"]
        cells = [f'<td class="metric">{html.escape(row["label"])}</td>']
        for config in ABLATION_CONFIGS:
            value = row["values"].get(config)
            text = _format_value(metric, value)
            cls = ' class="na"' if value is None else ""
            emphasis = config == FULL_NEURO_SYMBOLIC
            inner = f"<b>{text}</b>" if emphasis else text
            cells.append(f"<td{cls}>{inner}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    config_rows = "".join(
        f"<tr><td class=\"metric\">{html.escape(CONFIG_LABELS[config])} "
        f"(<code>{html.escape(config)}</code>)</td>"
        f"<td>{html.escape(CONFIG_DESCRIPTIONS[config])}</td></tr>"
        for config in ABLATION_CONFIGS
    )

    css = """
    body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
           margin: 0; color: #1a1a1a; background: #f6f7f9; line-height: 1.5; }
    header { background: #0f2540; color: #fff; padding: 24px 32px; }
    header h1 { margin: 0 0 4px; font-size: 22px; }
    header p { margin: 0; color: #b9c7d8; font-size: 14px; }
    main { max-width: 1040px; margin: 0 auto; padding: 24px 32px 64px; }
    .section { background:#fff; border:1px solid #e2e6ea; border-radius:10px;
               padding:20px 24px; margin:20px 0; }
    .ablation { background:#fff3e0; border:1px solid #ffcc80; color:#7a4f01;
                border-radius:8px; padding:12px 16px; font-size:13px; }
    .real { background:#e8f5e9; border:1px solid #a5d6a7; color:#1b5e20;
            border-radius:8px; padding:10px 14px; font-size:13px; margin-top:10px; }
    table { width:100%; border-collapse:collapse; margin-top:8px; font-size:14px; }
    th, td { padding:10px 12px; border-bottom:1px solid #e2e6ea; text-align:right;
             vertical-align:top; }
    th:first-child, td:first-child { text-align:left; }
    thead th { background:#eef1f4; }
    td.metric { font-weight:600; }
    td.na { color:#9aa3ad; font-style:italic; }
    code { background:#eef1f4; padding:1px 5px; border-radius:4px; font-size:12px; }
    .kv { font-size:13px; color:#444; }
    footer { text-align:center; color:#98a2ad; font-size:12px; padding:20px; }
    """

    domain_counts: dict[str, int] = {}
    for item in dataset:
        domain_counts[item.domain.value] = domain_counts.get(item.domain.value, 0) + 1
    domain_list = ", ".join(f"{d} ({n})" for d, n in sorted(domain_counts.items()))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NSR Ablation Study — GSM8K (real model via Ollama)</title>
<style>{css}</style>
</head>
<body>
<header>
  <h1>Neuro-Symbolic System — Ablation Study (GSM8K)</h1>
  <p>Four configurations on the SAME subset: plain-llm · constrained-decoding ·
     actr-no-validation · full-neuro-symbolic — real model {html.escape(str(model))}
     via Ollama</p>
</header>
<main>
  <div class="section">
    <p class="ablation"><b>Ablation study.</b> {html.escape(_ABLATION_NOTE)}</p>
    <p class="real"><b>Real model: {html.escape(str(model))} via Ollama.</b> Every
      configuration's answers are genuine model output; config D additionally produces a
      real Proof Trace and Faithfulness Score under live arithmetic validation.</p>
    <p class="kv">Dataset: <b>{len(dataset)}</b> items — {html.escape(domain_list)}.
      {html.escape(str(dataset_label))}. Model id:
      <b>{html.escape(primary_run.run_record.model_id)}</b>. Seed:
      <b>{primary_run.run_record.seed}</b>. Repeated runs:
      <b>{report.repeated_run_count}</b>.</p>
  </div>
  <div class="section">
    <h2>Configurations</h2>
    <table><tbody>
{config_rows}
    </tbody></table>
  </div>
  <div class="section">
    <h2>Per-component comparison</h2>
    <table>
      <thead><tr>{header_cells}</tr></thead>
      <tbody>
{chr(10).join(rows)}
      </tbody>
    </table>
    <p class="kv">Faithfulness and step-hallucination rate are <b>n/a</b> for plain-llm,
      constrained-decoding, and actr-no-validation (no real step rejection is possible) and
      are reported only for full-neuro-symbolic. Accuracy and latency are reported for all
      four. Latency overhead is measured relative to <code>plain-llm</code>.</p>
  </div>
</main>
<footer>Generated by demo/ablation.py — Neuro-Symbolic System-2 Reasoning Architecture</footer>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Public runner
# --------------------------------------------------------------------------- #


def generate_ablation_gsm8k(
    model: str = DEFAULT_GSM8K_MODEL,
    host: Optional[str] = None,
    dataset_path: os.PathLike | str | None = None,
    limit: int = 5,
    repeated_run_count: int = 1,
    output_dir: os.PathLike | str = OUTPUT_DIR,
) -> dict[str, Path]:
    """Run the four-config GSM8K ablation against a REAL Ollama model and write reports.

    Loads the GSM8K subset via :func:`run_benchmark._gsm8k_dataset` (so all four configs see
    the SAME items), runs the Evaluation Harness with :func:`run_benchmark.numeric_answer_match`
    mapping config D as the harness ``system`` and configs A/B/C as ``baselines``, builds the
    comparison report, and writes ``benchmark_report_ablation_gsm8k.html`` / ``.json``
    comparing the four configs across accuracy, faithfulness, step-hallucination rate, and
    latency (mean + overhead). Faithfulness/hallucination are recorded as ``n/a`` (None) for
    A/B/C and only carry real values for config D.

    Performs **no preflight** — callers (the CLI) should call
    :func:`nsr.llm_component.ollama_available` first. Exercised offline in tests by
    monkeypatching :data:`build_ollama_backend`. Returns the written file paths.
    """
    config = make_real_config(model, repeated_run_count=repeated_run_count)
    dataset, dataset_label = _gsm8k_dataset(dataset_path, limit)

    system, baselines = build_ablation_configs(model, host, config)

    harness = EvaluationHarness(
        system,
        baselines,
        answer_match=numeric_answer_match,
        system_method_name=FULL_NEURO_SYMBOLIC,
    )
    runs = [
        harness.run(dataset, config=config, model_id=f"ollama:{model}")
        for _ in range(config.repeated_run_count)
    ]
    report = build_comparison_report(
        runs,
        repeated_run_count=config.repeated_run_count,
        system_method_name=FULL_NEURO_SYMBOLIC,
    )
    primary = runs[0]

    # Config E (arithmetic+goal) is a validating config mapped as a harness baseline, so the
    # harness does not compute its trace-derived faithfulness/hallucination. Pull the REAL
    # values its wrapper captured from each Proof Trace and inject them into the table.
    goal_baseline = baselines.get(ARITHMETIC_GOAL)
    extra_values = {}
    if goal_baseline is not None:
        extra_values[ARITHMETIC_GOAL] = {
            "faithfulness": goal_baseline.mean_faithfulness,
            "step_hallucination_rate": goal_baseline.mean_hallucination,
        }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "html": out / "benchmark_report_ablation_gsm8k.html",
        "json": out / "benchmark_report_ablation_gsm8k.json",
    }
    paths["html"].write_text(
        _build_ablation_html(report, primary, dataset, dataset_label, model, extra_values),
        encoding="utf-8",
    )
    paths["json"].write_text(
        json.dumps(
            _ablation_to_dict(
                report, primary, dataset, dataset_label, model, extra_values
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "PLAIN_LLM",
    "CONSTRAINED_DECODING",
    "ACTR_NO_VALIDATION",
    "FULL_NEURO_SYMBOLIC",
    "ARITHMETIC_GOAL",
    "ABLATION_CONFIGS",
    "NoOpValidationEngine",
    "PlainLLMConfig",
    "ConstrainedDecodingConfig",
    "ActrNoValidationConfig",
    "FullNeuroSymbolicConfig",
    "FullPlusGoalConfig",
    "build_ablation_configs",
    "build_ablation_metrics",
    "generate_ablation_gsm8k",
]
