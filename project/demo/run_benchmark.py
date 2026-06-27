"""CLI: run a small offline multi-domain benchmark and write a comparison report.

Usage (from the ``project/`` directory)::

    python demo/run_benchmark.py                          # offline mock (default)
    python demo/run_benchmark.py --backend ollama --model llama3.1   # REAL model

By default this wires the **Neuro-Symbolic System** (the real orchestrator over a scripted
:class:`~nsr.llm_component.MockBackend`) against two baselines — **Plain LLM** (``llm-only``)
and **Chain-of-Thought** — also driven by scripted MockBackends. The real
:class:`~nsr.evaluation_harness.EvaluationHarness` runs every method over a small in-memory
dataset spanning three benchmark domains, repeated runs feed Reasoning Consistency, and a
:func:`~nsr.comparison_report.build_comparison_report` comparison is written to
``project/demo/output/`` as both HTML and JSON.

The default mode is **offline and deterministic**: answers come from scripted backends and
per-method latencies come from injected step clocks, so the numbers are reproducible. They
are illustrative of the architecture's behavior, not measurements of a real LLM.

With ``--backend ollama`` the *same* dataset/harness/report flow runs against a **real
model served by a local `Ollama <https://ollama.com>`_ instance** (Llama 3.1, Mistral,
Qwen, Phi, Gemma, ...). In that mode both the System and the baselines produce **genuine
model answers**; the report is written to ``benchmark_report_real.html`` / ``.json`` and is
clearly labelled with the model name. See :func:`generate_benchmark_real`.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

# --- Make ``nsr`` (in src/) and the sibling demo modules importable when run directly ---
_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _DEMO_DIR.parent
for _p in (str(_PROJECT_DIR / "src"), str(_DEMO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nsr.baselines import build_baseline  # noqa: E402
from nsr.comparison_report import REPORT_METRICS, build_comparison_report  # noqa: E402
from nsr.dataset_loader import load_dataset  # noqa: E402
from nsr.evaluation_harness import (  # noqa: E402
    LLM_ONLY_METHOD_NAME,
    SYSTEM_METHOD_NAME,
    EvaluationHarness,
)
from nsr.llm_component import (  # noqa: E402
    MockBackend,
    build_ollama_backend,
    ollama_available,
)
from nsr.models import Domain, ProductionRule, SystemConfig, VerifiedOutput  # noqa: E402

import datasets  # noqa: E402
import scenarios  # noqa: E402
from arithmetic_validation import ArithmeticValidationEngine  # noqa: E402
from goal_alignment import GoalAlignmentValidationEngine  # noqa: E402

#: Validation modes for the GSM8K System path. ``"arithmetic"`` (default) checks only that
#: each intermediate equation is arithmetically correct; ``"goal"`` additionally checks that
#: the final-answer step computes the QUANTITY the goal asked for (intent), rejecting a
#: correct-but-wrong-quantity step (e.g. answering the COST when the goal asks for PROFIT).
VALIDATION_ARITHMETIC = "arithmetic"
VALIDATION_GOAL = "goal"

OUTPUT_DIR = _DEMO_DIR / "output"

#: A rule that is always applicable and satisfied only when a step is marked "verified",
#: so the scripted System steps are validated before they advance the cycle.
VERIFY_RULE = ProductionRule(rule_id="calc-verified", condition="", action="THEN verified")

#: The baseline method labels compared against the System.
CHAIN_OF_THOUGHT = "chain-of-thought"
BASELINE_NAMES = (LLM_ONLY_METHOD_NAME, CHAIN_OF_THOUGHT)

#: Human-friendly names for the report.
DISPLAY_NAMES = {
    SYSTEM_METHOD_NAME: "Neuro-Symbolic System",
    LLM_ONLY_METHOD_NAME: "Plain LLM",
    CHAIN_OF_THOUGHT: "Chain-of-Thought",
}

#: Pretty metric labels for the report table.
METRIC_LABELS = {
    "final_answer_accuracy": "Accuracy (↑)",
    "step_hallucination_rate": "Step hallucination rate (↓)",
    "faithfulness": "Faithfulness (↑)",
    "mean_latency": "Mean latency",
    "p95_latency": "p95 latency",
    "latency_overhead": "Latency overhead vs Plain LLM",
    "reasoning_consistency": "Reasoning consistency (↑)",
}

_LATENCY_METRICS = {"mean_latency", "p95_latency", "latency_overhead"}

#: Per-method deterministic latency steps (milliseconds) injected via step clocks.
_SYSTEM_STEP_MS = 5.0
_BASELINE_STEP_MS = {LLM_ONLY_METHOD_NAME: 1.0, CHAIN_OF_THOUGHT: 2.5}


class StepClock:
    """A deterministic monotonic clock that advances by a fixed step on each call.

    Returns seconds. The difference between two consecutive calls is always ``step_ms``,
    so any ``end - start`` measurement yields exactly ``step_ms`` — giving reproducible,
    clearly-labelled mock latencies.
    """

    def __init__(self, step_ms: float) -> None:
        self._t = 0.0
        self._step = step_ms / 1000.0

    def __call__(self) -> float:
        value = self._t
        self._t += self._step
        return value


# --------------------------------------------------------------------------- #
# In-memory multi-domain dataset (loaded through the real DatasetLoader)
# --------------------------------------------------------------------------- #

RAW_ITEMS = [
    {"item_id": "math-1", "query": "Compute 2 plus 2.", "ground_truth": "4",
     "domain": Domain.MATH.value},
    {"item_id": "math-2", "query": "Compute 3 times 4.", "ground_truth": "12",
     "domain": Domain.MATH.value},
    {"item_id": "logic-1",
     "query": "All cats are mammals. then all mammals are animals. then are cats animals?",
     "ground_truth": "yes", "domain": Domain.LOGIC_PUZZLE.value},
    {"item_id": "commonsense-1", "query": "What color is the clear daytime sky?",
     "ground_truth": "blue", "domain": Domain.COMMONSENSE.value},
    {"item_id": "commonsense-2", "query": "What gas do plants absorb for photosynthesis?",
     "ground_truth": "carbon dioxide", "domain": Domain.COMMONSENSE.value},
]

#: System scripts: per item, the ordered logic forms (one per sub-goal). The last logic
#: form is the final answer and equals the ground truth, so the System is fully correct.
SYSTEM_LOGIC_FORMS = {
    "math-1": ["4"],
    "math-2": ["12"],
    "logic-1": ["cats_are_mammals", "mammals_are_animals", "yes"],
    "commonsense-1": ["blue"],
    "commonsense-2": ["carbon dioxide"],
}

#: Baseline completions (scripted). Plain LLM errs on the logic and science items;
#: Chain-of-Thought recovers the logic item but still errs on the harder science one.
BASELINE_COMPLETIONS = {
    LLM_ONLY_METHOD_NAME: {
        "math-1": "Answer: 4",
        "math-2": "Answer: 12",
        "logic-1": "Answer: no",
        "commonsense-1": "Answer: blue",
        "commonsense-2": "Answer: oxygen",
    },
    CHAIN_OF_THOUGHT: {
        "math-1": "Step by step: 2+2=4.\nAnswer: 4",
        "math-2": "Step by step: 3*4=12.\nAnswer: 12",
        "logic-1": "Cats→mammals→animals.\nAnswer: yes",
        "commonsense-1": "The sky scatters blue light.\nAnswer: blue",
        "commonsense-2": "Plants breathe.\nAnswer: oxygen",
    },
}


import re as _re


def lenient_answer_match(predicted: str, ground_truth: str) -> bool:
    """Lenient short-answer matcher used for the benchmark (applied to ALL methods).

    The harness default (`normalized_answer_match`) requires whitespace/case-normalized
    *equality*, which unfairly penalizes a correct-but-verbose answer (e.g. "Blue." or
    "Yes, cats are animals." against ground truth "blue" / "yes"). To compare reasoning
    fairly rather than output formatting, this matcher counts a prediction correct when,
    after lowercasing and stripping punctuation, the ground-truth phrase equals the
    prediction OR appears in it on word boundaries. The same matcher is applied to the
    System and every baseline, so no method gets a formatting advantage. (Caveat: pure
    containment can over-credit in adversarial cases; it is appropriate here for short
    factual answers.)
    """
    def norm(text: str) -> str:
        cleaned = _re.sub(r"[^0-9a-z\s]", " ", str(text or "").lower())
        return " ".join(cleaned.split())

    pred, gt = norm(predicted), norm(ground_truth)
    if not gt:
        return not pred
    if pred == gt:
        return True
    return _re.search(rf"\b{_re.escape(gt)}\b", pred) is not None


#: Matches a signed integer or decimal, optionally with thousands separators (``1,234``).
_NUMERIC_RE = _re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _last_number(text: object) -> float | None:
    """Extract the LAST number mentioned in ``text`` as a float, or ``None``.

    Tolerant of thousands separators (``1,200``), a leading ``$`` (``$5``), a trailing
    period (``72.``), and a trailing ``= N`` equation form (``x = 14`` → ``14``, since the
    last number is taken). Returns ``None`` when no number is present.
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


def numeric_answer_match(predicted: str, ground_truth: str) -> bool:
    """Numeric final-answer matcher for the GSM8K run (applied to ALL methods).

    Extracts the LAST number from the prediction and from the ground truth and compares
    them with a small tolerance, so "The answer is 72.", "72", "1,200", "$5", and the
    equation form "x = 14" all reduce to their final number. The same matcher is applied
    to the System and every baseline, so no method gets a formatting advantage. Returns
    ``False`` when either side has no parseable number.
    """
    predicted_value = _last_number(predicted)
    ground_truth_value = _last_number(ground_truth)
    if predicted_value is None or ground_truth_value is None:
        return False
    return abs(predicted_value - ground_truth_value) < 1e-6


def make_benchmark_config() -> SystemConfig:
    """The System configuration for the benchmark (repeated runs → consistency)."""
    return SystemConfig(
        max_cycle_limit=10,
        repair_attempt_limit=2,
        retry_count=0,
        llm_selection="gpt-4o-mini",
        output_format="json",
        conflict_resolution_policy="priority",
        generation_timeout_ms=30000,
        repeated_run_count=2,
        random_seed=7,
    )


class ScriptedSystem:
    """A System-under-test that builds a fresh, fully-wired offline orchestrator per query.

    Conforms to the harness's ``run(query) -> VerifiedOutput | ErrorRecord`` protocol. For
    each query it looks up the scripted logic forms for the matching dataset item and runs
    the **real** orchestrator over a scripted :class:`MockBackend`, so the System produces a
    genuine Proof Trace and Faithfulness Score with no network.
    """

    def __init__(self, query_to_item: dict[str, str], config: SystemConfig) -> None:
        self._query_to_item = query_to_item
        self._config = config

    def run(self, query: object):
        item_id = self._query_to_item.get(str(query))
        logic_forms = SYSTEM_LOGIC_FORMS.get(item_id, ["unknown"])
        script = [scenarios.scripted_step(lf, status="verified") for lf in logic_forms]
        orchestrator, _ = scenarios.build_orchestrator(
            script=script,
            procedural_memory=[VERIFY_RULE],
            config=self._config,
        )
        return orchestrator.run(query)


class ScriptedBaseline:
    """A baseline reasoning method backed by a per-query scripted MockBackend.

    Conforms to the harness's ``ReasoningMethod`` protocol (``name`` + ``run(query)``).
    Each call builds the real baseline method (Plain LLM or Chain-of-Thought) over a
    scripted backend and an injected step clock for a deterministic latency.
    """

    def __init__(self, method_name: str, query_to_item: dict[str, str]) -> None:
        self.name = method_name
        self._query_to_item = query_to_item
        self._step_ms = _BASELINE_STEP_MS[method_name]

    def run(self, query: str):
        item_id = self._query_to_item.get(str(query))
        completion = BASELINE_COMPLETIONS[self.name].get(item_id, "Answer: unknown")
        method = build_baseline(
            self.name, MockBackend([completion]), clock=StepClock(self._step_ms)
        )
        return method.run(query)


def _format_value(metric: str, value) -> str:
    if value is None:
        return "n/a"
    if metric in _LATENCY_METRICS:
        return f"{value:.2f} ms"
    return f"{value:.3f}"


def build_report_html(
    report,
    dataset,
    run_record,
    *,
    is_real: bool = False,
    model_label: str | None = None,
    dataset_label: str | None = None,
    arithmetic_validation: bool = False,
) -> str:
    """Render the comparison report as a self-contained, dependency-light HTML table.

    The same renderer serves both modes. When ``is_real`` is ``False`` (the default) the
    report carries the offline-mock framing. When ``is_real`` is ``True`` it states
    prominently that the results are **genuine answers from a real model** (named by
    ``model_label``) served via Ollama, and that faithfulness / step-hallucination are
    derived from the System's Proof Trace under the provided general starter rule set.
    """
    baselines = report.baseline_methods  # sorted baseline names
    header_cells = (
        "<th>Metric</th>"
        f"<th>{html.escape(DISPLAY_NAMES.get(report.system_method, report.system_method))}</th>"
        + "".join(
            f"<th>{html.escape(DISPLAY_NAMES.get(b, b))}</th>"
            f"<th>Δ (System − {html.escape(DISPLAY_NAMES.get(b, b))})</th>"
            for b in baselines
        )
    )

    rows = []
    for comparison in report.metrics:
        label = METRIC_LABELS.get(comparison.metric, comparison.metric)
        cells = [f'<td class="metric">{html.escape(label)}</td>']
        cells.append(f"<td><b>{_format_value(comparison.metric, comparison.system_value)}</b></td>")
        for b in baselines:
            cells.append(f"<td>{_format_value(comparison.metric, comparison.baseline_values.get(b))}</td>")
            diff = comparison.differences.get(b)
            cls = ""
            if diff is not None:
                cls = "pos" if diff > 0 else ("neg" if diff < 0 else "")
            diff_text = _format_value(comparison.metric, diff) if diff is not None else "n/a"
            sign = "+" if (diff is not None and diff > 0) else ""
            cells.append(f'<td class="{cls}">{sign}{diff_text}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    domain_counts: dict[str, int] = {}
    for item in dataset:
        domain_counts[item.domain.value] = domain_counts.get(item.domain.value, 0) + 1
    domain_list = ", ".join(f"{d} ({n})" for d, n in sorted(domain_counts.items()))

    css = """
    body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
           margin: 0; color: #1a1a1a; background: #f6f7f9; line-height: 1.5; }
    header { background: #0f2540; color: #fff; padding: 24px 32px; }
    header h1 { margin: 0 0 4px; font-size: 22px; }
    header p { margin: 0; color: #b9c7d8; font-size: 14px; }
    main { max-width: 1000px; margin: 0 auto; padding: 24px 32px 64px; }
    .section { background:#fff; border:1px solid #e2e6ea; border-radius:10px;
               padding:20px 24px; margin:20px 0; }
    .offline { background:#fff8e1; border:1px solid #ffe082; color:#6d5200;
               border-radius:8px; padding:10px 14px; font-size:13px; }
    .real { background:#e8f5e9; border:1px solid #a5d6a7; color:#1b5e20;
            border-radius:8px; padding:10px 14px; font-size:13px; }
    .scaffold { background:#e3f2fd; border:1px solid #90caf9; color:#0d47a1;
                border-radius:8px; padding:10px 14px; font-size:13px; margin-top:10px; }
    table { width:100%; border-collapse:collapse; margin-top:8px; font-size:14px; }
    th, td { padding:10px 12px; border-bottom:1px solid #e2e6ea; text-align:right; }
    th:first-child, td:first-child { text-align:left; }
    thead th { background:#eef1f4; }
    td.metric { font-weight:600; }
    td.pos { color:#2e7d32; font-weight:600; }
    td.neg { color:#c62828; font-weight:600; }
    .kv { font-size:13px; color:#444; }
    footer { text-align:center; color:#98a2ad; font-size:12px; padding:20px; }
    """

    if is_real:
        model_name = model_label or run_record.model_id
        page_title = "NSR Benchmark — REAL model via Ollama"
        header_title = "Neuro-Symbolic System — Real-Model Benchmark"
        header_sub = (
            f"Real model: {html.escape(str(model_name))} via Ollama — "
            "System vs Plain LLM vs Chain-of-Thought"
        )
        banner = (
            f'<p class="real"><b>Real model: {html.escape(str(model_name))} via Ollama.</b> '
            "These are <b>genuine results from an actual language model</b> — every System "
            "and baseline answer was produced by the model, not scripted. They are NOT the "
            "offline mock numbers (see <code>benchmark_report.html</code> for the "
            "deterministic mock comparison).</p>"
            '<p class="scaffold">What is genuinely real vs scaffold: the <b>baseline</b> and '
            "<b>System</b> final answers are real model output; the System also produces a "
            "real Proof Trace and a real Faithfulness Score. The System\u2019s step-level "
            "validation here uses a <b>general-purpose starter production-rule set</b> "
            "(accept any well-formed step that carries a non-empty conclusion, plus a "
            "consistency guard); domain-specific production rules would strengthen "
            "step-level validation.</p>"
        )
    else:
        page_title = "NSR Benchmark — System vs Baselines"
        header_title = "Neuro-Symbolic System — Benchmark Comparison"
        header_sub = (
            "System vs Plain LLM vs Chain-of-Thought across a small multi-domain dataset"
        )
        banner = (
            '<p class="offline">Offline &amp; deterministic: answers come from scripted '
            "MockBackends and per-method latencies come from injected step clocks. These "
            "numbers illustrate the architecture's behavior and are reproducible — they are "
            "not measurements of a real LLM.</p>"
        )

    # Optional extra banner lines: dataset label and the arithmetic-validation note,
    # appended for whichever mode is active so the GSM8K real run can state which dataset
    # was used and that intermediate-arithmetic checking is active.
    if dataset_label:
        banner += f'<p class="kv">Dataset: <b>{html.escape(str(dataset_label))}</b>.</p>'
    if arithmetic_validation:
        banner += (
            '<p class="scaffold"><b>Arithmetic validation is active.</b> Each intermediate '
            "calculation is checked by the ArithmeticValidationEngine: a wrong equation is "
            "REJECTED and routed to the bounded repair sub-loop rather than carried through "
            "to a wrong final answer. This is genuine content-level step validation on top "
            "of the general-purpose starter rule set.</p>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(page_title)}</title>
<style>{css}</style>
</head>
<body>
<header>
  <h1>{html.escape(header_title)}</h1>
  <p>{header_sub}</p>
</header>
<main>
  <div class="section">
    {banner}
    <p class="kv">Dataset: <b>{len(dataset)}</b> items across domains — {html.escape(domain_list)}.
      Model id: <b>{html.escape(run_record.model_id)}</b>. Seed: <b>{run_record.seed}</b>.
      Repeated runs: <b>{report.repeated_run_count}</b>.</p>
    <p class="kv">Note: faithfulness and step-hallucination rate are derived from the
      System's Proof Trace; baselines emit no trace, so those values are 0 for baselines by
      construction.</p>
  </div>
  <div class="section">
    <h2>Comparison across {len(REPORT_METRICS)} metrics</h2>
    <table>
      <thead><tr>{header_cells}</tr></thead>
      <tbody>
{chr(10).join(rows)}
      </tbody>
    </table>
  </div>
</main>
<footer>Generated by demo/run_benchmark.py — Neuro-Symbolic System-2 Reasoning Architecture</footer>
</body>
</html>
"""


def _report_to_dict(
    report,
    dataset,
    run_record,
    *,
    is_real: bool = False,
    model_label: str | None = None,
    dataset_label: str | None = None,
    arithmetic_validation: bool = False,
) -> dict:
    """A JSON-serializable view of the comparison report and run metadata."""
    if is_real:
        note = (
            f"REAL model results: every System and baseline answer was produced by "
            f"the model {model_label or run_record.model_id!r} served via Ollama — these "
            "are genuine model outputs, NOT the offline mock numbers. Faithfulness and "
            "step-hallucination rate are derived from the System's Proof Trace under a "
            "general-purpose starter production-rule set (domain-specific rules would "
            "strengthen step-level validation)."
        )
        if arithmetic_validation:
            note += (
                " Arithmetic validation is active: each intermediate equation is checked "
                "and a wrong calculation is rejected and routed to the bounded repair "
                "sub-loop rather than carried through to a wrong final answer."
            )
    else:
        note = (
            "Offline deterministic demo. Answers are from scripted MockBackends and "
            "latencies from injected step clocks; values are illustrative, not real-LLM "
            "measurements."
        )
    return {
        "mode": "real" if is_real else "mock",
        "is_real": is_real,
        "real_model": (model_label or run_record.model_id) if is_real else None,
        "dataset_label": dataset_label,
        "arithmetic_validation": arithmetic_validation,
        "note": note,
        "system_method": report.system_method,
        "baseline_methods": list(report.baseline_methods),
        "repeated_run_count": report.repeated_run_count,
        "model_id": run_record.model_id,
        "seed": run_record.seed,
        "dataset": [
            {"item_id": i.item_id, "domain": i.domain.value, "ground_truth": i.ground_truth}
            for i in dataset
        ],
        "metrics": [
            {
                "metric": m.metric,
                "system_value": m.system_value,
                "baseline_values": m.baseline_values,
                "differences": m.differences,
            }
            for m in report.metrics
        ],
        "reasoning_consistency": report.reasoning_consistency,
    }


def generate_benchmark(output_dir: os.PathLike | str = OUTPUT_DIR) -> dict[str, Path]:
    """Run the offline benchmark and write the HTML + JSON reports. Returns their paths."""
    config = make_benchmark_config()

    load_result = load_dataset(RAW_ITEMS)
    dataset = load_result.items
    query_to_item = {item.query: item.item_id for item in dataset}

    system = ScriptedSystem(query_to_item, config)
    baselines = {name: ScriptedBaseline(name, query_to_item) for name in BASELINE_NAMES}

    harness = EvaluationHarness(system, baselines, clock=StepClock(_SYSTEM_STEP_MS), answer_match=lenient_answer_match)
    runs = [
        harness.run(dataset, config=config, model_id="offline-mock-1")
        for _ in range(config.repeated_run_count)
    ]
    report = build_comparison_report(runs, repeated_run_count=config.repeated_run_count)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "html": out / "benchmark_report.html",
        "json": out / "benchmark_report.json",
    }
    paths["html"].write_text(
        build_report_html(report, dataset, runs[0].run_record), encoding="utf-8"
    )
    paths["json"].write_text(
        json.dumps(_report_to_dict(report, dataset, runs[0].run_record), indent=2),
        encoding="utf-8",
    )
    return paths


# --------------------------------------------------------------------------- #
# REAL-MODEL mode (driven by a local Ollama server)
# --------------------------------------------------------------------------- #

#: Default model used when ``--backend ollama`` is selected without ``--model``.
DEFAULT_OLLAMA_MODEL = "llama3.1"

#: A general-purpose "well-formed step" production rule for the real System path.
#:
#: It is *always applicable* (empty IF condition) and *vacuously satisfied* (empty THEN
#: action), so it accepts any step that reaches the Validation Engine. Because the
#: Constrained Decoder already guarantees a step carries a non-empty ``logic_form``
#: (conclusion/answer) before validation — regenerating/repairing non-conforming model
#: output up to the configured limits — this rule encodes "accept any well-formed step
#: that carries a non-empty conclusion". It is a GENERAL STARTER rule, not a
#: domain-specific validator: it does not check the *content* of a conclusion. Adding
#: domain-specific production rules (arithmetic checks, logical-entailment guards, etc.)
#: would strengthen step-level validation. Acceptance here is genuine — the step really
#: did pass format conformance and rule selection — it is not faked.
WELL_FORMED_STEP_RULE = ProductionRule(
    rule_id="well-formed-step",
    condition="",  # always applicable
    action="",  # satisfied by any conforming step (non-empty conclusion guaranteed)
)

#: An optional general consistency guard: it only becomes applicable when a step's text
#: mentions a "contradiction", and is satisfied only when the step also "flag"s it. A
#: model step that announces a contradiction without flagging it is rejected and routed
#: to the bounded repair sub-loop. This is a light, domain-agnostic guard included to show
#: a non-trivial rule in the starter set; it triggers rarely on well-behaved output.
CONSISTENCY_GUARD_RULE = ProductionRule(
    rule_id="consistency-guard",
    condition="IF contradiction",
    action="THEN flag",
)

#: The general-purpose starter Procedural_Memory seeded into the real System. Documented
#: as a scaffold: it produces a genuinely-functioning System path (real Proof Trace + real
#: Faithfulness Score over real model output), while domain-specific rules would add
#: real content-level step validation.
STARTER_PRODUCTION_RULES = [WELL_FORMED_STEP_RULE, CONSISTENCY_GUARD_RULE]


def make_real_config(model: str, repeated_run_count: int = 2) -> SystemConfig:
    """The System configuration for a REAL-model run against ``model``.

    Differs from the mock config in two honest ways: ``llm_selection`` records the real
    model name, and ``retry_count`` / ``repair_attempt_limit`` are non-zero so the
    Constrained Decoder and Repair Coordinator can tolerate a real model returning
    imperfect JSON before giving up. ``output_format="json"`` makes the decoder ask the
    model for a JSON object carrying a ``logic_form`` field. ``repeated_run_count`` feeds
    Reasoning Consistency (>= 2); set it to 1 for a faster single pass (consistency then
    stays unset) — useful for slow local models.
    """
    return SystemConfig(
        max_cycle_limit=10,
        repair_attempt_limit=2,
        retry_count=2,
        llm_selection=model,
        output_format="json",
        conflict_resolution_policy="priority",
        generation_timeout_ms=60000,
        repeated_run_count=max(1, int(repeated_run_count)),
        random_seed=7,
    )


class RealSystem:
    """System-under-test backed by a REAL Ollama model (no scripting).

    Conforms to the harness's ``run(query) -> VerifiedOutput | ErrorRecord`` protocol. For
    each query it wires the **real** orchestrator over a real
    :class:`~nsr.llm_component.OllamaBackend` via
    :func:`scenarios.build_orchestrator_with_backend`, seeded with the general-purpose
    starter production-rule set. The System therefore produces a genuine Proof Trace and
    Faithfulness Score from real model output — acceptance is never faked.
    """

    def __init__(self, model: str, host: str | None, config: SystemConfig) -> None:
        self._model = model
        self._host = host
        self._config = config

    def run(self, query: object):
        backend = build_ollama_backend(self._model, host=self._host)
        orchestrator = scenarios.build_orchestrator_with_backend(
            backend=backend,
            procedural_memory=STARTER_PRODUCTION_RULES,
            config=self._config,
            translation=scenarios.RealModelTranslationLayer(),
        )
        return orchestrator.run(query)


class RealBaseline:
    """A baseline reasoning method (Plain LLM / Chain-of-Thought) over a REAL model.

    Conforms to the harness's ``ReasoningMethod`` protocol (``name`` + ``run(query)``).
    Each call builds the real baseline via :func:`nsr.baselines.build_baseline` over a real
    :class:`~nsr.llm_component.OllamaBackend`, so the answer and its wall-clock latency are
    genuinely the model's.
    """

    def __init__(self, method_name: str, model: str, host: str | None) -> None:
        self.name = method_name
        self._model = model
        self._host = host

    def run(self, query: str):
        backend = build_ollama_backend(self._model, host=self._host)
        method = build_baseline(self.name, backend)
        return method.run(query)


def _run_real_benchmark(model: str, host: str | None, output_dir):
    """Core real-model run: returns ``(paths, runs, dataset)`` for the CLI to inspect.

    Shared by :func:`generate_benchmark_real` (public, returns only the written paths) and
    the CLI (which also inspects the runs to detect the degenerate "every item failed"
    case — for example when the model is not pulled — and report it honestly).
    """
    config = make_real_config(model)

    load_result = load_dataset(RAW_ITEMS)
    dataset = load_result.items

    system = RealSystem(model, host, config)
    baselines = {name: RealBaseline(name, model, host) for name in BASELINE_NAMES}

    harness = EvaluationHarness(system, baselines, answer_match=lenient_answer_match)
    runs = [
        harness.run(dataset, config=config, model_id=f"ollama:{model}")
        for _ in range(config.repeated_run_count)
    ]
    report = build_comparison_report(runs, repeated_run_count=config.repeated_run_count)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "html": out / "benchmark_report_real.html",
        "json": out / "benchmark_report_real.json",
    }
    paths["html"].write_text(
        build_report_html(
            report, dataset, runs[0].run_record, is_real=True, model_label=model
        ),
        encoding="utf-8",
    )
    paths["json"].write_text(
        json.dumps(
            _report_to_dict(
                report, dataset, runs[0].run_record, is_real=True, model_label=model
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths, runs, dataset


def generate_benchmark_real(
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str | None = None,
    output_dir: os.PathLike | str = OUTPUT_DIR,
) -> dict[str, Path]:
    """Run the benchmark against a REAL Ollama model and write ``*_real`` reports.

    Uses the *same* dataset, :class:`~nsr.evaluation_harness.EvaluationHarness`, and
    :func:`~nsr.comparison_report.build_comparison_report` flow as the offline benchmark,
    but drives both the System and the baselines with a real model served by Ollama. The
    HTML/JSON are written to ``benchmark_report_real.html`` / ``.json`` and are clearly
    labelled with the model name and as genuine model results. Returns their paths.

    This function performs **no preflight** — callers (the CLI) should call
    :func:`nsr.llm_component.ollama_available` first. It is exercised offline in tests by
    monkeypatching the backend factory.
    """
    paths, _runs, _dataset = _run_real_benchmark(model, host, output_dir)
    return paths


# --------------------------------------------------------------------------- #
# GSM8K REAL-MODEL mode: multi-step arithmetic with live arithmetic validation
# --------------------------------------------------------------------------- #

#: Default model used for the GSM8K real run when ``--model`` is not given.
DEFAULT_GSM8K_MODEL = "qwen3:8b"


class RealMathSystem:
    """System-under-test for GSM8K, backed by a REAL Ollama model with live math checking.

    Like :class:`RealSystem`, but wires the orchestrator with the
    :class:`~scenarios.MathTranslationLayer` (so the model emits checkable
    ``"<expr> = <result>"`` steps) and a content-level Validation Engine selected by
    ``validation_mode``:

    - ``"arithmetic"`` (default) -> :class:`~arithmetic_validation.ArithmeticValidationEngine`:
      a wrong intermediate calculation is rejected and routed to the bounded repair sub-loop.
    - ``"goal"`` -> :class:`~goal_alignment.GoalAlignmentValidationEngine`: in addition to
      arithmetic correctness, a final-answer step that computes the WRONG QUANTITY for the
      goal (e.g. the cost when the goal asks for profit) is rejected and repaired. The
      goal-alignment validator is built **per query** with that query's text, since the
      query IS the goal.

    Seeded with the same general-purpose starter production-rule set. Conforms to the
    harness ``run(query) -> VerifiedOutput | ErrorRecord`` protocol.
    """

    def __init__(
        self,
        model: str,
        host: str | None,
        config: SystemConfig,
        validation_mode: str = VALIDATION_ARITHMETIC,
    ) -> None:
        self._model = model
        self._host = host
        self._config = config
        self._validation_mode = validation_mode

    def _build_validation(self, query: object):
        """Build the Validation Engine for ``query`` according to ``validation_mode``.

        The goal-alignment validator MUST be constructed per query (the query text is the
        goal); the arithmetic validator is stateless but is built fresh here too for
        symmetry.
        """
        if self._validation_mode == VALIDATION_GOAL:
            return GoalAlignmentValidationEngine(goal_text=str(query))
        return ArithmeticValidationEngine()

    def run(self, query: object):
        backend = build_ollama_backend(self._model, host=self._host)
        orchestrator = scenarios.build_orchestrator_with_backend(
            backend=backend,
            procedural_memory=STARTER_PRODUCTION_RULES,
            config=self._config,
            translation=scenarios.MathTranslationLayer(),
            validation=self._build_validation(query),
        )
        return orchestrator.run(query)


def _gsm8k_dataset(dataset_path: str | os.PathLike | None, limit: int | None):
    """Load the GSM8K dataset for a run: the official file at ``dataset_path`` or, when
    ``dataset_path`` is ``None``, the bundled ORIGINAL sample. Returns ``(dataset, label)``.
    """
    if dataset_path is not None:
        dataset = datasets.load_gsm8k(dataset_path, limit=limit)
        label = f"GSM8K official format: {Path(dataset_path).name} ({len(dataset)} items)"
    else:
        dataset = datasets.load_benchmark("gsm8k", path=None, limit=limit)
        label = (
            f"bundled ORIGINAL sample ({len(dataset)} items) — NOT official GSM8K; "
            "pass --path <test.jsonl> to use the official corpus"
        )
    return dataset, label


def _run_gsm8k_benchmark(
    model,
    host,
    dataset_path,
    limit,
    output_dir,
    repeated_run_count=2,
    validation_mode=VALIDATION_ARITHMETIC,
):
    """Core GSM8K real-model run: returns ``(paths, runs, dataset)`` for the CLI to inspect.

    Uses the same harness/report flow as the multi-domain real benchmark, but with the
    :class:`RealMathSystem` (live arithmetic validation, or goal-aligned validation when
    ``validation_mode == "goal"``) and the numeric answer matcher applied symmetrically to
    the System and the baselines. ``repeated_run_count`` controls how many passes are made
    (1 = fast single pass; >= 2 also yields Reasoning Consistency).
    """
    config = make_real_config(model, repeated_run_count=repeated_run_count)
    dataset, dataset_label = _gsm8k_dataset(dataset_path, limit)

    system = RealMathSystem(model, host, config, validation_mode=validation_mode)
    baselines = {name: RealBaseline(name, model, host) for name in BASELINE_NAMES}

    harness = EvaluationHarness(system, baselines, answer_match=numeric_answer_match)
    runs = [
        harness.run(dataset, config=config, model_id=f"ollama:{model}")
        for _ in range(config.repeated_run_count)
    ]
    report = build_comparison_report(runs, repeated_run_count=config.repeated_run_count)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "html": out / "benchmark_report_gsm8k.html",
        "json": out / "benchmark_report_gsm8k.json",
    }
    paths["html"].write_text(
        build_report_html(
            report,
            dataset,
            runs[0].run_record,
            is_real=True,
            model_label=model,
            dataset_label=dataset_label,
            arithmetic_validation=True,
        ),
        encoding="utf-8",
    )
    paths["json"].write_text(
        json.dumps(
            _report_to_dict(
                report,
                dataset,
                runs[0].run_record,
                is_real=True,
                model_label=model,
                dataset_label=dataset_label,
                arithmetic_validation=True,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths, runs, dataset


def generate_benchmark_gsm8k(
    model: str = DEFAULT_GSM8K_MODEL,
    host: str | None = None,
    dataset_path: os.PathLike | str | None = None,
    limit: int = 5,
    output_dir: os.PathLike | str = OUTPUT_DIR,
    repeated_run_count: int = 2,
    validation_mode: str = VALIDATION_ARITHMETIC,
) -> dict[str, Path]:
    """Run the GSM8K benchmark against a REAL Ollama model and write ``*_gsm8k`` reports.

    Loads the dataset (the bundled original sample when ``dataset_path`` is ``None``, else
    the official GSM8K file), builds :class:`RealMathSystem` + :class:`RealBaseline`
    baselines, runs the :class:`~nsr.evaluation_harness.EvaluationHarness` with
    :func:`numeric_answer_match`, builds the comparison report, and writes
    ``benchmark_report_gsm8k.html`` / ``.json`` labelled with the model, the dataset, and
    that arithmetic validation is active. ``validation_mode`` selects ``"arithmetic"``
    (default) or goal-aligned (``"goal"``) System validation. Returns their paths.

    Performs **no preflight** — callers (the CLI) should call
    :func:`nsr.llm_component.ollama_available` first. Exercised offline in tests by
    monkeypatching the backend factory.
    """
    paths, _runs, _dataset = _run_gsm8k_benchmark(
        model, host, dataset_path, limit, output_dir, repeated_run_count, validation_mode
    )
    return paths


def _run_mock(argv_unused: object = None) -> int:
    """Run the offline mock benchmark and print where the reports landed."""
    paths = generate_benchmark()
    print(f"Offline benchmark complete. Reports written to {OUTPUT_DIR}:")
    for key, path in paths.items():
        size = path.stat().st_size
        print(f"  - {path.name}  ({size} bytes)")
    print("\nOpen benchmark_report.html in a browser to view the comparison table.")
    return 0


def _run_real(model: str, host: str | None) -> int:
    """Preflight Ollama, then run the real benchmark; return the process exit code.

    On an unreachable server, prints the friendly reason plus ``ollama serve`` /
    ``ollama pull`` hints and returns a non-zero code WITHOUT raising. If the server is up
    but the requested model is not listed, warns and still attempts the run (the model may
    be pullable on demand).
    """
    available, reason = ollama_available(host)
    if not available:
        print(f"Cannot run the real benchmark: {reason}")
        print("Hints:")
        print("  - start the server:  ollama serve")
        print(f"  - pull the model:    ollama pull {model}")
        return 1

    print(f"Ollama reachable — {reason}")
    if model not in reason and f"{model}:" not in reason:
        print(
            f"Warning: model {model!r} was not in the available list; attempting anyway "
            f"(it may be pulled on demand). If this fails, run: ollama pull {model}"
        )

    print(f"Running REAL benchmark against {model!r} via Ollama (this calls the model)...")
    try:
        paths, runs, _dataset = _run_real_benchmark(model, host, OUTPUT_DIR)
    except Exception as exc:  # surface a friendly message, no raw traceback
        print(f"Real benchmark failed while talking to the model: {exc}")
        print(f"Hint: ensure the model is pulled (ollama pull {model}) and the server is up.")
        return 1

    # Honesty check: if the model produced no usable results for ANY item (e.g. it is not
    # pulled and every call 404'd), the report would be all zeros — say so plainly and
    # exit non-zero rather than presenting an empty comparison as a successful benchmark.
    system_outcomes = sum(
        len(run.per_item_outcomes.get(SYSTEM_METHOD_NAME, [])) for run in runs
    )
    baseline_outcomes = sum(
        len(run.per_item_outcomes.get(name, []))
        for run in runs
        for name in BASELINE_NAMES
    )
    if system_outcomes == 0 and baseline_outcomes == 0:
        print(
            f"No results were produced: every item failed against {model!r}. "
            f"The model is most likely not pulled — run: ollama pull {model}"
        )
        print(f"(A labelled-but-empty report was still written to {OUTPUT_DIR}.)")
        return 1

    print(f"Real-model benchmark complete. Reports written to {OUTPUT_DIR}:")
    for path in paths.values():
        size = path.stat().st_size
        print(f"  - {path.name}  ({size} bytes)")
    print("\nOpen benchmark_report_real.html in a browser to view the real-model comparison.")
    return 0


def _run_real_gsm8k(
    model: str,
    host: str | None,
    dataset_path,
    limit: int,
    runs: int = 2,
    validation_mode: str = VALIDATION_ARITHMETIC,
) -> int:
    """Preflight Ollama, then run the GSM8K real benchmark; return the process exit code.

    Mirrors :func:`_run_real`: on an unreachable server it prints the friendly reason plus
    ``ollama serve`` / ``ollama pull`` hints and returns non-zero WITHOUT raising; if every
    item failed (e.g. the model is not pulled) it says so plainly and exits non-zero.
    ``validation_mode`` selects arithmetic-only (default) or goal-aligned System validation.
    """
    available, reason = ollama_available(host)
    if not available:
        print(f"Cannot run the real GSM8K benchmark: {reason}")
        print("Hints:")
        print("  - start the server:  ollama serve")
        print(f"  - pull the model:    ollama pull {model}")
        return 1

    print(f"Ollama reachable — {reason}")
    if model not in reason and f"{model}:" not in reason:
        print(
            f"Warning: model {model!r} was not in the available list; attempting anyway "
            f"(it may be pulled on demand). If this fails, run: ollama pull {model}"
        )

    source = "the official dataset" if dataset_path else "the bundled original sample"
    validation_label = (
        "goal-aligned validation" if validation_mode == VALIDATION_GOAL
        else "arithmetic validation"
    )
    print(
        f"Running REAL GSM8K benchmark against {model!r} via Ollama over {source} "
        f"with {validation_label} (this calls the model)..."
    )
    try:
        paths, runs, _dataset = _run_gsm8k_benchmark(
            model, host, dataset_path, limit, OUTPUT_DIR, runs, validation_mode
        )
    except Exception as exc:  # surface a friendly message, no raw traceback
        print(f"Real GSM8K benchmark failed while talking to the model: {exc}")
        print(f"Hint: ensure the model is pulled (ollama pull {model}) and the server is up.")
        return 1

    system_outcomes = sum(
        len(run.per_item_outcomes.get(SYSTEM_METHOD_NAME, [])) for run in runs
    )
    baseline_outcomes = sum(
        len(run.per_item_outcomes.get(name, []))
        for run in runs
        for name in BASELINE_NAMES
    )
    if system_outcomes == 0 and baseline_outcomes == 0:
        print(
            f"No results were produced: every item failed against {model!r}. "
            f"The model is most likely not pulled — run: ollama pull {model}"
        )
        print(f"(A labelled-but-empty report was still written to {OUTPUT_DIR}.)")
        return 1

    print(f"Real-model GSM8K benchmark complete. Reports written to {OUTPUT_DIR}:")
    for path in paths.values():
        size = path.stat().st_size
        print(f"  - {path.name}  ({size} bytes)")
    print("\nOpen benchmark_report_gsm8k.html in a browser to view the GSM8K comparison.")
    return 0


def _run_ablation_gsm8k(
    model: str, host: str | None, dataset_path, limit: int, runs: int = 1
) -> int:
    """Preflight Ollama, then run the four-config GSM8K ablation study; return exit code.

    Mirrors :func:`_run_real_gsm8k`: on an unreachable server it prints the friendly reason
    plus ``ollama serve`` / ``ollama pull`` hints and returns non-zero WITHOUT raising. The
    ablation evaluates the four configurations (plain-llm, constrained-decoding,
    actr-no-validation, full-neuro-symbolic) on the SAME GSM8K subset and writes
    ``benchmark_report_ablation_gsm8k.html`` / ``.json``.
    """
    import ablation  # local import keeps the module import cycle-free

    available, reason = ollama_available(host)
    if not available:
        print(f"Cannot run the GSM8K ablation study: {reason}")
        print("Hints:")
        print("  - start the server:  ollama serve")
        print(f"  - pull the model:    ollama pull {model}")
        return 1

    print(f"Ollama reachable — {reason}")
    if model not in reason and f"{model}:" not in reason:
        print(
            f"Warning: model {model!r} was not in the available list; attempting anyway "
            f"(it may be pulled on demand). If this fails, run: ollama pull {model}"
        )

    source = "the official dataset" if dataset_path else "the bundled original sample"
    print(
        f"Running GSM8K ABLATION study against {model!r} via Ollama over {source} "
        "(this calls the model for all four configurations)..."
    )
    try:
        paths = ablation.generate_ablation_gsm8k(
            model=model,
            host=host,
            dataset_path=dataset_path,
            limit=limit,
            repeated_run_count=runs,
            output_dir=OUTPUT_DIR,
        )
    except Exception as exc:  # surface a friendly message, no raw traceback
        print(f"GSM8K ablation study failed while talking to the model: {exc}")
        print(f"Hint: ensure the model is pulled (ollama pull {model}) and the server is up.")
        return 1

    print(f"GSM8K ablation study complete. Reports written to {OUTPUT_DIR}:")
    for path in paths.values():
        size = path.stat().st_size
        print(f"  - {path.name}  ({size} bytes)")
    print(
        "\nOpen benchmark_report_ablation_gsm8k.html in a browser to view the "
        "per-component comparison."
    )
    return 0


def _run_reasoning_stats(
    model: str, host: str | None, dataset_path, limit: int,
    validation_mode: str = VALIDATION_ARITHMETIC,
) -> int:
    """Preflight Ollama, then collect REASONING STATISTICS for the full System; exit code.

    Mirrors :func:`_run_real_gsm8k`: on an unreachable server it prints the friendly reason
    plus ``ollama serve`` / ``ollama pull`` hints and returns non-zero WITHOUT raising. The
    statistics run drives only the FULL neuro-symbolic system over the SAME GSM8K subset and
    writes ``reasoning_stats.html`` / ``.json`` describing what happened inside the reasoning
    loop (first-pass acceptances, repairs, rejections, rule utilization, repair triggers,
    termination reasons, and per-item correctness). ``validation_mode`` selects
    arithmetic-only (default) or goal-aligned System validation.
    """
    available, reason = ollama_available(host)
    if not available:
        print(f"Cannot collect reasoning statistics: {reason}")
        print("Hints:")
        print("  - start the server:  ollama serve")
        print(f"  - pull the model:    ollama pull {model}")
        return 1

    print(f"Ollama reachable — {reason}")
    if model not in reason and f"{model}:" not in reason:
        print(
            f"Warning: model {model!r} was not in the available list; attempting anyway "
            f"(it may be pulled on demand). If this fails, run: ollama pull {model}"
        )

    source = "the official dataset" if dataset_path else "the bundled original sample"
    print(
        f"Collecting REASONING STATISTICS for the full system against {model!r} via Ollama "
        f"over {source} (this calls the model)..."
    )
    # Lazy import keeps the module import cycle-free (reasoning_stats imports this module).
    import reasoning_stats

    try:
        paths = reasoning_stats.generate_reasoning_stats(
            model, host, dataset_path, limit, validation_mode=validation_mode
        )
    except Exception as exc:  # surface a friendly message, no raw traceback
        print(f"Reasoning-statistics run failed while talking to the model: {exc}")
        print(f"Hint: ensure the model is pulled (ollama pull {model}) and the server is up.")
        return 1

    print(f"Reasoning statistics complete. Reports written to {OUTPUT_DIR}:")
    for path in paths.values():
        size = path.stat().st_size
        print(f"  - {path.name}  ({size} bytes)")
    print(
        "\nOpen reasoning_stats.html in a browser to view the reasoning-loop breakdown."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Defaults to the offline mock benchmark (no args, unchanged).

    ``--backend ollama`` runs the same flow against a real model served by Ollama.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the Neuro-Symbolic benchmark (System vs Plain LLM vs Chain-of-Thought). "
            "Defaults to a fully-offline, deterministic mock benchmark."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("mock", "ollama"),
        default="mock",
        help="mock (offline, default) or ollama (real model via a local Ollama server)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OLLAMA_MODEL,
        help=(
            "model name for --backend ollama (e.g. llama3.1, mistral, qwen2.5, phi3, "
            f"gemma2); default {DEFAULT_OLLAMA_MODEL!r}"
        ),
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Ollama base URL (overrides the NSR_OLLAMA_HOST env var)",
    )
    parser.add_argument(
        "--dataset",
        choices=("builtin", "gsm8k"),
        default="builtin",
        help=(
            "builtin (the small multi-domain dataset, default) or gsm8k (multi-step "
            "arithmetic with live arithmetic validation; --backend ollama only)"
        ),
    )
    parser.add_argument(
        "--path",
        default=None,
        help=(
            "for --dataset gsm8k: path to an official GSM8K-format JSONL (e.g. the "
            "openai/grade-school-math test.jsonl). Omit to use the bundled original sample."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="for --dataset gsm8k: evaluate only the first N items (default 5)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help=(
            "for --dataset gsm8k: number of repeated passes (default 2; use 1 for a "
            "faster single pass — Reasoning Consistency needs >= 2)"
        ),
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "for --backend ollama --dataset gsm8k: run the four-configuration ABLATION "
            "study (plain-llm, constrained-decoding, actr-no-validation, "
            "full-neuro-symbolic) on the SAME subset and write "
            "benchmark_report_ablation_gsm8k.html/.json"
        ),
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help=(
            "for --backend ollama --dataset gsm8k: collect REASONING STATISTICS for the "
            "full system only (first-pass acceptances, repairs, rejections, rule "
            "utilization, repair triggers, termination reasons, per-item correctness) and "
            "write reasoning_stats.html/.json"
        ),
    )
    parser.add_argument(
        "--goal-validation",
        action="store_true",
        help=(
            "for --backend ollama --dataset gsm8k: use GOAL-ALIGNED semantic validation "
            "instead of arithmetic-only. In addition to checking that each equation is "
            "arithmetically correct, the System checks that the final-answer step computes "
            "the QUANTITY the goal asked for (intent) — e.g. it rejects answering the COST "
            "when the goal asks for PROFIT — and routes a mismatch to the bounded repair "
            "sub-loop. Default is arithmetic-only, so Baseline -> Arithmetic -> "
            "Arithmetic+Goal can be compared. Applies to the GSM8K System run and --stats."
        ),
    )
    args = parser.parse_args(argv)

    validation_mode = VALIDATION_GOAL if args.goal_validation else VALIDATION_ARITHMETIC

    if args.backend == "ollama":
        if args.dataset == "gsm8k":
            if args.stats:
                if args.ablation:
                    print(
                        "Error: --stats and --ablation are mutually exclusive. --stats "
                        "applies to the full system only; drop one and re-run."
                    )
                    return 2
                return _run_reasoning_stats(
                    args.model, args.host, args.path, args.limit, validation_mode
                )
            if args.ablation:
                return _run_ablation_gsm8k(
                    args.model, args.host, args.path, args.limit, args.runs
                )
            return _run_real_gsm8k(
                args.model, args.host, args.path, args.limit, args.runs, validation_mode
            )
        return _run_real(args.model, args.host)
    return _run_mock()


if __name__ == "__main__":
    raise SystemExit(main())
