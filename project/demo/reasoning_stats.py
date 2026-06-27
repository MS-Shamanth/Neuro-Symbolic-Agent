"""REASONING-STATISTICS collector + report for the Neuro-Symbolic demo (GSM8K).

The benchmark/ablation reports answer "how *accurate* and *fast* is each configuration?".
This module answers a different, complementary question about the FULL system only:
**what happened inside the reasoning loop?** It walks the :class:`~nsr.models.ProofTrace`
of every processed query and tallies, per reasoning step, how the symbolic layer treated
it — accepted on the first pass, rejected outright, or rejected-then-repaired — plus which
production rules were applied, which rules fired the repair sub-loop, and why each cycle
terminated.

Two layers
----------

- :func:`aggregate_trace_stats` is a **pure function** over a list of
  :class:`~nsr.models.ProofTrace`. It performs all the counting and is what the unit tests
  exercise directly with hand-built synthetic traces. It has no I/O and no network.
- :func:`generate_reasoning_stats` runs the FULL System
  (:class:`demo.ablation.FullNeuroSymbolicConfig`) over a GSM8K subset, collects each
  :class:`~nsr.models.VerifiedOutput`'s proof trace, aggregates them, scores per-item
  correctness against the ground truth, and writes ``reasoning_stats.html`` / ``.json``.
  It performs **no Ollama preflight** — the CLI does that before calling in.

Step categories (a partition)
------------------------------

Every recorded reasoning step is counted **exactly once** into one of three buckets:

- ``accepted_first_pass`` — status ``ACCEPTED`` and **no** repair attempts: the symbolic
  layer accepted the model's step as first emitted.
- ``repaired_successfully`` — status ``REPAIRED``: the step was rejected, routed to the
  bounded repair sub-loop, and an accepted replacement was produced (rejected-then-accepted).
- ``rejected`` — status ``REJECTED`` at the end: the step was never accepted.

``repair_failed`` is a **sub-count of ``rejected``** (not a fourth bucket): a rejected step
whose repair attempts did not yield an accepted replacement — detected when the step
carries repair attempts and/or its trace terminated with ``repair-exhausted``. So
``accepted_first_pass + repaired_successfully + rejected == total_reasoning_steps`` always,
and ``repair_failed <= rejected``.
"""

from __future__ import annotations

import html
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --- Make ``nsr`` (in src/) and the sibling demo modules importable when run directly ---
_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _DEMO_DIR.parent
for _p in (str(_PROJECT_DIR / "src"), str(_DEMO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nsr.models import ErrorRecord, ProofTrace, TerminationReason, ValidationStatus  # noqa: E402
from nsr.proof_trace import NO_RULE_APPLIED  # noqa: E402

# Rule ids whose firing we expose as report-only validation-trigger metrics.
from arithmetic_validation import ARITHMETIC_RULE  # noqa: E402
from goal_alignment import GOAL_ALIGNMENT_RULE_ID  # noqa: E402

#: The arithmetic-correctness rule id ("arithmetic-correctness").
ARITHMETIC_RULE_ID = ARITHMETIC_RULE.rule_id

# Reuse the full-System wiring and the GSM8K scaffolding from the existing demo modules.
import ablation  # noqa: E402
from ablation import FullNeuroSymbolicConfig, FullPlusGoalConfig  # noqa: E402
from run_benchmark import (  # noqa: E402
    DEFAULT_GSM8K_MODEL,
    OUTPUT_DIR,
    VALIDATION_ARITHMETIC,
    VALIDATION_GOAL,
    _gsm8k_dataset,
    make_real_config,
    numeric_answer_match,
)


# --------------------------------------------------------------------------- #
# The aggregate statistics container
# --------------------------------------------------------------------------- #


@dataclass
class ReasoningStats:
    """Aggregate statistics over a collection of :class:`~nsr.models.ProofTrace`.

    The step counters ``accepted_first_pass``, ``repaired_successfully`` and ``rejected``
    PARTITION every recorded reasoning step (each step is counted once), so their sum
    equals ``total_reasoning_steps``. ``repair_failed`` is a sub-count of ``rejected`` —
    rejected steps whose repair attempts did not yield an accepted replacement — so it
    satisfies ``repair_failed <= rejected`` and is NOT added into the partition again.
    """

    total_questions: int = 0
    total_reasoning_steps: int = 0
    accepted_first_pass: int = 0
    rejected: int = 0
    repaired_successfully: int = 0
    repair_failed: int = 0
    #: Number of STEPS where the goal-alignment rule fired — counted once per step whether
    #: the id appears in the step's ``violated_rule_ids`` or in any of its repair attempts'
    #: ``violated_rule_ids`` (report-only metric; does NOT touch the step partition).
    goal_validation_triggered: int = 0
    #: Same per-step count for the arithmetic-correctness rule (for symmetry/completeness).
    arithmetic_validation_triggered: int = 0
    termination_reason_counts: Counter = field(default_factory=Counter)
    #: Counter of ``applied_rule_id`` across every step; a ``None`` applied rule is counted
    #: under the explicit ``"no-rule-applied"`` key (mirrors the Proof_Trace convention).
    rule_utilization: Counter = field(default_factory=Counter)
    #: Counter over every repair attempt's ``violated_rule_ids`` — i.e. how often each rule
    #: (e.g. ``"arithmetic-correctness"``) actually fired the repair sub-loop.
    repair_violation_counts: Counter = field(default_factory=Counter)

    @property
    def repair_success_rate(self) -> Optional[float]:
        """Fraction of repair-triggering steps that ended accepted, or ``None``.

        ``repaired_successfully / (repaired_successfully + repair_failed)`` when the
        denominator is positive; ``None`` when no step ever entered the repair sub-loop
        (so the rate is undefined rather than a misleading ``0.0`` or ``1.0``).
        """
        denom = self.repaired_successfully + self.repair_failed
        if denom <= 0:
            return None
        return self.repaired_successfully / denom

    @property
    def goal_trigger_rate(self) -> float:
        """Fraction of reasoning steps where the goal-alignment rule fired.

        ``goal_validation_triggered / total_reasoning_steps``, or ``0.0`` when there are no
        steps (so the report never divides by zero). Always a float fraction in [0, 1].
        """
        if self.total_reasoning_steps <= 0:
            return 0.0
        return self.goal_validation_triggered / self.total_reasoning_steps

    @property
    def arithmetic_trigger_rate(self) -> float:
        """Fraction of reasoning steps where the arithmetic-correctness rule fired.

        ``arithmetic_validation_triggered / total_reasoning_steps``, or ``0.0`` when there
        are no steps. Always a float fraction in [0, 1].
        """
        if self.total_reasoning_steps <= 0:
            return 0.0
        return self.arithmetic_validation_triggered / self.total_reasoning_steps

    def to_dict(self) -> dict:
        """A JSON-serializable view of the statistics (Counters become plain dicts)."""
        return {
            "total_questions": self.total_questions,
            "total_reasoning_steps": self.total_reasoning_steps,
            "accepted_first_pass": self.accepted_first_pass,
            "rejected": self.rejected,
            "repaired_successfully": self.repaired_successfully,
            "repair_failed": self.repair_failed,
            "repair_success_rate": self.repair_success_rate,
            "goal_validation_triggered": self.goal_validation_triggered,
            "goal_trigger_rate": self.goal_trigger_rate,
            "arithmetic_validation_triggered": self.arithmetic_validation_triggered,
            "arithmetic_trigger_rate": self.arithmetic_trigger_rate,
            "termination_reason_counts": dict(self.termination_reason_counts),
            "rule_utilization": dict(self.rule_utilization),
            "repair_violation_counts": dict(self.repair_violation_counts),
        }


def _termination_key(reason: Optional[TerminationReason]) -> str:
    """Render a termination reason as its string value, or ``"unset"`` when absent."""
    if reason is None:
        return "unset"
    return getattr(reason, "value", str(reason))


def aggregate_trace_stats(traces: list[ProofTrace]) -> ReasoningStats:
    """Compute :class:`ReasoningStats` from a list of proof traces (pure function).

    Each trace contributes one question. For every step:

    - ``ACCEPTED`` with no repair attempts -> ``accepted_first_pass``;
    - ``REPAIRED`` -> ``repaired_successfully`` (rejected-then-accepted via repair);
    - ``REJECTED`` -> ``rejected`` (and additionally ``repair_failed`` when the step
      carries repair attempts OR the trace terminated with ``repair-exhausted``, i.e. the
      repair sub-loop ran but never produced an accepted replacement).

    ``rule_utilization`` counts ``applied_rule_id`` for every step (``None`` ->
    ``"no-rule-applied"``). ``repair_violation_counts`` counts the ``violated_rule_ids`` of
    every recorded repair attempt across all steps. ``termination_reason_counts`` tallies
    each trace's termination reason. No I/O, no mutation of the inputs.
    """
    stats = ReasoningStats(total_questions=len(traces))

    for trace in traces:
        stats.termination_reason_counts[_termination_key(trace.termination_reason)] += 1
        exhausted = trace.termination_reason == TerminationReason.REPAIR_EXHAUSTED

        for step in trace.steps:
            stats.total_reasoning_steps += 1

            # Rule utilization across every step (None -> explicit no-rule indicator).
            rule_key = step.applied_rule_id if step.applied_rule_id is not None else NO_RULE_APPLIED
            stats.rule_utilization[rule_key] += 1

            # Report-only validation-trigger metrics: did the goal-alignment /
            # arithmetic-correctness rule fire for THIS step? Scan the step's own
            # violated_rule_ids PLUS every repair attempt's violated_rule_ids, counting a
            # rule at most once per step (a step that cites the id in both places is not
            # double counted). This does NOT affect the accept/repair/reject partition.
            step_violated: set = set(step.violated_rule_ids)
            for attempt in step.repair_attempts:
                step_violated.update(attempt.violated_rule_ids)
            if GOAL_ALIGNMENT_RULE_ID in step_violated:
                stats.goal_validation_triggered += 1
            if ARITHMETIC_RULE_ID in step_violated:
                stats.arithmetic_validation_triggered += 1

            # Which rules actually fired the repair sub-loop (per repair attempt).
            for attempt in step.repair_attempts:
                for violated in attempt.violated_rule_ids:
                    stats.repair_violation_counts[violated] += 1

            # Partition the step into exactly one of the three buckets.
            if step.status == ValidationStatus.REPAIRED:
                stats.repaired_successfully += 1
            elif step.status == ValidationStatus.ACCEPTED and not step.repair_attempts:
                stats.accepted_first_pass += 1
            elif step.status == ValidationStatus.ACCEPTED:
                # Defensive: an accepted step that nonetheless carries repair attempts is
                # treated as a successful repair (rejected-then-accepted).
                stats.repaired_successfully += 1
            else:  # ValidationStatus.REJECTED (rejected at the end)
                stats.rejected += 1
                if step.repair_attempts or exhausted:
                    stats.repair_failed += 1

    return stats


# --------------------------------------------------------------------------- #
# Report rendering (HTML + JSON)
# --------------------------------------------------------------------------- #

#: Honest framing reused in both the HTML and JSON reports.
HONEST_NOTE = (
    "With the current general-purpose starter rule set, rule utilization is dominated by "
    "the well-formed-step rule plus arithmetic-correctness repairs; richer domain rules "
    "(Phase 3) would make this analysis more interesting."
)


def _fmt_rate(rate: Optional[float]) -> str:
    """Render a repair-success rate as a percentage, or ``n/a`` when undefined."""
    if rate is None:
        return "n/a"
    return f"{rate * 100:.1f}%"


def _fmt_pct(rate: float) -> str:
    """Render a trigger-rate fraction as a percentage (e.g. ``0.0168`` -> ``1.68%``)."""
    return f"{rate * 100:.2f}%"


def _headcount_rows(stats: ReasoningStats) -> str:
    pairs = [
        ("Total questions", stats.total_questions),
        ("Total reasoning steps", stats.total_reasoning_steps),
        ("Accepted on first pass", stats.accepted_first_pass),
        ("Rejected (final)", stats.rejected),
        ("Repaired successfully", stats.repaired_successfully),
        ("Repair failed (subset of rejected)", stats.repair_failed),
        ("Repair success rate", _fmt_rate(stats.repair_success_rate)),
        ("Goal validation triggered", stats.goal_validation_triggered),
        ("Goal trigger rate", _fmt_pct(stats.goal_trigger_rate)),
        ("Arithmetic validation triggered", stats.arithmetic_validation_triggered),
        ("Arithmetic trigger rate", _fmt_pct(stats.arithmetic_trigger_rate)),
    ]
    return "".join(
        f"<tr><td class=\"metric\">{html.escape(label)}</td><td>{html.escape(str(value))}</td></tr>"
        for label, value in pairs
    )


def _counter_rows_with_pct(counter: Counter) -> str:
    total = sum(counter.values())
    if total <= 0:
        return '<tr><td colspan="3" class="na">none recorded</td></tr>'
    rows = []
    for key, count in counter.most_common():
        pct = (count / total) * 100.0
        rows.append(
            f"<tr><td class=\"metric\"><code>{html.escape(str(key))}</code></td>"
            f"<td>{count}</td><td>{pct:.1f}%</td></tr>"
        )
    return "".join(rows)


def _counter_rows(counter: Counter) -> str:
    if not counter:
        return '<tr><td colspan="2" class="na">none recorded</td></tr>'
    return "".join(
        f"<tr><td class=\"metric\"><code>{html.escape(str(key))}</code></td><td>{count}</td></tr>"
        for key, count in counter.most_common()
    )


def _per_item_rows(per_item: list[dict]) -> str:
    if not per_item:
        return '<tr><td colspan="4" class="na">no items evaluated</td></tr>'
    rows = []
    for entry in per_item:
        correct = entry.get("correct")
        if correct is None:
            badge = '<span class="na">n/a</span>'
        elif correct:
            badge = '<span class="pos">✓ correct</span>'
        else:
            badge = '<span class="neg">✗ wrong</span>'
        rows.append(
            "<tr>"
            f"<td class=\"metric\">{html.escape(str(entry.get('item_id', '')))}</td>"
            f"<td>{html.escape(str(entry.get('final_answer', '')))}</td>"
            f"<td>{html.escape(str(entry.get('ground_truth', '')))}</td>"
            f"<td>{badge}</td>"
            "</tr>"
        )
    return "".join(rows)


def _build_stats_html(
    stats: ReasoningStats,
    per_item: list[dict],
    *,
    model: str,
    dataset_label: str,
    accuracy: Optional[float],
    evaluated: int,
    failed_items: int,
) -> str:
    """Render the reasoning-statistics report as a self-contained HTML document."""
    accuracy_text = "n/a" if accuracy is None else f"{accuracy * 100:.1f}% ({evaluated} scored)"

    css = """
    body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
           margin: 0; color: #1a1a1a; background: #f6f7f9; line-height: 1.5; }
    header { background: #0f2540; color: #fff; padding: 24px 32px; }
    header h1 { margin: 0 0 4px; font-size: 22px; }
    header p { margin: 0; color: #b9c7d8; font-size: 14px; }
    main { max-width: 1000px; margin: 0 auto; padding: 24px 32px 64px; }
    .section { background:#fff; border:1px solid #e2e6ea; border-radius:10px;
               padding:20px 24px; margin:20px 0; }
    .note { background:#fff3e0; border:1px solid #ffcc80; color:#7a4f01;
            border-radius:8px; padding:12px 16px; font-size:13px; }
    .real { background:#e8f5e9; border:1px solid #a5d6a7; color:#1b5e20;
            border-radius:8px; padding:10px 14px; font-size:13px; margin-top:10px; }
    table { width:100%; border-collapse:collapse; margin-top:8px; font-size:14px; }
    th, td { padding:10px 12px; border-bottom:1px solid #e2e6ea; text-align:right; }
    th:first-child, td:first-child { text-align:left; }
    thead th { background:#eef1f4; }
    td.metric { font-weight:600; }
    td.na, span.na { color:#9aa3ad; font-style:italic; }
    span.pos { color:#2e7d32; font-weight:600; }
    span.neg { color:#c62828; font-weight:600; }
    code { background:#eef1f4; padding:1px 5px; border-radius:4px; font-size:12px; }
    .kv { font-size:13px; color:#444; }
    footer { text-align:center; color:#98a2ad; font-size:12px; padding:20px; }
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NSR Reasoning Statistics — GSM8K (full system via Ollama)</title>
<style>{css}</style>
</head>
<body>
<header>
  <h1>Neuro-Symbolic System — Reasoning Statistics (GSM8K)</h1>
  <p>What happened inside the reasoning loop of the FULL system — real model
     {html.escape(str(model))} via Ollama</p>
</header>
<main>
  <div class="section">
    <p class="real"><b>Full neuro-symbolic system only.</b> Every step below was produced by
      the real model under live arithmetic validation and the bounded repair sub-loop; the
      counts are read from each query's genuine Proof Trace.</p>
    <p class="kv">Dataset: {html.escape(str(dataset_label))}. Questions evaluated:
      <b>{evaluated}</b>; items that errored (no trace): <b>{failed_items}</b>.
      Final-answer accuracy: <b>{html.escape(accuracy_text)}</b>.</p>
    <p class="note"><b>Honest note.</b> {html.escape(HONEST_NOTE)}</p>
  </div>

  <div class="section">
    <h2>Headcount</h2>
    <table><tbody>
{_headcount_rows(stats)}
    </tbody></table>
    <p class="kv">The first three rows partition every reasoning step
      (accepted-first-pass + repaired-successfully + rejected = total steps); repair-failed
      is the subset of rejected whose repair attempts never yielded an accepted step. The
      goal/arithmetic validation-trigger rows are report-only counts of the STEPS where
      each rule fired (once per step, whether the rule appears in the step's violated rules
      or in any of its repair attempts) over the total step count — they do not affect the
      partition above.</p>
  </div>

  <div class="section">
    <h2>Rule utilization</h2>
    <table>
      <thead><tr><th>Applied rule</th><th>Count</th><th>Share</th></tr></thead>
      <tbody>
{_counter_rows_with_pct(stats.rule_utilization)}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>Repair triggers (violated rules)</h2>
    <table>
      <thead><tr><th>Violated rule</th><th>Times fired</th></tr></thead>
      <tbody>
{_counter_rows(stats.repair_violation_counts)}
      </tbody>
    </table>
    <p class="kv">How often each rule (e.g. <code>arithmetic-correctness</code>) routed a
      step into the bounded repair sub-loop, counted per repair attempt.</p>
  </div>

  <div class="section">
    <h2>Termination reasons</h2>
    <table>
      <thead><tr><th>Reason</th><th>Count</th></tr></thead>
      <tbody>
{_counter_rows(stats.termination_reason_counts)}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>Per-item final answers</h2>
    <table>
      <thead><tr><th>Item</th><th>Final answer</th><th>Ground truth</th><th>Result</th></tr></thead>
      <tbody>
{_per_item_rows(per_item)}
      </tbody>
    </table>
  </div>
</main>
<footer>Generated by demo/reasoning_stats.py — Neuro-Symbolic System-2 Reasoning Architecture</footer>
</body>
</html>
"""


def _stats_to_dict(
    stats: ReasoningStats,
    per_item: list[dict],
    *,
    model: str,
    dataset_label: str,
    accuracy: Optional[float],
    evaluated: int,
    failed_items: int,
) -> dict:
    """A JSON-serializable view of the reasoning-statistics report."""
    return {
        "report": "reasoning-stats",
        "is_real": True,
        "real_model": model,
        "dataset_label": dataset_label,
        "note": HONEST_NOTE,
        "questions_evaluated": evaluated,
        "failed_items": failed_items,
        "final_answer_accuracy": accuracy,
        "stats": stats.to_dict(),
        "per_item": per_item,
    }


# --------------------------------------------------------------------------- #
# Public runner
# --------------------------------------------------------------------------- #


def generate_reasoning_stats(
    model: str = DEFAULT_GSM8K_MODEL,
    host: Optional[str] = None,
    dataset_path: os.PathLike | str | None = None,
    limit: int = 20,
    output_dir: os.PathLike | str = OUTPUT_DIR,
    validation_mode: str = VALIDATION_ARITHMETIC,
) -> dict[str, Path]:
    """Run the FULL System over a GSM8K subset and write reasoning-statistics reports.

    Loads the GSM8K subset via :func:`run_benchmark._gsm8k_dataset`, runs the FULL system
    over each item, collects every :class:`~nsr.models.VerifiedOutput`'s proof trace (an
    :class:`~nsr.models.ErrorRecord` is counted as a failed item and skipped), aggregates
    via :func:`aggregate_trace_stats`, scores per-item correctness with
    :func:`run_benchmark.numeric_answer_match`, and writes ``reasoning_stats.html`` /
    ``.json``. Returns their paths.

    ``validation_mode`` selects which full-system validator drives the run:
    ``"arithmetic"`` (default) uses :class:`demo.ablation.FullNeuroSymbolicConfig`
    (arithmetic correctness only); ``"goal"`` uses
    :class:`demo.ablation.FullPlusGoalConfig` (arithmetic + goal-aligned intent), so the
    statistics reflect goal-alignment rejections and their repairs.

    Performs **no Ollama preflight** — the CLI calls
    :func:`nsr.llm_component.ollama_available` before invoking this. Exercised offline in
    tests by monkeypatching the backend factory and the dataset loader.
    """
    config = make_real_config(model, repeated_run_count=1)
    dataset, dataset_label = _gsm8k_dataset(dataset_path, limit)

    if validation_mode == VALIDATION_GOAL:
        system = FullPlusGoalConfig(model, host, config)
    else:
        system = FullNeuroSymbolicConfig(model, host, config)

    traces: list[ProofTrace] = []
    per_item: list[dict] = []
    failed_items = 0
    correct_count = 0
    evaluated = 0

    for item in dataset:
        result = system.run(item.query)
        if isinstance(result, ErrorRecord):
            failed_items += 1
            per_item.append(
                {
                    "item_id": item.item_id,
                    "final_answer": None,
                    "ground_truth": item.ground_truth,
                    "correct": None,
                    "error": f"{result.failed_component}: {result.reason}",
                }
            )
            continue

        traces.append(result.proof_trace)
        is_correct = numeric_answer_match(result.final_answer, item.ground_truth)
        evaluated += 1
        if is_correct:
            correct_count += 1
        per_item.append(
            {
                "item_id": item.item_id,
                "final_answer": result.final_answer,
                "ground_truth": item.ground_truth,
                "correct": bool(is_correct),
            }
        )

    stats = aggregate_trace_stats(traces)
    accuracy = (correct_count / evaluated) if evaluated > 0 else None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "html": out / "reasoning_stats.html",
        "json": out / "reasoning_stats.json",
    }
    paths["html"].write_text(
        _build_stats_html(
            stats,
            per_item,
            model=model,
            dataset_label=dataset_label,
            accuracy=accuracy,
            evaluated=evaluated,
            failed_items=failed_items,
        ),
        encoding="utf-8",
    )
    paths["json"].write_text(
        json.dumps(
            _stats_to_dict(
                stats,
                per_item,
                model=model,
                dataset_label=dataset_label,
                accuracy=accuracy,
                evaluated=evaluated,
                failed_items=failed_items,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "ReasoningStats",
    "aggregate_trace_stats",
    "generate_reasoning_stats",
    "HONEST_NOTE",
]
