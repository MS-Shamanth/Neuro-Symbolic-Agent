"""CLI: run a demo scenario offline and write its reasoning artifacts.

Usage (from the ``project/`` directory)::

    python demo/run_demo.py                 # runs the default scenario
    python demo/run_demo.py syllogism       # runs a named scenario
    python demo/run_demo.py --list          # lists available scenarios

Everything runs **offline and deterministically** via a scripted
:class:`~nsr.llm_component.MockBackend` (no network, no API key). The run writes four
artifacts into ``project/demo/output/``:

- ``<scenario>_proof_trace.txt`` — the human-readable Proof Trace (``render_trace``).
- ``<scenario>_reasoning.mmd``   — the Mermaid reasoning-visualization source.
- ``<scenario>_reasoning.dot``   — the Graphviz DOT reasoning-visualization source.
- ``<scenario>_reasoning_report.html`` — a self-contained report that embeds the Mermaid
  diagram (via the mermaid.js CDN) plus step-by-step cards and a working-memory panel.
"""

from __future__ import annotations

import argparse
import html
import os
import sys
from pathlib import Path

# --- Make ``nsr`` (in src/) and the sibling demo modules importable when run directly ---
_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _DEMO_DIR.parent
for _p in (str(_PROJECT_DIR / "src"), str(_DEMO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nsr.models import VerifiedOutput  # noqa: E402
from nsr.proof_trace import NO_RULE_APPLIED, applied_rule_label  # noqa: E402
from nsr.proof_trace_export import render_trace  # noqa: E402
from nsr.trace_visualizer import to_dot, to_mermaid  # noqa: E402

import scenarios  # noqa: E402

#: Default output directory (created if absent).
OUTPUT_DIR = _DEMO_DIR / "output"

#: Outcome → (label, css class) styling shared by the step cards.
_STATUS_STYLE = {
    "accepted": ("Validation ✓ accepted", "accepted"),
    "rejected": ("Validation ✗ rejected", "rejected"),
    "repaired": ("Repaired → accepted", "repaired"),
}

_CSS = """
:root { --green:#2e7d32; --red:#c62828; --amber:#f9a825; --ink:#1a1a1a; --muted:#666; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0; color: var(--ink); background: #f6f7f9; line-height: 1.5; }
header { background: #0f2540; color: #fff; padding: 24px 32px; }
header h1 { margin: 0 0 4px; font-size: 22px; }
header p { margin: 0; color: #b9c7d8; font-size: 14px; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 32px 64px; }
.section { background: #fff; border: 1px solid #e2e6ea; border-radius: 10px;
           padding: 20px 24px; margin: 20px 0; }
.section h2 { margin: 0 0 12px; font-size: 17px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px;
         font-weight: 600; color: #fff; }
.badge.accepted { background: var(--green); }
.badge.rejected { background: var(--red); }
.badge.repaired { background: var(--amber); color: #3a2c00; }
.offline { background: #fff8e1; border: 1px solid #ffe082; color: #6d5200;
           border-radius: 8px; padding: 10px 14px; font-size: 13px; margin: 0 0 8px; }
.cards { display: grid; gap: 14px; }
.card { border: 1px solid #e2e6ea; border-left-width: 6px; border-radius: 8px;
        padding: 14px 16px; background: #fcfcfd; }
.card.accepted { border-left-color: var(--green); }
.card.rejected { border-left-color: var(--red); }
.card.repaired { border-left-color: var(--amber); }
.card .row { display: flex; justify-content: space-between; align-items: center;
             gap: 12px; flex-wrap: wrap; }
.card .lf { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 14px;
            background: #eef1f4; padding: 6px 10px; border-radius: 6px; margin: 8px 0;
            display: inline-block; }
.meta { font-size: 13px; color: var(--muted); }
.meta b { color: var(--ink); }
.rule { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }
.origin { font-size: 11px; font-weight: 600; padding: 1px 7px; border-radius: 10px;
          border: 1px solid #cbd2d9; color: #44525f; }
.origin.learned { background: #e8f0fe; border-color: #aac4f6; color: #1a3e8c; }
.origin.seeded { background: #f0f0f0; }
.repairs { margin: 8px 0 0; padding: 8px 12px; background: #fff6da; border-radius: 6px;
           font-size: 13px; }
.panel { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.panel .box { border: 1px solid #e2e6ea; border-radius: 8px; padding: 12px 14px;
              background: #fcfcfd; }
.panel .box h3 { margin: 0 0 8px; font-size: 14px; }
.kv { font-size: 13px; }
.kv .k { color: var(--muted); }
ul.tight { margin: 6px 0 0; padding-left: 18px; }
ul.tight li { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 13px; }
.metric { display: inline-block; margin-right: 22px; }
.metric .v { font-size: 24px; font-weight: 700; }
.metric .l { font-size: 12px; color: var(--muted); display: block; }
.mermaid { background: #fff; border: 1px dashed #cbd2d9; border-radius: 8px; padding: 12px; }
footer { text-align: center; color: #98a2ad; font-size: 12px; padding: 20px; }
"""


def _origin_label(step) -> tuple[str, str]:
    """Return ``(text, css)`` for the learned/seeded marker of an applied rule."""
    origin = getattr(step, "applied_rule_origin", None)
    if origin is None:
        return ("—", "")
    value = getattr(origin, "value", str(origin))
    return (value, value)


def _step_logic_form(step) -> str:
    """The machine-checkable logic form of a step, or its raw text as a fallback."""
    if step.representation is not None and step.representation.logic_form:
        return step.representation.logic_form
    return step.step_text or "(empty)"


def _render_step_card(step) -> str:
    status = step.status.value
    label, css = _STATUS_STYLE.get(status, (status, ""))
    rule_id = applied_rule_label(step)
    origin_text, origin_css = _origin_label(step)
    origin_html = ""
    if rule_id != NO_RULE_APPLIED:
        origin_html = f'<span class="origin {origin_css}">{html.escape(origin_text)}</span>'

    parts = [
        f'<div class="card {css}">',
        '  <div class="row">',
        f'    <span class="badge {css}">{html.escape(label)}</span>',
        f'    <span class="meta">step #{step.sequence}</span>',
        "  </div>",
        f'  <div class="lf">{html.escape(_step_logic_form(step))}</div>',
        '  <div class="meta">'
        f'applied rule: <span class="rule">{html.escape(rule_id)}</span> {origin_html}'
        "</div>",
    ]
    if step.violated_rule_ids:
        parts.append(
            '  <div class="meta">violated rules: '
            f'<b>{html.escape(", ".join(step.violated_rule_ids))}</b></div>'
        )
    for attempt in step.repair_attempts:
        repaired = (
            attempt.repaired_step.logic_form
            if attempt.repaired_step is not None
            else "(unrepaired)"
        )
        violated = ", ".join(attempt.violated_rule_ids) or NO_RULE_APPLIED
        parts.append(
            f'  <div class="repairs">repair attempt #{attempt.attempt_index}: '
            f"violated [{html.escape(violated)}] → repaired: "
            f"<b>{html.escape(repaired)}</b></div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def _render_buffer_panel(run: "scenarios.ScenarioRun") -> str:
    orchestrator = run.orchestrator
    controller = orchestrator._controller  # read-only access for the demo snapshot
    goal = controller.goal_buffer
    state = controller.state()

    sub_goal_items = "".join(
        f'<li>{"✓" if sg.satisfied else "•"} {html.escape(sg.description)}</li>'
        for sg in goal.sub_goals
    ) or "<li>(none)</li>"

    declarative = state.declarative_memory
    decl_items = "".join(
        f"<li>{html.escape(r.logic_form)}</li>" for r in declarative
    ) or "<li>(empty)</li>"

    imaginal = state.imaginal_buffer.logic_form if state.imaginal_buffer else "(empty)"

    rules = orchestrator.procedural_memory
    learned_ids = orchestrator.learned_rule_ids
    rule_items = "".join(
        f'<li>{html.escape(r.rule_id)} — IF {html.escape(r.condition or "(always)")} '
        f'THEN {html.escape(r.action or "(none)")}'
        + (' <span class="origin learned">learned</span>' if r.rule_id in learned_ids else "")
        + "</li>"
        for r in rules
    ) or "<li>(none)</li>"

    result = run.result
    if isinstance(result, VerifiedOutput):
        faithfulness = f"{result.faithfulness_score:.2f}"
        final_answer = html.escape(result.final_answer)
        trace = result.proof_trace
    else:
        faithfulness = "n/a"
        final_answer = "n/a"
        trace = orchestrator.last_trace
    termination = (
        trace.termination_reason.value
        if trace is not None and trace.termination_reason is not None
        else "n/a"
    )

    return f"""
    <div class="section">
      <h2>Outcome</h2>
      <div class="metric"><span class="v">{faithfulness}</span>
        <span class="l">Faithfulness Score</span></div>
      <div class="metric"><span class="v">{html.escape(termination)}</span>
        <span class="l">Termination reason</span></div>
      <div class="metric"><span class="v">{len(trace.steps) if trace else 0}</span>
        <span class="l">Reasoning steps</span></div>
      <p class="kv"><span class="k">Final answer:</span>
        <b class="rule">{final_answer}</b></p>
    </div>
    <div class="section">
      <h2>Working-memory buffers (final snapshot)</h2>
      <div class="panel">
        <div class="box">
          <h3>Goal Buffer</h3>
          <div class="kv">{html.escape(goal.description)}
            (goal satisfied: <b>{goal.satisfied}</b>)</div>
          <ul class="tight">{sub_goal_items}</ul>
        </div>
        <div class="box">
          <h3>Declarative Memory (accepted conclusions)</h3>
          <ul class="tight">{decl_items}</ul>
        </div>
        <div class="box">
          <h3>Imaginal Buffer (partial representation)</h3>
          <div class="lf">{html.escape(imaginal)}</div>
        </div>
        <div class="box">
          <h3>Procedural Memory (available rules)</h3>
          <ul class="tight">{rule_items}</ul>
        </div>
      </div>
    </div>
    """


def build_html(run: "scenarios.ScenarioRun", mermaid_src: str) -> str:
    """Render the self-contained reasoning report for a scenario run."""
    scenario = run.scenario
    trace = (
        run.result.proof_trace
        if isinstance(run.result, VerifiedOutput)
        else run.orchestrator.last_trace
    )
    step_cards = "\n".join(_render_step_card(s) for s in (trace.steps if trace else []))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NSR Reasoning Report — {html.escape(scenario.title)}</title>
<style>{_CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>document.addEventListener("DOMContentLoaded",function(){{
  if (window.mermaid) {{ mermaid.initialize({{ startOnLoad: true, theme: "neutral" }}); }}
}});</script>
</head>
<body>
<header>
  <h1>Neuro-Symbolic Reasoning Report</h1>
  <p>{html.escape(scenario.title)}</p>
</header>
<main>
  <div class="section">
    <p class="offline">Offline &amp; deterministic demo: every reasoning step is produced
      by a scripted MockBackend (no network, no API key). The only external resource is the
      mermaid.js diagram library loaded from a CDN for rendering below.</p>
    <h2>Scenario</h2>
    <p>{html.escape(scenario.description)}</p>
    <p class="kv"><span class="k">Query:</span> <b>{html.escape(scenario.query)}</b></p>
  </div>

  <div class="section">
    <h2>Reasoning visualization</h2>
    <pre class="mermaid">
{mermaid_src}
    </pre>
  </div>

  <div class="section">
    <h2>Reasoning steps</h2>
    <div class="cards">
{step_cards}
    </div>
  </div>

  {_render_buffer_panel(run)}
</main>
<footer>Generated by demo/run_demo.py — Neuro-Symbolic System-2 Reasoning Architecture</footer>
</body>
</html>
"""


def generate_demo(
    scenario_name: str = scenarios.DEFAULT_SCENARIO,
    output_dir: os.PathLike | str = OUTPUT_DIR,
) -> dict[str, Path]:
    """Run ``scenario_name`` offline and write all four artifacts. Returns their paths."""
    run = scenarios.run_scenario(scenario_name)
    trace = (
        run.result.proof_trace
        if isinstance(run.result, VerifiedOutput)
        else run.orchestrator.last_trace
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = scenario_name

    mermaid_src = to_mermaid(trace)
    dot_src = to_dot(trace)
    trace_txt = render_trace(trace)
    html_doc = build_html(run, mermaid_src)

    paths = {
        "trace_txt": out / f"{stem}_proof_trace.txt",
        "mermaid": out / f"{stem}_reasoning.mmd",
        "dot": out / f"{stem}_reasoning.dot",
        "html": out / f"{stem}_reasoning_report.html",
    }
    paths["trace_txt"].write_text(trace_txt, encoding="utf-8")
    paths["mermaid"].write_text(mermaid_src, encoding="utf-8")
    paths["dot"].write_text(dot_src, encoding="utf-8")
    paths["html"].write_text(html_doc, encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Neuro-Symbolic reasoning demo (offline).")
    parser.add_argument(
        "scenario",
        nargs="?",
        default=scenarios.DEFAULT_SCENARIO,
        help=f"scenario to run (default: {scenarios.DEFAULT_SCENARIO})",
    )
    parser.add_argument("--list", action="store_true", help="list available scenarios and exit")
    args = parser.parse_args(argv)

    if args.list:
        print("Available scenarios:")
        for name in sorted(scenarios.SCENARIOS):
            print(f"  - {name}: {scenarios.get_scenario(name).title}")
        return 0

    if args.scenario not in scenarios.SCENARIOS:
        print(f"unknown scenario {args.scenario!r}; use --list to see options", file=sys.stderr)
        return 2

    paths = generate_demo(args.scenario)
    print(f"Scenario '{args.scenario}' ran offline. Artifacts written to {OUTPUT_DIR}:")
    for key, path in paths.items():
        size = path.stat().st_size
        print(f"  - {path.name}  ({size} bytes)")
    print("\nOpen the *_reasoning_report.html file in a browser to view the full report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
