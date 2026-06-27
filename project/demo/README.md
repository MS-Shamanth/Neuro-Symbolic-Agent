# Neuro-Symbolic Reasoning — Demo Package

A runnable, **fully offline** demonstration of the Neuro-Symbolic System-2 Reasoning
Architecture. It drives the real reasoning pipeline end-to-end and produces shareable
artifacts: a human-readable proof trace, Mermaid and Graphviz reasoning visualizations, a
self-contained HTML reasoning report, and a benchmark comparison against baseline methods.

Run all commands from the **`project/`** directory.

## Quick start

```bash
# Run a reasoning scenario and generate its artifacts (proof trace, .mmd, .dot, HTML)
python demo/run_demo.py                 # default scenario (arithmetic-repair)
python demo/run_demo.py syllogism       # a specific scenario
python demo/run_demo.py --list          # list available scenarios

# Run the multi-domain benchmark comparing the System vs Plain LLM vs Chain-of-Thought
python demo/run_benchmark.py
```

All outputs are written to `project/demo/output/`.

## GSM8K multi-step arithmetic benchmark

`run_benchmark.py` also has a GSM8K mode that stresses the reason → validate → repair loop
on multi-step arithmetic, with the `ArithmeticValidationEngine` actually checking each
intermediate calculation (a wrong step is caught and repaired instead of being carried
through to a wrong final answer). It compares the System against Plain LLM and
Chain-of-Thought using a numeric final-answer matcher applied symmetrically to every method.

```bash
# REAL model (local Ollama) over the bundled ORIGINAL sample (first 5 items by default)
python demo/run_benchmark.py --backend ollama --dataset gsm8k --model qwen3:8b

# Point --path at the OFFICIAL GSM8K test set and evaluate the first 50 items
python demo/run_benchmark.py --backend ollama --dataset gsm8k --model qwen3:8b \
    --path path/to/grade-school-math/grade_school_math/data/test.jsonl --limit 50
```

The reports land in `benchmark_report_gsm8k.html` / `.json`, labelled with the model, the
dataset, and that arithmetic validation is active. A real run is slow (≈45s/problem on
`qwen3:8b`), so keep `--limit` small.

> **The bundled `demo/data/gsm8k_sample.jsonl` is an ORIGINAL sample written for this
> offline demo — it is NOT the official GSM8K corpus.** It exists so the GSM8K mode runs
> out-of-the-box and is clearly labelled as a sample everywhere it is reported. To run the
> real benchmark, download the official `test.jsonl` from the
> [openai/grade-school-math](https://github.com/openai/grade-school-math) repository
> (`grade_school_math/data/test.jsonl`) and pass it with `--path`. Nothing is downloaded by
> the demo.

### Other standard benchmark formats

`demo/datasets.py` also includes drop-in loaders for other standard formats so official
files can be used later without code changes (stdlib `json` only, no network):

- `load_gsm8k(path)` — official GSM8K JSONL (`{"question", "answer"}` ending with
  `#### <number>`).
- `load_multiple_choice(path)` — ARC / CommonsenseQA-style multiple choice (nested or flat
  shapes); query is the stem plus enumerated choices, ground truth is the correct choice
  text. Official files: [ARC](https://allenai.org/data/arc),
  [CommonsenseQA](https://www.tau-nlp.sites.tau.ac.il/commonsenseqa).
- `load_strategyqa(path)` — StrategyQA-style `{"question", "answer": true/false}` →
  `yes`/`no`. Official file: [StrategyQA](https://allenai.org/data/strategyqa).
- `load_benchmark(name, path, limit)` — dispatcher over
  `{gsm8k, multiple-choice, strategyqa}`.

Every loader skips malformed lines and runs items through `nsr.dataset_loader.load_dataset`
for the same validation any dataset receives.

## Web interface

Prefer to browse interactively? A tiny, **zero-dependency** web app (Python standard
library only — no Flask/FastAPI) lets you pick a scenario in the browser and view the same
reasoning report the file export produces:

```bash
python demo/web_app.py                 # serve on http://127.0.0.1:8000/
python demo/web_app.py --port 9000     # choose a different port
python demo/web_app.py --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000/** and pick a scenario. The index page lists every
scenario; selecting one runs it in-process via `scenarios.run_scenario` and renders the
exact same self-contained report `demo/run_demo.py` writes (step cards, Mermaid diagram,
working-memory buffers, Faithfulness Score).

> **Localhost only.** The server binds to `127.0.0.1` (loopback) and is an
> **unauthenticated local demo server intended for localhost use only** — do not expose it
> to other hosts. Like the rest of the demo it runs fully offline via the scripted
> `MockBackend` (no network, no API key), and only ever executes the predefined scenarios
> selected by name (nothing from the request is evaluated as code).

## What's in here

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` | Architecture overview with a Mermaid diagram of the dual-process pipeline and the four-stage cycle. |
| `scenarios.py` | Reusable helpers that wire a fully-offline `PipelineOrchestrator` over a scripted `MockBackend`, plus three example scenarios. |
| `run_demo.py` | CLI: run a scenario and write its reasoning artifacts. |
| `web_app.py` | Zero-dependency stdlib web interface: pick a scenario in the browser and view its reasoning report (localhost only, offline). |
| `run_benchmark.py` | CLI: run a small multi-domain benchmark and write the comparison report. |
| `output/` | Generated artifacts (created on first run). |

## Scenarios

- **`syllogism`** — a three-step deduction where every step is accepted (a perfect
  faithfulness score and `goal-satisfied` termination).
- **`arithmetic-repair`** — the second step is rejected by a production rule, repaired,
  and then accepted, so the visualization shows the **Validation ✗ → Repair →
  Validation ✓** path.
- **`multi-hop`** — a three-hop derivation where each hop builds on the accepted
  conclusions already in Declarative Memory.

## Output files

Running `python demo/run_demo.py <scenario>` produces, in `demo/output/`:

- **`<scenario>_proof_trace.txt`** — the human-readable proof trace (`render_trace`): each
  step in execution order with its outcome, applied rule, and repair attempts.
- **`<scenario>_reasoning.mmd`** — the Mermaid source for the reasoning-flow diagram.
- **`<scenario>_reasoning.dot`** — the Graphviz DOT source for the same diagram (render
  with `dot -Tpng <file>.dot -o out.png` if Graphviz is installed).
- **`<scenario>_reasoning_report.html`** — a self-contained report that embeds the Mermaid
  diagram (rendered via the mermaid.js CDN) and shows step-by-step cards (text, validation
  outcome with color, applied rule id + learned/seeded marker) plus a working-memory panel
  (Goal Buffer, Declarative Memory, Imaginal Buffer, available rules) and the Faithfulness
  Score and termination reason. Open it in any browser.

Running `python demo/run_benchmark.py` produces:

- **`benchmark_report.html`** — a comparison table of **Plain LLM** vs **Chain-of-Thought**
  vs the **Neuro-Symbolic System** across the report metrics (accuracy, step hallucination
  rate, faithfulness, mean / p95 latency, latency overhead, reasoning consistency), showing
  the System value, each baseline value, and the difference.
- **`benchmark_report.json`** — the same comparison as machine-readable JSON.

## Offline & deterministic

The demo is **fully offline and deterministic**. Every reasoning step is produced by a
scripted `nsr.llm_component.MockBackend` — no network calls and no API key are required.
Random seeding is fixed, and benchmark latencies come from injected deterministic step
clocks, so results are reproducible. The benchmark's answers and latencies are illustrative
of the architecture's behavior and clearly labelled as mock values, not measurements of a
real LLM.

The **only** external resource used is the mermaid.js library, loaded from a CDN *inside the
generated HTML report* purely for diagram rendering in the browser. The proof trace, `.mmd`,
`.dot`, and benchmark JSON are produced entirely offline.

## Plugging in a real LLM

Switching from the offline mock to a real model is a **configuration change, not a source
edit**. The `LLMComponent` selects its backend from `SystemConfig.llm_selection` via
`nsr.llm_component.build_backend`:

- A hosted-API selection (e.g. `gpt-4o-mini`) builds a `HostedAPIBackend` whose endpoint
  and credentials are read from configuration / environment variables
  (`NSR_LLM_ENDPOINT`, `NSR_LLM_API_KEY`, `NSR_LLM_MODEL_ID`) — never from source.
- A `local-*` selection (e.g. `local-llama3`) builds a `LocalRuntimeBackend`.

In other words, the demo wires the orchestrator with `MockBackend` for reproducibility; a
production wiring would build the backend with `build_backend(config)` and supply
credentials through the environment. No reasoning logic changes.
