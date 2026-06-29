# Neuro-Symbolic System-2 Reasoning Architecture

> A hybrid reasoning-control architecture that makes multi-step LLM reasoning
> **traceable, verifiable, and repairable** — pairing a neural LLM (System 1) with an
> ACT-R-style symbolic controller (System 2) that validates every intermediate step
> *before* it propagates.

[![tests](https://img.shields.io/badge/tests-~650%20passing-brightgreen)](#testing)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

---

## What this is

Large Language Models produce fluent multi-step reasoning but give no guarantee that
intermediate steps are well-formed, computed correctly, or aligned with the user's goal.
A single bad step compounds into a confidently wrong answer, and the final answer alone
hides *where* it went wrong.

This project wraps an LLM in a deliberate symbolic controller. Each candidate reasoning
step is forced into a structured form, translated into a machine-checkable symbolic
representation, and **accepted, rejected, or repaired** before the next step is generated.
Every action is journaled into an append-only **proof trace** that backs both the metrics
and the visualizations.

The contribution is **observability and control**, not better language modeling. By
validating each step it detects and repairs the *classes of hallucination the symbolic
layer can verify* (structural, arithmetic, and goal-alignment errors). It does **not** claim
to eliminate hallucination it cannot check (e.g. a fluent false world-knowledge premise).

## Architecture

```
            ┌──────────────────────────────────────────────────────────────┐
            │                    Closed reasoning cycle                      │
            │                                                                │
   query ─► │  ┌───────────┐   ┌────────────┐   ┌──────────────┐            │
            │  │ System 1  │   │ Translate  │   │  System 2    │            │
            │  │   (LLM)   │──►│ step → sym │──►│  ACT-R       │            │
            │  │ generate  │   │  symbolic  │   │  controller  │            │
            │  └───────────┘   └────────────┘   └──────┬───────┘            │
            │        ▲                                  │                    │
            │        │                          ┌───────▼────────┐           │
            │        │   reject → repair        │  3-layer       │           │
            │        └──────────────────────────┤  validation    │           │
            │                                    └───────┬────────┘           │
            │                                    accept  │                    │
            │                                            ▼                    │
            │                                  Declarative Memory ─► answer   │
            └──────────────────────────────────────────────────────────────┘
                                    every event ─► append-only Proof Trace
```

ACT-R working memory: **Goal**, **Declarative**, **Procedural** (IF–THEN production rules),
and **Imaginal** buffers. When multiple rules match, exactly one is chosen by a deterministic
conflict-resolution policy (priority → specificity → recency), so runs are reproducible
under a single seed.

## Three layers of verification

The conceptual core: validation organized by **increasing semantic depth**.

| Layer | Question it answers | Example rejection |
|-------|---------------------|-------------------|
| **Structural** | Is the reasoning step well-formed? | Step does not parse into a valid symbolic form. |
| **Arithmetic** | Is the computation correct? | `40 * 13 = 540` (true value 520) → rejected, repaired. |
| **Goal-aligned** | Is the step solving the intended objective? | Goal = *profit*; step computes cost `40 * 13 = 520`. Math is right, but the goal operation (subtraction) is absent → goal mismatch. |

**Goal-aligned validation** — checking *intent*, not just numbers — is the novel element.
Many systems check arithmetic; very few check whether a step serves the goal.

## Installation

Requires Python 3.10+.

```bash
cd project
python -m pip install -e ".[dev]"
```

Optional, for real-model runs: install [Ollama](https://ollama.com) and pull a chat model:

```bash
ollama pull qwen3:8b
```

## Quickstart

```bash
cd project

# Run the offline demo (deterministic mock backend, no network needed)
python demo/run_demo.py

# Launch the local web UI (stdlib only, no extra deps)
python demo/web_app.py        # then open http://127.0.0.1:8000

# Benchmark against baselines on the bundled sample
python demo/run_benchmark.py --stats
```

## Reproducing the experiments

All runs are seeded and write a proof trace; configuration and seed are persisted per run.

```bash
cd project

# 5-config ablation (Plain LLM | +structural | +ACT-R | +arithmetic | +goal-aligned)
python demo/run_benchmark.py --backend ollama --model qwen3:8b --dataset gsm8k \
    --ablation --limit 50 --path /path/to/gsm8k/test.jsonl

# Stats + goal-aligned validation pass (repair rate, goal-trigger rate, rule utilization)
python demo/run_benchmark.py --backend ollama --model qwen3:8b --dataset gsm8k \
    --stats --goal-validation --limit 50 --path /path/to/gsm8k/test.jsonl
```

> Real-model runs are slow (~30–45s/problem) and make many local LLM calls. The mock backend
> path stays available for fast, deterministic, offline runs and is what the test suite uses.

## Testing

~650 unit and Hypothesis property-based tests, all offline and deterministic.

```bash
cd project
pytest
```

## Project layout

```
project/
  src/nsr/        # package source
    models/       # core data models (enums + dataclasses)
  demo/           # web UI, benchmark + ablation runners, Ollama backend,
                  # GSM8K loader, arithmetic + goal-aligned validation
  tests/          # pytest + Hypothesis test suite
  pyproject.toml
paper/            # IEEE-style draft paper
docs/             # presentations and architecture PDF
.kiro/specs/      # requirements, design, and task spec
```

## Status

The architecture, test suite, demo, and ablation tooling are complete. The immediate
priority is collecting evidence at scale: an official GSM8K run (50–100 problems) to quantify
repair rate, goal-validation trigger rate, and rule utilization. See `paper/` for the draft
write-up.

## License

MIT (see `LICENSE`).
