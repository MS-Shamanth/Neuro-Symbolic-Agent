# Project Backlog & Resume Notes

> Picked up when the user says **"continue"**. This file is the source of truth for
> where we stopped and what to do next.

_Last updated: end of session before the user went to sleep._

---

## CURRENT STATE (verified)

- The base spec (Requirements 1–13) is **fully implemented and tested**.
  - Code lives under `project/` (`src/nsr/`, `tests/`, `pyproject.toml`).
  - Full suite: **375 tests passing, 0 failures** (incl. all Hypothesis property tests).
  - All 53 original tasks in `tasks.md` are marked complete.
- We then started adding **two new features** to the spec:
  1. **Requirement 14 — Adaptive Rule Learning** ✅ added to `requirements.md`
  2. **Requirement 15 — Reasoning Visualization** ✅ added to `requirements.md`
  - New glossary terms added: Learned_Rule, Seeded_Rule, Candidate_Rule,
    Rule_Provenance, Corroboration_Threshold, Rule_Learning, Learned_Rule_Store,
    Reasoning_Visualization.
  - `requirements.md` passed diagnostics (clean).

### Where we stopped (IMPORTANT)

- The **design phase was IN PROGRESS but CANCELLED** mid-run. `design.md` has **NOT**
  yet been extended for Requirements 14 & 15. **This is the first thing to redo.**
- `tasks.md` has **NOT** been extended yet (still the original 53 tasks).

---

## RESUME PLAN — when user says "continue", do these IN ORDER

### Step 1 — Extend `design.md` for Requirements 14 & 15 (was cancelled; redo)
Append (do not rewrite existing sections):
- **Adaptive Rule Learning subsystem** (new "Rule Learner" component):
  - Induce Candidate_Rules from accepted steps of a goal-satisfied Proof_Trace.
  - `Learned_Rule_Store` data model: versioned, persisted, holds candidates +
    provenance (trace ids, step ids) + corroboration counts + learned/seeded mark.
  - Promotion rule: promote only after Corroboration_Threshold (default 2) independent
    successful traces AND no contradiction with existing rules (a candidate must never
    accept a step an existing rule rejects); discard + log conflicts.
  - Determinism/reproducibility: seeded induction/corroboration/promotion recorded in
    RunRecord; integrate with ReproducibilityManager + persistence.
  - New config: `rule_learning_enabled` (default false), `corroboration_threshold`
    (default 2), `max_learned_rules` (cap) — wire into ConfigManager defaults/ranges.
  - Extend ProofStep to record learned-vs-seeded applied rule (backward compatible).
  - Disabled-path guarantee: identical to Reqs 1–13 when off.
  - New dataclasses + interface sketches: CandidateRule, LearnedRule, RuleProvenance,
    LearnedRuleStore; RuleLearner.induce(trace)/.corroborate()/.promote().
- **Reasoning Visualization exporter**:
  - Pure function of the existing ProofTrace (no new state), lossless.
  - Emit Mermaid and/or Graphviz DOT text: Goal Buffer → Step → Validation
    (accepted/rejected/repaired) → Repair branch → … → Final Answer / termination.
  - Distinguish accepted/rejected/repaired; annotate applied rule id (or
    no-rule-applied) + learned-vs-seeded; empty-trace placeholder.
  - Interface: `to_mermaid(trace) -> str`, `to_dot(trace) -> str` alongside
    `proof_trace_export.py`.
- Update the High-Level Component View diagram + decision tables to include both.
- Map each design element to the Req 14/15 acceptance criteria.

### Step 2 — Extend `tasks.md` with implementation tasks for Reqs 14 & 15
- Add tasks (with sub-tasks + PBT/unit/integration tests) for the Rule Learner and the
  Visualization exporter, following the existing tasks.md style and dependency-graph
  (waves) format. Reference Req 14.x / 15.x clauses.

### Step 3 — Run all the NEW tasks (orchestrator mode)
- Queue + dispatch via the spec-task-execution subagents, same as before.
- Verify full suite stays green after each wave.

---

## FEATURE BACKLOG (user's longer-term wishlist — spec these AFTER 14 & 15)

The user listed these for later. Treat each as a future spec/feature; **do not start
without confirming scope**, since several have heavy external dependencies.

1. **Learning production rules** — (this is Requirement 14, in progress now).
2. **Memory retrieval upgrade** — replace simple retrieval with a **vector database +
   symbolic memory hierarchy** (Declarative_Memory becomes a tiered store).
3. **Multi-agent reasoning** — one agent proposes, one critiques, ACT-R arbitrates.
4. **Tool execution** — let ACT-R call Python / calculators / databases / search tools
   when a production rule requires external verification.
5. **Formal verification** — use an SMT solver or symbolic theorem prover (e.g. Z3) for
   mathematical proofs. ("Genuinely novel.")

## USER'S EXPLICIT DIRECTION FOR AFTER FEATURES

> "Since the implementation is done, stop adding features for a while. Instead, focus
> on making it demonstrable."

Once Reqs 14 & 15 (and any agreed features) are done, **shift focus to demo/presentation**:
- A clean **architecture diagram**.
- A simple **web interface** where users see each reasoning step live.
- **Visualizations** of the Goal Buffer, Imaginal Buffer, and the validation process.
- **Benchmark results** comparing: Plain LLM vs Chain-of-Thought vs the Neuro-Symbolic
  System (quantitative tables/plots).
- Rationale: a strong demo + quantitative results makes a bigger impression than
  another module.

---

## QUICK CHECKLIST
- [x] Step 1: Extend design.md (Reqs 14 & 15)
- [x] Step 2: Extend tasks.md (Reqs 14 & 15) — Tasks 16-19 appended
- [x] Step 3: Run all new tasks, keep suite green — DONE: 480 tests passing, reliably green
- [ ] NEXT: pause feature work, build the demo (web UI + diagrams + benchmark report)
- [ ] Future specs (confirm scope first): vector memory, multi-agent, tool execution, SMT/formal verification

## STATUS UPDATE — Requirements 14 & 15 COMPLETE

Both new features are fully implemented and verified (480 tests, green across repeated runs):
- **Requirement 14 — Adaptive Rule Learning**: `src/nsr/rule_learner.py` (induce/corroborate/
  promote/contradicts), `src/nsr/models/learning.py` (CandidateRule, LearnedRule,
  LearnedRuleStore, provenance, versioned store_to_dict/from_dict), config + run-record
  extensions, orchestrator wiring on the goal-satisfied path (best-effort, disabled-path
  identical to Req 1-13), learned-vs-seeded markers in the Proof_Trace. All 10 properties tested.
- **Requirement 15 — Reasoning Visualization**: `src/nsr/trace_visualizer.py` (`to_mermaid`,
  `to_dot`) — pure, lossless functions over the Proof_Trace; outcome-distinguished, rule-annotated
  nodes; empty-trace placeholder. All 3 properties tested.
- Added `tests/conftest.py` with a "stable" Hypothesis profile so the property suite is not
  timing-flaky (no assertions weakened).

### THE DEMO PHASE IS DONE ✅
Built under `project/demo/` (fully offline/deterministic via scripted MockBackend; full suite green):
- `ARCHITECTURE.md` — clean Mermaid architecture diagram + component/four-stage-cycle write-up.
- `scenarios.py` — wires the real pipeline over a scripted backend; 3 scenarios incl. a
  rejection→repair→accept path; optional rule-learning variant.
- `run_demo.py` — CLI producing per-scenario artifacts in `demo/output/`: proof_trace.txt,
  reasoning.mmd, reasoning.dot, and a self-contained reasoning_report.html (step cards with
  accept/reject/repair colors, applied rule + learned/seeded marker, Goal/Declarative/
  Procedural/Imaginal buffer panels, faithfulness + termination).
- `run_benchmark.py` — benchmark_report.html + .json comparing Plain LLM vs Chain-of-Thought
  vs the Neuro-Symbolic System across all report metrics (System value, baseline value, diff).
- `web_app.py` — zero-dependency stdlib web UI (localhost only, unauthenticated, offline):
  `python demo/web_app.py` → http://127.0.0.1:8000/ , pick a scenario, view the live report.
- `README.md` + smoke tests (`tests/test_demo_smoke.py`, `tests/test_web_app_smoke.py`).

### REAL-MODEL BENCHMARK — DONE (honest result) ✅
- Ollama backend added (`src/nsr/llm_component.py`: `OllamaBackend`, `build_ollama_backend`,
  `ollama_available`), stdlib-only, mocked-HTTP tests. MockBackend untouched for tests.
- `demo/run_benchmark.py --backend ollama --model qwen3:8b` runs the System + Plain LLM +
  Chain-of-Thought against a REAL model via Ollama; writes `benchmark_report_real.html/.json`.
- Fixed the real System path with `scenarios.RealModelTranslationLayer` (tells the model the
  exact `{"logic_form": "..."}` schema; final step = concise answer).
- Fixed a measurement artifact: switched to `lenient_answer_match` (applied to ALL methods)
  so correct-but-verbose baseline answers ("Blue.", "Yes, cats are animals.") count.
- HONEST real qwen3:8b numbers (5 easy items): accuracy 1.0 / 1.0 / 1.0; **faithfulness 1.0
  vs 0.0 vs 0.0**; latency +6.8s System overhead; consistency 1.0 / 1.0(LLM) / 0.7(CoT).
  Story = verifiability at a measurable cost, NOT "we beat the LLM on accuracy".
- The user has Ollama models: `qwen3:8b` (chat) and `bge-m3` (embeddings). Use `qwen3:8b`.

## NEXT SESSION PLAN — say "continue" to start (user's explicit priority)
multi-step problems, with a verifiable trace." Framing for any writeup: the contribution is
**traceable, verifiable, repairable reasoning at competitive accuracy** — do NOT claim
"we outperform GPT". Don't chase 100% vs 99%; the novelty is Reasoning → Validation →
Repair → Traceability.

PHASE 1 (highest priority) — Run on GSM8K (real multi-step math):
- Add a GSM8K dataset loader. NOTE offline constraint: support loading from a local
  JSONL file the user provides (and bundle a small sample subset for a runnable default);
  do NOT assume network/HF access. Map each GSM8K item to a DatasetItem (query=problem,
  ground_truth=final numeric answer, domain=mathematical-reasoning).
- Run Plain LLM, Chain-of-Thought, and the Neuro-Symbolic System (real model via Ollama
  qwen3:8b) and compare accuracy, faithfulness, step-hallucination rate, latency, and
  reasoning consistency. Numeric-answer matching (extract the final number).

PHASE 2 — Arithmetic production rules (make validation MEANINGFUL):
- The current Validation Engine does IF/THEN substring matching only — it cannot check
  "7×8=56". Add an arithmetic-checking validator (likely a demo-level ValidationEngine
  extension or a custom rule type) that parses an arithmetic step (from logic_form or
  predicates like {lhs, op, rhs, result}) and ACCEPTS 7×8=56, REJECTS 7×8=54 → triggers
  the repair sub-loop. Have RealModelTranslationLayer prompt the model to emit each math
  step in a checkable form so the validator can verify it.
- This is the key piece that lets the System actually CATCH and REPAIR a wrong
  intermediate step — i.e. demonstrate accuracy improvement over plain CoT, not just
  faithfulness.

PHASE 3 — Multi-hop relational reasoning:
- Add relational items (e.g. "Alice is Bob's sister. Bob is Carol's father. What is Alice
  to Carol?") and relational production rules (kinship inference) so the ACT-R controller's
  declarative-memory + rule selection is exercised on multi-hop chains.

Success criterion: if the System improves accuracy even ~5-10% over Plain LLM / CoT on
GSM8K (or multi-hop) WHILE providing a verifiable proof trace, that's evidence matching the
design goals. Report honestly (small sample sizes, mock vs real clearly labelled).

### FUTURE FEATURE SPECS — confirm scope before building (heavier deps)
- Vector-DB + symbolic memory hierarchy (Declarative_Memory tiering).
- Multi-agent: propose / critique / ACT-R arbitrate.
- Tool execution: ACT-R calls Python / calculator / DB / search for external verification.
- Formal verification: SMT solver / theorem prover (e.g. Z3) for math proofs.


---

## PROGRESS UPDATE — Phases 1 & 2 DONE; real GSM8K run produced

DONE this session (full suite green, ~616 tests; all under demo/+tests/, no src/nsr edits):
- **Phase 2 — arithmetic validation (the meaningful piece):** `demo/arithmetic.py`
  (`safe_eval_arithmetic` AST-based, no eval/exec; `parse_equation`),
  `demo/arithmetic_validation.py` (`ArithmeticValidationEngine`: checks each equation,
  ACCEPTS 7*8=56, REJECTS 7*8=54 → routes to the repair sub-loop). Injectable via
  `build_orchestrator_with_backend(validation=...)`. `MathTranslationLayer` makes the model
  emit `{"logic_form":"<expr> = <result>"}`. Proven catch-and-repair end-to-end offline.
- **Phase 1 — standard benchmark pipeline:** `demo/datasets.py` loaders for OFFICIAL
  formats — `load_gsm8k` (#### answer), `load_multiple_choice` (ARC/CommonsenseQA),
  `load_strategyqa`; dispatcher `load_benchmark(name, path, limit)`. Bundled offline sample
  `demo/data/gsm8k_sample.jsonl` (10 original multi-step problems, clearly NOT official).
  `numeric_answer_match` for math. CLI: `--dataset gsm8k --path <official test.jsonl>
  --limit N --runs K`.
- **Real run produced** (qwen3:8b, bundled sample, 3 items, 1 pass, ~4 min):
  accuracy System/CoT/LLM = 1.0/1.0/1.0; **faithfulness 1.0 vs 0.0 vs 0.0**;
  latency System ~34s vs ~10.5s baselines (overhead ~+24s); consistency unset (1 pass).
  Arithmetic validation was active; on these 3 the model's arithmetic was correct so no
  repair fired. HONEST takeaway: pipeline works end-to-end on real GSM8K-format data with
  genuine equation checking; differentiator is still verifiability + measured latency cost.
  3 easy items is too few/easy to show an accuracy gain from catching errors.

REAL-RUN OPS NOTES (important for next time):
- qwen3:8b "thinking" makes the System SLOW (~34s/problem; the System makes many calls per
  problem). Run real benchmarks as a BACKGROUND process (control_pwsh start) and poll —
  foreground commands hit the 25-min timeout. Use `--runs 1` for speed; `--limit` small.
- Ollama models present: `qwen3:8b` (chat), `bge-m3` (embeddings). Use qwen3:8b.

REMAINING:
1. **Bigger/harder real evidence (highest value):** run the OFFICIAL GSM8K test set via
   `--path <test.jsonl> --limit 30..50` (background process). This is where the System can
   actually CATCH a wrong calculation and beat a baseline that carries the error — the
   accuracy-improvement evidence the user wants. Needs the user to drop the official
   test.jsonl locally (or point --path at it).
2. **Phase 3 — multi-hop relational reasoning:** relational dataset + kinship/relational
   production rules (Alice→Bob→Carol) to exercise declarative memory + rule chaining.
3. (Optional, bigger) Orchestrator extension for true multi-intermediate-step decomposition
   per sub-goal — would need a spec change; current GSM8K multi-step only triggers when the
   problem text contains connectives ("then"/"and").


---

## ABLATION STUDY — DONE (real qwen3:8b)

Built `demo/ablation.py` (4 configs A/B/C/D as SystemUnderTest; NoOpValidationEngine for C),
`generate_ablation_gsm8k`, CLI `--ablation`, tests `tests/test_ablation.py` (10). Full suite
green (626). Run: `python demo/run_benchmark.py --backend ollama --dataset gsm8k --ablation
--model qwen3:8b --limit N --runs 1` (background process; ~5 min for 4 items).

Real result (qwen3:8b, 4 bundled items, 1 pass) — demo/output/benchmark_report_ablation_gsm8k.*:
| Config | Accuracy | Faithfulness | Hallucination | Mean latency |
| A Plain LLM | 1.00 | n/a | n/a | 10.0s |
| B + Constrained decoding | 0.00 | n/a | n/a | 8.0s |
| C + ACT-R (no validation) | 1.00 | n/a | n/a | 32.4s |
| D Full neuro-symbolic | 1.00 | 1.00 | 0.00 | 33.9s |

Story (why each component exists): constrained decoding ALONE hurts (single-shot equation
can't express multi-step) → 0.0; ACT-R's multi-step loop recovers accuracy → 1.0; validation
adds verifiability (faithfulness 1.0/halluc 0.0, only config with real values) at ~+1.5s over
C. KEY latency insight: the ~+22s overhead is the ACT-R multi-step LOOP, not the validation.
Faithfulness/hallucination correctly n/a for A/B/C (no real rejection possible).

CAVEATS (state in any writeup): 4 items = directional only; B's 0.0 reflects its single-shot
design (one equation, take RHS) — label it honestly; on these easy items A/C/D tie at 1.0 so
D doesn't yet show catch-and-repair beating C (qwen3 arithmetic was correct). Needs official
GSM8K --limit 30+ to be solid.

STILL REMAINING:
- Larger run on OFFICIAL GSM8K (--path <test.jsonl> --limit 30..50) — the real evidence.
- Phase 3: multi-hop relational reasoning + relational production rules.


---

## REASONING STATISTICS + 30-problem run — DONE (real qwen3:8b)

Built `demo/reasoning_stats.py` (ReasoningStats, aggregate_trace_stats pure fn,
generate_reasoning_stats), CLI `--stats`, tests `tests/test_reasoning_stats.py`. Expanded
`demo/data/gsm8k_sample.jsonl` to 30 original multi-step problems (arithmetic verified).
Reframed the ablation constrained-decoding note to "insufficient, not bad". Full suite green.
Run: `python demo/run_benchmark.py --backend ollama --dataset gsm8k --stats --model qwen3:8b
--limit 30` (background; ~17 min). Reports: demo/output/reasoning_stats.{html,json}.

Real stats (30 problems, full System): accuracy 29/30 = 96.7%; 97 reasoning steps, ALL 97
accepted first pass; 0 rejected / 0 repaired / 0 repair-failed (repair never fired); rule
utilization 100% well-formed-step; all goal-satisfied.

KEY FINDINGS (state honestly in any writeup):
1. The single failure (store-profit item, gt 380) is the most useful data: the model
   answered "40*13=520" — the COST, not the PROFIT (900-520=380). The arithmetic validator
   ACCEPTED it because 40*13=520 is arithmetically correct. => the arithmetic rule validates
   COMPUTATION, not SEMANTIC/GOAL alignment (did it answer the right question). Clear,
   honest limitation.
2. Zero repairs fired across 97 steps — qwen3:8b's arithmetic was reliable on this set, so
   this sample does NOT yet demonstrate the repair loop catching an error. Need harder/larger
   problems (where the model slips arithmetically) to show repair value.
3. Rule utilization trivially 100% the generic well-formed-step rule (starter rules).

WHAT THIS POINTS TO NEXT (priority order):
1. OFFICIAL GSM8K, larger N (--path test.jsonl --limit 50+) — harder items surface real
   arithmetic slips for the repair loop to catch; gives CIs/significance.
2. SEMANTIC / goal-alignment production rules (not just arithmetic) — to catch the
   "right computation, wrong quantity" failure mode the 30-run exposed.
3. Phase 3: multi-hop relational reasoning + relational rules (makes rule-utilization
   analysis meaningful).


---

## GOAL-ALIGNED SEMANTIC VALIDATION — DONE (built + proven; real-model quantification pending)

Built `demo/goal_alignment.py`: infer_goal_operation(query) [keyword->op: profit/left/
difference->subtract, total/altogether->add, each/per->divide, product/times->multiply;
conservative, returns None if unclear or multi-match], expression_operations(expr) [AST,
safe], GoalAlignmentValidationEngine(goal_text) — composes ArithmeticValidationEngine, never
overturns an arithmetic rejection, and ADDS a "goal-alignment" rejection when the goal's
expected operation is ENTIRELY ABSENT from the step's equation (accepts compound exprs that
contain it). Wired: RealMathSystem validation_mode, CLI `--goal-validation`, ablation 5th
config "arithmetic+goal" (FullPlusGoalConfig). Tests tests/test_goal_alignment.py (17). Full
suite green (649+). NO src/nsr edits, no new deps.

PROVEN (deterministic, offline): the headline test reproduces the real profit failure —
goal "What is the profit?", step "40*13=520" (cost, arithmetically correct) is REJECTED by
goal-alignment, routed to repair, regenerated as "900-520=380", goal-satisfied → 380. This
is the concrete "reasoning about INTENT, not just numbers" demonstration.

REAL-MODEL CHECK (qwen3:8b, the profit item, 1 run each): BOTH arithmetic-only and
arithmetic+goal answered 380 correctly — because on this run the model decomposed properly
(40*13=520 then 900-520=380; final step is a subtraction → already goal-aligned, rule didn't
need to fire). The earlier 30-run failure (stopping at 520) was STOCHASTIC. So on the real
model goal-alignment's benefit is a stochastic improvement whose rate needs a LARGER sample
to quantify — it doesn't change easy/correctly-decomposed items.

ROADMAP (user's, confirmed) — paper story = Baseline -> Arithmetic Validation -> Arithmetic
+ Goal Validation:
1. OFFICIAL GSM8K, N=50-100 (--path <test.jsonl>): the ablation
   `--ablation` now includes config E "arithmetic+goal"; run it on the official set to
   quantify arithmetic vs arithmetic+goal (and finally see repair TRIGGER on real slips).
   NEEDS the user to drop the official test.jsonl locally.
2. (optional) per-sub-goal goal alignment (current v1 is final-answer-targeted; documented).
3. Then Phase 3 multi-hop relational reasoning.

CLI: `python demo/run_benchmark.py --backend ollama --dataset gsm8k --ablation --model
qwen3:8b --limit 50 --path <official test.jsonl>` (5-config ablation incl. arithmetic+goal),
or `... --stats --goal-validation ...` for the reasoning-loop breakdown with goal validation.


---

## LOCKED PLAN (user-confirmed) — PAUSE new architecture; get evidence, then write up

Priority order (do NOT build more architecture until evidence is in):
1. OFFICIAL GSM8K, N=50-100. Measure: Accuracy, Faithfulness, Repair rate, Goal-validation
   trigger rate, Rule utilization. (BLOCKED: needs the user to drop the official test.jsonl
   locally; then run the 5-config ablation + --stats as background jobs.)
2. Write the IEEE paper.
3. Polish GitHub + the demo.
4. ONLY after that: revisit multi-hop (Phase 3) / other extensions.

### PAPER FRAMING (user's suggestion) — present validators as THREE LAYERS OF VERIFICATION
(increasing semantic depth; all already implemented, no new architecture needed):
- Structural Validation — "Is the reasoning well-formed?" — Constrained Decoder + the
  well-formed-step rule (schema/format conformance).
- Arithmetic Validation — "Is the computation correct?" — ArithmeticValidationEngine
  (arithmetic-correctness).
- Goal-Aligned Validation — "Is the computation solving the intended objective?" —
  GoalAlignmentValidationEngine (goal-alignment).
Contribution statement to use: the architecture is a cognitive control layer that makes
reasoning traceable, verifiable, and repairable at competitive accuracy — NOT a claim to
beat the LLM. Ablation columns: Plain LLM | + Structural | + Arithmetic | + Goal-Aligned.

### METRICS THE OFFICIAL-GSM8K RUN MUST EMIT (mostly already collected by --stats/--ablation):
accuracy, faithfulness, step-hallucination rate, repair rate (= repaired+repair_failed /
total steps), repair success rate, GOAL-VALIDATION TRIGGER RATE (how often goal-alignment
rejected — add if not already surfaced), rule utilization, latency + overhead, per-config
in the ablation. (Note: reasoning_stats already emits most; confirm goal-trigger count is
surfaced when validation_mode=goal.)
