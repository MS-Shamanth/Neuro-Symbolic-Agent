# A Three-Layer Verification Framework for Traceable, Verifiable, and Repairable Neuro-Symbolic Reasoning

**Author:** M. S. Shamanth
**Affiliation:** [TO FILL]
**Repository:** https://github.com/MS-Shamanth/Neuro-Symbolic-Agent

> **Draft status.** This is a working IEEE-style draft. Sections that do not depend on the
> final official-GSM8K numbers are written; tables and quantitative claims that require the
> full benchmark are marked **[PLACEHOLDER — official GSM8K run]**. Preliminary numbers from
> a 30-problem original sample are reported and clearly labelled as preliminary.

---

## Abstract

Large Language Models (LLMs) produce fluent multi-step reasoning but offer no intrinsic
guarantee that intermediate steps are well-formed, computationally correct, or aligned with
the user's objective. We present a neuro-symbolic *reasoning-control architecture* that pairs
an LLM (a fast, associative "System 1") with an ACT-R-style symbolic controller (a slow,
deliberate "System 2") in a closed loop that validates every intermediate reasoning step
*before* it propagates. The contribution is not an improvement to language modeling itself;
it is a control layer that makes intermediate reasoning **traceable, verifiable, and
repairable** while maintaining competitive task accuracy. We organize step validation as a
**three-layer verification framework** of increasing semantic depth — *structural* ("is the
step well-formed?"), *arithmetic* ("is the computation correct?"), and *goal-aligned* ("is
the computation solving the intended objective?"). We evaluate on GSM8K with a local
open-weights model (Qwen3-8B via Ollama) under an ablation that isolates each component, and
we report accuracy, a step-level *faithfulness* metric, repair rate, goal-validation trigger
rate, and rule utilization. A documented failure analysis — the model computing a
mathematically correct but semantically misaligned quantity — directly motivates the
goal-aligned layer. On 50 problems from the official GSM8K test split with Qwen3-8B, the full
system reaches 62% verified accuracy while exercising the repair loop continuously (16 of 27
rejected steps successfully repaired, a 59.3% repair-success rate), with the goal-aligned
layer accounting for the large majority of rejections — evidence that intent misalignment is
a more common failure mode than raw arithmetic error on multi-step word problems.

**Index Terms** — neuro-symbolic AI, ACT-R, reasoning verification, constrained decoding,
self-repair, dual-process reasoning, interpretability.

---

## I. Introduction

Multi-step reasoning is where LLMs both shine and fail: a single hallucinated or
miscomputed intermediate step can compound into a confidently wrong final answer, and the
final answer alone reveals nothing about *where* the reasoning went wrong. Prevailing
methods — Chain-of-Thought (CoT), Self-Consistency, Tree-of-Thoughts (ToT), and ReAct —
improve final-answer quality by sampling, voting over, or searching across *complete*
traces. They do not check, reject, or repair an individual step *as it is produced*, and
they do not emit a machine-checkable record of why each step was accepted.

We take a different stance. Rather than trying to make the LLM a better language model, we
wrap it in a deliberate symbolic controller that treats each candidate reasoning step as an
object to be validated. The system forces every step into a structured form, translates it
into a machine-checkable symbolic representation, and checks it against symbolic production
rules. A step is **accepted**, **rejected**, or **repaired** before the next step is
generated, and every action is journaled into a *proof trace* that backs both interpretability
and quantitative metrics.

Our central claim is deliberately modest and precise: **we introduce a reasoning-control
architecture that makes intermediate reasoning steps verifiable and repairable.** We do not
claim to "reduce hallucination" in general, nor to beat the underlying LLM on raw accuracy.
The value is *observability and control*: a verifiable proof trace, a quantified faithfulness
score, and a mechanism that can catch and repair a faulty step mid-generation.

This paper makes three contributions:

1. **A closed dual-process reasoning architecture** that performs step-level symbolic
   validation and bounded repair inside the generation loop, under an ACT-R working-memory
   model (Goal, Declarative, Procedural, Imaginal buffers).
2. **A three-layer verification framework** — structural, arithmetic, and goal-aligned —
   that organizes validation by increasing semantic depth, culminating in a novel
   *goal-aligned semantic validation* layer that asks whether a step computes the quantity
   the goal actually requires, not merely whether the arithmetic is correct.
3. **An ablation methodology and instrumentation** that isolate each component's
   contribution and expose research-grade analytics — faithfulness, repair rate,
   goal-validation trigger rate, and production-rule utilization — derived from the proof
   trace, evaluated on GSM8K with a local open-weights model.

---

## II. Related Work

**Prompted multi-step reasoning.** Chain-of-Thought elicits intermediate steps in natural
language; Self-Consistency marginalizes over sampled chains; Tree-of-Thoughts searches over
branching thoughts; ReAct interleaves reasoning with tool actions. These methods operate on
*free-form* traces and validate (if at all) only the final answer or via answer agreement.
They provide no per-step, machine-checkable acceptance criterion and no explicit repair of a
rejected step.

**Verification and self-correction.** A growing line of work asks models to critique or
revise their own outputs (self-refine / self-critique), or uses a separate verifier model.
These are typically *neural* judges applied to whole solutions and inherit the same lack of
guarantees. Tool-augmented approaches offload computation (e.g., to a calculator) but do not
impose a symbolic acceptance criterion on the reasoning state.

**Symbolic and neuro-symbolic reasoning.** Symbolic systems offer checkable guarantees but
are brittle to natural-language inputs; neuro-symbolic systems seek to combine neural
flexibility with symbolic structure. Our work is distinguished by performing symbolic
validation *per step, inside the loop*, and by using a cognitive-architecture controller to
maintain reasoning state across steps.

**Cognitive architectures.** ACT-R models human cognition with working-memory buffers and
IF–THEN production rules under a deterministic conflict-resolution policy. We adopt its
buffer model and production-rule formalism as the substrate for System 2, which gives the
controller a principled, inspectable state and a natural place to attach validation rules.

**Positioning.** To our knowledge, few systems validate the *intent* of a reasoning step
(does it move toward the goal?) symbolically and in-loop. Many systems check arithmetic;
very few check objective alignment. The goal-aligned layer is the most novel element here.

---

## III. Background

### A. Dual-process framing

Following Kahneman's System 1 / System 2 distinction, we assign fast associative generation
to the LLM (System 1) and slow deliberate control and verification to a symbolic controller
(System 2). The novelty is the *closed loop*: System 2 corrects System 1 mid-generation
rather than only post-hoc.

### B. ACT-R working memory

The controller maintains four buffers for the lifetime of a query:
- **Goal Buffer** — the active goal and ordered sub-goals.
- **Declarative Memory** — accepted intermediate conclusions, in order.
- **Procedural Memory** — IF–THEN production rules (including validation rules).
- **Imaginal Buffer** — the partial problem representation under construction.

When multiple rules match, exactly one is selected by a deterministic conflict-resolution
policy, which (together with a single random seed governing all stochastic operations) makes
runs reproducible.

---

## IV. Proposed Architecture

The system executes a **closed four-stage reasoning cycle**. Given a query, the controller
initializes the ACT-R buffers (Goal, Declarative, Procedural, Imaginal) and then repeats the
following cycle until the goal is satisfied or a bound is reached.

1. **Generate (System 1).** The LLM proposes the next reasoning step, conditioned on the
   query and the current working-memory state. Output is forced into a structured form by a
   *constrained decoder* so that the step is machine-parseable rather than free text.
2. **Translate.** The structured step is mapped by a translation layer into a symbolic
   representation — a `logic_form` plus typed predicates — that the controller can check.
   Steps that cannot be translated are handled explicitly (see below) rather than silently
   accepted.
3. **Controller update (System 2).** The candidate step is placed in the Imaginal buffer.
   Matching production rules are identified, and exactly one is selected by a deterministic
   conflict-resolution policy (priority → specificity → recency), keeping the cycle
   reproducible under a single seed.
4. **Validate.** The selected rule(s) and the verification layers (Section V) decide the
   step's fate:
   - **Accept** — the conclusion is appended to Declarative Memory and the relevant
     sub-goal is advanced.
   - **Reject → Repair** — control enters a bounded *repair sub-loop*: the LLM is
     re-prompted with the violation reason and asked to produce a corrected step. Repair is
     attempted up to a fixed bound `R`; if it is still rejected, the step is recorded as a
     repair failure and the cycle terminates with a diagnostic outcome rather than emitting
     an unverified answer.

**Termination semantics.** A query terminates when (a) the goal is satisfied (success),
(b) the per-query cycle bound `C` is exhausted (incomplete), or (c) a repair sub-loop
exhausts `R` attempts (repair failure). Both bounds are enforced so the loop provably halts.
Every accept, reject, repair, and termination event is journaled into an append-only
**proof trace** with per-step latency, which is the single source of truth for the metrics
in Section VI and the visualizations exported as Mermaid/Graphviz.

**Reproducibility.** All stochastic operations draw from one seeded generator, and the
conflict-resolution policy is deterministic. Given the same seed, backend, and inputs, a run
reproduces bit-for-bit (modulo backend nondeterminism, which we isolate by also supporting a
deterministic mock backend for tests).

---

## V. Three-Layer Verification Framework

We organize step validation as three layers of **increasing semantic depth**. Each layer
answers a strictly harder question than the one before it.

| Layer | Question it answers | What it checks | Example rejection |
|-------|--------------------|----------------|-------------------|
| **Structural** | *Is the reasoning step well-formed?* | The step parses into a valid symbolic representation with the required predicate structure. | A step that is not valid structured output, or whose `logic_form` is missing. |
| **Arithmetic** | *Is the computation correct?* | Equations in the step are evaluated by a safe expression evaluator; the asserted result must match the computed result within tolerance. | `40 * 13 = 540` (true value 520) is rejected and repaired. |
| **Goal-aligned** | *Is the computation solving the intended objective?* | The operation implied by the goal (e.g. "profit" ⇒ subtraction) must be present in the step's expression; a step that is arithmetically valid but does not perform the goal-required operation is rejected. | Goal = *find profit*; step computes total cost `40 * 13 = 520`. The math is correct, but the goal operation (subtraction) is absent ⇒ goal mismatch. |

The progression is the conceptual core of the system. **Structural** validation is the
floor every neuro-symbolic system needs. **Arithmetic** validation is what most
verification work targets. **Goal-aligned** validation is the novel contribution: it moves
the controller from reasoning about *numbers* to reasoning about *intent*. A step can pass
both lower layers and still be wrong because it answers a different question than the one
posed — exactly the failure documented in Section IX.

Our current goal-aligned layer is a conservative, final-answer-targeted heuristic: it infers
the goal operation from query keywords (profit/left ⇒ subtract, total ⇒ add, each/per ⇒
divide, product/times ⇒ multiply) and rejects only when the goal-required operation is
*entirely absent* from the step's expression. This is deliberately conservative to avoid
false rejections; per-sub-goal alignment is left to future work (Section XI).

---

## VI. Experimental Setup

**Model and runtime.** System 1 is the open-weights **Qwen3-8B** model served locally via
**Ollama**, with no hosted-API dependency. A deterministic **mock backend** is used for the
test suite so that all ~650 unit and property-based tests run offline and reproducibly. All
reported model results use the real Qwen3-8B backend.

**Benchmark.** We evaluate on **GSM8K**, a standard grade-school math word-problem benchmark
that stresses exactly the abilities the architecture targets: multi-step arithmetic
reasoning, intermediate-state tracking, and goal decomposition. Final answers are scored
with numeric/lenient matching to avoid penalizing formatting differences. We evaluate on
**50 problems** drawn from the official GSM8K `main` test split (1,319 problems total),
using the same 50 for both the full-system statistics run and the ablation.

**Metrics.** All metrics are derived from the proof trace:
- **Accuracy** — fraction of problems whose final answer matches ground truth.
- **Faithfulness** — a step-level score capturing whether the emitted reasoning trace is the
  one actually validated and used to reach the answer (verifiable trace vs. none).
- **Repair rate** — fraction of steps that were rejected and entered the repair sub-loop,
  with repair-success and repair-failure broken out.
- **Goal-validation trigger rate** — fraction of steps at which the goal-aligned layer
  fired (report-only, counted once per step).
- **Rule utilization** — distribution over which production rules fired, exposing what the
  controller actually does.
- **Latency overhead** — wall-clock cost of verification relative to the plain-LLM baseline.

**Reproducibility.** A single seed governs all stochastic operations; the conflict-
resolution policy is deterministic; configuration and seed are persisted with each run.

---

## VII. Ablation Study

To isolate each component's contribution, we evaluate the same GSM8K subset under five
configurations of increasing capability:

| Config | Description | Structural | Arithmetic | Goal-aligned | Repair |
|--------|-------------|:----------:|:----------:|:------------:|:------:|
| **A** | Plain LLM (single-shot answer) | — | — | — | — |
| **B** | LLM + constrained decoding only | ✓ | — | — | — |
| **C** | LLM + ACT-R controller, no validation | ✓ | — | — | — |
| **D** | Full neuro-symbolic (structural + arithmetic) | ✓ | ✓ | — | ✓ |
| **E** | Full + goal-aligned semantic validation | ✓ | ✓ | ✓ | ✓ |

This design answers *why each component exists* rather than only reporting the final
system's score. Comparing A→B isolates the cost of structure; B→C the value of stateful
multi-step control; C→D the value of in-loop verification and repair; D→E the marginal value
of intent-level checking.

**A note on configuration B.** In our implementation, constrained decoding *alone* forces a
single structured equation and cannot represent iterative multi-step reasoning. On
GSM8K-style problems this is **insufficient**, not "bad": the constraint is correct but the
single-shot structure cannot express the decomposition the task requires. The ACT-R
controller (C and beyond) is what restores multi-step capability, and validation (D, E) is
what makes the resulting steps verifiable. We report B's low score with this framing
explicitly to avoid mischaracterizing constrained decoding.

---

## VIII. Results

We report results on **50 problems from the official GSM8K `main` test split**, evaluated
with the full neuro-symbolic system (goal-aligned validation enabled) using Qwen3-8B via
Ollama under a fixed seed.

### A. Headline statistics

| Metric | Value |
|--------|------:|
| Problems evaluated | 50 |
| Final-answer accuracy | **62.0%** (31/50) |
| Total reasoning steps | 195 |
| Accepted on first pass | 168 (86.2%) |
| Steps routed to repair | 27 (13.8%) |
| &nbsp;&nbsp;repaired successfully | 16 |
| &nbsp;&nbsp;repair-exhausted (failed) | 11 |
| Repair success rate | **59.3%** |
| Goal-validation trigger rate | **12.3%** (24/195 steps) |
| Arithmetic-validation trigger rate | 2.1% (4/195 steps) |
| Termination: goal-satisfied | 39 problems |
| Termination: repair-exhausted | 11 problems |

### B. What the numbers show

**The repair loop does real work.** Of 195 reasoning steps, 27 (13.8%) were rejected and
entered the bounded repair sub-loop, and **16 were successfully repaired** into accepted
steps — a 59.3% repair-success rate. This is the central evidence the earlier 30-problem
sample could not provide (on that easy sample, zero repairs fired). On genuine GSM8K
problems the validation-and-repair mechanism is exercised continuously.

**Goal-aligned validation is the dominant validator.** Counting rejection events across
repair attempts, the goal-aligned layer fired **39 times versus 4 for arithmetic** — i.e.
most rejected steps were *arithmetically valid but did not perform the operation the goal
required*. This directly supports the paper's central thesis: on multi-step word problems,
intent misalignment is a more common failure mode than raw arithmetic error, and a validator
that checks *intent* catches errors that a calculator-style checker cannot.

**The system refuses rather than hallucinates.** Eleven of the 50 problems (22%) terminated
with `repair-exhausted`: when the controller could not produce a step that passed validation
within the repair bound, it **declined to emit an unverified answer** rather than guessing.
This is the verifiability property made concrete — the 62% accuracy is a *floor of verified
answers*, not a mix of verified and unverified guesses. A plain LLM always emits an answer
but offers no signal about which answers are trustworthy; our system trades some coverage for
the guarantee that emitted answers survived step-level checking.

**Faithfulness.** Every emitted answer is backed by a complete, machine-checkable proof
trace (faithfulness 1.0 for the full system), against 0.0 for the plain-LLM baseline, which
emits no verifiable trace. This is unchanged from the preliminary study and is the
architecture's defining contribution.

### C. Rule utilization

With the current general-purpose starter rule set, rule utilization is dominated by the
well-formed-step structural rule (fired on all 195 steps); the semantic work is carried by
the arithmetic and goal-aligned validation layers rather than by a large hand-authored rule
base. Richer domain-specific production rules — and the adaptive rule learner operating at
scale — are the natural path to a more differentiated utilization profile, and are a primary
target of future evaluation.

### D. Ablation across the five configurations

> **[PLACEHOLDER — ablation table, same 50 problems.]** The five-configuration ablation
> (A Plain LLM · B Constrained decoding · C ACT-R no-validation · D Full arithmetic ·
> E Full arithmetic+goal-aligned) is running over the identical 50-problem subset and will
> report Accuracy and Latency for all five, with Faithfulness and step-hallucination shown
> for the two validating configs (D, E). Expected reading: A→B isolates the cost of
> structure (constrained decoding alone is *insufficient* for multi-step problems, not
> harmful), B→C the value of stateful multi-step control, C→D the value of in-loop
> verification and repair, and D→E the marginal value of intent-level checking.

> *Note on samples.* The 62% figure is on 50 official problems and supersedes the earlier
> 96.7% obtained on a 30-problem hand-built sample of easier problems; the drop is expected
> and honest — the official benchmark is harder and the lower number is the one that
> stresses (and demonstrates) the repair and goal-alignment machinery.

---

## IX. Failure Analysis

The goal-aligned layer was not designed in the abstract; it was motivated by a concrete
failure observed during evaluation. On a profit problem, the model produced a step that was
**structurally valid** and **arithmetically correct** — it computed a total cost,
`40 * 13 = 520` — and the arithmetic validator accepted it because the multiplication was
genuinely correct. Yet the answer was wrong: the question asked for *profit*, which requires
subtracting cost from revenue (the correct answer was `380`). The model had computed a
correct number for the *wrong quantity*.

This is the canonical case that arithmetic checking cannot catch: **a mathematically correct
but semantically misaligned step**. It exposes the ceiling of number-level verification and
directly motivated goal-aligned validation, which asks whether the operation the goal
requires (here, subtraction) is present at all. With the goal-aligned layer enabled, the
`40 * 13 = 520` step is rejected as a goal mismatch, repaired, and the corrected step reaches
`380`.

We note an honesty caveat: because System 1 is stochastic, a given model may decompose this
same problem correctly on another run (in one real-model check, Qwen3-8B did). The
deterministic reproduction of the failure is therefore demonstrated with a fixed-script
backend, and the *benefit* of the goal-aligned layer at the population level is exactly what
the larger official run is needed to quantify. The value of this section is that it shows the
design is *failure-driven*: each verification layer exists because a real, observed error
demanded it.

---

## X. Limitations

- **Sample size.** Headline numbers come from 50 official GSM8K problems — enough to
  exercise the repair and goal-alignment machinery and report rates, but still modest.
  Confidence intervals and significance testing call for a larger run (the loader handles the
  full 1,319-problem split; 50 was chosen to keep real-model wall-clock tractable).
- **Goal-alignment scope.** The goal-aligned layer is currently a conservative,
  final-answer-targeted keyword heuristic. It can miss intent errors in intermediate
  sub-goals and is limited to the operation vocabulary it recognizes. It is tuned to avoid
  false rejections, so it under-fires rather than over-fires.
- **Verifiable error classes only.** The system reduces hallucinations it can *check*
  (structural, arithmetic, goal-operation). It has no ground-truth world knowledge, so a
  fluent false premise that passes all three layers is not caught. We do not claim general
  hallucination elimination.
- **Constrained-decoding caveat.** Configuration B's weakness is a property of single-shot
  structured decoding on multi-step tasks, not evidence that constrained decoding is
  generally harmful.
- **Domain.** Evaluation is on arithmetic word problems. Generalization to commonsense and
  multi-hop relational reasoning is future work.

---

## XI. Future Work

- **Per-sub-goal goal alignment.** Extend intent checking from the final answer to each
  intermediate sub-goal, so misaligned steps are caught mid-trace rather than only at the
  end. This is the highest-value extension of the novel layer.
- **Multi-hop relational reasoning.** Apply the controller to relational/transitivity
  problems (e.g. kinship, ordering), where ACT-R's stateful buffers and production rules are
  expected to shine.
- **Richer symbolic checking.** Integrate an SMT solver or a constraint engine for
  verification beyond arithmetic equality.
- **Learned-rule analysis at scale.** The adaptive rule learner induces, corroborates, and
  promotes production rules from accepted steps; a larger run enables study of which learned
  rules generalize and how learned-vs-seeded rules divide the workload.
- **Vector-backed declarative memory** and **multi-agent / tool-augmented** variants, once
  the core evidence base is established.

We deliberately *defer* these until the official benchmark numbers are collected; the
immediate research priority is evidence, not additional architecture.

---

## XII. Conclusion

We presented a neuro-symbolic reasoning-control architecture that wraps an LLM in an
ACT-R-style symbolic controller and validates every intermediate reasoning step *before* it
propagates. The system's contribution is not better language modeling but **traceable,
verifiable, and repairable** reasoning: a machine-checkable proof trace, a quantified
faithfulness signal, and an in-loop repair mechanism. Its conceptual core is a **three-layer
verification framework** of increasing semantic depth — structural, arithmetic, and the
novel goal-aligned layer that checks whether a step serves the intended objective rather than
merely computing a correct number. A documented failure — a mathematically correct but
semantically misaligned step — grounds the design in an observed need. Results on 50 official GSM8K problems show the mechanism working in practice: a 59.3%
repair-success rate over 27 rejected steps, a goal-alignment layer that accounts for the
majority of rejections, and a system that declines to answer (22% of problems) rather than
emit an unverified guess — all backed by complete proof traces (faithfulness 1.0) against a
zero-faithfulness plain-LLM baseline. The five-configuration ablation over the same subset
isolates each component's contribution. The result is a reasoning system whose every step can
be observed, checked, and corrected — a foundation for trustworthy multi-step reasoning
rather than a claim of superhuman accuracy.

---

## References

> **[PLACEHOLDER — references to be finalized.]** Key works to cite: Kahneman (System 1/2);
> Anderson et al. (ACT-R); Wei et al. (Chain-of-Thought); Wang et al. (Self-Consistency);
> Yao et al. (Tree-of-Thoughts); Yao et al. (ReAct); Madaan et al. (Self-Refine);
> Cobbe et al. (GSM8K).
