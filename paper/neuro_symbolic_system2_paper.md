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
goal-aligned layer. **[PLACEHOLDER — official GSM8K headline result]**

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
