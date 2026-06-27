# Implementation Plan: Neuro-Symbolic System-2 Reasoning Architecture

## Overview

This plan implements the dual-process reasoning system in **Python**, using **Hypothesis**
for property-based testing as specified in the design. Work proceeds bottom-up: shared
data models and configuration first, then the symbolic state (ACT-R) and translation
machinery, then neural generation and validation/repair, then the proof trace and
metrics, then the orchestrator that wires every component into the four-stage cycle,
and finally the evaluation harness, dataset loader, baselines, and comparison report.

Each task builds on prior tasks and ends with integration so no code is left orphaned.
Property tests are derived from the universal, quantifiable guarantees in the
requirements (faithfulness math, determinism, lossless serialization, configuration
bounds, monotonic memory growth). Sub-tasks marked with `*` are optional test tasks.

## Tasks

- [x] 1. Set up project structure and core data models
  - Create the package layout (e.g. `src/nsr/`, `tests/`), `pyproject.toml` with
    dependencies (LLM SDK, Hypothesis, pytest), and a test runner configuration
  - Implement the core enums and dataclasses from the design: `ValidationStatus`,
    `TerminationReason`, `Goal`, `SubGoal`, `SymbolicRepresentation`, `ProductionRule`,
    `WorkingMemoryState`, `Domain`, `DatasetItem`
  - Implement the proof/result dataclasses: `RepairAttempt`, `ProofStep`,
    `LatencyRecord`, `ProofTrace`, `ErrorRecord`, `VerifiedOutput`
  - Implement the metrics/config/run dataclasses: `QueryMetrics`, `MethodMetrics`,
    `SystemConfig`, `RunRecord`
  - _Requirements: 1.5, 4.1, 5.1, 8.2, 12.1_

- [x] 2. Implement configuration and reproducibility foundations
  - [x] 2.1 Implement the Config Manager
    - Read max cycle limit, repair attempt limit, retry count, LLM selection, output
      format, conflict-resolution policy, generation timeout, latency budget, and
      repeated-run count from configuration at initialization
    - Apply documented defaults for absent values and record applied defaults
    - Halt initialization with a parameter-identifying error for out-of-range numeric
      values, disallowed enum values, or unparseable values
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [x] 2.2 Write property test for numeric range validation
    - **Property 5: Out-of-range numeric config is always rejected**
    - For any integer outside the documented range (cycle limit 1..10000, repair limit
      0..1000, retry count 0..1000), initialization halts with an error naming the
      parameter; any in-range value initializes successfully
    - **Validates: Requirements 12.3**

  - [x] 2.3 Write property test for enum config validation
    - **Property 6: Disallowed enum values are always rejected**
    - For any LLM selection, output format, or conflict-resolution policy value not in
      the allowed set, initialization halts with a parameter-identifying error
    - **Validates: Requirements 12.4**

  - [x] 2.4 Implement the Reproducibility Manager
    - Build the run record (config, dataset ids, model id, effective seed, applied
      defaults) with all required fields non-empty
    - Apply a supplied seed to all stochastic operations; generate and record a seed
      when none is supplied
    - Persist the run record together with metrics durably; return a persistence error
      record on failure
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

  - [x] 2.5 Write unit tests for seeding and persistence
    - Test seed generation when absent, seed application, and the persistence-failure
      error path
    - _Requirements: 13.2, 13.3, 13.5_

- [x] 3. Implement the ACT-R Controller and working-memory buffers (System 2)
  - [x] 3.1 Implement buffer maintenance and accepted-step integration
    - Maintain Goal_Buffer, Declarative_Memory, Procedural_Memory, and Imaginal_Buffer
      for the lifetime of a query
    - On acceptance, append the conclusion as a distinct Declarative_Memory entry and
      replace the Imaginal_Buffer with a representation reflecting the accepted step
    - Retain all prior accepted conclusions until query termination
    - _Requirements: 4.1, 4.2, 4.4, 4.5_

  - [x] 3.2 Implement sub-goal advancement and deterministic rule selection
    - Advance the Goal_Buffer to the next unmet sub-goal; mark the goal satisfied when
      none remain
    - Select exactly one rule deterministically via the configured conflict-resolution
      policy when multiple match; record `no-rule-matched` and route to repair when none
      match
    - _Requirements: 4.3, 4.6, 4.7, 4.8_

  - [x] 3.3 Write property test for deterministic conflict resolution
    - **Property 3: Same state and seed always select the same rule**
    - For any working-memory state with multiple matching rules, repeated selection
      under the same seed and policy yields the identical rule id
    - **Validates: Requirements 4.6, 13.2**

  - [x] 3.4 Write property test for declarative memory retention
    - **Property 9: Declarative memory grows monotonically and retains all conclusions**
    - For any sequence of accepted steps, every accepted conclusion remains present and
      ordered in Declarative_Memory until termination, and each is a distinct entry
    - **Validates: Requirements 4.2, 4.4**

- [x] 4. Implement the Translation Layer
  - [x] 4.1 Implement forward and backward translation
    - Forward: convert a structured step into a `SymbolicRepresentation` conforming to
      the machine-checkable encoding before the controller updates
    - Backward: convert Goal_Buffer, Imaginal_Buffer, and Declarative_Memory into LLM
      prompt context for the next generation
    - _Requirements: 5.1, 5.2_

  - [x] 4.2 Implement untranslatable and back-translation failure handling
    - On an untranslatable step, flag it, leave buffers unchanged, and route to repair
    - On back-translation failure, flag it, journal it, and return an error record
      naming the Translation_Layer
    - Record every translation outcome and direction in the Proof_Trace
    - _Requirements: 5.3, 5.4, 5.5_

  - [x] 4.3 Write property test for untranslatable buffer invariance
    - **Property 13: Untranslatable steps leave working memory unchanged**
    - For any state and any step that fails translation, the working-memory buffers
      after the attempt equal the buffers before it, and the step is routed to repair
    - **Validates: Requirements 5.3**

- [x] 5. Implement the LLM Component and Constrained Decoder (System 1)
  - [x] 5.1 Implement the LLM Component with pluggable backend
    - Generate exactly one candidate step per request for the active sub-goal, including
      the symbolic-state context from the Translation_Layer
    - Select hosted-API (endpoint and credentials from config, never source) or local
      runtime by configuration
    - Enforce the generation timeout with bounded retries; record failure with reason
      and return an error record naming the LLM after retries are exhausted
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x] 5.2 Implement the Constrained Decoder
    - Restrict LLM output to the configured structured format before return
    - Derive active constraints from the current Goal_Buffer, Declarative_Memory,
      Procedural_Memory, and Imaginal_Buffer contents
    - Mark and journal non-conforming output, regenerate up to the retry count, and
      terminate with `constraint-unsatisfied` on exhaustion
    - _Requirements: 3.1, 3.3, 3.4, 3.5_

  - [x] 5.3 Write unit tests for retry and timeout paths
    - Test the timeout-then-retry behavior, retry exhaustion error record, and the
      `constraint-unsatisfied` termination on repeated non-conforming output
    - _Requirements: 2.5, 2.6, 3.3, 3.4_

- [x] 6. Implement the Validation Engine and Repair Coordinator
  - [x] 6.1 Implement the Validation Engine
    - Evaluate a symbolic step against every applicable production rule and record an
      accepted or rejected outcome before the next step is generated
    - Mark accepted only when all applicable rules are satisfied; on rejection record
      every violated rule
    - _Requirements: 6.1, 6.2, 6.3, 6.7_

  - [x] 6.2 Implement the Repair Coordinator
    - Drive the shared repair sub-loop for rejection, untranslatable, and
      no-rule-matched outcomes: build a repair prompt referencing the offending
      constraints, regenerate, re-translate, and re-validate
    - Increment the repair attempt count up to the configured limit; terminate with
      `repair-exhausted` when the limit is reached without acceptance
    - _Requirements: 6.4, 6.5, 6.6_

  - [x] 6.3 Write property test for the repair attempt bound
    - **Property 8: Repair attempts never exceed the configured limit**
    - For any repair limit N and any sequence of rejections, the recorded repair attempt
      count is at most N, and exhaustion yields a `repair-exhausted` termination
    - **Validates: Requirements 6.4, 6.6**

  - [x] 6.4 Write unit tests for validation outcomes
    - Test all-rules-satisfied acceptance, partial-violation rejection with recorded
      violated rules, and re-validation of a repaired step
    - _Requirements: 6.2, 6.3, 6.5_

- [x] 7. Checkpoint - core reasoning components
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement the Proof Trace and exporters
  - [x] 8.1 Implement the append-only Proof Trace and latency recording
    - Record every step in execution order with sequence position, outcome, and applied
      rule id or explicit `no-rule-applied`; record per-attempt repair details
    - Record pipeline, System-2 (Validation + ACT-R), and LLM latencies, and a
      latency-budget-exceeded flag when the configured budget is exceeded
    - _Requirements: 8.1, 8.2, 8.3, 11.1, 11.2, 11.4_

  - [x] 8.2 Implement machine-readable and human-readable exporters
    - Export a lossless machine-readable form whose every field parses back without loss
    - Produce a human-readable rendering presenting each step, outcome, and applied rule
      in execution order
    - _Requirements: 8.4, 8.5_

  - [x] 8.3 Write property test for lossless trace round-trip
    - **Property 4: Proof_Trace machine-readable export round-trips losslessly**
    - For any generated `ProofTrace`, exporting then re-parsing yields a structure equal
      to the original in every recorded field
    - **Validates: Requirements 8.4**

  - [x] 8.4 Write unit tests for trace rendering and latency flag
    - Test `no-rule-applied` rendering, repair-attempt ordering, and the
      latency-budget-exceeded indication
    - _Requirements: 8.2, 8.3, 11.4_

- [x] 9. Implement the Metrics Engine
  - [x] 9.1 Implement per-query and consistency metrics
    - Compute Faithfulness_Score as accepted/total (0.0 for an empty trace) and
      Step_Level_Hallucination_Rate as rejected/total, each in [0.0, 1.0]
    - Compute Reasoning_Consistency as the fraction of runs matching the modal answer
      only when the repeated-run count is 2 or greater; otherwise leave it unset
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 9.2 Write property test for faithfulness and hallucination bounds
    - **Property 1: Faithfulness_Score equals accepted/total and lies in [0,1]**
    - **Property 2: Step_Level_Hallucination_Rate equals rejected/total and lies in [0,1]**
    - For any trace, both metrics fall in [0.0, 1.0]; an empty trace yields a
      Faithfulness_Score of exactly 0.0
    - **Validates: Requirements 7.1, 7.2, 7.3**

  - [x] 9.3 Write property test for reasoning consistency
    - **Property 10: Reasoning_Consistency is the modal-answer fraction in [0,1]**
    - For any multiset of run answers with repeated-run count >= 2, consistency equals
      the modal-answer fraction in [0.0, 1.0]; counts < 2 leave it unset
    - **Validates: Requirements 7.4, 7.5**

- [x] 10. Implement the Pipeline Orchestrator and wire the reasoning cycle
  - [x] 10.1 Implement query intake and cycle execution
    - Validate the query and initialize the Goal_Buffer before any step; reject empty or
      unparseable queries with an error record before starting the cycle
    - Run cycles in the fixed four-stage order (generate, translate, controller update,
      validate) bounded by the max cycle limit
    - _Requirements: 1.1, 1.2, 1.7_

  - [x] 10.2 Implement termination, output emission, and error handling
    - Terminate and emit a Verified_Output on goal satisfaction; terminate with
      `cycle-limit-reached` at the cycle bound; surface `constraint-unsatisfied` and
      `repair-exhausted` from sub-components
    - Convert component failures (LLM unavailable, back-translation failure) into error
      records while preserving the Proof_Trace; attach the Faithfulness_Score to output
    - Journal each step, outcome, and applied rule to the Proof_Trace
    - _Requirements: 1.3, 1.4, 1.5, 1.6, 7.6_

  - [x] 10.3 Write property test for the cycle bound
    - **Property 7: Completed cycles never exceed the maximum cycle limit**
    - For any max cycle limit M and any query that does not satisfy its goal, the number
      of completed cycles is at most M and termination is `cycle-limit-reached`
    - **Validates: Requirements 1.2, 1.4**

  - [x] 10.4 Write integration tests for end-to-end query flows
    - Test a goal-satisfied run emitting a Verified_Output with attached score, an
      empty-query rejection, and an LLM-unavailable error path preserving the trace
    - _Requirements: 1.1, 1.3, 1.6, 1.7, 7.6_

- [x] 11. Checkpoint - full single-query pipeline
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Implement the Dataset Loader and step-level benchmark
  - [x] 12.1 Implement dataset loading and validation
    - Validate that each item has a non-empty unique id, non-empty query, non-empty
      ground truth, and a recognized domain label before evaluation begins
    - Exclude items missing a field or carrying an unrecognized domain label, retain the
      remaining valid items, and log excluded id with the missing field or bad label
    - Record total items loaded, validated, and excluded per domain; associate each item
      with exactly one of the six benchmark domains
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [x] 12.2 Write property test for dataset validation partitioning
    - **Property 11: Loading partitions items into retained-valid and excluded-invalid**
    - For any mix of valid and invalid items, every valid item is retained and every
      invalid item is excluded and logged; the two sets are disjoint and cover the input
    - **Validates: Requirements 10.2, 10.3, 10.5**

- [x] 13. Implement baseline methods
  - [x] 13.1 Implement the baseline reasoning methods
    - Implement Chain-of-Thought, Self-Consistency, Tree-of-Thoughts, ReAct, and the
      LLM-only baseline behind a common method interface that returns a final answer and
      latency for a query
    - _Requirements: 9.3_

  - [x] 13.2 Write unit tests for baseline interface conformance
    - Test that each baseline returns a final answer and latency and conforms to the
      shared method interface
    - _Requirements: 9.3_

- [x] 14. Implement the Evaluation Harness and comparison report
  - [x] 14.1 Implement evaluation execution and per-method metrics
    - Execute the System and each configured baseline over every dataset item; record
      the run record before the first item (Reproducibility Manager)
    - Compute per-method final-answer accuracy, Step_Level_Hallucination_Rate,
      Faithfulness_Score, mean and p95 pipeline latency, and latency overhead
    - Exclude failing items per method, log the failed method and item id, and continue
    - _Requirements: 9.1, 9.2, 9.7, 11.3, 13.1_

  - [x] 14.2 Implement latency overhead, consistency, and the comparison report
    - Compute latency overhead as the mean per-query difference versus the
      LLM-only baseline over the same query set
    - Compute per-method Reasoning_Consistency across repeated runs when configured
    - Produce a comparison report listing each metric's System value, each baseline
      value, and the numeric difference; persist the report with the run record
    - _Requirements: 9.4, 9.5, 9.6, 13.4_

  - [x] 14.3 Write property test for latency overhead computation
    - **Property 12: Latency overhead is the mean per-query latency difference**
    - For any per-query latency sets, the reported overhead equals the mean of
      (method latency - LLM-only latency) over the shared query set
    - **Validates: Requirements 9.5**

  - [x] 14.4 Write integration test for an end-to-end evaluation run
    - Run a small multi-domain dataset through the harness and assert the report
      contains all methods, all metrics with differences, and a persisted run record
    - _Requirements: 9.1, 9.2, 9.4, 13.4_

- [x] 15. Final checkpoint - full system and evaluation
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. Implement adaptive-rule-learning data models and configuration
  - [x] 16.1 Add adaptive-rule-learning data models and backward-compatible extensions
    - Create `src/nsr/models/learning.py` with `RuleOrigin`, `RuleProvenance`,
      `CandidateRule`, `LearnedRule`, `DiscardedCandidate`, `PromotionDecision`,
      `PromotionResult`, and the versioned `LearnedRuleStore`, reusing the existing
      `ProductionRule` IF/THEN string form so learned rules stay evaluable by the
      unchanged `ACTRController`/`ValidationEngine`
    - Add the optional, default-valued `applied_rule_origin: Optional[RuleOrigin] = None`
      field to `ProofStep` (`models/trace.py`) so existing traces still construct and
      round-trip unchanged
    - Add `rule_learning_enabled` (default `False`), `corroboration_threshold` (default
      `2`), and `max_learned_rules` (default `64`) to `SystemConfig`, and add
      `learned_rules`, `induction_seed`, `corroboration_threshold`, and
      `promotion_decisions` (all default-empty/optional) to `RunRecord` in
      `models/config.py`
    - _Requirements: 14.2, 14.5, 14.6, 14.7, 14.8_

  - [x] 16.2 Extend the Config Manager with rule-learning parameters
    - Add `rule_learning_enabled` (parsed as bool, default `False`),
      `corroboration_threshold` (`CORROBORATION_THRESHOLD_RANGE = (1, 1000)`, default `2`),
      and `max_learned_rules` (`MAX_LEARNED_RULES_RANGE = (1, 100000)`, default `64`) to
      `DEFAULTS` in `config_manager.py`, resolving, range-checking, and recording each
      applied default in `applied_defaults` exactly like the existing numeric parameters
    - Halt initialization with a parameter-identifying `ConfigError` for out-of-range
      `corroboration_threshold` or `max_learned_rules`
    - _Requirements: 14.8_

  - [x] 16.3 Write property test for rule-learning config defaults and ranges
    - **Property 8: Config applies documented rule-learning defaults**
    - For any config mapping omitting the three keys, the documented defaults
      (`false`, `2`, `64`) are applied and recorded in `applied_defaults`; any
      out-of-range `corroboration_threshold` or `max_learned_rules` halts with a
      parameter-identifying error
    - **Validates: Requirements 14.8**

- [x] 17. Implement the Rule Learner and wire adaptive rule learning
  - [x] 17.1 Implement candidate-rule induction
    - Create `src/nsr/rule_learner.py` with `RuleLearner.induce(trace, *, trace_id)` that
      generalizes the accepted (or accepted-after-repair) `ProofStep`s of a
      goal-satisfied trace into `CandidateRule`s using the same term decomposition the
      controller/validator use, recording `RuleProvenance` (trace id + accepted step ids)
    - Return `[]` for a trace with no accepted steps or a non-`goal-satisfied`
      termination reason
    - _Requirements: 14.1, 14.2_

  - [x] 17.2 Write property test for induction over accepted steps only
    - **Property 1: Induction generalizes only accepted steps**
    - **Validates: Requirements 14.1**

  - [x] 17.3 Write property test for candidate provenance traceability
    - **Property 2: Candidates are provenance-traceable**
    - **Validates: Requirements 14.2**

  - [x] 17.4 Implement corroboration, promotion, contradiction check, cap, and persistence
    - Implement `corroborate` (merge candidates into the `LearnedRuleStore`, incrementing
      corroboration counts at most once per distinct source trace id) and `promote`
      (promote candidates with count `>= corroboration_threshold` that do not contradict
      any existing rule, up to `max_learned_rules`)
    - Implement `contradicts` reusing `ValidationEngine.validate` semantics over the
      witness set (provenance + corroborating accepted steps); discard contradicting
      candidates and append a `DiscardedCandidate` with the conflicting `rule_id`; log a
      `cap-reached` decision when the cap is hit
    - Register a seed hook via `ReproducibilityManager.register_seed_hook` and process
      candidates/promotions in canonical order (normalized IF/THEN key, then provenance
      trace id) for determinism; record `learned_rules`, `induction_seed`,
      `corroboration_threshold`, and `promotion_decisions` in the `RunRecord`; add
      `store_to_dict`/`store_from_dict` and persist the versioned `LearnedRuleStore`
      durably via the `ReproducibilityManager`
    - _Requirements: 14.3, 14.4, 14.6, 14.7, 14.9_

  - [x] 17.5 Write property test for promotion gating
    - **Property 3: Promotion requires corroboration and no contradiction**
    - **Validates: Requirements 14.3**

  - [x] 17.6 Write property test for contradiction discard-and-log
    - **Property 4: Contradicting candidates are discarded and logged**
    - **Validates: Requirements 14.4**

  - [x] 17.7 Write property test for deterministic rule learning
    - **Property 6: Rule learning is deterministic under a fixed seed**
    - **Validates: Requirements 14.6**

  - [x] 17.8 Write property test for learned-rule-store round-trip
    - **Property 7: Learned rule store serialization round-trips losslessly**
    - **Validates: Requirements 14.7**

  - [x] 17.9 Write property test for the learned-rule cap
    - **Property 9: Promoted learned rules never exceed the cap**
    - **Validates: Requirements 14.9**

  - [x] 17.10 Extend the proof-trace exporter for the learned-vs-seeded marker
    - Add an `"applied_rule_origin"` key to `_step_to_dict`/`_step_from_dict` in
      `proof_trace_export.py` (serialized as the enum `value`, omitted/`None`-tolerant on
      read) so the marker survives the machine-readable round-trip while older artifacts
      without the key still parse
    - _Requirements: 14.5_

  - [x] 17.11 Write property test for learned-vs-seeded marker round-trip
    - **Property 5: Learned-vs-seeded marker is preserved across trace round-trip**
    - **Validates: Requirements 14.5**

  - [x] 17.12 Wire the Rule Learner into the orchestrator/evaluation harness
    - On the `goal-satisfied` termination path only, gated by `rule_learning_enabled`,
      invoke `induce -> corroborate -> promote` after `_emit_verified_output` and extend
      the controller's Procedural_Memory with the promoted `Learned_Rules` for subsequent
      queries in the run; make the block best-effort (caught/logged, never corrupting the
      emitted result)
    - Ensure the disabled path skips the entire block so Procedural_Memory holds only
      `Seeded_Rules` and behavior is identical to Requirements 1-13
    - _Requirements: 14.1, 14.10_

  - [x] 17.13 Write property test for disabled-path equivalence
    - **Property 10: Disabled rule learning is behaviorally identical to Req 1-13**
    - Model-based equivalence test asserting equal final answer, step sequence, outcomes,
      applied rule ids, and termination reason versus the no-rule-learning baseline
    - **Validates: Requirements 14.10**

  - [x] 17.14 Write unit/integration tests for durable learned-rule-store persistence
    - Assert the versioned store writes to disk and reloads equal, and that a write
      failure yields an `ErrorRecord` naming the failed persistence operation
    - _Requirements: 14.7_

- [x] 18. Implement reasoning visualization (Trace Visualizer)
  - [x] 18.1 Implement the Mermaid and Graphviz DOT exporters
    - Create `src/nsr/trace_visualizer.py` alongside `proof_trace_export.py` with pure
      `to_mermaid(trace) -> str` and `to_dot(trace) -> str` functions that never mutate
      the trace, rendering `Goal Buffer -> step nodes -> validation outcome -> repair
      branches -> terminal node` from the append-only `ProofTrace`
    - Style step nodes by `ValidationStatus` (accepted/rejected/repaired visually
      distinct), annotate each node with its applied rule id or the explicit
      `no-rule-applied` indicator (reusing `applied_rule_label`/`NO_RULE_APPLIED`) plus a
      learned/seeded marker from `ProofStep.applied_rule_origin`, emit one branch node per
      `RepairAttempt` ordered by `attempt_index`, render the terminal as the
      `Verified_Output` or `termination_reason`, and render a well-formed minimal
      placeholder for an empty trace
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_

  - [x] 18.2 Write property test for visualization structural completeness
    - **Property 11: Visualization is structurally complete**
    - **Validates: Requirements 15.1**

  - [x] 18.3 Write property test for lossless pure-function visualization
    - **Property 12: Visualization is a lossless pure function of the trace**
    - **Validates: Requirements 15.2**

  - [x] 18.4 Write property test for outcome-distinguished, rule-annotated nodes
    - **Property 13: Nodes are outcome-distinguished and rule-annotated**
    - **Validates: Requirements 15.4**

  - [x] 18.5 Write example/edge unit tests for visualization formats
    - Assert `to_mermaid` output begins with a `flowchart` header and `to_dot` output is a
      `digraph { ... }` on a representative trace (15.3); assert both exporters return a
      well-formed minimal diagram for an empty `ProofTrace` without raising (15.5)
    - _Requirements: 15.3, 15.5_

- [x] 19. Checkpoint - adaptive rule learning and visualization
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks and can be skipped for a faster MVP;
  core implementation tasks are never optional.
- Each task references specific requirement sub-clauses for traceability.
- Property tests use Hypothesis and validate the universal correctness guarantees
  derived from the requirements; unit and integration tests cover specific examples,
  error paths, and end-to-end flows.
- Checkpoints provide incremental validation at natural integration boundaries.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "2.4", "3.1", "4.1", "5.1", "8.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.5", "3.2", "4.2", "5.2", "6.1", "8.2", "9.1", "12.1", "13.1"] },
    { "id": 3, "tasks": ["3.3", "3.4", "4.3", "5.3", "6.2", "8.3", "8.4", "9.2", "9.3", "12.2", "13.2"] },
    { "id": 4, "tasks": ["6.3", "6.4", "10.1"] },
    { "id": 5, "tasks": ["10.2", "14.1"] },
    { "id": 6, "tasks": ["10.3", "10.4", "14.2"] },
    { "id": 7, "tasks": ["14.3", "14.4"] },
    { "id": 8, "tasks": ["16.1"] },
    { "id": 9, "tasks": ["16.2", "17.1", "17.10", "18.1"] },
    { "id": 10, "tasks": ["16.3", "17.2", "17.3", "17.4", "17.11", "18.2", "18.3", "18.4", "18.5"] },
    { "id": 11, "tasks": ["17.5", "17.6", "17.7", "17.8", "17.9", "17.12"] },
    { "id": 12, "tasks": ["17.13", "17.14"] }
  ]
}
```
