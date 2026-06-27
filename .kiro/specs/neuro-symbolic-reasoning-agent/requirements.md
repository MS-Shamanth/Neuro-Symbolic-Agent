# Requirements Document

## Introduction

This document specifies the requirements for the **Neuro-Symbolic System-2 Reasoning Architecture**, a hybrid artificial intelligence system that reduces hallucinations in multi-step reasoning tasks. The system integrates a neural Large Language Model (operating as Kahneman's "System 1", fast and associative) with the ACT-R symbolic cognitive architecture (operating as "System 2", slow and deliberate) inside a unified inference pipeline.

The central novelty is **real-time, step-level symbolic validation**: each intermediate reasoning step produced by the LLM is translated into a symbolic representation, checked against symbolic production rules and constraints, and accepted, rejected, or repaired *before* it propagates to the next step. This forms a closed dual-process feedback loop in which the deliberate symbolic controller corrects the fast neural generator mid-generation rather than only checking the final answer. The system tracks reasoning state through ACT-R-style working-memory buffers, enforces structured outputs through constrained decoding, quantifies reasoning quality through a novel faithfulness score, and is evaluated against a reusable step-level hallucination benchmark spanning multiple domains.

This specification covers the hybrid pipeline, each architectural component (LLM, ACT-R Controller, Translation Layer, Validation Engine), the working-memory buffers, constrained decoding, step-level validation and repair, the faithfulness metric, the benchmarking and evaluation harness, the evaluation datasets, success criteria, and non-functional requirements for latency, interpretability, and reproducibility.

### Scope

In scope: multi-step mathematical reasoning, logical puzzle solving, abstract reasoning, structured problem-solving, and AI alignment / reasoning-validation research.

Out of scope: training a Large Language Model from scratch, and real-time conversational agent behavior.

## Glossary

- **System**: The complete Neuro-Symbolic System-2 Reasoning Architecture described in this document.
- **Pipeline**: The end-to-end inference flow that processes a user query into a verified output through iterative reasoning steps.
- **LLM**: The Large Language Model component acting as System 1, which generates candidate reasoning steps. The model may be accessed through a hosted API or run locally.
- **System_1**: The fast, associative generation role fulfilled by the LLM.
- **System_2**: The slow, deliberate validation and control role fulfilled by the ACT-R Controller and Validation Engine.
- **ACT-R Controller**: The symbolic cognitive controller that maintains working-memory buffers, selects production rules, and manages reasoning state across steps.
- **Goal_Buffer**: An ACT-R working-memory buffer that holds the current active goal and active sub-goal.
- **Declarative_Memory**: An ACT-R memory store that holds facts and previously established intermediate conclusions.
- **Procedural_Memory**: An ACT-R memory store that holds IF-THEN production rules that govern reasoning operations.
- **Imaginal_Buffer**: An ACT-R working-memory buffer that holds the partial problem representation under active construction.
- **Translation_Layer**: The component that maps an LLM-generated reasoning step into a symbolic representation, and maps symbolic state back into an LLM prompt context.
- **Symbolic_Representation**: A structured, machine-checkable encoding of a reasoning step (for example, a logical form or structured record).
- **Validation_Engine**: The component that evaluates a candidate reasoning step against symbolic production rules and constraints and returns an accept, reject, or repair outcome.
- **Constrained_Decoding**: A generation technique that restricts LLM output to a predefined structured format (for example, JSON or a defined logic format).
- **Reasoning_Step**: One discrete intermediate inference produced during the reasoning process.
- **Repair**: A corrective action in which a rejected reasoning step is regenerated or adjusted to satisfy symbolic constraints.
- **Faithfulness_Score**: A metric defined as the fraction of reasoning steps in a solution trace that pass symbolic validation.
- **Step_Level_Hallucination_Rate**: The fraction of reasoning steps that are factually or logically invalid as determined by validation or ground-truth annotation.
- **Proof_Trace**: The ordered record of reasoning steps, validation outcomes, and applied production rules that justifies the final output.
- **Verified_Output**: The final answer accompanied by its Proof_Trace.
- **Evaluation_Harness**: The component that runs the System and baseline methods over datasets and computes metrics.
- **Baseline_Method**: An existing reasoning method used for comparison, specifically Chain-of-Thought, Self-Consistency, Tree-of-Thoughts, and ReAct.
- **Benchmark**: The reusable step-level hallucination evaluation suite spanning multiple domains.
- **Reasoning_Consistency**: The degree to which repeated runs on the same query produce mutually agreeing final answers.
- **LLM_Only_Baseline**: The LLM generating a final answer directly without symbolic validation, used as the reference point for latency overhead.
- **Learned_Rule**: A production rule that is induced by the ACT-R Controller from successful reasoning traces and added to Procedural_Memory, as distinct from a Seeded_Rule that is supplied at initialization.
- **Seeded_Rule**: A production rule that is supplied to Procedural_Memory at initialization rather than induced from reasoning traces.
- **Candidate_Rule**: A provisional production rule generalized from accepted Reasoning_Steps that has not yet been promoted into active Procedural_Memory.
- **Rule_Provenance**: The record of the origin of a Learned_Rule, including the identifiers of the traces and the Reasoning_Steps from which the rule was induced, and whether the rule is learned or seeded.
- **Corroboration_Threshold**: The configurable minimum number of independent successful traces in which a Candidate_Rule must appear before it is promoted into active Procedural_Memory.
- **Rule_Learning**: The process by which the ACT-R Controller induces Candidate_Rules from successful reasoning traces and promotes corroborated, non-conflicting Candidate_Rules into Procedural_Memory.
- **Learned_Rule_Store**: The versioned, persisted store that holds Learned_Rules together with their Rule_Provenance across runs.
- **Reasoning_Visualization**: A machine-renderable diagram, emitted as text, that depicts the reasoning flow recorded in a Proof_Trace from the Goal_Buffer through each Reasoning_Step, its validation outcome, any Repair, and the final answer or termination reason.

## Requirements

### Requirement 1: Hybrid Dual-Process Inference Pipeline

**User Story:** As a researcher, I want a unified pipeline that combines the LLM (System 1) and the ACT-R Controller (System 2), so that fast neural generation and deliberate symbolic reasoning operate together on a single query.

#### Acceptance Criteria

1. WHEN a non-empty user query is submitted, THE Pipeline SHALL initialize the Goal_Buffer with the query goal before generating any Reasoning_Step.
2. THE Pipeline SHALL process a query through repeated cycles bounded by the configured maximum cycle limit, where each cycle consists, in order, of LLM generation, translation, ACT-R Controller update, and validation.
3. WHEN the active goal in the Goal_Buffer is satisfied, THE Pipeline SHALL terminate the reasoning cycle and emit a Verified_Output.
4. WHEN the number of completed reasoning cycles reaches the configured maximum cycle limit, THE Pipeline SHALL terminate the reasoning cycle and emit the current Proof_Trace with a termination reason of cycle-limit-reached.
5. THE Pipeline SHALL record each Reasoning_Step, its validation outcome, and the applied production rule into the Proof_Trace.
6. IF the LLM component is unavailable when generation is requested, THEN THE Pipeline SHALL halt the current query, preserve the existing Proof_Trace contents, and return an error record identifying the failed component.
7. IF a submitted user query is empty or cannot be parsed into a query goal, THEN THE Pipeline SHALL reject the query without initializing the reasoning cycle and return an error record identifying the invalid query.

### Requirement 2: Large Language Model Component (System 1)

**User Story:** As a developer, I want the LLM to generate one candidate reasoning step at a time, so that each step can be validated before the next step is produced.

#### Acceptance Criteria

1. WHEN the Pipeline requests a reasoning step, THE LLM SHALL generate exactly one candidate Reasoning_Step for the active sub-goal currently held in the Goal_Buffer.
2. WHEN the Pipeline requests a reasoning step, THE LLM SHALL include the current symbolic state supplied by the Translation_Layer in its generation context.
3. WHERE the LLM is configured to use a hosted API, THE System SHALL read the model endpoint and credentials from configuration rather than from source code.
4. WHERE the LLM is configured to run locally, THE System SHALL load the model through the configured local runtime.
5. IF an LLM generation request does not return a response within the configured generation timeout, THEN THE System SHALL treat the attempt as a failed generation request and SHALL retry it up to the configured retry count.
6. IF an LLM generation request fails after the configured retry count, THEN THE System SHALL record the failure with its failure reason in the Proof_Trace, SHALL preserve the existing Proof_Trace contents, and SHALL return an error record identifying the LLM component.

### Requirement 3: Constrained Decoding

**User Story:** As a developer, I want each LLM output constrained to a defined structured format, so that every reasoning step is machine-parseable and logically well-formed.

#### Acceptance Criteria

1. WHEN the Pipeline requests a Reasoning_Step, THE System SHALL constrain the LLM output to the configured structured format before the output is returned.
2. WHEN a generated Reasoning_Step conforms to the configured structured format, THE Translation_Layer SHALL parse it into a Symbolic_Representation before the next Reasoning_Step is requested.
3. IF a generated Reasoning_Step does not conform to the configured structured format, THEN THE System SHALL mark the step as non-conforming, record the attempt in the Proof_Trace, and request regeneration up to the configured retry count.
4. IF the configured retry count is exhausted without a conforming Reasoning_Step, THEN THE System SHALL terminate the query and emit the Proof_Trace with a termination reason of constraint-unsatisfied.
5. THE System SHALL derive the active decoding constraints from the current contents of the Goal_Buffer, Declarative_Memory, Procedural_Memory, and Imaginal_Buffer.

### Requirement 4: ACT-R Controller and Working-Memory Buffers (System 2)

**User Story:** As a researcher, I want ACT-R-style working-memory buffers that persist reasoning state, so that sub-goals and context are tracked across all reasoning steps.

#### Acceptance Criteria

1. WHILE a query is being processed, THE ACT-R Controller SHALL maintain a Goal_Buffer, a Declarative_Memory store, a Procedural_Memory store, and an Imaginal_Buffer.
2. WHEN a Reasoning_Step is accepted, THE ACT-R Controller SHALL store the resulting intermediate conclusion as a distinct entry in Declarative_Memory before the next Reasoning_Step is generated.
3. WHEN a sub-goal is satisfied and at least one unmet sub-goal remains, THE ACT-R Controller SHALL update the Goal_Buffer to the next unmet sub-goal.
4. WHILE a query is being processed, THE ACT-R Controller SHALL retain all previously accepted intermediate conclusions in Declarative_Memory until the query terminates.
5. WHEN the ACT-R Controller processes an accepted Reasoning_Step, THE ACT-R Controller SHALL replace the contents of the Imaginal_Buffer with a partial problem representation that reflects that accepted Reasoning_Step.
6. WHEN multiple production rules in Procedural_Memory match the current state, THE ACT-R Controller SHALL select exactly one rule deterministically using the configured conflict-resolution policy.
7. WHEN a sub-goal is satisfied and no unmet sub-goal remains, THE ACT-R Controller SHALL mark the active goal in the Goal_Buffer as satisfied.
8. IF no production rule in Procedural_Memory matches the current state, THEN THE ACT-R Controller SHALL record a no-rule-matched outcome in the Proof_Trace and route the current state to the repair process.

### Requirement 5: Translation Layer

**User Story:** As a researcher, I want a bidirectional translation layer between neural and symbolic representations, so that the LLM and the ACT-R Controller can exchange reasoning state.

#### Acceptance Criteria

1. WHEN the LLM produces a structured Reasoning_Step, THE Translation_Layer SHALL convert it into a Symbolic_Representation that conforms to the machine-checkable encoding before the ACT-R Controller is updated.
2. WHEN the ACT-R Controller updates its working-memory buffers, THE Translation_Layer SHALL convert the active goal from the Goal_Buffer, the partial problem representation from the Imaginal_Buffer, and the accepted intermediate conclusions from Declarative_Memory into context supplied to the LLM for the next generation.
3. IF a Reasoning_Step cannot be converted into a Symbolic_Representation, THEN THE Translation_Layer SHALL flag the step as untranslatable, leave the working-memory buffers unchanged, and route the step to the repair process.
4. THE Translation_Layer SHALL record each translation outcome, including the direction of translation and any untranslatable flag, in the Proof_Trace.
5. IF the symbolic state cannot be converted into LLM context, THEN THE Translation_Layer SHALL flag the back-translation as failed, record the failure in the Proof_Trace, and return an error record identifying the Translation_Layer.

### Requirement 6: Step-Level Validation and Repair

**User Story:** As a researcher, I want each intermediate reasoning step validated and repaired before it propagates, so that invalid reasoning is corrected mid-generation rather than after the final answer.

#### Acceptance Criteria

1. WHEN a Symbolic_Representation of a Reasoning_Step is produced, THE Validation_Engine SHALL evaluate it against every applicable production rule in Procedural_Memory and record an accepted or rejected outcome before the next Reasoning_Step is generated.
2. WHEN a Reasoning_Step satisfies every applicable production rule, THE Validation_Engine SHALL mark the step as accepted.
3. IF a Reasoning_Step violates one or more applicable production rules, THEN THE Validation_Engine SHALL mark the step as rejected and record each violated rule.
4. WHEN a Reasoning_Step is marked as rejected, THE System SHALL initiate a Repair by requesting a regenerated step constrained by the violated rules, incrementing the repair attempt count, up to the configured repair attempt limit.
5. WHEN a repaired Reasoning_Step is produced, THE Validation_Engine SHALL re-validate it against every applicable production rule before it propagates to the next Reasoning_Step.
6. IF the configured repair attempt limit is reached without an accepted step, THEN THE System SHALL terminate the query and emit the Proof_Trace with a termination reason of repair-exhausted.
7. THE Validation_Engine SHALL record the validation outcome of every Reasoning_Step and every repair attempt in the Proof_Trace.

### Requirement 7: Faithfulness Score and Reasoning Metrics

**User Story:** As a researcher, I want a quantitative faithfulness score and related metrics, so that reasoning quality can be measured and compared.

#### Acceptance Criteria

1. WHEN a query terminates with a non-empty Proof_Trace, THE System SHALL compute the Faithfulness_Score as the number of Reasoning_Steps marked accepted divided by the total number of Reasoning_Steps, as a value between 0.0 and 1.0 inclusive.
2. IF a query terminates with an empty Proof_Trace, THEN THE System SHALL set the Faithfulness_Score to 0.0.
3. WHEN a query terminates, THE System SHALL compute the Step_Level_Hallucination_Rate as the number of Reasoning_Steps marked rejected divided by the total number of Reasoning_Steps, as a value between 0.0 and 1.0 inclusive.
4. WHEN the Evaluation_Harness runs a query with a configured repeated-run count of 2 or greater, THE System SHALL compute Reasoning_Consistency as the fraction of runs whose final answer matches the modal final answer.
5. IF the configured repeated-run count is less than 2, THEN THE System SHALL not compute Reasoning_Consistency for that query.
6. WHEN the System emits a Verified_Output, THE System SHALL attach the computed Faithfulness_Score to it.

### Requirement 8: Proof Trace and Interpretability

**User Story:** As a researcher, I want a human-readable proof trace, so that I can inspect and explain how the system reached its answer.

#### Acceptance Criteria

1. WHEN a query terminates, whether by emitting a Verified_Output, by returning an error record, or by reaching a configured termination reason, THE System SHALL produce a Proof_Trace that records every Reasoning_Step executed before termination.
2. THE Proof_Trace SHALL list each Reasoning_Step in execution order, and for each step SHALL record its sequence position, its validation outcome of accepted, rejected, or repaired, and the identifier of the applied production rule, or an explicit no-rule-applied indicator when no production rule was applied.
3. WHEN a Reasoning_Step undergoes one or more Repair attempts, THE Proof_Trace SHALL record, for each attempt in execution order, the rejected step, the violated production rule, and the resulting repaired step.
4. THE System SHALL export the Proof_Trace in a machine-readable structured format such that every recorded field can be parsed back without loss of recorded content.
5. WHEN a Proof_Trace export is requested, THE System SHALL produce a human-readable rendering that presents each Reasoning_Step, its validation outcome, and its applied production rule in execution order.

### Requirement 9: Evaluation Harness and Baseline Benchmarking

**User Story:** As a researcher, I want an evaluation harness that benchmarks the system against established baselines, so that I can demonstrate measurable improvement.

#### Acceptance Criteria

1. WHEN an evaluation run starts, THE Evaluation_Harness SHALL execute the System and each configured Baseline_Method over every item in the configured dataset.
2. WHEN an evaluation run completes, THE Evaluation_Harness SHALL compute, for each method over all successfully evaluated items, final-answer accuracy, Step_Level_Hallucination_Rate, Faithfulness_Score, and latency overhead.
3. THE Evaluation_Harness SHALL support Chain-of-Thought, Self-Consistency, Tree-of-Thoughts, and ReAct as Baseline_Methods.
4. WHEN an evaluation run completes, THE Evaluation_Harness SHALL produce a comparison report that lists, for each computed metric, the value for the System, the value for each Baseline_Method, and the numeric difference between the System value and each Baseline_Method value.
5. THE Evaluation_Harness SHALL compute latency overhead for a method as the mean per-query difference between that method's wall-clock latency and the LLM_Only_Baseline latency over the same query set.
6. WHERE repeated runs are configured, WHEN an evaluation run completes, THE Evaluation_Harness SHALL compute Reasoning_Consistency for each method across the repeated runs.
7. IF a method fails to produce a result for a dataset item, THEN THE Evaluation_Harness SHALL exclude that item's result for that method, record the exclusion with the failed method and item identifier in the run log, and continue evaluating the remaining items.

### Requirement 10: Evaluation Datasets and Step-Level Benchmark

**User Story:** As a researcher, I want a reusable multi-domain benchmark, so that step-level hallucination can be measured consistently across domains.

#### Acceptance Criteria

1. THE Benchmark SHALL include at least one dataset for each of the following six domains — mathematical reasoning, commonsense reasoning, multi-hop reasoning, science reasoning, logical puzzles, and legal question answering — with each domain providing at least 50 evaluation items.
2. WHEN a dataset is loaded, THE Evaluation_Harness SHALL validate, before evaluation begins, that each item contains a non-empty unique identifier, a non-empty query, a non-empty ground-truth final answer, and a domain label.
3. IF a dataset item is missing its identifier, query, ground-truth final answer, or domain label, THEN THE Evaluation_Harness SHALL exclude the item from evaluation, retain all remaining valid items, and record the excluded item's identifier and the missing field name in the run log.
4. THE Benchmark SHALL associate each evaluated item with exactly one domain label drawn from the six benchmark domains so that metrics can be reported per domain.
5. IF a dataset item carries a domain label that is not one of the six benchmark domains, THEN THE Evaluation_Harness SHALL exclude the item from evaluation and record the excluded item's identifier and the unrecognized label in the run log.
6. WHEN dataset loading completes, THE Evaluation_Harness SHALL record in the run log the total number of items loaded, validated, and excluded per domain.

### Requirement 11: Performance and Latency

**User Story:** As a developer, I want the validation loop overhead to be measured and bounded, so that the accuracy gains can be weighed against the added cost.

#### Acceptance Criteria

1. WHEN a query is processed, THE System SHALL record the wall-clock latency of the complete Pipeline in milliseconds into the Proof_Trace.
2. WHEN a query is processed, THE System SHALL record, in milliseconds and into the Proof_Trace, the cumulative latency attributable to the Validation_Engine and the ACT-R Controller separately from the cumulative LLM generation latency.
3. WHEN an evaluation run completes, THE Evaluation_Harness SHALL report, in milliseconds per method, the mean Pipeline latency and the 95th-percentile Pipeline latency over all queries in the run.
4. WHERE a latency budget is configured, IF the cumulative Validation_Engine and ACT-R Controller latency for a query exceeds the configured latency budget, THEN THE System SHALL record a latency-budget-exceeded indication for that query in the Proof_Trace.

### Requirement 12: Configurability

**User Story:** As a developer, I want runtime parameters to be configurable, so that experiments can be reproduced and tuned without code changes.

#### Acceptance Criteria

1. WHEN the System initializes, THE System SHALL read the maximum cycle limit, the repair attempt limit, the retry count, the LLM selection, the structured output format, and the conflict-resolution policy from configuration before processing any query.
2. WHEN a configuration value is absent at initialization, THE System SHALL apply the documented default value for that parameter and record the applied default in the run record.
3. IF a numeric configuration value is outside its documented valid range (maximum cycle limit: integer 1 to 10000; repair attempt limit: integer 0 to 1000; retry count: integer 0 to 1000), THEN THE System SHALL halt initialization without processing any query and report an error identifying the invalid parameter and its permitted range.
4. IF an enumerated configuration value among the LLM selection, the structured output format, or the conflict-resolution policy is not one of its documented allowed values, THEN THE System SHALL halt initialization without processing any query and report an error identifying the invalid parameter.
5. IF a configuration value cannot be parsed as the documented type for its parameter, THEN THE System SHALL halt initialization without processing any query and report an error identifying the malformed parameter.

### Requirement 13: Reproducibility

**User Story:** As a researcher, I want runs to be reproducible, so that reported results can be independently verified.

#### Acceptance Criteria

1. WHEN an evaluation run starts and before the first dataset item is evaluated, THE Evaluation_Harness SHALL record a run record containing the complete set of configuration parameters, the dataset identifiers, the model identifier, and the random seed in effect for the run, where each of these fields is non-empty.
2. WHEN a random seed is supplied in configuration, THE System SHALL apply that seed to every controllable stochastic operation, including LLM sampling, production-rule conflict resolution, and dataset item ordering and sampling.
3. IF no random seed is supplied in configuration, THEN THE System SHALL generate a seed, apply it to every controllable stochastic operation, and record the generated seed in the run record.
4. WHEN an evaluation run completes, THE Evaluation_Harness SHALL persist the run record together with the computed metrics to an output location that retains the data after the process terminates, such that the run record and its metrics remain associated.
5. IF persisting the run record or the computed metrics to the output location fails, THEN THE Evaluation_Harness SHALL not report the run as successful and SHALL return an error record identifying the failed persistence operation.

### Requirement 14: Adaptive Rule Learning

**User Story:** As a researcher, I want the ACT-R Controller to learn new symbolic production rules from successful reasoning traces, so that the system improves its Procedural_Memory beyond the seeded rules while preserving soundness and reproducibility.

#### Acceptance Criteria

1. WHERE Rule_Learning is enabled, WHEN a query terminates with the active goal in the Goal_Buffer marked as satisfied, THE ACT-R Controller MAY induce one or more Candidate_Rules generalized from the accepted Reasoning_Steps recorded in the Proof_Trace.
2. WHEN a Candidate_Rule is induced, THE ACT-R Controller SHALL record the Candidate_Rule with its Rule_Provenance, including the trace identifiers and the Reasoning_Step identifiers from which the Candidate_Rule was induced.
3. WHEN a Candidate_Rule has been corroborated across at least the configured Corroboration_Threshold of independent successful traces AND the Candidate_Rule does not contradict any existing rule in Procedural_Memory, THE ACT-R Controller SHALL promote the Candidate_Rule into active Procedural_Memory as a Learned_Rule.
4. IF a Candidate_Rule would validate a Reasoning_Step that any existing rule in Procedural_Memory rejects, THEN THE ACT-R Controller SHALL discard the Candidate_Rule without promoting it and SHALL record the discarded Candidate_Rule and the conflicting rule identifier in the run log.
5. THE ACT-R Controller SHALL record each Learned_Rule as distinct from each Seeded_Rule, such that for every accepted Reasoning_Step the Proof_Trace identifies whether the applied production rule was a Learned_Rule or a Seeded_Rule.
6. WHEN a random seed is supplied in configuration, THE ACT-R Controller SHALL make the induction of Candidate_Rules, the corroboration evaluation, and the promotion decisions deterministic under that seed, and SHALL record the resulting Learned_Rule set, the induction seed, the Corroboration_Threshold, and each promotion decision in the run record.
7. THE ACT-R Controller SHALL persist the Learned_Rule_Store with a version identifier to an output location that retains the Learned_Rules and their Rule_Provenance after the process terminates.
8. WHEN the System initializes, THE System SHALL read the Rule_Learning enabled state, the Corroboration_Threshold, and the maximum number of Learned_Rules from configuration, applying the documented defaults of Rule_Learning disabled, a Corroboration_Threshold of 2, and the documented default Learned_Rule cap when a value is absent, and recording each applied default in the run record.
9. IF the number of promoted Learned_Rules reaches the configured maximum number of Learned_Rules, THEN THE ACT-R Controller SHALL stop promoting further Candidate_Rules and SHALL record the cap-reached condition in the run log.
10. WHILE Rule_Learning is disabled, THE System SHALL process queries using only the Seeded_Rules in Procedural_Memory, producing behavior identical to the fixed-rule behavior defined in Requirements 1 through 13.

### Requirement 15: Reasoning Visualization

**User Story:** As a researcher, I want a visual rendering of the reasoning flow derived from the Proof_Trace, so that I can inspect and present how the system progressed from goal to final answer.

#### Acceptance Criteria

1. WHEN a Proof_Trace is exported for visualization, THE System SHALL produce a Reasoning_Visualization that renders every Reasoning_Step in execution order as nodes, indicates the validation outcome of accepted, rejected, or repaired for each step, shows each Repair attempt as a branch, and renders edges representing the flow from each step to the next step, terminating at the Verified_Output or the termination reason.
2. THE System SHALL derive the Reasoning_Visualization solely from the existing append-only Proof_Trace without introducing any new state, such that the Reasoning_Visualization preserves without loss the Reasoning_Step order, the validation outcomes, the applied production rule identifiers, and the termination reason.
3. THE System SHALL emit the Reasoning_Visualization in at least one machine-renderable diagram format as text, such as Mermaid or Graphviz DOT, so that the Reasoning_Visualization can be embedded in documents and demonstrations.
4. THE Reasoning_Visualization SHALL visually distinguish accepted, rejected, and repaired Reasoning_Steps from one another, and SHALL annotate each Reasoning_Step with its applied production rule identifier or an explicit no-rule-applied indicator, and, where a production rule validated the step, whether that rule was a Learned_Rule or a Seeded_Rule.
5. IF the Proof_Trace is empty, THEN THE System SHALL render a well-formed placeholder Reasoning_Visualization rather than failing.
