"""Property test for deterministic rule learning (Task 17.7).

**Property 6: Rule learning is deterministic under a fixed seed.**

The RuleLearner pipeline ``induce -> corroborate -> promote`` is a pure,
ordering-stable function of its inputs and the supplied seed (Req 14.6). For any
fixed inputs (goal-satisfied traces, an existing rule set, a corroboration
threshold, a learned-rule cap) and a fixed seed, running the full pipeline twice
with two freshly constructed, identically seeded ``RuleLearner`` instances yields:

- equal ``LearnedRuleStore`` contents (candidates + promoted learned rules),
- an identical sequence of ``PromotionDecision`` objects, and
- equal recorded run-record fields (``learned_rules``, ``induction_seed``,
  ``corroboration_threshold``, ``promotion_decisions``).

**Validates: Requirements 14.6**
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from nsr import RuleLearner, ValidationEngine
from nsr.models import (
    LearnedRuleStore,
    ProductionRule,
    SymbolicRepresentation,
    store_to_dict,
)
from nsr.models.config import RunRecord, SystemConfig
from nsr.models.enums import TerminationReason, ValidationStatus
from nsr.models.trace import ProofStep, ProofTrace

# --------------------------------------------------------------------------- #
# Generators - constrain to the pipeline's input space.                       #
# --------------------------------------------------------------------------- #

# A small token alphabet keeps generalizations colliding often enough that
# corroboration, contradiction and the cap are all genuinely exercised.
_TOKENS = ["foo", "bar", "baz", "qux", "add", "mul", "rel", "ok", "x", "y"]
_PRED_KEYS = ["operation", "operands", "op", "kind", "lhs", "rhs", "k", "p"]

_token = st.sampled_from(_TOKENS)
_pred_key = st.sampled_from(_PRED_KEYS)


@st.composite
def _representations(draw):
    """A SymbolicRepresentation drawn from the validator-searchable text space."""
    logic_tokens = draw(st.lists(_token, min_size=1, max_size=4))
    # Numeric literals are instance-specific and generalized away; mix some in.
    logic_form = " ".join(logic_tokens) + f"({draw(st.integers(0, 9))})"
    predicates = draw(
        st.dictionaries(
            _pred_key,
            st.one_of(st.integers(-5, 5), _token),
            max_size=3,
        )
    )
    return SymbolicRepresentation(
        logic_form=logic_form,
        source_text=" ".join(logic_tokens),
        predicates=predicates,
    )


@st.composite
def _proof_steps(draw):
    status = draw(
        st.sampled_from(
            [
                ValidationStatus.ACCEPTED,
                ValidationStatus.REPAIRED,
                ValidationStatus.REJECTED,
            ]
        )
    )
    return ProofStep(
        sequence=draw(st.integers(0, 20)),
        step_text=f"step-{draw(st.integers(0, 20))}",
        representation=draw(_representations()),
        status=status,
    )


@st.composite
def _traces(draw):
    """A trace whose termination reason is usually goal-satisfied (induction fires)."""
    reason = draw(
        st.sampled_from(
            [
                TerminationReason.GOAL_SATISFIED,
                TerminationReason.GOAL_SATISFIED,
                TerminationReason.GOAL_SATISFIED,
                TerminationReason.CYCLE_LIMIT_REACHED,
            ]
        )
    )
    steps = draw(st.lists(_proof_steps(), min_size=0, max_size=4))
    return ProofTrace(steps=steps, termination_reason=reason)


@st.composite
def _existing_rules(draw):
    """Existing Procedural_Memory rules a promoted candidate must not contradict."""
    n = draw(st.integers(0, 3))
    rules = []
    for i in range(n):
        cond_terms = draw(st.lists(_token, min_size=0, max_size=3))
        act_terms = draw(st.lists(_token, min_size=0, max_size=3))
        condition = "IF" + ("" if not cond_terms else " " + " AND ".join(cond_terms))
        action = "THEN" + ("" if not act_terms else " " + " AND ".join(act_terms))
        rules.append(
            ProductionRule(rule_id=f"seed-{i}", condition=condition, action=action)
        )
    return rules


def _make_run_record(seed: int) -> RunRecord:
    return RunRecord(
        config=SystemConfig(
            max_cycle_limit=10,
            repair_attempt_limit=1,
            retry_count=0,
            llm_selection="x",
            output_format="json",
            conflict_resolution_policy="specificity",
            generation_timeout_ms=1000,
        ),
        dataset_ids=["d1"],
        model_id="m1",
        seed=seed,
    )


def _run_pipeline(traces, trace_ids, existing_rules, threshold, cap, seed):
    """Run induce -> corroborate -> promote with a fresh learner, then record the run.

    Returns the store dict (lossless contents), the promotion decisions, and the
    run record so two independent runs can be compared for equality.
    """
    learner = RuleLearner(
        LearnedRuleStore(),
        ValidationEngine(),
        corroboration_threshold=threshold,
        max_learned_rules=cap,
        seed=seed,
    )
    for trace, trace_id in zip(traces, trace_ids):
        learner.corroborate(learner.induce(trace, trace_id=trace_id))
    result = learner.promote(procedural_memory=existing_rules)

    run_record = _make_run_record(seed)
    learner.record_run(run_record, result)

    return store_to_dict(learner.store), result.decisions, run_record


@given(
    traces=st.lists(_traces(), min_size=0, max_size=6),
    existing_rules=_existing_rules(),
    threshold=st.integers(1, 3),
    cap=st.integers(0, 5),
    seed=st.integers(0, 2**31 - 1),
)
def test_rule_learning_is_deterministic_under_fixed_seed(
    traces, existing_rules, threshold, cap, seed
):
    """Property 6: identical inputs + seed => identical store, decisions, run record.

    **Validates: Requirements 14.6**
    """
    trace_ids = [f"trace-{i}" for i in range(len(traces))]

    store_a, decisions_a, record_a = _run_pipeline(
        traces, trace_ids, existing_rules, threshold, cap, seed
    )
    store_b, decisions_b, record_b = _run_pipeline(
        traces, trace_ids, existing_rules, threshold, cap, seed
    )

    # Equal LearnedRuleStore contents (candidates + promoted learned rules).
    assert store_a == store_b

    # Identical sequence of PromotionDecisions.
    assert decisions_a == decisions_b

    # Equal recorded run-record fields.
    assert record_a.learned_rules == record_b.learned_rules
    assert record_a.induction_seed == record_b.induction_seed
    assert record_a.corroboration_threshold == record_b.corroboration_threshold
    assert record_a.promotion_decisions == record_b.promotion_decisions

    # The seed is faithfully threaded into the run record (Req 14.6).
    assert record_a.induction_seed == seed
    assert record_a.corroboration_threshold == threshold
