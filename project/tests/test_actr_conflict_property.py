"""Property-based test for deterministic ACT-R conflict resolution (Task 3.3).

**Property 3: Same state and seed always select the same rule.**

For any working-memory state in which *multiple* production rules match, repeated
:meth:`ACTRController.select_rule` calls — on the same controller, on a fresh controller
built with the same seed and policy, and on an equal (deep-copied) state — always return
the identical rule id, for every configured conflict-resolution policy.

**Validates: Requirements 4.6, 13.2**

Rule matching in the controller is text-based: a rule matches when every term in its
``IF`` condition appears in the lower-cased text of the working-memory state (goal,
sub-goals, declarative memory, imaginal buffer). The generator below exploits this by
first drawing a vocabulary of tokens, planting every token in the state's text, and then
building rule conditions exclusively from those planted tokens. This guarantees that
every generated rule matches, so each generated state has multiple matching rules and the
conflict-resolution policy is always exercised (never short-circuited by a no-match).
"""

from __future__ import annotations

import copy

from hypothesis import given, settings
from hypothesis import strategies as st

from nsr.actr_controller import ACTRController
from nsr.config_manager import ALLOWED_CONFLICT_POLICIES
from nsr.models import (
    Goal,
    ProductionRule,
    SubGoal,
    SymbolicRepresentation,
    WorkingMemoryState,
)

POLICIES = sorted(ALLOWED_CONFLICT_POLICIES)  # ["priority", "recency", "specificity"]

# Tokens are short lower-case words. They are the atoms both planted in the state's text
# and used to build rule conditions, so every generated rule is guaranteed to match.
_tokens = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=6)


@st.composite
def states_with_multiple_matching_rules(draw):
    """Draw a (state, expected_match_count) pair whose rules all match the state.

    Every vocabulary token is planted in the goal description and across distinct
    Declarative_Memory entries (so the ``recency`` policy has ordered support to rank
    on). Each production rule's condition is a conjunction of a non-empty subset of those
    planted tokens, guaranteeing the rule matches the state's text.
    """
    vocab = draw(st.lists(_tokens, min_size=2, max_size=6, unique=True))

    # Plant every token in the goal description and in a distinct declarative entry each,
    # so the global haystack contains all tokens and recency indices are well-defined.
    goal = Goal(
        description="goal " + " ".join(vocab),
        sub_goals=[SubGoal(description="sub " + tok) for tok in vocab],
    )
    declarative = [
        SymbolicRepresentation(
            logic_form=f"fact({tok})",
            predicates={"tok": tok},
            source_text=f"established {tok}",
        )
        for tok in vocab
    ]

    # Build matching rules: each condition is a conjunction of planted tokens. Rule ids
    # are unique so an "identical rule id" assertion is meaningful and selection is
    # uniquely determined.
    n_rules = draw(st.integers(min_value=2, max_value=6))
    rules: list[ProductionRule] = []
    for i in range(n_rules):
        terms = draw(st.lists(st.sampled_from(vocab), min_size=1, max_size=len(vocab)))
        condition = "IF " + " AND ".join(terms)
        rules.append(
            ProductionRule(rule_id=f"R{i}", condition=condition, action=f"THEN act{i}")
        )

    imaginal = SymbolicRepresentation(
        logic_form="partial(" + ",".join(vocab) + ")",
        source_text="partial " + " ".join(vocab),
    )

    state = WorkingMemoryState(
        goal_buffer=goal,
        declarative_memory=declarative,
        procedural_memory=rules,
        imaginal_buffer=imaginal,
    )
    return state, n_rules


def _make_controller(policy: str, seed, state: WorkingMemoryState) -> ACTRController:
    controller = ACTRController(conflict_resolution_policy=policy, seed=seed)
    controller.initialize(state.goal_buffer, state.procedural_memory)
    return controller


@settings(max_examples=200, deadline=None)
@given(
    data=states_with_multiple_matching_rules(),
    policy=st.sampled_from(POLICIES),
    seed=st.one_of(st.none(), st.integers(min_value=0, max_value=2**32 - 1)),
)
def test_same_state_and_seed_always_select_the_same_rule(data, policy, seed):
    """Property 3 (Validates Requirements 4.6, 13.2)."""
    state, n_rules = data

    controller = _make_controller(policy, seed, state)
    first = controller.select_rule(state)

    # Precondition for this property: the state has multiple matching rules, so a single
    # rule must be selected deterministically (never a NoRuleMatched outcome here).
    assert isinstance(first, ProductionRule), (
        f"expected a matching rule among {n_rules} rules, got {first!r}"
    )

    # 1) Repeated selection on the same controller is stable.
    for _ in range(5):
        again = controller.select_rule(state)
        assert isinstance(again, ProductionRule)
        assert again.rule_id == first.rule_id

    # 2) A fresh controller with the same seed and policy selects the same rule id.
    fresh = _make_controller(policy, seed, state)
    assert fresh.select_rule(state).rule_id == first.rule_id

    # 3) Selection depends only on state *value*, not identity: an equal (deep-copied)
    #    state under the same seed/policy yields the same rule id.
    state_copy = copy.deepcopy(state)
    fresh_on_copy = _make_controller(policy, seed, state_copy)
    assert fresh_on_copy.select_rule(state_copy).rule_id == first.rule_id
