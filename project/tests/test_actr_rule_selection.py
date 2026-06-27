"""Unit tests for sub-goal advancement and deterministic rule selection (Task 3.2).

These cover the ACT-R Controller behaviour required by Requirements 4.3, 4.6, 4.7, and
4.8:

- the Goal_Buffer advances to the next unmet sub-goal (Req 4.3);
- when no unmet sub-goal remains, the active goal is marked satisfied (Req 4.7);
- when multiple rules match, exactly one is selected deterministically via the
  configured conflict-resolution policy (Req 4.6);
- when no rule matches, a ``NoRuleMatched`` outcome routes the state to repair (Req 4.8).
"""

from __future__ import annotations

import pytest

from nsr import ACTRController, NoRuleMatched
from nsr.models import (
    Goal,
    ProductionRule,
    SubGoal,
    SymbolicRepresentation,
    WorkingMemoryState,
)


def _rep(form: str, **predicates) -> SymbolicRepresentation:
    return SymbolicRepresentation(
        logic_form=form,
        predicates=dict(predicates),
        source_text=form,
    )


def _controller(policy: str = "priority", *, goal: Goal, rules=None) -> ACTRController:
    ctrl = ACTRController(conflict_resolution_policy=policy)
    ctrl.initialize(goal, procedural_memory=rules or [])
    return ctrl


# --------------------------------------------------------------- sub-goal advancement


def test_advance_moves_to_next_unmet_sub_goal():
    # Req 4.3: advancing satisfies the active sub-goal and returns the next unmet one.
    goal = Goal(
        description="solve",
        sub_goals=[SubGoal("a"), SubGoal("b"), SubGoal("c")],
    )
    ctrl = _controller(goal=goal)

    assert ctrl.active_sub_goal().description == "a"

    nxt = ctrl.advance_sub_goal()
    assert nxt is not None
    assert nxt.description == "b"
    # The previously active sub-goal is now satisfied; the goal is not yet satisfied.
    assert ctrl.goal_buffer.sub_goals[0].satisfied is True
    assert ctrl.goal_buffer.satisfied is False
    assert ctrl.active_sub_goal().description == "b"


def test_advance_marks_goal_satisfied_when_no_unmet_remain():
    # Req 4.7: advancing the last unmet sub-goal marks the active goal satisfied.
    goal = Goal(description="solve", sub_goals=[SubGoal("only")])
    ctrl = _controller(goal=goal)

    result = ctrl.advance_sub_goal()
    assert result is None
    assert ctrl.goal_buffer.satisfied is True
    assert ctrl.active_sub_goal() is None


def test_advance_through_all_sub_goals_in_order():
    goal = Goal(
        description="solve",
        sub_goals=[SubGoal("s0"), SubGoal("s1"), SubGoal("s2")],
    )
    ctrl = _controller(goal=goal)

    assert ctrl.advance_sub_goal().description == "s1"
    assert ctrl.advance_sub_goal().description == "s2"
    assert ctrl.advance_sub_goal() is None
    assert ctrl.goal_buffer.satisfied is True


def test_advance_goal_with_no_sub_goals_marks_satisfied():
    # A goal carrying no sub-goals has no unmet sub-goal: it is marked satisfied (4.7).
    ctrl = _controller(goal=Goal(description="atomic"))
    assert ctrl.active_sub_goal() is None
    assert ctrl.advance_sub_goal() is None
    assert ctrl.goal_buffer.satisfied is True


def test_advance_before_initialize_raises():
    ctrl = ACTRController()
    with pytest.raises(RuntimeError):
        ctrl.advance_sub_goal()
    with pytest.raises(RuntimeError):
        ctrl.active_sub_goal()


def test_active_sub_goal_returns_copy():
    goal = Goal(description="g", sub_goals=[SubGoal("a")])
    ctrl = _controller(goal=goal)
    sg = ctrl.active_sub_goal()
    sg.satisfied = True
    # Mutating the returned copy must not affect the controller's buffer.
    assert ctrl.goal_buffer.sub_goals[0].satisfied is False


# ------------------------------------------------------------------- rule matching


def test_no_matching_rule_returns_no_rule_matched():
    # Req 4.8: when nothing matches, return a NoRuleMatched routed to repair.
    goal = Goal(description="prove triangle")
    rules = [ProductionRule(rule_id="R1", condition="IF circle", action="THEN x")]
    ctrl = _controller(goal=goal, rules=rules)

    outcome = ctrl.select_rule(ctrl.state())
    assert isinstance(outcome, NoRuleMatched)
    assert outcome.route_to_repair is True
    assert isinstance(outcome.state, WorkingMemoryState)


def test_empty_procedural_memory_returns_no_rule_matched():
    ctrl = _controller(goal=Goal(description="anything"), rules=[])
    assert isinstance(ctrl.select_rule(ctrl.state()), NoRuleMatched)


def test_single_matching_rule_is_selected():
    goal = Goal(description="add two numbers")
    rules = [
        ProductionRule(rule_id="R1", condition="IF subtract", action="THEN x"),
        ProductionRule(rule_id="R2", condition="IF add", action="THEN y"),
    ]
    ctrl = _controller(goal=goal, rules=rules)
    selected = ctrl.select_rule(ctrl.state())
    assert isinstance(selected, ProductionRule)
    assert selected.rule_id == "R2"


def test_empty_condition_matches_unconditionally():
    # A bare/empty condition acts as a fallback default rule that always matches.
    goal = Goal(description="anything")
    rules = [ProductionRule(rule_id="DEFAULT", condition="", action="THEN fallback")]
    ctrl = _controller(goal=goal, rules=rules)
    selected = ctrl.select_rule(ctrl.state())
    assert selected.rule_id == "DEFAULT"


def test_conjunctive_condition_requires_all_terms():
    goal = Goal(description="add numbers")  # has "add" but not "fraction"
    rules = [
        ProductionRule(rule_id="R1", condition="IF add AND fraction", action="THEN x"),
        ProductionRule(rule_id="R2", condition="IF add", action="THEN y"),
    ]
    ctrl = _controller(goal=goal, rules=rules)
    selected = ctrl.select_rule(ctrl.state())
    # R1 fails because "fraction" is absent; only R2 matches.
    assert selected.rule_id == "R2"


# ---------------------------------------------------- conflict-resolution policies


def _three_matching_rules():
    # All three match a state mentioning "add"; they differ in specificity/order.
    return [
        ProductionRule(rule_id="B", condition="IF add", action="t"),
        ProductionRule(rule_id="A", condition="IF add AND sum", action="t"),
        ProductionRule(rule_id="C", condition="IF add", action="t"),
    ]


def test_priority_policy_picks_earliest_position():
    # Req 4.6: priority selects the earliest matching rule in Procedural_Memory order.
    goal = Goal(description="add the sum")
    ctrl = _controller("priority", goal=goal, rules=_three_matching_rules())
    selected = ctrl.select_rule(ctrl.state())
    assert selected.rule_id == "B"  # position 0


def test_specificity_policy_picks_most_specific():
    # Req 4.6: specificity selects the rule with the most match terms.
    goal = Goal(description="add the sum")
    ctrl = _controller("specificity", goal=goal, rules=_three_matching_rules())
    selected = ctrl.select_rule(ctrl.state())
    assert selected.rule_id == "A"  # two terms ("add" AND "sum")


def test_recency_policy_prefers_most_recent_declarative_support():
    # Req 4.6: recency prefers the rule supported by the most recently accepted fact.
    goal = Goal(description="solve")
    rules = [
        ProductionRule(rule_id="OLD", condition="IF first", action="t"),
        ProductionRule(rule_id="NEW", condition="IF third", action="t"),
    ]
    ctrl = _controller("recency", goal=goal, rules=rules)
    ctrl.integrate_accepted(_rep("first"))   # declarative index 0
    ctrl.integrate_accepted(_rep("second"))  # declarative index 1
    ctrl.integrate_accepted(_rep("third"))   # declarative index 2 (most recent)

    selected = ctrl.select_rule(ctrl.state())
    assert selected.rule_id == "NEW"


def test_specificity_tie_breaks_on_rule_id():
    # Equal specificity -> deterministic tie-break on rule_id (lexicographic).
    goal = Goal(description="add")
    rules = [
        ProductionRule(rule_id="Z", condition="IF add", action="t"),
        ProductionRule(rule_id="A", condition="IF add", action="t"),
    ]
    ctrl = _controller("specificity", goal=goal, rules=rules)
    assert ctrl.select_rule(ctrl.state()).rule_id == "A"


# --------------------------------------------------------------------- determinism


def test_selection_is_deterministic_across_repeated_calls():
    # Property 3 foundation: same state + policy -> identical rule id, every time.
    goal = Goal(description="add the sum")
    for policy in ("priority", "specificity", "recency"):
        ctrl = _controller(policy, goal=goal, rules=_three_matching_rules())
        state = ctrl.state()
        first = ctrl.select_rule(state)
        for _ in range(20):
            again = ctrl.select_rule(state)
            assert again.rule_id == first.rule_id


def test_selection_independent_of_seed():
    # Policies consume no randomness: any seed yields the same selection.
    goal = Goal(description="add the sum")
    rules = _three_matching_rules()
    a = ACTRController("specificity", seed=1)
    a.initialize(goal, procedural_memory=rules)
    b = ACTRController("specificity", seed=999)
    b.initialize(goal, procedural_memory=rules)
    assert a.select_rule(a.state()).rule_id == b.select_rule(b.state()).rule_id


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        ACTRController(conflict_resolution_policy="bogus")
