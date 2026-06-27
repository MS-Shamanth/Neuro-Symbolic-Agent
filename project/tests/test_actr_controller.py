"""Unit tests for the ACT-R Controller buffer maintenance (Task 3.1).

These cover the buffer-maintenance and accepted-step integration behaviour required by
Requirements 4.1, 4.2, 4.4, and 4.5:

- the four buffers are maintained for the lifetime of a query;
- an accepted step appends a distinct Declarative_Memory entry;
- the Imaginal_Buffer is replaced to reflect the accepted step;
- all prior accepted conclusions are retained, in order, until termination.
"""

from __future__ import annotations

import pytest

from nsr import ACTRController
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


def test_initialize_sets_up_all_four_buffers():
    # Req 4.1: maintain Goal/Declarative/Procedural/Imaginal buffers for the query.
    ctrl = ACTRController()
    goal = Goal(description="solve", sub_goals=[SubGoal(description="step 1")])
    rules = [ProductionRule(rule_id="R1", condition="IF a", action="THEN b")]

    ctrl.initialize(goal, procedural_memory=rules)
    state = ctrl.state()

    assert isinstance(state, WorkingMemoryState)
    assert state.goal_buffer.description == "solve"
    assert state.declarative_memory == []
    assert [r.rule_id for r in state.procedural_memory] == ["R1"]
    assert state.imaginal_buffer is None


def test_state_before_initialize_raises():
    ctrl = ACTRController()
    with pytest.raises(RuntimeError):
        ctrl.state()
    with pytest.raises(RuntimeError):
        ctrl.integrate_accepted(_rep("x"))


def test_initialize_defaults_procedural_memory_to_empty():
    ctrl = ACTRController()
    ctrl.initialize(Goal(description="g"))
    assert ctrl.state().procedural_memory == []


def test_integrate_accepted_appends_distinct_declarative_entry():
    # Req 4.2: store the accepted conclusion as a distinct Declarative_Memory entry.
    ctrl = ACTRController()
    ctrl.initialize(Goal(description="g"))

    step = _rep("add(2,2)=4")
    ctrl.integrate_accepted(step)

    dm = ctrl.state().declarative_memory
    assert len(dm) == 1
    assert dm[0].logic_form == "add(2,2)=4"
    # The stored entry must be a distinct object from the caller's step (Req 4.2/4.4),
    # so later mutation of the caller's object never alters the recorded conclusion.
    assert dm[0] is not step


def test_integrate_accepted_replaces_imaginal_buffer():
    # Req 4.5: replace the Imaginal_Buffer with a representation reflecting the step.
    ctrl = ACTRController()
    ctrl.initialize(Goal(description="g"))

    ctrl.integrate_accepted(_rep("first"))
    assert ctrl.state().imaginal_buffer.logic_form == "first"

    ctrl.integrate_accepted(_rep("second"))
    # The buffer holds only the most recently accepted step.
    assert ctrl.state().imaginal_buffer.logic_form == "second"


def test_declarative_memory_retains_all_conclusions_in_order():
    # Req 4.4: retain all previously accepted conclusions, ordered, until termination.
    ctrl = ACTRController()
    ctrl.initialize(Goal(description="g"))

    forms = ["s0", "s1", "s2", "s3"]
    for f in forms:
        ctrl.integrate_accepted(_rep(f))

    dm = ctrl.state().declarative_memory
    assert [r.logic_form for r in dm] == forms
    # Imaginal buffer reflects only the latest step, but declarative memory keeps all.
    assert ctrl.state().imaginal_buffer.logic_form == "s3"


def test_each_declarative_entry_is_distinct_even_for_equal_steps():
    # Req 4.2: each accepted conclusion is a distinct entry, even when steps are equal.
    ctrl = ACTRController()
    ctrl.initialize(Goal(description="g"))

    shared = _rep("dup")
    ctrl.integrate_accepted(shared)
    ctrl.integrate_accepted(shared)

    dm = ctrl.state().declarative_memory
    assert len(dm) == 2
    assert dm[0] is not dm[1]
    assert dm[0].logic_form == dm[1].logic_form == "dup"


def test_state_snapshot_is_isolated_from_later_updates():
    # A returned snapshot must not change when the controller integrates further steps.
    ctrl = ACTRController()
    ctrl.initialize(Goal(description="g"))
    ctrl.integrate_accepted(_rep("a"))

    snapshot = ctrl.state()
    assert len(snapshot.declarative_memory) == 1

    ctrl.integrate_accepted(_rep("b"))
    # The earlier snapshot is unaffected by the later acceptance.
    assert len(snapshot.declarative_memory) == 1
    assert len(ctrl.state().declarative_memory) == 2


def test_mutating_caller_step_after_accept_does_not_change_memory():
    # Distinct-entry guarantee: mutating the caller's object post-accept is isolated.
    ctrl = ACTRController()
    ctrl.initialize(Goal(description="g"))

    step = _rep("orig")
    ctrl.integrate_accepted(step)
    step.logic_form = "mutated"
    step.predicates["x"] = 1

    dm = ctrl.state().declarative_memory
    assert dm[0].logic_form == "orig"
    assert dm[0].predicates == {}


def test_reinitialize_clears_buffers_for_new_query():
    # Buffers are scoped to a query lifetime; re-initializing starts a clean state.
    ctrl = ACTRController()
    ctrl.initialize(Goal(description="g1"))
    ctrl.integrate_accepted(_rep("a"))
    assert len(ctrl.state().declarative_memory) == 1

    ctrl.initialize(Goal(description="g2"))
    state = ctrl.state()
    assert state.goal_buffer.description == "g2"
    assert state.declarative_memory == []
    assert state.imaginal_buffer is None
