"""Tests for the Translation Layer forward and backward translation (Task 4.1).

Covers the success paths for Requirements 5.1 (forward: structured step ->
Symbolic_Representation) and 5.2 (backward: working-memory state -> LLM prompt
context). Untranslatable/back-translation failure handling (Task 4.2) and the named
buffer-invariance property (Task 4.3) are intentionally out of scope here.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from nsr.models import (
    CandidateStep,
    Goal,
    PromptContext,
    SubGoal,
    SymbolicRepresentation,
    Untranslatable,
    WorkingMemoryState,
)
from nsr.translation_layer import TranslationLayer


# --------------------------------------------------------------------------- #
# Forward translation: to_symbolic (Requirement 5.1)
# --------------------------------------------------------------------------- #


def test_to_symbolic_converts_structured_step():
    layer = TranslationLayer()
    step = CandidateStep(
        raw_text="Two plus two equals four.",
        structured={"logic_form": "add(2,2)=4", "predicates": {"op": "add", "result": 4}},
        sub_goal="compute sum",
    )

    rep = layer.to_symbolic(step)

    assert isinstance(rep, SymbolicRepresentation)
    assert rep.logic_form == "add(2,2)=4"
    assert rep.predicates == {"op": "add", "result": 4}
    assert rep.source_text == "Two plus two equals four."


def test_to_symbolic_defaults_predicates_when_absent():
    layer = TranslationLayer()
    step = CandidateStep(raw_text="x", structured={"logic_form": "p(x)"})

    rep = layer.to_symbolic(step)

    assert isinstance(rep, SymbolicRepresentation)
    assert rep.predicates == {}


def test_to_symbolic_ignores_non_dict_predicates():
    layer = TranslationLayer()
    step = CandidateStep(
        raw_text="x", structured={"logic_form": "p(x)", "predicates": "not-a-dict"}
    )

    rep = layer.to_symbolic(step)

    assert isinstance(rep, SymbolicRepresentation)
    assert rep.predicates == {}


def test_to_symbolic_returns_untranslatable_when_logic_form_missing():
    layer = TranslationLayer()
    step = CandidateStep(raw_text="free-form prose", structured={"predicates": {}})

    result = layer.to_symbolic(step)

    assert isinstance(result, Untranslatable)
    assert result.step is step
    assert result.reason


def test_to_symbolic_returns_untranslatable_when_logic_form_blank():
    layer = TranslationLayer()
    step = CandidateStep(raw_text="x", structured={"logic_form": "   "})

    result = layer.to_symbolic(step)

    assert isinstance(result, Untranslatable)


# --------------------------------------------------------------------------- #
# Backward translation: to_context (Requirement 5.2)
# --------------------------------------------------------------------------- #


def test_to_context_includes_goal_subgoal_conclusions_and_imaginal():
    layer = TranslationLayer()
    state = WorkingMemoryState(
        goal_buffer=Goal(
            description="Solve the equation",
            sub_goals=[
                SubGoal(description="isolate x", satisfied=True),
                SubGoal(description="simplify", satisfied=False),
            ],
        ),
        declarative_memory=[
            SymbolicRepresentation(logic_form="eq(2x+4,10)"),
            SymbolicRepresentation(logic_form="eq(2x,6)"),
        ],
        imaginal_buffer=SymbolicRepresentation(logic_form="eq(x,3)"),
    )

    ctx = layer.to_context(state)

    assert isinstance(ctx, PromptContext)
    assert ctx.goal_description == "Solve the equation"
    # First unsatisfied sub-goal is selected as the active sub-goal.
    assert ctx.active_sub_goal == "simplify"
    assert ctx.partial_representation == "eq(x,3)"
    assert ctx.established_conclusions == ["eq(2x+4,10)", "eq(2x,6)"]
    # Rendered prompt surfaces every piece of symbolic state.
    assert "Goal: Solve the equation" in ctx.prompt_text
    assert "Current sub-goal: simplify" in ctx.prompt_text
    assert "- eq(2x+4,10)" in ctx.prompt_text
    assert "Partial representation: eq(x,3)" in ctx.prompt_text


def test_to_context_with_no_subgoals_has_no_active_subgoal():
    layer = TranslationLayer()
    state = WorkingMemoryState(goal_buffer=Goal(description="g"))

    ctx = layer.to_context(state)

    assert ctx.active_sub_goal is None
    assert ctx.partial_representation is None
    assert ctx.established_conclusions == []
    assert "Established conclusions: (none)" in ctx.prompt_text


def test_to_context_active_subgoal_is_first_unsatisfied():
    layer = TranslationLayer()
    state = WorkingMemoryState(
        goal_buffer=Goal(
            description="g",
            sub_goals=[
                SubGoal(description="a", satisfied=True),
                SubGoal(description="b", satisfied=False),
                SubGoal(description="c", satisfied=False),
            ],
        )
    )

    ctx = layer.to_context(state)

    assert ctx.active_sub_goal == "b"


# --------------------------------------------------------------------------- #
# Property tests for success-path translation invariants
# --------------------------------------------------------------------------- #

_non_blank_logic_form = st.text(min_size=1).filter(lambda s: s.strip() != "")
_predicate_dicts = st.dictionaries(
    keys=st.text(min_size=1, max_size=8),
    values=st.one_of(st.integers(), st.text(max_size=8), st.booleans()),
    max_size=5,
)


@given(
    raw_text=st.text(max_size=40),
    logic_form=_non_blank_logic_form,
    predicates=_predicate_dicts,
)
def test_to_symbolic_preserves_encoding_and_source(raw_text, logic_form, predicates):
    """Forward translation preserves the logic form, predicates, and source text."""
    layer = TranslationLayer()
    step = CandidateStep(
        raw_text=raw_text,
        structured={"logic_form": logic_form, "predicates": predicates},
    )

    rep = layer.to_symbolic(step)

    assert isinstance(rep, SymbolicRepresentation)
    assert rep.logic_form == logic_form
    assert rep.predicates == predicates
    assert rep.source_text == raw_text


@given(conclusions=st.lists(st.text(min_size=1, max_size=12), max_size=8))
def test_to_context_preserves_declarative_order_and_count(conclusions):
    """Backward translation surfaces every declarative conclusion in order."""
    layer = TranslationLayer()
    state = WorkingMemoryState(
        goal_buffer=Goal(description="g"),
        declarative_memory=[SymbolicRepresentation(logic_form=c) for c in conclusions],
    )

    ctx = layer.to_context(state)

    assert ctx.established_conclusions == conclusions
