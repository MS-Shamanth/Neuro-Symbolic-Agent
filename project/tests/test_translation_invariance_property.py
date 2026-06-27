"""Property test for the Translation_Layer's untranslatable-step invariance (Task 4.3).

**Property 13: Untranslatable steps leave working memory unchanged.**

For any working-memory state and any candidate step that fails forward translation,
the working-memory buffers after the translation attempt equal the buffers before it,
and the step is routed to repair (the forward translation yields an
:class:`~nsr.models.Untranslatable` outcome rather than a Symbolic_Representation).

**Validates: Requirements 5.3**
"""

from __future__ import annotations

import copy

from hypothesis import given
from hypothesis import strategies as st

from nsr.models import (
    Goal,
    ProductionRule,
    SubGoal,
    SymbolicRepresentation,
    Untranslatable,
    WorkingMemoryState,
)
from nsr.models.translation import CandidateStep
from nsr.translation_layer import LOGIC_FORM_KEY, PREDICATES_KEY, TranslationLayer


# --- Generators for arbitrary working-memory state --------------------------------

text = st.text(max_size=40)


@st.composite
def sub_goals(draw):
    return SubGoal(description=draw(text), satisfied=draw(st.booleans()))


@st.composite
def symbolic_representations(draw):
    """A non-empty logic_form keeps these representations validly translated already."""
    return SymbolicRepresentation(
        logic_form=draw(st.text(min_size=1, max_size=30)),
        predicates=draw(
            st.dictionaries(st.text(max_size=8), st.integers(), max_size=3)
        ),
        source_text=draw(text),
    )


@st.composite
def production_rules(draw):
    return ProductionRule(
        rule_id=draw(st.text(min_size=1, max_size=8)),
        condition=draw(text),
        action=draw(text),
    )


@st.composite
def working_memory_states(draw):
    """Generate an arbitrary :class:`WorkingMemoryState` across all four buffers."""
    goal = Goal(
        description=draw(text),
        sub_goals=draw(st.lists(sub_goals(), max_size=4)),
        satisfied=draw(st.booleans()),
    )
    return WorkingMemoryState(
        goal_buffer=goal,
        declarative_memory=draw(st.lists(symbolic_representations(), max_size=4)),
        procedural_memory=draw(st.lists(production_rules(), max_size=4)),
        imaginal_buffer=draw(st.one_of(st.none(), symbolic_representations())),
    )


# --- Generators for candidate steps that fail forward translation -----------------

# Forward translation (to_symbolic) returns Untranslatable when the candidate carries
# no non-empty *string* logic_form. We cover every such case:
#   - the logic_form key is absent,
#   - it is blank / whitespace-only,
#   - it is a non-string value.
blank_logic_forms = st.sampled_from(["", "   ", "\t", "\n  \n"])
non_string_logic_forms = st.one_of(
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.lists(st.integers(), max_size=3),
    st.dictionaries(st.text(max_size=4), st.integers(), max_size=2),
    st.none(),
)


@st.composite
def untranslatable_structured(draw):
    """A ``structured`` payload guaranteed to lack a non-empty string logic_form."""
    payload = {}
    # Optionally carry along arbitrary predicates and noise fields.
    if draw(st.booleans()):
        payload[PREDICATES_KEY] = draw(
            st.dictionaries(st.text(max_size=6), st.integers(), max_size=3)
        )
    if draw(st.booleans()):
        payload["note"] = draw(text)

    kind = draw(st.sampled_from(["absent", "blank", "non_string"]))
    if kind == "absent":
        # Leave the logic_form key out entirely.
        pass
    elif kind == "blank":
        payload[LOGIC_FORM_KEY] = draw(blank_logic_forms)
    else:
        payload[LOGIC_FORM_KEY] = draw(non_string_logic_forms)
    return payload


@st.composite
def untranslatable_steps(draw):
    return CandidateStep(
        raw_text=draw(text),
        structured=draw(untranslatable_structured()),
        sub_goal=draw(st.one_of(st.none(), text)),
    )


# --- The property -----------------------------------------------------------------


@given(state=working_memory_states(), step=untranslatable_steps())
def test_untranslatable_step_leaves_working_memory_unchanged(state, step):
    """Property 13 — untranslatable steps leave working memory unchanged.

    **Validates: Requirements 5.3**
    """
    layer = TranslationLayer()

    # Snapshot every buffer before the translation attempt.
    before = copy.deepcopy(state)

    result = layer.forward(step)

    # The step fails translation and is routed to repair (an Untranslatable outcome,
    # never a SymbolicRepresentation).
    assert isinstance(result, Untranslatable)
    assert result.step is step
    assert isinstance(result.reason, str) and result.reason

    # The working-memory buffers are byte-for-byte unchanged after the attempt.
    assert state == before
    assert state.goal_buffer == before.goal_buffer
    assert state.declarative_memory == before.declarative_memory
    assert state.procedural_memory == before.procedural_memory
    assert state.imaginal_buffer == before.imaginal_buffer
