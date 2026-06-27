"""Property-based test for declarative memory retention (Task 3.4).

**Property 9: Declarative memory grows monotonically and retains all conclusions**

For any sequence of accepted reasoning steps, every accepted conclusion remains present
and ordered in Declarative_Memory until termination, and each is a distinct entry.

**Validates: Requirements 4.2, 4.4**

Requirement 4.2 requires that, when a Reasoning_Step is accepted, the ACT-R Controller
stores the resulting intermediate conclusion as a *distinct* entry in Declarative_Memory
before the next step is generated. Requirement 4.4 requires that all previously accepted
intermediate conclusions are retained in Declarative_Memory until the query terminates.

Together these yield a monotonic-growth invariant: after each acceptance the
Declarative_Memory is exactly the ordered list of all conclusions accepted so far, every
earlier conclusion is still present in its original position, and no entry shares object
identity with another (so mutating one entry never affects another).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from nsr import ACTRController
from nsr.models import Goal, SymbolicRepresentation


# --------------------------------------------------------------------- generators

# Predicate values are kept to simple, hashable-friendly JSON-like scalars so the
# generated SymbolicRepresentation objects mirror realistic parsed-step fields without
# straying outside the input space the controller actually handles.
_predicate_values = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.text(max_size=20),
    st.booleans(),
)


@st.composite
def symbolic_representations(draw: st.DrawFn) -> SymbolicRepresentation:
    """Generate a single accepted-step SymbolicRepresentation.

    ``logic_form`` and ``source_text`` are arbitrary text; ``predicates`` is a small
    dict of scalar fields. The generator deliberately allows duplicate content across
    draws so the distinct-entry guarantee (Req 4.2) is exercised even when two accepted
    steps are value-equal.
    """
    logic_form = draw(st.text(max_size=40))
    source_text = draw(st.text(max_size=40))
    predicates = draw(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=10),
            values=_predicate_values,
            max_size=4,
        )
    )
    return SymbolicRepresentation(
        logic_form=logic_form,
        predicates=predicates,
        source_text=source_text,
    )


# A non-empty sequence of accepted steps; bounded so the test stays fast.
step_sequences = st.lists(symbolic_representations(), min_size=1, max_size=30)


# ---------------------------------------------------------------------- the property


@settings(max_examples=200)
@given(steps=step_sequences)
def test_declarative_memory_grows_monotonically_and_retains_all(steps):
    """Property 9: Declarative_Memory retains every accepted conclusion, in order.

    Validates: Requirements 4.2, 4.4
    """
    ctrl = ACTRController()
    ctrl.initialize(Goal(description="retention property goal"))

    expected_forms: list[str] = []

    for index, step in enumerate(steps):
        ctrl.integrate_accepted(step)
        expected_forms.append(step.logic_form)

        dm = ctrl.declarative_memory

        # (a) Length equals the number of accepted steps so far: each acceptance adds
        # exactly one entry (monotonic growth by one, no drops, no duplicates dropped).
        assert len(dm) == index + 1

        # (b) Order is preserved and matches insertion order by logic_form: every prior
        # conclusion is still present at its original position (Req 4.4 retention).
        assert [entry.logic_form for entry in dm] == expected_forms

        # (c) Entries are distinct objects: no two Declarative_Memory entries share
        # identity, so the store is composed of distinct entries (Req 4.2).
        ids = [id(entry) for entry in dm]
        assert len(ids) == len(set(ids))

    # After integrating the whole sequence the store holds exactly all conclusions, in
    # the order they were accepted (retention until termination, Req 4.4).
    final_dm = ctrl.declarative_memory
    assert len(final_dm) == len(steps)
    assert [entry.logic_form for entry in final_dm] == [s.logic_form for s in steps]


@settings(max_examples=200)
@given(steps=step_sequences)
def test_mutating_one_entry_does_not_affect_others(steps):
    """Distinct-entry guarantee (Req 4.2): entries are independent objects.

    Mutating any single Declarative_Memory entry (or the caller's original step) must
    not alter any other recorded conclusion. We verify this against the controller's
    internal store via a fresh snapshot taken after mutation.

    Validates: Requirements 4.2, 4.4
    """
    ctrl = ACTRController()
    ctrl.initialize(Goal(description="distinctness property goal"))

    for step in steps:
        ctrl.integrate_accepted(step)

    # Mutate the caller's original step objects after acceptance: because each accepted
    # conclusion is stored as a distinct, independent entry, the recorded conclusions
    # must be unchanged.
    expected_forms = [s.logic_form for s in steps]
    for step in steps:
        step.logic_form = "MUTATED-AFTER-ACCEPT"
        step.predicates["__mutated__"] = True

    dm = ctrl.declarative_memory
    assert [entry.logic_form for entry in dm] == expected_forms
    assert all("__mutated__" not in entry.predicates for entry in dm)

    # Mutating one returned entry must not change any sibling entry either: the entries
    # share no aliased sub-structure.
    if len(dm) >= 2:
        dm[0].predicates["__local_mutation__"] = 123
        siblings = ctrl.declarative_memory[1:]
        assert all("__local_mutation__" not in entry.predicates for entry in siblings)
