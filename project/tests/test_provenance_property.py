"""Property-based test for Candidate_Rule provenance traceability (Task 17.3).

Property 2: Candidates are provenance-traceable.

For any :class:`CandidateRule` produced by :meth:`RuleLearner.induce` over a
goal-satisfied trace, its :class:`RuleProvenance` records:

- ``trace_ids`` equal to exactly the one source trace id supplied to ``induce``,
- a non-empty ``step_ids`` list, and
- every referenced ``step_id`` is the ``sequence`` of an accepted (or
  accepted-after-repair) ``ProofStep`` that is actually present in the source trace.

This is the basis for later corroboration/promotion logging: a learned generalization
must always be traceable back to the concrete accepted steps it was induced from.

**Validates: Requirements 14.2**
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from nsr import RuleLearner, ValidationEngine
from nsr.models import LearnedRuleStore, SymbolicRepresentation
from nsr.models.enums import TerminationReason, ValidationStatus
from nsr.models.trace import ProofStep, ProofTrace

# Statuses that make a step a valid induction source (Req 14.1): a directly accepted
# step and a step accepted only after repair both contribute to a generalization.
_ACCEPTED_STATUSES = [ValidationStatus.ACCEPTED, ValidationStatus.REPAIRED]

# A small token alphabet keeps logic forms (and therefore generalizations) overlapping
# enough that distinct accepted steps frequently collapse into a shared candidate,
# exercising the multi-step provenance path as well as the single-step one.
_TOKENS = ["sum", "rel", "eq", "add", "mul", "p", "q", "x", "y", "z"]

_PREDICATE_KEYS = ["operation", "operands", "subject", "relation", "value", "k"]


def _logic_forms() -> st.SearchStrategy[str]:
    """Identifier-ish logic forms, optionally with instance-specific numeric literals."""
    return st.builds(
        lambda head, args: f"{head}({','.join(args)})",
        head=st.sampled_from(_TOKENS),
        args=st.lists(
            st.one_of(st.sampled_from(_TOKENS), st.integers(0, 9).map(str)),
            min_size=0,
            max_size=3,
        ),
    )


def _representations() -> st.SearchStrategy[SymbolicRepresentation]:
    return st.builds(
        SymbolicRepresentation,
        logic_form=_logic_forms(),
        predicates=st.dictionaries(
            keys=st.sampled_from(_PREDICATE_KEYS),
            values=st.one_of(st.integers(0, 9), st.sampled_from(_TOKENS)),
            max_size=4,
        ),
        source_text=st.text(max_size=12),
    )


# Non-empty trace ids: ``induce`` rejects an empty trace_id, so the generator must never
# produce one.
_TRACE_IDS = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
    max_size=16,
)


@st.composite
def goal_satisfied_traces(draw: st.DrawFn) -> ProofTrace:
    """A goal-satisfied trace with unique step sequences and >= 1 accepted step.

    Each step gets a distinct ``sequence`` (its index) so that a referenced provenance
    ``step_id`` maps unambiguously back to a single source step. At least one step is
    forced to an accepted/repaired status so that induction always yields candidates.
    """
    n = draw(st.integers(min_value=1, max_value=8))
    statuses = draw(
        st.lists(st.sampled_from(list(ValidationStatus)), min_size=n, max_size=n)
    )
    # Force one step to be an induction source so the trace is genuinely learnable.
    forced_index = draw(st.integers(min_value=0, max_value=n - 1))
    statuses[forced_index] = draw(st.sampled_from(_ACCEPTED_STATUSES))

    steps = [
        ProofStep(
            sequence=sequence,
            step_text=f"step-{sequence}",
            representation=draw(_representations()),
            status=status,
        )
        for sequence, status in enumerate(statuses)
    ]
    return ProofTrace(
        steps=steps, termination_reason=TerminationReason.GOAL_SATISFIED
    )


def _learner(seed: int) -> RuleLearner:
    return RuleLearner(LearnedRuleStore(), ValidationEngine(), seed=seed)


@given(
    trace=goal_satisfied_traces(),
    trace_id=_TRACE_IDS,
    seed=st.integers(min_value=0, max_value=1024),
)
def test_candidates_are_provenance_traceable(
    trace: ProofTrace, trace_id: str, seed: int
) -> None:
    """Every induced candidate is traceable to accepted steps of its source trace."""
    learner = _learner(seed)

    candidates = learner.induce(trace, trace_id=trace_id)

    # The trace is goal-satisfied with at least one accepted step, so induction must
    # produce at least one candidate to be traceable.
    assert candidates, "expected at least one candidate from a goal-satisfied trace"

    # The sequences of the steps that are valid induction sources in this trace.
    accepted_sequences = {
        step.sequence
        for step in trace.steps
        if step.status in _ACCEPTED_STATUSES and step.representation is not None
    }

    for candidate in candidates:
        provenance = candidate.provenance

        # trace_ids points to exactly the one source trace.
        assert provenance.trace_ids == [trace_id]

        # step_ids is non-empty: a candidate always generalizes >= 1 accepted step.
        assert provenance.step_ids, "candidate provenance must reference >= 1 step"

        # Every referenced step id is the sequence of an accepted/repaired step that is
        # actually present in the source trace.
        for step_id in provenance.step_ids:
            assert step_id in accepted_sequences

        # The seed is faithfully recorded for reproducibility (Req 14.2/14.6).
        assert provenance.induction_seed == seed


@given(
    trace=goal_satisfied_traces(),
    trace_id=_TRACE_IDS,
    seed=st.integers(min_value=0, max_value=1024),
)
def test_provenance_step_ids_partition_accepted_steps(
    trace: ProofTrace, trace_id: str, seed: int
) -> None:
    """Collected provenance step ids are exactly the trace's accepted step sequences.

    No accepted step is dropped and no non-accepted/absent step is invented: the union
    of all candidates' ``step_ids`` equals the set of accepted-step sequences.
    """
    learner = _learner(seed)

    candidates = learner.induce(trace, trace_id=trace_id)

    accepted_sequences = {
        step.sequence
        for step in trace.steps
        if step.status in _ACCEPTED_STATUSES and step.representation is not None
    }
    referenced = {
        step_id
        for candidate in candidates
        for step_id in candidate.provenance.step_ids
    }

    assert referenced == accepted_sequences
