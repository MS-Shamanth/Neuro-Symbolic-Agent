"""Property-based test for the lossless Proof_Trace machine-readable round-trip.

Task 8.3 / **Property 4: Proof_Trace machine-readable export round-trips losslessly.**

For any generated :class:`~nsr.models.trace.ProofTrace`, exporting it through the
machine-readable serializers and re-parsing yields a structure equal to the original in
every recorded field. This exercises both the dict form
(``trace_from_dict(trace_to_dict(t)) == t``) and the JSON form
(``trace_from_json(trace_to_json(t)) == t``).

**Validates: Requirements 8.4**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from nsr.models.enums import TerminationReason, ValidationStatus
from nsr.models.reasoning import SymbolicRepresentation
from nsr.models.trace import (
    ErrorRecord,
    LatencyRecord,
    ProofStep,
    ProofTrace,
    RepairAttempt,
)
from nsr.proof_trace_export import (
    trace_from_dict,
    trace_from_json,
    trace_to_dict,
    trace_to_json,
)

# --------------------------------------------------------------------------- #
# Leaf strategies
# --------------------------------------------------------------------------- #

# Bounded, finite-only text and numbers keep the JSON form losslessly comparable:
# NaN/inf are excluded because they either break equality (NaN != NaN) or are not
# valid JSON, and they are never produced by the real pipeline.
_text = st.text(max_size=24)
_rule_ids = st.lists(st.text(min_size=1, max_size=8), max_size=4)

# JSON-compatible scalar values that survive a json.dumps/json.loads round-trip and
# compare equal afterward. Floats are finite to avoid NaN/inf equality pitfalls.
_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**6), max_value=10**6),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    st.text(max_size=16),
)


def _json_values(max_leaves: int = 8):
    """Recursive JSON-compatible values (scalars, lists, string-keyed dicts).

    Dict keys are constrained to strings because JSON only supports string keys; this
    mirrors the structured ``predicates`` / ``translation_outcomes`` schema the real
    pipeline emits, so the generated values are representative and round-trip cleanly.
    """

    return st.recursive(
        _json_scalars,
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.dictionaries(st.text(max_size=8), children, max_size=4),
        ),
        max_leaves=max_leaves,
    )


_predicates = st.dictionaries(st.text(max_size=8), _json_values(), max_size=4)
_translation_outcomes = st.lists(
    st.dictionaries(st.text(max_size=8), _json_values(), max_size=4),
    max_size=3,
)


# --------------------------------------------------------------------------- #
# Composite strategies for the nested trace records
# --------------------------------------------------------------------------- #


@st.composite
def representations(draw) -> SymbolicRepresentation:
    """A nested :class:`SymbolicRepresentation` with arbitrary structured predicates."""

    return SymbolicRepresentation(
        logic_form=draw(_text),
        predicates=draw(_predicates),
        source_text=draw(_text),
    )


@st.composite
def repair_attempts(draw) -> RepairAttempt:
    """A :class:`RepairAttempt`; ``rejected_step`` is non-optional on the dataclass."""

    return RepairAttempt(
        attempt_index=draw(st.integers(min_value=0, max_value=1000)),
        rejected_step=draw(representations()),
        violated_rule_ids=draw(_rule_ids),
        # repaired_step is optional: sometimes absent (unrepaired attempt).
        repaired_step=draw(st.one_of(st.none(), representations())),
    )


@st.composite
def proof_steps(draw) -> ProofStep:
    """A :class:`ProofStep` spanning all ValidationStatus values and optional fields."""

    return ProofStep(
        sequence=draw(st.integers(min_value=0, max_value=10_000)),
        step_text=draw(_text),
        # representation is Optional: exercise both the present and None branches.
        representation=draw(st.one_of(st.none(), representations())),
        status=draw(st.sampled_from(list(ValidationStatus))),
        applied_rule_id=draw(st.one_of(st.none(), st.text(min_size=1, max_size=8))),
        violated_rule_ids=draw(_rule_ids),
        repair_attempts=draw(st.lists(repair_attempts(), max_size=3)),
        translation_outcomes=draw(_translation_outcomes),
    )


@st.composite
def latency_records(draw) -> LatencyRecord:
    """A :class:`LatencyRecord` with finite millisecond fields."""

    ms = st.floats(
        min_value=0.0, max_value=1e7, allow_nan=False, allow_infinity=False
    )
    return LatencyRecord(
        pipeline_ms=draw(ms),
        system2_ms=draw(ms),
        llm_ms=draw(ms),
        latency_budget_exceeded=draw(st.booleans()),
    )


@st.composite
def error_records(draw) -> ErrorRecord:
    """An :class:`ErrorRecord` identifying a failed component and reason."""

    return ErrorRecord(
        failed_component=draw(_text),
        reason=draw(_text),
    )


@st.composite
def proof_traces(draw) -> ProofTrace:
    """An arbitrary :class:`ProofTrace` exercising every recorded field.

    Varied steps (all ValidationStatus values), nested representations with predicates,
    repair attempts, translation outcomes, an optional latency record, an optional
    termination reason, and an optional error record.
    """

    return ProofTrace(
        steps=draw(st.lists(proof_steps(), max_size=5)),
        termination_reason=draw(
            st.one_of(st.none(), st.sampled_from(list(TerminationReason)))
        ),
        latency=draw(st.one_of(st.none(), latency_records())),
        error_record=draw(st.one_of(st.none(), error_records())),
    )


# --------------------------------------------------------------------------- #
# Property 4: lossless machine-readable round-trip
# --------------------------------------------------------------------------- #


@settings(max_examples=200)
@given(trace=proof_traces())
def test_dict_round_trip_is_lossless(trace: ProofTrace) -> None:
    """Property 4 (dict form): ``trace_from_dict(trace_to_dict(t)) == t``.

    **Validates: Requirements 8.4**
    """

    assert trace_from_dict(trace_to_dict(trace)) == trace


@settings(max_examples=200)
@given(trace=proof_traces())
def test_json_round_trip_is_lossless(trace: ProofTrace) -> None:
    """Property 4 (JSON form): ``trace_from_json(trace_to_json(t)) == t``.

    **Validates: Requirements 8.4**
    """

    assert trace_from_json(trace_to_json(trace)) == trace
