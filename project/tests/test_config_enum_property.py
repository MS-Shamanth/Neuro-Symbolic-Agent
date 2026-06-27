"""Property-based test for enum configuration validation (Task 2.3).

Property 6: Disallowed enum values are always rejected.

For any LLM selection, output format, or conflict-resolution policy value not in the
allowed set, initialization halts with a parameter-identifying error.

**Validates: Requirements 12.4**
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nsr.config_manager import (
    ALLOWED_CONFLICT_POLICIES,
    ALLOWED_LLM_SELECTIONS,
    ALLOWED_OUTPUT_FORMATS,
    ConfigError,
    load_config,
)

# The three enumerated parameters and their documented allowed sets (Req 12.4).
_ENUM_PARAMETERS: dict[str, frozenset[str]] = {
    "llm_selection": ALLOWED_LLM_SELECTIONS,
    "output_format": ALLOWED_OUTPUT_FORMATS,
    "conflict_resolution_policy": ALLOWED_CONFLICT_POLICIES,
}

# Every allowed value across all enum parameters; used to exclude accidental hits.
_ALL_ALLOWED: frozenset[str] = (
    ALLOWED_LLM_SELECTIONS | ALLOWED_OUTPUT_FORMATS | ALLOWED_CONFLICT_POLICIES
)


def _valid_raw() -> dict:
    """A fully valid configuration mapping that initializes without error."""
    return {
        "max_cycle_limit": 100,
        "repair_attempt_limit": 5,
        "retry_count": 2,
        "llm_selection": "gpt-4o",
        "output_format": "json",
        "conflict_resolution_policy": "priority",
        "generation_timeout_ms": 15000,
        "repeated_run_count": 3,
        "latency_budget_ms": 2000,
        "random_seed": 42,
    }


# A string that is guaranteed not to be any allowed enum value. We filter out the rare
# generated string that happens to collide with an allowed value for any parameter.
_disallowed_strings = st.text().filter(lambda s: s not in _ALL_ALLOWED)


@given(parameter=st.sampled_from(sorted(_ENUM_PARAMETERS)), value=_disallowed_strings)
def test_disallowed_enum_value_is_always_rejected(parameter: str, value: str) -> None:
    """Any out-of-set string for an enum parameter halts with a parameter-naming error.

    **Validates: Requirements 12.4**
    """
    raw = _valid_raw()
    raw[parameter] = value
    with pytest.raises(ConfigError) as exc_info:
        load_config(raw)
    # The error must identify the offending parameter (Req 12.4).
    assert exc_info.value.parameter == parameter
    assert parameter in str(exc_info.value)


@given(value=_disallowed_strings)
def test_disallowed_llm_selection_is_rejected(value: str) -> None:
    """Disallowed LLM selection halts naming ``llm_selection``.

    **Validates: Requirements 12.4**
    """
    raw = _valid_raw()
    raw["llm_selection"] = value
    with pytest.raises(ConfigError) as exc_info:
        load_config(raw)
    assert exc_info.value.parameter == "llm_selection"


@given(value=_disallowed_strings)
def test_disallowed_output_format_is_rejected(value: str) -> None:
    """Disallowed output format halts naming ``output_format``.

    **Validates: Requirements 12.4**
    """
    raw = _valid_raw()
    raw["output_format"] = value
    with pytest.raises(ConfigError) as exc_info:
        load_config(raw)
    assert exc_info.value.parameter == "output_format"


@given(value=_disallowed_strings)
def test_disallowed_conflict_policy_is_rejected(value: str) -> None:
    """Disallowed conflict-resolution policy halts naming ``conflict_resolution_policy``.

    **Validates: Requirements 12.4**
    """
    raw = _valid_raw()
    raw["conflict_resolution_policy"] = value
    with pytest.raises(ConfigError) as exc_info:
        load_config(raw)
    assert exc_info.value.parameter == "conflict_resolution_policy"


@given(
    llm=st.sampled_from(sorted(ALLOWED_LLM_SELECTIONS)),
    fmt=st.sampled_from(sorted(ALLOWED_OUTPUT_FORMATS)),
    policy=st.sampled_from(sorted(ALLOWED_CONFLICT_POLICIES)),
)
def test_allowed_enum_combinations_always_succeed(
    llm: str, fmt: str, policy: str
) -> None:
    """Conversely, every allowed enum combination initializes successfully.

    **Validates: Requirements 12.4**
    """
    raw = _valid_raw()
    raw["llm_selection"] = llm
    raw["output_format"] = fmt
    raw["conflict_resolution_policy"] = policy
    loaded = load_config(raw)
    assert loaded.config.llm_selection == llm
    assert loaded.config.output_format == fmt
    assert loaded.config.conflict_resolution_policy == policy
