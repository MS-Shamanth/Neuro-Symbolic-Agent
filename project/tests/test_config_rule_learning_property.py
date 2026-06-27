"""Property-based test for rule-learning configuration defaults and ranges (Task 16.3).

**Property 8: Config applies documented rule-learning defaults.**

For any configuration mapping that omits ``rule_learning_enabled``,
``corroboration_threshold``, or ``max_learned_rules``, the loaded config takes the
documented defaults (``False``, ``2``, and ``64``) and records each applied default in
``applied_defaults``. For any out-of-range ``corroboration_threshold`` or
``max_learned_rules``, loading halts with a parameter-identifying :class:`ConfigError`.

**Validates: Requirements 14.8**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nsr.config_manager import (
    CORROBORATION_THRESHOLD_RANGE,
    MAX_LEARNED_RULES_RANGE,
    ConfigError,
    load_config,
)

# The three rule-learning keys governed by this property and their documented defaults.
_RULE_LEARNING_KEYS = ("rule_learning_enabled", "corroboration_threshold", "max_learned_rules")
_DOCUMENTED_DEFAULTS = {
    "rule_learning_enabled": False,
    "corroboration_threshold": 2,
    "max_learned_rules": 64,
}

# Other valid keys that may freely co-occur in a mapping. The property concerns only the
# rule-learning keys, so any extra keys here are drawn from values known to be valid.
_OTHER_VALID_ENTRIES = st.fixed_dictionaries(
    {},
    optional={
        "max_cycle_limit": st.integers(min_value=1, max_value=10000),
        "retry_count": st.integers(min_value=0, max_value=1000),
        "llm_selection": st.sampled_from(["gpt-4o", "gpt-4o-mini", "local-llama3"]),
        "output_format": st.sampled_from(["json", "logic-form", "yaml"]),
    },
)


@settings(max_examples=200, deadline=None)
@given(extra=_OTHER_VALID_ENTRIES)
def test_documented_defaults_applied_and_recorded_when_keys_absent(extra: dict[str, Any]):
    """When the three keys are absent, defaults (False, 2, 64) are applied and recorded.

    Validates: Requirements 14.8
    """
    # ``extra`` never contains the rule-learning keys, so they are always absent here.
    raw = dict(extra)
    loaded = load_config(raw)

    # Documented defaults are applied to the resulting config.
    assert loaded.config.rule_learning_enabled is False
    assert loaded.config.corroboration_threshold == 2
    assert loaded.config.max_learned_rules == 64

    # ...and each applied default is recorded for reproducibility.
    for key in _RULE_LEARNING_KEYS:
        assert key in loaded.applied_defaults
        assert loaded.applied_defaults[key] == _DOCUMENTED_DEFAULTS[key]


@settings(max_examples=200, deadline=None)
@given(
    corroboration_threshold=st.integers(*CORROBORATION_THRESHOLD_RANGE),
    max_learned_rules=st.integers(*MAX_LEARNED_RULES_RANGE),
    enabled=st.booleans(),
)
def test_in_range_values_are_accepted_and_not_recorded_as_defaults(
    corroboration_threshold: int, max_learned_rules: int, enabled: bool
):
    """Supplied in-range values are accepted verbatim and not recorded as defaults.

    Validates: Requirements 14.8
    """
    raw = {
        "rule_learning_enabled": enabled,
        "corroboration_threshold": corroboration_threshold,
        "max_learned_rules": max_learned_rules,
    }
    loaded = load_config(raw)

    assert loaded.config.rule_learning_enabled is enabled
    assert loaded.config.corroboration_threshold == corroboration_threshold
    assert loaded.config.max_learned_rules == max_learned_rules

    # Present values are never recorded as applied defaults.
    for key in _RULE_LEARNING_KEYS:
        assert key not in loaded.applied_defaults


def _out_of_range(bounds: tuple[int, int]) -> st.SearchStrategy[int]:
    """Integers strictly outside the inclusive ``(low, high)`` range."""
    low, high = bounds
    return st.integers().filter(lambda v: v < low or v > high)


@settings(max_examples=200, deadline=None)
@given(value=_out_of_range(CORROBORATION_THRESHOLD_RANGE))
def test_out_of_range_corroboration_threshold_halts_with_named_error(value: int):
    """Out-of-range corroboration_threshold halts with a parameter-identifying error.

    Validates: Requirements 14.8
    """
    raw = {"corroboration_threshold": value}
    with pytest.raises(ConfigError) as exc_info:
        load_config(raw)
    assert exc_info.value.parameter == "corroboration_threshold"
    assert "corroboration_threshold" in str(exc_info.value)


@settings(max_examples=200, deadline=None)
@given(value=_out_of_range(MAX_LEARNED_RULES_RANGE))
def test_out_of_range_max_learned_rules_halts_with_named_error(value: int):
    """Out-of-range max_learned_rules halts with a parameter-identifying error.

    Validates: Requirements 14.8
    """
    raw = {"max_learned_rules": value}
    with pytest.raises(ConfigError) as exc_info:
        load_config(raw)
    assert exc_info.value.parameter == "max_learned_rules"
    assert "max_learned_rules" in str(exc_info.value)
