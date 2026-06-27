"""Property-based tests for numeric configuration range validation (Task 2.2).

Property 5: Out-of-range numeric config is always rejected.

For any integer outside the documented range, initialization halts with a
``ConfigError`` naming the offending parameter; any in-range integer initializes
successfully.

Documented numeric ranges (inclusive):
    - ``max_cycle_limit``     : 1 .. 10000
    - ``repair_attempt_limit``: 0 .. 1000
    - ``retry_count``         : 0 .. 1000

**Validates: Requirements 12.3**
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nsr.config_manager import ConfigError, ConfigManager, load_config
from nsr.models import SystemConfig

# The three numeric parameters under test and their documented inclusive ranges.
NUMERIC_RANGES: dict[str, tuple[int, int]] = {
    "max_cycle_limit": (1, 10000),
    "repair_attempt_limit": (0, 1000),
    "retry_count": (0, 1000),
}

# A baseline configuration whose every value is valid. Each property overrides exactly
# one numeric parameter so that any rejection is attributable to that parameter alone.
BASE_VALID_CONFIG: dict[str, object] = {
    "max_cycle_limit": 50,
    "repair_attempt_limit": 3,
    "retry_count": 3,
    "llm_selection": "gpt-4o-mini",
    "output_format": "json",
    "conflict_resolution_policy": "specificity",
    "generation_timeout_ms": 30000,
    "repeated_run_count": 1,
}


def _below_range(param: str) -> st.SearchStrategy[int]:
    """Integers strictly below the parameter's documented lower bound."""
    low, _high = NUMERIC_RANGES[param]
    return st.integers(max_value=low - 1)


def _above_range(param: str) -> st.SearchStrategy[int]:
    """Integers strictly above the parameter's documented upper bound."""
    _low, high = NUMERIC_RANGES[param]
    return st.integers(min_value=high + 1)


def _in_range(param: str) -> st.SearchStrategy[int]:
    """Integers within the parameter's documented inclusive range."""
    low, high = NUMERIC_RANGES[param]
    return st.integers(min_value=low, max_value=high)


def _config_with(param: str, value: int) -> dict[str, object]:
    cfg = dict(BASE_VALID_CONFIG)
    cfg[param] = value
    return cfg


# ---------------------------------------------------------------------------
# Out-of-range integers are always rejected with a parameter-identifying error.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("param", list(NUMERIC_RANGES))
@given(data=st.data())
def test_out_of_range_numeric_config_is_rejected(param: str, data: st.DataObject) -> None:
    """Any integer outside the documented range halts loading naming the parameter."""
    # Draw an out-of-range value from either below or above the range.
    value = data.draw(st.one_of(_below_range(param), _above_range(param)))

    with pytest.raises(ConfigError) as exc_info:
        ConfigManager().load(_config_with(param, value))

    # The error must identify the offending parameter (Req 12.3).
    assert exc_info.value.parameter == param
    assert param in str(exc_info.value)


# ---------------------------------------------------------------------------
# In-range integers always initialize successfully.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("param", list(NUMERIC_RANGES))
@given(data=st.data())
def test_in_range_numeric_config_initializes(param: str, data: st.DataObject) -> None:
    """Any in-range integer yields a SystemConfig carrying that value."""
    value = data.draw(_in_range(param))

    loaded = ConfigManager().load(_config_with(param, value))

    assert isinstance(loaded.config, SystemConfig)
    assert getattr(loaded.config, param) == value


@given(
    max_cycle_limit=_in_range("max_cycle_limit"),
    repair_attempt_limit=_in_range("repair_attempt_limit"),
    retry_count=_in_range("retry_count"),
)
def test_all_in_range_together_initialize(
    max_cycle_limit: int, repair_attempt_limit: int, retry_count: int
) -> None:
    """All three numeric parameters in range simultaneously initialize successfully."""
    cfg = dict(BASE_VALID_CONFIG)
    cfg["max_cycle_limit"] = max_cycle_limit
    cfg["repair_attempt_limit"] = repair_attempt_limit
    cfg["retry_count"] = retry_count

    loaded = load_config(cfg)

    assert loaded.config.max_cycle_limit == max_cycle_limit
    assert loaded.config.repair_attempt_limit == repair_attempt_limit
    assert loaded.config.retry_count == retry_count
