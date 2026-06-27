"""Unit tests for the Config Manager (Task 2.1, Requirement 12).

These cover reading parameters, applying and recording documented defaults, and
halting with a parameter-identifying error for out-of-range numeric values, disallowed
enum values, and unparseable values.
"""

from __future__ import annotations

import pytest

from nsr.config_manager import (
    ALLOWED_CONFLICT_POLICIES,
    ALLOWED_LLM_SELECTIONS,
    ALLOWED_OUTPUT_FORMATS,
    DEFAULTS,
    ConfigError,
    ConfigManager,
    load_config,
)
from nsr.models import SystemConfig


def _valid_raw() -> dict:
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
        "rule_learning_enabled": False,
        "corroboration_threshold": 2,
        "max_learned_rules": 64,
    }


# --- Reading parameters (Req 12.1) ------------------------------------------


def test_reads_all_parameters_from_configuration():
    loaded = load_config(_valid_raw())
    cfg = loaded.config
    assert isinstance(cfg, SystemConfig)
    assert cfg.max_cycle_limit == 100
    assert cfg.repair_attempt_limit == 5
    assert cfg.retry_count == 2
    assert cfg.llm_selection == "gpt-4o"
    assert cfg.output_format == "json"
    assert cfg.conflict_resolution_policy == "priority"
    assert cfg.generation_timeout_ms == 15000
    assert cfg.repeated_run_count == 3
    assert cfg.latency_budget_ms == 2000
    assert cfg.random_seed == 42
    # Nothing was defaulted because every parameter was supplied.
    assert loaded.applied_defaults == {}


# --- Defaults (Req 12.2) ----------------------------------------------------


def test_applies_documented_defaults_when_absent():
    loaded = load_config({})
    cfg = loaded.config
    assert cfg.max_cycle_limit == DEFAULTS["max_cycle_limit"]
    assert cfg.repair_attempt_limit == DEFAULTS["repair_attempt_limit"]
    assert cfg.retry_count == DEFAULTS["retry_count"]
    assert cfg.llm_selection == DEFAULTS["llm_selection"]
    assert cfg.output_format == DEFAULTS["output_format"]
    assert cfg.conflict_resolution_policy == DEFAULTS["conflict_resolution_policy"]
    assert cfg.generation_timeout_ms == DEFAULTS["generation_timeout_ms"]
    assert cfg.repeated_run_count == DEFAULTS["repeated_run_count"]
    assert cfg.latency_budget_ms is None
    assert cfg.random_seed is None


def test_records_only_applied_defaults():
    raw = {"max_cycle_limit": 7, "llm_selection": "local-llama3"}
    loaded = load_config(raw)
    # Supplied parameters are not recorded as defaults.
    assert "max_cycle_limit" not in loaded.applied_defaults
    assert "llm_selection" not in loaded.applied_defaults
    # Absent parameters are recorded with their documented default value.
    assert loaded.applied_defaults["retry_count"] == DEFAULTS["retry_count"]
    assert loaded.applied_defaults["output_format"] == DEFAULTS["output_format"]
    assert loaded.applied_defaults["latency_budget_ms"] is None


def test_none_value_is_treated_as_absent_and_defaulted():
    raw = _valid_raw()
    raw["retry_count"] = None
    loaded = load_config(raw)
    assert loaded.config.retry_count == DEFAULTS["retry_count"]
    assert loaded.applied_defaults["retry_count"] == DEFAULTS["retry_count"]


def test_none_for_optional_parameter_passes_through():
    loaded = load_config(None)
    assert loaded.config.max_cycle_limit == DEFAULTS["max_cycle_limit"]


# --- Out-of-range numeric values (Req 12.3) ---------------------------------


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("max_cycle_limit", 0),
        ("max_cycle_limit", 10001),
        ("repair_attempt_limit", -1),
        ("repair_attempt_limit", 1001),
        ("retry_count", -1),
        ("retry_count", 1001),
        ("generation_timeout_ms", 0),
        ("repeated_run_count", 0),
        ("latency_budget_ms", 0),
        ("random_seed", -1),
    ],
)
def test_out_of_range_numeric_halts_with_named_error(parameter, value):
    raw = _valid_raw()
    raw[parameter] = value
    with pytest.raises(ConfigError) as exc_info:
        load_config(raw)
    assert exc_info.value.parameter == parameter
    assert parameter in str(exc_info.value)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("max_cycle_limit", 1),
        ("max_cycle_limit", 10000),
        ("repair_attempt_limit", 0),
        ("repair_attempt_limit", 1000),
        ("retry_count", 0),
        ("retry_count", 1000),
    ],
)
def test_boundary_values_are_accepted(parameter, value):
    raw = _valid_raw()
    raw[parameter] = value
    loaded = load_config(raw)
    assert getattr(loaded.config, parameter) == value


# --- Disallowed enum values (Req 12.4) --------------------------------------


@pytest.mark.parametrize(
    "parameter",
    ["llm_selection", "output_format", "conflict_resolution_policy"],
)
def test_disallowed_enum_halts_with_named_error(parameter):
    raw = _valid_raw()
    raw[parameter] = "definitely-not-allowed"
    with pytest.raises(ConfigError) as exc_info:
        load_config(raw)
    assert exc_info.value.parameter == parameter


def test_allowed_enum_sets_are_accepted():
    for value in ALLOWED_LLM_SELECTIONS:
        raw = _valid_raw()
        raw["llm_selection"] = value
        assert load_config(raw).config.llm_selection == value
    for value in ALLOWED_OUTPUT_FORMATS:
        raw = _valid_raw()
        raw["output_format"] = value
        assert load_config(raw).config.output_format == value
    for value in ALLOWED_CONFLICT_POLICIES:
        raw = _valid_raw()
        raw["conflict_resolution_policy"] = value
        assert load_config(raw).config.conflict_resolution_policy == value


# --- Unparseable values (Req 12.5) ------------------------------------------


def test_unparseable_integer_string_halts_with_named_error():
    raw = _valid_raw()
    raw["max_cycle_limit"] = "not-a-number"
    with pytest.raises(ConfigError) as exc_info:
        load_config(raw)
    assert exc_info.value.parameter == "max_cycle_limit"


def test_numeric_string_is_parsed():
    raw = _valid_raw()
    raw["retry_count"] = "4"
    assert load_config(raw).config.retry_count == 4


def test_boolean_is_rejected_as_integer():
    raw = _valid_raw()
    raw["retry_count"] = True
    with pytest.raises(ConfigError) as exc_info:
        load_config(raw)
    assert exc_info.value.parameter == "retry_count"


def test_non_integer_float_is_rejected():
    raw = _valid_raw()
    raw["max_cycle_limit"] = 5.5
    with pytest.raises(ConfigError) as exc_info:
        load_config(raw)
    assert exc_info.value.parameter == "max_cycle_limit"


def test_whole_number_float_is_accepted():
    raw = _valid_raw()
    raw["max_cycle_limit"] = 50.0
    assert load_config(raw).config.max_cycle_limit == 50


def test_non_string_enum_is_rejected():
    raw = _valid_raw()
    raw["output_format"] = 123
    with pytest.raises(ConfigError) as exc_info:
        load_config(raw)
    assert exc_info.value.parameter == "output_format"


def test_config_manager_class_matches_module_helper():
    raw = _valid_raw()
    assert ConfigManager().load(raw).config == load_config(raw).config


# --- Adaptive rule-learning parameters (Req 14.8) ---------------------------


def test_rule_learning_defaults_applied_and_recorded_when_absent():
    loaded = load_config({})
    cfg = loaded.config
    assert cfg.rule_learning_enabled is False
    assert cfg.corroboration_threshold == DEFAULTS["corroboration_threshold"] == 2
    assert cfg.max_learned_rules == DEFAULTS["max_learned_rules"] == 64
    # Absent rule-learning parameters are recorded as applied defaults.
    assert loaded.applied_defaults["rule_learning_enabled"] is False
    assert loaded.applied_defaults["corroboration_threshold"] == 2
    assert loaded.applied_defaults["max_learned_rules"] == 64


def test_rule_learning_supplied_values_accepted_and_not_recorded():
    raw = _valid_raw()
    raw["rule_learning_enabled"] = True
    raw["corroboration_threshold"] = 5
    raw["max_learned_rules"] = 1000
    loaded = load_config(raw)
    cfg = loaded.config
    assert cfg.rule_learning_enabled is True
    assert cfg.corroboration_threshold == 5
    assert cfg.max_learned_rules == 1000
    assert "rule_learning_enabled" not in loaded.applied_defaults
    assert "corroboration_threshold" not in loaded.applied_defaults
    assert "max_learned_rules" not in loaded.applied_defaults


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("corroboration_threshold", 1),
        ("corroboration_threshold", 1000),
        ("max_learned_rules", 1),
        ("max_learned_rules", 100000),
    ],
)
def test_rule_learning_boundary_values_are_accepted(parameter, value):
    raw = _valid_raw()
    raw[parameter] = value
    loaded = load_config(raw)
    assert getattr(loaded.config, parameter) == value


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("corroboration_threshold", 0),
        ("corroboration_threshold", 1001),
        ("max_learned_rules", 0),
        ("max_learned_rules", 100001),
    ],
)
def test_rule_learning_out_of_range_halts_with_named_error(parameter, value):
    raw = _valid_raw()
    raw[parameter] = value
    with pytest.raises(ConfigError) as exc_info:
        load_config(raw)
    assert exc_info.value.parameter == parameter
    assert parameter in str(exc_info.value)


@pytest.mark.parametrize("value", [True, False])
def test_rule_learning_enabled_accepts_booleans(value):
    raw = _valid_raw()
    raw["rule_learning_enabled"] = value
    assert load_config(raw).config.rule_learning_enabled is value


@pytest.mark.parametrize(
    "value, expected",
    [
        ("true", True),
        ("True", True),
        ("FALSE", False),
        ("false", False),
    ],
)
def test_rule_learning_enabled_accepts_boolean_strings(value, expected):
    raw = _valid_raw()
    raw["rule_learning_enabled"] = value
    assert load_config(raw).config.rule_learning_enabled is expected


def test_rule_learning_enabled_rejects_unparseable_value():
    raw = _valid_raw()
    raw["rule_learning_enabled"] = "maybe"
    with pytest.raises(ConfigError) as exc_info:
        load_config(raw)
    assert exc_info.value.parameter == "rule_learning_enabled"


def test_corroboration_threshold_unparseable_halts_with_named_error():
    raw = _valid_raw()
    raw["corroboration_threshold"] = "not-a-number"
    with pytest.raises(ConfigError) as exc_info:
        load_config(raw)
    assert exc_info.value.parameter == "corroboration_threshold"
