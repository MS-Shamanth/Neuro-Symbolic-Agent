"""Config Manager for the Neuro-Symbolic System-2 Reasoning Architecture.

Implements Requirement 12 (Configurability):

- Reads every runtime parameter from a configuration mapping at initialization
  (Req 12.1).
- Applies documented defaults for absent values and records which defaults were
  applied (Req 12.2).
- Halts initialization with a parameter-identifying error for out-of-range numeric
  values (Req 12.3), disallowed enumerated values (Req 12.4), or values that cannot be
  parsed as the documented type (Req 12.5).

The resulting :class:`~nsr.models.SystemConfig` is returned together with the
``applied_defaults`` mapping, which callers may store on
:attr:`~nsr.models.RunRecord.applied_defaults` for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .models import SystemConfig

# ---------------------------------------------------------------------------
# Documented allowed enumerated values (Req 12.4)
# ---------------------------------------------------------------------------

#: Allowed LLM selections. Hosted-API and local-runtime backends are distinguished
#: elsewhere; here we only constrain the selection identifier to a documented set.
ALLOWED_LLM_SELECTIONS: frozenset[str] = frozenset(
    {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "local-llama3",
        "local-mistral",
    }
)

#: Allowed structured output formats for constrained decoding.
ALLOWED_OUTPUT_FORMATS: frozenset[str] = frozenset({"json", "logic-form", "yaml"})

#: Allowed deterministic conflict-resolution policies for ACT-R rule selection.
ALLOWED_CONFLICT_POLICIES: frozenset[str] = frozenset(
    {"priority", "specificity", "recency"}
)


# ---------------------------------------------------------------------------
# Documented numeric ranges (Req 12.3). (min, max) inclusive.
# ---------------------------------------------------------------------------

MAX_CYCLE_LIMIT_RANGE = (1, 10000)
REPAIR_ATTEMPT_LIMIT_RANGE = (0, 1000)
RETRY_COUNT_RANGE = (0, 1000)
GENERATION_TIMEOUT_MS_RANGE = (1, 3_600_000)
REPEATED_RUN_COUNT_RANGE = (1, 10000)
LATENCY_BUDGET_MS_RANGE = (1, 3_600_000)
RANDOM_SEED_RANGE = (0, 2**32 - 1)
CORROBORATION_THRESHOLD_RANGE = (1, 1000)
MAX_LEARNED_RULES_RANGE = (1, 100000)


# ---------------------------------------------------------------------------
# Documented defaults (Req 12.2)
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "max_cycle_limit": 50,
    "repair_attempt_limit": 3,
    "retry_count": 3,
    "llm_selection": "gpt-4o-mini",
    "output_format": "json",
    "conflict_resolution_policy": "specificity",
    "generation_timeout_ms": 30000,
    "repeated_run_count": 1,
    "latency_budget_ms": None,
    "random_seed": None,
    "rule_learning_enabled": False,
    "corroboration_threshold": 2,
    "max_learned_rules": 64,
}


class ConfigError(ValueError):
    """Raised when a configuration value is invalid.

    Always identifies the offending ``parameter`` so initialization can be halted with
    a parameter-identifying error (Req 12.3, 12.4, 12.5).
    """

    def __init__(self, parameter: str, message: str) -> None:
        self.parameter = parameter
        super().__init__(f"Invalid configuration for '{parameter}': {message}")


@dataclass
class LoadedConfig:
    """Result of loading configuration: the validated config plus applied defaults."""

    config: SystemConfig
    applied_defaults: dict[str, Any]


def _parse_int(parameter: str, value: Any) -> int:
    """Coerce ``value`` to an ``int`` or raise a parameter-identifying error (Req 12.5).

    Booleans are rejected (``bool`` is a subclass of ``int`` in Python but is never a
    valid numeric configuration value here). Whole-number floats and numeric strings
    are accepted; everything else is treated as unparseable.
    """
    if isinstance(value, bool):
        raise ConfigError(parameter, f"expected an integer, got boolean {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ConfigError(parameter, f"expected an integer, got non-integer float {value!r}")
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError as exc:
            raise ConfigError(
                parameter, f"could not be parsed as an integer (got {value!r})"
            ) from exc
    raise ConfigError(
        parameter, f"could not be parsed as an integer (got {type(value).__name__})"
    )


def _parse_bool(parameter: str, value: Any) -> bool:
    """Coerce ``value`` to a ``bool`` or raise a parameter-identifying error (Req 12.5).

    Genuine booleans pass through. For convenience, the case-insensitive strings
    ``"true"``/``"false"`` (and their common synonyms ``"1"``/``"0"``,
    ``"yes"``/``"no"``) are accepted, mirroring the lenient string handling used for
    numeric parameters. All other values are treated as unparseable and halt loading
    with an error naming ``parameter``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    raise ConfigError(
        parameter, f"could not be parsed as a boolean (got {value!r})"
    )


def _check_range(parameter: str, value: int, bounds: tuple[int, int]) -> int:
    """Validate an integer against an inclusive ``(low, high)`` range (Req 12.3)."""
    low, high = bounds
    if value < low or value > high:
        raise ConfigError(
            parameter,
            f"value {value} is outside the permitted range [{low}, {high}]",
        )
    return value


def _check_enum(parameter: str, value: Any, allowed: frozenset[str]) -> str:
    """Validate an enumerated value against its documented allowed set (Req 12.4)."""
    if not isinstance(value, str):
        raise ConfigError(
            parameter,
            f"expected a string, got {type(value).__name__}",
        )
    if value not in allowed:
        permitted = ", ".join(sorted(allowed))
        raise ConfigError(
            parameter,
            f"value {value!r} is not one of the allowed values: {{{permitted}}}",
        )
    return value


def _resolve(
    raw: Mapping[str, Any],
    applied_defaults: dict[str, Any],
    parameter: str,
) -> Any:
    """Return the raw value for ``parameter`` or its default, recording the default.

    A value is considered absent when the key is missing or maps to ``None`` (Req 12.2).
    """
    if parameter in raw and raw[parameter] is not None:
        return raw[parameter]
    default = DEFAULTS[parameter]
    applied_defaults[parameter] = default
    return default


class ConfigManager:
    """Reads, defaults, and validates runtime parameters at initialization (Req 12)."""

    def load(self, raw: Optional[Mapping[str, Any]] = None) -> LoadedConfig:
        """Build a validated :class:`SystemConfig` from a configuration mapping.

        Absent values fall back to documented defaults, which are recorded in
        ``applied_defaults``. Out-of-range numeric values, disallowed enum values, and
        unparseable values halt loading by raising :class:`ConfigError` that identifies
        the offending parameter.
        """
        raw = raw or {}
        applied_defaults: dict[str, Any] = {}

        # --- Numeric parameters with documented ranges (Req 12.3, 12.5) ---
        max_cycle_limit = _check_range(
            "max_cycle_limit",
            _parse_int("max_cycle_limit", _resolve(raw, applied_defaults, "max_cycle_limit")),
            MAX_CYCLE_LIMIT_RANGE,
        )
        repair_attempt_limit = _check_range(
            "repair_attempt_limit",
            _parse_int(
                "repair_attempt_limit",
                _resolve(raw, applied_defaults, "repair_attempt_limit"),
            ),
            REPAIR_ATTEMPT_LIMIT_RANGE,
        )
        retry_count = _check_range(
            "retry_count",
            _parse_int("retry_count", _resolve(raw, applied_defaults, "retry_count")),
            RETRY_COUNT_RANGE,
        )
        generation_timeout_ms = _check_range(
            "generation_timeout_ms",
            _parse_int(
                "generation_timeout_ms",
                _resolve(raw, applied_defaults, "generation_timeout_ms"),
            ),
            GENERATION_TIMEOUT_MS_RANGE,
        )
        repeated_run_count = _check_range(
            "repeated_run_count",
            _parse_int(
                "repeated_run_count",
                _resolve(raw, applied_defaults, "repeated_run_count"),
            ),
            REPEATED_RUN_COUNT_RANGE,
        )

        # --- Optional numeric parameters (absent stays None, not defaulted further) ---
        latency_budget_ms = self._optional_int(
            raw, applied_defaults, "latency_budget_ms", LATENCY_BUDGET_MS_RANGE
        )
        random_seed = self._optional_int(
            raw, applied_defaults, "random_seed", RANDOM_SEED_RANGE
        )

        # --- Enumerated parameters (Req 12.4) ---
        llm_selection = _check_enum(
            "llm_selection",
            _resolve(raw, applied_defaults, "llm_selection"),
            ALLOWED_LLM_SELECTIONS,
        )
        output_format = _check_enum(
            "output_format",
            _resolve(raw, applied_defaults, "output_format"),
            ALLOWED_OUTPUT_FORMATS,
        )
        conflict_resolution_policy = _check_enum(
            "conflict_resolution_policy",
            _resolve(raw, applied_defaults, "conflict_resolution_policy"),
            ALLOWED_CONFLICT_POLICIES,
        )

        # --- Adaptive rule-learning parameters (Req 14.8) ---
        rule_learning_enabled = _parse_bool(
            "rule_learning_enabled",
            _resolve(raw, applied_defaults, "rule_learning_enabled"),
        )
        corroboration_threshold = _check_range(
            "corroboration_threshold",
            _parse_int(
                "corroboration_threshold",
                _resolve(raw, applied_defaults, "corroboration_threshold"),
            ),
            CORROBORATION_THRESHOLD_RANGE,
        )
        max_learned_rules = _check_range(
            "max_learned_rules",
            _parse_int(
                "max_learned_rules",
                _resolve(raw, applied_defaults, "max_learned_rules"),
            ),
            MAX_LEARNED_RULES_RANGE,
        )

        config = SystemConfig(
            max_cycle_limit=max_cycle_limit,
            repair_attempt_limit=repair_attempt_limit,
            retry_count=retry_count,
            llm_selection=llm_selection,
            output_format=output_format,
            conflict_resolution_policy=conflict_resolution_policy,
            generation_timeout_ms=generation_timeout_ms,
            repeated_run_count=repeated_run_count,
            latency_budget_ms=latency_budget_ms,
            random_seed=random_seed,
            rule_learning_enabled=rule_learning_enabled,
            corroboration_threshold=corroboration_threshold,
            max_learned_rules=max_learned_rules,
        )
        return LoadedConfig(config=config, applied_defaults=applied_defaults)

    @staticmethod
    def _optional_int(
        raw: Mapping[str, Any],
        applied_defaults: dict[str, Any],
        parameter: str,
        bounds: tuple[int, int],
    ) -> Optional[int]:
        """Resolve an optional integer parameter.

        When absent, the documented default (``None``) is applied and recorded. When
        present, it is parsed and range-checked like any other numeric parameter.
        """
        if parameter not in raw or raw[parameter] is None:
            applied_defaults[parameter] = DEFAULTS[parameter]
            return DEFAULTS[parameter]
        return _check_range(parameter, _parse_int(parameter, raw[parameter]), bounds)


def load_config(raw: Optional[Mapping[str, Any]] = None) -> LoadedConfig:
    """Module-level convenience wrapper around :meth:`ConfigManager.load`."""
    return ConfigManager().load(raw)
