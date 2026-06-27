"""System configuration and reproducibility run record."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .learning import LearnedRule, PromotionDecision


@dataclass
class SystemConfig:
    """Runtime parameters read at initialization.

    Documented valid ranges (enforced by the Config Manager in a later task):
    - ``max_cycle_limit``: integer 1..10000
    - ``repair_attempt_limit``: integer 0..1000
    - ``retry_count``: integer 0..1000
    The enumerated fields (``llm_selection``, ``output_format``,
    ``conflict_resolution_policy``) must be one of their documented allowed values.
    """

    max_cycle_limit: int
    repair_attempt_limit: int
    retry_count: int
    llm_selection: str
    output_format: str
    conflict_resolution_policy: str
    generation_timeout_ms: int
    repeated_run_count: int = 1
    latency_budget_ms: Optional[int] = None
    random_seed: Optional[int] = None
    rule_learning_enabled: bool = False
    """Adaptive Rule_Learning enabled state (Req 14.8 default: disabled)."""

    corroboration_threshold: int = 2
    """Required corroborating traces before promotion (Req 14.8 default: 2)."""

    max_learned_rules: int = 64
    """Cap on promoted Learned_Rules (Req 14.8 documented default)."""


@dataclass
class RunRecord:
    """Reproducibility record persisted before the first dataset item is evaluated."""

    config: SystemConfig
    dataset_ids: list[str]
    model_id: str
    seed: int
    """The effective seed in use (supplied or generated)."""

    applied_defaults: dict[str, Any] = field(default_factory=dict)
    learned_rules: list[LearnedRule] = field(default_factory=list)
    """Learned_Rule set persisted for reproducibility (Req 14.6)."""

    induction_seed: Optional[int] = None
    """Effective seed governing induction/promotion (Req 14.6)."""

    corroboration_threshold: Optional[int] = None
    """Corroboration_Threshold in effect for this run (Req 14.6)."""

    promotion_decisions: list[PromotionDecision] = field(default_factory=list)
    """Recorded promote/discard/cap-reached decisions (Req 14.6)."""
