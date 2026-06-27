"""Core data models for the Neuro-Symbolic System-2 Reasoning Architecture.

This package gathers every enum and dataclass defined in the design's Data Models
section and re-exports them so callers can ``from nsr.models import ...`` directly.
"""

from __future__ import annotations

from .config import RunRecord, SystemConfig
from .dataset import DatasetItem
from .enums import Domain, TerminationReason, ValidationStatus
from .learning import (
    CandidateRule,
    DiscardedCandidate,
    LearnedRule,
    LearnedRuleStore,
    PromotionDecision,
    PromotionResult,
    RuleOrigin,
    RuleProvenance,
    store_from_dict,
    store_to_dict,
)
from .memory import WorkingMemoryState
from .metrics import MethodMetrics, QueryMetrics
from .reasoning import Goal, ProductionRule, SubGoal, SymbolicRepresentation
from .translation import (
    BackTranslationError,
    CandidateStep,
    PromptContext,
    Untranslatable,
)
from .trace import (
    ErrorRecord,
    LatencyRecord,
    ProofStep,
    ProofTrace,
    RepairAttempt,
    VerifiedOutput,
)

__all__ = [
    # enums
    "ValidationStatus",
    "TerminationReason",
    "Domain",
    # reasoning
    "Goal",
    "SubGoal",
    "SymbolicRepresentation",
    "ProductionRule",
    # working memory
    "WorkingMemoryState",
    # translation-layer boundary types
    "CandidateStep",
    "PromptContext",
    "Untranslatable",
    "BackTranslationError",
    # proof trace / results
    "RepairAttempt",
    "ProofStep",
    "LatencyRecord",
    "ProofTrace",
    "ErrorRecord",
    "VerifiedOutput",
    # metrics
    "QueryMetrics",
    "MethodMetrics",
    # dataset
    "DatasetItem",
    # config / run record
    "SystemConfig",
    "RunRecord",
    # adaptive rule learning (Req 14)
    "RuleOrigin",
    "RuleProvenance",
    "CandidateRule",
    "LearnedRule",
    "DiscardedCandidate",
    "PromotionDecision",
    "PromotionResult",
    "LearnedRuleStore",
    "store_to_dict",
    "store_from_dict",
]
