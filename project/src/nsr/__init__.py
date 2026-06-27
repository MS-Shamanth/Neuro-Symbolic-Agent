"""Neuro-Symbolic System-2 Reasoning Architecture (nsr).

A hybrid inference system pairing a neural LLM (System 1) with an ACT-R-style symbolic
controller (System 2) that performs step-level symbolic validation inside a closed
dual-process loop.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .actr_controller import ACTRController, NoRuleMatched
from .dataset_loader import (
    DatasetLoader,
    DatasetLoadResult,
    DomainStats,
    ExclusionRecord,
    LoadReport,
    load_dataset,
)
from .orchestrator import (
    CycleStage,
    PipelineOrchestrator,
    PipelineResult,
    STAGE_ORDER,
    parse_query,
)
from .rule_learner import RuleLearner
from .validation_engine import RuleEvaluation, ValidationEngine, ValidationOutcome

__all__ = [
    "ACTRController",
    "NoRuleMatched",
    "RuleLearner",
    "ValidationEngine",
    "ValidationOutcome",
    "RuleEvaluation",
    "DatasetLoader",
    "DatasetLoadResult",
    "DomainStats",
    "ExclusionRecord",
    "LoadReport",
    "load_dataset",
    "PipelineOrchestrator",
    "PipelineResult",
    "CycleStage",
    "STAGE_ORDER",
    "parse_query",
    "__version__",
]
