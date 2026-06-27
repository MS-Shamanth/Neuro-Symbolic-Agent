"""Reasoning quality and per-method evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class QueryMetrics:
    """Metrics computed from a single terminated Proof_Trace."""

    faithfulness_score: float
    """accepted / total; 0.0 for an empty trace."""

    step_hallucination_rate: float
    """rejected / total."""

    reasoning_consistency: Optional[float] = None
    """None when the repeated-run count is less than 2."""


@dataclass
class MethodMetrics:
    """Aggregate metrics for a single method over an evaluation run."""

    method: str
    final_answer_accuracy: float
    step_hallucination_rate: float
    faithfulness_score: float
    latency_overhead_ms: float
    mean_latency_ms: float
    p95_latency_ms: float
    reasoning_consistency: Optional[float] = None
