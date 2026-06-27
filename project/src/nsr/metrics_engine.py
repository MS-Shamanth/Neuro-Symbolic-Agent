"""Metrics Engine: per-query reasoning-quality metrics and run consistency.

This module computes the quantitative metrics defined in the design's *Metrics Engine*
component from a terminated :class:`~nsr.models.trace.ProofTrace`:

- **Faithfulness_Score** -- the fraction of Reasoning_Steps marked ``ACCEPTED`` out of
  the total number of steps. An empty trace yields exactly ``0.0`` -- Req 7.1, 7.2.
- **Step_Level_Hallucination_Rate** -- the fraction of Reasoning_Steps marked
  ``REJECTED`` out of the total number of steps -- Req 7.3.
- **Reasoning_Consistency** -- the fraction of repeated runs whose final answer equals
  the modal final answer, computed only when the repeated-run count is 2 or greater;
  otherwise left unset (``None``) -- Req 7.4, 7.5.

Both per-query rates are guaranteed to lie in the closed interval ``[0.0, 1.0]``.

Step counting follows the design's :class:`~nsr.models.enums.ValidationStatus`:
``ACCEPTED`` steps count toward faithfulness, ``REJECTED`` steps count toward the
hallucination rate, and ``REPAIRED`` is a distinct terminal status that counts toward
neither numerator. In every case the denominator is the total number of steps recorded
in the trace, so the two rates are well defined and bounded.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

from .models.enums import ValidationStatus
from .models.metrics import QueryMetrics
from .models.trace import ProofTrace


def compute_faithfulness_score(trace: ProofTrace) -> float:
    """Return accepted/total over the trace's steps, or ``0.0`` for an empty trace.

    The result is guaranteed to lie in ``[0.0, 1.0]`` because the accepted count can
    never exceed the total step count (Req 7.1, 7.2).
    """
    total = len(trace.steps)
    if total == 0:
        return 0.0
    accepted = sum(1 for step in trace.steps if step.status == ValidationStatus.ACCEPTED)
    return accepted / total


def compute_step_hallucination_rate(trace: ProofTrace) -> float:
    """Return rejected/total over the trace's steps, or ``0.0`` for an empty trace.

    The result is guaranteed to lie in ``[0.0, 1.0]`` because the rejected count can
    never exceed the total step count (Req 7.3).
    """
    total = len(trace.steps)
    if total == 0:
        return 0.0
    rejected = sum(1 for step in trace.steps if step.status == ValidationStatus.REJECTED)
    return rejected / total


def compute_reasoning_consistency(
    run_answers: Sequence[str],
    repeated_run_count: int,
) -> Optional[float]:
    """Return the modal-answer fraction across repeated runs, or ``None``.

    Reasoning_Consistency is computed only when ``repeated_run_count`` is 2 or greater;
    otherwise it is left unset and ``None`` is returned (Req 7.4, 7.5). When computed,
    the value is the count of runs whose final answer equals the most common
    (modal) final answer divided by the total number of runs, which lies in
    ``[0.0, 1.0]``.
    """
    if repeated_run_count < 2:
        return None
    if not run_answers:
        return None
    counts = Counter(run_answers)
    modal_count = counts.most_common(1)[0][1]
    return modal_count / len(run_answers)


def compute_query_metrics(
    trace: ProofTrace,
    run_answers: Optional[Sequence[str]] = None,
    repeated_run_count: int = 1,
) -> QueryMetrics:
    """Compute the full :class:`QueryMetrics` bundle for a terminated trace.

    ``run_answers`` and ``repeated_run_count`` drive Reasoning_Consistency; when the
    repeated-run count is below 2 (the default), consistency is left unset (Req 7.5).
    """
    return QueryMetrics(
        faithfulness_score=compute_faithfulness_score(trace),
        step_hallucination_rate=compute_step_hallucination_rate(trace),
        reasoning_consistency=compute_reasoning_consistency(
            run_answers or [],
            repeated_run_count,
        ),
    )
