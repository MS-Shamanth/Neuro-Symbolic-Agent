"""Baseline reasoning methods for comparative evaluation.

This module implements the four established :term:`Baseline_Method` strategies the
design names -- **Chain-of-Thought**, **Self-Consistency**, **Tree-of-Thoughts**, and
**ReAct** -- plus the **LLM-only** reference baseline, behind a single common interface
(:class:`ReasoningMethod`). Each method consumes a pluggable
:class:`~nsr.llm_component.LLMBackend` (so they are fully testable with
:class:`~nsr.llm_component.MockBackend`, no network or local model required) and returns
a :class:`BaselineResult` carrying a final answer and the wall-clock latency for a query
(Req 9.3).

The ``LLM_Only_Baseline`` is the reference point for latency overhead: it asks the
backend once and returns that answer directly, performing no extra reasoning passes
(see Req 9.5). The other baselines layer additional generation passes on top:

- **Chain-of-Thought** elicits a single step-by-step trace and reads off its answer.
- **Self-Consistency** samples several independent traces and takes the *modal* answer.
- **Tree-of-Thoughts** expands several candidate thoughts and continues each branch,
  then selects the modal answer across the explored leaves.
- **ReAct** interleaves reasoning ("thought") and action/observation steps in a bounded
  loop until the model emits a final answer.

All five share the timing, result construction, and answer-extraction logic in
:class:`BaseReasoningMethod`; each subclass only implements its reasoning strategy.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional, Protocol, runtime_checkable

from .llm_component import LLMBackend, OutputSchema

#: Default per-attempt generation timeout (milliseconds) for a baseline backend call.
DEFAULT_TIMEOUT_MS = 30_000

#: Conventional prefix the methods look for when reading a final answer from output.
ANSWER_PREFIX = "answer:"


# ---------------------------------------------------------------------------
# Common result + interface
# ---------------------------------------------------------------------------


@dataclass
class BaselineResult:
    """The outcome of running a baseline method over a single query.

    Carries the two quantities the evaluation harness needs from every method
    (Req 9.3): the ``final_answer`` and the wall-clock ``latency_ms``. ``raw_outputs``
    keeps every backend completion produced (useful for inspection and for
    consistency/voting), and ``metadata`` records strategy-specific details.
    """

    method: str
    final_answer: str
    latency_ms: float
    raw_outputs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ReasoningMethod(Protocol):
    """The common interface every baseline (and the LLM-only reference) conforms to.

    A reasoning method exposes a human-readable :attr:`name` and a :meth:`run` that maps
    a query string to a :class:`BaselineResult` carrying a final answer and latency.
    """

    name: str

    def run(self, query: str) -> BaselineResult:
        """Reason over ``query`` and return its final answer and wall-clock latency."""
        ...


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_final_answer(text: str) -> str:
    """Read a final answer out of a raw completion using simple conventions.

    Resolution order:

    1. A ReAct-style ``Finish[...]`` marker -- the text inside the brackets.
    2. The text following the last ``Answer:`` marker (case-insensitive), whether it
       begins a line or appears inline, trimmed to the end of that line.
    3. Otherwise the last non-empty line of the completion.

    An empty or whitespace-only completion yields an empty string.
    """
    if text is None:
        return ""

    # 1. ReAct finish marker: Finish[<answer>]
    lowered = text.lower()
    marker = "finish["
    idx = lowered.rfind(marker)
    if idx != -1:
        end = text.find("]", idx)
        if end != -1:
            return text[idx + len(marker) : end].strip()

    # 2. Last "Answer:" marker wins (line-leading or inline). Take the remainder of the
    #    line that follows the marker.
    ans_idx = lowered.rfind(ANSWER_PREFIX)
    if ans_idx != -1:
        after = text[ans_idx + len(ANSWER_PREFIX) :]
        return after.splitlines()[0].strip() if after.strip() else ""

    # 3. Fall back to the last non-empty line.
    last_non_empty: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            last_non_empty = line
    return last_non_empty if last_non_empty is not None else ""


def _modal_answer(answers: list[str]) -> str:
    """Return the most common answer, breaking ties by first appearance.

    :class:`collections.Counter` preserves first-encountered order among equal counts,
    so selection is deterministic for a given input ordering. An empty input yields an
    empty string.
    """
    non_empty = [a for a in answers if a]
    pool = non_empty or answers
    if not pool:
        return ""
    return Counter(pool).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Shared base method
# ---------------------------------------------------------------------------


class BaseReasoningMethod(ABC):
    """Shared timing, backend invocation, and result construction for baselines.

    Subclasses implement :meth:`_reason`, returning ``(final_answer, raw_outputs)``;
    this base measures wall-clock latency around that call and packages the
    :class:`BaselineResult`. The backend is invoked through :meth:`_generate`, which
    forwards the configured :class:`OutputSchema` and per-attempt timeout.
    """

    #: Method name reported in results and used by the harness; overridden per subclass.
    name: ClassVar[str] = "baseline"

    def __init__(
        self,
        backend: LLMBackend,
        *,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        schema: Optional[OutputSchema] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self._backend = backend
        self._timeout_s = timeout_ms / 1000.0
        self._schema = schema if schema is not None else OutputSchema(format="text")
        self._clock = clock

    def _generate(self, prompt: str) -> str:
        """Produce one raw completion for ``prompt`` from the pluggable backend."""
        return self._backend.generate(prompt, self._schema, self._timeout_s)

    @abstractmethod
    def _reason(self, query: str) -> tuple[str, list[str]]:
        """Run the strategy and return ``(final_answer, raw_outputs)``."""
        raise NotImplementedError

    def run(self, query: str) -> BaselineResult:
        """Reason over ``query`` and return its final answer and wall-clock latency.

        The latency covers the entire reasoning strategy (all backend passes), so it can
        be compared against the LLM-only reference to compute overhead (Req 9.5).
        """
        start = self._clock()
        final_answer, raw_outputs = self._reason(query)
        latency_ms = (self._clock() - start) * 1000.0
        return BaselineResult(
            method=self.name,
            final_answer=final_answer,
            latency_ms=latency_ms,
            raw_outputs=raw_outputs,
        )


# ---------------------------------------------------------------------------
# LLM-only reference baseline
# ---------------------------------------------------------------------------


class LLMOnly(BaseReasoningMethod):
    """The LLM-only reference baseline: one call, answer returned directly (Req 9.5).

    Performs no extra reasoning passes, so its latency is the natural reference point
    for measuring the overhead added by every other method.
    """

    name: ClassVar[str] = "llm-only"

    def _reason(self, query: str) -> tuple[str, list[str]]:
        prompt = f"Answer the following question directly.\n\nQuestion: {query}\nAnswer:"
        raw = self._generate(prompt)
        return extract_final_answer(raw), [raw]


# ---------------------------------------------------------------------------
# Chain-of-Thought
# ---------------------------------------------------------------------------


class ChainOfThought(BaseReasoningMethod):
    """Chain-of-Thought: elicit a single step-by-step trace and read its answer."""

    name: ClassVar[str] = "chain-of-thought"

    def _reason(self, query: str) -> tuple[str, list[str]]:
        prompt = (
            "Answer the following question. Think step by step, then state the final "
            "answer on a line beginning with 'Answer:'.\n\n"
            f"Question: {query}\nLet's think step by step:"
        )
        raw = self._generate(prompt)
        return extract_final_answer(raw), [raw]


# ---------------------------------------------------------------------------
# Self-Consistency
# ---------------------------------------------------------------------------


class SelfConsistency(BaseReasoningMethod):
    """Self-Consistency: sample several traces and take the modal final answer.

    ``num_samples`` independent Chain-of-Thought completions are drawn; the answer that
    appears most often (ties broken by first appearance) is returned.
    """

    name: ClassVar[str] = "self-consistency"

    def __init__(self, backend: LLMBackend, *, num_samples: int = 5, **kwargs: Any) -> None:
        super().__init__(backend, **kwargs)
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        self._num_samples = num_samples

    def _reason(self, query: str) -> tuple[str, list[str]]:
        prompt = (
            "Answer the following question. Think step by step, then state the final "
            "answer on a line beginning with 'Answer:'.\n\n"
            f"Question: {query}\nLet's think step by step:"
        )
        raw_outputs: list[str] = []
        answers: list[str] = []
        for _ in range(self._num_samples):
            raw = self._generate(prompt)
            raw_outputs.append(raw)
            answers.append(extract_final_answer(raw))
        return _modal_answer(answers), raw_outputs


# ---------------------------------------------------------------------------
# Tree-of-Thoughts
# ---------------------------------------------------------------------------


class TreeOfThoughts(BaseReasoningMethod):
    """Tree-of-Thoughts: expand candidate thoughts, continue branches, vote.

    A bounded exploration: ``breadth`` candidate first thoughts are generated, then each
    branch is continued (``depth`` continuation rounds) toward a final answer. The modal
    answer across the explored leaves is returned. Defaults are kept small so the method
    stays fast and testable.
    """

    name: ClassVar[str] = "tree-of-thoughts"

    def __init__(
        self,
        backend: LLMBackend,
        *,
        breadth: int = 3,
        depth: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(backend, **kwargs)
        if breadth < 1:
            raise ValueError("breadth must be at least 1")
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self._breadth = breadth
        self._depth = depth

    def _reason(self, query: str) -> tuple[str, list[str]]:
        raw_outputs: list[str] = []
        leaf_answers: list[str] = []

        for branch in range(self._breadth):
            thought_prompt = (
                "Propose one promising line of reasoning (a single 'thought') toward "
                f"answering the question. This is candidate branch {branch + 1}.\n\n"
                f"Question: {query}\nThought:"
            )
            thought = self._generate(thought_prompt)
            raw_outputs.append(thought)
            scratchpad = thought

            for _ in range(self._depth):
                continue_prompt = (
                    "Continue this line of reasoning. If you can conclude, state the "
                    "final answer on a line beginning with 'Answer:'.\n\n"
                    f"Question: {query}\nReasoning so far:\n{scratchpad}\nNext:"
                )
                continuation = self._generate(continue_prompt)
                raw_outputs.append(continuation)
                scratchpad = f"{scratchpad}\n{continuation}"

            leaf_answers.append(extract_final_answer(scratchpad))

        return _modal_answer(leaf_answers), raw_outputs


# ---------------------------------------------------------------------------
# ReAct
# ---------------------------------------------------------------------------


class ReAct(BaseReasoningMethod):
    """ReAct: interleave reasoning and action steps until a final answer is emitted.

    A bounded loop (``max_steps``) accumulates a scratchpad of thought/action/observation
    turns. Each turn the backend is asked for the next thought or action; when it emits a
    ``Finish[...]`` marker or an ``Answer:`` line, the loop stops and that answer is
    returned. If the budget is exhausted, the answer is read from the last completion.
    """

    name: ClassVar[str] = "react"

    def __init__(self, backend: LLMBackend, *, max_steps: int = 5, **kwargs: Any) -> None:
        super().__init__(backend, **kwargs)
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._max_steps = max_steps

    def _reason(self, query: str) -> tuple[str, list[str]]:
        raw_outputs: list[str] = []
        scratchpad = ""

        for step in range(self._max_steps):
            prompt = (
                "Solve the question by interleaving Thought, Action, and Observation "
                "steps. When you know the answer, emit 'Finish[<answer>]'.\n\n"
                f"Question: {query}\n{scratchpad}Thought {step + 1}:"
            )
            raw = self._generate(prompt)
            raw_outputs.append(raw)
            scratchpad += f"Thought {step + 1}: {raw}\n"

            lowered = raw.lower()
            if "finish[" in lowered or ANSWER_PREFIX in lowered:
                return extract_final_answer(raw), raw_outputs

            # No conclusion yet: append a neutral observation and continue.
            scratchpad += f"Observation {step + 1}: (no external tool; continue)\n"

        # Budget exhausted: read the best answer we have from the last completion.
        final = extract_final_answer(raw_outputs[-1]) if raw_outputs else ""
        return final, raw_outputs


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

#: The four established baselines plus the LLM-only reference, keyed by their name.
BASELINE_METHODS: dict[str, type[BaseReasoningMethod]] = {
    LLMOnly.name: LLMOnly,
    ChainOfThought.name: ChainOfThought,
    SelfConsistency.name: SelfConsistency,
    TreeOfThoughts.name: TreeOfThoughts,
    ReAct.name: ReAct,
}


def build_baseline(
    name: str, backend: LLMBackend, **kwargs: Any
) -> BaseReasoningMethod:
    """Construct a baseline method by its registered ``name``.

    Raises :class:`KeyError` (via a clear ``ValueError``) for an unknown name.
    """
    try:
        method_cls = BASELINE_METHODS[name]
    except KeyError:
        allowed = ", ".join(sorted(BASELINE_METHODS))
        raise ValueError(
            f"unknown baseline method {name!r}; expected one of: {allowed}"
        ) from None
    return method_cls(backend, **kwargs)


__all__ = [
    "BaselineResult",
    "ReasoningMethod",
    "BaseReasoningMethod",
    "LLMOnly",
    "ChainOfThought",
    "SelfConsistency",
    "TreeOfThoughts",
    "ReAct",
    "BASELINE_METHODS",
    "build_baseline",
    "extract_final_answer",
]
