"""Basic unit tests for the baseline reasoning methods (Task 13.1).

These verify each baseline runs against the scriptable :class:`MockBackend` (no network
or local model), conforms to the shared :class:`ReasoningMethod` interface, and returns
a final answer with a wall-clock latency. The dedicated interface-conformance suite is
Task 13.2; these cover the core behaviour of each strategy.

_Requirements: 9.3_
"""

from __future__ import annotations

import pytest

from nsr.baselines import (
    BASELINE_METHODS,
    BaselineResult,
    BaseReasoningMethod,
    ChainOfThought,
    LLMOnly,
    ReAct,
    ReasoningMethod,
    SelfConsistency,
    TreeOfThoughts,
    build_baseline,
    extract_final_answer,
)
from nsr.llm_component import MockBackend


# A deterministic fake clock so latency is exercised without real wall time.
class FakeClock:
    """Returns a monotonically increasing time, +1.0s per call."""

    def __init__(self) -> None:
        self._t = 0.0

    def __call__(self) -> float:
        now = self._t
        self._t += 1.0
        return now


# ---------------------------------------------------------------------------
# extract_final_answer
# ---------------------------------------------------------------------------


def test_extract_answer_prefix():
    assert extract_final_answer("reasoning...\nAnswer: 42") == "42"


def test_extract_finish_marker():
    assert extract_final_answer("Thought: done. Finish[the cat]") == "the cat"


def test_extract_last_line_fallback():
    assert extract_final_answer("step one\nstep two\n7") == "7"


def test_extract_empty():
    assert extract_final_answer("") == ""
    assert extract_final_answer("   \n  ") == ""


def test_extract_last_answer_line_wins():
    assert extract_final_answer("Answer: first\nmore\nAnswer: final") == "final"


# ---------------------------------------------------------------------------
# LLM-only reference baseline
# ---------------------------------------------------------------------------


def test_llm_only_single_call_returns_answer():
    backend = MockBackend(["Answer: 4"])
    method = LLMOnly(backend)
    result = method.run("What is 2+2?")
    assert isinstance(result, BaselineResult)
    assert result.method == "llm-only"
    assert result.final_answer == "4"
    # The reference baseline performs exactly one generation pass.
    assert backend.call_count == 1
    assert result.raw_outputs == ["Answer: 4"]


def test_llm_only_reports_latency():
    backend = MockBackend(["Answer: 4"])
    method = LLMOnly(backend, clock=FakeClock())
    result = method.run("q")
    # One clock tick before and after _reason -> 1.0s -> 1000ms.
    assert result.latency_ms == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Chain-of-Thought
# ---------------------------------------------------------------------------


def test_chain_of_thought_reads_answer_from_trace():
    backend = MockBackend(["First 2. Then 2. Answer: 4"])
    method = ChainOfThought(backend)
    result = method.run("What is 2+2?")
    assert result.method == "chain-of-thought"
    assert result.final_answer == "4"
    assert backend.call_count == 1


# ---------------------------------------------------------------------------
# Self-Consistency
# ---------------------------------------------------------------------------


def test_self_consistency_takes_modal_answer():
    # Three samples: two say 4, one says 5 -> modal is 4.
    backend = MockBackend(["Answer: 4", "Answer: 5", "Answer: 4"])
    method = SelfConsistency(backend, num_samples=3)
    result = method.run("What is 2+2?")
    assert result.method == "self-consistency"
    assert result.final_answer == "4"
    assert backend.call_count == 3
    assert len(result.raw_outputs) == 3


def test_self_consistency_requires_positive_samples():
    with pytest.raises(ValueError):
        SelfConsistency(MockBackend(), num_samples=0)


# ---------------------------------------------------------------------------
# Tree-of-Thoughts
# ---------------------------------------------------------------------------


def test_tree_of_thoughts_explores_and_votes():
    # breadth=2, depth=1 -> 2 thoughts + 2 continuations = 4 calls.
    script = [
        "Thought A",
        "Answer: 10",  # continuation of branch A
        "Thought B",
        "Answer: 10",  # continuation of branch B
    ]
    backend = MockBackend(script)
    method = TreeOfThoughts(backend, breadth=2, depth=1)
    result = method.run("solve it")
    assert result.method == "tree-of-thoughts"
    assert result.final_answer == "10"
    assert backend.call_count == 4


def test_tree_of_thoughts_validates_params():
    with pytest.raises(ValueError):
        TreeOfThoughts(MockBackend(), breadth=0)
    with pytest.raises(ValueError):
        TreeOfThoughts(MockBackend(), depth=0)


# ---------------------------------------------------------------------------
# ReAct
# ---------------------------------------------------------------------------


def test_react_finishes_on_marker():
    backend = MockBackend(["I should compute. Finish[42]"])
    method = ReAct(backend, max_steps=5)
    result = method.run("the question")
    assert result.method == "react"
    assert result.final_answer == "42"
    # Stops as soon as the finish marker appears.
    assert backend.call_count == 1


def test_react_interleaves_until_answer():
    script = [
        "Need more info",          # step 1: no conclusion
        "Still thinking",          # step 2: no conclusion
        "Answer: done",            # step 3: concludes
    ]
    backend = MockBackend(script)
    method = ReAct(backend, max_steps=5)
    result = method.run("q")
    assert result.final_answer == "done"
    assert backend.call_count == 3


def test_react_exhausts_budget():
    # Never concludes: budget of 2 steps, reads last completion's answer (fallback).
    backend = MockBackend(["thinking 1", "thinking 2", "thinking 3"])
    method = ReAct(backend, max_steps=2)
    result = method.run("q")
    assert backend.call_count == 2
    assert result.final_answer == "thinking 2"


# ---------------------------------------------------------------------------
# Common interface conformance (light-touch; full suite is Task 13.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(BASELINE_METHODS))
def test_every_baseline_conforms_to_interface(name):
    backend = MockBackend(["Answer: x"])
    method = build_baseline(name, backend)
    assert isinstance(method, BaseReasoningMethod)
    assert isinstance(method, ReasoningMethod)  # runtime-checkable Protocol
    result = method.run("question")
    assert isinstance(result, BaselineResult)
    assert isinstance(result.final_answer, str)
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms >= 0.0


def test_build_baseline_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_baseline("does-not-exist", MockBackend())
