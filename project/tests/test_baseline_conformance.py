"""Interface-conformance suite for the baseline reasoning methods (Task 13.2).

Where ``test_baselines.py`` exercises each strategy's specific behaviour, this suite
focuses narrowly on the *shared contract* every baseline in :data:`BASELINE_METHODS`
must honour, driven entirely by the registry so adding a baseline automatically extends
coverage. For each of the five registered methods (``llm-only``, ``chain-of-thought``,
``self-consistency``, ``tree-of-thoughts``, ``react``) it asserts that:

* the constructed method conforms to the runtime-checkable :class:`ReasoningMethod`
  Protocol (and is a :class:`BaseReasoningMethod`);
* it exposes a non-empty string :attr:`name` that matches its registry key;
* ``run(query)`` returns a :class:`BaselineResult` whose ``final_answer`` is a ``str``
  and whose ``latency_ms`` is a non-negative number; and
* the result echoes the method's own name.

All runs use the scriptable :class:`MockBackend`, so no network or local model is
needed. ``MockBackend`` repeats its last scripted item once exhausted, so a single
answer suffices regardless of how many generation passes a strategy makes.

_Requirements: 9.3_
"""

from __future__ import annotations

from numbers import Real

import pytest

from nsr.baselines import (
    BASELINE_METHODS,
    BaselineResult,
    BaseReasoningMethod,
    ReasoningMethod,
    build_baseline,
)
from nsr.llm_component import MockBackend


# Every registered baseline name; the suite is parametrised over the live registry so
# coverage tracks the registry rather than a hand-maintained list.
BASELINE_NAMES = sorted(BASELINE_METHODS)


def _make(name: str) -> BaseReasoningMethod:
    """Build a registered baseline backed by a scriptable, deterministic MockBackend."""
    return build_baseline(name, MockBackend(["Answer: 42"]))


def test_registry_has_all_five_baselines():
    """The registry exposes exactly the five expected methods (four baselines + ref)."""
    assert set(BASELINE_NAMES) == {
        "llm-only",
        "chain-of-thought",
        "self-consistency",
        "tree-of-thoughts",
        "react",
    }


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_baseline_conforms_to_reasoning_method_protocol(name):
    """Each baseline satisfies the runtime-checkable shared interface."""
    method = _make(name)
    assert isinstance(method, BaseReasoningMethod)
    assert isinstance(method, ReasoningMethod)  # runtime-checkable Protocol


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_baseline_name_is_set_and_matches_registry_key(name):
    """Every baseline carries a non-empty name equal to its registry key."""
    method = _make(name)
    assert isinstance(method.name, str)
    assert method.name != ""
    assert method.name == name


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_run_returns_baseline_result_with_answer_and_latency(name):
    """run(query) yields a BaselineResult with a string answer and numeric latency."""
    method = _make(name)
    result = method.run("What is 6 times 7?")

    assert isinstance(result, BaselineResult)

    # A final answer is always a string (extraction never returns None).
    assert isinstance(result.final_answer, str)

    # Latency is a real, non-negative number of milliseconds.
    assert isinstance(result.latency_ms, Real)
    assert not isinstance(result.latency_ms, bool)
    assert result.latency_ms >= 0.0

    # The result is attributed to the method that produced it.
    assert result.method == name


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_run_invokes_backend_at_least_once(name):
    """Producing an answer requires consulting the backend at least once."""
    backend = MockBackend(["Answer: 42"])
    method = build_baseline(name, backend)
    method.run("anything")
    assert backend.call_count >= 1
