"""Smoke tests for the offline demo web interface (``project/demo/web_app.py``).

These tests exercise the request routing **in-process, without binding a real socket or
port**: they call the pure :func:`web_app.handle` function (which both the HTTP handler and
these tests share) directly. Like the demo itself, everything runs fully offline via the
scripted :class:`~nsr.llm_component.MockBackend` (no network, no API key).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# The demo modules live in ``project/demo`` (a sibling of ``tests``); add it to the path so
# they import the same way they do when run as ``python demo/web_app.py``.
_DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

web_app = importlib.import_module("web_app")
scenarios = importlib.import_module("scenarios")


def test_index_lists_scenarios():
    """``GET /`` returns 200 HTML listing every available scenario."""
    status, content_type, body = web_app.handle("/", "")

    assert status == 200
    assert "text/html" in content_type
    # The localhost-only / unauthenticated banner is present.
    assert "unauthenticated local demo server" in body
    # Every scenario name and title is listed.
    for name in scenarios.SCENARIOS:
        assert name in body
        assert scenarios.get_scenario(name).title in body


@pytest.mark.parametrize("scenario_name", sorted(scenarios.SCENARIOS))
def test_run_returns_reasoning_report(scenario_name):
    """``GET /run?scenario=<valid>`` returns 200 with the reasoning-report markers."""
    status, content_type, body = web_app.handle("/run", f"scenario={scenario_name}")

    assert status == 200
    assert "text/html" in content_type
    # The same markers the file-export report carries.
    assert "Faithfulness Score" in body
    assert "mermaid" in body
    assert 'class="card' in body


def test_run_accepts_rule_learning_flag():
    """The optional ``&rule_learning=1`` flag is accepted and still returns the report."""
    status, _ct, body = web_app.handle("/run", "scenario=arithmetic-repair&rule_learning=1")

    assert status == 200
    assert "Faithfulness Score" in body


def test_unknown_scenario_returns_helpful_4xx():
    """An unknown scenario name returns a 4xx with a helpful message (no stack trace)."""
    status, content_type, body = web_app.handle("/run", "scenario=does-not-exist")

    assert 400 <= status < 500
    assert "text/html" in content_type
    assert "Unknown scenario" in body
    assert "does-not-exist" in body
    # The available scenarios are suggested.
    for name in scenarios.SCENARIOS:
        assert name in body
    # No raw traceback leaked.
    assert "Traceback" not in body


def test_missing_scenario_returns_4xx():
    """``/run`` with no scenario param returns a helpful 4xx, not a crash."""
    status, _ct, body = web_app.handle("/run", "")

    assert 400 <= status < 500
    assert "scenario" in body.lower()


def test_unknown_path_returns_404():
    """An unknown path returns a clean 404 message page."""
    status, content_type, body = web_app.handle("/nope", "")

    assert status == 404
    assert "text/html" in content_type
    assert "Not found" in body
    assert "Traceback" not in body
