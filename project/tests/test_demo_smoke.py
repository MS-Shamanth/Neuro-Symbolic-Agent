"""Smoke tests for the offline demo package (``project/demo/``).

These tests call the demo's ``generate_*`` entry points in-process, writing to a pytest
``tmp_path``, and assert that every advertised artifact is created and non-empty. They keep
the demo runnable as the codebase evolves, and -- like the demo itself -- run fully offline
via the scripted :class:`~nsr.llm_component.MockBackend` (no network, no API key).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# The demo modules live in ``project/demo`` (a sibling of ``tests``); add it to the path so
# they import the same way they do when run as ``python demo/run_demo.py``.
_DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

run_demo = importlib.import_module("run_demo")
run_benchmark = importlib.import_module("run_benchmark")
scenarios = importlib.import_module("scenarios")


@pytest.mark.parametrize("scenario_name", sorted(scenarios.SCENARIOS))
def test_run_demo_generates_nonempty_artifacts(scenario_name, tmp_path):
    """Each scenario writes a proof trace, Mermaid, DOT, and HTML report, all non-empty."""
    paths = run_demo.generate_demo(scenario_name, output_dir=tmp_path)

    assert set(paths) == {"trace_txt", "mermaid", "dot", "html"}
    for key, path in paths.items():
        assert path.exists(), f"{key} artifact was not created"
        assert path.stat().st_size > 0, f"{key} artifact is empty"

    # The HTML embeds the mermaid CDN and at least one reasoning-step card.
    html = paths["html"].read_text(encoding="utf-8")
    assert "mermaid" in html
    assert "Faithfulness Score" in html
    assert 'class="card' in html

    # The Mermaid and DOT sources carry the expected diagram framing.
    assert paths["mermaid"].read_text(encoding="utf-8").startswith("flowchart TD")
    assert paths["dot"].read_text(encoding="utf-8").startswith("digraph ProofTrace")


def test_arithmetic_repair_scenario_shows_repair_path(tmp_path):
    """The repair scenario records a rejected-then-repaired step in its proof trace."""
    paths = run_demo.generate_demo("arithmetic-repair", output_dir=tmp_path)
    trace_text = paths["trace_txt"].read_text(encoding="utf-8")

    assert "repaired" in trace_text
    assert "goal-satisfied" in trace_text


def test_run_benchmark_generates_nonempty_reports(tmp_path):
    """The benchmark writes a non-empty HTML and JSON report comparing the methods."""
    paths = run_benchmark.generate_benchmark(output_dir=tmp_path)

    assert set(paths) == {"html", "json"}
    for key, path in paths.items():
        assert path.exists(), f"{key} report was not created"
        assert path.stat().st_size > 0, f"{key} report is empty"

    import json

    document = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert document["system_method"] == "neuro-symbolic"
    # Every report metric is present with a System value and per-baseline differences.
    metric_names = {m["metric"] for m in document["metrics"]}
    assert metric_names == set(run_benchmark.REPORT_METRICS)
    for metric in document["metrics"]:
        assert metric["system_value"] is not None
        assert set(metric["differences"]) == set(document["baseline_methods"])
