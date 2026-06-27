"""Tests for the REAL-MODEL benchmark mode (``demo/run_benchmark.py --backend ollama``).

Every test here runs **fully offline with no Ollama server**: the real pipeline is driven
over an injected :class:`~nsr.llm_component.MockBackend`, and the preflight / network call
is stubbed via ``monkeypatch``. They confirm:

- :func:`scenarios.build_orchestrator_with_backend` runs the *real* four-stage pipeline
  over an injected backend (the new injection path), producing a genuine
  :class:`~nsr.models.VerifiedOutput` with a Proof Trace + Faithfulness Score.
- The CLI preflight exits non-zero (without raising) and prints the friendly reason +
  hints when Ollama is unreachable.
- With Ollama "reachable" and the backend factory stubbed to a canned response,
  :func:`run_benchmark.generate_benchmark_real` runs end-to-end and writes a non-empty
  ``benchmark_report_real.html`` / ``.json`` clearly labelled as real-model results.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

# The demo modules live in ``project/demo`` (a sibling of ``tests``); add to the path.
_DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

run_benchmark = importlib.import_module("run_benchmark")
scenarios = importlib.import_module("scenarios")

from nsr.llm_component import MockBackend  # noqa: E402
from nsr.models import VerifiedOutput  # noqa: E402

#: A conforming JSON step the constrained decoder accepts (non-empty ``logic_form``).
_CANNED_STEP = json.dumps({"logic_form": "blue", "predicates": {}})


def _canned_backend_factory(*_args, **_kwargs) -> MockBackend:
    """Stand-in for ``build_ollama_backend`` returning a canned, conforming backend.

    A fresh :class:`MockBackend` per call mirrors how the real factory yields a fresh
    backend per query; the backend repeats its single canned JSON completion, which is a
    well-formed step for the System and a (genuine, if wrong) answer for the baselines.
    """
    return MockBackend([_CANNED_STEP])


# --------------------------------------------------------------------------- #
# 1. The new injection path: real pipeline over an injected MockBackend
# --------------------------------------------------------------------------- #


def test_build_orchestrator_with_backend_runs_real_pipeline_offline():
    """An injected MockBackend drives the real orchestrator to a VerifiedOutput."""
    backend = MockBackend([_CANNED_STEP])
    config = run_benchmark.make_real_config("llama3.1")

    orchestrator = scenarios.build_orchestrator_with_backend(
        backend=backend,
        procedural_memory=run_benchmark.STARTER_PRODUCTION_RULES,
        config=config,
    )
    result = orchestrator.run("What color is the clear daytime sky?")

    # The real pipeline produced a genuine VerifiedOutput with a Proof Trace + score.
    assert isinstance(result, VerifiedOutput)
    assert result.proof_trace is not None
    assert 0.0 <= result.faithfulness_score <= 1.0
    # The System really invoked the injected backend (no scripting shortcut).
    assert backend.call_count >= 1
    # The accepted answer is the model's conclusion, surfaced verbatim.
    assert result.final_answer == "blue"


def test_build_orchestrator_delegates_to_injection_path():
    """The scripted convenience wrapper still returns an orchestrator + its MockBackend."""
    orchestrator, backend = scenarios.build_orchestrator(
        script=[_CANNED_STEP],
        procedural_memory=run_benchmark.STARTER_PRODUCTION_RULES,
    )
    assert isinstance(backend, MockBackend)
    result = orchestrator.run("What color is the clear daytime sky?")
    assert isinstance(result, VerifiedOutput)


# --------------------------------------------------------------------------- #
# 2. Preflight / CLI behaviour when Ollama is unreachable
# --------------------------------------------------------------------------- #


def test_cli_exits_nonzero_when_ollama_unavailable(monkeypatch, capsys):
    """An unreachable server makes main() return non-zero + print the reason, no raise."""
    reason = "could not reach Ollama at http://localhost:11434 (Connection refused)"
    monkeypatch.setattr(run_benchmark, "ollama_available", lambda host=None: (False, reason))

    # Should NOT raise; should return a non-zero exit code.
    exit_code = run_benchmark.main(["--backend", "ollama", "--model", "llama3.1"])

    assert exit_code != 0
    out = capsys.readouterr().out
    assert reason in out
    # Friendly hints are surfaced.
    assert "ollama serve" in out
    assert "ollama pull llama3.1" in out


def test_cli_unavailable_does_not_write_real_reports(monkeypatch, capsys, tmp_path):
    """When unavailable, generation is never attempted (the core must not be called)."""

    def _boom(*_a, **_k):  # pragma: no cover - must never run
        raise AssertionError("the benchmark core should not be called when down")

    monkeypatch.setattr(
        run_benchmark, "ollama_available", lambda host=None: (False, "down")
    )
    monkeypatch.setattr(run_benchmark, "_run_real_benchmark", _boom)

    assert run_benchmark.main(["--backend", "ollama"]) != 0


# --------------------------------------------------------------------------- #
# 3. End-to-end real generation, fully offline via a stubbed backend factory
# --------------------------------------------------------------------------- #


def test_generate_benchmark_real_writes_labelled_reports_offline(monkeypatch, tmp_path):
    """With the backend factory stubbed, the real flow writes non-empty labelled reports."""
    monkeypatch.setattr(run_benchmark, "build_ollama_backend", _canned_backend_factory)

    paths = run_benchmark.generate_benchmark_real(
        model="mistral", host=None, output_dir=tmp_path
    )

    assert set(paths) == {"html", "json"}
    for key, path in paths.items():
        assert path.exists(), f"{key} report was not created"
        assert path.stat().st_size > 0, f"{key} report is empty"
        # The files carry the *_real naming.
        assert path.name.endswith(("_real.html", "_real.json"))

    html = paths["html"].read_text(encoding="utf-8")
    # The HTML prominently states it is a real model via Ollama.
    assert "Real model: mistral via Ollama" in html
    assert "genuine results from an actual language model" in html
    # It distinguishes the starter rule-set scaffold honestly.
    assert "general-purpose starter production-rule set" in html

    document = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert document["is_real"] is True
    assert document["mode"] == "real"
    assert document["real_model"] == "mistral"
    assert document["system_method"] == "neuro-symbolic"
    assert document["model_id"] == "ollama:mistral"
    # Every report metric is present with per-baseline differences.
    metric_names = {m["metric"] for m in document["metrics"]}
    assert metric_names == set(run_benchmark.REPORT_METRICS)
    for metric in document["metrics"]:
        assert set(metric["differences"]) == set(document["baseline_methods"])


def test_cli_real_success_path_offline(monkeypatch, capsys, tmp_path):
    """main() with a reachable (stubbed) server runs generation offline and returns 0."""
    monkeypatch.setattr(
        run_benchmark,
        "ollama_available",
        lambda host=None: (True, "1 models available: llama3.1:8b"),
    )
    # Drive the whole real flow offline via the canned backend factory, and redirect the
    # output directory the CLI writes to so the test never touches demo/output.
    monkeypatch.setattr(run_benchmark, "build_ollama_backend", _canned_backend_factory)
    monkeypatch.setattr(run_benchmark, "OUTPUT_DIR", tmp_path)

    exit_code = run_benchmark.main(["--backend", "ollama", "--model", "llama3.1"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Ollama reachable" in out
    assert "Real-model benchmark complete" in out
    assert (tmp_path / "benchmark_report_real.html").exists()
    assert (tmp_path / "benchmark_report_real.json").exists()


def test_cli_real_reports_all_excluded_as_failure(monkeypatch, capsys, tmp_path):
    """If every model call fails (e.g. model not pulled), main() exits non-zero + warns."""
    from nsr.llm_component import BackendUnavailable

    def _failing_factory(*_a, **_k):
        backend = MockBackend([BackendUnavailable("HTTP 404: model not pulled")])
        return backend

    monkeypatch.setattr(
        run_benchmark, "ollama_available", lambda host=None: (True, "0 models available")
    )
    monkeypatch.setattr(run_benchmark, "build_ollama_backend", _failing_factory)
    monkeypatch.setattr(run_benchmark, "OUTPUT_DIR", tmp_path)

    exit_code = run_benchmark.main(["--backend", "ollama", "--model", "llama3.1"])

    assert exit_code != 0
    out = capsys.readouterr().out
    assert "No results were produced" in out
    assert "ollama pull llama3.1" in out
    # A labelled-but-empty report is still written.
    assert (tmp_path / "benchmark_report_real.json").exists()
