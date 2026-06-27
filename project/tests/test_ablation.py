"""Tests for the GSM8K ablation study (``demo/ablation.py``).

Every test here runs **fully offline with no Ollama server**: the per-query backend factory
:func:`ablation.build_ollama_backend` is monkeypatched to return a scripted
:class:`~nsr.llm_component.MockBackend`, and the CLI preflight is stubbed. They confirm:

- :class:`ablation.NoOpValidationEngine` always accepts (config C semantics).
- Each of the four configurations (A plain-llm, B constrained-decoding, C
  actr-no-validation, D full-neuro-symbolic) constructed over a stubbed backend produces a
  :class:`~nsr.models.VerifiedOutput` with a numeric final answer.
- :func:`ablation.generate_ablation_gsm8k` over a 2-item stubbed dataset writes the
  ``benchmark_report_ablation_gsm8k`` HTML + JSON, both naming all FOUR configs, with
  faithfulness/step-hallucination ``None`` (n/a) for A/B/C and present for D, and accuracy +
  latency present for all four.
- The ``--ablation`` CLI flag wires through ``main()`` offline and returns 0.
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

ablation = importlib.import_module("ablation")
run_benchmark = importlib.import_module("run_benchmark")

from nsr.llm_component import MockBackend  # noqa: E402
from nsr.models import (  # noqa: E402
    DatasetItem,
    Domain,
    ProductionRule,
    SymbolicRepresentation,
    ValidationStatus,
    VerifiedOutput,
)

#: A conforming JSON step: a checkable equation whose RHS (42) is the final answer.
_CANNED_STEP = json.dumps(
    {"logic_form": "6 * 7 = 42", "predicates": {"lhs": 6, "op": "*", "rhs": 7, "result": 42}}
)


def _canned_backend_factory(*_args, **_kwargs) -> MockBackend:
    """Stand-in for ``build_ollama_backend``: a fresh MockBackend repeating the canned step.

    A fresh backend per call mirrors how the real factory yields one backend per query; the
    single canned completion is a well-formed step for the constrained decoder / orchestrator
    and (via last-number extraction) a usable answer for the plain-llm baseline.
    """
    return MockBackend([_CANNED_STEP])


def _stub_dataset(*_args, **_kwargs):
    """A 2-item stubbed GSM8K dataset (ground truth 42) + a label, replacing the loader."""
    items = [
        DatasetItem(
            item_id="ab-1",
            query="Compute 6 times 7.",
            ground_truth="42",
            domain=Domain.MATH,
        ),
        DatasetItem(
            item_id="ab-2",
            query="What is 6 multiplied by 7?",
            ground_truth="42",
            domain=Domain.MATH,
        ),
    ]
    return items, "stubbed ablation sample (2 items)"


@pytest.fixture()
def _offline(monkeypatch):
    """Drive all four configs offline via the canned backend factory + stub dataset."""
    monkeypatch.setattr(ablation, "build_ollama_backend", _canned_backend_factory)
    monkeypatch.setattr(ablation, "_gsm8k_dataset", _stub_dataset)


# --------------------------------------------------------------------------- #
# 1. NoOpValidationEngine always accepts
# --------------------------------------------------------------------------- #


def test_noop_validation_engine_always_accepts():
    """The no-op engine returns ACCEPTED with empty violated lists for any rep."""
    engine = ablation.NoOpValidationEngine()
    rules = [
        ProductionRule(rule_id="never", condition="IF contradiction", action="THEN flag"),
    ]

    for logic_form in ("7 * 8 = 54", "anything", ""):
        rep = SymbolicRepresentation(logic_form=logic_form)
        outcome = engine.validate(rep, rules)
        assert outcome.status == ValidationStatus.ACCEPTED
        assert outcome.accepted is True
        assert outcome.rejected is False
        assert outcome.violated_rule_ids == []
        assert outcome.violated_rules == []


# --------------------------------------------------------------------------- #
# 2. Each of the four configs produces a VerifiedOutput + numeric final answer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "factory",
    [
        ablation.PlainLLMConfig,
        ablation.ConstrainedDecodingConfig,
        ablation.ActrNoValidationConfig,
        ablation.FullNeuroSymbolicConfig,
        ablation.FullPlusGoalConfig,
    ],
)
def test_each_config_runs_to_verified_output(monkeypatch, factory):
    """Every config, over a stubbed backend, yields a VerifiedOutput whose answer is 42."""
    monkeypatch.setattr(ablation, "build_ollama_backend", _canned_backend_factory)
    config = run_benchmark.make_real_config("qwen3:8b", repeated_run_count=1)

    system = factory("qwen3:8b", None, config)
    result = system.run("Compute 6 times 7.")

    assert isinstance(result, VerifiedOutput)
    assert result.proof_trace is not None
    assert len(result.proof_trace.steps) >= 1
    # The final answer reduces to the equation RHS (42) under numeric matching.
    assert run_benchmark.numeric_answer_match(result.final_answer, "42")


def test_config_b_extracts_equation_rhs(monkeypatch):
    """Config B takes the '<expr> = <result>' RHS via numeric extraction as the answer."""
    monkeypatch.setattr(ablation, "build_ollama_backend", _canned_backend_factory)
    config = run_benchmark.make_real_config("qwen3:8b", repeated_run_count=1)

    system = ablation.ConstrainedDecodingConfig("qwen3:8b", None, config)
    result = system.run("Compute 6 times 7.")

    assert result.final_answer == "42"


# --------------------------------------------------------------------------- #
# 3. End-to-end ablation run writes labelled reports with honest n/a handling
# --------------------------------------------------------------------------- #


def test_generate_ablation_writes_reports_with_na_handling(_offline, tmp_path):
    """The ablation runner writes HTML+JSON naming all four configs with honest n/a."""
    paths = ablation.generate_ablation_gsm8k(
        model="qwen3:8b", limit=2, repeated_run_count=1, output_dir=tmp_path
    )

    assert set(paths) == {"html", "json"}
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0
    assert paths["html"].name == "benchmark_report_ablation_gsm8k.html"
    assert paths["json"].name == "benchmark_report_ablation_gsm8k.json"

    html_text = paths["html"].read_text(encoding="utf-8")
    json_text = paths["json"].read_text(encoding="utf-8")

    # All FIVE config names appear in both reports.
    for name in (
        "plain-llm",
        "constrained-decoding",
        "actr-no-validation",
        "full-neuro-symbolic",
        "arithmetic+goal",
    ):
        assert name in html_text, f"{name} missing from HTML"
        assert name in json_text, f"{name} missing from JSON"

    document = json.loads(json_text)
    assert document["mode"] == "ablation"
    assert document["system_method"] == "full-neuro-symbolic"
    assert document["configs"] == [
        "plain-llm",
        "constrained-decoding",
        "actr-no-validation",
        "full-neuro-symbolic",
        "arithmetic+goal",
    ]

    metrics = {m["metric"]: m["values"] for m in document["metrics"]}

    # Faithfulness + step-hallucination are n/a (None) for the non-validating configs
    # A/B/C, and present for the two validating configs D and E.
    for metric in ("faithfulness", "step_hallucination_rate"):
        values = metrics[metric]
        assert values["plain-llm"] is None
        assert values["constrained-decoding"] is None
        assert values["actr-no-validation"] is None
        assert values["full-neuro-symbolic"] is not None
        assert values["arithmetic+goal"] is not None

    # Accuracy + latency are present (not None) for ALL five configs.
    for metric in ("final_answer_accuracy", "mean_latency", "latency_overhead"):
        values = metrics[metric]
        for config in (
            "plain-llm",
            "constrained-decoding",
            "actr-no-validation",
            "full-neuro-symbolic",
            "arithmetic+goal",
        ):
            assert values[config] is not None, f"{metric}/{config} should be present"

    # The report renders the n/a marker and a prominent ablation note.
    assert "n/a" in html_text
    assert "ABLATION STUDY" in html_text or "Ablation study" in html_text


def test_generate_ablation_accuracy_perfect_on_matching_stub(_offline, tmp_path):
    """With every config answering 42 and ground truth 42, accuracy is 1.0 for all five."""
    paths = ablation.generate_ablation_gsm8k(
        model="qwen3:8b", limit=2, repeated_run_count=1, output_dir=tmp_path
    )
    document = json.loads(paths["json"].read_text(encoding="utf-8"))
    accuracy = {m["metric"]: m["values"] for m in document["metrics"]}["final_answer_accuracy"]
    for config in (
        "plain-llm",
        "constrained-decoding",
        "actr-no-validation",
        "full-neuro-symbolic",
        "arithmetic+goal",
    ):
        assert accuracy[config] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 4. CLI wiring for --ablation (offline)
# --------------------------------------------------------------------------- #


def test_cli_ablation_success_path_offline(monkeypatch, capsys, tmp_path):
    """main() --ablation with a reachable (stubbed) server runs offline and returns 0."""
    monkeypatch.setattr(
        run_benchmark,
        "ollama_available",
        lambda host=None: (True, "1 models available: qwen3:8b"),
    )
    monkeypatch.setattr(ablation, "build_ollama_backend", _canned_backend_factory)
    monkeypatch.setattr(ablation, "_gsm8k_dataset", _stub_dataset)
    monkeypatch.setattr(run_benchmark, "OUTPUT_DIR", tmp_path)

    exit_code = run_benchmark.main(
        ["--backend", "ollama", "--dataset", "gsm8k", "--ablation",
         "--model", "qwen3:8b", "--limit", "2", "--runs", "1"]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "ABLATION" in out
    assert "ablation study complete" in out.lower()
    assert (tmp_path / "benchmark_report_ablation_gsm8k.html").exists()
    assert (tmp_path / "benchmark_report_ablation_gsm8k.json").exists()


def test_cli_ablation_exits_nonzero_when_ollama_unavailable(monkeypatch, capsys):
    """An unreachable server makes --ablation return non-zero + print hints, no raise."""
    reason = "could not reach Ollama at http://localhost:11434 (Connection refused)"
    monkeypatch.setattr(run_benchmark, "ollama_available", lambda host=None: (False, reason))

    def _boom(*_a, **_k):  # pragma: no cover - must never run when unavailable
        raise AssertionError("the ablation runner should not be called when down")

    monkeypatch.setattr(ablation, "generate_ablation_gsm8k", _boom)

    exit_code = run_benchmark.main(
        ["--backend", "ollama", "--dataset", "gsm8k", "--ablation", "--model", "qwen3:8b"]
    )

    assert exit_code != 0
    out = capsys.readouterr().out
    assert reason in out
    assert "ollama serve" in out
    assert "ollama pull qwen3:8b" in out
