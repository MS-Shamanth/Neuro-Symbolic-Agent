"""Tests for the GSM8K real-model runner (``run_benchmark.generate_benchmark_gsm8k``).

Fully offline with no Ollama server: the backend factory is monkeypatched to a canned
:class:`~nsr.llm_component.MockBackend` and the preflight is stubbed. They confirm the
GSM8K flow runs the real pipeline (with live arithmetic validation) end-to-end and writes
a non-empty ``benchmark_report_gsm8k.html`` / ``.json`` carrying every report metric and
the model / dataset / arithmetic-validation labels.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

run_benchmark = importlib.import_module("run_benchmark")

from nsr.llm_component import MockBackend  # noqa: E402

#: A conforming, arithmetically-correct equation step: the constrained decoder accepts it,
#: the ArithmeticValidationEngine verifies 6*7=42, and numeric matching reduces it to 42.
#: For the baselines the same text yields a final number of 42 via the numeric matcher.
_CANNED_STEP = json.dumps(
    {"logic_form": "6 * 7 = 42", "predicates": {"lhs": 6, "op": "*", "rhs": 7, "result": 42}}
)


def _canned_backend_factory(*_args, **_kwargs) -> MockBackend:
    """Stand-in for ``build_ollama_backend`` returning a canned, conforming backend."""
    return MockBackend([_CANNED_STEP])


def test_generate_benchmark_gsm8k_writes_labelled_reports_offline(monkeypatch, tmp_path):
    """With the backend factory stubbed, the GSM8K flow writes non-empty labelled reports."""
    monkeypatch.setattr(run_benchmark, "build_ollama_backend", _canned_backend_factory)

    paths = run_benchmark.generate_benchmark_gsm8k(
        model="qwen3:8b", host=None, dataset_path=None, limit=3, output_dir=tmp_path
    )

    assert set(paths) == {"html", "json"}
    for key, path in paths.items():
        assert path.exists(), f"{key} report was not created"
        assert path.stat().st_size > 0, f"{key} report is empty"
        assert path.name.startswith("benchmark_report_gsm8k")

    html = paths["html"].read_text(encoding="utf-8")
    assert "Real model: qwen3:8b via Ollama" in html
    assert "Arithmetic validation is active" in html
    assert "bundled ORIGINAL sample" in html  # dataset label for the sample path

    document = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert document["is_real"] is True
    assert document["mode"] == "real"
    assert document["real_model"] == "qwen3:8b"
    assert document["arithmetic_validation"] is True
    assert "sample" in (document["dataset_label"] or "").lower()
    assert document["model_id"] == "ollama:qwen3:8b"
    assert document["system_method"] == "neuro-symbolic"

    # Every report metric is present with per-baseline differences.
    metric_names = {m["metric"] for m in document["metrics"]}
    assert metric_names == set(run_benchmark.REPORT_METRICS)
    for metric in document["metrics"]:
        assert set(metric["differences"]) == set(document["baseline_methods"])


def test_generate_benchmark_gsm8k_with_official_path(monkeypatch, tmp_path):
    """Pointing at an official-format file labels the report with the dataset file name."""
    monkeypatch.setattr(run_benchmark, "build_ollama_backend", _canned_backend_factory)

    dataset_file = tmp_path / "test.jsonl"
    dataset_file.write_text(
        '{"question": "6 times 7?", "answer": "6 * 7 = 42\\n#### 42"}\n'
        '{"question": "another?", "answer": "#### 10"}\n',
        encoding="utf-8",
    )

    paths = run_benchmark.generate_benchmark_gsm8k(
        model="mistral", host=None, dataset_path=dataset_file, limit=2, output_dir=tmp_path
    )

    document = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert "test.jsonl" in document["dataset_label"]
    assert document["real_model"] == "mistral"
    # The official-format items were loaded (gsm8k-ids, 2 items capped by limit).
    assert len(document["dataset"]) == 2
    assert all(entry["item_id"].startswith("gsm8k-") for entry in document["dataset"])


def test_cli_gsm8k_success_path_offline(monkeypatch, capsys, tmp_path):
    """main() --dataset gsm8k with a reachable (stubbed) server runs offline, returns 0."""
    monkeypatch.setattr(
        run_benchmark,
        "ollama_available",
        lambda host=None: (True, "1 models available: qwen3:8b"),
    )
    monkeypatch.setattr(run_benchmark, "build_ollama_backend", _canned_backend_factory)
    monkeypatch.setattr(run_benchmark, "OUTPUT_DIR", tmp_path)

    exit_code = run_benchmark.main(
        ["--backend", "ollama", "--dataset", "gsm8k", "--model", "qwen3:8b", "--limit", "2"]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Ollama reachable" in out
    assert "Real-model GSM8K benchmark complete" in out
    assert (tmp_path / "benchmark_report_gsm8k.html").exists()
    assert (tmp_path / "benchmark_report_gsm8k.json").exists()


def test_cli_gsm8k_exits_nonzero_when_unavailable(monkeypatch, capsys):
    """An unreachable server makes the GSM8K CLI return non-zero + print hints, no raise."""
    reason = "could not reach Ollama at http://localhost:11434 (Connection refused)"
    monkeypatch.setattr(
        run_benchmark, "ollama_available", lambda host=None: (False, reason)
    )

    exit_code = run_benchmark.main(["--backend", "ollama", "--dataset", "gsm8k"])

    assert exit_code != 0
    out = capsys.readouterr().out
    assert reason in out
    assert "ollama serve" in out
