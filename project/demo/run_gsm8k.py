"""GSM8K-style multi-step math benchmark: System vs Plain LLM vs Chain-of-Thought.

This is the Phase 1/2 experiment: it stresses the architecture's reason -> validate ->
repair loop on multi-step arithmetic, with the :class:`ArithmeticValidationEngine`
actually checking each intermediate calculation (so a wrong step is caught and repaired
rather than carried through to a wrong final answer).

Usage (from the ``project/`` directory)::

    python demo/run_gsm8k.py                          # MOCK mode (offline, deterministic)
    python demo/run_gsm8k.py --backend ollama --model qwen3:8b
    python demo/run_gsm8k.py --backend ollama --model qwen3:8b --dataset path/to/gsm8k.jsonl --limit 50

MOCK mode (the default, used by tests) is fully offline and deterministic: the System and
baselines are driven by scripted MockBackends on a small set of controlled items,
*including a planted arithmetic error*. It demonstrates the thesis reproducibly — the
System's wrong first step is REJECTED by arithmetic validation and REPAIRED to the correct
value (right final answer), while the Chain-of-Thought baseline commits the same error and
ends wrong. These are scripted illustrations, NOT a real model.

OLLAMA mode runs genuine answers from a local model (e.g. ``qwen3:8b``) over the bundled
ORIGINAL sample dataset (``--dataset`` for the real GSM8K corpus). The bundled sample is a
small set of original problems for demonstration, not the official GSM8K.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _DEMO_DIR.parent
for _p in (str(_PROJECT_DIR / "src"), str(_DEMO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nsr.baselines import build_baseline  # noqa: E402
from nsr.comparison_report import build_comparison_report  # noqa: E402
from nsr.evaluation_harness import (  # noqa: E402
    LLM_ONLY_METHOD_NAME,
    SYSTEM_METHOD_NAME,
    EvaluationHarness,
)
from nsr.llm_component import MockBackend, build_ollama_backend, ollama_available  # noqa: E402
from nsr.models import DatasetItem, Domain, ProductionRule, SystemConfig, VerifiedOutput  # noqa: E402

import datasets  # noqa: E402
import run_benchmark as rb  # noqa: E402 (reuse report rendering + StepClock)
import scenarios  # noqa: E402
from arithmetic_validation import ArithmeticValidationEngine  # noqa: E402

OUTPUT_DIR = _DEMO_DIR / "output"

CHAIN_OF_THOUGHT = "chain-of-thought"
BASELINE_NAMES = (LLM_ONLY_METHOD_NAME, CHAIN_OF_THOUGHT)

DEFAULT_OLLAMA_MODEL = "qwen3:8b"

# Per-method deterministic mock latencies (ms) so the offline report is reproducible.
_SYSTEM_STEP_MS = 6.0
_BASELINE_STEP_MS = {LLM_ONLY_METHOD_NAME: 1.5, CHAIN_OF_THOUGHT: 3.0}

#: A permissive "well-formed step" rule so the ACT-R controller always has a rule to
#: select (avoiding a spurious no-rule-matched → repair on correct steps). Its empty
#: IF/THEN makes it always applicable and vacuously satisfied; the genuine content check
#: comes from the ArithmeticValidationEngine layered on top.
PERMISSIVE_RULE = ProductionRule(rule_id="well-formed-step", condition="", action="")


def _json_step(logic_form, lhs=None, op=None, rhs=None, result=None) -> str:
    """A constrained-decoder-shaped JSON completion, optionally carrying an equation."""
    preds = {}
    if op is not None:
        preds = {"lhs": lhs, "op": op, "rhs": rhs, "result": result}
    return json.dumps({"logic_form": logic_form, "predicates": preds})


# --------------------------------------------------------------------------- #
# MOCK mode: controlled single-sub-goal items, one with a planted arithmetic error
# --------------------------------------------------------------------------- #
#
# Each item's query is a single sub-goal (no "then"/"and"/sentence breaks) so the System
# runs exactly one accept cycle (plus a repair when the first scripted step is wrong). The
# System script for a "planted error" item is [wrong-equation, correct-equation]: the
# ArithmeticValidationEngine rejects the wrong one, the Repair Coordinator regenerates the
# correct one, and the goal is satisfied with the right answer.

_MOCK_ITEMS = [
    {
        "item_id": "mock-1",
        "query": "Compute 7 times 8. then subtract 9 from the result.",
        "ground_truth": "47",
        # Step 1 correct (accepted), step 2 wrong (56-9=45) -> rejected by arithmetic
        # validation -> repaired to 56-9=47 (right). One clean accept + one repair.
        "system_script": [
            _json_step("7 * 8 = 56", 7, "*", 8, 56),
            _json_step("56 - 9 = 45", 56, "-", 9, 45),
            _json_step("56 - 9 = 47", 56, "-", 9, 47),
        ],
        "cot": "7 times 8 is 56. 56 minus 9 is 45.\nAnswer: 45",  # same error, uncaught
        "llm": "Answer: 45",                                        # wrong
    },
    {
        "item_id": "mock-2",
        "query": "Compute 6 times 9. then add 4 to the result.",
        "ground_truth": "58",
        # Both steps correct on the first try -> two clean accepts (faithfulness 1.0).
        "system_script": [
            _json_step("6 * 9 = 54", 6, "*", 9, 54),
            _json_step("54 + 4 = 58", 54, "+", 4, 58),
        ],
        "cot": "6 times 9 is 54. 54 plus 4 is 58.\nAnswer: 58",  # correct
        "llm": "Answer: 58",                                      # correct
    },
    {
        "item_id": "mock-3",
        "query": "Compute 12 times 3. then subtract 5 from the result.",
        "ground_truth": "31",
        # Step 1 correct, step 2 wrong (36-5=30) -> rejected -> repaired to 31.
        "system_script": [
            _json_step("12 * 3 = 36", 12, "*", 3, 36),
            _json_step("36 - 5 = 30", 36, "-", 5, 30),
            _json_step("36 - 5 = 31", 36, "-", 5, 31),
        ],
        "cot": "12 times 3 is 36. 36 minus 5 is 30.\nAnswer: 30",  # wrong
        "llm": "Answer: 31",                                        # correct
    },
]


def _mock_dataset() -> list[DatasetItem]:
    return [
        DatasetItem(
            item_id=it["item_id"],
            query=it["query"],
            ground_truth=it["ground_truth"],
            domain=Domain.MATH,
        )
        for it in _MOCK_ITEMS
    ]


def make_math_config(model: str = "mock") -> SystemConfig:
    """System config for the math benchmark: repair enabled so wrong steps can be fixed."""
    return SystemConfig(
        max_cycle_limit=12,
        repair_attempt_limit=2,
        retry_count=2,
        llm_selection=model,
        output_format="json",
        conflict_resolution_policy="priority",
        generation_timeout_ms=60000,
        repeated_run_count=1,
        random_seed=7,
    )


class _MockMathSystem:
    """Scripted, offline System-under-test with real arithmetic validation + repair."""

    def __init__(self, config: SystemConfig) -> None:
        self._config = config
        self._scripts = {it["query"]: it["system_script"] for it in _MOCK_ITEMS}

    def run(self, query: object):
        script = self._scripts.get(str(query), [_json_step("0")])
        orchestrator = scenarios.build_orchestrator_with_backend(
            backend=MockBackend(list(script)),
            procedural_memory=[PERMISSIVE_RULE],  # always-applicable; arithmetic engine checks content
            config=self._config,
            validation=ArithmeticValidationEngine(),
        )
        return orchestrator.run(query)


class _MockMathBaseline:
    """Scripted, offline baseline (Plain LLM / Chain-of-Thought) over a MockBackend."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._key = "cot" if name == CHAIN_OF_THOUGHT else "llm"
        self._completions = {it["query"]: it[self._key] for it in _MOCK_ITEMS}
        self._step_ms = _BASELINE_STEP_MS[name]

    def run(self, query: str):
        completion = self._completions.get(str(query), "Answer: 0")
        method = build_baseline(
            self.name, MockBackend([completion]), clock=rb.StepClock(self._step_ms)
        )
        return method.run(query)


# --------------------------------------------------------------------------- #
# OLLAMA mode: genuine answers from a local model
# --------------------------------------------------------------------------- #


class _RealMathSystem:
    def __init__(self, model: str, host, config: SystemConfig) -> None:
        self._model = model
        self._host = host
        self._config = config

    def run(self, query: object):
        orchestrator = scenarios.build_orchestrator_with_backend(
            backend=build_ollama_backend(self._model, host=self._host),
            procedural_memory=[PERMISSIVE_RULE],
            config=self._config,
            validation=ArithmeticValidationEngine(),
            translation=scenarios.MathReasoningTranslationLayer(),
        )
        return orchestrator.run(query)


class _RealMathBaseline:
    def __init__(self, name: str, model: str, host) -> None:
        self.name = name
        self._model = model
        self._host = host

    def run(self, query: str):
        return build_baseline(
            self.name, build_ollama_backend(self._model, host=self._host)
        ).run(query)


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #


def _write_reports(report, dataset, run_record, *, stem, is_real, model_label):
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"html": out / f"{stem}.html", "json": out / f"{stem}.json"}
    paths["html"].write_text(
        rb.build_report_html(
            report, dataset, run_record, is_real=is_real, model_label=model_label
        ),
        encoding="utf-8",
    )
    paths["json"].write_text(
        json.dumps(
            rb._report_to_dict(
                report, dataset, run_record, is_real=is_real, model_label=model_label
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def generate_gsm8k_mock(output_dir: os.PathLike | str = OUTPUT_DIR) -> dict[str, Path]:
    """Run the offline, deterministic mock math benchmark and write the reports."""
    config = make_math_config("mock")
    dataset = _mock_dataset()
    system = _MockMathSystem(config)
    baselines = {name: _MockMathBaseline(name) for name in BASELINE_NAMES}
    harness = EvaluationHarness(
        system,
        baselines,
        clock=rb.StepClock(_SYSTEM_STEP_MS),
        answer_match=datasets.numeric_answer_match,
    )
    runs = [harness.run(dataset, config=config, model_id="offline-mock-gsm8k")]
    report = build_comparison_report(runs, repeated_run_count=config.repeated_run_count)
    global OUTPUT_DIR
    saved = OUTPUT_DIR
    OUTPUT_DIR = Path(output_dir)
    try:
        paths = _write_reports(
            report, dataset, runs[0].run_record,
            stem="gsm8k_report", is_real=False, model_label=None,
        )
    finally:
        OUTPUT_DIR = saved
    return paths


def _run_gsm8k_real(model, host, dataset_path, limit, output_dir):
    config = make_math_config(model)
    if dataset_path:
        dataset = datasets.load_gsm8k_jsonl(dataset_path, limit=limit)
    else:
        dataset = datasets.load_sample(limit=limit)
    system = _RealMathSystem(model, host, config)
    baselines = {name: _RealMathBaseline(name, model, host) for name in BASELINE_NAMES}
    harness = EvaluationHarness(
        system, baselines, answer_match=datasets.numeric_answer_match
    )
    runs = [
        harness.run(dataset, config=config, model_id=f"ollama:{model}")
        for _ in range(config.repeated_run_count)
    ]
    report = build_comparison_report(runs, repeated_run_count=config.repeated_run_count)
    global OUTPUT_DIR
    saved = OUTPUT_DIR
    OUTPUT_DIR = Path(output_dir)
    try:
        paths = _write_reports(
            report, dataset, runs[0].run_record,
            stem="gsm8k_report_real", is_real=True, model_label=model,
        )
    finally:
        OUTPUT_DIR = saved
    return paths, runs, dataset


def generate_gsm8k_real(
    model: str = DEFAULT_OLLAMA_MODEL,
    host=None,
    dataset_path=None,
    limit=None,
    output_dir: os.PathLike | str = OUTPUT_DIR,
) -> dict[str, Path]:
    """Run the real-model math benchmark (no preflight; the CLI preflights)."""
    paths, _runs, _dataset = _run_gsm8k_real(model, host, dataset_path, limit, output_dir)
    return paths


def _print_paths(paths) -> None:
    print(f"Reports written to {OUTPUT_DIR}:")
    for path in paths.values():
        print(f"  - {path.name}  ({path.stat().st_size} bytes)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "GSM8K-style math benchmark (System vs Plain LLM vs Chain-of-Thought). "
            "Defaults to an offline, deterministic mock illustration."
        )
    )
    parser.add_argument("--backend", choices=["mock", "ollama"], default="mock")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--host", default=None)
    parser.add_argument("--dataset", default=None, help="path to a GSM8K-format JSONL")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    if args.backend == "mock":
        paths = generate_gsm8k_mock()
        print("Offline mock GSM8K benchmark complete (deterministic illustration).")
        _print_paths(paths)
        return 0

    ok, reason = ollama_available(args.host)
    if not ok:
        print(f"Ollama not available: {reason}", file=sys.stderr)
        print("Start it with `ollama serve` and pull a model, e.g. "
              f"`ollama pull {args.model}`.", file=sys.stderr)
        return 2
    print(f"Ollama reachable - {reason}")
    print(f"Running REAL GSM8K benchmark against '{args.model}' (this calls the model)...")
    paths, runs, dataset = _run_gsm8k_real(
        args.model, args.host, args.dataset, args.limit, OUTPUT_DIR
    )
    _print_paths(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
