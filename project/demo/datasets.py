"""Standard-benchmark dataset loaders + numeric answer matching (demo, Phase 1).

This module loads **standard benchmark file formats** so the official datasets can be
dropped in later without code changes (stdlib ``json`` only — no network, no downloads).

Standard-format loaders (Phase 1 deliverable)
---------------------------------------------

- :func:`load_gsm8k` — parse the **official GSM8K** JSONL format. Each line is a JSON
  object ``{"question": str, "answer": str}`` where the ``answer`` string ends with a line
  ``#### <number>`` (GSM8K's final-answer convention). ``ground_truth`` is that final
  number (commas / ``$`` stripped); ``domain`` is :attr:`Domain.MATH`.

  Obtain the official file from the **openai/grade-school-math** repository
  (``grade_school_math/data/test.jsonl``) — https://github.com/openai/grade-school-math —
  then run with ``--path <test.jsonl>``. Nothing is downloaded by this module.

- :func:`load_multiple_choice` — parse ARC / CommonsenseQA-style multiple-choice JSONL.
  Tolerant of the common shape variants: the nested
  ``{"question": {"stem": ..., "choices": [{"label", "text"}, ...]}, "answerKey": "B"}``
  form, a flat ``{"question", "choices", "answerKey"}`` form, and choices given either as a
  list of ``{"label", "text"}`` objects, a list of plain strings, or a parallel
  ``{"label": [...], "text": [...]}`` object. ``query`` is the stem followed by the
  enumerated choices; ``ground_truth`` is the correct choice **text**. ``domain`` defaults
  to :attr:`Domain.COMMONSENSE` (configurable).

  Official files: ARC — https://allenai.org/data/arc ;
  CommonsenseQA — https://www.tau-nlp.sites.tau.ac.il/commonsenseqa .

- :func:`load_strategyqa` — parse StrategyQA-style JSONL ``{"question", "answer": bool}``
  into ``ground_truth`` of ``"yes"`` / ``"no"`` with ``domain`` :attr:`Domain.MULTI_HOP`.

  Official file: StrategyQA — https://allenai.org/data/strategyqa .

- :func:`load_benchmark` — a dispatcher over ``{"gsm8k", "multiple-choice",
  "strategyqa"}`` that applies an optional ``limit`` (first ``N`` items). For ``gsm8k``
  with no ``path`` it loads the bundled original sample.

Every loader builds raw item dicts, **skips malformed/incomplete lines**, and runs the
result through :func:`nsr.dataset_loader.load_dataset` for the final
:class:`~nsr.models.DatasetItem` validation (non-empty unique id / query / ground-truth and
a recognized domain), so a dropped-in official file is validated exactly like any other.

Legacy demo helpers (kept for the GSM8K runner / existing tests)
----------------------------------------------------------------

- :func:`load_gsm8k_jsonl` / :func:`load_sample` — the original demo loaders used by
  ``demo/run_gsm8k.py``; :func:`load_sample` returns the bundled set of ~10 **original**
  multi-step problems (``demo/data/gsm8k_sample.jsonl``), which is NOT the official GSM8K
  corpus and is clearly labelled as a sample everywhere it is reported.

:func:`numeric_answer_match` is the answer matcher used by the Evaluation Harness for math:
it extracts the final number from a (possibly verbose) prediction and compares it
numerically to the ground-truth number, so "The answer is 18." matches "18".
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, Optional

from nsr.dataset_loader import load_dataset
from nsr.models import DatasetItem, Domain

#: The bundled original sample dataset (NOT official GSM8K).
SAMPLE_PATH = Path(__file__).resolve().parent / "data" / "gsm8k_sample.jsonl"

#: Matches a signed integer or decimal, optionally with thousands separators, e.g.
#: ``1,234``, ``-5``, ``3.5``. Used to pull the final number out of free text.
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _clean_number(token: str) -> Optional[float]:
    """Parse a number token (stripping ``$``, commas, trailing period) to a float."""
    token = token.strip().strip("$").rstrip(".")
    token = token.replace(",", "")
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def extract_final_number(text: str) -> Optional[float]:
    """Return the final numeric value mentioned in ``text``, or ``None``.

    Resolution order, most explicit first:

    1. the number following the last ``####`` marker (GSM8K's final-answer convention);
    2. the number following the last ``answer:`` marker (case-insensitive);
    3. otherwise the last number that appears anywhere in the text.
    """
    if text is None:
        return None
    s = str(text)

    marker = s.rfind("####")
    if marker != -1:
        m = _NUMBER_RE.search(s, marker)
        if m:
            return _clean_number(m.group())

    lowered = s.lower()
    ans = lowered.rfind("answer:")
    if ans != -1:
        m = _NUMBER_RE.search(s, ans)
        if m:
            return _clean_number(m.group())

    numbers = _NUMBER_RE.findall(s)
    if numbers:
        return _clean_number(numbers[-1])
    return None


def numeric_answer_match(predicted: str, ground_truth: str) -> bool:
    """Numeric final-answer match used by the harness for math items.

    Extracts the final number from ``predicted`` and from ``ground_truth`` and compares
    them numerically (so ``"18"``, ``"18.0"``, and ``"The answer is 18."`` all match).
    Returns ``False`` when either side has no parseable number.
    """
    p = extract_final_number(predicted)
    g = extract_final_number(ground_truth)
    if p is None or g is None:
        return False
    return abs(p - g) < 1e-9


def _ground_truth_from_answer(answer: str) -> str:
    """Extract the canonical final-answer string from a GSM8K ``answer`` field."""
    value = extract_final_number(answer)
    if value is None:
        return ""
    # Render integers without a trailing ".0" so ground_truth reads naturally.
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return str(value)


def load_gsm8k_jsonl(path: str | Path, *, limit: Optional[int] = None) -> list[DatasetItem]:
    """Load GSM8K-format JSONL into :class:`DatasetItem`s (domain = mathematical-reasoning).

    Each non-empty line must be a JSON object with a non-empty ``question`` and an
    ``answer`` carrying the final number after ``####`` (the GSM8K convention). Lines
    whose answer has no parseable final number are skipped. ``limit`` caps the count.
    """
    path = Path(path)
    items: list[DatasetItem] = []
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = str(record.get("question", "")).strip()
            answer = str(record.get("answer", ""))
            ground_truth = _ground_truth_from_answer(answer)
            if not question or not ground_truth:
                continue
            items.append(
                DatasetItem(
                    item_id=f"gsm8k-{index + 1}",
                    query=question,
                    ground_truth=ground_truth,
                    domain=Domain.MATH,
                )
            )
            if limit is not None and len(items) >= limit:
                break
    return items


def load_sample(*, limit: Optional[int] = None) -> list[DatasetItem]:
    """Load the bundled ORIGINAL sample dataset (not official GSM8K)."""
    return load_gsm8k_jsonl(SAMPLE_PATH, limit=limit)


# --------------------------------------------------------------------------- #
# Standard-benchmark format loaders (Phase 1)
#
# Each loader parses one standard file format into raw item dicts, skips malformed or
# incomplete lines, and runs the result through nsr.dataset_loader.load_dataset for the
# final DatasetItem validation. Stdlib json only; no network.
# --------------------------------------------------------------------------- #

#: Default labels A, B, C, ... used to enumerate multiple-choice options that carry no
#: explicit label, and to map a letter ``answerKey`` to a positional choice.
_CHOICE_LETTERS = [chr(ord("A") + i) for i in range(26)]


def _iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict]]:
    """Yield ``(line_index, record)`` for each well-formed JSON object line in ``path``.

    Blank lines and lines that are not valid JSON objects are skipped (malformed-line
    tolerance), so a single bad line never aborts loading an otherwise-valid file.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(record, dict):
                yield index, record


def _gsm8k_ground_truth(answer: object) -> Optional[str]:
    """Extract the final number after the last ``####`` marker, or ``None``.

    Strips thousands separators and a leading ``$`` and renders an integral value without
    a trailing ``.0`` (the GSM8K convention is an integer final answer).
    """
    text = str(answer if answer is not None else "")
    marker = text.rfind("####")
    if marker == -1:
        return None
    match = _NUMBER_RE.search(text, marker)
    if match is None:
        return None
    value = _clean_number(match.group())
    if value is None:
        return None
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return str(value)


def load_gsm8k(path: str | Path, *, limit: Optional[int] = None) -> list[DatasetItem]:
    """Load the **official GSM8K** JSONL format into validated :class:`DatasetItem`s.

    Each line is ``{"question": str, "answer": str}`` where ``answer`` ends with a line
    ``#### <number>``. ``ground_truth`` is that final number (commas / ``$`` stripped),
    ``query`` is the question, ``domain`` is :attr:`Domain.MATH`, and ``item_id`` is
    ``f"gsm8k-{i}"`` (``i`` is the source line index). Lines lacking a question or a
    parseable ``#### <number>`` are skipped. ``limit`` keeps the first ``N`` valid items.
    """
    raw: list[dict] = []
    for index, record in _iter_jsonl(path):
        question = str(record.get("question", "")).strip()
        ground_truth = _gsm8k_ground_truth(record.get("answer", ""))
        if not question or not ground_truth:
            continue
        raw.append(
            {
                "item_id": f"gsm8k-{index}",
                "query": question,
                "ground_truth": ground_truth,
                "domain": Domain.MATH.value,
            }
        )
        if limit is not None and len(raw) >= limit:
            break
    return load_dataset(raw).items


def _normalize_choices(record: dict) -> list[tuple[str, str]]:
    """Return ``[(label, text), ...]`` from the choices in a multiple-choice ``record``.

    Tolerates the nested ``question.choices`` form, a flat top-level ``choices`` field, a
    list of ``{"label"/"key", "text"/"value"}`` objects, a list of plain strings, and a
    parallel ``{"label": [...], "text": [...]}`` object. Missing labels are filled with
    positional letters (A, B, C, ...).
    """
    question = record.get("question")
    if isinstance(question, dict) and question.get("choices") is not None:
        choices = question.get("choices")
    else:
        choices = record.get("choices")

    pairs: list[tuple[str, str]] = []
    if isinstance(choices, dict):
        labels = choices.get("label") or choices.get("labels") or []
        texts = choices.get("text") or choices.get("texts") or []
        for position, text in enumerate(texts):
            label = str(labels[position]) if position < len(labels) else _letter(position)
            pairs.append((label, str(text)))
    elif isinstance(choices, list):
        for position, choice in enumerate(choices):
            if isinstance(choice, dict):
                label = choice.get("label", choice.get("key"))
                label = str(label) if label is not None else _letter(position)
                text = choice.get("text", choice.get("value", ""))
                pairs.append((label, str(text)))
            else:
                pairs.append((_letter(position), str(choice)))
    return pairs


def _letter(position: int) -> str:
    """Return the positional choice label (A, B, ...) or a numeric fallback."""
    if 0 <= position < len(_CHOICE_LETTERS):
        return _CHOICE_LETTERS[position]
    return str(position + 1)


def _correct_choice_text(pairs: list[tuple[str, str]], answer_key: object) -> Optional[str]:
    """Resolve the correct choice TEXT from ``answer_key`` against ``pairs``.

    Matches the key against a choice label (case-insensitive); failing that, treats an
    integer-like key as a 0-based or 1-based positional index. Returns ``None`` when the
    key cannot be resolved to a choice.
    """
    if answer_key is None or not pairs:
        return None
    key = str(answer_key).strip()
    for label, text in pairs:
        if label.strip().lower() == key.lower():
            return text
    # Positional fallback for integer keys (support both 0-based and 1-based).
    try:
        position = int(float(key))
    except ValueError:
        return None
    for candidate in (position, position - 1):
        if 0 <= candidate < len(pairs):
            return pairs[candidate][1]
    return None


def _stem(record: dict) -> str:
    """Return the question stem from either the nested or flat shape."""
    question = record.get("question")
    if isinstance(question, dict):
        return str(question.get("stem", "")).strip()
    return str(question if question is not None else "").strip()


def load_multiple_choice(
    path: str | Path,
    *,
    domain: Domain = Domain.COMMONSENSE,
    limit: Optional[int] = None,
) -> list[DatasetItem]:
    """Load ARC / CommonsenseQA-style multiple-choice JSONL into :class:`DatasetItem`s.

    ``query`` is the stem followed by the enumerated choices (``"A) ...\\nB) ..."``);
    ``ground_truth`` is the correct choice **text**. ``domain`` defaults to
    :attr:`Domain.COMMONSENSE`. Lines lacking a stem, choices, or a resolvable answer key
    are skipped. ``item_id`` is ``f"mc-{i}"``. ``limit`` keeps the first ``N`` valid items.
    """
    raw: list[dict] = []
    for index, record in _iter_jsonl(path):
        stem = _stem(record)
        pairs = _normalize_choices(record)
        answer_key = record.get("answerKey", record.get("answer"))
        correct_text = _correct_choice_text(pairs, answer_key)
        if not stem or not pairs or not correct_text:
            continue
        enumerated = "\n".join(f"{label}) {text}" for label, text in pairs)
        query = f"{stem}\n{enumerated}"
        raw.append(
            {
                "item_id": f"mc-{index}",
                "query": query,
                "ground_truth": correct_text,
                "domain": domain.value,
            }
        )
        if limit is not None and len(raw) >= limit:
            break
    return load_dataset(raw).items


def _strategyqa_answer(value: object) -> Optional[str]:
    """Map a StrategyQA boolean (or boolean-like) answer to ``"yes"`` / ``"no"``."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("true", "yes"):
            return "yes"
        if token in ("false", "no"):
            return "no"
    return None


def load_strategyqa(path: str | Path, *, limit: Optional[int] = None) -> list[DatasetItem]:
    """Load StrategyQA-style JSONL ``{"question", "answer": bool}`` into items.

    ``ground_truth`` is ``"yes"`` / ``"no"``; ``domain`` is :attr:`Domain.MULTI_HOP`;
    ``item_id`` is ``f"strategyqa-{i}"``. Lines lacking a question or a boolean answer are
    skipped. ``limit`` keeps the first ``N`` valid items.
    """
    raw: list[dict] = []
    for index, record in _iter_jsonl(path):
        question = str(record.get("question", "")).strip()
        answer = _strategyqa_answer(record.get("answer"))
        if not question or answer is None:
            continue
        raw.append(
            {
                "item_id": f"strategyqa-{index}",
                "query": question,
                "ground_truth": answer,
                "domain": Domain.MULTI_HOP.value,
            }
        )
        if limit is not None and len(raw) >= limit:
            break
    return load_dataset(raw).items


#: Recognized dispatcher names mapped to their loader callables.
_BENCHMARK_LOADERS = {
    "gsm8k": load_gsm8k,
    "multiple-choice": load_multiple_choice,
    "strategyqa": load_strategyqa,
}


def load_benchmark(
    name: str,
    path: str | Path | None = None,
    limit: Optional[int] = None,
) -> list[DatasetItem]:
    """Dispatch to the loader for ``name`` and apply ``limit`` (first ``N`` items).

    ``name`` is one of ``{"gsm8k", "multiple-choice", "strategyqa"}``. For ``"gsm8k"`` a
    missing ``path`` loads the bundled original sample (``demo/data/gsm8k_sample.jsonl``);
    the other benchmarks require a ``path`` to an official file.
    """
    key = str(name).strip().lower()
    loader = _BENCHMARK_LOADERS.get(key)
    if loader is None:
        allowed = ", ".join(sorted(_BENCHMARK_LOADERS))
        raise ValueError(f"unknown benchmark {name!r}; expected one of: {allowed}")
    if path is None:
        if key == "gsm8k":
            path = SAMPLE_PATH
        else:
            raise ValueError(f"benchmark {name!r} requires a dataset --path")
    return loader(path, limit=limit)


__all__ = [
    "SAMPLE_PATH",
    "extract_final_number",
    "numeric_answer_match",
    "load_gsm8k_jsonl",
    "load_sample",
    "load_gsm8k",
    "load_multiple_choice",
    "load_strategyqa",
    "load_benchmark",
]
