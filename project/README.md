# Neuro-Symbolic System-2 Reasoning Architecture (nsr)

A hybrid AI system that reduces hallucinations in multi-step reasoning by pairing a
neural Large Language Model (System 1) with an ACT-R-style symbolic cognitive
controller (System 2). Each intermediate reasoning step is translated into a
machine-checkable symbolic representation, validated against symbolic production
rules, and accepted, rejected, or repaired *before* it propagates to the next step.

## Project layout

```
src/nsr/        # package source
  models/       # core data models (enums + dataclasses)
tests/          # pytest + Hypothesis test suite
pyproject.toml  # build config, dependencies, pytest configuration
```

## Development setup

```bash
python -m pip install -e ".[dev]"
pytest
```
