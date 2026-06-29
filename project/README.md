# Neuro-Symbolic System-2 Reasoning Architecture (nsr)

A hybrid reasoning-control architecture that makes multi-step reasoning
**verifiable and repairable** by pairing a neural Large Language Model (System 1)
with an ACT-R-style symbolic cognitive controller (System 2). Each intermediate
reasoning step is translated into a machine-checkable symbolic representation,
validated against symbolic production rules, and accepted, rejected, or repaired
*before* it propagates to the next step.

By validating each step before it propagates, the system detects and repairs
structural, arithmetic, and goal-alignment errors — reducing the classes of
hallucination the symbolic layer can verify. It does not claim to eliminate
hallucination outside those checkable classes (e.g. false world-knowledge
premises the controller has no ground truth for); its contribution is making
reasoning **traceable, verifiable, and controllable** while maintaining
competitive task performance.

### Three layers of verification

| Layer       | Question it answers                                 |
|-------------|-----------------------------------------------------|
| Structural  | Is the reasoning step well-formed?                  |
| Arithmetic  | Is the computation correct?                         |
| Goal-aligned| Is the step solving the intended objective?         |

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
