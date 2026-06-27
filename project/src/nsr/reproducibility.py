"""Reproducibility Manager (Requirement 13).

Builds the run record, seeds every controllable stochastic operation, generates a
seed when none is supplied, and persists the run record together with the computed
metrics to a durable output location.

The manager exposes a *seeding hook* mechanism so other components (LLM sampling,
production-rule conflict resolution, dataset ordering/sampling) can register their own
deterministic re-seeding callbacks and have them invoked with the effective seed.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .models import ErrorRecord, RunRecord, SystemConfig

# A seeding hook receives the effective integer seed and is responsible for seeding
# one stochastic subsystem (e.g. LLM sampling, conflict resolution, dataset ordering).
SeedHook = Callable[[int], None]

# Bound for an auto-generated seed: a non-negative 63-bit integer, comfortably within
# the range accepted by Python's ``random`` and common numeric libraries.
_MAX_GENERATED_SEED = 2**63 - 1


class ReproducibilityManager:
    """Coordinates seeding, run-record construction, and durable persistence."""

    def __init__(self) -> None:
        self._seed_hooks: list[SeedHook] = []
        self._effective_seed: Optional[int] = None

    # ------------------------------------------------------------------ seeding

    def register_seed_hook(self, hook: SeedHook) -> None:
        """Register a callback to be invoked with the effective seed.

        Other components (LLM sampling, conflict resolution, dataset ordering and
        sampling) register here so a single resolved seed governs every controllable
        stochastic operation (Req 13.2). If a seed has already been applied, the hook
        is invoked immediately so late registrants are still seeded deterministically.
        """
        if not callable(hook):
            raise TypeError("seed hook must be callable")
        self._seed_hooks.append(hook)
        if self._effective_seed is not None:
            hook(self._effective_seed)

    def resolve_seed(self, config: SystemConfig) -> int:
        """Return the effective seed: the supplied one, or a freshly generated one.

        When ``config.random_seed`` is supplied it is used as-is (Req 13.2). When it is
        absent a seed is generated (Req 13.3). This does not apply the seed; call
        :meth:`apply_seed` (or :meth:`seed_everything`) to do that.
        """
        if config.random_seed is not None:
            return int(config.random_seed)
        return self._generate_seed()

    @staticmethod
    def _generate_seed() -> int:
        """Generate a fresh, non-negative seed from a non-deterministic source."""
        return int.from_bytes(os.urandom(8), "big") & _MAX_GENERATED_SEED

    def apply_seed(self, seed: int) -> int:
        """Apply ``seed`` to Python's ``random``, any numeric libs, and all hooks.

        Returns the applied seed. Seeds ``random`` and, when available, ``numpy`` and
        sets ``PYTHONHASHSEED`` for child processes, then invokes every registered
        seeding hook so component-level stochastic operations are seeded too.
        """
        seed = int(seed)
        self._effective_seed = seed

        # Python standard library RNG.
        random.seed(seed)

        # Hash seed for any subprocesses spawned later (best-effort, in-process only).
        os.environ["PYTHONHASHSEED"] = str(seed)

        # NumPy, if it is installed and in use by other components.
        try:  # pragma: no cover - exercised only when numpy is present
            import numpy as np

            np.random.seed(seed % (2**32))
        except Exception:
            # NumPy is optional; absence or any seeding error must not break seeding
            # of the libraries that are present.
            pass

        # Fan out to component-registered hooks (LLM sampling, conflict resolution,
        # dataset ordering/sampling, ...).
        for hook in self._seed_hooks:
            hook(seed)

        return seed

    def seed_everything(self, config: SystemConfig) -> int:
        """Resolve the effective seed from ``config`` and apply it everywhere.

        Convenience method combining :meth:`resolve_seed` and :meth:`apply_seed`.
        Returns the effective seed that was applied.
        """
        return self.apply_seed(self.resolve_seed(config))

    @property
    def effective_seed(self) -> Optional[int]:
        """The most recently applied seed, or ``None`` if none has been applied."""
        return self._effective_seed

    # -------------------------------------------------------------- run record

    def build_run_record(
        self,
        config: SystemConfig,
        dataset_ids: list[str],
        model_id: str,
        seed: Optional[int] = None,
        applied_defaults: Optional[dict[str, Any]] = None,
    ) -> RunRecord:
        """Build a :class:`RunRecord` with every required field non-empty (Req 13.1).

        The effective ``seed`` is the supplied argument, else the already-applied seed,
        else a value resolved from ``config`` (generated when absent, per Req 13.3).
        Raises :class:`ValueError` if a required field would be empty.
        """
        if config is None:
            raise ValueError("run record requires a non-empty config")

        if not dataset_ids:
            raise ValueError("run record requires non-empty dataset_ids")
        if any(not str(d).strip() for d in dataset_ids):
            raise ValueError("run record dataset_ids must not contain empty ids")

        if not model_id or not str(model_id).strip():
            raise ValueError("run record requires a non-empty model_id")

        effective_seed = seed
        if effective_seed is None:
            effective_seed = self._effective_seed
        if effective_seed is None:
            effective_seed = self.resolve_seed(config)

        return RunRecord(
            config=config,
            dataset_ids=list(dataset_ids),
            model_id=str(model_id),
            seed=int(effective_seed),
            applied_defaults=dict(applied_defaults or {}),
        )

    # -------------------------------------------------------------- persistence

    def persist(
        self,
        run_record: RunRecord,
        metrics: Any,
        output_path: Union[str, os.PathLike[str]],
    ) -> Optional[ErrorRecord]:
        """Persist ``run_record`` together with ``metrics`` durably as JSON (Req 13.4).

        The run record and its metrics are written under a single document so they
        remain associated. The bytes are flushed and ``fsync``'d so the data survives
        process termination.

        Returns ``None`` on success. On any failure (Req 13.5) returns an
        :class:`ErrorRecord` naming the failed persistence operation rather than
        raising, so the caller can refrain from reporting the run as successful.
        """
        try:
            document = {
                "run_record": _to_serializable(run_record),
                "metrics": _to_serializable(metrics),
            }
            payload = json.dumps(document, indent=2, sort_keys=True, default=str)

            path = Path(output_path)
            if path.parent and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

            return None
        except Exception as exc:  # noqa: BLE001 - failure must be reported, not raised
            return ErrorRecord(
                failed_component="ReproducibilityManager",
                reason=f"run-record-and-metrics-persistence failed: {exc!r}",
            )

    def persist_learned_rule_store(
        self,
        store: Any,
        output_path: Union[str, os.PathLike[str]],
    ) -> Optional[ErrorRecord]:
        """Persist the versioned Learned_Rule_Store durably as JSON (Req 14.7).

        The store is serialized with its version identifier (via
        :func:`nsr.models.learning.store_to_dict`) so the Learned_Rules and their
        Rule_Provenance survive process termination. The bytes are flushed and
        ``fsync``'d. Returns ``None`` on success, or an :class:`ErrorRecord` naming the
        failed persistence operation on any failure rather than raising (Req 13.5).
        """
        try:
            from .models.learning import LearnedRuleStore, store_to_dict

            document = store_to_dict(store) if isinstance(store, LearnedRuleStore) else store
            payload = json.dumps(document, indent=2, sort_keys=True, default=str)

            path = Path(output_path)
            if path.parent and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

            return None
        except Exception as exc:  # noqa: BLE001 - failure must be reported, not raised
            return ErrorRecord(
                failed_component="ReproducibilityManager",
                reason=f"learned-rule-store-persistence failed: {exc!r}",
            )


def _to_serializable(value: Any) -> Any:
    """Recursively convert dataclasses/containers into JSON-serializable structures."""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _to_serializable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value
