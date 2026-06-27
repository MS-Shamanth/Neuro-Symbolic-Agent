"""Evaluation dataset item model."""

from __future__ import annotations

from dataclasses import dataclass

from .enums import Domain


@dataclass
class DatasetItem:
    """A single benchmark evaluation item belonging to exactly one domain."""

    item_id: str
    query: str
    ground_truth: str
    domain: Domain
