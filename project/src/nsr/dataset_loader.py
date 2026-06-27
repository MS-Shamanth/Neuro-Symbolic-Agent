"""Dataset Loader for the Neuro-Symbolic System-2 Reasoning Architecture.

Implements the dataset-loading and validation portion of Requirement 10 (Evaluation
Datasets and Step-Level Benchmark):

- Validates, before evaluation begins, that each item has a non-empty unique
  identifier, a non-empty query, a non-empty ground-truth final answer, and a
  recognized domain label (Req 10.2).
- Excludes items missing a required field, retains all remaining valid items, and
  records the excluded item's identifier together with the missing field name in the
  run log (Req 10.3).
- Associates each retained item with exactly one of the six benchmark domains so that
  metrics can be reported per domain (Req 10.4).
- Excludes items whose domain label is not one of the six benchmark domains and records
  the excluded item's identifier together with the unrecognized label (Req 10.5).
- Records, when loading completes, the total number of items loaded, validated, and
  excluded per domain (Req 10.6).

The loader takes raw items (a sequence of mappings, e.g. parsed JSON) and returns a
:class:`DatasetLoadResult` holding the retained :class:`~nsr.models.DatasetItem`
objects, a per-domain :class:`LoadReport`, and the run-log of :class:`ExclusionRecord`
entries. The retained-valid and excluded-invalid sets partition the input: they are
disjoint and together cover every input item.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .models import DatasetItem, Domain

#: Placeholder identifier used in the run log when an item has no usable identifier.
MISSING_ID_PLACEHOLDER = "<missing-id>"

#: The four required fields on a raw dataset item, checked in this order.
REQUIRED_FIELDS: tuple[str, ...] = ("item_id", "query", "ground_truth", "domain")


@dataclass(frozen=True)
class ExclusionRecord:
    """A single run-log entry describing why an input item was excluded.

    Exactly one of ``missing_field`` or ``bad_label`` is set for field/label problems;
    duplicate-identifier exclusions set neither and rely on ``reason``.
    """

    item_id: str
    reason: str
    missing_field: Optional[str] = None
    bad_label: Optional[str] = None


@dataclass
class DomainStats:
    """Per-domain load counters (Req 10.6).

    ``loaded`` counts input items carrying this (recognized) domain label, ``validated``
    counts those retained, and ``excluded`` counts those dropped for a missing field or
    a duplicate identifier.
    """

    loaded: int = 0
    validated: int = 0
    excluded: int = 0


@dataclass
class LoadReport:
    """Aggregate statistics produced when dataset loading completes (Req 10.6)."""

    total_loaded: int = 0
    total_validated: int = 0
    total_excluded: int = 0
    per_domain: dict[Domain, DomainStats] = field(default_factory=dict)
    #: Loaded items whose domain label was missing or unrecognized and so could not be
    #: attributed to one of the six benchmark domains.
    unassigned_loaded: int = 0
    #: Excluded items that could not be attributed to a recognized domain.
    unassigned_excluded: int = 0
    exclusions: list[ExclusionRecord] = field(default_factory=list)


@dataclass
class DatasetLoadResult:
    """Result of loading a dataset: retained items plus the load report."""

    items: list[DatasetItem]
    report: LoadReport


def _is_nonempty_str(value: Any) -> bool:
    """Return ``True`` when ``value`` is a string with non-whitespace content."""
    return isinstance(value, str) and value.strip() != ""


def _recognize_domain(label: Any) -> Optional[Domain]:
    """Map a raw domain label to a :class:`Domain`, or ``None`` if unrecognized.

    Accepts either a :class:`Domain` member or its string value (case-sensitive against
    the six documented domain values).
    """
    if isinstance(label, Domain):
        return label
    if isinstance(label, str):
        try:
            return Domain(label)
        except ValueError:
            return None
    return None


def _display_id(raw_id: Any) -> str:
    """Return a printable identifier for the run log, even when the id is unusable."""
    if isinstance(raw_id, str) and raw_id.strip() != "":
        return raw_id
    if raw_id is None:
        return MISSING_ID_PLACEHOLDER
    # Non-empty non-string ids are surfaced as their repr so the log stays informative.
    text = str(raw_id).strip()
    return text if text else MISSING_ID_PLACEHOLDER


class DatasetLoader:
    """Validates raw dataset items and partitions them into retained and excluded.

    A single :class:`DatasetLoader` instance is stateless across calls; each
    :meth:`load` call validates one batch of raw items independently.
    """

    def load(self, raw_items: Sequence[Mapping[str, Any]]) -> DatasetLoadResult:
        """Validate ``raw_items`` and return retained items with a load report.

        Every input item is either retained as a valid :class:`DatasetItem` or recorded
        as an :class:`ExclusionRecord`; the two outcomes are mutually exclusive and
        together account for every input item (Req 10.2, 10.3, 10.5).
        """
        report = LoadReport(per_domain={d: DomainStats() for d in Domain})
        items: list[DatasetItem] = []
        seen_ids: set[str] = set()

        for raw in raw_items:
            report.total_loaded += 1

            raw_id = raw.get("item_id")
            query = raw.get("query")
            ground_truth = raw.get("ground_truth")
            domain_label = raw.get("domain")

            recognized = _recognize_domain(domain_label)

            # --- Loaded-per-domain accounting (Req 10.6) ---
            if recognized is not None:
                report.per_domain[recognized].loaded += 1
            else:
                report.unassigned_loaded += 1

            exclusion = self._validate(
                raw_id=raw_id,
                query=query,
                ground_truth=ground_truth,
                domain_label=domain_label,
                recognized=recognized,
                seen_ids=seen_ids,
            )

            if exclusion is not None:
                report.total_excluded += 1
                if recognized is not None:
                    report.per_domain[recognized].excluded += 1
                else:
                    report.unassigned_excluded += 1
                report.exclusions.append(exclusion)
                continue

            # Valid: recognized is guaranteed non-None here.
            assert recognized is not None  # for type-checkers; enforced by _validate
            item_id = raw_id  # known non-empty str by validation
            items.append(
                DatasetItem(
                    item_id=item_id,
                    query=query,
                    ground_truth=ground_truth,
                    domain=recognized,
                )
            )
            seen_ids.add(item_id)
            report.total_validated += 1
            report.per_domain[recognized].validated += 1

        return DatasetLoadResult(items=items, report=report)

    @staticmethod
    def _validate(
        *,
        raw_id: Any,
        query: Any,
        ground_truth: Any,
        domain_label: Any,
        recognized: Optional[Domain],
        seen_ids: set[str],
    ) -> Optional[ExclusionRecord]:
        """Return an :class:`ExclusionRecord` if the item is invalid, else ``None``.

        Checks run in a fixed order so the reported problem is deterministic: missing
        identifier, query, or ground truth (Req 10.3); missing or unrecognized domain
        label (Req 10.3, 10.5); then duplicate identifier (uniqueness, Req 10.2).
        """
        display_id = _display_id(raw_id)

        # Required scalar fields must be non-empty strings (Req 10.2, 10.3).
        if not _is_nonempty_str(raw_id):
            return ExclusionRecord(
                item_id=display_id,
                reason="missing or empty required field 'item_id'",
                missing_field="item_id",
            )
        if not _is_nonempty_str(query):
            return ExclusionRecord(
                item_id=display_id,
                reason="missing or empty required field 'query'",
                missing_field="query",
            )
        if not _is_nonempty_str(ground_truth):
            return ExclusionRecord(
                item_id=display_id,
                reason="missing or empty required field 'ground_truth'",
                missing_field="ground_truth",
            )

        # Domain label must be present (Req 10.3) and recognized (Req 10.5).
        if domain_label is None or (
            isinstance(domain_label, str) and domain_label.strip() == ""
        ):
            return ExclusionRecord(
                item_id=display_id,
                reason="missing or empty required field 'domain'",
                missing_field="domain",
            )
        if recognized is None:
            label_text = (
                domain_label if isinstance(domain_label, str) else str(domain_label)
            )
            return ExclusionRecord(
                item_id=display_id,
                reason=f"unrecognized domain label {label_text!r}",
                bad_label=label_text,
            )

        # Identifier uniqueness (Req 10.2).
        if raw_id in seen_ids:
            return ExclusionRecord(
                item_id=display_id,
                reason=f"duplicate item_id {raw_id!r}",
            )

        return None


def load_dataset(raw_items: Sequence[Mapping[str, Any]]) -> DatasetLoadResult:
    """Module-level convenience wrapper around :meth:`DatasetLoader.load`."""
    return DatasetLoader().load(raw_items)
