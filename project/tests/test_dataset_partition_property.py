"""Property-based test for dataset validation partitioning (Task 12.2).

Property 11: Loading partitions items into retained-valid and excluded-invalid.

For any mix of valid and invalid items, every valid item is retained and every invalid
item is excluded and logged; the retained-valid and excluded-invalid sets are disjoint
and together cover the input.

Duplicate-id handling (aligned to the loader's documented behavior): among structurally
valid items, the *first* occurrence of an identifier is retained and any later item
carrying an already-retained identifier is excluded as a duplicate. Only retained
identifiers populate the loader's seen-id set, so duplicate detection compares against
successfully retained items only.

**Validates: Requirements 10.2, 10.3, 10.5**
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from nsr.dataset_loader import ExclusionRecord, load_dataset
from nsr.models import Domain

# The six recognized benchmark domain label values (Req 10.2, 10.4).
_DOMAIN_VALUES: list[str] = [d.value for d in Domain]

# Non-empty domain labels guaranteed NOT to be any of the six recognized values
# (Req 10.5). An empty/whitespace label is a *missing field*, not a bad label, so it is
# deliberately excluded here and exercised by the ``missing_domain`` kind instead.
_BAD_DOMAIN_LABELS: list[str] = ["astrology", "unknown", "xyz", "not-a-domain"]

# The kinds of item the generator can emit. Each maps to a known expected outcome.
_KINDS: list[str] = [
    "valid",          # structurally valid, unique id  -> retained
    "dup",            # structurally valid, duplicate id -> excluded (duplicate)
    "empty_id",       # empty/whitespace id            -> excluded (missing field)
    "empty_query",    # empty/whitespace query         -> excluded (missing field)
    "empty_gt",       # empty/whitespace ground truth  -> excluded (missing field)
    "missing_domain", # domain key absent              -> excluded (missing field)
    "bad_domain",     # unrecognized domain label      -> excluded (bad label)
]


@st.composite
def _dataset(draw):
    """Build a raw-item list with a parallel list of expected per-item outcome tags.

    Identifiers for structurally valid items are made unique by construction (keyed on
    their slot index) so that the only duplicates are the ones we deliberately inject.
    Expected outcome tags are one of: ``retain``, ``missing``, ``bad``, ``dup``.
    """
    n = draw(st.integers(min_value=0, max_value=15))
    raw_items: list[dict] = []
    expected: list[str] = []
    retained_ids: list[str] = []  # ids that will be retained, in retention order

    for i in range(n):
        kind = draw(st.sampled_from(_KINDS))
        slot_id = f"id-{i}"
        domain = draw(st.sampled_from(_DOMAIN_VALUES))

        if kind == "valid":
            raw_items.append(
                {"item_id": slot_id, "query": "q", "ground_truth": "a", "domain": domain}
            )
            expected.append("retain")
            retained_ids.append(slot_id)

        elif kind == "dup":
            if retained_ids:
                dup_id = draw(st.sampled_from(retained_ids))
                raw_items.append(
                    {
                        "item_id": dup_id,
                        "query": "q",
                        "ground_truth": "a",
                        "domain": domain,
                    }
                )
                expected.append("dup")
            else:
                # No prior retained id to duplicate; emit a normal valid item instead.
                raw_items.append(
                    {
                        "item_id": slot_id,
                        "query": "q",
                        "ground_truth": "a",
                        "domain": domain,
                    }
                )
                expected.append("retain")
                retained_ids.append(slot_id)

        elif kind == "empty_id":
            raw_items.append(
                {
                    "item_id": draw(st.sampled_from(["", "   "])),
                    "query": "q",
                    "ground_truth": "a",
                    "domain": domain,
                }
            )
            expected.append("missing")

        elif kind == "empty_query":
            raw_items.append(
                {
                    "item_id": slot_id,
                    "query": draw(st.sampled_from(["", "   "])),
                    "ground_truth": "a",
                    "domain": domain,
                }
            )
            expected.append("missing")

        elif kind == "empty_gt":
            raw_items.append(
                {
                    "item_id": slot_id,
                    "query": "q",
                    "ground_truth": draw(st.sampled_from(["", "   "])),
                    "domain": domain,
                }
            )
            expected.append("missing")

        elif kind == "missing_domain":
            raw_items.append({"item_id": slot_id, "query": "q", "ground_truth": "a"})
            expected.append("missing")

        else:  # bad_domain
            raw_items.append(
                {
                    "item_id": slot_id,
                    "query": "q",
                    "ground_truth": "a",
                    "domain": draw(st.sampled_from(_BAD_DOMAIN_LABELS)),
                }
            )
            expected.append("bad")

    return raw_items, expected


@given(_dataset())
def test_loading_partitions_into_retained_and_excluded(data) -> None:
    """Retained-valid and excluded-invalid sets are disjoint and cover the input.

    Every structurally valid, uniquely identified item is retained; every item that is
    missing a required field, carries an unrecognized domain, or duplicates an already
    retained identifier is excluded and logged.

    **Validates: Requirements 10.2, 10.3, 10.5**
    """
    raw_items, expected = data
    result = load_dataset(raw_items)

    retained = result.items
    exclusions = result.report.exclusions
    report = result.report

    # --- Coverage: the two outcomes together account for every input item ---
    assert len(retained) + len(exclusions) == len(raw_items)
    assert report.total_loaded == len(raw_items)
    assert report.total_validated == len(retained)
    assert report.total_excluded == len(exclusions)

    # --- Partition matches the expected classification (disjoint by construction) ---
    expected_retain = [i for i, tag in enumerate(expected) if tag == "retain"]
    expected_excluded = [i for i, tag in enumerate(expected) if tag != "retain"]
    assert len(retained) == len(expected_retain)
    assert len(exclusions) == len(expected_excluded)

    # --- Every retained item is genuinely valid (Req 10.2) ---
    for item in retained:
        assert isinstance(item.item_id, str) and item.item_id.strip() != ""
        assert isinstance(item.query, str) and item.query.strip() != ""
        assert isinstance(item.ground_truth, str) and item.ground_truth.strip() != ""
        assert item.domain in set(Domain)

    # Retained identifiers are unique: the retained set is internally disjoint.
    retained_ids = [item.item_id for item in retained]
    assert len(retained_ids) == len(set(retained_ids))

    # Retained items are exactly the expected-valid ones, in input order.
    assert retained_ids == [raw_items[i]["item_id"] for i in expected_retain]

    # --- Every excluded item is logged with a reason (Req 10.3, 10.5) ---
    assert all(isinstance(rec, ExclusionRecord) for rec in exclusions)
    assert all(rec.reason for rec in exclusions)

    # The recorded exclusion categories match the expected invalid kinds.
    missing_field_recs = [r for r in exclusions if r.missing_field is not None]
    bad_label_recs = [r for r in exclusions if r.bad_label is not None]
    duplicate_recs = [
        r for r in exclusions if r.missing_field is None and r.bad_label is None
    ]
    assert len(missing_field_recs) == sum(1 for t in expected if t == "missing")
    assert len(bad_label_recs) == sum(1 for t in expected if t == "bad")
    assert len(duplicate_recs) == sum(1 for t in expected if t == "dup")

    # A missing-field record always names one of the four required fields (Req 10.3).
    for rec in missing_field_recs:
        assert rec.missing_field in {"item_id", "query", "ground_truth", "domain"}


@given(_dataset())
def test_no_input_item_is_both_retained_and_excluded(data) -> None:
    """Disjointness in counts: validated and excluded totals sum to items loaded.

    No input item can be simultaneously retained and excluded, so the validated and
    excluded counts form an exact partition of the loaded total.

    **Validates: Requirements 10.2, 10.3, 10.5**
    """
    raw_items, _ = data
    result = load_dataset(raw_items)
    report = result.report

    assert report.total_validated + report.total_excluded == report.total_loaded
    # Per-domain and unassigned accounting also covers every loaded item exactly once.
    per_domain_loaded = sum(stats.loaded for stats in report.per_domain.values())
    assert per_domain_loaded + report.unassigned_loaded == report.total_loaded
