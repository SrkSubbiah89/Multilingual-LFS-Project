"""
eval/legacy824/dataset_gate.py

Task 39: refuses any heldout WISCO path or row count before any
classification is attempted. Also validates the development input
schema (four-digit gold ISCO-08 codes only, unique case IDs) without
inventing, correcting, or silently dropping a malformed row.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

EXPECTED_DEV_ROW_COUNT = 2013
EXPECTED_HELDOUT_ROW_COUNT = 18747

# Substrings that, if present anywhere in a candidate input path, mark
# it as a heldout artifact -- refused outright regardless of row count.
_HELDOUT_PATH_MARKERS = ("heldout",)

_ISCO4_RE = re.compile(r"^\d{4}$")


class HeldoutAccessRefusedError(ValueError):
    """Raised when a caller attempts to point the adapter at a heldout
    path or a row count matching the heldout split."""


class MalformedDevRowError(ValueError):
    """Raised when a development row's gold code is missing or not a
    valid four-digit ISCO-08 code. The row is rejected, never repaired."""


def assert_not_heldout_path(path: Path | str) -> None:
    name = str(path).lower()
    for marker in _HELDOUT_PATH_MARKERS:
        if marker in name:
            raise HeldoutAccessRefusedError(
                f"refusing path {path!r}: filename indicates a heldout artifact"
            )


def load_and_validate_dev_rows(path: Path | str) -> list[dict]:
    """
    Reads *path* as the WISCO development-split CSV, refuses it outright
    if it looks like a heldout file (by name) or has the heldout row
    count, and validates every row's gold code. Returns the validated
    row list. Raises (never silently drops) on any malformed row.
    """
    path = Path(path)
    assert_not_heldout_path(path)

    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) == EXPECTED_HELDOUT_ROW_COUNT:
        raise HeldoutAccessRefusedError(
            f"refusing {path}: row count ({len(rows)}) matches the heldout split size"
        )
    if len(rows) != EXPECTED_DEV_ROW_COUNT:
        raise ValueError(
            f"{path}: expected exactly {EXPECTED_DEV_ROW_COUNT} development rows, got {len(rows)}"
        )

    ids = [r["case_id"] for r in rows]
    if len(set(ids)) != len(ids):
        seen, dupes = set(), []
        for i in ids:
            if i in seen:
                dupes.append(i)
            seen.add(i)
        raise ValueError(f"{path}: duplicate case_id values: {dupes[:20]}")

    bad = [r["case_id"] for r in rows if not _ISCO4_RE.match((r.get("gold_isco_4digit") or "").strip())]
    if bad:
        raise MalformedDevRowError(
            f"{path}: {len(bad)} row(s) have a missing/malformed gold_isco_4digit "
            f"(must be exactly 4 digits): {bad[:20]}"
        )

    return rows
