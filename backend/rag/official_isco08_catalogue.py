"""
backend/rag/official_isco08_catalogue.py

Conference I Reviewer #2 response, Task 21: production loader for the
verified, primary-source official ILO ISCO-08 catalogue (Task 20). This
module never embeds a catalogue row list itself -- it reads two local
file paths supplied by the caller: a normalized catalogue CSV (the
``level,code,parent_code,label`` shape ``eval/normalize_ilo_isco08_
catalogue.py`` produces from the official ILO workbook) and the
project's verified-count metadata file (``eval/verified_catalogue_
counts.yaml``, Task 20).

Fail-closed contract
---------------------
``load_official_catalogue()`` raises ``OfficialISCO08CatalogueError``
(never returns a partial record list) on: a missing catalogue or
metadata file, malformed/incomplete metadata, a catalogue SHA-256 that
does not match the metadata's recorded ``normalized_catalogue_sha256``,
observed per-level counts that differ from the metadata's own
``verified_counts`` (or the fixed official figures 10/43/130/436, both
are checked), a missing required CSV column, a malformed/duplicate/
wrong-length code, an invalid or out-of-order parent link, or a blank
title.

WISCO independence
--------------------
This module takes only a catalogue path and a metadata path as input.
It contains no reference to WISCO, ``backend/evaluation/wisco/``, or
``eval/local_benchmarks/`` anywhere in its source, and never reads any
file other than the two explicitly supplied paths.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_PROFILE = "official_ilo2021_v1"
E5LARGE_PROFILE = "official_ilo2021_v1_e5large"
DEFAULT_METADATA_PATH = Path(__file__).resolve().parents[2] / "eval" / "verified_catalogue_counts.yaml"

LEVELS = ("major", "submajor", "minor", "unit")
_LEVEL_CODE_LENGTH = {"major": 1, "submajor": 2, "minor": 3, "unit": 4}
_LEVEL_PARENT = {"major": None, "submajor": "major", "minor": "submajor", "unit": "minor"}
_CODE_RE = re.compile(r"^\d+$")
_REQUIRED_CSV_COLUMNS = ("level", "code", "parent_code", "label")

# Fixed official ISCO-08 hierarchy counts (ILO, 2021 revision) -- checked
# independently of whatever the metadata file happens to say, so a
# corrupted/edited metadata file can never silently authorize a different
# catalogue shape.
OFFICIAL_EXPECTED_COUNTS = {"major": 10, "submajor": 43, "minor": 130, "unit": 436}

# Single source of truth for versioned official collection names, shared
# by backend/rag/hierarchical_store.py (retrieval) and backend/rag/
# build_official_isco08_collections.py (dry-run planning) so the two can
# never drift apart. Collection-name suffixes are declared explicitly
# per profile (NOT derived by string-formatting the profile name) since
# this task's own literal spec uses a different suffix ("_ilo2021_v1")
# than the profile identifier itself ("official_ilo2021_v1").
PROFILE_COLLECTION_NAMES: dict[str, dict[str, str]] = {
    DEFAULT_PROFILE: {
        "major": "isco08_major_groups_ilo2021_v1",
        "submajor": "isco08_submajor_groups_ilo2021_v1",
        "minor": "isco08_minor_groups_ilo2021_v1",
        "unit": "isco08_unit_groups_ilo2021_v1",
        "flat": "isco08_unit_groups_flat_ilo2021_v1",
    },
    # 2026-08-24: same official ILO 2021 catalogue records, embedded with
    # intfloat/multilingual-e5-large (1024-dim) instead of -small (384-dim)
    # -- a direct retrieval-recall comparison earlier this project measured
    # +11.1pp Recall@1 for -large over -small. Distinct collection names
    # (required -- Qdrant collections are fixed-dimension, so a larger
    # vector size can never reuse the -small collections) and a distinct
    # profile identifier so this is an additive, opt-in comparator, never a
    # silent change to the already-published official_ilo2021_v1 Tier-1
    # result.
    E5LARGE_PROFILE: {
        "major": "isco08_major_groups_ilo2021_v1_e5large",
        "submajor": "isco08_submajor_groups_ilo2021_v1_e5large",
        "minor": "isco08_minor_groups_ilo2021_v1_e5large",
        "unit": "isco08_unit_groups_ilo2021_v1_e5large",
        "flat": "isco08_unit_groups_flat_ilo2021_v1_e5large",
    },
}

# Per-profile embedding identity. Every profile not listed here defaults to
# (intfloat/multilingual-e5-small, 384) -- the project's long-standing
# default -- via PROFILE_EMBEDDING_CONFIG.get(profile, _DEFAULT_EMBEDDING).
_DEFAULT_EMBEDDING = ("intfloat/multilingual-e5-small", 384)
PROFILE_EMBEDDING_CONFIG: dict[str, tuple[str, int]] = {
    E5LARGE_PROFILE: ("intfloat/multilingual-e5-large", 1024),
}


def embedding_config_for_profile(profile: str) -> tuple[str, int]:
    """Return (model_name, vector_dim) for *profile*. Every profile not
    explicitly listed in PROFILE_EMBEDDING_CONFIG (i.e. every profile that
    existed before 2026-08-24) resolves to the unchanged
    intfloat/multilingual-e5-small / 384 default -- this function can only
    ever return something *different* for a profile that opts in."""
    return PROFILE_EMBEDDING_CONFIG.get(profile, _DEFAULT_EMBEDDING)


class OfficialISCO08CatalogueError(Exception):
    """Raised on any fail-closed violation while loading the official
    ISCO-08 catalogue. Callers must not catch this and proceed with a
    partial record list -- none is ever returned when this is raised."""


@dataclass(frozen=True)
class OfficialCatalogueRecord:
    code: str
    level: str
    parent_code: str
    title_en: str
    embedding_text: str
    profile: str
    source_catalogue_sha256: str


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified_metadata(metadata_path: Path = DEFAULT_METADATA_PATH) -> dict:
    """Read and structurally validate ``eval/verified_catalogue_counts.yaml``'s
    ``isco08`` entry. Raises OfficialISCO08CatalogueError on any structural
    problem -- this function never returns a partially-populated dict."""
    if not Path(metadata_path).exists():
        raise OfficialISCO08CatalogueError(f"verified-catalogue metadata file not found: {metadata_path}")
    try:
        data = yaml.safe_load(Path(metadata_path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise OfficialISCO08CatalogueError(f"metadata file {metadata_path} is not valid YAML: {exc}") from exc

    if not isinstance(data, dict) or "isco08" not in data:
        raise OfficialISCO08CatalogueError(f"metadata file {metadata_path} has no top-level 'isco08' entry")
    entry = data["isco08"]
    if not isinstance(entry, dict):
        raise OfficialISCO08CatalogueError(f"metadata file {metadata_path}'s 'isco08' entry is not a mapping")

    required_keys = ("normalized_catalogue_sha256", "verified_counts")
    missing = [k for k in required_keys if k not in entry]
    if missing:
        raise OfficialISCO08CatalogueError(
            f"metadata file {metadata_path}'s 'isco08' entry is missing required key(s): {missing}"
        )
    counts = entry["verified_counts"]
    if not isinstance(counts, dict) or set(counts) != set(LEVELS):
        raise OfficialISCO08CatalogueError(
            f"metadata file {metadata_path}'s 'isco08.verified_counts' must have exactly "
            f"keys {sorted(LEVELS)}, got {counts!r}"
        )
    if not isinstance(entry["normalized_catalogue_sha256"], str) or not entry["normalized_catalogue_sha256"]:
        raise OfficialISCO08CatalogueError(
            f"metadata file {metadata_path}'s 'isco08.normalized_catalogue_sha256' must be a nonblank string"
        )
    return entry


def _load_and_validate_rows(catalogue_path: Path) -> list[dict]:
    with catalogue_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing_cols = [c for c in _REQUIRED_CSV_COLUMNS if c not in fieldnames]
        if missing_cols:
            raise OfficialISCO08CatalogueError(
                f"catalogue {catalogue_path} is missing required column(s) {missing_cols}; "
                f"found: {fieldnames}"
            )
        rows = list(reader)

    seen_codes_by_level: dict[str, set] = {lv: set() for lv in LEVELS}
    for row_number, row in enumerate(rows, start=2):
        level = (row.get("level") or "").strip()
        code = (row.get("code") or "").strip()
        parent_code = (row.get("parent_code") or "").strip()
        title = (row.get("label") or "").strip()

        if level not in LEVELS:
            raise OfficialISCO08CatalogueError(f"row {row_number}: unknown level {level!r} (expected one of {LEVELS})")
        if not code or not _CODE_RE.match(code):
            raise OfficialISCO08CatalogueError(f"row {row_number}: blank or non-numeric code {row.get('code')!r} at level {level!r}")
        if len(code) != _LEVEL_CODE_LENGTH[level]:
            raise OfficialISCO08CatalogueError(
                f"row {row_number}: code {code!r} has length {len(code)}, expected "
                f"{_LEVEL_CODE_LENGTH[level]} for level {level!r}"
            )
        if code in seen_codes_by_level[level]:
            raise OfficialISCO08CatalogueError(f"row {row_number}: duplicate code {code!r} at level {level!r}")

        parent_level = _LEVEL_PARENT[level]
        if parent_level is None:
            if parent_code:
                raise OfficialISCO08CatalogueError(
                    f"row {row_number}: level {level!r} is top-level; parent_code must be blank, got {parent_code!r}"
                )
        else:
            if not parent_code:
                raise OfficialISCO08CatalogueError(f"row {row_number}: level {level!r} requires a non-empty parent_code")
            if parent_code not in seen_codes_by_level[parent_level]:
                raise OfficialISCO08CatalogueError(
                    f"row {row_number}: code {code!r}'s parent_code {parent_code!r} has not appeared "
                    f"yet among level {parent_level!r} -- rows must be in top-down order"
                )

        if not title:
            raise OfficialISCO08CatalogueError(f"row {row_number}: blank title (label) for code {code!r} at level {level!r}")

        seen_codes_by_level[level].add(code)

    return rows


def load_official_catalogue(
    catalogue_path: Path,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    profile: str = DEFAULT_PROFILE,
    expected_counts: Optional[dict[str, int]] = None,
) -> list[OfficialCatalogueRecord]:
    """Load, hash-verify, count-verify, and structurally validate the
    normalized official ISCO-08 catalogue at *catalogue_path* against
    *metadata_path*. Raises OfficialISCO08CatalogueError and returns
    nothing on any violation -- never a partial record list.

    *expected_counts* defaults to the real official ILO figures
    (``OFFICIAL_EXPECTED_COUNTS``, 10/43/130/436) for production use.
    Tests pass a small, fixture-equivalent dict instead of requiring a
    619-row synthetic workbook -- the counts are always cross-checked
    against *metadata_path*'s own recorded ``verified_counts`` too, so a
    caller cannot use a lenient *expected_counts* to bypass what the
    trusted metadata file actually records."""
    catalogue_path = Path(catalogue_path)
    if not catalogue_path.exists():
        raise OfficialISCO08CatalogueError(f"catalogue file not found: {catalogue_path}")

    metadata = load_verified_metadata(metadata_path)
    expected = expected_counts if expected_counts is not None else OFFICIAL_EXPECTED_COUNTS

    actual_hash = _sha256_file(catalogue_path)
    expected_hash = metadata["normalized_catalogue_sha256"]
    if actual_hash != expected_hash:
        raise OfficialISCO08CatalogueError(
            f"catalogue {catalogue_path} sha256 mismatch: expected {expected_hash}, got {actual_hash}"
        )

    rows = _load_and_validate_rows(catalogue_path)

    counts: dict[str, int] = {lv: 0 for lv in LEVELS}
    for row in rows:
        counts[row["level"].strip()] += 1

    for lv in LEVELS:
        if counts[lv] != expected[lv]:
            raise OfficialISCO08CatalogueError(
                f"catalogue {catalogue_path} has {counts[lv]} {lv!r}-level codes; "
                f"expected {expected[lv]}"
            )
        metadata_expected = metadata["verified_counts"].get(lv)
        if counts[lv] != metadata_expected:
            raise OfficialISCO08CatalogueError(
                f"catalogue {catalogue_path} has {counts[lv]} {lv!r}-level codes; "
                f"metadata {metadata_path} records verified_counts.{lv}={metadata_expected}"
            )

    records: list[OfficialCatalogueRecord] = []
    for row in rows:
        code = row["code"].strip()
        title = row["label"].strip()
        records.append(OfficialCatalogueRecord(
            code=code,
            level=row["level"].strip(),
            parent_code=row["parent_code"].strip(),
            title_en=title,
            embedding_text=f"{code} {title}",
            profile=profile,
            source_catalogue_sha256=actual_hash,
        ))
    return records


def records_by_level(records: list[OfficialCatalogueRecord]) -> dict[str, list[OfficialCatalogueRecord]]:
    by_level: dict[str, list[OfficialCatalogueRecord]] = {lv: [] for lv in LEVELS}
    for r in records:
        by_level[r.level].append(r)
    return by_level
