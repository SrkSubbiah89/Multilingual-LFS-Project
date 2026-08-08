"""
backend/rag/build_official_isco08_collections.py

Conference I Reviewer #2 response, Task 21: operator-only builder for
the versioned official ISCO-08 Qdrant collections. Adapts the existing
official-catalogue loader (backend/rag/official_isco08_catalogue.py) --
this module implements no new validation algorithm of its own; it only
turns already-validated records into a collection plan.

Dry-run only in this task
---------------------------
Without ``--execute``, this module validates the supplied official
catalogue (via the existing loader) and prints/returns the resulting
collection plan: five versioned collection names, their per-level record
counts, and the source catalogue hash. It never instantiates a
``QdrantClient``, a ``SentenceTransformer``, or any classifier.

``--execute`` is accepted as a CLI flag (per this task's own required
shape) but is refused unconditionally by ``main()`` -- Task 21's own
brief states live execution "may be used only in a later separately
approved task." This module never performs a live Qdrant write, not
even if a caller passes ``--execute`` by mistake.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from backend.rag.official_isco08_catalogue import (
    DEFAULT_METADATA_PATH,
    DEFAULT_PROFILE,
    LEVELS,
    PROFILE_COLLECTION_NAMES,
    OfficialISCO08CatalogueError,
    load_official_catalogue,
    records_by_level,
)


class UnknownOfficialProfileError(Exception):
    """Raised when a profile has no registered collection-name mapping."""


class OfficialCollectionBuildNotApproved(Exception):
    """Raised whenever ``--execute`` is requested. Task 21 implements the
    dry-run planning path only; live collection execution is explicitly
    gated for a separately approved future task and is never performed
    by this module, regardless of caller intent."""


@dataclass(frozen=True)
class CollectionPlanEntry:
    name: str
    level: str
    record_count: int
    profile: str
    source_catalogue_sha256: str


@dataclass(frozen=True)
class CollectionPlan:
    profile: str
    catalogue_path: str
    metadata_path: str
    entries: list  # list[CollectionPlanEntry]


def collection_names_for_profile(profile: str) -> dict[str, str]:
    if profile not in PROFILE_COLLECTION_NAMES:
        raise UnknownOfficialProfileError(
            f"profile {profile!r} has no registered collection-name mapping; "
            f"known profiles: {sorted(PROFILE_COLLECTION_NAMES)}"
        )
    return PROFILE_COLLECTION_NAMES[profile]


def build_plan(
    catalogue_path: Path,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    profile: str = DEFAULT_PROFILE,
    expected_counts: Optional[dict] = None,
) -> CollectionPlan:
    """Validate the official catalogue (via the existing loader -- no
    duplicated validation logic here) and return a dry-run collection
    plan. Never instantiates Qdrant, an embedding model, or a classifier;
    performs no network or Qdrant call of any kind."""
    names = collection_names_for_profile(profile)
    records = load_official_catalogue(
        catalogue_path, metadata_path, profile=profile, expected_counts=expected_counts,
    )
    by_level = records_by_level(records)
    source_hash = records[0].source_catalogue_sha256 if records else ""

    entries = [
        CollectionPlanEntry(
            name=names[level], level=level, record_count=len(by_level[level]),
            profile=profile, source_catalogue_sha256=source_hash,
        )
        for level in LEVELS
    ]
    # Flat collection: a distinctly-named collection over the same
    # unit-level record set (one record per official four-digit unit
    # group, zero 1/2/3-digit records) -- not a duplicate data source,
    # just a second collection identity for direct unfiltered retrieval.
    entries.append(CollectionPlanEntry(
        name=names["flat"], level="unit", record_count=len(by_level["unit"]),
        profile=profile, source_catalogue_sha256=source_hash,
    ))

    return CollectionPlan(
        profile=profile,
        catalogue_path=str(catalogue_path),
        metadata_path=str(metadata_path),
        entries=entries,
    )


def plan_to_dict(plan: CollectionPlan) -> dict:
    return {
        "profile": plan.profile,
        "catalogue_path": plan.catalogue_path,
        "metadata_path": plan.metadata_path,
        "entries": [asdict(e) for e in plan.entries],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalogue", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.execute:
        print(
            "REFUSED: --execute is not implemented by this task (Task 21) and "
            "requires a separately approved future task. Re-run without "
            "--execute to see the validated dry-run plan."
        )
        raise SystemExit(1)

    try:
        plan = build_plan(args.catalogue, args.metadata, args.profile)
    except (OfficialISCO08CatalogueError, UnknownOfficialProfileError) as exc:
        print(f"CATALOGUE/PROFILE VALIDATION FAILURE: {exc}")
        raise SystemExit(1) from exc

    print(json.dumps(plan_to_dict(plan), indent=2, ensure_ascii=False))
    print("\n--dry-run: no Qdrant collection was created, connected to, or modified.")


if __name__ == "__main__":
    main()
