"""
backend/rag/build_official_isco08_collections.py

Conference I Reviewer #2 response. Task 21 built the dry-run-only
collection planner for the versioned official ISCO-08 Qdrant
collections. Task 22 adds the actual, guarded, local-only live
execution path -- still never invoked by any command in this project as
of Task 22 (no test, script, or CLI call in this codebase passes
``--execute``; that remains for a separately approved future task to
authorize and run).

Dry-run (default; no code changed since Task 21)
--------------------------------------------------
Without ``--execute`` (or with ``--dry-run``), this module validates the
supplied official catalogue (via the existing loader,
``backend/rag/official_isco08_catalogue.py`` -- no duplicated validation
logic here) and prints/returns the resulting collection plan. It never
instantiates a ``QdrantClient``, a ``SentenceTransformer``, or any
classifier.

Guarded live execution (Task 22, new)
----------------------------------------
``--execute`` now has a real implementation, gated behind BOTH mandatory
acknowledgements (``--confirm-profile official_ilo2021_v1`` exactly, and
``--allow-local-qdrant-mutation``). Missing either fails before any
Qdrant/embedder import or connection. Execution:

1. re-validates the catalogue via ``load_official_catalogue()`` (same
   fail-closed hash/count/format checks as the dry-run path);
2. only THEN lazily imports ``qdrant_client``/``sentence_transformers``
   (see "Lazy dependency boundaries" below) and connects to a
   **local-only** Qdrant instance (``QDRANT_HOST``/``QDRANT_PORT`` env
   vars, defaulting to ``localhost``/``6333`` -- no remote URL/token CLI
   option exists anywhere in this module);
3. refuses if ANY of the five target collection names already exists
   (empty or not) -- no auto-replace, delete, recreate, upsert-into-
   existing, alias swap, or overwrite is ever performed;
4. creates and verifies the five collections in a fixed order (major,
   submajor, minor, unit hierarchical, unit flat), verifying each one's
   exact point count and payload identity immediately after writing it;
5. writes an ignored local JSON manifest only after every collection is
   verified (success), or a failure manifest identifying exactly which
   targets were created/verified/partial if anything goes wrong mid-build
   (never deleted or auto-remediated -- a separate explicit task must
   handle that).

Lazy dependency boundaries
-----------------------------
At module import time and throughout the entire dry-run code path, this
module imports NEITHER ``qdrant_client`` NOR ``sentence_transformers``
(confirmed by a hermetic test that AST-inspects this file's *top-level*
import statements only -- the two lazy ``import`` statements inside
``_default_qdrant_client_factory()``/``_default_embedder_factory()``
below are real Python `import` statements, deliberately nested inside
function bodies so they never execute until the guarded execution path
itself calls them, strictly after both CLI acknowledgements and full
catalogue validation succeed).

Reused local embedding configuration
----------------------------------------
``_EMBEDDING_MODEL_NAME`` / ``_EMBEDDING_VECTOR_DIM`` / the ``"passage: "``
E5 prefix convention below are the exact same values already used by
``backend/rag/hierarchical_store.py`` and ``backend/rag/load_full_isco.py``
(``intfloat/multilingual-e5-small``, 384-dim, cosine distance) -- copied
as literal constants here, not imported from those modules, because both
of them import ``qdrant_client``/``sentence_transformers`` at their own
module top level, which would defeat this module's lazy-import
guarantee above. No new embedding dimension or model was invented for
this task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from backend.rag.official_isco08_catalogue import (
    DEFAULT_METADATA_PATH,
    DEFAULT_PROFILE,
    LEVELS,
    PROFILE_COLLECTION_NAMES,
    OfficialCatalogueRecord,
    OfficialISCO08CatalogueError,
    load_official_catalogue,
    records_by_level,
)

# Reused, not reinvented -- see module docstring "Reused local embedding
# configuration". Source: backend/rag/hierarchical_store.py's MODEL_NAME/
# VECTOR_DIM and backend/rag/load_full_isco.py's MODEL_NAME/VECTOR_DIM/
# _PREFIX (both currently intfloat/multilingual-e5-small, 384-dim).
_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
_EMBEDDING_VECTOR_DIM = 384
_EMBED_PREFIX = "passage: "
_EMBED_BATCH_SIZE = 32
_UPSERT_BATCH_SIZE = 64

_PAYLOAD_SCHEMA_VERSION = "official_isco08_v1"

# Local-only Qdrant target. No CLI flag or environment variable in this
# module accepts a remote URL, API key, or token of any kind.
_QDRANT_HOST_ENV = "QDRANT_HOST"
_QDRANT_PORT_ENV = "QDRANT_PORT"
_LOCAL_QDRANT_HOST_DEFAULT = "localhost"
_LOCAL_QDRANT_PORT_DEFAULT = 6333

_ISCO4_RE = re.compile(r"^[0-9]{4}$")

# collection_role -> the OfficialCatalogueRecord.level its records must
# all carry (both unit_hierarchical and unit_flat draw from the "unit"
# level -- two distinct collection identities over the same 436 records).
_ROLE_SOURCE_LEVEL = {
    "major": "major", "submajor": "submajor", "minor": "minor",
    "unit_hierarchical": "unit", "unit_flat": "unit",
}
_TARGET_ROLE_ORDER = ("major", "submajor", "minor", "unit_hierarchical", "unit_flat")


class UnknownOfficialProfileError(Exception):
    """Raised when a profile has no registered collection-name mapping."""


class BuildAcknowledgementError(Exception):
    """Raised when --execute is requested without both required
    acknowledgements, or with a profile/confirm-profile mismatch. Always
    raised before any Qdrant/embedder import or connection."""


class BuildPreflightError(Exception):
    """Raised when a target collection already exists (empty or not) --
    always raised before any collection is created or written to."""


class BuildExecutionError(Exception):
    """Raised on any failure during or after collection creation. A
    failure manifest identifying exactly which targets were created/
    verified/partial is always written before this is raised; no target
    is ever deleted or auto-remediated."""


# ---------------------------------------------------------------------------
# Dry-run plan (Task 21, unchanged)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CollectionPlanEntry:
    name: str
    level: str
    record_count: int
    profile: str
    source_catalogue_sha256: str
    collection_role: str = ""


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
            collection_role=("unit_hierarchical" if level == "unit" else level),
        )
        for level in LEVELS
    ]
    entries.append(CollectionPlanEntry(
        name=names["flat"], level="unit", record_count=len(by_level["unit"]),
        profile=profile, source_catalogue_sha256=source_hash,
        collection_role="unit_flat",
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


# ---------------------------------------------------------------------------
# Acknowledgement gate (Task 22)
# ---------------------------------------------------------------------------

def check_execution_acknowledgements(profile: str, confirm_profile: str, allow_local_qdrant_mutation: bool) -> None:
    """Raises BuildAcknowledgementError unless BOTH acknowledgements are
    present and consistent. Pure string/bool logic -- no I/O, no import
    of Qdrant/embedder, called before anything else in execute_build()."""
    if confirm_profile != DEFAULT_PROFILE:
        raise BuildAcknowledgementError(
            f"--confirm-profile must be exactly {DEFAULT_PROFILE!r}; got {confirm_profile!r}"
        )
    if confirm_profile != profile:
        raise BuildAcknowledgementError(
            f"--confirm-profile ({confirm_profile!r}) must exactly equal --profile ({profile!r})"
        )
    if not allow_local_qdrant_mutation:
        raise BuildAcknowledgementError(
            "--allow-local-qdrant-mutation is required in addition to --confirm-profile for --execute"
        )


# ---------------------------------------------------------------------------
# Target build derivation (Task 22) -- deterministic ordering
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetBuild:
    name: str
    collection_role: str
    records: tuple  # tuple[OfficialCatalogueRecord, ...], deterministic order


def target_builds_for_profile(
    by_level: dict[str, list[OfficialCatalogueRecord]], profile: str,
) -> list[TargetBuild]:
    names = collection_names_for_profile(profile)
    builds = []
    for role in _TARGET_ROLE_ORDER:
        source_level = _ROLE_SOURCE_LEVEL[role]
        name = names["flat"] if role == "unit_flat" else names[source_level]
        builds.append(TargetBuild(name=name, collection_role=role, records=tuple(by_level[source_level])))
    return builds


def _stable_point_id(code: str, collection_role: str) -> int:
    """Same scheme as backend/rag/vector_store.py's _stable_id(): first
    15 hex digits of a sha256 hash, a stable 60-bit non-negative integer.
    collection_role is folded in only for defensive uniqueness (each
    collection already has its own independent Qdrant ID space)."""
    payload = f"{collection_role}:{code}"
    return int(hashlib.sha256(payload.encode()).hexdigest()[:15], 16)


def _build_payload(record: OfficialCatalogueRecord, collection_role: str) -> dict:
    return {
        "code": record.code,
        "level": record.level,
        "parent_code": record.parent_code,
        "title_en": record.title_en,
        "profile": record.profile,
        "source_catalogue_sha256": record.source_catalogue_sha256,
        "collection_role": collection_role,
        "embedding_text": record.embedding_text,
    }


# ---------------------------------------------------------------------------
# Lazy dependency factories -- the ONLY two places qdrant_client /
# sentence_transformers may ever be imported in this module.
# ---------------------------------------------------------------------------

def _resolve_local_qdrant_target() -> tuple[str, int]:
    host = os.getenv(_QDRANT_HOST_ENV, _LOCAL_QDRANT_HOST_DEFAULT)
    port = int(os.getenv(_QDRANT_PORT_ENV, _LOCAL_QDRANT_PORT_DEFAULT))
    return host, port


def _default_qdrant_client_factory(host: str, port: int):
    from qdrant_client import QdrantClient  # lazy: see module docstring
    return QdrantClient(host=host, port=port)


def _default_embedder_factory():
    from sentence_transformers import SentenceTransformer  # lazy: see module docstring
    return SentenceTransformer(_EMBEDDING_MODEL_NAME)


# ---------------------------------------------------------------------------
# Guarded live execution (Task 22)
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(manifest: dict, output_manifest_path: Path) -> None:
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _create_and_populate_collection(client, embedder, target: TargetBuild) -> None:
    from qdrant_client.models import Distance, PointStruct, VectorParams  # lazy: see module docstring

    client.create_collection(
        collection_name=target.name,
        vectors_config=VectorParams(size=_EMBEDDING_VECTOR_DIM, distance=Distance.COSINE),
    )

    texts = [f"{_EMBED_PREFIX}{r.embedding_text}" for r in target.records]
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=_EMBED_BATCH_SIZE)

    points = []
    for i, record in enumerate(target.records):
        vec = vectors[i]
        vec = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        points.append(PointStruct(
            id=_stable_point_id(record.code, target.collection_role),
            vector=vec,
            payload=_build_payload(record, target.collection_role),
        ))

    for i in range(0, len(points), _UPSERT_BATCH_SIZE):
        client.upsert(collection_name=target.name, points=points[i:i + _UPSERT_BATCH_SIZE])


def _verify_collection(client, target: TargetBuild) -> dict:
    expected_count = len(target.records)
    observed_count = client.count(collection_name=target.name, exact=True).count
    if observed_count != expected_count:
        raise BuildExecutionError(
            f"collection {target.name!r}: expected {expected_count} points, observed {observed_count}"
        )

    expected_codes = {r.code for r in target.records}
    expected_level = _ROLE_SOURCE_LEVEL[target.collection_role]
    points, _next_offset = client.scroll(collection_name=target.name, limit=max(expected_count, 1), with_payload=True)
    if len(points) != expected_count:
        raise BuildExecutionError(
            f"collection {target.name!r}: expected {expected_count} points on scroll, got {len(points)}"
        )

    seen_codes = set()
    for point in points:
        payload = point.payload
        code = payload.get("code", "")
        if code not in expected_codes:
            raise BuildExecutionError(f"collection {target.name!r}: unexpected code {code!r} in payload")
        if payload.get("level") != expected_level:
            raise BuildExecutionError(
                f"collection {target.name!r}: code {code!r} has level {payload.get('level')!r}, expected {expected_level!r}"
            )
        if payload.get("collection_role") != target.collection_role:
            raise BuildExecutionError(
                f"collection {target.name!r}: code {code!r} has collection_role "
                f"{payload.get('collection_role')!r}, expected {target.collection_role!r}"
            )
        if payload.get("profile") != DEFAULT_PROFILE:
            raise BuildExecutionError(f"collection {target.name!r}: code {code!r} has unexpected profile in payload")
        if not payload.get("source_catalogue_sha256"):
            raise BuildExecutionError(f"collection {target.name!r}: code {code!r} is missing source_catalogue_sha256")
        if target.collection_role == "unit_flat" and not _ISCO4_RE.match(code):
            raise BuildExecutionError(
                f"collection {target.name!r}: non-four-digit code {code!r} found in the flat collection"
            )
        seen_codes.add(code)

    if seen_codes != expected_codes:
        raise BuildExecutionError(f"collection {target.name!r}: verified code set does not match the planned code set")

    return {"name": target.name, "collection_role": target.collection_role,
            "expected_count": expected_count, "observed_count": observed_count, "verified": True}


def execute_build(
    catalogue_path: Path,
    metadata_path: Path,
    profile: str,
    confirm_profile: str,
    allow_local_qdrant_mutation: bool,
    output_manifest_path: Path,
    expected_counts: Optional[dict] = None,
    qdrant_client_factory: Optional[Callable] = None,
    embedder_factory: Optional[Callable] = None,
) -> dict:
    """Guarded live-execution path. `qdrant_client_factory`/
    `embedder_factory` are dependency-injection seams -- tests always
    supply fakes here; production callers omit them and get the real,
    lazily-imported `_default_qdrant_client_factory`/
    `_default_embedder_factory`. No test and no command in this
    repository as of Task 22 calls this function without injecting
    fakes for both.

    Order (see module docstring): acknowledgements -> catalogue
    validation (no Qdrant/embedder import yet) -> lazy client/embedder
    construction -> preflight (all 5 targets absent) -> create+verify
    each target in order -> write manifest.
    """
    # 1. Acknowledgements -- before any import beyond argparse/stdlib.
    check_execution_acknowledgements(profile, confirm_profile, allow_local_qdrant_mutation)

    # 2. Catalogue validation -- reuses load_official_catalogue(); still
    #    no Qdrant/embedder import anywhere in this call chain.
    records = load_official_catalogue(
        catalogue_path, metadata_path, profile=profile, expected_counts=expected_counts,
    )
    by_level = records_by_level(records)
    target_builds = target_builds_for_profile(by_level, profile)
    source_hash = records[0].source_catalogue_sha256 if records else ""

    # 3. Only now: lazily construct Qdrant client + embedder.
    host, port = _resolve_local_qdrant_target()
    client = (qdrant_client_factory or _default_qdrant_client_factory)(host, port)
    embedder = (embedder_factory or _default_embedder_factory)()

    manifest: dict = {
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "builder_script_sha256": _sha256_file(Path(__file__)),
        "catalogue_path": str(catalogue_path),
        "catalogue_sha256": source_hash,
        "metadata_path": str(metadata_path),
        "profile": profile,
        "qdrant_host": host,
        "qdrant_port": port,
        "embedding_model_identity": _EMBEDDING_MODEL_NAME,
        "embedding_vector_dim": _EMBEDDING_VECTOR_DIM,
        "payload_schema_version": _PAYLOAD_SCHEMA_VERSION,
        "acknowledgements": {
            "confirm_profile": confirm_profile,
            "allow_local_qdrant_mutation": allow_local_qdrant_mutation,
        },
        "planned_targets": [
            {"name": tb.name, "collection_role": tb.collection_role, "planned_count": len(tb.records)}
            for tb in target_builds
        ],
        "status": "in_progress",
    }

    # 4. Preflight: ALL five targets must be absent before ANY mutation.
    existing_names = {c.name for c in client.get_collections().collections}
    conflicting = [tb.name for tb in target_builds if tb.name in existing_names]
    if conflicting:
        manifest["status"] = "preflight_failed_existing_target"
        manifest["failure_reason"] = (
            f"target collection(s) already exist -- refusing to create/overwrite: {conflicting}"
        )
        manifest["targets_created_or_partial"] = []
        _write_manifest(manifest, output_manifest_path)
        raise BuildPreflightError(manifest["failure_reason"])

    # 5. Create + verify each target, in deterministic order.
    created_or_partial: list[str] = []
    verified_targets: list[dict] = []
    try:
        for tb in target_builds:
            created_or_partial.append(tb.name)
            _create_and_populate_collection(client, embedder, tb)
            verification = _verify_collection(client, tb)
            verified_targets.append(verification)
    except Exception as exc:
        manifest["status"] = "failed_partial_build"
        manifest["failure_reason"] = str(exc)
        manifest["targets_created_or_partial"] = created_or_partial
        manifest["targets_verified_before_failure"] = verified_targets
        manifest["remediation_note"] = (
            "No collection was deleted or overwritten automatically. Remediation "
            "(inspecting/dropping the partially-built target(s) above) requires a "
            "separate, explicit future task -- this builder never self-heals."
        )
        _write_manifest(manifest, output_manifest_path)
        raise BuildExecutionError(str(exc)) from exc

    manifest["status"] = "success"
    manifest["targets_created_or_partial"] = created_or_partial
    manifest["verified_targets"] = verified_targets
    _write_manifest(manifest, output_manifest_path)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalogue", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-profile", default=None)
    parser.add_argument("--allow-local-qdrant-mutation", action="store_true")
    args = parser.parse_args()

    if args.execute and not args.dry_run:
        if args.output_manifest is None:
            parser.error("--output-manifest is required when --execute is passed.")
        try:
            manifest = execute_build(
                args.catalogue, args.metadata, args.profile,
                confirm_profile=args.confirm_profile or "",
                allow_local_qdrant_mutation=args.allow_local_qdrant_mutation,
                output_manifest_path=args.output_manifest,
            )
        except BuildAcknowledgementError as exc:
            print(f"ACKNOWLEDGEMENT REQUIRED: {exc}")
            raise SystemExit(1) from exc
        except (OfficialISCO08CatalogueError, UnknownOfficialProfileError) as exc:
            print(f"CATALOGUE/PROFILE VALIDATION FAILURE: {exc}")
            raise SystemExit(1) from exc
        except BuildPreflightError as exc:
            print(f"PREFLIGHT FAILURE (no collection created): {exc}")
            raise SystemExit(1) from exc
        except BuildExecutionError as exc:
            print(f"BUILD EXECUTION FAILURE (partial build recorded in manifest, no auto-remediation): {exc}")
            raise SystemExit(1) from exc

        print(f"BUILD SUCCEEDED. Manifest written to {args.output_manifest}")
        for t in manifest["verified_targets"]:
            print(f"  {t['name']}: {t['observed_count']}/{t['expected_count']} verified")
        return

    # Dry-run path (default whenever --execute is absent, or when
    # --dry-run is passed even alongside --execute -- --dry-run always wins).
    try:
        plan = build_plan(args.catalogue, args.metadata, args.profile)
    except (OfficialISCO08CatalogueError, UnknownOfficialProfileError) as exc:
        print(f"CATALOGUE/PROFILE VALIDATION FAILURE: {exc}")
        raise SystemExit(1) from exc

    print(json.dumps(plan_to_dict(plan), indent=2, ensure_ascii=False))
    print("\n--dry-run: no Qdrant collection was created, connected to, or modified.")
    if args.output_manifest:
        _write_manifest({"mode": "dry_run", "plan": plan_to_dict(plan)}, args.output_manifest)
        print(f"Dry-run plan also written to {args.output_manifest}")


if __name__ == "__main__":
    main()
