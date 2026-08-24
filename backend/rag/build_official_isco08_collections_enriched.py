"""
backend/rag/build_official_isco08_collections_enriched.py

Thesis accuracy-investigation work (2026-08-24). Builds the
ENRICHED_PROFILE ("official_ilo2021_v1_enriched") Qdrant collections:
the same verified official ILO ISCO-08 catalogue codes/hierarchy as
DEFAULT_PROFILE, same intfloat/multilingual-e5-small embedding model,
but with richer embedding_text -- the official ILO Definition +
Included-occupations text (previously discarded by the default 4-column
normalization pipeline; see eval/normalize_ilo_isco08_catalogue.py's
EnrichedRow and backend/rag/official_isco08_catalogue.py's
load_enriched_catalogue()).

Built after finding a real "magnet" effect in production evaluation
results: several unit groups whose embedding_text was just "{code}
{2-4 word title}" were predicted 15-55x more often than their true
frequency in the WISCO heldout set. Validated directly against the real
embedding model before building this collection: enrichment reduced
similarity to 19/20 real wrongly-matched queries pulled from actual
heldout results, and flipped 4/5 real cases from wrong to correct when
both the wrong and correct codes were enriched.

Deliberately a separate, simpler script, same reasoning as
build_official_isco08_collections_e5large.py: build_official_isco08_
collections.py is intentionally hardened to allow exactly one profile.
Shares the catalogue loader and the collection-name/embedding-identity
single source of truth in official_isco08_catalogue.py, so this can
never silently drift from what "official_ilo2021_v1_enriched" means.

Guarded live execution
-----------------------
--execute requires --allow-local-qdrant-mutation. Refuses if any of the
five target collections already exists (no overwrite/replace ever).
Local-only Qdrant target (QDRANT_HOST/QDRANT_PORT env vars, default
localhost:6333) -- no remote URL/token option exists in this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from backend.rag.official_isco08_catalogue import (
    DEFAULT_METADATA_PATH,
    ENRICHED_PROFILE,
    PROFILE_COLLECTION_NAMES,
    embedding_config_for_profile,
    load_enriched_catalogue,
    records_by_level,
    OfficialCatalogueRecord,
    OfficialISCO08CatalogueError,
)

_EMBED_MODEL_NAME, _EMBED_VECTOR_DIM = embedding_config_for_profile(ENRICHED_PROFILE)
_EMBED_PREFIX = "passage: "
_EMBED_BATCH_SIZE = 32
_UPSERT_BATCH_SIZE = 64

_QDRANT_HOST_ENV = "QDRANT_HOST"
_QDRANT_PORT_ENV = "QDRANT_PORT"
_LOCAL_QDRANT_HOST_DEFAULT = "localhost"
_LOCAL_QDRANT_PORT_DEFAULT = 6333

_ROLE_SOURCE_LEVEL = {
    "major": "major", "submajor": "submajor", "minor": "minor",
    "unit_hierarchical": "unit", "unit_flat": "unit",
}
_TARGET_ROLE_ORDER = ("major", "submajor", "minor", "unit_hierarchical", "unit_flat")


class BuildPreflightError(Exception):
    pass


class BuildExecutionError(Exception):
    pass


@dataclass(frozen=True)
class TargetBuild:
    name: str
    collection_role: str
    records: tuple


def _target_builds(by_level: dict[str, list[OfficialCatalogueRecord]]) -> list[TargetBuild]:
    names = PROFILE_COLLECTION_NAMES[ENRICHED_PROFILE]
    builds = []
    for role in _TARGET_ROLE_ORDER:
        source_level = _ROLE_SOURCE_LEVEL[role]
        name = names["flat"] if role == "unit_flat" else names[source_level]
        builds.append(TargetBuild(name=name, collection_role=role, records=tuple(by_level[source_level])))
    return builds


def _stable_point_id(code: str, collection_role: str) -> int:
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


def _resolve_local_qdrant_target() -> tuple[str, int]:
    host = os.getenv(_QDRANT_HOST_ENV, _LOCAL_QDRANT_HOST_DEFAULT)
    port = int(os.getenv(_QDRANT_PORT_ENV, _LOCAL_QDRANT_PORT_DEFAULT))
    return host, port


def _default_qdrant_client_factory(host: str, port: int):
    from qdrant_client import QdrantClient
    return QdrantClient(host=host, port=port)


def _default_embedder_factory():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_EMBED_MODEL_NAME)


def _create_and_populate_collection(client, embedder, target: TargetBuild) -> None:
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client.create_collection(
        collection_name=target.name,
        vectors_config=VectorParams(size=_EMBED_VECTOR_DIM, distance=Distance.COSINE),
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
        raise BuildExecutionError(f"{target.name!r}: expected {expected_count}, observed {observed_count}")
    return {"name": target.name, "collection_role": target.collection_role,
            "expected_count": expected_count, "observed_count": observed_count, "verified": True}


def execute_build(
    catalogue_path: Path,
    metadata_path: Path,
    allow_local_qdrant_mutation: bool,
    output_manifest_path: Path,
    qdrant_client_factory: Optional[Callable] = None,
    embedder_factory: Optional[Callable] = None,
) -> dict:
    if not allow_local_qdrant_mutation:
        raise BuildPreflightError("--allow-local-qdrant-mutation is required for --execute")

    records = load_enriched_catalogue(catalogue_path, metadata_path)
    by_level = records_by_level(records)
    target_builds = _target_builds(by_level)

    host, port = _resolve_local_qdrant_target()
    client = (qdrant_client_factory or _default_qdrant_client_factory)(host, port)

    manifest: dict = {
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "profile": ENRICHED_PROFILE,
        "qdrant_host": host, "qdrant_port": port,
        "embedding_model_identity": _EMBED_MODEL_NAME,
        "embedding_vector_dim": _EMBED_VECTOR_DIM,
        "status": "in_progress",
    }

    existing_names = {c.name for c in client.get_collections().collections}
    conflicting = [tb.name for tb in target_builds if tb.name in existing_names]
    if conflicting:
        manifest["status"] = "preflight_failed_existing_target"
        manifest["failure_reason"] = f"already exist, refusing to overwrite: {conflicting}"
        _write_manifest(manifest, output_manifest_path)
        raise BuildPreflightError(manifest["failure_reason"])

    embedder = (embedder_factory or _default_embedder_factory)()

    created_or_partial: list[str] = []
    verified_targets: list[dict] = []
    try:
        for tb in target_builds:
            created_or_partial.append(tb.name)
            print(f"Building {tb.name} ({len(tb.records)} records)...", flush=True)
            _create_and_populate_collection(client, embedder, tb)
            verification = _verify_collection(client, tb)
            verified_targets.append(verification)
            print(f"  verified: {verification['observed_count']}/{verification['expected_count']}", flush=True)
    except Exception as exc:
        manifest["status"] = "failed_partial_build"
        manifest["failure_reason"] = str(exc)
        manifest["targets_created_or_partial"] = created_or_partial
        manifest["targets_verified_before_failure"] = verified_targets
        _write_manifest(manifest, output_manifest_path)
        raise BuildExecutionError(str(exc)) from exc

    manifest["status"] = "success"
    manifest["targets_created_or_partial"] = created_or_partial
    manifest["verified_targets"] = verified_targets
    _write_manifest(manifest, output_manifest_path)
    return manifest


def _write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-local-qdrant-mutation", action="store_true")
    args = parser.parse_args()

    if not args.execute:
        print("Dry-run only supported via --execute (this script has one target profile). "
              "Pass --execute --allow-local-qdrant-mutation to build for real.")
        return

    try:
        manifest = execute_build(
            args.catalogue, args.metadata,
            allow_local_qdrant_mutation=args.allow_local_qdrant_mutation,
            output_manifest_path=args.output_manifest,
        )
    except (BuildPreflightError, BuildExecutionError, OfficialISCO08CatalogueError) as exc:
        print(f"BUILD FAILED: {exc}")
        raise SystemExit(1) from exc

    print(f"BUILD SUCCEEDED. Manifest: {args.output_manifest}")
    for t in manifest["verified_targets"]:
        print(f"  {t['name']}: {t['observed_count']}/{t['expected_count']}")


if __name__ == "__main__":
    main()
