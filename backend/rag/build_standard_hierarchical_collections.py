"""
backend/rag/build_standard_hierarchical_collections.py

Conference I Reviewer #2 response, Task 05. Opt-in, operator-run CLI to
build (create/upsert) the ISIC Rev.4 or ISCED-F 2013 hierarchical Qdrant
collections from the nodes derived by ``backend.rag.hierarchy_nodes``.

**This script was NOT executed against a live Qdrant instance as part of
Task 05** -- no collection was created or populated by that work. Running
it for real (without ``--dry-run``) is an explicit, separate operator
action; see Documentation/Conference_I_Reviewer_2/
ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md for when/why an
operator would do so.

Usage
-----
    # Safe, offline, no Qdrant/embedding-model/network dependency:
    python -m backend.rag.build_standard_hierarchical_collections --standard isic --dry-run
    python -m backend.rag.build_standard_hierarchical_collections --standard iscedf --dry-run

    # Operator-only, live Qdrant write (NOT run by this task):
    python -m backend.rag.build_standard_hierarchical_collections --standard isic --execute
    python -m backend.rag.build_standard_hierarchical_collections --standard isic --execute --recreate

``--dry-run`` runs node derivation (``backend.rag.hierarchy_nodes``) and
validation ONLY -- it imports neither ``qdrant_client.QdrantClient`` nor
``sentence_transformers.SentenceTransformer`` at call time (both are
imported lazily, inside the ``--execute`` code path only), makes no network
request, and writes no data anywhere. It prints per-level node counts, a
deterministic content hash per level (sha256 of sorted
``code|parent_code|label_en`` lines -- lets an operator detect whether the
embedded source tables changed since a prior dry run), and the exact
collection-name/stage-weight plan that a real ``--execute`` run would use.

``--execute`` (real run, operator-only) creates each required collection
if absent, or fails closed with a clear message if it already exists
unless ``--recreate`` is also passed (explicit, destructive opt-in).
"""

from __future__ import annotations

import argparse
import hashlib
import sys

from backend.rag.hierarchy_nodes import (
    HierarchyNode,
    derive_isic_nodes,
    derive_iscedf_nodes,
)
from backend.rag.standard_hierarchical_store import (
    ISCEDF_COLLECTIONS,
    ISCEDF_STAGE_WEIGHTS,
    ISIC_COLLECTIONS,
    ISIC_STAGE_WEIGHTS,
    MODEL_NAME,
    VECTOR_DIM,
)

_STANDARDS = {
    "isic": {
        "derive": derive_isic_nodes,
        "collections": ISIC_COLLECTIONS,
        "weights": ISIC_STAGE_WEIGHTS,
        "level_order": ("sections", "divisions", "groups", "classes"),
        "display_name": "ISIC Rev.4",
    },
    "iscedf": {
        "derive": derive_iscedf_nodes,
        "collections": ISCEDF_COLLECTIONS,
        "weights": ISCEDF_STAGE_WEIGHTS,
        "level_order": ("broad_fields", "narrow_fields", "detailed_fields"),
        "display_name": "ISCED-F 2013",
    },
}


def _level_hash(nodes: list[HierarchyNode]) -> str:
    """Deterministic content hash for one level -- sha256 over sorted
    "code|parent_code|label_en" lines. Changes iff the derived node set
    (codes, parents, or English titles) changes; does not depend on Python
    dict/list ordering."""
    lines = sorted(f"{n.code}|{n.parent_code}|{n.label_en}" for n in nodes)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def dry_run(standard_key: str) -> dict:
    """Node derivation + validation ONLY. No Qdrant, no embedding model, no
    network call, no data written anywhere. Returns the summary dict that
    was also printed, for test assertions."""
    spec = _STANDARDS[standard_key]
    nodes_by_level = spec["derive"]()  # raises HierarchyValidationError on any inconsistency

    summary = {
        "standard": spec["display_name"],
        "mode": "dry_run",
        "embedding_model": MODEL_NAME,
        "vector_dim": VECTOR_DIM,
        "levels": [],
    }
    for level, weight in zip(spec["level_order"], spec["weights"]):
        node_list = nodes_by_level[level]
        summary["levels"].append({
            "level": level,
            "collection": spec["collections"][level],
            "stage_weight": weight,
            "node_count": len(node_list),
            "content_sha256": _level_hash(node_list),
        })

    print(f"=== {spec['display_name']} hierarchical collection plan (DRY RUN) ===")
    print(f"embedding_model: {summary['embedding_model']} ({summary['vector_dim']}-dim)")
    print("No Qdrant connection, embedding model load, or network request was made.")
    print()
    for lvl in summary["levels"]:
        print(
            f"  {lvl['level']:16s} -> collection={lvl['collection']:28s} "
            f"weight={lvl['stage_weight']:.2f} nodes={lvl['node_count']:4d} "
            f"sha256={lvl['content_sha256'][:16]}..."
        )
    total = sum(lvl["node_count"] for lvl in summary["levels"])
    print(f"\nTotal derived nodes: {total}")
    print(
        "These counts reflect this repository's currently embedded records only -- "
        "not an official-catalogue coverage claim. No data was written."
    )
    return summary


def execute_run(standard_key: str, recreate: bool) -> None:
    """Operator-only, live Qdrant write. NOT called by --dry-run, and NOT
    executed anywhere in Task 05's own work. Imports Qdrant/embedding
    dependencies lazily, only on this path."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from sentence_transformers import SentenceTransformer

    spec = _STANDARDS[standard_key]
    nodes_by_level = spec["derive"]()

    print(f"Connecting to Qdrant ...")
    client = QdrantClient(host="localhost", port=6333)
    print(f"Loading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    existing = {c.name for c in client.get_collections().collections}

    for level in spec["level_order"]:
        collection = spec["collections"][level]
        node_list = nodes_by_level[level]

        if collection in existing and not recreate:
            raise RuntimeError(
                f"Collection {collection!r} already exists. Refusing to overwrite without "
                f"--recreate (explicit, destructive opt-in). Aborting before any write."
            )

        if recreate:
            client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
        else:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

        # "passage: " prefix at index time -- matches backend/rag/load_full_isco.py's
        # existing convention exactly (see that module's _PREFIX constant).
        texts = [f"passage: {n.code} {n.index_text}" for n in node_list]
        vectors = model.encode(texts, normalize_embeddings=True).tolist()
        points = [
            PointStruct(
                id=i + 1, vector=vectors[i],
                payload={
                    "code": n.code, "parent_code": n.parent_code,
                    "label_en": n.label_en, "label_ar": n.label_ar,
                },
            )
            for i, n in enumerate(node_list)
        ]
        client.upsert(collection_name=collection, points=points)
        print(f"  {collection}: {len(points)} points upserted")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--standard", required=True, choices=sorted(_STANDARDS))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Node derivation + validation only. No Qdrant/embedding/network call, no data written.")
    mode.add_argument("--execute", action="store_true", help="Operator-only: creates/upserts the live Qdrant collections. Requires a running Qdrant instance and downloads the embedding model.")
    parser.add_argument("--recreate", action="store_true", help="With --execute only: explicit, destructive opt-in to replace an already-existing collection.")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.standard)
        return

    if args.recreate and not args.execute:
        parser.error("--recreate requires --execute")

    try:
        execute_run(args.standard, recreate=args.recreate)
    except RuntimeError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
