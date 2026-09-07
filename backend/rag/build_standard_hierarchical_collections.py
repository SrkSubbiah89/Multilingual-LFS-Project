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
    ISCEDF_COLLECTIONS_BY_PROFILE,
    ISCEDF_FLAT_COLLECTIONS_BY_PROFILE,
    ISCEDF_STAGE_WEIGHTS,
    ISIC_COLLECTIONS_BY_PROFILE,
    ISIC_FLAT_COLLECTIONS_BY_PROFILE,
    ISIC_STAGE_WEIGHTS,
    PROFILE_MODEL_CONFIG,
    _hierarchical_collections_for,
)

_STANDARDS = {
    "isic": {
        "derive": derive_isic_nodes,
        "collections_by_profile": ISIC_COLLECTIONS_BY_PROFILE,
        "flat_collections_by_profile": ISIC_FLAT_COLLECTIONS_BY_PROFILE,
        "weights": ISIC_STAGE_WEIGHTS,
        "level_order": ("sections", "divisions", "groups", "classes"),
        "leaf_level": "classes",
        "display_name": "ISIC Rev.4",
    },
    "iscedf": {
        "derive": derive_iscedf_nodes,
        "collections_by_profile": ISCEDF_COLLECTIONS_BY_PROFILE,
        "flat_collections_by_profile": ISCEDF_FLAT_COLLECTIONS_BY_PROFILE,
        "weights": ISCEDF_STAGE_WEIGHTS,
        "level_order": ("broad_fields", "narrow_fields", "detailed_fields"),
        "leaf_level": "detailed_fields",
        "display_name": "ISCED-F 2013",
    },
}


def _enriched_flat_nodes(standard_key: str) -> list[HierarchyNode]:
    """Leaf-level nodes for the "enriched_e5large" flat profile -- same
    shape as hierarchy_nodes.py's derived nodes, but index_text comes from
    backend.rag.official_source_enrichment.build_enriched_text() (real
    official definitions/examples where a match exists, the existing
    title+keywords text otherwise). Sourced directly from _ISIC_DATA /
    _ISCED_FIELDS (not hierarchy_nodes.py, which only derives the plain
    keyword-based text) -- deliberately a separate, additive code path so
    the existing e5_small/e5_large flat profiles are untouched.

    **Excludes NON_STANDARD_ISIC_CODES / NON_STANDARD_ISCEDF_CODES
    entirely** -- a real regression found by live-testing this profile
    before disclosing it as done, not assumed: once every OTHER code's
    text became much richer, these already-known-non-standard codes'
    now-comparatively-thin fallback text started acting as a genuine
    magnet, exactly the mechanism ISCO-08's own pre-enrichment catalogue
    had (see CLAUDE.md's "magnet effect" finding) -- e.g. "I build mobile
    apps at a software company" and "construction labourer on a
    residential building site" both wrongly matched non-standard code
    8899 instead of their real codes (6201, 4100) before this exclusion.
    Since these codes were already confirmed to not correspond to any
    real official ISIC/ISCED-F code, dropping them from this collection
    is not a coverage loss in any meaningful sense -- a query that would
    have hit one now correctly falls through to its real neighbouring
    code instead (e.g. 7311 excluded -> "advertising agency" queries
    correctly land on 7310, the actual official code for the same
    concept)."""
    from backend.rag.official_source_enrichment import (
        NON_STANDARD_ISCEDF_CODES,
        NON_STANDARD_ISIC_CODES,
        build_enriched_text,
        load_isic_definitions,
        load_iscedf_definitions,
    )

    nodes: dict[str, HierarchyNode] = {}
    if standard_key == "isic":
        from backend.agents.isic_classifier import _ISIC_DATA
        definitions = load_isic_definitions()
        for row in _ISIC_DATA:
            code = row["class_code"]
            if code in nodes or code in NON_STANDARD_ISIC_CODES:
                continue
            text = build_enriched_text(code, row["class_title"], row.get("keywords", ""), definitions)
            nodes[code] = HierarchyNode(
                code=code, parent_code=row["group_code"], label_en=row["class_title"], label_ar="",
                index_text=text, level="class",
            )
    else:
        from backend.agents.isced_classifier import _ISCED_FIELDS
        definitions = load_iscedf_definitions()
        for row in _ISCED_FIELDS:
            code = row["detailed_code"]
            if code in nodes or code in NON_STANDARD_ISCEDF_CODES:
                continue
            text = build_enriched_text(code, row["detailed_title"], row.get("keywords", ""), definitions)
            nodes[code] = HierarchyNode(
                code=code, parent_code=row["narrow_code"], label_en=row["detailed_title"], label_ar="",
                index_text=text, level="detailed_field",
            )
    return list(nodes.values())


def _flat_leaf_nodes(standard_key: str, profile: str) -> list[HierarchyNode]:
    """Dispatches to the enriched source for "enriched_e5large", the
    ordinary hierarchy_nodes.py derivation otherwise (e5_small/e5_large --
    unchanged behaviour)."""
    if profile == "enriched_e5large":
        return _enriched_flat_nodes(standard_key)
    spec = _STANDARDS[standard_key]
    return spec["derive"]()[spec["leaf_level"]]


def _level_hash(nodes: list[HierarchyNode]) -> str:
    """Deterministic content hash for one level -- sha256 over sorted
    "code|parent_code|label_en" lines. Changes iff the derived node set
    (codes, parents, or English titles) changes; does not depend on Python
    dict/list ordering."""
    lines = sorted(f"{n.code}|{n.parent_code}|{n.label_en}" for n in nodes)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def dry_run(standard_key: str, profile: str = "e5_small") -> dict:
    """Node derivation + validation ONLY. No Qdrant, no embedding model, no
    network call, no data written anywhere. Returns the summary dict that
    was also printed, for test assertions."""
    spec = _STANDARDS[standard_key]
    model_name, vector_dim = PROFILE_MODEL_CONFIG[profile]
    collections = _hierarchical_collections_for(spec["collections_by_profile"], profile, spec["display_name"])
    nodes_by_level = spec["derive"]()  # raises HierarchyValidationError on any inconsistency

    summary = {
        "standard": spec["display_name"],
        "mode": "dry_run",
        "profile": profile,
        "embedding_model": model_name,
        "vector_dim": vector_dim,
        "levels": [],
    }
    for level, weight in zip(spec["level_order"], spec["weights"]):
        node_list = nodes_by_level[level]
        summary["levels"].append({
            "level": level,
            "collection": collections[level],
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


def dry_run_flat(standard_key: str, profile: str = "e5_small") -> dict:
    """Same contract as dry_run(), for the single flat (leaf-level-only)
    collection -- see standard_hierarchical_store.py's "Flat (single-
    collection...)" comment block for why this exists alongside the
    hierarchical build above. Reuses the SAME leaf-level derived nodes
    (e.g. ISIC "classes") -- no new node derivation logic."""
    spec = _STANDARDS[standard_key]
    model_name, vector_dim = PROFILE_MODEL_CONFIG[profile]
    collection = spec["flat_collections_by_profile"][profile]
    leaf_level = spec["leaf_level"]
    node_list = _flat_leaf_nodes(standard_key, profile)

    summary = {
        "standard": spec["display_name"],
        "mode": "dry_run_flat",
        "profile": profile,
        "embedding_model": model_name,
        "vector_dim": vector_dim,
        "collection": collection,
        "node_count": len(node_list),
        "content_sha256": _level_hash(node_list),
    }

    print(f"=== {spec['display_name']} FLAT collection plan (DRY RUN) ===")
    print(f"embedding_model: {summary['embedding_model']} ({summary['vector_dim']}-dim)")
    print("No Qdrant connection, embedding model load, or network request was made.")
    print(
        f"\n  {leaf_level:16s} -> collection={collection:34s} "
        f"nodes={summary['node_count']:4d} sha256={summary['content_sha256'][:16]}..."
    )
    print(
        "\nThis is the SAME leaf-level records already used by the hierarchical "
        "build's final stage -- just written into a separately-named collection "
        "queried directly (no parent-chain filtering). No official-catalogue "
        "coverage claim. No data was written."
    )
    return summary


def execute_run_flat(standard_key: str, recreate: bool, profile: str = "e5_small") -> None:
    """Operator-only, live Qdrant write for the flat collection. Same
    fail-closed-on-existing-without-recreate contract as execute_run()."""
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from sentence_transformers import SentenceTransformer

    from backend.rag import make_qdrant_client

    spec = _STANDARDS[standard_key]
    model_name, vector_dim = PROFILE_MODEL_CONFIG[profile]
    collection = spec["flat_collections_by_profile"][profile]
    node_list = _flat_leaf_nodes(standard_key, profile)

    print(f"Connecting to Qdrant ...")
    client = make_qdrant_client()
    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)

    existing = {c.name for c in client.get_collections().collections}
    if collection in existing and not recreate:
        raise RuntimeError(
            f"Collection {collection!r} already exists. Refusing to overwrite without "
            f"--recreate (explicit, destructive opt-in). Aborting before any write."
        )

    if recreate:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
    else:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

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


def execute_run(standard_key: str, recreate: bool, profile: str = "e5_small") -> None:
    """Operator-only, live Qdrant write. NOT called by --dry-run, and NOT
    executed anywhere in Task 05's own work. Imports Qdrant/embedding
    dependencies lazily, only on this path."""
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from sentence_transformers import SentenceTransformer

    from backend.rag import make_qdrant_client

    spec = _STANDARDS[standard_key]
    model_name, vector_dim = PROFILE_MODEL_CONFIG[profile]
    collections = _hierarchical_collections_for(spec["collections_by_profile"], profile, spec["display_name"])
    nodes_by_level = spec["derive"]()

    print(f"Connecting to Qdrant ...")
    client = make_qdrant_client()
    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)

    existing = {c.name for c in client.get_collections().collections}

    for level in spec["level_order"]:
        collection = collections[level]
        node_list = nodes_by_level[level]

        if collection in existing and not recreate:
            raise RuntimeError(
                f"Collection {collection!r} already exists. Refusing to overwrite without "
                f"--recreate (explicit, destructive opt-in). Aborting before any write."
            )

        if recreate:
            client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )
        else:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
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
    parser.add_argument(
        "--profile", default="e5_small", choices=sorted(PROFILE_MODEL_CONFIG),
        help="Embedding profile. 'e5_small' (default) reproduces today's collections "
             "exactly. 'e5_large' builds additive, separately-named collections "
             "(e.g. isic_rev4_sections_e5large) -- mirrors the e5-large profile already "
             "built for ISCO-08; never overwrites or is read by the e5_small collections.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Node derivation + validation only. No Qdrant/embedding/network call, no data written.")
    mode.add_argument("--execute", action="store_true", help="Operator-only: creates/upserts the live Qdrant collections. Requires a running Qdrant instance and downloads the embedding model.")
    parser.add_argument("--recreate", action="store_true", help="With --execute only: explicit, destructive opt-in to replace an already-existing collection.")
    parser.add_argument(
        "--flat", action="store_true",
        help="Build the single flat (leaf-level-only) collection instead of the "
             "4-stage/3-stage hierarchical set -- the architecturally-identical "
             "counterpart to ISCO-08's own best-tested flat retrieval. Reuses the "
             "same leaf-level derived nodes; writes to a separately-named collection.",
    )
    args = parser.parse_args()

    if args.dry_run:
        if args.flat:
            dry_run_flat(args.standard, profile=args.profile)
        else:
            dry_run(args.standard, profile=args.profile)
        return

    if args.recreate and not args.execute:
        parser.error("--recreate requires --execute")

    try:
        if args.flat:
            execute_run_flat(args.standard, recreate=args.recreate, profile=args.profile)
        else:
            execute_run(args.standard, recreate=args.recreate, profile=args.profile)
    except RuntimeError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
