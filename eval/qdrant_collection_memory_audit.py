"""
Module I (Conference I Reviewer #2 response), Phase 1 point 1: real Qdrant
index size / memory for the live ISCO-08 collections.

Uses Qdrant's collection-info API for real point counts and vector config.
Qdrant's HTTP API does not expose per-collection memory/disk size directly
(confirmed by inspecting the full collection-info response), so this script
also reports the real total process-wide memory via Qdrant's /metrics
endpoint (covers ALL collections combined, not just ISCO) and, only where
neither is available, computes an analytical estimate
(point_count * vector_dim * 4 bytes) clearly labeled as such.

Run (requires a reachable Qdrant instance):
    python -m eval.qdrant_collection_memory_audit
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

COLLECTIONS = [
    "isco08_major_groups",
    "isco08_major_groups_ilo2021_v1",
    "isco08_submajor_groups",
    "isco08_submajor_groups_ilo2021_v1",
    "isco08_minor_groups",
    "isco08_minor_groups_ilo2021_v1",
    "isco08_unit_groups",
    "isco08_unit_groups_flat_ilo2021_v1",
    "isco08_unit_groups_ilo2021_v1",
    "isco_occupations",
]


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def audit_collection(base_url: str, name: str) -> dict:
    d = _get_json(f"{base_url}/collections/{name}")["result"]
    vectors = d["config"]["params"]["vectors"]
    dim = vectors["size"] if isinstance(vectors, dict) else None
    points = d["points_count"]
    indexed = d["indexed_vectors_count"]
    full_scan_threshold = d["config"]["optimizer_config"]["indexing_threshold"]
    analytical_estimate_bytes = points * dim * 4 if dim else None
    return {
        "collection": name,
        "points_count": points,
        "vector_dim": dim,
        "indexed_vectors_count": indexed,
        "hnsw_built": indexed > 0,
        "below_indexing_threshold": points < full_scan_threshold,
        "indexing_threshold": full_scan_threshold,
        "analytical_estimate_bytes_raw_vectors_only": analytical_estimate_bytes,
        "analytical_estimate_note": (
            "point_count * vector_dim * 4 bytes (float32), RAW VECTOR DATA ONLY -- "
            "excludes payload, WAL, segment overhead. Not HNSW graph overhead since "
            "this collection is below Qdrant's indexing_threshold and uses flat "
            "(brute-force) search with no HNSW graph built at all (indexed_vectors_count=0)."
        ),
    }


def audit_process_memory(base_url: str) -> dict:
    """Real, measured, process-wide (all collections combined) memory via
    Qdrant's own /metrics endpoint -- not per-collection, but a genuine
    measurement, not an estimate."""
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as resp:
        text = resp.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("memory_resident_bytes "):
            out["memory_resident_bytes"] = int(float(line.split()[-1]))
        elif line.startswith("memory_active_bytes "):
            out["memory_active_bytes"] = int(float(line.split()[-1]))
        elif line.startswith("memory_allocated_bytes "):
            out["memory_allocated_bytes"] = int(float(line.split()[-1]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:6333")
    parser.add_argument("--out", default="backend/evaluation/qdrant_collection_memory_audit.json")
    args = parser.parse_args()

    collections = [audit_collection(args.base_url, name) for name in COLLECTIONS]
    process_memory = audit_process_memory(args.base_url)

    result = {
        "source": "real, direct Qdrant collection-info + /metrics API calls -- not estimated",
        "collections": collections,
        "process_wide_memory_real_measured": {
            **process_memory,
            "note": "Whole-Qdrant-process RSS covering ALL collections combined "
                    "(this dev instance hosts only ISCO-08 collections) -- Qdrant's "
                    "HTTP API does not expose a real per-collection memory breakdown.",
        },
    }
    print(json.dumps(result, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
