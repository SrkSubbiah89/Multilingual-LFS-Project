"""
eval/legacy824/build_legacy_collection.py

Task 39: builds (or confirms already-built) the historical
`isco_occupations` collection in the isolated Qdrant instance only,
using the unmodified historical VectorStore loaded verbatim from the
detached LEGACY_SHA worktree. Never touches the current project's
Qdrant instance -- `validate_isolated_endpoint` refuses port 6333 and
any known current-project endpoint before anything else runs.

This step needs only the historical embedding model (already verified
cached -- see the Task 39 final report's cache-gate evidence); it does
not require ANTHROPIC_API_KEY, since building/populating the vector
index never calls the LLM.

Usage
-----
python eval/legacy824/build_legacy_collection.py \
    --worktree C:/task39_legacy824_worktree \
    --qdrant-host localhost --qdrant-port 17333
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.legacy824.historical_loader import load_historical_isco_classifier  # noqa: E402
from eval.legacy824.isolation import validate_isolated_endpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--qdrant-host", required=True)
    parser.add_argument("--qdrant-port", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    validate_isolated_endpoint(args.qdrant_host, args.qdrant_port)
    os.environ["QDRANT_HOST"] = args.qdrant_host
    os.environ["QDRANT_PORT"] = str(args.qdrant_port)

    loaded = load_historical_isco_classifier(args.worktree)

    store = loaded.vector_store.get_vector_store()
    count = store._client.count(collection_name=loaded.vector_store.COLLECTION_NAME, exact=True).count

    manifest = {
        "isolated_qdrant_endpoint": f"{args.qdrant_host}:{args.qdrant_port}",
        "collection_name": loaded.vector_store.COLLECTION_NAME,
        "embedding_model": loaded.vector_store.MODEL_NAME,
        "vector_dim": loaded.vector_store.VECTOR_DIM,
        "point_count": count,
        "worktree": str(args.worktree),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
