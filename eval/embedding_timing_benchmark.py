"""
Module I (Conference I Reviewer #2 response), Phase 1 point 2: real
embedding computation cost per turn.

Times sentence-transformers' encode() call directly, using the exact call
pattern backend/rag/hierarchical_store.py's _embed_query uses in production
(E5 "query: " prefix, normalize_embeddings=True, batch_size=1), batch=1,
averaged over >=100 calls.

Run:
    python -m eval.embedding_timing_benchmark
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

MODEL_NAME = "intfloat/multilingual-e5-small"

SAMPLE_TEXTS = [
    "query: software engineer developing mobile applications",
    "query: registered nurse caring for patients in the ICU",
    "query: secondary school teacher teaching mathematics",
    "query: مهندس برمجيات يطور تطبيقات الجوال",
    "query: اکاؤنٹنٹ مالی حسابات اور آڈٹ کرتا ہوں",
    "query: ड्राइवर ट्रक और डिलीवरी वाहन चलाता हूं",
    "query: nars na nag-aalaga ng pasyente sa ospital",
    "query: operations manager managing daily operations",
]


def _pct(lst: list[float], p: int) -> float:
    s = sorted(lst)
    idx = math.ceil(p / 100 * len(s)) - 1
    return s[max(0, idx)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--out", default="eval/results/legacy_thesis_ch6/embedding_timing_benchmark.json")
    args = parser.parse_args()

    t_import0 = time.perf_counter()
    from sentence_transformers import SentenceTransformer
    t_import = time.perf_counter() - t_import0

    t_load0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    t_load = time.perf_counter() - t_load0

    # Warm-up (excluded -- first call pays a one-time lazy-init cost)
    model.encode([SAMPLE_TEXTS[0]], normalize_embeddings=True, show_progress_bar=False, batch_size=1)

    latencies_ms: list[float] = []
    for i in range(args.n):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        t0 = time.perf_counter()
        vec = model.encode([text], normalize_embeddings=True, show_progress_bar=False, batch_size=1)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    result = {
        "model": MODEL_NAME,
        "call_pattern": "matches backend/rag/hierarchical_store.py._embed_query exactly "
                        "('query: ' E5 prefix, normalize_embeddings=True, batch_size=1)",
        "hardware_disclosure": "resource-constrained CPU-only local sandbox, NOT the target "
                                "deployment host -- see report for full disclosure",
        "import_time_s": round(t_import, 2),
        "model_load_time_s": round(t_load, 2),
        "n_calls": args.n,
        "vector_dim": len(vec[0]),
        "mean_ms": round(statistics.mean(latencies_ms), 2),
        "median_ms": round(statistics.median(latencies_ms), 2),
        "p95_ms": round(_pct(latencies_ms, 95), 2),
        "p99_ms": round(_pct(latencies_ms, 99), 2),
        "min_ms": round(min(latencies_ms), 2),
        "max_ms": round(max(latencies_ms), 2),
    }
    print(json.dumps(result, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
