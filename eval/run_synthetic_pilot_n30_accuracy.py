"""
eval/run_synthetic_pilot_n30_accuracy.py

Companion to run_synthetic_pilot_n30_live.py -- computes ISCO-08
classification ACCURACY for the same n=30 synthetic cases, using this
project's real, heldout-confirmed best-tested configuration
(force_flat=True, isco_catalogue_profile=ENRICHED_E5LARGE_PROFILE --
40.95% vs. the live server's current default 21.19% on the full
18,747-case WISCO heldout; see CLAUDE.md).

Deliberately a SEPARATE process from the live-conversation driver, and
deliberately never invoked through the live HTTP server: 2026-10-01,
this exact config was attempted on the live survey path and reverted
after a real, reproduced crash -- loading multilingual-e5-large under
this machine's current low free memory sometimes raises a catchable
RuntimeError, sometimes segfaults the whole process outright (see
CLAUDE.md's full writeup). Running it here, standalone, means a crash
only kills this one offline script -- never the live demo, never the
conversation-level pilot results already written by the live script.
Checks free memory before attempting, and reports a clear, honest
failure rather than a silent partial result if it can't run.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _free_memory_gb() -> float | None:
    """Best-effort free-memory check (Windows-specific, matches this
    project's own established `wmic`/PowerShell checks elsewhere) --
    returns None if it can't be determined, never raises."""
    try:
        import subprocess
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            capture_output=True, text=True, timeout=10,
        )
        kb = int(result.stdout.strip())
        return round(kb / (1024 * 1024), 2)
    except Exception:
        return None


BENCHMARK_CSV = Path(__file__).resolve().parent / "results" / "synthetic_coordination_benchmark" / "benchmark.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "synthetic_pilot_n30"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "accuracy_results.csv")
    ap.add_argument("--min-free-gb", type=float, default=1.0,
                     help="Refuse to attempt the e5-large load below this much free RAM "
                          "(matches this project's own documented low-RAM crash threshold).")
    args = ap.parse_args()

    free_gb = _free_memory_gb()
    print(f"Free memory: {free_gb} GB" if free_gb is not None else "Free memory: could not determine")
    if free_gb is not None and free_gb < args.min_free_gb:
        print(f"\nERROR: only {free_gb}GB free, below the {args.min_free_gb}GB safety floor.", file=sys.stderr)
        print("This is the exact condition that caused a real segfault on 2026-10-01 "
              "(see CLAUDE.md). Close other applications and retry rather than push through.",
              file=sys.stderr)
        sys.exit(1)

    with BENCHMARK_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))[: args.n]

    print(f"Loading ISCOClassifier(force_flat=True, isco_catalogue_profile=ENRICHED_E5LARGE_PROFILE)...")
    t0 = time.perf_counter()
    from backend.agents.isco_classifier import ISCOClassifier
    from backend.rag.official_isco08_catalogue import ENRICHED_E5LARGE_PROFILE
    clf = ISCOClassifier(force_flat=True, isco_catalogue_profile=ENRICHED_E5LARGE_PROFILE, enable_llm=False)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    results = []
    n_correct = 0
    for i, row in enumerate(rows, start=1):
        gold = row["gold_isco_4digit"].strip()
        job_title = row["job_title"]
        t0 = time.perf_counter()
        try:
            r = clf.classify(job_title)
            pred = r.primary_code
            conf = r.confidence
            method = r.method
            err = None
        except Exception as exc:
            pred, conf, method, err = None, None, None, str(exc)
        ms = int((time.perf_counter() - t0) * 1000)
        correct = (pred == gold)
        n_correct += int(correct)
        results.append({
            "case_idx": i, "gold_isco_4digit": gold, "job_title": job_title,
            "predicted_isco_4digit": pred, "correct": correct,
            "confidence": conf, "method": method, "latency_ms": ms, "error": err,
        })
        print(f"[{i}/{len(rows)}] gold={gold} pred={pred} correct={correct} ({ms}ms)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    acc = n_correct / len(results) if results else 0.0
    print(f"\n=== SUMMARY ===")
    print(f"n = {len(results)}")
    print(f"Exact 4-digit ISCO-08 accuracy: {n_correct}/{len(results)} = {acc:.1%}")
    print(f"(Reference: full 18,747-case WISCO heldout with this same config: 40.95%, CLAUDE.md)")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
