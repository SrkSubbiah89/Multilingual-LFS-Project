"""
eval/emotional_intelligence_warmed_comparison.py

Module J (Conference I Reviewer #2 response): GENERAL-tier model ablation
for EmotionalIntelligence's real analyze() task, properly warmed to avoid
the cold-start confound found during ConversationManager testing.

Must be invoked once per model, with OLLAMA_MODEL set as a real process
environment variable BEFORE this script (and therefore
backend.llm.llm_client) is imported -- that module reads OLLAMA_MODEL into
a module-level constant exactly once at import time, same pattern as
eval/ner_benchmark_runner.py.

Sequence: one throwaway warm-up call (timed separately, never counted in
the comparison) -> 3 full runs over the whole test set.

Run:
    OLLAMA_MODEL=llama3.2 python -m eval.emotional_intelligence_warmed_comparison --model llama3.2 --out r1.json
    OLLAMA_MODEL=qwen2.5:3b python -m eval.emotional_intelligence_warmed_comparison --model qwen2.5:3b --out r2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    actual_env = os.environ.get("OLLAMA_MODEL", "(unset)")
    print(f"OLLAMA_MODEL={actual_env!r} (--model label: {args.model!r})", flush=True)

    from backend.agents.emotional_intelligence import EmotionalIntelligence
    from eval.general_tier_ablation_emotional_intelligence import CASES

    ei = EmotionalIntelligence()

    # Warm-up: throwaway call, NOT counted in any comparison timing.
    t0 = time.perf_counter()
    ei.analyze("This is a warm-up call, ignore the result.", language="en")
    cold_start_s = time.perf_counter() - t0
    print(f"COLD START (throwaway, not counted): {cold_start_s:.2f}s", flush=True)

    result = {"model_label": args.model, "ollama_model_env": actual_env,
              "cold_start_s": round(cold_start_s, 2), "runs": []}

    for run_num in range(1, 4):
        run_results = []
        for case in CASES:
            t0 = time.perf_counter()
            analysis = ei.analyze(case["text"], language=case["language"])
            lat = time.perf_counter() - t0
            correct = analysis.emotional_state.value == case["gold_state"]
            run_results.append({"id": case["id"], "gold": case["gold_state"],
                                 "predicted": analysis.emotional_state.value,
                                 "correct": correct, "latency_s": round(lat, 2)})
            print(f"  run{run_num} {case['id']:6s} gold={case['gold_state']:10s} "
                  f"pred={analysis.emotional_state.value:10s} correct={correct!s:5s} latency={lat:.2f}s", flush=True)
        n_correct = sum(r["correct"] for r in run_results)
        mean_lat = sum(r["latency_s"] for r in run_results) / len(run_results)
        print(f"RUN {run_num}: {n_correct}/{len(run_results)} correct, mean_latency={mean_lat:.2f}s", flush=True)
        result["runs"].append({"run": run_num, "n_correct": n_correct, "n_total": len(run_results),
                                "mean_latency_s": round(mean_lat, 2), "cases": run_results})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
