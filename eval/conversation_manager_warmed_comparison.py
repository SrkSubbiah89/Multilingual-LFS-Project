"""
eval/conversation_manager_warmed_comparison.py

Module J (Conference I Reviewer #2 response): GENERAL-tier model ablation
for ConversationManager's real _llm_extract_correction task, properly
warmed to avoid the cold-start confound found during earlier testing
(a single throwaway call is not enough on a memory-constrained host --
see backend/agents/conversation_manager.py's _CORRECTION_TIMEOUT comment
for the real cold=78s vs warm=4.7-6.1s measurement this was based on).

Unlike eval/ner_benchmark_runner.py, this does NOT need a fresh subprocess
per model -- ConversationManager._call_ollama_json reads OLLAMA_MODEL from
os.environ fresh on every call (not cached at import time), so both models
are compared within a single process here.

Sequence per model: one throwaway warm-up call (timed separately, never
counted in the comparison) -> FSM-sensitivity check (first 3 cases) ->
if that passes, the full 3-run comparison over all 10 cases.

Run:
    python -m eval.conversation_manager_warmed_comparison
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.agents.conversation_manager import ConversationManager, ConversationContext  # noqa: E402
from eval.general_tier_ablation_conversation_manager import CASES, _check  # noqa: E402


def run_model(cm: ConversationManager, model: str) -> dict:
    os.environ["OLLAMA_MODEL"] = model
    print(f"\n=========== {model} ===========", flush=True)

    # Warm-up: throwaway call, NOT counted in any comparison timing.
    t0 = time.perf_counter()
    ctx = ConversationContext(session_id=999, language="en")
    cm._llm_extract_correction(ctx, "warm up call, ignore result")
    cold_start_s = time.perf_counter() - t0
    print(f"COLD START (throwaway, not counted): {cold_start_s:.2f}s", flush=True)

    print("--- FSM-sensitivity check (3 cases, warm) ---", flush=True)
    fsm_results = []
    for case in CASES[:3]:
        ctx = ConversationContext(session_id=1, language=case["language"])
        ctx.collected_data.update(case["baseline"])
        ctx.corrected_fields = set()
        ctx.correction_rejected_field = None
        t0 = time.perf_counter()
        ok = cm._llm_extract_correction(ctx, case["message"])
        lat = time.perf_counter() - t0
        correct = _check(case, ctx)
        fsm_results.append({"id": case["id"], "ok": ok, "correct": correct, "latency_s": round(lat, 2)})
        print(f"  {case['id']:6s} ok={ok!s:5s} correct={correct!s:5s} latency={lat:.2f}s "
              f"actual={ctx.collected_data.get(case['expected_field'])!r}", flush=True)

    fsm_pass = all(r["correct"] for r in fsm_results)
    print(f"FSM-sensitivity check: {'PASS' if fsm_pass else 'FAIL'} "
          f"({sum(r['correct'] for r in fsm_results)}/3)", flush=True)

    model_result = {"cold_start_s": round(cold_start_s, 2), "fsm_check": fsm_results,
                     "fsm_pass": fsm_pass, "runs": []}

    if not fsm_pass:
        print("Skipping full comparison -- FSM-sensitivity check failed.", flush=True)
        return model_result

    print("--- Full 3-run comparison (all 10 cases, warm) ---", flush=True)
    for run_num in range(1, 4):
        run_results = []
        for case in CASES:
            ctx = ConversationContext(session_id=1, language=case["language"])
            ctx.collected_data.update(case["baseline"])
            ctx.corrected_fields = set()
            ctx.correction_rejected_field = None
            t0 = time.perf_counter()
            ok = cm._llm_extract_correction(ctx, case["message"])
            lat = time.perf_counter() - t0
            correct = _check(case, ctx)
            run_results.append({"id": case["id"], "ok": ok, "correct": correct, "latency_s": round(lat, 2)})
        n_correct = sum(r["correct"] for r in run_results)
        mean_lat = sum(r["latency_s"] for r in run_results) / len(run_results)
        print(f"  RUN {run_num}: {n_correct}/{len(run_results)} correct, mean_latency={mean_lat:.2f}s", flush=True)
        model_result["runs"].append({"run": run_num, "n_correct": n_correct, "n_total": len(run_results),
                                      "mean_latency_s": round(mean_lat, 2), "cases": run_results})
    return model_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["llama3.2", "qwen2.5:3b"])
    parser.add_argument("--out", default="backend/evaluation/conversation_manager_warmed_comparison.json")
    args = parser.parse_args()

    cm = ConversationManager()
    results = {model: run_model(cm, model) for model in args.models}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
