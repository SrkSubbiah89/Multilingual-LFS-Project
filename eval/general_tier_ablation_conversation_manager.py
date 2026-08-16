"""
Module J (Conference I Reviewer #2 response, Prompt 7): GENERAL-tier model
ablation for ConversationManager's REAL LLM-touching task.

Tracing backend/agents/conversation_manager.py directly: the FSM's primary
field extraction (_extract_fields) is fully regex/rule-based -- it never
calls the LLM. The one genuine, checkable, always-active (not bypassed by
LFS_FAST_MODE, unlike the CrewAI response-generation path) LLM code path is
_llm_extract_correction (VALIDATING state, "I meant X not Y" free-text
correction parsing), called via a direct Ollama REST call (bypasses CrewAI
overhead), hard 45-second timeout, format=json.

This script calls _llm_extract_correction directly (isolated, no HTTP
server, no FSM state machine driving it) against a small gold-labeled set
of correction messages, checking whether the right field got the right
value. Supports a `--subset N` isolation check (Prompt 7's required
FSM-sensitivity check, run first with a small N before the full set) and
a full run.

Run:
    OLLAMA_MODEL=llama3.2 python -m eval.general_tier_ablation_conversation_manager --model llama3.2 --out r1.json
    OLLAMA_MODEL=qwen2.5:3b python -m eval.general_tier_ablation_conversation_manager --model qwen2.5:3b --out r2.json
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

# Each case: baseline collected_data, a free-text correction message, and the
# expected (field, value) outcome -- matches _llm_extract_correction's own
# real prompt examples/schema exactly (backend/agents/conversation_manager.py
# _CORRECTION_FIELD_SCHEMA).
CASES = [
    {
        "id": "cm1", "language": "en",
        "baseline": {"nationality": "Pakistani"},
        "message": "Actually I am from India, not Pakistan",
        "expected_field": "nationality", "expected_value_contains": "indian",
    },
    {
        "id": "cm2", "language": "en",
        "baseline": {"employment_type": "full_time"},
        "message": "Sorry, I actually work part-time",
        "expected_field": "employment_type", "expected_value": "part_time",
    },
    {
        "id": "cm3", "language": "en",
        "baseline": {"employment_sector": "government"},
        "message": "Correction: I work in the private sector",
        "expected_field": "employment_sector", "expected_value": "private",
    },
    {
        "id": "cm4", "language": "en",
        "baseline": {"marital_status": "single"},
        "message": "I need to correct that, I'm actually married",
        "expected_field": "marital_status", "expected_value": "married",
    },
    {
        "id": "cm5", "language": "en",
        "baseline": {"job_title": "Accountant"},
        "message": "My job title should be Software Engineer, not Accountant",
        "expected_field": "job_title", "expected_value_contains": "software engineer",
    },
    {
        "id": "cm6", "language": "ar",
        "baseline": {"nationality": "Pakistani"},
        "message": "أنا هندي مو باكستاني",
        "expected_field": "nationality", "expected_value_contains": "indian",
    },
    {
        "id": "cm7", "language": "en",
        "baseline": {"gender": "female"},
        "message": "Please correct that, I'm male",
        "expected_field": "gender", "expected_value": "male",
    },
    {
        "id": "cm8", "language": "en",
        "baseline": {"health_insurance": "none"},
        "message": "Actually I do have full health insurance coverage",
        "expected_field": "health_insurance", "expected_value": "full",
    },
    {
        "id": "cm9", "language": "en",
        "baseline": {"emirate": "dubai"},
        "message": "Change my emirate to Sharjah",
        "expected_field": "emirate", "expected_value": "sharjah",
    },
    {
        "id": "cm10", "language": "en",
        "baseline": {"vocational_training": "no"},
        "message": "Actually yes, I did complete vocational training",
        "expected_field": "vocational_training", "expected_value": "yes",
    },
]


def _check(case: dict, ctx: ConversationContext) -> bool:
    field = case["expected_field"]
    val = ctx.collected_data.get(field)
    if val is None:
        return False
    val_norm = str(val).strip().casefold()
    if "expected_value" in case:
        return val_norm == case["expected_value"].casefold()
    if "expected_value_contains" in case:
        return case["expected_value_contains"] in val_norm
    return False


def run(cases: list[dict], model_label: str) -> list[dict]:
    cm = ConversationManager()
    results = []
    for case in cases:
        ctx = ConversationContext(session_id=1, language=case["language"])
        ctx.collected_data.update(case["baseline"])
        ctx.corrected_fields = set()
        ctx.correction_rejected_field = None

        t0 = time.perf_counter()
        ok = cm._llm_extract_correction(ctx, case["message"])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        correct = _check(case, ctx)
        results.append({
            "id": case["id"], "language": case["language"],
            "llm_ok": ok, "correct": correct,
            "actual_value": ctx.collected_data.get(case["expected_field"]),
            "expected_field": case["expected_field"],
            "latency_ms": round(latency_ms, 1),
        })
        print(f"  {case['id']:6s} lang={case['language']:4s} ok={ok!s:5s} "
              f"correct={correct!s:5s} latency={latency_ms:.0f}ms "
              f"actual={ctx.collected_data.get(case['expected_field'])!r}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--subset", type=int, default=None,
                        help="Run only the first N cases (isolation/FSM-sensitivity check)")
    args = parser.parse_args()

    cases = CASES[: args.subset] if args.subset else CASES
    actual_env = os.environ.get("OLLAMA_MODEL", "(unset -> defaults to llama3.2)")
    print(f"Running with OLLAMA_MODEL={actual_env!r} (--model label: {args.model!r}), n={len(cases)}")

    results = run(cases, args.model)
    n_correct = sum(1 for r in results if r["correct"])
    out = {
        "model_label": args.model,
        "ollama_model_env": actual_env,
        "n_total": len(results),
        "n_correct": n_correct,
        "accuracy": n_correct / len(results),
        "cases": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\naccuracy: {n_correct}/{len(results)} = {n_correct/len(results):.1%}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
