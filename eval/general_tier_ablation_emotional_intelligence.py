"""
Module J (Conference I Reviewer #2 response, Prompt 7): GENERAL-tier model
ablation for EmotionalIntelligence's real analyze() task.

Unlike ConversationManager's response-generation path, EmotionalIntelligence's
Stage-2 LLM call in analyze() is NOT gated by LFS_FAST_MODE -- it always runs
(backend/agents/emotional_intelligence.py, analyze() -> _analyze_with_llm),
making this a genuinely representative live task to benchmark.

Calls EmotionalIntelligence().analyze(text, language) directly (real CrewAI
call, not mocked) against a small gold-labeled test set, checking whether
the predicted emotional_state matches gold. Must be invoked once per model
via a fresh subprocess with OLLAMA_MODEL set as a real env var, same pattern
as eval/ner_benchmark_runner.py (backend.llm.llm_client reads OLLAMA_MODEL
into a module-level constant once at import time).

Run:
    OLLAMA_MODEL=llama3.2 python -m eval.general_tier_ablation_emotional_intelligence --model llama3.2 --out r1.json
    OLLAMA_MODEL=qwen2.5:3b python -m eval.general_tier_ablation_emotional_intelligence --model qwen2.5:3b --out r2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.agents.emotional_intelligence import EmotionalIntelligence  # noqa: E402

CASES = [
    {"id": "ei1", "language": "en", "text": "I don't understand any of this, can you please help me?",
     "gold_state": "confused"},
    {"id": "ei2", "language": "en", "text": "This is ridiculous, I've been asked the same question three times already!!",
     "gold_state": "frustrated"},
    {"id": "ei3", "language": "en", "text": "I'm really worried about losing my job, this is very stressful for me.",
     "gold_state": "stressed"},
    {"id": "ei4", "language": "en", "text": "Sure, happy to answer more questions, this is interesting!",
     "gold_state": "engaged"},
    {"id": "ei5", "language": "en", "text": "I work as an accountant in the private sector.",
     "gold_state": "neutral"},
    {"id": "ei6", "language": "ar", "text": "أنا قلق جداً بشأن فقدان وظيفتي، هذا مرهق للغاية بالنسبة لي.",
     "gold_state": "stressed"},
    {"id": "ei7", "language": "ar", "text": "هذا سخيف، لقد سُئلت نفس السؤال ثلاث مرات!!",
     "gold_state": "frustrated"},
    {"id": "ei8", "language": "en", "text": "I'm not sure what you mean by that question, could you explain?",
     "gold_state": "confused"},
    {"id": "ei9", "language": "en", "text": "I currently work 40 hours a week at a hospital.",
     "gold_state": "neutral"},
    {"id": "ei10", "language": "en", "text": "Great, let's continue, I have plenty of time to answer everything.",
     "gold_state": "engaged"},
]


def run(cases: list[dict], model_label: str) -> list[dict]:
    ei = EmotionalIntelligence()
    results = []
    for case in cases:
        t0 = time.perf_counter()
        result = ei.analyze(case["text"], language=case["language"])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        correct = result.emotional_state.value == case["gold_state"]
        results.append({
            "id": case["id"], "language": case["language"],
            "gold_state": case["gold_state"], "predicted_state": result.emotional_state.value,
            "correct": correct, "confidence": result.confidence,
            "latency_ms": round(latency_ms, 1),
        })
        print(f"  {case['id']:6s} lang={case['language']:4s} gold={case['gold_state']:10s} "
              f"pred={result.emotional_state.value:10s} correct={correct!s:5s} latency={latency_ms:.0f}ms")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    actual_env = os.environ.get("OLLAMA_MODEL", "(unset -> defaults to llama3.2)")
    print(f"Running with OLLAMA_MODEL={actual_env!r} (--model label: {args.model!r}), n={len(CASES)}")

    results = run(CASES, args.model)
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
