"""
eval/ner_benchmark_runner.py

Real (non-mocked) GENERAL-tier NER accuracy benchmark, built to compare
Llama 3.2 vs Qwen2.5 for backend.agents.language_processor.LanguageProcessor's
NER step -- the only part of the GENERAL tier's 3 consumers (LanguageProcessor,
ConversationManager, EmotionalIntelligence) that has a clean, measurable,
per-language accuracy interpretation. No prior benchmark of this kind existed
in the repo; every existing test in backend/tests/test_language_processor*.py
mocks the LLM's JSON output rather than invoking a real model.

Must be invoked once per model, with OLLAMA_MODEL set as a real process
environment variable BEFORE this script (and therefore before
backend.llm.llm_client) is imported -- that module reads OLLAMA_MODEL into a
module-level constant exactly once at import time, so setting it later in a
running process has no effect. Run as a genuinely separate subprocess per
model, matching how the reversible env-var swap would actually be used in
production, e.g.:

    OLLAMA_MODEL=llama3.2 python eval/ner_benchmark_runner.py --model llama3.2 --out results_llama.json
    OLLAMA_MODEL=qwen2.5:3b python eval/ner_benchmark_runner.py --model qwen2.5:3b --out results_qwen.json

Entity matching: a predicted entity counts as a true positive against a gold
entity if their (label, normalized text) pair matches exactly (casefold +
whitespace-collapse only -- no fuzzy matching, so partial-span credit is
never silently given).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.agents.language_processor import LanguageProcessor  # noqa: E402


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _match(gold: list[dict], pred: list[dict]) -> tuple[int, int, int]:
    """Returns (tp, fp, fn) via one-to-one greedy matching on (label, normalized text)."""
    gold_remaining = [(g["label"], _norm(g["text"])) for g in gold]
    tp = 0
    for p in pred:
        key = (p.label, _norm(p.text))
        if key in gold_remaining:
            gold_remaining.remove(key)
            tp += 1
    fp = len(pred) - tp
    fn = len(gold_remaining)
    return tp, fp, fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model label for the report (informational only)")
    parser.add_argument("--data", default=str(Path(__file__).parent / "ner_benchmark_data.json"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    actual_ollama_model_env = os.environ.get("OLLAMA_MODEL", "(unset -> defaults to llama3.2)")
    print(f"Running with OLLAMA_MODEL={actual_ollama_model_env!r} (--model label: {args.model!r})")

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    cases = data["cases"]

    processor = LanguageProcessor()
    if not processor._agent_available:
        print("FATAL: LanguageProcessor's NER agent is not available (LLM construction failed). Aborting.")
        sys.exit(1)

    per_case_results = []
    for case in cases:
        t0 = time.perf_counter()
        result = processor.process(case["text"])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        pred = [{"text": e.text, "label": e.label} for e in result.entities]
        tp, fp, fn = _match(case["gold"], result.entities)

        per_case_results.append({
            "id": case["id"],
            "language": case["language"],
            "latency_ms": round(latency_ms, 1),
            "gold": case["gold"],
            "predicted": pred,
            "tp": tp, "fp": fp, "fn": fn,
        })
        print(f"  {case['id']:10s} lang={case['language']:8s} tp={tp} fp={fp} fn={fn} latency={latency_ms:.0f}ms")

    out = {
        "model_label": args.model,
        "ollama_model_env": actual_ollama_model_env,
        "cases": per_case_results,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(per_case_results)} case results to {args.out}")


if __name__ == "__main__":
    main()
