"""
eval/wisco_groq_zeroshot_classify.py

Zero-shot ISCO-08 classification via Groq gpt-oss-120b -- NO retrieval, NO
candidate list, just the job text and a request for the 4-digit code. A
genuinely new, previously-untested angle (every prior Groq test in this
project used it as a RERANKER on top of retrieval, never as the primary
classifier).

Resumable by design: appends one JSON line per case to the output file and
skips any case_id already present there, so a re-run after Groq's daily
200,000-token quota is hit picks up exactly where it left off -- same
--skip-existing pattern already used for the synthetic ISIC/ISCED-F
benchmark generation elsewhere in this project.

Run:
    python eval/wisco_groq_zeroshot_classify.py \
        --data eval/local_benchmarks/wisco_isco08_v3_dev_validation_split/heldout_run_eval_format.csv \
        --out eval/results/wisco_groq_zeroshot/results.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["LLM_FALLBACK_EXCLUDE"] = "anthropic,gemini,ollama"

from backend.llm.llm_client import get_llm_strict  # noqa: E402

_CODE_RE = re.compile(r"\b(\d{4})\b")


def load_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["case_id"])
                except Exception:
                    continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--delay", type=float, default=1.6)
    parser.add_argument("--max-calls", type=int, default=100000)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    print(f"Already completed: {len(done)} cases", flush=True)

    with open(args.data, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    remaining = [r for r in rows if r["case_id"] not in done]
    print(f"Remaining: {len(remaining)} / {len(rows)} total", flush=True)

    llm = get_llm_strict("groq/openai/gpt-oss-120b", temperature=0.0)

    n_correct = 0
    n_done_this_run = 0
    for row in remaining:
        if n_done_this_run >= args.max_calls:
            print(f"Reached --max-calls={args.max_calls} for this run. Stopping cleanly.", flush=True)
            break

        prompt = (
            f"Job description ({row['input_language']}): {row['input_text']}\n"
            "Respond with ONLY the 4-digit ISCO-08 occupation code, nothing else."
        )

        for attempt in range(4):
            try:
                resp = llm.call([{"role": "user", "content": prompt}])
                break
            except Exception as exc:
                msg = str(exc)
                if "Limit 200000" in msg or "TPD" in msg or "tokens per day" in msg.lower():
                    print(f"DAILY QUOTA HIT after {n_done_this_run} calls this run "
                          f"({len(done) + n_done_this_run} total complete). Stopping for today. "
                          f"Error: {msg[:200]}", flush=True)
                    print(f"FINAL: {len(done) + n_done_this_run} / {len(rows)} complete, "
                          f"{n_correct} correct this run.", flush=True)
                    sys.exit(0)
                if "rate_limit" in msg.lower() or "RateLimitError" in msg:
                    wait = 5 * (attempt + 1)
                    print(f"  rate limited, waiting {wait}s (attempt {attempt+1})", flush=True)
                    time.sleep(wait)
                    continue
                print(f"  call failed (non-rate-limit): {msg[:150]}", flush=True)
                resp = ""
                break
        else:
            resp = ""

        m = _CODE_RE.search(resp or "")
        pred_code = m.group(1) if m else None
        correct = pred_code == row["gold_isco_4digit"]
        if correct:
            n_correct += 1
        n_done_this_run += 1

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "case_id": row["case_id"], "language": row["input_language"],
                "gold": row["gold_isco_4digit"], "pred": pred_code,
                "raw_response": (resp or "")[:80], "correct": correct,
            }, ensure_ascii=False) + "\n")

        if n_done_this_run % 25 == 0:
            print(f"  progress: {n_done_this_run} this run, "
                  f"{n_correct}/{n_done_this_run} correct so far this run "
                  f"({100*n_correct/n_done_this_run:.1f}%)", flush=True)

        time.sleep(args.delay)

    print(f"RUN COMPLETE: {n_done_this_run} calls this run, {n_correct} correct "
          f"({100*n_correct/n_done_this_run:.1f}% this run)." if n_done_this_run else "Nothing to do.", flush=True)
    print(f"TOTAL PROGRESS: {len(done) + n_done_this_run} / {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
