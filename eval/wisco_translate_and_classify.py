"""
eval/wisco_translate_and_classify.py

Translate every non-English WISCO heldout case to English via local
qwen2.5:3b (Ollama, no daily quota, unlike Groq), then re-run the SAME
proven flat + official_ilo2021_v1_enriched_e5large retrieval config
(via run_eval.py, no reranking) that produced the project's headline
40.95% ISCO-08 number -- the only variable changed is the input text
language, not the retrieval pipeline itself.

Resumable: translation writes one JSONL line per case, skips any
case_id already present. Safe to interrupt and re-run.

Run:
    python eval/wisco_translate_and_classify.py --stage translate
    python eval/wisco_translate_and_classify.py --stage build-csv
    (then run_eval.py separately on the resulting CSV)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["OLLAMA_MODEL"] = "qwen2.5:3b"

from backend.llm.llm_client import get_llm_strict  # noqa: E402


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


def stage_translate(args) -> None:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    print(f"Already translated: {len(done)}", flush=True)

    with open(args.data, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["input_language"] != "en"]
    remaining = [r for r in rows if r["case_id"] not in done]
    print(f"Remaining: {len(remaining)} / {len(rows)} non-English cases", flush=True)

    llm = get_llm_strict("ollama/qwen2.5:3b", temperature=0.0)

    n = 0
    t_start = time.perf_counter()
    for row in remaining:
        prompt = (
            "Translate this job/occupation description to English. "
            "Respond with ONLY the translation, nothing else.\n\n"
            f"Text: {row['input_text']}"
        )
        translated = row["input_text"]
        for attempt in range(3):
            try:
                translated = llm.call([{"role": "user", "content": prompt}]).strip()
                break
            except Exception as exc:
                print(f"  retry {attempt} for {row['case_id']}: {str(exc)[:100]}", flush=True)
                time.sleep(3)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "case_id": row["case_id"], "language": row["input_language"],
                "original": row["input_text"], "translated": translated,
            }, ensure_ascii=False) + "\n")
        n += 1
        if n % 100 == 0:
            elapsed = time.perf_counter() - t_start
            rate = n / elapsed
            eta_min = (len(remaining) - n) / rate / 60 if rate else 0
            print(f"  progress: {n}/{len(remaining)} this run, "
                  f"{rate:.2f} cases/s, ETA {eta_min:.0f} min", flush=True)

    print(f"TRANSLATE STAGE COMPLETE: {n} translated this run, "
          f"{len(done) + n} total done.", flush=True)


def stage_build_csv(args) -> None:
    """Merge translations back into a full run_eval.py-ready CSV
    (English cases pass through unchanged; non-English cases use their
    translated text; any not-yet-translated non-English case is skipped)."""
    translations = {}
    with open(args.translations, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            translations[d["case_id"]] = d["translated"]

    with open(args.data, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    skipped = 0
    for r in rows:
        if r["input_language"] == "en":
            out_rows.append(r)
        elif r["case_id"] in translations:
            r2 = dict(r)
            r2["input_text"] = translations[r["case_id"]]
            out_rows.append(r2)
        else:
            skipped += 1

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"Wrote {len(out_rows)} rows ({skipped} non-English cases skipped, not yet translated) to {args.out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["translate", "build-csv"])
    parser.add_argument("--data", default="eval/local_benchmarks/wisco_isco08_v3_dev_validation_split/heldout_run_eval_format.csv")
    parser.add_argument("--out", default="eval/results/wisco_groq_zeroshot/heldout_translations.jsonl")
    parser.add_argument("--translations", default="eval/results/wisco_groq_zeroshot/heldout_translations.jsonl")
    args = parser.parse_args()

    if args.stage == "translate":
        stage_translate(args)
    else:
        stage_build_csv(args)


if __name__ == "__main__":
    main()
