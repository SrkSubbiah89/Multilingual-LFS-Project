"""
eval/generate_synthetic_coordination_benchmark.py

Real, disclosed, LLM-grounded synthetic benchmark for evaluating Item 1
of the 2026-09-12 multi-agent RAG work (cross-standard coordination
between ISCOClassifier/ISICClassifier/ISCEDClassifier).

Why this exists, and why it's the first thing needed before Item 1 can be
evaluated at all: WISCO -- this project's only real, external, occupation
evaluation dataset -- was directly checked (not assumed) and confirmed to
carry ZERO industry/education text or gold labels in any of its 18,747
heldout rows (`gold_isic`/`gold_isced` columns exist in the CSV schema but
are blank for every single row). The existing synthetic ISIC/ISCED-F
benchmark (eval/generate_synthetic_isic_iscedf_benchmark.py) has real
industry/education text but NO occupation/ISCO gold label at all. Neither
dataset can test whether occupation+industry+education coordination
actually improves ISCO accuracy, because no dataset in this project pairs
all three dimensions for the same case. This script closes that specific
gap using the SAME taxonomy-grounded LLM paraphrase methodology already
used and disclosed for the ISIC/ISCED-F benchmark -- not a new, unvetted
technique.

Method: sample N real ISCO-08 unit-group codes (with real official ILO
definitions, from the same enriched catalogue CSV used to build the
enriched_e5large Qdrant collections). For each, ask a local LLM to
generate ONE consistent (job_title, industry_text, education_text) triple
-- casual respondent language, explicitly instructed not to echo the
official definition/title verbatim. The ISCO gold label is fixed BY
CONSTRUCTION (the code whose definition grounded the generation), never
inferred or guessed.

Scope, disclosed up front, not discovered later: English only for this
first pass. Earlier synthetic-generation work in this project (see
CLAUDE.md's ISIC/ISCED-F benchmark entry) found real, disqualifying
multilingual quality problems with local models (Arabic/Chinese character
mixing, broken Urdu) -- English-only avoids repeating that failure mode
while this specific coordination question is being answered for the
first time. Multilingual expansion is a real, separate follow-up, not
attempted here.

What this is NOT: not real respondent data, not WISCO-equivalent, not a
substitute for a real pilot. A small (default n=40), first-look synthetic
set for directional signal on one specific question (does cross-standard
coordination change ISCO accuracy), not a citable headline number.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.llm import get_llm_strict

CATALOGUE_PATH = Path(__file__).resolve().parents[1] / (
    "eval/local_catalogues/ilo_isco08_2021/normalized/isco08_official_normalized_enriched.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "synthetic_coordination_benchmark" / "benchmark.csv"

_REFUSAL_MARKERS = (
    "i'm sorry", "i’m sorry", "i cannot", "i can't", "as an ai",
)


def load_grounded_codes() -> list[dict]:
    with CATALOGUE_PATH.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            r for r in reader
            if r.get("level") == "unit" and r.get("definition", "").strip()
        ]
    if not rows:
        raise RuntimeError(f"No grounded unit-level codes found in {CATALOGUE_PATH}")
    return rows


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def generate_triple(entry: dict, model: str, retries: int = 2) -> dict | None:
    """Generate one (job_title, industry_text, education_text) triple
    grounded in *entry*'s real official definition. Returns None (never
    raises) on repeated failure -- callers must skip the code, never
    fabricate a placeholder row."""
    definition = entry["definition"].strip()
    label = entry["label"].strip()
    code = entry["code"].strip()

    prompt = (
        f"An ISCO-08 occupation is officially defined as follows:\n\n"
        f'Title: "{label}"\n'
        f'Definition: "{definition}"\n\n'
        "Imagine a real survey respondent who holds this occupation. Write, in "
        "casual first-person survey-answer language (NOT the official title or "
        "definition wording):\n"
        "1. job_title: how they'd describe their job in a few words\n"
        "2. industry_text: what industry/business they work in\n"
        "3. education_text: what education/qualification they'd plausibly have\n\n"
        "Return ONLY a JSON object, no markdown fences:\n"
        '{"job_title": "...", "industry_text": "...", "education_text": "..."}'
    )

    for attempt in range(retries + 1):
        try:
            llm = get_llm_strict(model, temperature=0.7)
            raw = llm.call([{"role": "user", "content": prompt}]).strip()
        except Exception as exc:
            print(f"  [{code}] LLM call failed (attempt {attempt+1}): {exc}")
            continue

        if _looks_like_refusal(raw):
            print(f"  [{code}] refusal detected, retrying (attempt {attempt+1})")
            continue

        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", clean, re.DOTALL)
            if not m:
                print(f"  [{code}] no JSON found in response (attempt {attempt+1})")
                continue
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                print(f"  [{code}] JSON parse failed (attempt {attempt+1})")
                continue

        job_title = str(data.get("job_title", "")).strip()
        industry_text = str(data.get("industry_text", "")).strip()
        education_text = str(data.get("education_text", "")).strip()
        if job_title and industry_text and education_text:
            return {
                "gold_isco_4digit": code,
                "gold_isco_label": label,
                "job_title": job_title,
                "industry_text": industry_text,
                "education_text": education_text,
            }
        print(f"  [{code}] incomplete triple (attempt {attempt+1}): {data}")

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40, help="number of codes to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="ollama/qwen2.5:3b")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    codes = load_grounded_codes()
    random.Random(args.seed).shuffle(codes)
    sample = codes[: args.n]

    print(f"Generating {len(sample)} synthetic (occupation, industry, education) triples via {args.model}...")
    rows: list[dict] = []
    for i, entry in enumerate(sample, 1):
        t0 = time.perf_counter()
        row = generate_triple(entry, args.model)
        elapsed = time.perf_counter() - t0
        if row is None:
            print(f"[{i}/{len(sample)}] {entry['code']} SKIPPED (generation failed)")
            continue
        rows.append(row)
        print(f"[{i}/{len(sample)}] {entry['code']} ({elapsed:.1f}s): {row['job_title']!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["gold_isco_4digit", "gold_isco_label", "job_title", "industry_text", "education_text"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)}/{len(sample)} rows to {args.output}")
    if len(rows) < len(sample):
        print(f"({len(sample) - len(rows)} codes skipped after generation failures -- see log above)")


if __name__ == "__main__":
    main()
