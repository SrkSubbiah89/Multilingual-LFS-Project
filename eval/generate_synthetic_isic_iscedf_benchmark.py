"""
eval/generate_synthetic_isic_iscedf_benchmark.py

Real, disclosed, LLM-grounded synthetic benchmark generator for ISIC Rev.4
/ ISCED-F 2013, covering all 5 project languages (6 language codes: en,
ar, ar-gulf, ur, hi, tl -- see backend/agents/language_processor.py's
SUPPORTED_LANGUAGES) -- built because no real labelled evaluation dataset
exists for either standard (WISCO is occupation-only; the IPUMS
International correspondence requesting an equivalent real dataset
concluded 2026-08-27 with a definitive negative answer -- IPUMS does not
have access to, and cannot distribute, original verbatim occupation/
industry/education text for any sample, see
Documentation/Phase_2/Week_1/ipums_correspondence_log.md -- and a real
pilot (Module E) has not started).

**The method, stated plainly**: this is taxonomy-grounded LLM paraphrase
data augmentation -- a standard, widely-used technique for bootstrapping
labelled evaluation/training data when real annotated examples are scarce
or unavailable (the general family sometimes called "weak supervision" /
"synthetic data augmentation from label descriptions": an LLM is given a
category's own official definition and asked to produce natural-language
text a real person might actually say, and the generated text's label is
known BY CONSTRUCTION -- it's exactly the category the definition came
from -- not inferred, not guessed, not scored by a second model). This is
the same category of technique used across NLP when building intent- or
taxonomy-classification benchmarks from a schema document rather than
from collected user data. This is also the closest real analogue to how
national statistics offices pilot-test a new survey instrument's coding
scheme before real fieldwork -- using domain-expert-simulated example
responses grounded in the coding manual -- except here the "domain
expert" is an LLM prompted with the manual's own real text, disclosed as
such throughout, never presented as expert-authored.

**What makes this defensible rather than circular**: every generated
example is grounded in a REAL official definition (the same
backend/rag/official_source_enrichment.py data used to build the
enriched_e5large Qdrant collections), explicitly instructed to paraphrase
into casual survey-respondent language and AVOID reusing the official
title/keywords verbatim -- so a keyword-matching classifier cannot
trivially "win" just by echoing the source text back. It is still
fundamentally an LLM's simulation of what a respondent might say, not a
real respondent's actual words -- see "What this is NOT" below.

**What this is NOT**:
- NOT real respondent data. NOT a substitute for real correspondence-
  sourced data (the IPUMS avenue is now closed, see above) or the
  Module E pilot. Every row is machine-generated.
- NOT randomly guessed text -- grounded in the real official
  definition/examples for the class it's labelled with.
- NOT evaluated by a second LLM (no LLM-as-judge circularity) -- the gold
  label is fixed at generation time by which definition was used as the
  prompt, never re-derived.

**Generation model, and why it changed from an initial local-only
attempt**: a first pass used local Ollama (qwen2.5:3b) exclusively.
Directly inspecting the output caught real, disqualifying quality
problems specific to this generation task -- Arabic output mixed in
literal Chinese characters mid-sentence (e.g. "أنا程序员" -- "I程序员"),
Urdu output was grammatically broken/nonsensical, and a much larger
multilingual-specialist model (aya:latest) was too slow on this hardware
(>120s per call, timed out). Switched to Groq (`groq/openai/gpt-oss-120b`
-- the same model already used successfully for ISCO-08 reranking, see
CLAUDE.md) after directly comparing output: clean, natural text in every
tested language, ~1-2s/call vs 15-30s locally. English/Hindi generation
via local Ollama had been acceptable quality-wise but the cloud model was
used for ALL languages for consistency (one grounded quality bar, not a
per-language patchwork) and speed (this benchmark spans hundreds of
generations across 6 language codes; the wall-clock difference matters).
get_llm_strict() still pins to exactly this one model -- no silent
fallback substitution mid-run.

Scope: only codes with a REAL official-document match are used (see
backend.rag.official_source_enrichment.NON_STANDARD_ISIC_CODES /
NON_STANDARD_ISCEDF_CODES) -- 121 ISIC classes, 61 ISCED-F detailed
fields.

Usage
-----
    python -m eval.generate_synthetic_isic_iscedf_benchmark \\
        --examples-per-code 2 --concurrency 5 \\
        --out eval/results/synthetic_isic_iscedf_benchmark/
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import hashlib
import logging
import re
import sys
from pathlib import Path

from backend.llm.llm_client import get_llm_strict

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "groq/openai/gpt-oss-120b"
_PROMPT_VERSION = "v2_2026-08-27_multilingual"

# Matches backend/agents/language_processor.py's SUPPORTED_LANGUAGES minus
# "other" -- the project's real 5 languages / 6 language codes (Arabic has
# two dialect variants).
_LANGUAGES: dict[str, str] = {
    "en": "English",
    "ar": "Modern Standard Arabic",
    "ar-gulf": "Gulf Arabic dialect (colloquial Khaleeji Arabic, as spoken in the UAE/Gulf states -- not Modern Standard Arabic)",
    "ur": "Urdu",
    "hi": "Hindi",
    "tl": "Tagalog",
}

# Kept deliberately short -- token usage directly determines throughput
# under Groq's real, confirmed 8000 TPM ceiling (see module docstring).
_ISIC_PROMPT_TEMPLATE = (
    'Simulate a survey respondent answering "What does your job/business do?" '
    "IN {language_name}. Real job: {title}. {definition}\n"
    "Write ONE casual first-person sentence (8-20 words) IN {language_name} ONLY, "
    "plain everyday wording (not official terms), no other language mixed in. "
    "Reply with ONLY that sentence."
)

_ISCEDF_PROMPT_TEMPLATE = (
    'Simulate a survey respondent answering "What did you study?" IN '
    "{language_name}. Real field: {title}. {definition}\n"
    "Write ONE casual first-person sentence (6-16 words) IN {language_name} ONLY, "
    "plain everyday wording (not official terms), no other language mixed in. "
    "Reply with ONLY that sentence."
)


def _load_matched_isic_entries() -> list[dict]:
    from backend.agents.isic_classifier import _ISIC_DATA
    from backend.rag.official_source_enrichment import (
        NON_STANDARD_ISIC_CODES,
        load_isic_definitions,
    )
    definitions = load_isic_definitions()
    seen = set()
    entries = []
    for row in _ISIC_DATA:
        code = row["class_code"]
        if code in seen or code in NON_STANDARD_ISIC_CODES:
            continue
        entry = definitions.get(code)
        if entry is None or not (entry.get("definition") or entry.get("examples")):
            continue
        seen.add(code)
        entries.append({"code": code, "title": entry["title"],
                         "definition": entry.get("definition", ""),
                         "examples": entry.get("examples", [])})
    return entries


def _load_matched_iscedf_entries() -> list[dict]:
    from backend.agents.isced_classifier import _ISCED_FIELDS
    from backend.rag.official_source_enrichment import (
        NON_STANDARD_ISCEDF_CODES,
        load_iscedf_definitions,
    )
    definitions = load_iscedf_definitions()
    seen = set()
    entries = []
    for row in _ISCED_FIELDS:
        code = row["detailed_code"]
        if code in seen or code in NON_STANDARD_ISCEDF_CODES:
            continue
        entry = definitions.get(code)
        if entry is None or not (entry.get("definition") or entry.get("examples")):
            continue
        seen.add(code)
        entries.append({"code": code, "title": entry["title"],
                         "definition": entry.get("definition", ""),
                         "examples": entry.get("examples", [])})
    return entries


_MAX_DEFINITION_CHARS = 220  # bounds token usage per call regardless of source length


def _build_prompt(template: str, entry: dict, language_name: str) -> str:
    definition = entry["definition"] or "; ".join(entry["examples"][:3])
    if len(definition) > _MAX_DEFINITION_CHARS:
        definition = definition[:_MAX_DEFINITION_CHARS].rsplit(" ", 1)[0] + "..."
    return template.format(language_name=language_name, title=entry["title"], definition=definition)


_MAX_RETRIES = 4
_BASE_BACKOFF_SECONDS = 3.0

# Real, live-caught defect (2026-08-27, found by a first run of
# eval/validate_synthetic_benchmark_quality.py against the actual 449-row
# dataset): SYN-ISIC-9609-hi-1 and SYN-ISIC-9609-tl-1 (gold code 9609,
# "Other personal service activities n.e.c.", whose official examples
# include "escort services, dating services, services of marriage
# bureaux") both contain the literal text "I'm sorry, but I can't help
# with that." -- a plain LLM safety-filter refusal that the prior
# `len(text) >= 3` acceptance check happily accepted as valid respondent
# text. The same code succeeded normally for the other 4 languages, so
# this is a probabilistic per-call refusal, not a deterministic block --
# retrying the same prompt is a real, appropriate fix, not a workaround.
# ['’]? (not a plain '?) is deliberate -- see eval/validate_synthetic_
# benchmark_quality.py's identical pattern for why: the real refusal rows
# use a curly/typographic apostrophe (U+2019), not a straight ASCII one,
# and a straight-quote-only version of this pattern was verified NOT to
# match them before this was corrected.
_REFUSAL_PATTERNS = re.compile(
    "i['’]?m sorry|i am sorry|i cannot help|i can['’]?t help|"
    "i cannot assist|i can['’]?t assist|as an ai|i am unable to|"
    "i['’]?m unable to|i cannot fulfill|i can['’]?t fulfill|"
    "i['’]?m not able to|i am not able to",
    re.IGNORECASE,
)


def _looks_like_refusal(text: str) -> bool:
    return bool(_REFUSAL_PATTERNS.search(text))


async def _generate_one(llm, sem: asyncio.Semaphore, template: str, entry: dict,
                         lang_code: str, lang_name: str, standard: str, idx: int,
                         pacing_delay: float) -> dict | None:
    """Returns a completed row dict, or None on any failure (never raises
    -- a single failed generation must not abort the whole run). Retries
    with exponential backoff specifically on HTTP 429 (rate limit) --
    confirmed necessary by direct testing: Groq's free-tier key used here
    has a real, hard 8000 TOKENS-PER-MINUTE ceiling (confirmed directly
    from the API's own error message, not assumed -- the same constraint
    CLAUDE.md already documented hitting during corrective-retry work).
    Also retries (same backoff budget) when the model returns a refusal/
    apology instead of respondent text -- see _REFUSAL_PATTERNS above --
    rather than accepting it as a valid row. pacing_delay adds a
    proactive, deliberate pause after every call (success or failure) so
    the run stays under that budget by design rather than only reacting
    to 429s after the fact -- reactive-only backoff was tried first and
    still spent most of its wall-clock time retrying. Any other exception
    fails immediately, no retry (matches this project's existing
    corrective-retry precedent of not retrying non-transient failures)."""
    prompt = _build_prompt(template, entry, lang_name)
    async with sem:
        text = None
        for attempt in range(_MAX_RETRIES):
            try:
                raw = await llm.acall(prompt)
            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "Too Many Requests" in str(exc)
                if is_rate_limit and attempt < _MAX_RETRIES - 1:
                    backoff = _BASE_BACKOFF_SECONDS * (2 ** attempt)
                    await asyncio.sleep(backoff)
                    continue
                log.warning("Generation failed for %r/%s: %s", entry["code"], lang_code, exc)
                await asyncio.sleep(pacing_delay)
                return None
            candidate = str(raw).strip().strip('"').strip("'").strip()
            if _looks_like_refusal(candidate):
                if attempt < _MAX_RETRIES - 1:
                    log.warning(
                        "Refusal-pattern output for %r/%s (attempt %d), retrying: %r",
                        entry["code"], lang_code, attempt, candidate,
                    )
                    await asyncio.sleep(_BASE_BACKOFF_SECONDS)
                    continue
                log.warning(
                    "Refusal-pattern output for %r/%s persisted after %d attempts, skipping row.",
                    entry["code"], lang_code, _MAX_RETRIES,
                )
                await asyncio.sleep(pacing_delay)
                return None
            text = candidate
            break
        await asyncio.sleep(pacing_delay)
    if not text or len(text) < 3:
        return None
    return {
        "case_id": f"SYN-{standard.upper()}-{entry['code']}-{lang_code}-{idx+1}",
        "input_text": text,
        "input_language": lang_code,
        "standard": standard,
        "gold_code": entry["code"],
        "gold_title": entry["title"],
        "generation_model": _DEFAULT_MODEL,
        "prompt_version": _PROMPT_VERSION,
        "source_definition_hash": hashlib.sha256(
            (entry["definition"] + "|".join(entry["examples"])).encode("utf-8")
        ).hexdigest()[:16],
    }


def _load_existing_case_ids(path: str | None) -> set[str]:
    """Case IDs already present in a prior run's output CSV -- case_id is
    deterministic (f"SYN-{standard}-{code}-{lang}-{idx+1}"), so this is an
    exact, reliable way to skip only what already succeeded rather than
    re-spending Groq's confirmed 8000 TPM budget re-generating rows a
    first pass already got. Returns an empty set if no path is given."""
    if not path:
        return set()
    import csv as _csv
    with open(path, encoding="utf-8") as f:
        return {row["case_id"] for row in _csv.DictReader(f)}


async def _generate_all(standards: list[str], languages: list[str], examples_per_code: int,
                         model: str, concurrency: int, limit: int | None, pacing_delay: float,
                         skip_existing: str | None = None) -> list[dict]:
    llm = get_llm_strict(model, temperature=0.8)
    sem = asyncio.Semaphore(concurrency)
    existing = _load_existing_case_ids(skip_existing)
    if existing:
        print(f"Resuming: {len(existing)} case IDs already present in {skip_existing!r}, will be skipped.")

    tasks = []
    skipped = 0
    for standard in standards:
        entries = _load_matched_isic_entries() if standard == "isic" else _load_matched_iscedf_entries()
        template = _ISIC_PROMPT_TEMPLATE if standard == "isic" else _ISCEDF_PROMPT_TEMPLATE
        if limit is not None:
            entries = entries[:limit]
        for entry in entries:
            for lang_code in languages:
                lang_name = _LANGUAGES[lang_code]
                for idx in range(examples_per_code):
                    case_id = f"SYN-{standard.upper()}-{entry['code']}-{lang_code}-{idx+1}"
                    if case_id in existing:
                        skipped += 1
                        continue
                    tasks.append(_generate_one(llm, sem, template, entry, lang_code, lang_name, standard, idx, pacing_delay))
    if skipped:
        print(f"Skipped {skipped} already-generated case(s); {len(tasks)} remain to generate.")

    total = len(tasks)
    print(f"Dispatching {total} generation calls (concurrency={concurrency}) ...")
    results = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        row = await coro
        done += 1
        if row is None:
            if done % 25 == 0 or done == total:
                print(f"  [{done}/{total}] ...")
            continue
        if done % 25 == 0 or done == total:
            print(f"  [{done}/{total}] {row['gold_code']}/{row['input_language']}: {row['input_text']!r}")
        results.append(row)
    return results


def main() -> None:
    # Windows console (cp1252) cannot render Arabic/Urdu/Hindi/Tagalog text
    # directly -- confirmed by a real crash (UnicodeEncodeError) on the
    # very first multilingual smoke run. Reconfigure stdout to UTF-8 with a
    # lossy fallback rather than crash the whole generation run over a
    # progress-print.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--standard", choices=["isic", "iscedf", "both"], default="both")
    parser.add_argument("--languages", default="en,ar,ar-gulf,ur,hi,tl",
                         help="Comma-separated language codes (default: all 6).")
    parser.add_argument("--examples-per-code", type=int, default=2)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=1,
                         help="1 (default) is deliberate -- Groq's confirmed 8000 TPM ceiling means "
                              "higher concurrency just produces more 429s and retries, not more throughput.")
    parser.add_argument("--pacing-delay", type=float, default=1.5,
                         help="Seconds to wait after every call (success or failure) -- proactively "
                              "keeps the run under the TPM budget instead of only reacting to 429s.")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of source codes (for a quick smoke run).")
    parser.add_argument("--skip-existing", default=None,
                         help="Path to a prior run's output CSV -- case IDs already present there are "
                              "skipped (not re-generated) and merged into this run's output, so a "
                              "rate-limit-truncated first pass can be topped up without re-spending "
                              "budget on rows that already succeeded.")
    parser.add_argument("--out", default="eval/results/synthetic_isic_iscedf_benchmark")
    args = parser.parse_args()

    languages = [l.strip() for l in args.languages.split(",") if l.strip()]
    for l in languages:
        if l not in _LANGUAGES:
            parser.error(f"Unknown language code {l!r}; must be one of {sorted(_LANGUAGES)}")

    standards = ["isic", "iscedf"] if args.standard == "both" else [args.standard]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    new_rows = asyncio.run(_generate_all(standards, languages, args.examples_per_code, args.model,
                                          args.concurrency, args.limit, args.pacing_delay, args.skip_existing))

    rows = new_rows
    if args.skip_existing:
        with open(args.skip_existing, encoding="utf-8") as f:
            rows = list(csv.DictReader(f)) + new_rows
        print(f"Merged {len(rows) - len(new_rows)} pre-existing rows with {len(new_rows)} newly generated rows.")

    out_path = out_dir / f"synthetic_isic_iscedf_benchmark_{ts}.csv"
    fieldnames = ["case_id", "input_text", "input_language", "standard", "gold_code",
                  "gold_title", "generation_model", "prompt_version", "source_definition_hash"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} synthetic rows to {out_path}")
    print("SYNTHETIC -- LLM-generated, grounded in real official definitions, "
          "NOT real respondent data. Never cite as WISCO-equivalent or pilot evidence.")


if __name__ == "__main__":
    main()
