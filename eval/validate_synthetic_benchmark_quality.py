"""
eval/validate_synthetic_benchmark_quality.py

Independent, automated quality check for the rows already produced by
`eval/generate_synthetic_isic_iscedf_benchmark.py`.

Why this exists: the only quality control the generator itself had was a
human (Sivarama) eyeballing a handful of outputs during model selection,
which is how the qwen2.5:3b Chinese-character-mixing and broken-Urdu
defects were originally caught (see that script's own docstring and
CLAUDE.md's "Knowledge base construction" log). That was real, but it was
a spot-check on the model-selection sample, not a systematic check on
every row of the dataset actually used for the published 82.85%/83.06%
accuracy numbers. This script closes that gap: it re-runs the same class
of check (does the text mix in a script it shouldn't) across every row,
automatically and reproducibly, rather than relying on a one-time human
read-through.

Four checks, none requiring a live LLM call (this stays hermetic and
free to re-run):

1. **Script contamination** -- does the row contain characters from a
   script that has NO business being there for that language (the exact
   defect class the qwen2.5:3b attempt produced for Arabic: CJK
   characters mixed mid-sentence)? Checked against every row regardless
   of which model generated it, so this also re-validates the Groq output
   that replaced qwen2.5:3b, not just the discarded attempt.
2. **Primary-script match** -- for scripted languages (ar/ar-gulf/ur ->
   Arabic-range; hi -> Devanagari; en/tl -> Latin), does the row's
   predominant script actually match what the language code claims?
   Catches wrong-language generation, not just contamination.
3. **Verbatim official-title leakage** -- the generation prompt
   explicitly instructs the model to paraphrase and avoid reusing the
   official title/keywords verbatim (see the generator's own docstring,
   "What makes this defensible rather than circular"). A row that leaks
   the English official title verbatim weakens that defensibility claim
   for THAT row specifically -- flagged, not silently dropped.
4. **LLM refusal/apology pattern** -- added after this exact defect was
   found by a first, manual run of this script against the real 449-row
   dataset (2026-08-27): two rows (SYN-ISIC-9609-hi-1, SYN-ISIC-9609-tl-1
   -- gold code 9609 "Other personal service activities n.e.c.", whose
   official examples include "escort services, dating services") contain
   the literal text "I'm sorry, but I can't help with that." -- a plain
   LLM safety-filter refusal, stored as if it were valid respondent text,
   because the generator's own acceptance check only tested
   `len(text) >= 3`. Both were silently included in the 449-row dataset
   used for the published 82.85%/83.06% accuracy numbers. The generator
   (`eval/generate_synthetic_isic_iscedf_benchmark.py`) was fixed the same
   day to reject refusal-pattern output at generation time (see its own
   `_looks_like_refusal()`); this check exists so every row, from every
   run past or future, is still independently re-verified rather than
   trusting the generator fix alone.

This is a QUALITY check, not a re-derivation of gold labels -- it never
changes, drops, or re-scores what class a row belongs to, and it is not
an LLM-as-judge (no live model call at all). It answers a different
question than `run_synthetic_isic_iscedf_eval.py`: not "does the
classifier get the right answer" but "is the generated text itself
usable evidence."

Usage
-----
    python -m eval.validate_synthetic_benchmark_quality \\
        --csv eval/results/synthetic_isic_iscedf_benchmark/<file>.csv \\
        --out eval/results/synthetic_isic_iscedf_benchmark/
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Reuse the exact same script-range regexes backend/agents/language_processor.py
# already uses for its own code-switch detection, so "what counts as Arabic
# script" stays defined in exactly one place in this codebase. Explicit
# \uXXXX escapes (not literal characters) throughout -- copied verbatim
# from language_processor.py's own _ARABIC_RE/_DEVANAGARI_RE for the two
# ranges shared with it, so codepoint ranges are directly verifiable
# rather than relying on how a literal character renders in an editor.
_ARABIC_RE = re.compile(
    "[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]+"
)
_DEVANAGARI_RE = re.compile("[ऀ-ॿ]+")
_LATIN_RE = re.compile(r"[A-Za-z]+")

# CJK Unicode ranges -- deliberately broader than just "Chinese" (also
# catches Japanese kana / Korean hangul mixing), since the one real,
# documented defect (qwen2.5:3b's Arabic output, see the generator's
# docstring) was specifically CJK characters appearing mid-sentence in
# text that should have been pure Arabic. Any hit here is a hard,
# unambiguous defect for every one of this project's 6 language codes --
# none of them use these scripts. Covers CJK Unified Ideographs
# (一-鿿), CJK Unified Ideographs Extension A (㐀-䶿),
# Hiragana (぀-ゟ), Katakana (゠-ヿ), and Hangul
# Syllables (가-힯).
_CJK_RE = re.compile(
    "[一-鿿㐀-䶿぀-ゟ゠-ヿ가-힯]+"
)

# Expected primary script family per language code. "latin" covers en/tl
# (Filipino/Tagalog is written in the Latin alphabet); ar/ar-gulf/ur all
# use Arabic-range Unicode (Urdu is Perso-Arabic script) so they share one
# family here -- this check cannot distinguish Urdu from Arabic by script
# alone, only catch a row written in the wrong SCRIPT family entirely.
_EXPECTED_SCRIPT: dict[str, str] = {
    "en": "latin", "tl": "latin",
    "ar": "arabic", "ar-gulf": "arabic", "ur": "arabic",
    "hi": "devanagari",
}

# Broad sanity bounds, deliberately looser than the generation prompt's own
# stated 6-20 word target (see generate_synthetic_isic_iscedf_benchmark.py) --
# this isn't policing prompt compliance, it's catching degenerate output
# (empty strings, single-word non-answers, runaway repetition).
_MIN_WORDS = 3
_MAX_WORDS = 40

# A leaked title is only flagged if the shared substring is at least this
# long -- avoids false positives on short, generic titles (e.g. a title
# like "Other" would trivially match many casual sentences).
_MIN_LEAK_TITLE_CHARS = 6

# Real, live-caught defect (2026-08-27): a small number of generation
# calls -- confirmed for gold_code 9609 "Other personal service
# activities n.e.c." in hi/tl, whose official examples list includes
# "escort services, dating services" -- returned a plain-English LLM
# refusal/apology instead of respondent text (e.g. "I'm sorry, but I
# can't help with that."). The generator's own row-acceptance check
# before this fix only tested `len(text) >= 3`, so a refusal sailed
# through as if it were a valid, gold-labelled dataset row -- silently
# included in earlier accuracy numbers. Checked here because it recurs
# probabilistically (the same code succeeded normally in the other 4
# languages), so it is not safely assumed fixed just because the
# generator itself was patched -- every row, from every run, still gets
# checked. English-only phrasing is deliberate: this is a heuristic over
# an LLM's own refusal phrasing, not a translated-refusal detector.
# ['’]? (not a plain '?) is deliberate: the real refusal rows this
# was built to catch use a curly/typographic apostrophe (U+2019, "I’m
# sorry"), not a straight ASCII one -- a straight-quote-only pattern
# silently failed to match the exact text it was written to catch,
# caught by testing this regex directly against the real row before
# trusting it (see CLAUDE.md's "Knowledge base construction" log).
_REFUSAL_PATTERNS = re.compile(
    "i['’]?m sorry|i am sorry|i cannot help|i can['’]?t help|"
    "i cannot assist|i can['’]?t assist|as an ai|i am unable to|"
    "i['’]?m unable to|i cannot fulfill|i can['’]?t fulfill|"
    "i['’]?m not able to|i am not able to",
    re.IGNORECASE,
)


@dataclass
class RowQualityResult:
    case_id: str
    input_language: str
    standard: str
    gold_code: str
    cjk_contamination: bool
    script_match: bool | None  # None when the language has no expected script (should not happen for these 6)
    dominant_script_fraction: float | None
    word_count: int
    length_ok: bool
    title_leak: bool
    refusal_pattern: bool
    passed: bool
    failure_reasons: list[str] = field(default_factory=list)


@dataclass
class LanguageQualitySummary:
    language: str
    n: int
    n_passed: int
    pass_rate: float
    n_cjk_contamination: int
    n_script_mismatch: int
    n_length_fail: int
    n_title_leak: int
    n_refusal_pattern: int


def _script_fraction(text: str, script: str) -> float:
    if script == "arabic":
        matches = _ARABIC_RE.findall(text)
    elif script == "devanagari":
        matches = _DEVANAGARI_RE.findall(text)
    elif script == "latin":
        matches = _LATIN_RE.findall(text)
    else:
        return 0.0
    script_chars = sum(len(m) for m in matches)
    total_alpha = sum(
        len(m) for m in (_ARABIC_RE.findall(text) + _DEVANAGARI_RE.findall(text) + _LATIN_RE.findall(text))
    )
    if total_alpha == 0:
        return 0.0
    return script_chars / total_alpha


def _normalise_for_leak_check(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def check_row(
    case_id: str, input_text: str, input_language: str, standard: str,
    gold_code: str, gold_title: str,
) -> RowQualityResult:
    reasons: list[str] = []

    cjk_hit = bool(_CJK_RE.search(input_text))
    if cjk_hit:
        reasons.append("cjk_contamination")

    expected_script = _EXPECTED_SCRIPT.get(input_language)
    script_match: bool | None = None
    dominant_fraction: float | None = None
    if expected_script is not None:
        dominant_fraction = _script_fraction(input_text, expected_script)
        script_match = dominant_fraction >= 0.5
        if not script_match:
            reasons.append("script_mismatch")

    words = input_text.split()
    word_count = len(words)
    length_ok = _MIN_WORDS <= word_count <= _MAX_WORDS
    if not length_ok:
        reasons.append("length_out_of_bounds")

    title_leak = False
    normalised_title = _normalise_for_leak_check(gold_title)
    normalised_text = _normalise_for_leak_check(input_text)
    if len(normalised_title) >= _MIN_LEAK_TITLE_CHARS and normalised_title in normalised_text:
        title_leak = True
        reasons.append("title_leak")

    refusal_hit = bool(_REFUSAL_PATTERNS.search(input_text))
    if refusal_hit:
        reasons.append("refusal_pattern")

    passed = not reasons
    return RowQualityResult(
        case_id=case_id, input_language=input_language, standard=standard,
        gold_code=gold_code, cjk_contamination=cjk_hit, script_match=script_match,
        dominant_script_fraction=dominant_fraction, word_count=word_count,
        length_ok=length_ok, title_leak=title_leak, refusal_pattern=refusal_hit, passed=passed,
        failure_reasons=reasons,
    )


def check_csv(rows: list[dict]) -> list[RowQualityResult]:
    return [
        check_row(
            case_id=r["case_id"], input_text=r["input_text"],
            input_language=r["input_language"], standard=r["standard"],
            gold_code=r["gold_code"], gold_title=r.get("gold_title", ""),
        )
        for r in rows
    ]


def summarise_by_language(results: list[RowQualityResult]) -> list[LanguageQualitySummary]:
    by_lang: dict[str, list[RowQualityResult]] = {}
    for r in results:
        by_lang.setdefault(r.input_language, []).append(r)
    summaries = []
    for lang, rs in sorted(by_lang.items()):
        n = len(rs)
        n_passed = sum(1 for r in rs if r.passed)
        summaries.append(LanguageQualitySummary(
            language=lang, n=n, n_passed=n_passed,
            pass_rate=round(n_passed / n, 4) if n else 0.0,
            n_cjk_contamination=sum(1 for r in rs if r.cjk_contamination),
            n_script_mismatch=sum(1 for r in rs if r.script_match is False),
            n_length_fail=sum(1 for r in rs if not r.length_ok),
            n_title_leak=sum(1 for r in rs if r.title_leak),
            n_refusal_pattern=sum(1 for r in rs if r.refusal_pattern),
        ))
    return summaries


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_report(results: list[RowQualityResult], summaries: list[LanguageQualitySummary], out_dir: Path, source_csv: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"quality_check_{ts}"

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps({
        "source_csv": str(source_csv),
        "generated_utc": ts,
        "n_rows": len(results),
        "n_passed": sum(1 for r in results if r.passed),
        "overall_pass_rate": round(sum(1 for r in results if r.passed) / len(results), 4) if results else 0.0,
        "by_language": [asdict(s) for s in summaries],
        "failed_rows": [asdict(r) for r in results if not r.passed],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else [])
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["failure_reasons"] = ";".join(row["failure_reasons"])
            writer.writerow(row)

    md_path = out_dir / f"{stem}.md"
    n = len(results)
    n_passed = sum(1 for r in results if r.passed)
    lines = [
        "# Synthetic ISIC/ISCED-F benchmark -- automated quality check",
        "",
        f"Source: `{source_csv}`",
        f"Generated: {ts}",
        "",
        f"**Overall: {n_passed}/{n} rows passed all checks "
        f"({round(100 * n_passed / n, 2) if n else 0.0}%).**",
        "",
        "This is an automated, non-LLM, script/length/leakage check -- it does",
        "NOT re-derive or re-score gold labels. See this script's module",
        "docstring for exactly what each check does and does not verify.",
        "",
        "| Language | n | Passed | Pass rate | CJK contamination | Script mismatch | Length fail | Title leak | Refusal pattern |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.language} | {s.n} | {s.n_passed} | {round(100 * s.pass_rate, 2)}% "
            f"| {s.n_cjk_contamination} | {s.n_script_mismatch} | {s.n_length_fail} | {s.n_title_leak} "
            f"| {s.n_refusal_pattern} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"json": json_path, "csv": csv_path, "md": md_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = load_csv(args.csv)
    results = check_csv(rows)
    summaries = summarise_by_language(results)
    paths = write_report(results, summaries, args.out, args.csv)

    n = len(results)
    n_passed = sum(1 for r in results if r.passed)
    print(f"{n_passed}/{n} rows passed ({round(100 * n_passed / n, 2) if n else 0.0}%)")
    for s in summaries:
        print(f"  {s.language}: {s.n_passed}/{s.n} ({round(100 * s.pass_rate, 2)}%)")
    print(f"Report written to: {paths['md']}")


if __name__ == "__main__":
    main()
