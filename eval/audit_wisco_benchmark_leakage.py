"""
eval/audit_wisco_benchmark_leakage.py

Conference I Reviewer #2 response, Step 7A: strict leakage audit for a
WISCO-derived controlled_benchmark_schema.BenchmarkRecord package (see
eval/build_wisco_isco_benchmark.py). Performs every Phase B check from
Documentation/Conference_I_Reviewer_2/WISCO_LEAKAGE_AUDIT_AND_RUN_PLAN.md
WITHOUT invoking any classifier, LLM, Qdrant, or network call -- pure,
deterministic, offline data auditing.

benchmark_id format is "WISCO-{source_key}-{language}" (see
build_wisco_isco_benchmark.py's build_records()) -- source_key is the raw
WISCO occupation key, the canonical grouping unit for leakage purposes
(all of one occupation's language variants must share a split).

Usage
-----
    from audit_wisco_benchmark_leakage import run_full_audit
    report = run_full_audit(records, expected_dataset_hash="...")
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

_ISCO4_RE = re.compile(r"^\d{4}$")


def parse_source_key(benchmark_id: str) -> tuple[str, str]:
    """'WISCO-{source_key}-{language}' -> (source_key, language). Raises
    ValueError if the ID doesn't match this exact 3-part shape -- a
    malformed ID must never be silently mis-parsed into a wrong group."""
    parts = benchmark_id.split("-")
    if len(parts) != 3 or parts[0] != "WISCO":
        raise ValueError(f"benchmark_id {benchmark_id!r} does not match the expected 'WISCO-<key>-<lang>' shape")
    return parts[1], parts[2]


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# ---------------------------------------------------------------------------
# Phase B.1 -- source-family split check
# ---------------------------------------------------------------------------

def check_source_family_split(records: list[dict]) -> dict:
    """Every record's source_key must map to exactly one split among
    {dev, heldout} (an 'excluded' record is not itself a leak partner, but
    a source_key spanning dev+excluded or heldout+excluded is still fine --
    only a dev+heldout co-occurrence is leakage)."""
    key_to_splits: dict[str, set[str]] = defaultdict(set)
    for r in records:
        source_key, _lang = parse_source_key(r["benchmark_id"])
        key_to_splits[source_key].add(r["split"])

    leaking_keys = {
        k: sorted(splits) for k, splits in key_to_splits.items()
        if "dev" in splits and "heldout" in splits
    }
    return {
        "n_source_keys": len(key_to_splits),
        "n_leaking_source_keys": len(leaking_keys),
        "leaking_source_keys_sample": dict(list(leaking_keys.items())[:20]),
        "ok": len(leaking_keys) == 0,
    }


# ---------------------------------------------------------------------------
# Phase B.2 -- text-duplicate check
# ---------------------------------------------------------------------------

def check_text_duplicates(records: list[dict]) -> dict:
    # (language, normalized_text) -> list of (benchmark_id, split)
    groups: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for r in records:
        key = (r["language"], normalize_text(r["input_text"]))
        groups[key].append((r["benchmark_id"], r["split"]))

    within_split_dupes = 0
    cross_split_dupes = 0
    cross_split_examples = []
    for (lang, text), members in groups.items():
        if len(members) < 2:
            continue
        splits_present = {m[1] for m in members}
        if len(splits_present) > 1 and "dev" in splits_present and "heldout" in splits_present:
            cross_split_dupes += 1
            if len(cross_split_examples) < 20:
                cross_split_examples.append({"language": lang, "n_records": len(members), "splits": sorted(splits_present)})
        else:
            within_split_dupes += 1

    # Cross-language translation-family duplicate check: same source_key
    # appearing with the SAME normalized text in two different languages
    # would indicate an untranslated/copy-pasted title -- informational
    # only, not a leakage signal (translation-family membership is already
    # guaranteed correct by construction, since benchmark_id embeds source_key).
    by_key_text: dict[str, set[str]] = defaultdict(set)
    for r in records:
        source_key, _lang = parse_source_key(r["benchmark_id"])
        by_key_text[source_key].add(normalize_text(r["input_text"]))
    keys_with_identical_cross_lang_text = sum(1 for texts in by_key_text.values() if len(texts) == 1 and len(texts) > 0)

    return {
        "n_exact_duplicate_groups_within_split": within_split_dupes,
        "n_exact_duplicate_groups_cross_split": cross_split_dupes,
        "cross_split_duplicate_examples": cross_split_examples,
        "n_source_keys_with_identical_text_across_all_their_languages": keys_with_identical_cross_lang_text,
        "ok": cross_split_dupes == 0,
    }


# ---------------------------------------------------------------------------
# Phase B.3 -- near-duplicate check (deterministic, non-LLM)
# ---------------------------------------------------------------------------

def _blocking_key(text: str) -> str:
    """Cheap fingerprint to bound near-duplicate comparison to plausible
    candidates only -- first 3 normalized characters. Two strings that
    could plausibly be near-duplicates of each other overwhelmingly share
    a prefix; this keeps the O(n^2) SequenceMatcher comparison tractable."""
    return text[:3]


def check_near_duplicates(records: list[dict], similarity_threshold: float = 0.90, max_reported: int = 50) -> dict:
    """Reports candidate cross-split near-duplicate PAIRS for human review.
    Never removes/alters data -- see this function's docstring and the
    module docstring's restriction. Blocked by (language, first-3-chars) to
    keep runtime bounded on ~20k records."""
    dev_by_block: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    heldout_by_block: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for r in records:
        norm = normalize_text(r["input_text"])
        block = (r["language"], _blocking_key(norm))
        entry = (r["benchmark_id"], norm, r["input_text"])
        if r["split"] == "dev":
            dev_by_block[block].append(entry)
        elif r["split"] == "heldout":
            heldout_by_block[block].append(entry)

    candidates = []
    for block, dev_entries in dev_by_block.items():
        heldout_entries = heldout_by_block.get(block, [])
        if not heldout_entries:
            continue
        for dev_id, dev_norm, dev_raw in dev_entries:
            for held_id, held_norm, held_raw in heldout_entries:
                if dev_norm == held_norm:
                    continue  # already reported by check_text_duplicates
                ratio = difflib.SequenceMatcher(None, dev_norm, held_norm).ratio()
                if ratio >= similarity_threshold:
                    candidates.append({
                        "dev_benchmark_id": dev_id, "heldout_benchmark_id": held_id,
                        "similarity": round(ratio, 4),
                    })

    candidates.sort(key=lambda c: -c["similarity"])
    return {
        "similarity_threshold": similarity_threshold,
        "n_candidate_near_duplicate_pairs": len(candidates),
        "candidate_pairs_sample": candidates[:max_reported],
        "note": "Candidates are reported for human review only -- no data was altered or removed by this check.",
    }


# ---------------------------------------------------------------------------
# Phase B.4 -- code distribution check
# ---------------------------------------------------------------------------

def check_code_distribution(records: list[dict]) -> dict:
    by_split: dict[str, Counter] = {"dev": Counter(), "heldout": Counter(), "excluded": Counter()}
    unit_by_split: dict[str, set] = defaultdict(set)
    for r in records:
        code = r["gold_code"]
        split = r["split"]
        if not code:
            continue
        unit_by_split[split].add(code)
        for level, n in (("major", 1), ("submajor", 2), ("minor", 3), ("unit", 4)):
            by_split[split][f"{level}:{code[:n]}"] += 1

    dev_units = unit_by_split.get("dev", set())
    heldout_units = unit_by_split.get("heldout", set())
    dev_only = sorted(dev_units - heldout_units)
    heldout_only = sorted(heldout_units - dev_units)

    unit_counts_overall = Counter(r["gold_code"] for r in records if r["gold_code"])
    rare_units = sorted([u for u, c in unit_counts_overall.items() if c < 5])

    return {
        "n_distinct_unit_groups_dev": len(dev_units),
        "n_distinct_unit_groups_heldout": len(heldout_units),
        "unit_groups_only_in_dev": dev_only,
        "unit_groups_only_in_heldout_count": len(heldout_only),
        "n_rare_unit_groups_overall_lt5_records": len(rare_units),
        "rare_unit_groups_sample": rare_units[:20],
    }


# ---------------------------------------------------------------------------
# Phase B.5 -- language distribution check
# ---------------------------------------------------------------------------

def check_language_distribution(records: list[dict], min_heldout_per_language: int = 30) -> dict:
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        counts[r["language"]][r["split"]] += 1

    flagged = [lang for lang, c in counts.items() if c.get("heldout", 0) < min_heldout_per_language]
    return {
        "counts_by_language_and_split": {lang: dict(c) for lang, c in counts.items()},
        "min_heldout_per_language_threshold": min_heldout_per_language,
        "languages_with_insufficient_heldout_support": flagged,
        "all_languages_have_split_assignment": all(
            "dev" in c or "heldout" in c or "excluded" in c for c in counts.values()
        ),
    }


# ---------------------------------------------------------------------------
# Phase B.6 -- integrity check
# ---------------------------------------------------------------------------

def recompute_dataset_hash(records: list[dict]) -> str:
    """MUST match eval/build_wisco_isco_benchmark.py's own hash computation
    exactly (sha256 of the sorted-key JSON dump of the records payload) --
    this is an independent recomputation, not a re-read of the stored value."""
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def check_integrity(records: list[dict], expected_dataset_hash: Optional[str]) -> dict:
    recomputed = recompute_dataset_hash(records)
    hash_match = (recomputed == expected_dataset_hash) if expected_dataset_hash else None

    malformed_codes = [
        r["benchmark_id"] for r in records
        if r["gold_code"] and not _ISCO4_RE.match(r["gold_code"])
    ]
    return {
        "recomputed_dataset_hash": recomputed,
        "expected_dataset_hash": expected_dataset_hash,
        "hash_match": hash_match,
        "n_malformed_isco_codes": len(malformed_codes),
        "malformed_code_examples": malformed_codes[:20],
        "ok": (hash_match in (True, None)) and not malformed_codes,
    }


# ---------------------------------------------------------------------------
# Full audit
# ---------------------------------------------------------------------------

READY_FOR_EVALUATION = "ready_for_evaluation"
NOT_READY_UNRESOLVED_LEAKAGE = "not_ready_for_evaluation_due_to_unresolved_cross_split_leakage"


def determine_evaluation_readiness(audit_report: dict) -> str:
    """Step 7A, Phase C.3: an evaluation plan must never be prepared for a
    benchmark version with unresolved source-family or exact-duplicate
    cross-split leakage. Returns the exact required status string --
    callers (e.g. a plan-generation step) must check this before doing
    anything else."""
    if audit_report.get("leakage_found"):
        return NOT_READY_UNRESOLVED_LEAKAGE
    return READY_FOR_EVALUATION


def run_full_audit(records: list[dict], expected_dataset_hash: Optional[str] = None) -> dict:
    source_family = check_source_family_split(records)
    text_dupes = check_text_duplicates(records)
    near_dupes = check_near_duplicates(records)
    code_dist = check_code_distribution(records)
    lang_dist = check_language_distribution(records)
    integrity = check_integrity(records, expected_dataset_hash)

    leakage_found = not source_family["ok"] or not text_dupes["ok"]
    report = {
        "n_records": len(records),
        "source_family_split_check": source_family,
        "text_duplicate_check": text_dupes,
        "near_duplicate_check": near_dupes,
        "code_distribution_check": code_dist,
        "language_distribution_check": lang_dist,
        "integrity_check": integrity,
        "leakage_found": leakage_found,
        "overall_ok_no_leakage": not leakage_found and integrity["ok"],
    }
    report["evaluation_readiness"] = determine_evaluation_readiness(report)
    return report
