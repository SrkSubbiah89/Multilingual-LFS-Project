"""
eval/sre_expanded_validation.py

Module D: expand Semantic Relation Engine (SRE) validation from the
existing 10-case backend/evaluation/semantic_demo.py to >=50 cases,
covering coherent / LOW / MODERATE / HIGH severity, with each case's
predicted severity computed directly from _ISCO_MAJOR_TO_ISIC /
_ISCO_SUBMAJOR_TO_ISIC / _ISCO_MAJOR_TO_ISCED / _ISCO_SUBMAJOR_TO_ISCED_MIN
(the real tables the engine actually uses) BEFORE running the case, then
checked against the engine's real output -- this is the actual
validation, not just recording whatever the engine happens to produce.

IMPORTANT: as of 2026-08-16, these tables are documented (see
semantic_relation.py's own corrected module docstring) as hand-built
domain-reasoning heuristics, NOT a transcription of an official ILO/UNESCO
correspondence table -- verified directly that neither originally-cited
document (ILO "ISCO-08 Correspondence Table with ISIC Rev.4", UNESCO
"ISCED 2011 Operational Manual" Table 7) actually contains this content.
This script validates the engine's OWN internal consistency (does its
output match what its own tables predict), not correctness against an
external authority that turned out not to exist for this purpose.

use_llm=False throughout, per this task's explicit instruction not to
touch the optional LLM re-inference sub-feature -- this keeps the run
fully deterministic (required for the 3x reproducibility check) and
avoids invoking it at all, not just avoiding changing its logic.

Usage
-----
    python eval/sre_expanded_validation.py --out backend/evaluation/sre_expanded_validation.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.agents.semantic_relation import (  # noqa: E402
    SemanticRelationEngine,
    _ISCO_MAJOR_TO_ISIC,
    _ISCO_SUBMAJOR_TO_ISIC,
    _ISCO_MAJOR_TO_ISCED,
    _ISCO_SUBMAJOR_TO_ISCED_MIN,
)


def _real_isco_pool() -> dict[str, list[str]]:
    """Real 4-digit ISCO codes actually defined in the KB, grouped by
    2-digit submajor prefix, pulled directly from load_full_isco.py
    (not hand-copied)."""
    text = Path("backend/rag/load_full_isco.py").read_text(encoding="utf-8")
    m = re.search(r"_UNIT: list\[tuple\[str, str\]\] = \[(.*?)\n\]", text, re.S)
    codes = re.findall(r'\("(\d{4})",\s*"([^"]+)"\)', m.group(1))
    by_submajor: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for code, label in codes:
        by_submajor[code[:2]].append((code, label))
    return by_submajor


_ISCO_POOL = _real_isco_pool()

# All ISIC section letters this engine's tables reference
_ALL_ISIC_SECTIONS = sorted({s for v in _ISCO_MAJOR_TO_ISIC.values() for s in v if s != "ANY"}
                             | {s for v in _ISCO_SUBMAJOR_TO_ISIC.values() for s in v})


def _pick_isco(submajor: str) -> tuple[str, str]:
    """Real (code, job_title) for a given 2-digit submajor prefix."""
    candidates = _ISCO_POOL.get(submajor)
    if not candidates:
        raise RuntimeError(f"no real ISCO codes found for submajor {submajor!r}")
    return candidates[0]


def _predicted_isic_severity(major: str, submajor: str, isic: str) -> tuple[str, bool]:
    """Returns (severity_or_'COHERENT', is_compatible) predicted from the
    real tables, mirroring _check_isco_isic's own logic exactly."""
    submajor_allowed = _ISCO_SUBMAJOR_TO_ISIC.get(submajor)
    allowed = submajor_allowed if submajor_allowed is not None else _ISCO_MAJOR_TO_ISIC.get(major, ["ANY"])
    if "ANY" in allowed or isic in allowed:
        return "COHERENT", True
    major_allowed = _ISCO_MAJOR_TO_ISIC.get(major, [])
    if "ANY" not in major_allowed and isic in major_allowed:
        return "MODERATE", True
    return "HIGH", False


def _predicted_isced_severity(major: str, submajor: str, isced: int) -> tuple[str, bool]:
    min_l, max_l, _typical = _ISCO_MAJOR_TO_ISCED.get(major, (0, 8, 4))
    sub_min = _ISCO_SUBMAJOR_TO_ISCED_MIN.get(submajor)
    if sub_min is not None:
        min_l = max(min_l, sub_min)
    if min_l <= isced <= max_l:
        return "COHERENT", True
    gap = (min_l - isced) if isced < min_l else (isced - max_l)
    if gap <= 1:
        return "LOW", False
    elif gap == 2:
        return "MODERATE", False
    return "HIGH", False


def _severity_rank(sev: str) -> int:
    return {"COHERENT": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3}[sev]


def build_cases() -> list[dict]:
    """Constructs cases targeting each severity band precisely, computed
    from the real tables (not guessed), using real ISCO codes from the KB."""
    cases = []

    def add(desc: str, submajor: str, isic: str, isced: int, expect_overall: str):
        major = submajor[0]
        code, title = _pick_isco(submajor)
        isic_sev, _ = _predicted_isic_severity(major, submajor, isic)
        isced_sev, _ = _predicted_isced_severity(major, submajor, isced)
        predicted_overall = max(isic_sev, isced_sev, key=_severity_rank)
        cases.append({
            "description": desc,
            "isco_code": code,
            "job_title": title,
            "isic_section": isic,
            "isced_level": isced,
            "predicted_isic_severity": isic_sev,
            "predicted_isced_severity": isced_sev,
            "predicted_overall_severity": predicted_overall,
        })
        assert predicted_overall == expect_overall, (
            f"{desc}: computed predicted severity {predicted_overall!r} != "
            f"declared expectation {expect_overall!r} (isic={isic_sev}, isced={isced_sev})"
        )

    # --- COHERENT: fully compatible on both ISIC and ISCED ---
    add("22 Health Professional, ISIC Q, ISCED 7 (Master's, within submajor min)", "22", "Q", 7, "COHERENT")
    add("25 ICT Professional, ISIC J, ISCED 6 (Bachelor's, meets submajor min)", "25", "J", 6, "COHERENT")
    add("23 Teaching Professional, ISIC P, ISCED 6", "23", "P", 6, "COHERENT")
    add("26 Legal Professional, ISIC M, ISCED 7", "26", "M", 7, "COHERENT")
    add("11 Manager (major-level ANY ISIC), ISIC C, ISCED 6 (within major 1-8)", "11", "C", 6, "COHERENT")
    add("61 Market Gardener, ISIC A, ISCED 1 (within major-6 range 0-3)", "61", "A", 1, "COHERENT")
    add("82 Assembler, ISIC C, ISCED 3 (within major-8 range 2-4)", "82", "C", 3, "COHERENT")
    add("32 Health Technician, ISIC Q, ISCED 5 (within major-3 range 4-6)", "32", "Q", 5, "COHERENT")
    add("71 Building Trades, ISIC F, ISCED 3 (within major-7 range 2-5)", "71", "F", 3, "COHERENT")
    add("83 Drivers, ISIC H, ISCED 3 (within major-8 range 2-4)", "83", "H", 3, "COHERENT")
    add("21 Science/Engineering Professional, ISIC J, ISCED 7 (within major-2 6-8)", "21", "J", 7, "COHERENT")
    add("31 Science/Engineering Technician, ISIC F, ISCED 5 (within major-3 4-6)", "31", "F", 5, "COHERENT")
    add("33 Business Technician, ISIC K, ISCED 4 (within major-3 4-6)", "33", "K", 4, "COHERENT")
    add("62 Subsistence Farmer, ISIC A, ISCED 0 (within major-6 0-3)", "62", "A", 0, "COHERENT")

    # --- LOW: ISCED gap<=1 outside range, ISIC fully compatible ---
    add("22 Health Professional (submajor min ISCED>=7), ISIC Q, ISCED 6 (gap=1 below)", "22", "Q", 6, "LOW")
    add("23 Teaching Professional (submajor min ISCED>=6), ISIC P, ISCED 5 (gap=1 below)", "23", "P", 5, "LOW")
    add("25 ICT Professional (submajor min ISCED>=6), ISIC J, ISCED 5 (gap=1 below)", "25", "J", 5, "LOW")
    add("26 Legal Professional (submajor min ISCED>=7), ISIC M, ISCED 6 (gap=1 below)", "26", "M", 6, "LOW")
    add("11 Manager (major range 5-8), ISIC C, ISCED 4 (gap=1 below)", "11", "C", 4, "LOW")
    add("61 Market Gardener (major range 0-3), ISIC A, ISCED 4 (gap=1 above)", "61", "A", 4, "LOW")
    add("82 Assembler (major range 2-4), ISIC C, ISCED 5 (gap=1 above)", "82", "C", 5, "LOW")
    add("32 Health Technician (major range 4-6), ISIC Q, ISCED 3 (gap=1 below)", "32", "Q", 3, "LOW")
    add("71 Building Trades (major range 2-5), ISIC F, ISCED 6 (gap=1 above)", "71", "F", 6, "LOW")
    add("83 Drivers (major range 2-4), ISIC H, ISCED 5 (gap=1 above)", "83", "H", 5, "LOW")
    add("21 Science/Eng Professional (major range 6-8), ISIC J, ISCED 5 (gap=1 below)", "21", "J", 5, "LOW")
    add("62 Subsistence Farmer (major range 0-3), ISIC A, ISCED 4 (gap=1 above)", "62", "A", 4, "LOW")

    # --- MODERATE: ISIC section in major-list but not stricter submajor-list ---
    add("22 Health Professional, ISIC J (major-2 allows J, submajor-22 only Q) MODERATE-ISIC", "22", "J", 7, "MODERATE")
    add("25 ICT Professional, ISIC M (major-2 allows M, submajor-25 only J) MODERATE-ISIC", "25", "M", 6, "MODERATE")
    add("23 Teaching Professional, ISIC J (major-2 allows J, submajor-23 only P) MODERATE-ISIC", "23", "J", 6, "MODERATE")
    add("26 Legal Professional, ISIC Q (major-2 allows Q, submajor-26 only M/R/J) MODERATE-ISIC", "26", "Q", 7, "MODERATE")
    add("32 Health Technician, ISIC C (major-3 allows C, submajor-32 only Q) MODERATE-ISIC", "32", "C", 5, "MODERATE")
    # --- MODERATE: ISCED gap==2 ---
    add("22 Health Professional (submajor min>=7), ISIC Q, ISCED 5 (gap=2 below)", "22", "Q", 5, "MODERATE")
    add("25 ICT Professional (submajor min>=6), ISIC J, ISCED 4 (gap=2 below)", "25", "J", 4, "MODERATE")
    add("11 Manager (major range 5-8), ISIC C, ISCED 3 (gap=2 below)", "11", "C", 3, "MODERATE")
    add("61 Market Gardener (major range 0-3), ISIC A, ISCED 5 (gap=2 above)", "61", "A", 5, "MODERATE")
    add("82 Assembler (major range 2-4), ISIC C, ISCED 6 (gap=2 above)", "82", "C", 6, "MODERATE")
    add("71 Building Trades (major range 2-5), ISIC F, ISCED 7 (gap=2 above)", "71", "F", 7, "MODERATE")
    add("32 Health Technician (major range 4-6), ISIC Q, ISCED 2 (gap=2 below)", "32", "Q", 2, "MODERATE")
    add("83 Drivers (major range 2-4), ISIC H, ISCED 6 (gap=2 above)", "83", "H", 6, "MODERATE")

    # --- HIGH: ISIC section not in major-list at all ---
    add("22 Health Professional, ISIC A (not in major-2 list at all) HIGH-ISIC", "22", "A", 7, "HIGH")
    add("6 Agriculture worker equivalent -- 61 Market Gardener, ISIC J (major-6 only A) HIGH-ISIC", "61", "J", 1, "HIGH")
    add("25 ICT Professional, ISIC A (not in major-2 list) HIGH-ISIC", "25", "A", 6, "HIGH")
    add("82 Assembler, ISIC Q (major-8 does not include Q) HIGH-ISIC", "82", "Q", 3, "HIGH")
    add("71 Building Trades, ISIC Q (major-7 does not include Q) HIGH-ISIC", "71", "Q", 3, "HIGH")
    add("83 Drivers, ISIC A (major-8 does not include A) HIGH-ISIC", "83", "A", 3, "HIGH")
    # --- HIGH: ISCED gap>=3 ---
    add("22 Health Professional (submajor min>=7), ISIC Q, ISCED 3 (gap=4 below)", "22", "Q", 3, "HIGH")
    add("25 ICT Professional (submajor min>=6), ISIC J, ISCED 2 (gap=4 below)", "25", "J", 2, "HIGH")
    add("11 Manager (major range 5-8), ISIC C, ISCED 1 (gap=4 below)", "11", "C", 1, "HIGH")
    add("61 Market Gardener (major range 0-3), ISIC A, ISCED 7 (gap=4 above)", "61", "A", 7, "HIGH")
    add("82 Assembler (major range 2-4), ISIC C, ISCED 8 (gap=4 above)", "82", "C", 8, "HIGH")
    add("71 Building Trades (major range 2-5), ISIC F, ISCED 0 (gap=2 below)" , "71", "F", 0, "MODERATE")
    add("32 Health Technician (major range 4-6), ISIC Q, ISCED 0 (gap=4 below)", "32", "Q", 0, "HIGH")
    add("83 Drivers (major range 2-4), ISIC H, ISCED 8 (gap=4 above)", "83", "H", 8, "HIGH")
    add("23 Teaching Professional (submajor min>=6), ISIC P, ISCED 2 (gap=4 below)", "23", "P", 2, "HIGH")
    add("26 Legal Professional (submajor min>=7), ISIC M, ISCED 3 (gap=4 below)", "26", "M", 3, "HIGH")

    # --- additional COHERENT to balance counts ---
    add("24 Business/Admin Professional, ISIC K, ISCED 6 (within major-2 6-8)", "24", "K", 6, "COHERENT")
    add("34 Legal/Social Technician, ISIC P, ISCED 5 (within major-3 4-6)", "34", "P", 5, "COHERENT")
    add("72 Metal/Machinery Trades, ISIC C, ISCED 3 (within major-7 2-5)", "72", "C", 3, "COHERENT")
    add("81 Stationary Plant Operators, ISIC C, ISCED 3 (within major-8 2-4)", "81", "C", 3, "COHERENT")
    add("73 Precision/Handicraft, ISIC C, ISCED 3 (within major-7 2-5)", "73", "C", 3, "COHERENT")
    add("74 Electrical Trades, ISIC C, ISCED 3 (within major-7 2-5)", "74", "C", 3, "COHERENT")

    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="backend/evaluation/sre_expanded_validation.json")
    args = parser.parse_args()

    cases = build_cases()
    print(f"Built {len(cases)} cases (target >=50)")
    assert len(cases) >= 50, f"only {len(cases)} cases built, need >=50"

    engine = SemanticRelationEngine(use_llm=False)  # never touches the LLM re-inference sub-feature

    results = []
    for c in cases:
        r = engine.analyse(
            isco_code=c["isco_code"],
            isic_section=c["isic_section"],
            isced_level=c["isced_level"],
            job_title=c["job_title"],
            language="en",
        )
        actual_violation_severities = sorted({v.severity for v in r.violations}, key=_severity_rank, reverse=True)
        actual_overall = actual_violation_severities[0] if actual_violation_severities else "COHERENT"
        results.append({
            **c,
            "actual_score": r.score,
            "actual_is_coherent": r.is_coherent,
            "actual_violations": [
                {"severity": v.severity, "rule_id": v.rule_id, "violation_type": v.violation_type}
                for v in r.violations
            ],
            "actual_overall_severity": actual_overall,
            "prediction_matches_actual": actual_overall == c["predicted_overall_severity"],
        })

    mismatches = [r for r in results if not r["prediction_matches_actual"]]

    by_severity = defaultdict(list)
    for r in results:
        by_severity[r["actual_overall_severity"]].append(r["actual_score"])

    aggregate = {
        sev: {
            "n": len(scores),
            "mean_score": round(sum(scores) / len(scores), 4),
        }
        for sev, scores in by_severity.items()
    }

    out = {
        "n_cases": len(results),
        "n_mismatches": len(mismatches),
        "all_predictions_matched": len(mismatches) == 0,
        "severity_distribution": {sev: len(v) for sev, v in by_severity.items()},
        "aggregate_by_severity": aggregate,
        "mismatches": mismatches,
        "cases": results,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"severity_distribution: {out['severity_distribution']}")
    print(f"mismatches: {len(mismatches)}")


if __name__ == "__main__":
    main()
