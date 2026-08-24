"""
eval/legacy_thesis_ch6/semantic_demo.py

Presentation-ready demo of the Semantic Relation (Three-Way Crosswalk) layer.

Shows 10 cases spanning COHERENT, MODERATE, and HIGH-violation scenarios
in both English and Arabic -- designed for manager / thesis-panel presentation.

Usage:
    python -m eval.legacy_thesis_ch6.semantic_demo
    python -m eval.legacy_thesis_ch6.semantic_demo --arabic
"""

from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from backend.agents.semantic_relation import get_semantic_relation_engine, SemanticCoherence

# ---- ANSI colours -----------------------------------------------------------
G   = "\033[92m";  Y = "\033[93m";  R = "\033[91m"
C   = "\033[96m";  W = "\033[97m";  DIM= "\033[2m";  RST= "\033[0m"
BLD = "\033[1m"

def col(text, c): return f"{c}{text}{RST}"

# ---- 10 Hard-coded test cases -----------------------------------------------
# (isco_code, isic_section, isced_level, job_title, language, expected_label)
CASES = [
    # COHERENT
    ("2211", "Q", 7, "Medical doctor",                       "en", "COHERENT"),
    ("5131", "I", 3, "Restaurant waiter",                    "en", "COHERENT"),
    ("6111", "A", 2, "Rice farmer",                          "en", "COHERENT"),
    ("2411", "K", 6, "Accountant",                           "en", "COHERENT"),
    # ARABIC case
    ("5414", "O", 4, "police officer",                       "ar", "COHERENT"),
    # MODERATE violations
    ("2211", "C", 7, "Medical doctor in factory",            "en", "MODERATE"),
    ("3115", "Q", 3, "Mechanical engineer in hospital",      "en", "MODERATE"),
    # HIGH violations (strong inconsistency)
    ("1211", "F", 2, "Finance director on construction site","en", "HIGH"),
    ("9111", "M", 7, "Cleaner with doctoral degree",         "en", "HIGH"),
    ("6111", "J", 8, "Farmer in IT sector with postgrad",    "en", "HIGH"),
]

SEP = "=" * 78

def _bar(score: float) -> str:
    filled = round(score * 20)
    bar = "#" * filled + "-" * (20 - filled)
    pct = f"{score:.0%}"
    if score >= 0.70:
        return col(bar, G) + col(f" {pct}", G)
    elif score >= 0.50:
        return col(bar, Y) + col(f" {pct}", Y)
    else:
        return col(bar, R) + col(f" {pct}", R)

def _sev_col(sev: str) -> str:
    s = sev.upper()
    if s == "HIGH":     return col(f"[{s}]", R)
    if s == "MODERATE": return col(f"[{s}]", Y)
    return col(f"[{s}]", DIM)

def print_case(idx: int, sc: SemanticCoherence, case: tuple, show_arabic: bool):
    isco, isic, isced, title, lang, expected = case
    actual = "COHERENT" if sc.is_coherent else "INCONSISTENT"

    bdr = G if sc.is_coherent else (Y if sc.score >= 0.50 else R)

    print(col("+" + "-" * 76 + "+", bdr))
    header = f" Case {idx:>2} | {title:<35} | Expected: {expected}"
    print(col("|", bdr) + col(f"{header:<76}", W) + col("|", bdr))
    print(col("+" + "-" * 76 + "+", bdr))

    meta = (f" ISCO {isco}  |  ISIC Section {isic or '-'}  |"
            f"  ISCED Level {isced if isced is not None else '-'}  |  lang={lang}")
    print(col("|", bdr) + col(f"{meta:<76}", C) + col("|", bdr))

    pct = int(sc.score * 100)
    status_str = col(f"[{actual}]", G if sc.is_coherent else R)
    print(col("|", bdr) + f" Score: {_bar(sc.score)}   {status_str}")

    isic_flag  = col("[OK] ISIC",  G) if sc.isco_isic_compatible  else col("[X]  ISIC", R)
    isced_flag = col("[OK] ISCED", G) if sc.isco_isced_compatible else col("[X]  ISCED", R)
    conf_adj   = sc.confidence_adjustment
    adj_str    = col(f"  conf adj {conf_adj:+.0%}", G if conf_adj >= 0 else R)
    print(col("|", bdr) + f"  {isic_flag}   {isced_flag}{adj_str}")

    expl = sc.explanation_ar if show_arabic else sc.explanation_en
    words = expl.split()
    line = " "; lines = []
    for w in words:
        if len(line) + len(w) + 1 > 73:
            lines.append(line)
            line = " " + w
        else:
            line += " " + w
    if line.strip():
        lines.append(line)
    for ln in lines[:3]:
        print(col("|", bdr) + col(f"{ln:<76}", DIM) + col("|", bdr))

    if sc.violations:
        print(col("|", bdr) + col(f"  Violations ({len(sc.violations)}):", W) +
              " " * (54 - len(f"  Violations ({len(sc.violations)}):")) + col("|", bdr))
        for v in sc.violations:
            msg = v.message_ar if show_arabic else v.message_en
            sev_str = _sev_col(v.severity)
            print(col("|", bdr) + f"  {sev_str} {col(msg[:65], DIM)}")

    print(col("+" + "-" * 76 + "+", bdr))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arabic", action="store_true", help="Show Arabic explanations")
    args = ap.parse_args()

    engine = get_semantic_relation_engine(use_llm=False)

    print()
    print(col(SEP, C))
    print(col("  SEMANTIC RELATION DEMO -- ISCO <-> ISIC <-> ISCED Three-Way Crosswalk", W))
    print(col("  Thesis Contribution: Cross-Standard Coherence Scoring", DIM))
    print(col(SEP, C))
    print()

    coherent_count = 0
    for i, case in enumerate(CASES, 1):
        isco, isic, isced, title, lang, _ = case
        sc = engine.analyse(
            isco_code    = isco,
            isic_section = isic,
            isced_level  = isced,
            job_title    = title,
            language     = lang,
        )
        if sc.is_coherent:
            coherent_count += 1
        print_case(i, sc, case, args.arabic)

    print(col(SEP, C))
    print(col("  SUMMARY", W))
    print(col(SEP, C))
    print(f"  Total cases:   {len(CASES)}")
    print(f"  Coherent:      {col(str(coherent_count), G)}")
    print(f"  Inconsistent:  {col(str(len(CASES) - coherent_count), R)}")
    print()
    print(col("  Coherence score interpretation:", W))
    print(f"  {col('>= 0.90', G)}  Perfect alignment  ->  ISCO confidence  {col('+10%', G)}")
    print(f"  {col('>= 0.70', G)}  Good alignment     ->  ISCO confidence  {col('+5%',  G)}")
    print(f"  {col('>= 0.50', Y)}  Partial mismatch   ->  ISCO confidence  {col('-5%',  Y)}")
    print(f"  {col('< 0.50',  R)}  Strong mismatch    ->  ISCO confidence  {col('-20%', R)}  + HITL flag")
    print()
    print(col("  Crosswalk tables used:", DIM))
    print(col("  * Hand-built ISCO<->ISIC domain-reasoning table (no official ILO", DIM))
    print(col("    ISCO-08-to-ISIC-Rev.4 correspondence table was found to exist)", DIM))
    print(col("  * Hand-built ISCO<->ISCED domain-reasoning table (the real UNESCO", DIM))
    print(col("    ISCED 2011 Operational Manual does not map occupations)", DIM))
    print(col("  * ISCO-08 Major Group definitions (ILO 2012)", DIM))
    print()


if __name__ == "__main__":
    main()
