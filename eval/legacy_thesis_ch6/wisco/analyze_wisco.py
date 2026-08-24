"""
Deeper analysis pass, run after inspect_wisco.py's first look. Answers the
specific questions module_a_week1_report.md and language_mapping_note.md need:
  - What are the base language codes, and how do the locale columns map onto them?
  - Per target-language column(s), how many titles have non-empty text?
  - Are the Arabic country-locale columns genuinely dialectal, or duplicated MSA text?
    (Corrected 2026-08-02: they are duplicated MSA -- see arabic_locale_check below.)
  - What does the MAPPINGS sheet's ISCO code column actually look like (digit count,
    leading zeros, join-key cardinality, duplicates)?
  - Is NACE2.0 (MAPPINGS) or the OCC>>INDUSTRY NACE2004 cross-tab the better Module D
    source? (Corrected 2026-08-02: neither alone is sufficient -- see nace_comparison.)
  - What does LABELSET contain?
  - Are there duplicate column headers in the source (data-quality check)?

CANONICAL FILE (corrected 2026-08-02): targets the 2023-08-18 file, not the
2023-02-02 file originally used. See PROVENANCE.md for why.

Usage:
    python backend/evaluation/wisco/analyze_wisco.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
WORKBOOK = RAW_DIR / "occupations_ISCO08_5dgt_55languages_4000titles_surveycodings_20230818.xlsx"
OUT_JSON = Path(__file__).resolve().parent / "data" / "interim" / "wisco_language_analysis.json"

KEY_COL = "occupai3_API_13dgt"
MASTER_LABEL_COL = "MASTER LABEL 4000"

# Our 5 target categories and the locale column(s) considered primary for each,
# chosen because they are the only single-country option (ur, hi, tl) or the
# most relevant option given the LFS system targets UAE respondents (ar -> ar_AE,
# en -> en_US as the most populated global-English variant, checked below).
TARGET_PRIMARY_COL = {
    "en": "en_US",
    "ar": "ar_AE",
    "ur": "ur_PK",
    "hi": "hi_IN",
    "tl": "tl_PH",
}


def main() -> None:
    wb = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)

    result = {}

    # ---- CODESET: language column analysis -------------------------------
    ws = wb["CODESET"]
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])
    data_rows = rows[1:]
    n_data_rows = len(data_rows)

    # Data-quality check: duplicate column headers in the source (found in this
    # file: 'it_IT' appears twice). header.index() only ever finds the first
    # occurrence, so any duplicate column's data is silently unreachable via
    # simple name lookup -- flag it rather than let it pass silently.
    header_counts = Counter(h for h in header if h is not None)
    duplicate_headers = {h: c for h, c in header_counts.items() if c > 1}
    result["duplicate_column_headers"] = duplicate_headers

    key_col_idx = header.index(KEY_COL)
    master_col_idx = header.index(MASTER_LABEL_COL)
    lang_cols = [h for h in header if h not in (KEY_COL, MASTER_LABEL_COL)]

    base_lang_counter = Counter()
    locale_by_base = defaultdict(list)
    for col in lang_cols:
        base = col.split("_")[0]
        base_lang_counter[base] += 1
        locale_by_base[base].append(col)

    result["n_data_rows_codeset"] = n_data_rows
    result["n_locale_columns"] = len(lang_cols)
    result["n_distinct_base_languages"] = len(base_lang_counter)
    result["locales_by_base_language"] = {k: v for k, v in locale_by_base.items()}
    result["target_language_locale_counts"] = {
        base: len(locale_by_base.get(base, [])) for base in TARGET_PRIMARY_COL
    }

    keys = [r[key_col_idx] for r in data_rows]

    # Non-empty fill rate for every locale column belonging to our 5 target bases
    fill_counts = Counter()
    key_non_null = 0
    master_non_null = 0
    for r in data_rows:
        if r[key_col_idx] not in (None, ""):
            key_non_null += 1
        if r[master_col_idx] not in (None, ""):
            master_non_null += 1
        for base in TARGET_PRIMARY_COL:
            for col in locale_by_base.get(base, []):
                idx = header.index(col)
                val = r[idx] if idx < len(r) else None
                if val not in (None, ""):
                    fill_counts[col] += 1

    result["key_col_non_null"] = key_non_null
    result["master_label_non_null"] = master_non_null
    result["fill_counts_target_locales"] = dict(fill_counts)

    # Duplicate key check
    key_counter = Counter(keys)
    dupes = {k: c for k, c in key_counter.items() if c > 1 and k not in (None, "")}
    result["duplicate_keys_in_codeset"] = {"count": len(dupes), "examples": {str(k): v for k, v in list(dupes.items())[:5]}}

    # ---- Arabic locale distinctness check (corrected 2026-08-02) ----------
    # Tests whether the Arabic country-locale columns actually contain
    # dialect-distinct text, or are the same MSA translation duplicated.
    ar_ae_idx = header.index("ar_AE")
    arabic_check = {}
    for col in locale_by_base.get("ar", []):
        if col == "ar_AE":
            continue
        idx = header.index(col)
        both, diff = 0, 0
        for r in data_rows:
            ae_val = r[ar_ae_idx]
            val = r[idx] if idx < len(r) else None
            if ae_val in (None, "") or val in (None, ""):
                continue
            both += 1
            if val != ae_val:
                diff += 1
        arabic_check[col] = {"both_present": both, "differ_from_ar_AE": diff,
                              "pct_differ": round(100 * diff / both, 2) if both else None}
    result["arabic_locale_distinctness_vs_ar_AE"] = arabic_check
    total_diff = sum(v["differ_from_ar_AE"] for v in arabic_check.values())
    result["arabic_locales_are_msa_duplicates"] = total_diff <= (0.01 * n_data_rows * len(arabic_check))

    # Sample rows for our 5 primary target columns
    sample = []
    for r in data_rows[:8]:
        row = {"key": r[key_col_idx], "master_label": r[master_col_idx]}
        for base, col in TARGET_PRIMARY_COL.items():
            idx = header.index(col)
            row[col] = r[idx] if idx < len(r) else None
        sample.append(row)
    result["sample_rows_codeset"] = sample

    # ---- MAPPINGS: ISCO code format + join key ----------------------------
    ws2 = wb["MAPPINGS"]
    rows2 = list(ws2.iter_rows(values_only=True))
    header2 = list(rows2[0])
    data2 = rows2[1:]

    key2_idx = header2.index(KEY_COL)
    isco4_idx = header2.index("ISCO0804")
    isco3_idx = header2.index("ISCO0803")
    isco2_idx = header2.index("ISCO0802")
    isco1_idx = header2.index("ISCO0801")
    isco08lv_idx = header2.index("ISCO08lv")
    nace2_idx = header2.index("NACE2.0")

    isco4_values = [r[isco4_idx] for r in data2 if r[isco4_idx] not in (None, "")]
    isco4_types = Counter(type(v).__name__ for v in isco4_values)
    isco4_sample = isco4_values[:15]
    isco4_lengths = Counter(len(str(v)) for v in isco4_values)
    leading_zero_examples = [v for v in isco4_values if str(v).startswith("0")][:10]

    keys2 = [r[key2_idx] for r in data2]
    key2_counter = Counter(keys2)
    dupes2 = {k: c for k, c in key2_counter.items() if c > 1 and k not in (None, "")}

    codeset_keys = set(k for k in keys if k not in (None, ""))
    mappings_keys = set(k for k in keys2 if k not in (None, ""))
    orphans_codeset_only = codeset_keys - mappings_keys
    orphans_mappings_only = mappings_keys - codeset_keys

    result["mappings_header"] = header2
    result["isco4_value_types"] = dict(isco4_types)
    result["isco4_sample"] = isco4_sample
    result["isco4_string_lengths"] = dict(isco4_lengths)
    result["isco4_leading_zero_examples"] = leading_zero_examples
    result["mappings_duplicate_keys"] = {"count": len(dupes2), "examples": {str(k): v for k, v in list(dupes2.items())[:5]}}
    result["orphans_codeset_only_count"] = len(orphans_codeset_only)
    result["orphans_mappings_only_count"] = len(orphans_mappings_only)
    result["orphans_codeset_only_examples"] = [str(x) for x in list(orphans_codeset_only)[:5]]
    result["orphans_mappings_only_examples"] = [str(x) for x in list(orphans_mappings_only)[:5]]

    # ---- ISCO08lv (skill level) --------------------------------------------
    isco08lv_values = [r[isco08lv_idx] for r in data2 if r[isco08lv_idx] not in (None, "")]
    result["isco08lv_non_null"] = len(isco08lv_values)
    result["isco08lv_distribution"] = dict(Counter(isco08lv_values))

    # ---- NACE2.0 (MAPPINGS) analysis ---------------------------------------
    nace2_by_key = {r[key2_idx]: r[nace2_idx] for r in data2}
    nace2_non_null = [v for v in nace2_by_key.values() if v not in (None, "")]
    result["nace2_0_rows_with_value"] = len(nace2_non_null)
    result["nace2_0_total_rows"] = len(data2)
    result["nace2_0_fill_rate"] = round(len(nace2_non_null) / len(data2), 4)
    result["nace2_0_value_lengths"] = dict(Counter(len(str(v)) for v in nace2_non_null))
    result["nace2_0_distinct_values"] = len(set(nace2_non_null))

    # ---- OCC>>INDUSTRY usability -------------------------------------------
    ws3 = wb["OCC>>INDUSTRY"]
    rows3 = list(ws3.iter_rows(values_only=True))
    header3 = list(rows3[0])
    data3 = rows3[1:]
    key3_idx = header3.index("ISCO0813") if "ISCO0813" in header3 else None
    nace_cols = [h for h in header3 if h and h.startswith("NACE")]
    non_empty_cells = 0
    total_cells = 0
    distinct_values = Counter()
    rows_with_real_value = 0
    for r in data3:
        vals = [r[header3.index(h)] for h in nace_cols]
        real_vals = [v for v in vals if v not in (None, "", 99999)]
        total_cells += len(nace_cols)
        non_empty_cells += len(real_vals)
        for v in real_vals:
            distinct_values[v] += 1
        if real_vals:
            rows_with_real_value += 1

    result["occ_industry_rows"] = len(data3)
    result["occ_industry_nace_cols"] = len(nace_cols)
    result["occ_industry_fill_rate_all_cells"] = round(non_empty_cells / total_cells, 4) if total_cells else None
    result["occ_industry_rows_with_ge1_real_value"] = rows_with_real_value
    result["occ_industry_row_coverage_pct"] = round(100 * rows_with_real_value / len(data3), 1) if data3 else None
    result["occ_industry_distinct_cell_values"] = dict(distinct_values.most_common(10))

    # ---- Cross-check: does NACE2.0 (Rev.2) agree with OCC>>INDUSTRY (Rev.1.1)? ----
    if key3_idx is not None:
        crosstab_by_key = {r[key3_idx]: r for r in data3}
        match, total_checked, mismatch_examples = 0, 0, []
        for key, nace2_val in nace2_by_key.items():
            if nace2_val in (None, ""):
                continue
            r = crosstab_by_key.get(key)
            if r is None:
                continue
            vals = [r[header3.index(h)] for h in nace_cols]
            real_vals = set(v for v in vals if v not in (None, "", 99999))
            total_checked += 1
            if nace2_val in real_vals:
                match += 1
            elif len(mismatch_examples) < 5:
                mismatch_examples.append({"key": str(key), "nace2_0": nace2_val, "crosstab_nace2004": list(real_vals)[:5]})
        result["nace_comparison"] = {
            "nace2_0_value_found_in_crosstab_pct": round(100 * match / total_checked, 1) if total_checked else None,
            "checked": total_checked, "matched": match,
            "mismatch_examples": mismatch_examples,
        }

    # ---- LABELSET content ---------------------------------------------------
    ws4 = wb["LABELSET ISCO-1-4dgt NACE"]
    rows4 = list(ws4.iter_rows(values_only=True))
    header4 = list(rows4[0])
    data4 = rows4[1:]
    dataset_types = Counter(r[header4.index("DATASET TYPE")] for r in data4 if r[header4.index("DATASET TYPE")])
    result["labelset_header"] = header4
    result["labelset_dataset_types"] = dict(dataset_types)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    # ASCII-safe console summary (avoid Windows cp1252 crash on Arabic/Devanagari text)
    print("n_data_rows_codeset:", result["n_data_rows_codeset"])
    print("n_distinct_base_languages:", result["n_distinct_base_languages"])
    print("duplicate_column_headers:", result["duplicate_column_headers"])
    print("target_language_locale_counts:", result["target_language_locale_counts"])
    print("arabic_locales_are_msa_duplicates:", result["arabic_locales_are_msa_duplicates"],
          "(total differing cells across all 21 non-AE Arabic columns:", total_diff, ")")
    print("isco08lv_distribution:", result["isco08lv_distribution"])
    print("nace2_0_fill_rate:", result["nace2_0_fill_rate"], f"({result['nace2_0_rows_with_value']}/{result['nace2_0_total_rows']})")
    print("occ_industry_row_coverage_pct:", result["occ_industry_row_coverage_pct"])
    if "nace_comparison" in result:
        print("nace2_0 value found in crosstab (agreement rate):", result["nace_comparison"]["nace2_0_value_found_in_crosstab_pct"], "%")
    print(f"\nFull report (incl. non-ASCII samples) written to {OUT_JSON}")


if __name__ == "__main__":
    main()
