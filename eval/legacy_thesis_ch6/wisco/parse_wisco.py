"""
Parses the raw WISCO workbook into a flat, per-occupation JSON dataset covering
the system's 5 target languages, using the verified contract established in
wisco_structure_inspection.md (updated 2026-08-02 against the corrected file
-- do not assume a different sheet/column layout without re-running
inspect_wisco.py / analyze_wisco.py first).

CANONICAL FILE (corrected 2026-08-02): the 2023-02-02 file originally parsed
here was the oldest of 4 Zenodo versions under this concept DOI. This parser
now targets the 2023-08-18 file -- the correct/latest one, and the exact
filename the original Module A plan expected. See PROVENANCE.md.

Key facts this parser depends on (see wisco_structure_inspection.md):
  - CODESET is wide-format: one row per occupation, one column per locale.
    The master/reference-English column is 'MASTER LABEL 4000' in this file
    (was 'MASTER LABEL' in the superseded 2023-02-02 file).
  - MAPPINGS' ISCO0804 and NACE2.0 columns are both read as Python ints,
    which silently drops leading zeros (ISCO0804 for major-group-0 Armed
    Forces codes; NACE2.0 for any 3-digit-after-strip NACE Rev.2 code).
    Both are str().zfill()'d here before being treated as code strings.
  - occupai3_API_13dgt is the join key between CODESET and MAPPINGS.
  - Arabic's 22 country-locale columns are duplicated MSA text, not
    dialect-distinct (confirmed: 0 differing cells vs ar_AE across all 21
    other Arabic columns in this file). ar_AE is still used as the single
    Arabic column -- see language_mapping_note.md for why the locale choice
    no longer carries the "Gulf-dialect" justification it was originally
    given.
  - Industry crosswalk has TWO non-interchangeable sources, both kept:
      * NACE2.0 (MAPPINGS): NACE Rev.2, one value per occupation, maps
        cleanly to ISIC Rev.4 at class level, but only ~17% row coverage.
      * NACE2004_01..33 (OCC>>INDUSTRY): NACE Rev.1.1, up to 33 values per
        occupation, 100% row coverage, but needs an extra Rev.1.1-to-Rev.2
        hop before it's ISIC Rev.4-comparable. The two sources agree on
        only ~26% of rows where both exist -- they are not redundant, and
        Module D should pick per its own precision/coverage needs rather
        than this parser choosing for it.
  - ISCO08lv (ISCO-08 skill level, 1-4) is extracted for the Week 7
    ISCO<->ISCED cross-check (skill level is defined to align with ISCED
    education groups).
  - CODESET's key column is not uniformly typed. 246 rows have small
    NEGATIVE keys (e.g. -20509) whose "master label" is a section/category
    heading ("Waterworks", "Oil, gas", "Management, direction") -- these
    match the STRUCTURE sheet's own negative-numbered Level 1/2 grouping
    scheme and are not occupation titles at all; they are excluded here.
    A further 71 rows have real occupation titles ("Pre-school teacher",
    "Chef de Partie", "Yoga trainer", ...) but their key was read by
    openpyxl as an imprecise float (e.g. 1.34400030001688e+16), because the
    value's magnitude exceeds float64's ~15-17 significant-digit exact
    integer range. Rounding to the nearest int is attempted before joining
    against MAPPINGS, but checked directly: it does not recover a match for
    any of these 71 -- the precision loss happened upstream, in however the
    source workbook itself was generated/exported, not in openpyxl's read.
    These 71 are therefore genuine, currently-unrecoverable orphans (no
    alternative join path exists; MAPPINGS has no title text to fuzzy-match
    against). Together with the 8 Armed Forces rows below, they account for
    the entire orphan set this parser reports -- it is not random data
    loss, and every orphan is individually explained.
  - 8 rows are doubly broken in WISCO itself, independent of anything this
    parser does: (a) their MAPPINGS ISCO0801-0804 hierarchy is internally
    consistent but semantically wrong -- master labels are unambiguously
    Armed Forces occupations ("Commissioned officer armed forces",
    "Military weapons specialist", ...) but the code columns tag them as
    major group 1/2/3, not 0; and (b) their CODESET key
    (e.g. 1100100000008420) doesn't match their MAPPINGS key for the same
    occupation (110010000000) at all -- the two sheets disagree on this
    occupation's identity, not just its code. A theory that the join key's
    own leading digits could recover the correct ISCO code was checked and
    rejected (it also disagrees with known-correct rows like "Air force
    captain"). These 8 are quarantined/orphaned, not guessed at.

Usage:
    python eval/legacy_thesis_ch6/wisco/parse_wisco.py
"""
import json
from pathlib import Path

import openpyxl

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
WORKBOOK = RAW_DIR / "occupations_ISCO08_5dgt_55languages_4000titles_surveycodings_20230818.xlsx"
OUT_PATH = Path(__file__).resolve().parent / "data" / "processed" / "wisco_raw_parsed.json"
INDUSTRY_OUT_PATH = Path(__file__).resolve().parent / "data" / "processed" / "wisco_industry_crosswalk.json"

KEY_COL = "occupai3_API_13dgt"
MASTER_LABEL_COL = "MASTER LABEL 4000"
ISCO4_COL = "ISCO0804"
ISCO3_COL = "ISCO0803"
ISCO2_COL = "ISCO0802"
ISCO1_COL = "ISCO0801"
ISCO08LV_COL = "ISCO08lv"
NACE2_0_COL = "NACE2.0"
INDUSTRY_SENTINEL_NONE = 99999

TARGET_LOCALE_COL = {
    "en": "en_US",
    "ar": "ar_AE",
    "ur": "ur_PK",
    "hi": "hi_IN",
    "tl": "tl_PH",
}


def _sheet_to_rows(ws):
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])
    return header, rows[1:]


def _zfill_code(value, width: int) -> str | None:
    """int/str code -> zero-padded string, preserving leading zeros lost on read."""
    if value in (None, ""):
        return None
    return str(int(value)).zfill(width)


def _normalise_key(value):
    """
    CODESET's key column mixes three representations of what should be one
    integer join key: clean positive ints, small negative ints (these are
    section/category rows, not occupations -- see module docstring), and
    large positive values openpyxl reads as imprecise floats. Returns None
    for rows that aren't real occupation keys (negative/non-numeric);
    otherwise returns a clean int, rounding float-precision-loss values to
    their nearest integer so they join correctly against MAPPINGS' clean
    int keys.
    """
    if value in (None, ""):
        return None
    if not isinstance(value, (int, float)):
        return None
    if value < 0:
        return None  # section/category heading row, not an occupation
    return int(round(value))


def main() -> None:
    assert WORKBOOK.exists(), f"Workbook not found at {WORKBOOK}. Run the Day 1 download first."

    wb = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)

    codeset_header, codeset_rows = _sheet_to_rows(wb["CODESET"])
    mappings_header, mappings_rows = _sheet_to_rows(wb["MAPPINGS"])

    key_idx = codeset_header.index(KEY_COL)
    master_idx = codeset_header.index(MASTER_LABEL_COL)
    locale_idx = {lang: codeset_header.index(col) for lang, col in TARGET_LOCALE_COL.items()}

    # Build titles-by-key from CODESET
    titles_by_key: dict[int, dict] = {}
    excluded_category_rows = 0
    recovered_float_keys = 0
    codeset_duplicate_keys = 0
    for r in codeset_rows:
        raw_key = r[key_idx]
        key = _normalise_key(raw_key)
        if key is None:
            if isinstance(raw_key, (int, float)) and raw_key is not None and raw_key < 0:
                excluded_category_rows += 1
            continue
        if isinstance(raw_key, float):
            recovered_float_keys += 1
        titles = {}
        for lang, idx in locale_idx.items():
            val = r[idx] if idx < len(r) else None
            if val not in (None, ""):
                titles[lang] = str(val).strip()
        codeset_duplicate_keys = codeset_duplicate_keys + 1 if key in titles_by_key else codeset_duplicate_keys
        titles_by_key[key] = {
            "master_label_en": (str(r[master_idx]).strip() if r[master_idx] not in (None, "") else None),
            "titles": titles,
        }

    # Build codes-by-key from MAPPINGS, with leading-zero-safe conversion
    mkey_idx = mappings_header.index(KEY_COL)
    m4_idx = mappings_header.index(ISCO4_COL)
    m3_idx = mappings_header.index(ISCO3_COL)
    m2_idx = mappings_header.index(ISCO2_COL)
    m1_idx = mappings_header.index(ISCO1_COL)
    mlv_idx = mappings_header.index(ISCO08LV_COL)
    mnace2_idx = mappings_header.index(NACE2_0_COL)

    codes_by_key: dict[int, dict] = {}
    quarantined = []
    mappings_duplicate_keys = 0
    for r in mappings_rows:
        key = r[mkey_idx]
        if key in (None, ""):
            continue
        mappings_duplicate_keys = mappings_duplicate_keys + 1 if key in codes_by_key else mappings_duplicate_keys
        isco4 = _zfill_code(r[m4_idx], 4)
        isco3 = _zfill_code(r[m3_idx], 3)
        isco2 = _zfill_code(r[m2_idx], 2)
        isco1 = _zfill_code(r[m1_idx], 1)

        valid = (
            isco4 is not None and len(isco4) == 4 and isco4.isdigit()
            and isco3 is not None and len(isco3) == 3 and isco3.isdigit()
            and isco2 is not None and len(isco2) == 2 and isco2.isdigit()
            and isco1 is not None and len(isco1) == 1 and isco1.isdigit()
            # Prefix-consistency, adapted from the "5-to-4-digit derivation" check:
            # there is no 5-digit ISCO source column in this workbook (see
            # wisco_structure_inspection.md), so this instead verifies the
            # 4-digit code's own leading digits match its declared parents.
            and isco4.startswith(isco3) and isco4.startswith(isco2) and isco4.startswith(isco1)
        )
        if not valid:
            quarantined.append({"key": key, "isco4_raw": r[m4_idx], "isco3_raw": r[m3_idx],
                                 "isco2_raw": r[m2_idx], "isco1_raw": r[m1_idx]})
            continue

        skill_level = r[mlv_idx] if r[mlv_idx] not in (None, "") else None
        nace2_0 = _zfill_code(r[mnace2_idx], 4) if r[mnace2_idx] not in (None, "") else None

        codes_by_key[key] = {
            "isco4": isco4, "isco3": isco3, "isco2": isco2, "isco1": isco1,
            "isco08_skill_level": skill_level,
            "nace2_0": nace2_0,
        }

    # Join
    records = []
    orphans_no_code = []
    for key, t in titles_by_key.items():
        codes = codes_by_key.get(key)
        if codes is None:
            orphans_no_code.append(key)
            continue
        if not t["titles"]:
            continue
        records.append({
            "key": key,
            "isco08_unit": codes["isco4"],
            "isco08_minor": codes["isco3"],
            "isco08_submajor": codes["isco2"],
            "isco08_major": codes["isco1"],
            "isco08_skill_level": codes["isco08_skill_level"],
            "master_label_en": t["master_label_en"],
            "titles": t["titles"],
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # ---- Secondary output: industry crosswalk, both sources kept separate ----
    ws3 = wb["OCC>>INDUSTRY"]
    industry_header, industry_rows = _sheet_to_rows(ws3)
    ikey_idx = industry_header.index("ISCO0813")
    nace_cols = [h for h in industry_header if h and h.startswith("NACE2004")]
    nace_col_idx = {h: industry_header.index(h) for h in nace_cols}
    crosstab_by_key: dict[int, list] = {}
    for r in industry_rows:
        key = r[ikey_idx]
        if key in (None, ""):
            continue
        vals = [r[nace_col_idx[h]] for h in nace_cols]
        real_vals = sorted(set(str(int(v)) for v in vals if v not in (None, "", INDUSTRY_SENTINEL_NONE)))
        if real_vals:
            crosstab_by_key[key] = real_vals

    industry_records = []
    for key, codes in codes_by_key.items():
        nace2004 = crosstab_by_key.get(key, [])
        if codes["nace2_0"] is None and not nace2004:
            continue
        industry_records.append({
            "key": key,
            "isco08_unit": codes["isco4"],
            "nace2_0_rev2": codes["nace2_0"],           # sparse (~17%), single value, ISIC Rev.4-comparable at class level
            "nace2004_rev1_1": nace2004,                 # comprehensive (100% row coverage), list, one classification generation behind
        })

    INDUSTRY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDUSTRY_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(industry_records, f, ensure_ascii=False, indent=2)

    # ---- Summary printed to stdout (ASCII-only for Windows console safety) ----
    per_lang_counts = {lang: sum(1 for rec in records if lang in rec["titles"]) for lang in TARGET_LOCALE_COL}
    n_with_nace2_0 = sum(1 for r in industry_records if r["nace2_0_rev2"] is not None)
    n_with_nace2004 = sum(1 for r in industry_records if r["nace2004_rev1_1"])
    summary = {
        "total_codeset_rows": len(titles_by_key),
        "codeset_excluded_category_rows": excluded_category_rows,
        "codeset_recovered_float_precision_keys": recovered_float_keys,
        "codeset_duplicate_keys_last_wins": codeset_duplicate_keys,
        "total_mappings_rows": len(mappings_rows),
        "mappings_duplicate_keys_last_wins": mappings_duplicate_keys,
        "records_written": len(records),
        "quarantined_bad_codes": len(quarantined),
        "orphans_title_without_code": len(orphans_no_code),
        "per_language_title_counts": per_lang_counts,
        "industry_records_written": len(industry_records),
        "industry_records_with_nace2_0": n_with_nace2_0,
        "industry_records_with_nace2004_crosstab": n_with_nace2004,
        "output_path": str(OUT_PATH),
        "industry_output_path": str(INDUSTRY_OUT_PATH),
    }
    with open(OUT_PATH.parent / "wisco_parse_summary.json", "w", encoding="utf-8") as f:
        json.dump({**summary, "quarantined_examples": quarantined[:10],
                    "orphan_examples": [str(x) for x in orphans_no_code[:10]]}, f, ensure_ascii=False, indent=2, default=str)

    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
