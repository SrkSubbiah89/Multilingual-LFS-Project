"""
eval/parse_official_isic_iscedf_definitions.py

One-off, reproducible extraction of real official per-code definitions and
examples from the primary-source ISIC Rev.4 and ISCED-F 2013 publications,
mirroring the role `eval/normalize_ilo_isco08_catalogue.py` plays for
ISCO-08's own `ISCO-08_EN_Structure_and_definitions.xlsx`.

Sources (downloaded 2026-08-25, real published UN/UNESCO documents):
    eval/local_catalogues/isic_rev4_2008/ISIC_Rev_4_publication_English.pdf
        UN Statistics Division, "International Standard Industrial
        Classification of All Economic Activities (ISIC), Revision 4" --
        https://unstats.un.org/unsd/classifications/Econ/Download/In%20Text/
        ISIC_Rev_4_publication_English.pdf
    eval/local_catalogues/iscedf_2013/ISCEDF_2013_detailed_field_descriptions_2015.pdf
        UNESCO Institute for Statistics, "International Standard
        Classification of Education: Fields of education and training 2013
        (ISCED-F 2013) -- Detailed field descriptions" (2015) --
        https://www.uis.unesco.org/sites/default/files/medias/fichiers/2025/
        04/international-standard-classification-of-education-fields-of-
        education-and-training-2013-detailed-field-descriptions-2015-en.pdf

Output (git-ignored, same as the two source PDFs above -- eval/local_catalogues/
is excluded wholesale in .gitignore; consumed by
backend/rag/official_source_enrichment.py, which raises a clear
FileNotFoundError with regeneration instructions if these are absent):
    eval/local_catalogues/isic_rev4_2008/isic_rev4_definitions.json
    eval/local_catalogues/iscedf_2013/iscedf_2013_definitions.json

Both PDFs' "Detailed structure and explanatory notes" sections use a real
narrative format per code (a definition paragraph, then "This class
includes:" / "Programmes and qualifications with the following main
content are classified here:" followed by real example activities/subject
names) -- genuinely richer than backend/agents/isic_classifier.py's
_ISIC_DATA / backend/agents/isced_classifier.py's _ISCED_FIELDS own
hand-built "keywords" bag-of-terms field, the same kind of gap ISCO-08's
own catalogue had before its 2026-08-24 enrichment fix (see CLAUDE.md).

**A real, disclosed finding from this extraction, not fixed here**: 13 of
134 ISIC codes and 2 of 63 ISCED-F codes already hardcoded in _ISIC_DATA /
_ISCED_FIELDS do not match ANY code in the official structure document --
e.g. _ISIC_DATA's "7311 Advertising agencies" vs the real official "7310
Advertising"; "9001 Performing arts"/"9003 Artistic creation" vs the real
single class "9000 Creative, arts and entertainment activities" (no 9001/
9002/9003 split exists in ISIC Rev.4 at all). This is the same class of
bug Task 20/21's primary-source audit found and fixed in the ISCO-08
catalogue (19 non-standard codes there) -- but is NOT fixed by this
script, which only extracts official text for codes that already
correctly exist; see official_source_enrichment.py's
NON_STANDARD_ISIC_CODES / NON_STANDARD_ISCEDF_CODES for the full list and
COVERAGE_AUDIT_GUIDE.md / a future dedicated task for resolving them.

Requires `pdfplumber` (already an installed dependency). Re-run with:
    python -m eval.parse_official_isic_iscedf_definitions
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pdfplumber

_BASE = Path(__file__).resolve().parent / "local_catalogues"
_ISIC_PDF = _BASE / "isic_rev4_2008" / "ISIC_Rev_4_publication_English.pdf"
_ISIC_OUT = _BASE / "isic_rev4_2008" / "isic_rev4_definitions.json"
_ISCEDF_PDF = _BASE / "iscedf_2013" / "ISCEDF_2013_detailed_field_descriptions_2015.pdf"
_ISCEDF_OUT = _BASE / "iscedf_2013" / "iscedf_2013_definitions.json"

_HYPHEN_STOP_NEXT = {"or", "and", "the", "a", "an", "to", "of", "in"}


def _fix_hyphenation(text: str) -> str:
    """Repairs the PDF line-wrap artifact "supervi- sion" -> "supervision"
    without merging genuine abbreviated forms like "short- or long-term"
    (the word after the space is a common conjunction/article there, which
    a genuine mid-word break never is)."""
    def _repl(m: re.Match) -> str:
        if m.group(2).lower() in _HYPHEN_STOP_NEXT:
            return m.group(0)
        return m.group(1) + m.group(2)
    return re.sub(r"([a-z])- (\w+)", _repl, text)


def _extract_pdf_text(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


# ---------------------------------------------------------------------------
# ISIC Rev.4
# ---------------------------------------------------------------------------

_ISIC_CODE_RE = re.compile(r"^(\d{4})\s+(.+?)\s*$")
_ISIC_FOOTER_RE = re.compile(
    r"^\d{1,3}\s+International Standard Industrial Classification of All Economic Activities \(ISIC\), Revision 4\s*$"
    r"|^Detailed structure and explanatory notes\s+\d{1,3}\s*$"
)
# Verified boundaries (0-indexed, against text.split("\n")): "Part Three:
# Detailed structure and explanatory notes" (the real per-code narrative
# section) begins at line 3024 ("Section A"); the summary structure TABLE
# that precedes it (Division/Group/Class/Description columns, one pass
# through every section A-U) runs up through ~line 3017 and, critically,
# contains bare "NNNN Title" lines of its own (its last rows, e.g. "9602
# Hairdressing and other beauty treatment") that match the same anchor
# regex with no real body text -- an earlier boundary (3000) let a few of
# these leak in as false, empty-bodied anchors for codes whose real
# narrative entry appears ~200 lines later, silently overwriting nothing
# (setdefault) but only because the FALSE one was encountered first,
# which is backwards -- confirmed by a direct diff against a manual
# extraction run once this bug was found. class 9900 (the last real
# class) ends just before "Part Four: Alternative aggregations" begins,
# around line 11895 (unaffected by this specific fix).
_ISIC_SECTION_START = 3024
_ISIC_SECTION_END = 11895


def parse_isic() -> dict[str, dict]:
    text = _extract_pdf_text(_ISIC_PDF)
    lines = text.split("\n")[_ISIC_SECTION_START:_ISIC_SECTION_END]

    anchors = []
    for i, line in enumerate(lines):
        m = _ISIC_CODE_RE.match(line)
        if m and not _ISIC_FOOTER_RE.match(line):
            anchors.append((i, m.group(1), m.group(2)))

    entries: dict[str, dict] = {}
    for idx, (line_i, code, title) in enumerate(anchors):
        end_i = anchors[idx + 1][0] if idx + 1 < len(anchors) else len(lines)
        body_lines = [l for l in lines[line_i + 1:end_i] if not _ISIC_FOOTER_RE.match(l)]
        body = "\n".join(body_lines)

        first_includes = re.search(r"This class (?:also )?includes:", body)
        definition = body[:first_includes.start()].strip() if first_includes else body.strip()
        definition = re.sub(r"\s*\n\s*", " ", definition).strip()

        inc_block = ""
        for m in re.finditer(
            r"This class (?:also )?includes:\s*\n(.*?)(?=This class (?:also )?includes:|This class excludes:|$)",
            body, re.S,
        ):
            inc_block += m.group(1) + "\n"

        examples: list[str] = []
        if inc_block:
            current: list[str] = []
            for l in inc_block.split("\n"):
                l = l.strip()
                if not l:
                    continue
                if l.startswith("—"):  # em-dash bullet marker
                    if current:
                        examples.append(" ".join(current).strip())
                    current = [l.lstrip("—").strip()]
                else:
                    current.append(l)
            if current:
                examples.append(" ".join(current).strip())

        cleaned_examples = []
        for ex in examples:
            ex = ex.replace("\x99", ",")
            ex = re.sub(r":\s*,", ":", ex)
            ex = re.sub(r",\s*,", ",", ex)
            ex = re.sub(r"\s*,\s*", ", ", ex).strip().strip(",").strip()
            ex = re.sub(r"\s{2,}", " ", ex)
            if ex:
                cleaned_examples.append(_fix_hyphenation(ex))

        # Prefer the first NON-EMPTY match -- same rationale as parse_iscedf()'s
        # identical guard below (a cross-reference "see NNNN" sentence can
        # itself start a PDF-wrapped line and look like an empty anchor).
        existing = entries.get(code)
        has_content = bool(definition) or bool(cleaned_examples)
        if existing is None or (not (existing["definition"] or existing["examples"]) and has_content):
            entries[code] = {
                "title": title,
                "definition": _fix_hyphenation(definition),
                "examples": cleaned_examples,
            }

    return entries


# ---------------------------------------------------------------------------
# ISCED-F 2013
# ---------------------------------------------------------------------------

_ISCEDF_CODE_RE = re.compile(r"^(\d{4})\s+(.+?)\s*$")
_ISCEDF_FOOTER_RE = re.compile(r"^-\s*\d{1,3}\s*-\s*$")
_ISCEDF_CLASSIFIED_HERE_RE = re.compile(r"the following main[a-z ]*content are classified here:\s*")
_ISCEDF_NOT_FITTING_RE = re.compile(r"not fitting in the detailed fields? are classified here:\s*")


def parse_iscedf() -> dict[str, dict]:
    text = _extract_pdf_text(_ISCEDF_PDF)
    all_lines = text.split("\n")
    # Real descriptions run through just before "Appendix I: ISCED-F 2013:
    # List of possible codes" -- after that point the document is a
    # two-column code-list appendix and an alphabetical subject index,
    # both of which produce spurious/jumbled 4-digit-code-shaped matches
    # (pdfplumber merges adjacent table columns onto one text line there).
    appendix_start = next(i for i, l in enumerate(all_lines) if l.strip() == "Appendix I")
    lines = all_lines[:appendix_start]

    anchors = []
    for i, line in enumerate(lines):
        m = _ISCEDF_CODE_RE.match(line)
        if m and not _ISCEDF_FOOTER_RE.match(line):
            anchors.append((i, m.group(1), m.group(2)))

    entries: dict[str, dict] = {}
    for idx, (line_i, code, title) in enumerate(anchors):
        end_i = anchors[idx + 1][0] if idx + 1 < len(anchors) else len(lines)
        body_lines = [l for l in lines[line_i + 1:end_i] if not _ISCEDF_FOOTER_RE.match(l)]
        body = "\n".join(body_lines)

        for stop_marker in ("\nInclusions\n", "\nExclusions\n"):
            cut = body.find(stop_marker)
            if cut != -1:
                body = body[:cut]

        m = _ISCEDF_CLASSIFIED_HERE_RE.search(body) or _ISCEDF_NOT_FITTING_RE.search(body)
        if m:
            definition = body[:m.start()].strip()
            examples_block = body[m.end():].strip()
        else:
            definition = body.strip()
            examples_block = ""

        definition = re.sub(r"\s*\n\s*", " ", definition).strip()
        examples = [l.strip() for l in examples_block.split("\n") if l.strip()]

        # Prefer the first NON-EMPTY match -- a handful of cross-reference
        # sentences elsewhere in the body happen to start a PDF-wrapped
        # line with "NNNN <quoted title fragment>.", which looks like a
        # malformed anchor with no real definition/examples of its own.
        # Naively taking literally the first match (setdefault) locks in
        # an empty entry when that false anchor happens to appear BEFORE
        # the real one (confirmed for codes 0512/1013 -- their real entry
        # is the SECOND occurrence, the first being exactly this kind of
        # quoted cross-reference) -- so only skip a code once it already
        # has real content.
        existing = entries.get(code)
        has_content = bool(definition) or bool(examples)
        if existing is None or (not (existing["definition"] or existing["examples"]) and has_content):
            entries[code] = {
                "title": title,
                "definition": _fix_hyphenation(definition),
                "examples": [_fix_hyphenation(e) for e in examples],
            }

    return entries


def main() -> None:
    isic_entries = parse_isic()
    _ISIC_OUT.write_text(json.dumps(isic_entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ISIC Rev.4: {len(isic_entries)} entries -> {_ISIC_OUT}")

    iscedf_entries = parse_iscedf()
    _ISCEDF_OUT.write_text(json.dumps(iscedf_entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ISCED-F 2013: {len(iscedf_entries)} entries -> {_ISCEDF_OUT}")


if __name__ == "__main__":
    main()
