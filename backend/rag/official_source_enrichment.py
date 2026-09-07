"""
backend/rag/official_source_enrichment.py

Real per-code definitions/examples for ISIC Rev.4 and ISCED-F 2013, parsed
from the official primary-source publications (UN Statistics Division /
UNESCO UIS) by eval/parse_official_isic_iscedf_definitions.py -- the
ISIC/ISCED-F counterpart to how backend/rag/official_isco08_catalogue.py's
ENRICHED_PROFILE enriches ISCO-08's own catalogue text from the official
ILO ISCO-08 workbook.

**Why this exists**: backend/agents/isic_classifier.py's _ISIC_DATA and
backend/agents/isced_classifier.py's _ISCED_FIELDS carry only a hand-built
"keywords" bag-of-terms field (mean 11.4 / 9.7 words) -- real, but
nowhere near as rich as ISCO-08's own post-enrichment text (real official
definitions + example activity/subject lists, sourced directly from the
primary publication). This module closes that specific gap for the codes
where a real official match exists.

**A real, disclosed limitation this module does NOT resolve** -- found
while building it, not assumed: 13 of ISIC's 134 already-embedded class
codes, and 2 of ISCED-F's 63 already-embedded detailed-field codes, do not
match ANY code in the official structure document at all. This is the
same class of bug Task 20/21's primary-source audit found and fixed in the
ISCO-08 catalogue (19 non-standard codes there) -- but is genuinely
different work (determining what the correct replacement code should be
for each, and whether downstream consistency needs updating) and is
explicitly NOT done here. NON_STANDARD_ISIC_CODES / NON_STANDARD_ISCEDF_CODES
below name them precisely so nothing downstream can silently assume full
coverage. Entries for these codes fall back to the existing title+keywords
text -- never fabricated, never silently dropped.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent.parent / "eval" / "local_catalogues"
_ISIC_DEFINITIONS_PATH = _BASE / "isic_rev4_2008" / "isic_rev4_definitions.json"
_ISCEDF_DEFINITIONS_PATH = _BASE / "iscedf_2013" / "iscedf_2013_definitions.json"

# Real, verified findings (see this module's docstring and
# eval/parse_official_isic_iscedf_definitions.py's own docstring for the
# full audit) -- codes in _ISIC_DATA / _ISCED_FIELDS with NO match in the
# official structure document. Not fixed here; disclosed so a caller (or a
# future coverage-audit task) never has to rediscover this.
NON_STANDARD_ISIC_CODES = frozenset({
    "7311", "7430", "7739", "8424", "8531", "8559", "8560",
    "8621", "8622", "8623", "8899", "9001", "9003",
})
NON_STANDARD_ISCEDF_CODES = frozenset({"0224", "0919"})


def _read_definitions_json(path: Path, standard_name: str) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{standard_name} definitions file not found at {path}. This file is "
            f"git-ignored (see eval/local_catalogues/ in .gitignore -- same policy "
            f"as ISCO-08's own official workbook) and must be regenerated locally: "
            f"run `python -m eval.parse_official_isic_iscedf_definitions` after "
            f"downloading the two source PDFs named in that script's own docstring "
            f"(see also eval/verified_catalogue_counts.yaml for the exact URLs and "
            f"expected file hashes). Only reached when a caller explicitly requests "
            f"the 'enriched_e5large' profile -- default classify() calls never hit "
            f"this path."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_isic_definitions() -> dict[str, dict]:
    """{class_code: {"title": str, "definition": str, "examples": list[str]}}
    for every ISIC Rev.4 code the official structure document actually
    defines (419 entries -- the full standard, not just the 134 already
    embedded in _ISIC_DATA)."""
    return _read_definitions_json(_ISIC_DEFINITIONS_PATH, "ISIC Rev.4")


@lru_cache(maxsize=1)
def load_iscedf_definitions() -> dict[str, dict]:
    """Same shape as load_isic_definitions(), for ISCED-F 2013 detailed
    fields (92 entries parsed; the full standard's ~80 detailed fields are
    covered, some duplicated code-shaped index/appendix noise was excluded
    -- see the parser's own docstring)."""
    return _read_definitions_json(_ISCEDF_DEFINITIONS_PATH, "ISCED-F 2013")


def build_enriched_text(code: str, title: str, keywords: str, definitions: dict[str, dict]) -> str:
    """Same embedding-text shape as official_isco08_catalogue.py's
    ENRICHED_PROFILE: "{code} {title}. {definition} Examples: {examples}."
    when a real official match exists; falls back to the existing
    "{code} {title} {keywords}" shape (today's behaviour, unchanged) when
    it doesn't -- e.g. every NON_STANDARD_*_CODES entry, and any future
    catalogue addition this module hasn't been re-run against. Never
    fabricates a definition or example that isn't in the official source."""
    entry = definitions.get(code)
    if entry is None or not (entry.get("definition") or entry.get("examples")):
        return f"{code} {title} {keywords}".strip()

    parts = [f"{code} {title}."]
    if entry.get("definition"):
        parts.append(entry["definition"])
    if entry.get("examples"):
        parts.append("Examples: " + "; ".join(entry["examples"]) + ".")
    return " ".join(parts)
