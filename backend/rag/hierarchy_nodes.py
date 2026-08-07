"""
backend/rag/hierarchy_nodes.py

Conference I Reviewer #2 response, Task 05 ("Implement Genuine Hierarchical
Retrieval for ISIC and ISCED-F"). Deterministic derivation of Qdrant-
indexable hierarchy nodes from the existing embedded classification
tables:

    backend.agents.isic_classifier._ISIC_DATA     -- ISIC Rev.4
    backend.agents.isced_classifier._ISCED_FIELDS  -- ISCED-F 2013

**This module adds no new classification data.** Every node's code, title,
and index text is derived straight from those two existing, bounded,
already-embedded tables (the same tables the keyword classifiers already
use). It is an INDEXING REPRESENTATION of what already exists in this
codebase -- never a downloaded/scraped official catalogue, and never a
claim of complete official ISIC/ISCED-F coverage. See
Documentation/Conference_I_Reviewer_2/ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md
for the honest coverage statement.

Fails closed (raises ``HierarchyValidationError``) on:
  - a duplicate code with a conflicting parent or title
  - a malformed code (wrong length/pattern for its level)
  - a non-root node whose parent code does not exist
  - an empty/impossible derived level

No Qdrant, embedding model, or network dependency -- pure, deterministic,
same input always produces the same output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.agents.isced_classifier import _ISCED_FIELDS
from backend.agents.isic_classifier import _ISIC_DATA


class HierarchyValidationError(Exception):
    """Raised by validate_isic_nodes()/validate_iscedf_nodes() -- fail
    closed, never silently accept a malformed or internally-inconsistent
    derived hierarchy."""


@dataclass
class HierarchyNode:
    code: str
    parent_code: str  # "" for root-level nodes (ISIC sections, ISCED-F broad fields)
    label_en: str
    label_ar: str      # "" when the source table has no separate Arabic title
    index_text: str    # what actually gets embedded: label + aggregated descendant keywords
    level: str          # e.g. "section" | "division" | "group" | "class"


# ---------------------------------------------------------------------------
# ISIC Rev.4: section -> division -> group -> class
# ---------------------------------------------------------------------------

ISIC_LEVELS = ("sections", "divisions", "groups", "classes")

ISIC_CODE_PATTERNS = {
    "sections": re.compile(r"^[A-Z]$"),
    "divisions": re.compile(r"^\d{2}$"),
    "groups": re.compile(r"^\d{3}$"),
    "classes": re.compile(r"^\d{4}$"),
}


def derive_isic_nodes() -> dict[str, list[HierarchyNode]]:
    """Returns {"sections": [...], "divisions": [...], "groups": [...],
    "classes": [...]}, derived from _ISIC_DATA. Deterministic: the same
    _ISIC_DATA always produces the same output, no randomness, no I/O.
    Validates the result before returning (raises HierarchyValidationError
    on any inconsistency -- see module docstring)."""
    sections: dict[str, HierarchyNode] = {}
    divisions: dict[str, HierarchyNode] = {}
    groups: dict[str, HierarchyNode] = {}
    classes: dict[str, HierarchyNode] = {}

    # Aggregate descendant keywords bottom-up so an internal (non-leaf)
    # node's index_text isn't just its own short official label -- it also
    # carries the keyword vocabulary of everything beneath it, matching how
    # ISCO's own major/submajor/minor collections were built.
    section_kw: dict[str, list[str]] = {}
    division_kw: dict[str, list[str]] = {}
    group_kw: dict[str, list[str]] = {}
    for row in _ISIC_DATA:
        section_kw.setdefault(row["section"], []).append(row.get("keywords", ""))
        division_kw.setdefault(row["division_code"], []).append(row.get("keywords", ""))
        group_kw.setdefault(row["group_code"], []).append(row.get("keywords", ""))

    for row in _ISIC_DATA:
        sec = row["section"]
        if sec not in sections:
            sections[sec] = HierarchyNode(
                code=sec, parent_code="", label_en=row["section_title"], label_ar="",
                index_text=f"{row['section_title']} {' '.join(section_kw[sec])}".strip(),
                level="section",
            )
        elif sections[sec].label_en != row["section_title"]:
            raise HierarchyValidationError(
                f"ISIC section {sec!r}: conflicting titles {sections[sec].label_en!r} vs {row['section_title']!r}"
            )

        div = row["division_code"]
        if div not in divisions:
            divisions[div] = HierarchyNode(
                code=div, parent_code=sec, label_en=row["division_title"], label_ar="",
                index_text=f"{row['division_title']} {' '.join(division_kw[div])}".strip(),
                level="division",
            )
        elif divisions[div].parent_code != sec or divisions[div].label_en != row["division_title"]:
            raise HierarchyValidationError(f"ISIC division {div!r}: conflicting parent/title")

        grp = row["group_code"]
        if grp not in groups:
            groups[grp] = HierarchyNode(
                code=grp, parent_code=div, label_en=row["group_title"], label_ar="",
                index_text=f"{row['group_title']} {' '.join(group_kw[grp])}".strip(),
                level="group",
            )
        elif groups[grp].parent_code != div or groups[grp].label_en != row["group_title"]:
            raise HierarchyValidationError(f"ISIC group {grp!r}: conflicting parent/title")

        cls = row["class_code"]
        if cls not in classes:
            classes[cls] = HierarchyNode(
                code=cls, parent_code=grp, label_en=row["class_title"], label_ar="",
                index_text=f"{row['class_title']} {row.get('keywords', '')}".strip(),
                level="class",
            )
        elif classes[cls].parent_code != grp or classes[cls].label_en != row["class_title"]:
            raise HierarchyValidationError(f"ISIC class {cls!r}: conflicting parent/title")

    result = {
        "sections": list(sections.values()),
        "divisions": list(divisions.values()),
        "groups": list(groups.values()),
        "classes": list(classes.values()),
    }
    validate_isic_nodes(result)
    return result


def validate_isic_nodes(nodes: dict[str, list[HierarchyNode]]) -> None:
    _validate_common(nodes, ISIC_LEVELS, ISIC_CODE_PATTERNS, standard="ISIC")


# ---------------------------------------------------------------------------
# ISCED-F 2013: broad field -> narrow field -> detailed field
# ---------------------------------------------------------------------------

ISCEDF_LEVELS = ("broad_fields", "narrow_fields", "detailed_fields")

ISCEDF_CODE_PATTERNS = {
    "broad_fields": re.compile(r"^\d{2}$"),
    "narrow_fields": re.compile(r"^\d{3}$"),
    "detailed_fields": re.compile(r"^\d{4}$"),
}


def derive_iscedf_nodes() -> dict[str, list[HierarchyNode]]:
    """Returns {"broad_fields": [...], "narrow_fields": [...],
    "detailed_fields": [...]}, derived from _ISCED_FIELDS. Same
    determinism/validation contract as derive_isic_nodes() above. Note:
    ISCED 2011 attainment LEVEL (0-8, from the separate _ISCED_LEVELS
    table) is NOT part of this hierarchy and has no node here -- see
    module docstring and ISCEDClassifier.classify()'s docstring for why
    level and field are kept as two independent dimensions."""
    broad: dict[str, HierarchyNode] = {}
    narrow: dict[str, HierarchyNode] = {}
    detailed: dict[str, HierarchyNode] = {}

    broad_kw: dict[str, list[str]] = {}
    narrow_kw: dict[str, list[str]] = {}
    for row in _ISCED_FIELDS:
        broad_kw.setdefault(row["broad_code"], []).append(row.get("keywords", ""))
        narrow_kw.setdefault(row["narrow_code"], []).append(row.get("keywords", ""))

    for row in _ISCED_FIELDS:
        b = row["broad_code"]
        if b not in broad:
            broad[b] = HierarchyNode(
                code=b, parent_code="", label_en=row["broad_title"], label_ar="",
                index_text=f"{row['broad_title']} {' '.join(broad_kw[b])}".strip(),
                level="broad_field",
            )
        elif broad[b].label_en != row["broad_title"]:
            raise HierarchyValidationError(
                f"ISCED-F broad field {b!r}: conflicting titles {broad[b].label_en!r} vs {row['broad_title']!r}"
            )

        n = row["narrow_code"]
        if n not in narrow:
            narrow[n] = HierarchyNode(
                code=n, parent_code=b, label_en=row["narrow_title"], label_ar="",
                index_text=f"{row['narrow_title']} {' '.join(narrow_kw[n])}".strip(),
                level="narrow_field",
            )
        elif narrow[n].parent_code != b or narrow[n].label_en != row["narrow_title"]:
            raise HierarchyValidationError(f"ISCED-F narrow field {n!r}: conflicting parent/title")

        d = row["detailed_code"]
        if d not in detailed:
            detailed[d] = HierarchyNode(
                code=d, parent_code=n, label_en=row["detailed_title"], label_ar="",
                index_text=f"{row['detailed_title']} {row.get('keywords', '')}".strip(),
                level="detailed_field",
            )
        elif detailed[d].parent_code != n or detailed[d].label_en != row["detailed_title"]:
            raise HierarchyValidationError(f"ISCED-F detailed field {d!r}: conflicting parent/title")

    result = {
        "broad_fields": list(broad.values()),
        "narrow_fields": list(narrow.values()),
        "detailed_fields": list(detailed.values()),
    }
    validate_iscedf_nodes(result)
    return result


def validate_iscedf_nodes(nodes: dict[str, list[HierarchyNode]]) -> None:
    _validate_common(nodes, ISCEDF_LEVELS, ISCEDF_CODE_PATTERNS, standard="ISCED-F")


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------

def _validate_common(
    nodes: dict[str, list[HierarchyNode]],
    level_order: tuple[str, ...],
    patterns: dict[str, "re.Pattern"],
    standard: str,
) -> None:
    """Fail closed on: empty level, malformed code, duplicate code, or a
    non-root node whose parent_code does not exist in the immediately
    preceding level. level_order is authoritative for which level is each
    other level's parent -- never inferred from dict iteration order."""
    codes_by_level: dict[str, dict[str, HierarchyNode]] = {}

    for level in level_order:
        node_list = nodes.get(level, [])
        if not node_list:
            raise HierarchyValidationError(
                f"{standard}: level {level!r} has zero derived nodes -- empty/impossible hierarchy"
            )
        pattern = patterns[level]
        seen: dict[str, HierarchyNode] = {}
        for n in node_list:
            if not pattern.match(n.code):
                raise HierarchyValidationError(
                    f"{standard}: malformed code {n.code!r} for level {level!r} "
                    f"(expected pattern {pattern.pattern!r})"
                )
            if n.code in seen:
                raise HierarchyValidationError(
                    f"{standard}: duplicate code {n.code!r} within level {level!r}"
                )
            seen[n.code] = n
        codes_by_level[level] = seen

    for i, level in enumerate(level_order):
        if i == 0:
            continue  # root level -- no parent to check
        parent_level = level_order[i - 1]
        parent_codes = codes_by_level[parent_level]
        for n in nodes[level]:
            if n.parent_code not in parent_codes:
                raise HierarchyValidationError(
                    f"{standard}: {level} node {n.code!r} has parent_code={n.parent_code!r}, "
                    f"which does not exist in level {parent_level!r}"
                )
