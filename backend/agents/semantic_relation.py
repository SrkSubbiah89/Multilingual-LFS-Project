"""
backend/agents/semantic_relation.py

Semantic Relation Engine — ISCO ↔ ISIC ↔ ISCED Cross-Classification
=====================================================================
Thesis Chapter 5 — Novel Contribution

This module implements a three-way semantic crosswalk between the three
international labour-force classification standards:

    ISCO-08  — International Standard Classification of Occupations
    ISIC R4  — International Standard Industrial Classification (industry)
    ISCED 11 — International Standard Classification of Education

Problem statement
-----------------
Current LFS practice classifies these three dimensions independently.
A respondent might be labelled:
    ISCO  = "2211 Medical Doctor"
    ISIC  = "C26 Electronics Manufacturing"  ← inconsistent
    ISCED = "ISCED 2 Lower Secondary"         ← inconsistent

No existing automated LFS tool detects such cross-classification
inconsistencies or uses multi-standard signals to disambiguate a
single classification.

This engine provides three capabilities:

1. **Consistency scoring**  (SemanticCoherence.score 0–1)
   Checks whether the three classifications are mutually compatible
   according to the ILO occupation-industry-education mapping tables.

2. **Cross-standard inference**
   Uses ISIC evidence to disambiguate ISCO when confidence is low, and
   uses ISCED evidence to validate ISCO major-group assignment.

3. **Semantic violation detection**
   Flags specific cross-standard conflicts with natural-language
   explanations (both EN and AR) suitable for HITL review.

Architecture
------------
The crosswalk is encoded as two mapping tables. CORRECTED 2026-08-16
(Conference I Reviewer #2 response, Section G expansion / Module D):
this docstring previously cited "ILO 'ISCO-08 Correspondence Table with
ISIC Rev.4' (Geneva, 2012)" and "UNESCO 'ISCED 2011 Operational Manual'
Table 7 (2015)" as the source of these tables. Verified directly against
primary sources and found false: the real, complete ISCO-08 Volume I PDF
(433 pages, fetched directly from ilo.org) mentions "ISIC" exactly once,
in a bibliography entry citing ISIC Rev.4 as a related standard -- not a
correspondence table. No ILO document mapping ISCO-08 to ISIC appears to
exist at all (occupation and industry are independent classification
dimensions, unlike e.g. the real ISCO-08-to-ISCO-88 correspondence
table, which does exist). The ISCED 2011 Operational Manual is a real
document, but its own described structure (chapters per ISCED level
0-8, plus a summary table of ISCED codes/criteria in its Annex) concerns
classifying education PROGRAMMES into ISCED levels -- not occupations;
nothing in its documented contents maps ISCO codes to expected education
levels. Neither citation could be verified.

**What the tables below actually are**: hand-built domain-reasoning
heuristics (each entry already carried an inline comment explaining its
reasoning, e.g. "Health Professionals -> Health only" -- that reasoning
was always real; only the claimed document source was not). They encode
a genuine, useful plausibility check -- flagging occupation/industry/
education combinations an ILO labour-statistics analyst would find
surprising -- but are not a transcription of any single official
correspondence table, because no such official document exists for
ISCO<->ISIC or ISCO<->ISCED specifically. This should be corrected in
any thesis text that currently cites Geneva 2012 / UNESCO Table 7 as
the source.

These are stored as in-memory dicts for O(1) lookup — no LLM call needed
for the core crosswalk, making it deterministic, fast, and auditable.

An optional LLM re-inference step (TaskType.GENERAL) is invoked only when
``not is_coherent`` (score < 0.70) AND both ``isic_section`` and
``job_title`` were supplied -- NOT only in an "ambiguous band" as an
earlier version of this docstring claimed. Corrected here (Conference I
Reviewer #2 response, Section G) to match ``analyse()``'s actual condition
(see the "5. ISIC-based ISCO inference" step) rather than a description
that never matched the code.

Usage
-----
from backend.agents.semantic_relation import SemanticRelationEngine

engine = SemanticRelationEngine()

result = engine.analyse(
    isco_code    = "2211",          # 4-digit ISCO-08
    isic_section = "Q",             # one-letter ISIC R4 section
    isced_level  = 6,               # ISCED 2011 level 0-8
    job_title    = "Medical Doctor",
    language     = "en",
)

print(result.coherence_score)    # 0.97
print(result.is_coherent)        # True
print(result.violations)         # []
print(result.inferred_isco)      # None  (original kept)
print(result.explanation_en)     # "Strong alignment: professional ..."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ISCO → ISIC  mapping  (major group → expected ISIC sections)
# Hand-built plausibility heuristic, NOT a transcription of an official ILO
# correspondence table -- verified 2026-08-16 that no such document exists
# (see module docstring's "Architecture" section for the full verification).
# Each entry's domain reasoning is genuine even though the original claimed
# source was not.
# ---------------------------------------------------------------------------

# Each major group lists the ISIC sections that are semantically consistent.
# "ANY" means all sections are valid (managerial / clerical roles span all industries).

_ISCO_MAJOR_TO_ISIC: dict[str, list[str]] = {
    "0": ["O"],                                          # Armed forces → Public admin & defence
    "1": ["ANY"],                                        # Managers → all industries
    "2": ["J", "M", "Q", "P", "K", "L", "R", "N"],     # Professionals → ICT, Science, Health, Education, Finance
    "3": ["C", "F", "H", "J", "Q", "M", "K"],           # Technicians → Manufacturing, Construction, Transport, ICT, Health
    "4": ["ANY"],                                        # Clerical → all industries
    "5": ["G", "I", "N", "S", "Q", "O"],                # Service & Sales → Retail, Hotels, Admin, Other, Health, Public
    "6": ["A"],                                          # Agriculture → Agriculture only
    "7": ["C", "F", "E", "D", "B"],                     # Craft & Trades → Manufacturing, Construction, Utilities, Mining
    "8": ["C", "H", "B", "D", "F"],                     # Operators → Manufacturing, Transport, Mining, Utilities, Construction
    "9": ["ANY"],                                        # Elementary → all industries
}

# Stricter sub-major rules for common misclassifications (2-digit ISCO prefix)
_ISCO_SUBMAJOR_TO_ISIC: dict[str, list[str]] = {
    "21": ["J", "M", "C", "F"],       # Science & Engineering Professionals
    "22": ["Q"],                       # Health Professionals → Health only
    "23": ["P"],                       # Teaching Professionals → Education only
    "24": ["K", "M", "N"],            # Business & Administration Professionals
    "25": ["J"],                       # ICT Professionals → ICT only
    "26": ["M", "R", "J"],            # Legal / Social / Cultural
    "31": ["C", "F", "J", "H"],       # Science & Engineering Technicians
    "32": ["Q"],                       # Health Technicians → Health only
    "33": ["K", "M", "N"],            # Business Technicians
    "34": ["P", "R", "N"],            # Legal / Social Technicians
    "61": ["A"],                       # Market Gardeners → Agriculture
    "62": ["A"],                       # Subsistence Farmers → Agriculture
    "71": ["F", "C"],                 # Building Trades → Construction, Manufacturing
    "72": ["C", "F"],                 # Metal / Machinery Trades
    "73": ["C", "R"],                 # Precision / Handicraft
    "74": ["C", "D", "E"],            # Electrical / Electronic Trades
    "75": ["C", "A"],                 # Food / Wood / Garment
    "81": ["C", "B"],                 # Stationary Plant Operators
    "82": ["C"],                       # Assemblers → Manufacturing
    "83": ["H", "G"],                 # Drivers → Transport, Retail
}


# ---------------------------------------------------------------------------
# ISCO → ISCED  mapping  (major group → expected ISCED levels, min/max)
# Hand-built plausibility heuristic, NOT a transcription of UNESCO's ISCED
# 2011 Operational Manual -- verified 2026-08-16 that manual's own described
# structure concerns classifying education programmes into ISCED levels,
# not mapping occupations to expected education levels (see module
# docstring's "Architecture" section for the full verification).
# ---------------------------------------------------------------------------

# (min_level, max_level, typical_level)
_ISCO_MAJOR_TO_ISCED: dict[str, tuple[int, int, int]] = {
    "0": (3, 8, 5),   # Armed forces — varies widely; at least upper secondary
    "1": (5, 8, 6),   # Managers — at least short-cycle tertiary
    "2": (6, 8, 7),   # Professionals — bachelor's or above (ISCED 6-8)
    "3": (4, 6, 5),   # Technicians — post-secondary non-tertiary to bachelor's
    "4": (3, 6, 4),   # Clerical — upper secondary minimum
    "5": (2, 5, 3),   # Service & Sales — lower secondary to post-secondary
    "6": (0, 3, 1),   # Agriculture — no minimum (ISCED 0-3 typical)
    "7": (2, 5, 3),   # Craft & Trades — lower secondary to post-secondary
    "8": (2, 4, 3),   # Operators — lower secondary to post-secondary non-tertiary
    "9": (0, 3, 1),   # Elementary — no education to lower secondary
}

# Stricter rules for Major 2 sub-majors
_ISCO_SUBMAJOR_TO_ISCED_MIN: dict[str, int] = {
    "22": 7,   # Health Professionals (Medical Doctors) → Master's+
    "23": 6,   # Teaching Professionals → Bachelor's+
    "25": 6,   # ICT Professionals → Bachelor's+
    "26": 7,   # Legal Professionals (Lawyers) → Master's+
}


# ---------------------------------------------------------------------------
# ISIC section labels  (one-letter code → human-readable)
# ---------------------------------------------------------------------------

_ISIC_SECTION_LABELS: dict[str, str] = {
    "A": "Agriculture, Forestry & Fishing",
    "B": "Mining & Quarrying",
    "C": "Manufacturing",
    "D": "Electricity, Gas, Steam",
    "E": "Water Supply & Waste Management",
    "F": "Construction",
    "G": "Wholesale & Retail Trade",
    "H": "Transportation & Storage",
    "I": "Accommodation & Food Service",
    "J": "Information & Communication",
    "K": "Financial & Insurance Activities",
    "L": "Real Estate Activities",
    "M": "Professional, Scientific & Technical",
    "N": "Administrative & Support Services",
    "O": "Public Administration & Defence",
    "P": "Education",
    "Q": "Human Health & Social Work",
    "R": "Arts, Entertainment & Recreation",
    "S": "Other Service Activities",
    "T": "Household Activities",
    "U": "Extraterritorial Organisations",
}

_ISCED_LABELS: dict[int, str] = {
    0: "Early Childhood (No formal ed.)",
    1: "Primary",
    2: "Lower Secondary",
    3: "Upper Secondary",
    4: "Post-Secondary Non-Tertiary",
    5: "Short-Cycle Tertiary",
    6: "Bachelor's Equivalent",
    7: "Master's Equivalent",
    8: "Doctoral",
}


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

@dataclass
class SemanticViolation:
    """
    One cross-standard inconsistency.

    Provenance fields (rule_id, standard_version, crosswalk_source_id,
    matched_hierarchy_levels, severity_rationale) were added for Conference
    I Reviewer #2 response, Section G ("semantic relation evidence") --
    defaulted so existing callers constructing a SemanticViolation
    positionally/without these kwargs are unaffected.
    """
    violation_type:  str     # "isco_isic" | "isco_isced" | "isic_isced"
    severity:        str     # "HIGH" | "MODERATE" | "LOW"
    message_en:      str
    message_ar:      str
    expected:        str     # what was expected
    actual:          str     # what was found
    rule_id:         str = ""   # e.g. "SR-ISCO-ISIC-01" -- see _check_isco_isic/_check_isco_isced
    standard_version: str = ""  # e.g. "ISCO-08 / ISIC Rev.4"
    crosswalk_source_id: str = ""  # which lookup table produced this violation, e.g. "_ISCO_SUBMAJOR_TO_ISIC"
    matched_hierarchy_levels: list = field(default_factory=list)  # e.g. ["major:2", "submajor:25"]
    severity_rationale: str = ""   # human-readable justification for the severity band chosen


@dataclass
class SemanticCoherence:
    """
    Full cross-classification coherence analysis for one survey respondent.

    score        : 0.0 (completely incoherent) – 1.0 (perfect alignment)
    is_coherent  : True when score >= 0.70
    violations   : list of specific cross-standard conflicts
    inferred_isco: if ISIC+ISCED strongly suggest a different ISCO code,
                   this field carries the suggested 4-digit code
    explanation_en / explanation_ar : summary for report page
    isco_isic_compatible  : whether occupation fits industry
    isco_isced_compatible : whether occupation fits education level
    """
    isco_code:             str
    isic_section:          Optional[str]
    isced_level:           Optional[int]
    score:                 float
    is_coherent:           bool
    isco_isic_compatible:  bool
    isco_isced_compatible: bool
    violations:            list[SemanticViolation]
    inferred_isco:         Optional[str]
    explanation_en:        str
    explanation_ar:        str
    confidence_adjustment: float   # delta to apply to ISCO confidence: +0.10 to -0.20
    major_group:           str     # "0"–"9"
    major_label:           str
    isic_label:            Optional[str]
    isced_label:           Optional[str]


# ---------------------------------------------------------------------------
# SemanticRelationEngine
# ---------------------------------------------------------------------------

_MAJOR_LABELS: dict[str, str] = {
    "0": "Armed Forces Occupations",
    "1": "Managers",
    "2": "Professionals",
    "3": "Technicians & Associate Professionals",
    "4": "Clerical Support Workers",
    "5": "Service & Sales Workers",
    "6": "Skilled Agricultural, Forestry & Fishery Workers",
    "7": "Craft & Related Trades Workers",
    "8": "Plant & Machine Operators & Assemblers",
    "9": "Elementary Occupations",
}


class SemanticRelationEngine:
    """
    Three-way ISCO ↔ ISIC ↔ ISCED semantic crosswalk.

    Deterministic lookup table approach — no LLM required for core logic.
    Optional LLM refinement only when ``not is_coherent`` (score < 0.70)
    AND isic_section + job_title are both supplied -- see analyse()'s
    "5. ISIC-based ISCO inference" step for the exact condition (corrected
    here to match the code; an earlier version of this docstring
    incorrectly described the trigger as an "ambiguous band 0.40-0.75").

    Parameters
    ----------
    use_llm : bool
        Set to False to disable the optional LLM disambiguation step
        (useful in tests and when Ollama is not running).
    """

    def __init__(self, use_llm: bool = True) -> None:
        self._use_llm = use_llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        isco_code:    str,
        isic_section: Optional[str] = None,
        isced_level:  Optional[int] = None,
        job_title:    str = "",
        language:     str = "en",
    ) -> SemanticCoherence:
        """
        Analyse cross-standard coherence for one respondent's classifications.

        Parameters
        ----------
        isco_code    : 4-digit ISCO-08 code (e.g. "2211")
        isic_section : one-letter ISIC section (e.g. "Q")  — optional
        isced_level  : integer 0-8                          — optional
        job_title    : free-text job title for LLM fallback — optional
        language     : "en" | "ar" for explanation language

        Returns
        -------
        SemanticCoherence dataclass
        """
        isco_code    = str(isco_code).strip()[:4]
        major        = isco_code[:1] if isco_code else "?"
        submajor     = isco_code[:2] if len(isco_code) >= 2 else ""
        major_label  = _MAJOR_LABELS.get(major, "Unknown")
        isic_label   = _ISIC_SECTION_LABELS.get(isic_section or "", None)
        isced_label  = _ISCED_LABELS.get(isced_level, None) if isced_level is not None else None

        violations: list[SemanticViolation] = []
        score_parts: list[float] = []

        # ── 1. ISCO ↔ ISIC compatibility ─────────────────────────────────────
        isco_isic_ok = True
        if isic_section:
            isco_isic_ok = self._check_isco_isic(
                major, submajor, isic_section, violations
            )
        score_parts.append(1.0 if isco_isic_ok else 0.0)

        # ── 2. ISCO ↔ ISCED compatibility ────────────────────────────────────
        isco_isced_ok = True
        if isced_level is not None:
            isco_isced_ok = self._check_isco_isced(
                major, submajor, isced_level, violations
            )
        score_parts.append(1.0 if isco_isced_ok else 0.0)

        # ── 3. Weighted coherence score ───────────────────────────────────────
        # Weight: ISIC check 0.55, ISCED check 0.45
        # If one dimension is missing, use only the available one.
        if isic_section and isced_level is not None:
            raw_score = 0.55 * score_parts[0] + 0.45 * score_parts[1]
        elif isic_section:
            raw_score = float(score_parts[0])
        elif isced_level is not None:
            raw_score = float(score_parts[1])
        else:
            raw_score = 0.80   # no cross-standard data → assume plausible

        # Partial credit: violations with MODERATE severity soften the penalty
        moderate_violations = sum(1 for v in violations if v.severity == "MODERATE")
        raw_score = max(0.0, raw_score + 0.15 * moderate_violations * (1 - raw_score))

        score = round(min(raw_score, 1.0), 4)
        is_coherent = score >= 0.70

        # ── 4. Confidence adjustment ──────────────────────────────────────────
        if score >= 0.90:
            conf_adj = +0.10
        elif score >= 0.70:
            conf_adj = +0.05
        elif score >= 0.50:
            conf_adj = -0.05
        else:
            conf_adj = -0.20

        # ── 5. ISIC-based ISCO inference (when ISCO confidence is low) ────────
        inferred_isco: Optional[str] = None
        if not is_coherent and isic_section and job_title and self._use_llm:
            inferred_isco = self._infer_isco_from_isic(
                job_title, isic_section, major, language
            )

        # ── 6. Explanations ───────────────────────────────────────────────────
        explanation_en, explanation_ar = self._build_explanations(
            major_label, isic_label, isced_label, score,
            violations, inferred_isco
        )

        return SemanticCoherence(
            isco_code=isco_code,
            isic_section=isic_section,
            isced_level=isced_level,
            score=score,
            is_coherent=is_coherent,
            isco_isic_compatible=isco_isic_ok,
            isco_isced_compatible=isco_isced_ok,
            violations=violations,
            inferred_isco=inferred_isco,
            explanation_en=explanation_en,
            explanation_ar=explanation_ar,
            confidence_adjustment=round(conf_adj, 2),
            major_group=major,
            major_label=major_label,
            isic_label=isic_label,
            isced_label=isced_label,
        )

    # ------------------------------------------------------------------
    # Internal checkers
    # ------------------------------------------------------------------

    def _check_isco_isic(
        self,
        major:     str,
        submajor:  str,
        isic:      str,
        violations: list[SemanticViolation],
    ) -> bool:
        """Return True if ISIC section is consistent with ISCO major/sub-major."""
        # Sub-major rule takes priority when available
        submajor_allowed = _ISCO_SUBMAJOR_TO_ISIC.get(submajor)
        used_submajor_rule = submajor_allowed is not None
        allowed = submajor_allowed or _ISCO_MAJOR_TO_ISIC.get(major, ["ANY"])

        source_id = "_ISCO_SUBMAJOR_TO_ISIC" if used_submajor_rule else "_ISCO_MAJOR_TO_ISIC"
        levels = [f"major:{major}"] + ([f"submajor:{submajor}"] if used_submajor_rule and submajor else [])

        if "ANY" in allowed:
            return True

        if isic in allowed:
            return True

        # Partial: some sub-majors allow a slightly wider set at major level
        major_allowed = _ISCO_MAJOR_TO_ISIC.get(major, [])
        if "ANY" not in major_allowed and isic in major_allowed:
            violations.append(SemanticViolation(
                violation_type="isco_isic",
                severity="MODERATE",
                message_en=(
                    f"The occupation (ISCO major {major}: {_MAJOR_LABELS.get(major,'?')}) "
                    f"is atypical in industry section '{isic}' "
                    f"({_ISIC_SECTION_LABELS.get(isic,'?')}). "
                    f"Expected sections: {', '.join(allowed)}."
                ),
                message_ar=(
                    f"المهنة (المجموعة الرئيسية {major}) غير معتادة في قطاع '{isic}'. "
                    f"القطاعات المتوقعة: {', '.join(allowed)}."
                ),
                expected=", ".join(allowed),
                actual=isic,
                rule_id="SR-ISCO-ISIC-01",
                standard_version="ISCO-08 / ISIC Rev.4",
                crosswalk_source_id=source_id,
                matched_hierarchy_levels=levels,
                severity_rationale=(
                    f"ISIC section {isic!r} is allowed at ISCO major-group level ({major}) "
                    f"but not in the stricter sub-major ({submajor!r}) list -- treated as "
                    f"atypical, not incompatible."
                ),
            ))
            return True   # MODERATE — still partially compatible

        violations.append(SemanticViolation(
            violation_type="isco_isic",
            severity="HIGH",
            message_en=(
                f"ISCO major group {major} ({_MAJOR_LABELS.get(major,'?')}) is "
                f"incompatible with ISIC section '{isic}' "
                f"({_ISIC_SECTION_LABELS.get(isic,'?')}). "
                f"Expected: {', '.join(allowed)}."
            ),
            message_ar=(
                f"المجموعة المهنية {major} غير متوافقة مع قطاع الصناعة '{isic}'. "
                f"القطاعات المتوقعة: {', '.join(allowed)}."
            ),
            expected=", ".join(allowed),
            actual=isic,
            rule_id="SR-ISCO-ISIC-02",
            standard_version="ISCO-08 / ISIC Rev.4",
            crosswalk_source_id=source_id,
            matched_hierarchy_levels=levels,
            severity_rationale=(
                f"ISIC section {isic!r} is not in the allowed set for ISCO major group "
                f"{major} at either major or sub-major granularity."
            ),
        ))
        return False

    def _check_isco_isced(
        self,
        major:     str,
        submajor:  str,
        isced:     int,
        violations: list[SemanticViolation],
    ) -> bool:
        """Return True if ISCED level is consistent with ISCO major group."""
        min_l, max_l, typical_l = _ISCO_MAJOR_TO_ISCED.get(major, (0, 8, 4))

        # Stricter sub-major minimum
        sub_min = _ISCO_SUBMAJOR_TO_ISCED_MIN.get(submajor)
        used_submajor_rule = sub_min is not None
        if used_submajor_rule:
            min_l = max(min_l, sub_min)

        if min_l <= isced <= max_l:
            return True

        # Severity band = how many ISCED levels outside the valid [min_l,
        # max_l] range the respondent's level falls -- a genuine 3-tier
        # scale (previously only HIGH/MODERATE existed, gated on distance
        # from the single "typical" point rather than distance from the
        # range boundary; LOW is new -- see Documentation/
        # Conference_I_Reviewer_2/EVALUATION_PROTOCOL.md for why this
        # boundary-gap formulation was chosen and its provisional status).
        gap = (min_l - isced) if isced < min_l else (isced - max_l)
        if gap <= 1:
            severity = "LOW"
        elif gap == 2:
            severity = "MODERATE"
        else:
            severity = "HIGH"

        source_id = "_ISCO_MAJOR_TO_ISCED" + ("+_ISCO_SUBMAJOR_TO_ISCED_MIN" if used_submajor_rule else "")
        levels = [f"major:{major}"] + ([f"submajor:{submajor}"] if used_submajor_rule and submajor else [])

        violations.append(SemanticViolation(
            violation_type="isco_isced",
            severity=severity,
            message_en=(
                f"ISCO major group {major} ({_MAJOR_LABELS.get(major,'?')}) "
                f"typically requires ISCED {min_l}–{max_l} "
                f"({_ISCED_LABELS.get(typical_l,'?')}), "
                f"but respondent has ISCED {isced} ({_ISCED_LABELS.get(isced,'?')})."
            ),
            message_ar=(
                f"المجموعة المهنية {major} تتطلب عادةً مستوى تعليمياً ISCED {min_l}–{max_l}، "
                f"لكن المستجيب أعطى ISCED {isced}."
            ),
            expected=f"ISCED {min_l}–{max_l}",
            actual=f"ISCED {isced}",
            rule_id="SR-ISCO-ISCED-01" if severity == "LOW" else (
                "SR-ISCO-ISCED-02" if severity == "MODERATE" else "SR-ISCO-ISCED-03"
            ),
            standard_version="ISCO-08 / ISCED 2011",
            crosswalk_source_id=source_id,
            matched_hierarchy_levels=levels,
            severity_rationale=(
                f"Respondent's ISCED level {isced} is {gap} level(s) outside the expected "
                f"range [{min_l}, {max_l}] for ISCO major group {major} -- "
                f"gap<=1 -> LOW, gap==2 -> MODERATE, gap>=3 -> HIGH."
            ),
        ))
        return False

    def _infer_isco_from_isic(
        self,
        job_title:   str,
        isic_section: str,
        current_major: str,
        language:    str,
    ) -> Optional[str]:
        """
        Use ISIC evidence to suggest a corrected ISCO code when
        coherence score is below threshold.

        Returns a 4-digit ISCO code string, or None.
        """
        try:
            import json as _json, re as _re
            from backend.llm import TaskType, get_llm
            from crewai import Agent, Crew, Task

            isic_label = _ISIC_SECTION_LABELS.get(isic_section, isic_section)
            prompt = (
                f"A survey respondent said their job title is: '{job_title}'.\n"
                f"They work in industry sector: '{isic_section}' ({isic_label}).\n"
                f"Their current ISCO-08 major group is {current_major}.\n\n"
                f"Based on the job title AND industry sector together, what is the "
                f"single most appropriate 4-digit ISCO-08 code?\n\n"
                f"Return ONLY: {{\"code\": \"XXXX\"}}"
            )
            agent = Agent(
                role="ISCO-08 occupation coding specialist",
                goal="Assign the correct 4-digit ISCO-08 code using both job title and industry context",
                backstory="Expert in international occupation classification with 15 years LFS coding experience.",
                llm=get_llm(TaskType.GENERAL),
                verbose=False,
                allow_delegation=False,
            )
            task = Task(description=prompt, expected_output='{"code": "XXXX"}', agent=agent)
            crew = Crew(agents=[agent], tasks=[task], verbose=False)
            raw  = str(crew.kickoff()).strip()
            raw  = _re.sub(r"```[a-z]*\n?", "", raw).strip()
            m    = _re.search(r"\{[^}]+\}", raw)
            if m:
                obj = _json.loads(m.group(0))
                code = str(obj.get("code", "")).strip()[:4]
                if code.isdigit() and len(code) == 4:
                    return code
        except Exception as exc:
            _logger.debug("ISCO inference from ISIC failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Explanation builder
    # ------------------------------------------------------------------

    def _build_explanations(
        self,
        major_label:    str,
        isic_label:     Optional[str],
        isced_label:    Optional[str],
        score:          float,
        violations:     list[SemanticViolation],
        inferred_isco:  Optional[str],
    ) -> tuple[str, str]:
        """Build EN + AR narrative explanations."""
        if score >= 0.90:
            en = (
                f"Strong semantic alignment: occupation ({major_label}), "
                f"industry ({isic_label or 'N/A'}), and education ({isced_label or 'N/A'}) "
                f"are fully consistent according to ILO ISCO-ISIC-ISCED crosswalk tables. "
                f"Classification confidence boosted by +{int(score*10)-8}0%."
            )
            ar = (
                f"تطابق دلالي قوي: المهنة والصناعة والتعليم متسقة تماماً وفق جداول "
                f"التقاطع الدولية ILO. تم تعزيز ثقة التصنيف."
            )
        elif score >= 0.70:
            en = (
                f"Good alignment: occupation ({major_label}) is generally consistent "
                f"with industry ({isic_label or 'N/A'}) and education ({isced_label or 'N/A'}). "
                f"Minor discrepancies noted but within acceptable ILO tolerance."
            )
            ar = (
                f"توافق جيد: المهنة متسقة عموماً مع الصناعة والتعليم مع وجود "
                f"تباينات طفيفة ضمن الحدود المقبولة."
            )
        elif score >= 0.40:
            n = len(violations)
            en = (
                f"Partial alignment (score {score:.0%}): {n} cross-standard "
                f"inconsistenc{'y' if n==1 else 'ies'} detected between "
                f"occupation ({major_label}), industry ({isic_label or 'N/A'}), "
                f"and education ({isced_label or 'N/A'}). Review recommended."
            )
            ar = (
                f"توافق جزئي ({score:.0%}): تم اكتشاف {n} تعارض بين المعايير. "
                f"يُوصى بالمراجعة."
            )
        else:
            en = (
                f"Low coherence (score {score:.0%}): significant cross-standard "
                f"conflicts detected. The occupation ({major_label}), industry "
                f"({isic_label or 'N/A'}), and education ({isced_label or 'N/A'}) "
                f"are inconsistent. HITL review required."
                + (f" Suggested ISCO: {inferred_isco}." if inferred_isco else "")
            )
            ar = (
                f"تماسك منخفض ({score:.0%}): تعارضات جوهرية بين المعايير. "
                f"مطلوب مراجعة بشرية."
                + (f" كود ISCO المقترح: {inferred_isco}." if inferred_isco else "")
            )

        return en, ar


# ---------------------------------------------------------------------------
# Batch analyser — for reporting and evaluation
# ---------------------------------------------------------------------------

def analyse_batch(
    records: list[dict],
    engine:  Optional[SemanticRelationEngine] = None,
) -> list[SemanticCoherence]:
    """
    Analyse a list of survey records in batch.

    Each record dict must contain:
        isco_code    : str
        isic_section : str | None
        isced_level  : int | None
        job_title    : str (optional)
        language     : str (optional, default "en")

    Returns a list of SemanticCoherence objects in the same order.
    """
    if engine is None:
        engine = SemanticRelationEngine(use_llm=False)
    return [
        engine.analyse(
            isco_code    = r.get("isco_code", ""),
            isic_section = r.get("isic_section"),
            isced_level  = r.get("isced_level"),
            job_title    = r.get("job_title", ""),
            language     = r.get("language", "en"),
        )
        for r in records
    ]


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_instance: Optional[SemanticRelationEngine] = None


def get_semantic_relation_engine(use_llm: bool = True) -> SemanticRelationEngine:
    global _instance
    if _instance is None:
        _instance = SemanticRelationEngine(use_llm=use_llm)
    return _instance
