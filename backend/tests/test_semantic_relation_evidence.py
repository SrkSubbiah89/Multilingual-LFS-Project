"""
Tests for backend/agents/semantic_relation.py's Conference I Reviewer #2
Section G additions: structured provenance on SemanticViolation, a genuine
LOW severity tier (previously documented but never emitted), and the
corrected module/class docstrings.

No live LLM/Qdrant infra required -- SemanticRelationEngine's crosswalk
core is pure Python; tests use use_llm=False throughout since these tests
target the deterministic crosswalk, not the optional LLM re-inference path.
"""

from __future__ import annotations

from backend.agents.semantic_relation import SemanticRelationEngine, SemanticViolation


def make_engine():
    return SemanticRelationEngine(use_llm=False)


# ---------------------------------------------------------------------------
# Coherent cases: no violations at all
# ---------------------------------------------------------------------------

def test_fully_coherent_case_has_no_violations():
    engine = make_engine()
    result = engine.analyse(isco_code="2500", isic_section="J", isced_level=7)  # ICT professional
    assert result.violations == []
    assert result.is_coherent is True


def test_any_industry_major_group_is_always_isic_coherent():
    engine = make_engine()
    # Major "1" (Managers) is "ANY" -- compatible with every ISIC section.
    result = engine.analyse(isco_code="1100", isic_section="U", isced_level=None)
    assert result.isco_isic_compatible is True
    assert result.violations == []


# ---------------------------------------------------------------------------
# ISCED severity tiers: LOW / MODERATE / HIGH via boundary-gap
# ---------------------------------------------------------------------------
# Major "2" (Professionals), submajor "24" (not in _ISCO_SUBMAJOR_TO_ISCED_MIN,
# so the plain major-level range [6, 8] applies, unmodified by a sub-major
# override) -- isco_code "2400".

def test_isced_gap_one_is_low_severity():
    engine = make_engine()
    result = engine.analyse(isco_code="2400", isic_section=None, isced_level=5)  # min_l=6, gap=1
    assert len(result.violations) == 1
    v = result.violations[0]
    assert v.severity == "LOW"
    assert v.violation_type == "isco_isced"
    assert v.rule_id == "SR-ISCO-ISCED-01"


def test_isced_gap_two_is_moderate_severity():
    engine = make_engine()
    result = engine.analyse(isco_code="2400", isic_section=None, isced_level=4)  # gap=2
    v = result.violations[0]
    assert v.severity == "MODERATE"
    assert v.rule_id == "SR-ISCO-ISCED-02"


def test_isced_gap_three_or_more_is_high_severity():
    engine = make_engine()
    result = engine.analyse(isco_code="2400", isic_section=None, isced_level=3)  # gap=3
    v = result.violations[0]
    assert v.severity == "HIGH"
    assert v.rule_id == "SR-ISCO-ISCED-03"


def test_isced_within_range_no_violation():
    engine = make_engine()
    result = engine.analyse(isco_code="2400", isic_section=None, isced_level=7)  # within [6,8]
    assert result.violations == []


def test_isced_violation_provenance_fields_populated():
    engine = make_engine()
    result = engine.analyse(isco_code="2400", isic_section=None, isced_level=5)
    v = result.violations[0]
    assert v.standard_version == "ISCO-08 / ISCED 2011"
    assert v.crosswalk_source_id == "_ISCO_MAJOR_TO_ISCED"
    assert v.matched_hierarchy_levels == ["major:2"]
    assert "gap" in v.severity_rationale.lower() or "level(s)" in v.severity_rationale.lower()


def test_isced_submajor_override_reflected_in_provenance():
    # Submajor "25" (ICT Professionals) has a stricter min in
    # _ISCO_SUBMAJOR_TO_ISCED_MIN (6) -- same as major "2"'s own min here,
    # so use "22" (Health Professionals, sub_min=7) to force an override
    # that actually changes min_l (major alone would allow isced=6).
    engine = make_engine()
    result = engine.analyse(isco_code="2200", isic_section=None, isced_level=6)  # min_l becomes 7, gap=1
    v = result.violations[0]
    assert v.severity == "LOW"
    assert "_ISCO_SUBMAJOR_TO_ISCED_MIN" in v.crosswalk_source_id
    assert "submajor:22" in v.matched_hierarchy_levels


# ---------------------------------------------------------------------------
# ISIC severity tiers: MODERATE (atypical) / HIGH (incompatible)
# ---------------------------------------------------------------------------

def test_isic_major_allowed_but_not_submajor_is_moderate():
    engine = make_engine()
    # Major "7" allows C/F/E/D/B; submajor "71" allows only F/C -- "E" is
    # in the major list but not the sub-major list.
    result = engine.analyse(isco_code="7100", isic_section="E", isced_level=None)
    v = result.violations[0]
    assert v.severity == "MODERATE"
    assert v.rule_id == "SR-ISCO-ISIC-01"
    assert result.isco_isic_compatible is True  # MODERATE still counts as compatible


def test_isic_not_in_major_or_submajor_is_high():
    engine = make_engine()
    # Major "6" (Agriculture) only allows "A"; "C" is nowhere in that list.
    result = engine.analyse(isco_code="6300", isic_section="C", isced_level=None)
    v = result.violations[0]
    assert v.severity == "HIGH"
    assert v.rule_id == "SR-ISCO-ISIC-02"
    assert result.isco_isic_compatible is False


def test_isic_violation_provenance_fields_populated():
    engine = make_engine()
    result = engine.analyse(isco_code="7100", isic_section="E", isced_level=None)
    v = result.violations[0]
    assert v.standard_version == "ISCO-08 / ISIC Rev.4"
    assert v.crosswalk_source_id == "_ISCO_SUBMAJOR_TO_ISIC"
    assert "submajor:71" in v.matched_hierarchy_levels
    assert v.severity_rationale


def test_isic_major_level_source_when_no_submajor_rule():
    engine = make_engine()
    # Submajor "63" has no entry in _ISCO_SUBMAJOR_TO_ISIC -- falls back to
    # the major-level rule for "6".
    result = engine.analyse(isco_code="6300", isic_section="C", isced_level=None)
    v = result.violations[0]
    assert v.crosswalk_source_id == "_ISCO_MAJOR_TO_ISIC"
    assert v.matched_hierarchy_levels == ["major:6"]


# ---------------------------------------------------------------------------
# Backward compatibility: SemanticViolation still constructible without the
# new provenance kwargs (positional-only, matching pre-Section-G callers)
# ---------------------------------------------------------------------------

def test_semantic_violation_constructible_without_provenance_kwargs():
    v = SemanticViolation(
        violation_type="isco_isic", severity="HIGH",
        message_en="x", message_ar="y", expected="A", actual="B",
    )
    assert v.rule_id == ""
    assert v.standard_version == ""
    assert v.crosswalk_source_id == ""
    assert v.matched_hierarchy_levels == []
    assert v.severity_rationale == ""


# ---------------------------------------------------------------------------
# Combined ISIC + ISCED violations in one analyse() call
# ---------------------------------------------------------------------------

def test_combined_isic_and_isced_violations_both_carry_provenance():
    engine = make_engine()
    result = engine.analyse(isco_code="6300", isic_section="C", isced_level=8)  # ISIC HIGH, ISCED (major "6": 0-3) gap=5 -> HIGH
    assert len(result.violations) == 2
    types = {v.violation_type for v in result.violations}
    assert types == {"isco_isic", "isco_isced"}
    for v in result.violations:
        assert v.rule_id
        assert v.standard_version
