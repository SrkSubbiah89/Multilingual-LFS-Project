"""
Tests for backend/agents/validation_agent.py

Rule-based logic (_normalise, all R0x checks) is tested without mocking.
validate() tests that exercise the LLM path (stage 2) patch Agent/Crew
to avoid external API calls.
"""

from unittest.mock import MagicMock

import pytest

from backend.agents.validation_agent import (
    RuleViolation,
    ValidationAgent,
    ValidationResult,
    _EMPLOYED_STATUSES,
    _FULL_TIME_MIN,
    _HOURS_EXTREME,
    _HOURS_MAX,
    _HOURS_MIN,
    _PART_TIME_MAX,
    _PENALTY_ERROR,
    _PENALTY_WARNING,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_EMPLOYED = {
    "employment_status": "employed",
    "job_title":         "software engineer",
    "industry":          "technology",
    "hours_per_week":    "40",
    "employment_type":   "full_time",
}

_VALID_UNEMPLOYED = {
    "employment_status": "unemployed",
    "industry":          "technology",
}

_LLM_OK = (
    '{"is_semantically_consistent": true, "confidence": 0.95, '
    '"issues": [], '
    '"explanation_en": "Responses are consistent.", '
    '"explanation_ar": "الإجابات متسقة."}'
)

_LLM_FAIL = (
    '{"is_semantically_consistent": false, "confidence": 0.60, '
    '"issues": ["Job title does not match industry."], '
    '"explanation_en": "Inconsistency detected.", '
    '"explanation_ar": "تم اكتشاف تناقض."}'
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_crew(monkeypatch):
    crew_instance = MagicMock()
    crew_instance.kickoff.return_value = _LLM_OK
    crew_class = MagicMock(return_value=crew_instance)
    monkeypatch.setattr("backend.agents.validation_agent.Agent", MagicMock())
    monkeypatch.setattr("backend.agents.validation_agent.Crew",  crew_class)
    monkeypatch.setattr("backend.agents.validation_agent.Task",  MagicMock())
    return crew_instance


@pytest.fixture
def agent(monkeypatch, mock_crew):
    monkeypatch.setattr(
        "backend.agents.validation_agent.get_llm",
        lambda *a, **kw: MagicMock(),
    )
    return ValidationAgent()


# ---------------------------------------------------------------------------
# _normalise
# ---------------------------------------------------------------------------

class TestNormalise:
    @pytest.fixture
    def ag(self, monkeypatch, mock_crew):
        monkeypatch.setattr(
            "backend.agents.validation_agent.get_llm", lambda *a, **kw: MagicMock()
        )
        return ValidationAgent()

    def test_strips_whitespace_from_values(self, ag):
        n = ag._normalise({"employment_status": "  employed  "})
        assert n["employment_status"] == "employed"

    def test_lowercases_employment_status(self, ag):
        n = ag._normalise({"employment_status": "Employed"})
        assert n["employment_status"] == "employed"

    def test_lowercases_employment_type(self, ag):
        n = ag._normalise({"employment_type": "Full_Time"})
        assert n["employment_type"] == "full_time"

    def test_casts_hours_float_to_int_string(self, ag):
        n = ag._normalise({"hours_per_week": "40.0"})
        assert n["hours_per_week"] == "40"

    def test_preserves_invalid_hours_as_string(self, ag):
        n = ag._normalise({"hours_per_week": "forty"})
        assert n["hours_per_week"] == "forty"

    def test_preserves_extra_keys(self, ag):
        n = ag._normalise({"employment_status": "employed", "extra_field": "x"})
        assert "extra_field" in n

    def test_none_values_become_empty_string(self, ag):
        n = ag._normalise({"job_title": None})
        assert n["job_title"] == ""


# ---------------------------------------------------------------------------
# R01 — Required fields
# ---------------------------------------------------------------------------

class TestR01RequiredFields:
    @pytest.fixture
    def ag(self, monkeypatch, mock_crew):
        monkeypatch.setattr(
            "backend.agents.validation_agent.get_llm", lambda *a, **kw: MagicMock()
        )
        return ValidationAgent()

    def test_missing_employment_status_is_error(self, ag):
        v = ag._r01_required_fields({"industry": "tech"})
        codes = [r.rule_id for r in v]
        assert "R01" in codes
        fields = [r.field for r in v]
        assert "employment_status" in fields

    def test_missing_industry_is_error(self, ag):
        v = ag._r01_required_fields({"employment_status": "unemployed"})
        assert any(r.field == "industry" for r in v)

    def test_employed_missing_job_title_is_error(self, ag):
        data = {
            "employment_status": "employed",
            "industry": "tech",
            "hours_per_week": "40",
            "employment_type": "full_time",
        }
        v = ag._r01_required_fields(data)
        assert any(r.field == "job_title" for r in v)

    def test_unemployed_missing_job_title_is_not_error(self, ag):
        data = {"employment_status": "unemployed", "industry": "tech"}
        v = ag._r01_required_fields(data)
        assert not any(r.field == "job_title" for r in v)

    def test_all_fields_present_no_violations(self, ag):
        v = ag._r01_required_fields(ag._normalise(_VALID_EMPLOYED))
        assert v == []

    def test_empty_field_value_triggers_error(self, ag):
        data = {"employment_status": "", "industry": "tech"}
        v = ag._r01_required_fields(data)
        assert any(r.field == "employment_status" for r in v)


# ---------------------------------------------------------------------------
# R02 — hours_per_week must be numeric
# ---------------------------------------------------------------------------

class TestR02HoursNumeric:
    @pytest.fixture
    def ag(self, monkeypatch, mock_crew):
        monkeypatch.setattr(
            "backend.agents.validation_agent.get_llm", lambda *a, **kw: MagicMock()
        )
        return ValidationAgent()

    def test_non_numeric_string_is_error(self, ag):
        v = ag._r02_hours_numeric({"hours_per_week": "forty"})
        assert len(v) == 1
        assert v[0].rule_id == "R02"
        assert v[0].severity == "error"

    def test_integer_string_passes(self, ag):
        assert ag._r02_hours_numeric({"hours_per_week": "40"}) == []

    def test_float_string_passes(self, ag):
        assert ag._r02_hours_numeric({"hours_per_week": "37.5"}) == []

    def test_absent_field_passes(self, ag):
        assert ag._r02_hours_numeric({}) == []

    def test_empty_string_passes(self, ag):
        # Empty hours is caught by R01, not R02
        assert ag._r02_hours_numeric({"hours_per_week": ""}) == []


# ---------------------------------------------------------------------------
# R03 — hours_per_week range [1, 168]
# ---------------------------------------------------------------------------

class TestR03HoursRange:
    @pytest.fixture
    def ag(self, monkeypatch, mock_crew):
        monkeypatch.setattr(
            "backend.agents.validation_agent.get_llm", lambda *a, **kw: MagicMock()
        )
        return ValidationAgent()

    def test_zero_hours_is_error(self, ag):
        v = ag._r03_hours_range({"hours_per_week": "0"})
        assert len(v) == 1 and v[0].rule_id == "R03"

    def test_negative_hours_is_error(self, ag):
        v = ag._r03_hours_range({"hours_per_week": "-5"})
        assert v[0].rule_id == "R03"

    def test_over_168_is_error(self, ag):
        v = ag._r03_hours_range({"hours_per_week": "169"})
        assert v[0].rule_id == "R03"

    def test_exactly_168_passes(self, ag):
        assert ag._r03_hours_range({"hours_per_week": "168"}) == []

    def test_exactly_1_passes(self, ag):
        assert ag._r03_hours_range({"hours_per_week": "1"}) == []

    def test_normal_hours_passes(self, ag):
        assert ag._r03_hours_range({"hours_per_week": "40"}) == []

    def test_non_numeric_skipped(self, ag):
        # R02 handles non-numeric; R03 should return empty
        assert ag._r03_hours_range({"hours_per_week": "forty"}) == []


# ---------------------------------------------------------------------------
# R04 — hours_per_week extreme outlier
# ---------------------------------------------------------------------------

class TestR04HoursExtreme:
    @pytest.fixture
    def ag(self, monkeypatch, mock_crew):
        monkeypatch.setattr(
            "backend.agents.validation_agent.get_llm", lambda *a, **kw: MagicMock()
        )
        return ValidationAgent()

    def test_over_80_is_warning(self, ag):
        v = ag._r04_hours_extreme({"hours_per_week": "90"})
        assert len(v) == 1
        assert v[0].rule_id == "R04"
        assert v[0].severity == "warning"

    def test_exactly_80_passes(self, ag):
        assert ag._r04_hours_extreme({"hours_per_week": "80"}) == []

    def test_normal_hours_passes(self, ag):
        assert ag._r04_hours_extreme({"hours_per_week": "40"}) == []

    def test_absent_field_passes(self, ag):
        assert ag._r04_hours_extreme({}) == []


# ---------------------------------------------------------------------------
# R05 / R06 — Unemployed consistency
# ---------------------------------------------------------------------------

class TestR05R06UnemployedConsistency:
    @pytest.fixture
    def ag(self, monkeypatch, mock_crew):
        monkeypatch.setattr(
            "backend.agents.validation_agent.get_llm", lambda *a, **kw: MagicMock()
        )
        return ValidationAgent()

    def test_unemployed_with_job_title_is_r05_error(self, ag):
        data = {"employment_status": "unemployed", "job_title": "engineer"}
        v = ag._r05_r06_unemployed_consistency(data)
        assert any(r.rule_id == "R05" for r in v)

    def test_unemployed_with_positive_hours_is_r06_error(self, ag):
        data = {"employment_status": "unemployed", "hours_per_week": "40"}
        v = ag._r05_r06_unemployed_consistency(data)
        assert any(r.rule_id == "R06" for r in v)

    def test_unemployed_zero_hours_no_error(self, ag):
        data = {"employment_status": "unemployed", "hours_per_week": "0"}
        v = ag._r05_r06_unemployed_consistency(data)
        assert not any(r.rule_id == "R06" for r in v)

    def test_employed_with_job_title_no_error(self, ag):
        data = {"employment_status": "employed", "job_title": "engineer"}
        v = ag._r05_r06_unemployed_consistency(data)
        assert v == []

    def test_non_unemployed_status_skipped(self, ag):
        data = {"employment_status": "not_in_labour_force", "job_title": "teacher"}
        v = ag._r05_r06_unemployed_consistency(data)
        assert v == []


# ---------------------------------------------------------------------------
# R07 — Not-in-labour-force consistency
# ---------------------------------------------------------------------------

class TestR07NilfConsistency:
    @pytest.fixture
    def ag(self, monkeypatch, mock_crew):
        monkeypatch.setattr(
            "backend.agents.validation_agent.get_llm", lambda *a, **kw: MagicMock()
        )
        return ValidationAgent()

    def test_nilf_with_job_title_is_r07_error(self, ag):
        data = {"employment_status": "not_in_labour_force", "job_title": "teacher"}
        v = ag._r07_nilf_consistency(data)
        assert len(v) == 1 and v[0].rule_id == "R07"

    def test_nilf_without_job_title_no_error(self, ag):
        data = {"employment_status": "not_in_labour_force"}
        assert ag._r07_nilf_consistency(data) == []

    def test_employed_status_skipped(self, ag):
        data = {"employment_status": "employed", "job_title": "teacher"}
        assert ag._r07_nilf_consistency(data) == []


# ---------------------------------------------------------------------------
# R08 / R09 / R10 — hours × employment_type consistency
# ---------------------------------------------------------------------------

class TestR08R09R10HoursTypeConsistency:
    @pytest.fixture
    def ag(self, monkeypatch, mock_crew):
        monkeypatch.setattr(
            "backend.agents.validation_agent.get_llm", lambda *a, **kw: MagicMock()
        )
        return ValidationAgent()

    def test_full_time_low_hours_is_r08_warning(self, ag):
        data = {"employment_type": "full_time",
                "employment_status": "employed", "hours_per_week": "10"}
        v = ag._r08_r09_r10_hours_type_consistency(data)
        assert any(r.rule_id == "R08" and r.severity == "warning" for r in v)

    def test_full_time_40_hours_passes(self, ag):
        data = {"employment_type": "full_time",
                "employment_status": "employed", "hours_per_week": "40"}
        v = ag._r08_r09_r10_hours_type_consistency(data)
        assert not any(r.rule_id == "R08" for r in v)

    def test_part_time_high_hours_is_r09_warning(self, ag):
        data = {"employment_type": "part_time",
                "employment_status": "employed", "hours_per_week": "40"}
        v = ag._r08_r09_r10_hours_type_consistency(data)
        assert any(r.rule_id == "R09" and r.severity == "warning" for r in v)

    def test_part_time_low_hours_passes(self, ag):
        data = {"employment_type": "part_time",
                "employment_status": "employed", "hours_per_week": "20"}
        v = ag._r08_r09_r10_hours_type_consistency(data)
        assert not any(r.rule_id == "R09" for r in v)

    def test_employed_zero_hours_is_r10_error(self, ag):
        data = {"employment_type": "full_time",
                "employment_status": "employed", "hours_per_week": "0"}
        v = ag._r08_r09_r10_hours_type_consistency(data)
        assert any(r.rule_id == "R10" and r.severity == "error" for r in v)

    def test_missing_employment_type_no_violations(self, ag):
        data = {"employment_status": "employed", "hours_per_week": "40"}
        assert ag._r08_r09_r10_hours_type_consistency(data) == []

    def test_out_of_range_hours_skipped(self, ag):
        # R03 already flags this; R08/09/10 should not add duplicates
        data = {"employment_type": "full_time",
                "employment_status": "employed", "hours_per_week": "200"}
        v = ag._r08_r09_r10_hours_type_consistency(data)
        assert v == []


# ---------------------------------------------------------------------------
# _parse_semantic_response
# ---------------------------------------------------------------------------

class TestParseSemanticResponse:
    @pytest.fixture
    def ag(self, monkeypatch, mock_crew):
        monkeypatch.setattr(
            "backend.agents.validation_agent.get_llm", lambda *a, **kw: MagicMock()
        )
        return ValidationAgent()

    def test_valid_json_parsed_correctly(self, ag):
        is_v, conf, issues, en, ar = ag._parse_semantic_response(_LLM_OK)
        assert is_v is True
        assert conf == pytest.approx(0.95)
        assert issues == []
        assert "consistent" in en.lower()

    def test_inconsistent_response_sets_is_valid_false(self, ag):
        is_v, conf, issues, en, ar = ag._parse_semantic_response(_LLM_FAIL)
        assert is_v is False
        assert len(issues) == 1

    def test_strips_markdown_fences(self, ag):
        raw = f"```json\n{_LLM_OK}\n```"
        is_v, conf, issues, en, ar = ag._parse_semantic_response(raw)
        assert is_v is True

    def test_malformed_json_falls_back_to_valid(self, ag):
        is_v, conf, issues, en, ar = ag._parse_semantic_response("not json")
        assert is_v is True    # safe fallback
        assert 0.0 <= conf <= 1.0

    def test_embedded_json_extracted(self, ag):
        raw = f'Some text before. {_LLM_OK} Some text after.'
        is_v, conf, issues, en, ar = ag._parse_semantic_response(raw)
        assert is_v is True

    def test_confidence_clamped_to_unit_interval(self, ag):
        raw = ('{"is_semantically_consistent": true, "confidence": 1.5, '
               '"issues": [], "explanation_en": "", "explanation_ar": ""}')
        _, conf, _, _, _ = ag._parse_semantic_response(raw)
        assert conf <= 1.0

    def test_arabic_explanation_preserved(self, ag):
        _, _, _, _, ar = ag._parse_semantic_response(_LLM_OK)
        assert "متسقة" in ar


# ---------------------------------------------------------------------------
# validate — result shape and is_valid flag
# ---------------------------------------------------------------------------

class TestValidateResultShape:
    def test_returns_validation_result(self, agent):
        result = agent.validate(_VALID_EMPLOYED)
        assert isinstance(result, ValidationResult)

    def test_valid_employed_response_is_valid(self, agent):
        result = agent.validate(_VALID_EMPLOYED)
        assert result.is_valid is True

    def test_valid_unemployed_response_is_valid(self, agent):
        result = agent.validate(_VALID_UNEMPLOYED)
        assert result.is_valid is True

    def test_confidence_in_unit_interval(self, agent):
        result = agent.validate(_VALID_EMPLOYED)
        assert 0.0 <= result.confidence <= 1.0

    def test_rule_violations_is_list(self, agent):
        result = agent.validate(_VALID_EMPLOYED)
        assert isinstance(result.rule_violations, list)

    def test_semantic_issues_is_list(self, agent):
        result = agent.validate(_VALID_EMPLOYED)
        assert isinstance(result.semantic_issues, list)

    def test_explanation_en_is_non_empty_string(self, agent):
        result = agent.validate(_VALID_EMPLOYED)
        assert isinstance(result.explanation_en, str)
        assert len(result.explanation_en) > 0

    def test_explanation_ar_is_non_empty_string(self, agent):
        result = agent.validate(_VALID_EMPLOYED)
        assert isinstance(result.explanation_ar, str)
        assert len(result.explanation_ar) > 0

    def test_validated_data_echoes_input_keys(self, agent):
        result = agent.validate(_VALID_EMPLOYED)
        for k in _VALID_EMPLOYED:
            assert k in result.validated_data


# ---------------------------------------------------------------------------
# validate — error detection
# ---------------------------------------------------------------------------

class TestValidateErrors:
    def test_missing_employment_status_fails(self, agent):
        bad = dict(_VALID_EMPLOYED)
        del bad["employment_status"]
        result = agent.validate(bad)
        assert result.is_valid is False
        assert any(v.rule_id == "R01" for v in result.rule_violations)

    def test_non_numeric_hours_fails(self, agent):
        bad = {**_VALID_EMPLOYED, "hours_per_week": "forty"}
        result = agent.validate(bad)
        assert result.is_valid is False
        assert any(v.rule_id == "R02" for v in result.rule_violations)

    def test_hours_out_of_range_fails(self, agent):
        bad = {**_VALID_EMPLOYED, "hours_per_week": "200"}
        result = agent.validate(bad)
        assert result.is_valid is False
        assert any(v.rule_id == "R03" for v in result.rule_violations)

    def test_unemployed_with_job_title_fails(self, agent):
        bad = {**_VALID_UNEMPLOYED, "job_title": "engineer"}
        result = agent.validate(bad)
        assert result.is_valid is False
        assert any(v.rule_id == "R05" for v in result.rule_violations)

    def test_employed_zero_hours_fails(self, agent):
        bad = {**_VALID_EMPLOYED, "hours_per_week": "0"}
        result = agent.validate(bad)
        assert result.is_valid is False
        assert any(v.rule_id == "R10" for v in result.rule_violations)


# ---------------------------------------------------------------------------
# validate — warning detection
# ---------------------------------------------------------------------------

class TestValidateWarnings:
    def test_extreme_hours_is_warning_not_error(self, agent):
        data = {**_VALID_EMPLOYED, "hours_per_week": "90"}
        result = agent.validate(data)
        warnings = [v for v in result.rule_violations if v.severity == "warning"]
        errors   = [v for v in result.rule_violations if v.severity == "error"]
        assert any(v.rule_id == "R04" for v in warnings)
        assert not errors

    def test_part_time_high_hours_is_warning(self, agent):
        data = {**_VALID_EMPLOYED, "employment_type": "part_time",
                "hours_per_week": "40"}
        result = agent.validate(data)
        assert any(v.rule_id == "R09" for v in result.rule_violations)
        assert result.is_valid is True   # warnings don't invalidate

    def test_full_time_low_hours_is_warning(self, agent):
        data = {**_VALID_EMPLOYED, "hours_per_week": "10"}
        result = agent.validate(data)
        assert any(v.rule_id == "R08" for v in result.rule_violations)
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# validate — confidence scoring
# ---------------------------------------------------------------------------

class TestValidateConfidence:
    def test_valid_response_has_high_confidence(self, agent):
        result = agent.validate(_VALID_EMPLOYED)
        assert result.confidence >= 0.80

    def test_each_error_reduces_confidence(self, agent):
        # Two errors: missing employment_status AND non-numeric hours
        bad = {"industry": "tech", "hours_per_week": "forty",
               "job_title": "x", "employment_type": "full_time"}
        result = agent.validate(bad)
        # Should have at least two errors → confidence well below 1.0
        assert result.confidence < 1.0 - _PENALTY_ERROR

    def test_warning_reduces_confidence_less_than_error(self, agent):
        warn_result = agent.validate({**_VALID_EMPLOYED, "hours_per_week": "90"})
        error_result = agent.validate({**_VALID_EMPLOYED, "hours_per_week": "forty"})
        assert warn_result.confidence > error_result.confidence

    def test_confidence_never_below_zero(self, agent):
        # Force many errors
        bad = {}
        result = agent.validate(bad)
        assert result.confidence >= 0.0

    def test_confidence_never_above_one(self, agent):
        result = agent.validate(_VALID_EMPLOYED)
        assert result.confidence <= 1.0


# ---------------------------------------------------------------------------
# validate — semantic path
# ---------------------------------------------------------------------------

class TestValidateSemanticPath:
    def test_semantic_stage_called_for_clean_responses(self, agent, mock_crew):
        agent.validate(_VALID_EMPLOYED)
        mock_crew.kickoff.assert_called_once()

    def test_semantic_stage_skipped_when_errors_present(self, agent, mock_crew):
        bad = {**_VALID_EMPLOYED, "hours_per_week": "not_a_number"}
        agent.validate(bad)
        mock_crew.kickoff.assert_not_called()

    def test_llm_inconsistency_sets_is_valid_false(self, agent, mock_crew):
        mock_crew.kickoff.return_value = _LLM_FAIL
        result = agent.validate(_VALID_EMPLOYED)
        assert result.is_valid is False
        assert len(result.semantic_issues) > 0

    def test_llm_issues_appear_in_semantic_issues_list(self, agent, mock_crew):
        mock_crew.kickoff.return_value = _LLM_FAIL
        result = agent.validate(_VALID_EMPLOYED)
        assert "Job title does not match industry." in result.semantic_issues

    def test_llm_failure_degrades_confidence(self, agent, mock_crew):
        mock_crew.kickoff.return_value = _LLM_FAIL
        result = agent.validate(_VALID_EMPLOYED)
        # The LLM's low confidence (0.60) and semantic issues should lower overall
        assert result.confidence < 0.90


# ---------------------------------------------------------------------------
# validate — language parameter
# ---------------------------------------------------------------------------

class TestValidateLanguage:
    def test_arabic_language_produces_arabic_explanation(self, agent):
        result = agent.validate(_VALID_EMPLOYED, language="ar")
        # Arabic explanation should contain Arabic characters
        arabic_chars = [c for c in result.explanation_ar if "\u0600" <= c <= "\u06FF"]
        assert len(arabic_chars) > 0

    def test_english_language_produces_english_explanation(self, agent):
        result = agent.validate(_VALID_EMPLOYED, language="en")
        assert len(result.explanation_en) > 0


# ---------------------------------------------------------------------------
# RuleViolation model
# ---------------------------------------------------------------------------

class TestRuleViolationModel:
    def test_rule_violation_fields_accessible(self, agent):
        bad = {**_VALID_EMPLOYED, "hours_per_week": "forty"}
        result = agent.validate(bad)
        v = next(r for r in result.rule_violations if r.rule_id == "R02")
        assert v.field == "hours_per_week"
        assert v.severity == "error"
        assert len(v.message_en) > 0
        assert len(v.message_ar) > 0

    def test_violation_severities_are_valid_strings(self, agent):
        data = {**_VALID_EMPLOYED, "hours_per_week": "90"}  # R04 warning
        result = agent.validate(data)
        for v in result.rule_violations:
            assert v.severity in ("error", "warning")
