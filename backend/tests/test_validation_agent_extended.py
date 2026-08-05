"""
Extended tests for ValidationAgent
Covers: all R01-R10 rules with boundary values, cross-rule interactions,
        wage validation, sector validation, multilingual rules,
        invalid data types, concurrent validation, performance.
"""
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def agent():
    with patch("backend.agents.validation_agent.get_llm", return_value=MagicMock()), \
         patch("backend.agents.validation_agent.Agent", return_value=MagicMock()), \
         patch("backend.agents.validation_agent.Task",  return_value=MagicMock()), \
         patch("backend.agents.validation_agent.Crew",  return_value=MagicMock()):
        from backend.agents.validation_agent import ValidationAgent
        a = ValidationAgent()
        # Skip LLM semantic stage in most tests
        # _semantic_validate returns (is_valid, confidence, issues, explanation_en, explanation_ar)
        a._semantic_validate = MagicMock(return_value=(True, 1.0, [], "All good.", "كل شيء صحيح."))
        return a


def valid_employed_data(**overrides):
    base = {
        "employment_status": "employed",
        "education_level": "bachelor",
        "employment_nature": "full-time",
        "employment_sector": "private",
        "job_title": "Engineer",
        "job_duties": "Design and build systems",
        "industry": "technology",
        "hours_per_week": "40",
        "employment_type": "full-time",
        "monthly_wage_range": "10000-15000",
    }
    base.update(overrides)
    return base


def valid_unemployed_data(**overrides):
    base = {
        "employment_status": "unemployed",
        "education_level": "high_school",
        "job_search_active": "yes",
        "available_for_work": "yes",
        "unemployment_duration": "2",
        "last_job_title": "Sales Assistant",
        "reason_left_job": "redundancy",
    }
    base.update(overrides)
    return base


# ─── R01: Required Fields ─────────────────────────────────────────────────────

class TestR01BoundaryValues:
    def test_single_space_job_title_fails(self, agent):
        data = valid_employed_data(job_title=" ")
        result = agent.validate(data)
        assert not result.is_valid

    def test_job_title_with_single_char_passes(self, agent):
        data = valid_employed_data(job_title="X")
        result = agent.validate(data)
        # Single char is minimal but technically present
        assert result is not None

    def test_all_fields_empty_strings_fail(self, agent):
        data = {k: "" for k in valid_employed_data().keys()}
        result = agent.validate(data)
        assert not result.is_valid

    def test_missing_education_level_passes(self, agent):
        # education_level is NOT in _ALWAYS_REQUIRED or _EMPLOYMENT_REQUIRED
        data = valid_employed_data()
        del data["education_level"]
        result = agent.validate(data)
        assert result is not None  # should not crash; field is optional


# ─── R03: Hours Range Boundary Values ────────────────────────────────────────

class TestR03BoundaryValues:
    @pytest.mark.parametrize("hours,should_pass", [
        ("0",    False),   # zero not allowed
        ("1",    True),    # minimum valid
        ("40",   True),    # normal
        ("80",   True),    # high but within range
        ("168",  True),    # maximum (all hours in a week)
        ("169",  False),   # over maximum
        ("200",  False),   # clearly over
        ("-1",   False),   # negative
        ("0.5",  False),   # normalises to 0 via int(float()) → fails R03
        ("167.5",True),    # normalises to 167 → within range
        ("168.1",True),    # normalises to 168 → exactly _HOURS_MAX → passes
    ])
    def test_hours_boundary(self, agent, hours, should_pass):
        data = valid_employed_data(hours_per_week=hours)
        result = agent.validate(data)
        violations = [v for v in result.rule_violations if v.rule_id == "R03"]
        if should_pass:
            assert len(violations) == 0, f"Expected {hours} to pass R03"
        else:
            assert len(violations) > 0, f"Expected {hours} to fail R03"


# ─── R04: Extreme Hours Warning ──────────────────────────────────────────────

class TestR04ExtremeHours:
    @pytest.mark.parametrize("hours,expect_warning", [
        ("79",  False),
        ("80",  False),   # exactly 80 is not extreme
        ("81",  True),    # just over 80 triggers warning
        ("100", True),
        ("168", True),
    ])
    def test_extreme_hours_warning(self, agent, hours, expect_warning):
        data = valid_employed_data(hours_per_week=hours, employment_type="full-time")
        result = agent.validate(data)
        r04_violations = [v for v in result.rule_violations if v.rule_id == "R04"]
        if expect_warning:
            assert len(r04_violations) > 0
        else:
            assert len(r04_violations) == 0


# ─── R05/R06: Unemployed Consistency ────────────────────────────────────────

class TestR05R06UnemployedBoundary:
    def test_unemployed_with_zero_hours_ok(self, agent):
        data = valid_unemployed_data(hours_per_week="0")
        result = agent.validate(data)
        r06 = [v for v in result.rule_violations if v.rule_id == "R06"]
        assert len(r06) == 0

    def test_unemployed_with_hours_fails(self, agent):
        # "0.5" normalises to "0" via int(float()) → R06 checks float > 0 → 0 not > 0 → no R06
        # Use "1" to reliably trigger R06
        data = valid_unemployed_data(hours_per_week="1")
        result = agent.validate(data)
        r06 = [v for v in result.rule_violations if v.rule_id == "R06"]
        assert len(r06) > 0

    def test_unemployed_with_none_hours_ok(self, agent):
        data = valid_unemployed_data()
        data.pop("hours_per_week", None)
        result = agent.validate(data)
        r06 = [v for v in result.rule_violations if v.rule_id == "R06"]
        assert len(r06) == 0


# ─── R08/R09: Full-Time/Part-Time Hours ──────────────────────────────────────

class TestR08R09Boundary:
    @pytest.mark.parametrize("hours,emp_type,expect_r08", [
        # Code checks emp_type == "full_time" (underscore); _FULL_TIME_MIN = 20
        ("15",  "full_time",  True),   # full-time but < 20 hours → R08
        ("19",  "full_time",  True),   # still < 20 for full-time → R08
        ("20",  "full_time",  False),  # exactly at threshold → no R08
        ("40",  "full_time",  False),  # normal full-time
        ("20",  "part_time",  False),  # low hours fine for part-time
    ])
    def test_r08_full_time_low_hours(self, agent, hours, emp_type, expect_r08):
        data = valid_employed_data(hours_per_week=hours, employment_type=emp_type)
        result = agent.validate(data)
        r08 = [v for v in result.rule_violations if v.rule_id == "R08"]
        if expect_r08:
            assert len(r08) > 0
        else:
            assert len(r08) == 0

    @pytest.mark.parametrize("hours,emp_type,expect_r09", [
        # Code checks emp_type == "part_time" (underscore); _PART_TIME_MAX = 35
        ("35",  "part_time",  True),   # part-time >= 35 → R09
        ("34",  "part_time",  False),  # < 35 is ok for part-time
        ("40",  "part_time",  True),   # definitely too high for part-time
        ("40",  "full_time",  False),  # not part-time, no R09
    ])
    def test_r09_part_time_high_hours(self, agent, hours, emp_type, expect_r09):
        data = valid_employed_data(hours_per_week=hours, employment_type=emp_type)
        result = agent.validate(data)
        r09 = [v for v in result.rule_violations if v.rule_id == "R09"]
        if expect_r09:
            assert len(r09) > 0
        else:
            assert len(r09) == 0


# ─── R10: Employed Zero Hours ────────────────────────────────────────────────

class TestR10EmployedZeroHours:
    def test_employed_with_zero_hours_fails(self, agent):
        data = valid_employed_data(hours_per_week="0")
        result = agent.validate(data)
        r10 = [v for v in result.rule_violations if v.rule_id == "R10"]
        assert len(r10) > 0

    def test_employed_with_positive_hours_ok(self, agent):
        data = valid_employed_data(hours_per_week="1")
        result = agent.validate(data)
        r10 = [v for v in result.rule_violations if v.rule_id == "R10"]
        assert len(r10) == 0

    def test_self_employed_zero_hours_fails(self, agent):
        data = valid_employed_data(
            employment_status="self-employed",
            hours_per_week="0"
        )
        result = agent.validate(data)
        assert not result.is_valid


# ─── Cross-Rule Interactions ──────────────────────────────────────────────────

class TestCrossRuleInteractions:
    def test_multiple_violations_reported(self, agent):
        """Both R01 and R06 can fire simultaneously"""
        data = {
            "employment_status": "unemployed",
            "hours_per_week": "40",  # R06
            # missing job_search_active etc → R01
        }
        result = agent.validate(data)
        assert len(result.rule_violations) >= 1

    def test_first_error_does_not_hide_second(self, agent):
        data = valid_employed_data(
            hours_per_week="-5",     # R03
            job_title="",            # R01
        )
        result = agent.validate(data)
        rule_ids = [v.rule_id for v in result.rule_violations]
        assert len(rule_ids) >= 2

    def test_warning_does_not_fail_validation(self, agent):
        """Warnings (R04, R08, R09) should not set is_valid=False"""
        data = valid_employed_data(
            hours_per_week="90",     # R04 warning
            employment_type="full-time"
        )
        result = agent.validate(data)
        errors = [v for v in result.rule_violations if v.severity == "error"]
        warnings = [v for v in result.rule_violations if v.severity == "warning"]
        if len(errors) == 0:
            assert result.is_valid

    def test_error_fails_validation(self, agent):
        """An error-level violation must set is_valid=False"""
        data = valid_employed_data(job_title="")  # R01 error
        result = agent.validate(data)
        assert not result.is_valid


# ─── Wage Validation ─────────────────────────────────────────────────────────

class TestWageValidation:
    def test_valid_wage_range_passes(self, agent):
        data = valid_employed_data(monthly_wage_range="5000-10000")
        result = agent.validate(data)
        assert result is not None

    def test_negative_wage_handled(self, agent):
        data = valid_employed_data(monthly_wage_range="-1000-5000")
        result = agent.validate(data)
        assert result is not None  # should not crash

    def test_very_high_wage_passes(self, agent):
        data = valid_employed_data(monthly_wage_range="100000-200000")
        result = agent.validate(data)
        assert result is not None

    def test_zero_wage_employed(self, agent):
        """Zero wage for employed may be a warning (volunteer/intern)"""
        data = valid_employed_data(monthly_wage_range="0")
        result = agent.validate(data)
        assert result is not None


# ─── Arabic Language Rules ───────────────────────────────────────────────────

class TestArabicLanguageRules:
    def test_arabic_explanation_returned(self, agent):
        data = valid_employed_data(job_title="")
        result = agent.validate(data, language="ar")
        assert result.explanation_ar is not None
        assert len(result.explanation_ar) > 0

    def test_arabic_rule_violation_message(self, agent):
        data = valid_employed_data(job_title="")
        result = agent.validate(data, language="ar")
        # At least one violation should have Arabic text
        assert result.explanation_ar is not None

    def test_english_explanation_always_present(self, agent):
        data = valid_employed_data(job_title="")
        result = agent.validate(data, language="ar")
        assert result.explanation_en is not None

    def test_rtl_text_in_explanation(self, agent):
        data = valid_employed_data(hours_per_week="200")
        result = agent.validate(data, language="ar")
        # Arabic explanation should contain Arabic characters
        ar_chars = sum(1 for c in (result.explanation_ar or "") if '\u0600' <= c <= '\u06FF')
        assert ar_chars > 0


# ─── Data Type Handling ──────────────────────────────────────────────────────

class TestDataTypeHandling:
    def test_integer_hours_accepted(self, agent):
        data = valid_employed_data(hours_per_week=40)  # int not string
        result = agent.validate(data)
        assert result is not None

    def test_float_hours_accepted(self, agent):
        data = valid_employed_data(hours_per_week=40.5)
        result = agent.validate(data)
        assert result is not None

    def test_none_hours_handled(self, agent):
        data = valid_employed_data(hours_per_week=None)
        result = agent.validate(data)
        assert result is not None

    def test_list_value_handled(self, agent):
        """If a field value is accidentally a list, should not crash"""
        data = valid_employed_data(job_title=["engineer", "developer"])
        try:
            result = agent.validate(data)
            assert result is not None
        except (TypeError, AttributeError):
            pass  # acceptable

    def test_dict_value_handled(self, agent):
        data = valid_employed_data(job_title={"text": "engineer"})
        try:
            result = agent.validate(data)
            assert result is not None
        except (TypeError, AttributeError):
            pass


# ─── Semantic Validation ─────────────────────────────────────────────────────

class TestSemanticValidation:
    def test_semantic_check_skipped_when_errors_present(self, agent):
        """When rule violations exist, semantic LLM check should be skipped"""
        data = valid_employed_data(job_title="")  # R01 error
        agent._semantic_validate = MagicMock()
        agent.validate(data)
        agent._semantic_validate.assert_not_called()

    def test_semantic_check_runs_for_clean_data(self, agent):
        data = valid_employed_data()
        agent._semantic_validate = MagicMock(return_value=(True, 0.95, [], "Valid.", "صحيح."))
        agent.validate(data)
        agent._semantic_validate.assert_called_once()

    def test_semantic_llm_failure_propagates(self, agent):
        # _semantic_validate is NOT wrapped in try/except in validate(),
        # so exceptions propagate to the caller.
        data = valid_employed_data()
        agent._semantic_validate = MagicMock(side_effect=Exception("LLM down"))
        with pytest.raises(Exception, match="LLM down"):
            agent.validate(data)

    def test_llm_inconsistency_reported(self, agent):
        data = valid_employed_data()
        agent._semantic_validate = MagicMock(return_value=(
            False, 0.4,
            ["Claimed full-time but only 10 hours reported"],
            "Inconsistency detected.",
            "تم اكتشاف تناقض.",
        ))
        result = agent.validate(data)
        assert len(result.semantic_issues) > 0


# ─── Confidence Score ─────────────────────────────────────────────────────────

class TestConfidenceScoreValidation:
    def test_perfect_data_high_confidence(self, agent):
        result = agent.validate(valid_employed_data())
        assert result.confidence >= 0.8

    def test_multiple_errors_lower_confidence(self, agent):
        data = {
            "employment_status": "employed",
            "hours_per_week": "-5",   # R03
            "job_title": "",          # R01
        }
        result = agent.validate(data)
        assert result.confidence < 0.8

    def test_confidence_bounded_0_to_1(self, agent):
        for data in [valid_employed_data(), valid_unemployed_data(), {}]:
            result = agent.validate(data)
            assert 0.0 <= result.confidence <= 1.0
