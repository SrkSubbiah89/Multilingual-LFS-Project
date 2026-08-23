"""
Tests for backend/agents/report_generator.py

Strategy
--------
- SQLite in-memory database (module scope) replaces PostgreSQL.
- CrewAI components (get_llm, Agent, Task, Crew) are monkeypatched.
- ContextMemory is replaced with a no-op fake.
- All DB helper functions (_make_user, _make_session, etc.) mirror the
  pattern used in test_hitl_quality_manager.py.
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backend.agents.report_generator as _rg_mod
from backend.agents.report_generator import (
    EmploymentProfile,
    ReportGenerator,
    SurveyReport,
    get_report_generator,
)
from backend.database.models import (
    Base,
    QualityReview,
    SurveyReportRecord,
    SurveyResponse,
    SurveySession,
    User,
)

# ---------------------------------------------------------------------------
# Module-level alias (the module defines _build_fallback as a method; we test
# the fallback dict content through the instance method below)
# ---------------------------------------------------------------------------

_email_counter = itertools.count(1)


# ---------------------------------------------------------------------------
# SQLite in-memory engine (module scope for speed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def session_factory(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _make_user(session_factory) -> User:
    db = session_factory()
    try:
        u = User(email=f"user{next(_email_counter)}@test.com")
        db.add(u)
        db.commit()
        db.refresh(u)
        return u
    finally:
        db.close()


def _make_session(session_factory, user_id: int, status: str = "completed") -> SurveySession:
    db = session_factory()
    try:
        s = SurveySession(user_id=user_id, language="en", status=status)
        db.add(s)
        db.commit()
        db.refresh(s)
        return s
    finally:
        db.close()


def _make_response(
    session_factory,
    session_id: int,
    question_id: str,
    answer: str,
    isco_code: str | None = None,
    confidence_score: float | None = None,
) -> SurveyResponse:
    db = session_factory()
    try:
        r = SurveyResponse(
            session_id=session_id,
            question_id=question_id,
            answer=answer,
            isco_code=isco_code,
            confidence_score=confidence_score,
        )
        db.add(r)
        db.commit()
        db.refresh(r)
        return r
    finally:
        db.close()


def _make_quality_review(
    session_factory,
    session_id: int,
    passed: bool = True,
    escalated: bool = False,
    quality_score: float = 0.90,
    flagged_count: int = 0,
) -> QualityReview:
    db = session_factory()
    try:
        qr = QualityReview(
            session_id=session_id,
            quality_score=quality_score,
            passed=passed,
            flagged_count=flagged_count,
            escalated=escalated,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        db.add(qr)
        db.commit()
        db.refresh(qr)
        return qr
    finally:
        db.close()


# ---------------------------------------------------------------------------
# CrewAI mock helper
# ---------------------------------------------------------------------------

def _mock_crew(monkeypatch, response: str) -> None:
    class FakeTask:
        def __init__(self, **kwargs): pass

    class FakeCrew:
        def __init__(self, **kwargs): pass
        def kickoff(self): return response

    monkeypatch.setattr(_rg_mod, "Task", FakeTask)
    monkeypatch.setattr(_rg_mod, "Crew", FakeCrew)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mgr(monkeypatch, session_factory):
    monkeypatch.setattr(_rg_mod, "get_llm", lambda *a, **kw: object())

    class FakeAgent:
        def __init__(self, **kwargs): pass

    monkeypatch.setattr(_rg_mod, "Agent", FakeAgent)
    return ReportGenerator(session_factory=session_factory)


# ===========================================================================
# Tests
# ===========================================================================


class TestEmploymentProfileModel:
    def test_all_optional_defaults_none(self):
        p = EmploymentProfile()
        assert p.employment_status is None
        assert p.job_title is None
        assert p.isco_code is None
        assert p.isco_confidence is None
        assert p.industry is None
        assert p.hours_per_week is None
        assert p.employment_type is None

    def test_fields_populated(self):
        p = EmploymentProfile(
            employment_status="employed",
            job_title="nurse",
            isco_code="2221",
            isco_confidence=0.88,
            industry="healthcare",
            hours_per_week="40",
            employment_type="full_time",
        )
        assert p.employment_status == "employed"
        assert p.isco_confidence == 0.88

    def test_isco_confidence_bounds(self):
        with pytest.raises(Exception):
            EmploymentProfile(isco_confidence=1.5)


class TestSurveyReportModel:
    def test_required_fields(self):
        r = SurveyReport(
            report_id=1,
            session_id=1,
            language="en",
            profile=EmploymentProfile(),
            flagged_count=0,
            report_en="English report.",
            report_ar="تقرير عربي.",
            recommendations_en="No follow-up.",
            recommendations_ar="لا متابعة.",
            generated_at="2026-02-28T00:00:00",
        )
        assert r.report_id == 1
        assert r.language == "en"
        assert r.quality_score is None

    def test_optional_quality_fields_default_none(self):
        r = SurveyReport(
            report_id=1, session_id=1, language="en",
            profile=EmploymentProfile(), flagged_count=0,
            report_en="x", report_ar="x",
            recommendations_en="x", recommendations_ar="x",
            generated_at="2026-02-28T00:00:00",
        )
        assert r.quality_score is None
        assert r.quality_status is None


class TestBuildProfile:
    def test_empty_responses_returns_all_none(self, mgr):
        profile = mgr._build_profile([])
        assert profile.employment_status is None
        assert profile.job_title is None
        assert profile.isco_code is None

    def test_employment_status_extracted(self, session_factory, mgr):
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        _make_response(session_factory, s.id, "employment_status", "employed")
        db = session_factory()
        responses = db.query(SurveyResponse).filter(SurveyResponse.session_id == s.id).all()
        db.close()
        profile = mgr._build_profile(responses)
        assert profile.employment_status == "employed"

    def test_job_title_extracted(self, session_factory, mgr):
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        _make_response(session_factory, s.id, "job_title", "software engineer")
        db = session_factory()
        responses = db.query(SurveyResponse).filter(SurveyResponse.session_id == s.id).all()
        db.close()
        profile = mgr._build_profile(responses)
        assert profile.job_title == "software engineer"

    def test_best_isco_by_confidence(self, session_factory, mgr):
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        _make_response(session_factory, s.id, "job_title", "nurse", "2221", 0.70)
        _make_response(session_factory, s.id, "job_title", "nurse specialist", "2222", 0.92)
        db = session_factory()
        responses = db.query(SurveyResponse).filter(SurveyResponse.session_id == s.id).all()
        db.close()
        profile = mgr._build_profile(responses)
        assert profile.isco_code == "2222"
        assert profile.isco_confidence == pytest.approx(0.92)

    def test_no_isco_when_no_code_in_responses(self, session_factory, mgr):
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        _make_response(session_factory, s.id, "job_title", "teacher")  # no isco
        db = session_factory()
        responses = db.query(SurveyResponse).filter(SurveyResponse.session_id == s.id).all()
        db.close()
        profile = mgr._build_profile(responses)
        assert profile.isco_code is None
        assert profile.isco_confidence is None

    def test_latest_answer_wins_for_duplicate_field(self, session_factory, mgr):
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        _make_response(session_factory, s.id, "employment_status", "unemployed")
        _make_response(session_factory, s.id, "employment_status", "employed")
        db = session_factory()
        responses = db.query(SurveyResponse).filter(SurveyResponse.session_id == s.id).all()
        db.close()
        profile = mgr._build_profile(responses)
        assert profile.employment_status == "employed"

    def test_industry_extracted(self, session_factory, mgr):
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        _make_response(session_factory, s.id, "industry", "healthcare")
        db = session_factory()
        responses = db.query(SurveyResponse).filter(SurveyResponse.session_id == s.id).all()
        db.close()
        profile = mgr._build_profile(responses)
        assert profile.industry == "healthcare"

    def test_hours_per_week_extracted(self, session_factory, mgr):
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        _make_response(session_factory, s.id, "hours_per_week", "40")
        db = session_factory()
        responses = db.query(SurveyResponse).filter(SurveyResponse.session_id == s.id).all()
        db.close()
        profile = mgr._build_profile(responses)
        assert profile.hours_per_week == "40"

    def test_employment_type_extracted(self, session_factory, mgr):
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        _make_response(session_factory, s.id, "employment_type", "full_time")
        db = session_factory()
        responses = db.query(SurveyResponse).filter(SurveyResponse.session_id == s.id).all()
        db.close()
        profile = mgr._build_profile(responses)
        assert profile.employment_type == "full_time"


class TestGenerateErrors:
    def test_raises_for_unknown_session(self, mgr):
        with pytest.raises(ValueError, match="not found"):
            mgr.generate(session_id=999999)

    def test_raises_for_in_progress_session(self, session_factory, mgr):
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id, status="in_progress")
        with pytest.raises(ValueError, match="not completed"):
            mgr.generate(session_id=s.id)


class TestGenerateHappyPath:
    def test_returns_survey_report(self, monkeypatch, session_factory, mgr):
        _mock_crew(monkeypatch, json.dumps({
            "report_en": "Good report.", "report_ar": "تقرير جيد.",
            "recommendations_en": "None.", "recommendations_ar": "لا شيء.",
        }))
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        _make_response(session_factory, s.id, "employment_status", "employed")
        result = mgr.generate(session_id=s.id)
        assert isinstance(result, SurveyReport)
        assert result.session_id == s.id

    def test_report_persisted_to_db(self, monkeypatch, session_factory, mgr):
        _mock_crew(monkeypatch, json.dumps({
            "report_en": "EN.", "report_ar": "AR.",
            "recommendations_en": "Rec EN.", "recommendations_ar": "Rec AR.",
        }))
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        mgr.generate(session_id=s.id)
        db = session_factory()
        record = db.query(SurveyReportRecord).filter(
            SurveyReportRecord.session_id == s.id
        ).first()
        db.close()
        assert record is not None
        assert record.report_en == "EN."

    def test_cached_report_returned_on_second_call(self, monkeypatch, session_factory, mgr):
        call_count = {"n": 0}

        class CountingCrew:
            def __init__(self, **kwargs): pass
            def kickoff(self):
                call_count["n"] += 1
                return json.dumps({
                    "report_en": "First.", "report_ar": "أول.",
                    "recommendations_en": "R.", "recommendations_ar": "ر.",
                })

        monkeypatch.setattr(_rg_mod, "Task", lambda **kw: object())
        monkeypatch.setattr(_rg_mod, "Crew", CountingCrew)

        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        mgr.generate(session_id=s.id)
        mgr.generate(session_id=s.id)        # second call — should use cache
        assert call_count["n"] == 1           # LLM only called once

    def test_regenerate_forces_new_llm_call(self, monkeypatch, session_factory, mgr):
        call_count = {"n": 0}

        class CountingCrew:
            def __init__(self, **kwargs): pass
            def kickoff(self):
                call_count["n"] += 1
                return json.dumps({
                    "report_en": "New.", "report_ar": "جديد.",
                    "recommendations_en": "R.", "recommendations_ar": "ر.",
                })

        monkeypatch.setattr(_rg_mod, "Task", lambda **kw: object())
        monkeypatch.setattr(_rg_mod, "Crew", CountingCrew)

        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        mgr.generate(session_id=s.id)
        mgr.generate(session_id=s.id, regenerate=True)
        assert call_count["n"] == 2

    def test_report_en_content(self, monkeypatch, session_factory, mgr):
        _mock_crew(monkeypatch, json.dumps({
            "report_en": "Nurse employed full-time.",
            "report_ar": "ممرضة موظفة بدوام كامل.",
            "recommendations_en": "None.", "recommendations_ar": "لا شيء.",
        }))
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        r = mgr.generate(session_id=s.id)
        assert r.report_en == "Nurse employed full-time."
        assert r.report_ar == "ممرضة موظفة بدوام كامل."

    def test_language_preserved(self, monkeypatch, session_factory, mgr):
        _mock_crew(monkeypatch, json.dumps({
            "report_en": "x", "report_ar": "x",
            "recommendations_en": "x", "recommendations_ar": "x",
        }))
        db = session_factory()
        u = _make_user(session_factory)
        s_ar = SurveySession(user_id=u.id, language="ar", status="completed")
        db.add(s_ar)
        db.commit()
        db.refresh(s_ar)
        db.close()
        r = mgr.generate(session_id=s_ar.id)
        assert r.language == "ar"


class TestEnrichmentPersistence:
    """
    Regression tests for a real bug: semantic_coherence, isic_classification,
    and isced_classification were computed correctly on first generation but
    never written to survey_report_records, so any later cache-hit read of an
    already-generated report (regenerate=False, the default) came back with
    those three fields silently null even though the underlying data was
    present and classifiable. Reproduced live against a real account (session
    457: industry="government", education_level="bachelor",
    field_of_study="Engineering", job_title -> ISCO 2511) before being fixed.
    """

    def _make_classifiable_session(self, session_factory):
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        _make_response(session_factory, s.id, "employment_status", "employed")
        _make_response(session_factory, s.id, "industry", "government")
        _make_response(session_factory, s.id, "education_level", "bachelor")
        _make_response(session_factory, s.id, "field_of_study", "Engineering")
        _make_response(
            session_factory, s.id, "job_title", "Engineering",
            isco_code="2511", confidence_score=0.80,
        )
        return s

    def test_enrichment_populated_on_fresh_generation(self, monkeypatch, session_factory, mgr):
        _mock_crew(monkeypatch, json.dumps({
            "report_en": "EN.", "report_ar": "AR.",
            "recommendations_en": "R.", "recommendations_ar": "ر.",
        }))
        s = self._make_classifiable_session(session_factory)
        r = mgr.generate(session_id=s.id)
        assert r.isic_classification is not None
        assert r.isic_classification["section"] == "O"
        assert r.isced_classification is not None
        assert r.isced_classification["level"] == 6
        assert r.semantic_coherence is not None

    def test_enrichment_survives_cache_hit(self, monkeypatch, session_factory, mgr):
        _mock_crew(monkeypatch, json.dumps({
            "report_en": "EN.", "report_ar": "AR.",
            "recommendations_en": "R.", "recommendations_ar": "ر.",
        }))
        s = self._make_classifiable_session(session_factory)
        mgr.generate(session_id=s.id)                     # fresh generation
        r2 = mgr.generate(session_id=s.id)                 # cache-hit path
        assert r2.isic_classification is not None
        assert r2.isic_classification["section"] == "O"
        assert r2.isced_classification is not None
        assert r2.isced_classification["level"] == 6
        assert r2.semantic_coherence is not None

    def test_enrichment_persisted_to_db_columns(self, monkeypatch, session_factory, mgr):
        _mock_crew(monkeypatch, json.dumps({
            "report_en": "EN.", "report_ar": "AR.",
            "recommendations_en": "R.", "recommendations_ar": "ر.",
        }))
        s = self._make_classifiable_session(session_factory)
        mgr.generate(session_id=s.id)
        db = session_factory()
        record = db.query(SurveyReportRecord).filter(
            SurveyReportRecord.session_id == s.id
        ).first()
        db.close()
        assert record.isic_classification_json is not None
        assert record.isced_classification_json is not None
        assert record.semantic_coherence_json is not None

    def test_no_enrichment_when_no_isco_code(self, monkeypatch, session_factory, mgr):
        _mock_crew(monkeypatch, json.dumps({
            "report_en": "EN.", "report_ar": "AR.",
            "recommendations_en": "R.", "recommendations_ar": "ر.",
        }))
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        _make_response(session_factory, s.id, "employment_status", "employed")  # no job_title/isco
        r = mgr.generate(session_id=s.id)
        assert r.isic_classification is None
        assert r.isced_classification is None
        assert r.semantic_coherence is None


class TestGenerateWithQuality:
    def test_quality_score_included(self, monkeypatch, session_factory, mgr):
        _mock_crew(monkeypatch, json.dumps({
            "report_en": "EN.", "report_ar": "AR.",
            "recommendations_en": "R.", "recommendations_ar": "ر.",
        }))
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        _make_quality_review(session_factory, s.id, passed=True, quality_score=0.85)
        r = mgr.generate(session_id=s.id)
        assert r.quality_score == pytest.approx(0.85)
        assert r.quality_status == "pass"

    def test_failed_quality_status(self, monkeypatch, session_factory, mgr):
        _mock_crew(monkeypatch, json.dumps({
            "report_en": "EN.", "report_ar": "AR.",
            "recommendations_en": "R.", "recommendations_ar": "ر.",
        }))
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        _make_quality_review(session_factory, s.id, passed=False, quality_score=0.55)
        r = mgr.generate(session_id=s.id)
        assert r.quality_status == "fail"

    def test_escalated_quality_status(self, monkeypatch, session_factory, mgr):
        _mock_crew(monkeypatch, json.dumps({
            "report_en": "EN.", "report_ar": "AR.",
            "recommendations_en": "R.", "recommendations_ar": "ر.",
        }))
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        _make_quality_review(session_factory, s.id, passed=False, escalated=True, quality_score=0.30)
        r = mgr.generate(session_id=s.id)
        assert r.quality_status == "escalated"

    def test_flagged_count_included(self, monkeypatch, session_factory, mgr):
        _mock_crew(monkeypatch, json.dumps({
            "report_en": "EN.", "report_ar": "AR.",
            "recommendations_en": "R.", "recommendations_ar": "ر.",
        }))
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        _make_quality_review(session_factory, s.id, flagged_count=3)
        r = mgr.generate(session_id=s.id)
        assert r.flagged_count == 3


class TestParseNarrative:
    def test_clean_json_parsed(self, mgr):
        raw = json.dumps({
            "report_en": "English.", "report_ar": "عربي.",
            "recommendations_en": "None.", "recommendations_ar": "لا شيء.",
        })
        result = mgr._parse_narrative(raw, quality=None)
        assert result["report_en"] == "English."
        assert result["report_ar"] == "عربي."

    def test_markdown_fences_stripped(self, mgr):
        raw = "```json\n" + json.dumps({
            "report_en": "Fenced.", "report_ar": "محاط.",
            "recommendations_en": "R.", "recommendations_ar": "ر.",
        }) + "\n```"
        result = mgr._parse_narrative(raw, quality=None)
        assert result["report_en"] == "Fenced."

    def test_broken_json_falls_back(self, mgr):
        result = mgr._parse_narrative("not json at all {{{}}", quality=None)
        assert "report_en" in result
        assert len(result["report_en"]) > 0

    def test_missing_keys_falls_back(self, mgr):
        raw = json.dumps({"report_en": "Only English."})
        result = mgr._parse_narrative(raw, quality=None)
        # Missing report_ar → fallback used
        assert "report_ar" in result

    def test_pass_status_fallback(self, mgr):
        qr = QualityReview(
            session_id=1, quality_score=0.90, passed=True,
            flagged_count=0, escalated=False,
            created_at=datetime.utcnow(),
        )
        result = mgr._parse_narrative("INVALID", quality=qr)
        assert "satisfactory" in result["recommendations_en"].lower()

    def test_escalated_status_fallback(self, mgr):
        qr = QualityReview(
            session_id=1, quality_score=0.30, passed=False,
            flagged_count=4, escalated=True,
            created_at=datetime.utcnow(),
        )
        result = mgr._parse_narrative("INVALID", quality=qr)
        assert "supervisor" in result["recommendations_en"].lower() or \
               "escalate" in result["recommendations_en"].lower()

    def test_fail_status_fallback(self, mgr):
        qr = QualityReview(
            session_id=1, quality_score=0.55, passed=False,
            flagged_count=2, escalated=False,
            created_at=datetime.utcnow(),
        )
        result = mgr._parse_narrative("INVALID", quality=qr)
        assert "flagged" in result["recommendations_en"].lower() or \
               "review" in result["recommendations_en"].lower()


class TestGetSessionReport:
    def test_returns_none_for_unknown_session(self, mgr):
        assert mgr.get_session_report(session_id=888888) is None

    def test_returns_none_when_no_report_generated(self, session_factory, mgr):
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        assert mgr.get_session_report(session_id=s.id) is None

    def test_returns_report_after_generation(self, monkeypatch, session_factory, mgr):
        _mock_crew(monkeypatch, json.dumps({
            "report_en": "Cached.", "report_ar": "مخزن.",
            "recommendations_en": "R.", "recommendations_ar": "ر.",
        }))
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        mgr.generate(session_id=s.id)
        report = mgr.get_session_report(session_id=s.id)
        assert report is not None
        assert isinstance(report, SurveyReport)

    def test_returns_most_recent_report(self, monkeypatch, session_factory, mgr):
        _mock_crew(monkeypatch, json.dumps({
            "report_en": "Latest.", "report_ar": "أحدث.",
            "recommendations_en": "R.", "recommendations_ar": "ر.",
        }))
        u = _make_user(session_factory)
        s = _make_session(session_factory, u.id)
        mgr.generate(session_id=s.id)
        mgr.generate(session_id=s.id, regenerate=True)
        report = mgr.get_session_report(session_id=s.id)
        assert report.report_en == "Latest."


class TestFallbackTemplates:
    def test_pass_fallback_has_no_follow_up(self, mgr):
        result = mgr._build_fallback("pass")
        assert "satisfactory" in result["recommendations_en"].lower() or \
               "no" in result["recommendations_en"].lower()

    def test_fail_fallback_mentions_review(self, mgr):
        result = mgr._build_fallback("fail")
        assert "review" in result["recommendations_en"].lower() or \
               "flagged" in result["recommendations_en"].lower()

    def test_escalated_fallback_mentions_supervisor(self, mgr):
        result = mgr._build_fallback("escalated")
        assert "supervisor" in result["recommendations_en"].lower() or \
               "escalate" in result["recommendations_en"].lower()

    def test_unknown_fallback_keys_present(self, mgr):
        result = mgr._build_fallback("unknown")
        assert all(k in result for k in (
            "report_en", "report_ar", "recommendations_en", "recommendations_ar"
        ))

    def test_arabic_text_present_in_all_statuses(self, mgr):
        for status in ("pass", "fail", "escalated", "unknown"):
            result = mgr._build_fallback(status)
            assert len(result["report_ar"]) > 10


class TestGetReportGeneratorSingleton:
    def setup_method(self):
        _rg_mod._instance = None

    def teardown_method(self):
        _rg_mod._instance = None

    def test_returns_report_generator_instance(self):
        _rg_mod._instance = ReportGenerator.__new__(ReportGenerator)
        result = get_report_generator()
        assert result is _rg_mod._instance

    def test_same_instance_on_repeated_calls(self):
        _rg_mod._instance = ReportGenerator.__new__(ReportGenerator)
        assert get_report_generator() is get_report_generator()

    def test_instance_is_none_before_first_call(self):
        assert _rg_mod._instance is None
