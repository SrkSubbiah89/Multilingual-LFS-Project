"""
backend/agents/report_generator.py

Bilingual LFS survey report generator.

Responsibilities
----------------
1. Profile extraction  — Aggregates all SurveyResponse rows for a completed
   session into a structured EmploymentProfile (employment status, job title,
   best ISCO classification, industry, hours, employment type).

2. Narrative generation — A CrewAI agent (Claude 3.5 Sonnet) writes a
   professional 3-paragraph report in both English and Arabic, plus
   data-collector recommendations tailored to the quality assessment result.

3. Persistence          — The generated report is written to the
   ``survey_report_records`` PostgreSQL table and served from cache on
   subsequent calls (pass ``regenerate=True`` to force a new LLM run).

LLM
---
Claude 3.5 Sonnet (TaskType.CRITICAL) — final output that will be read by
national statistics office analysts; accuracy and bilingual quality matter.

Usage
-----
from backend.agents.report_generator import ReportGenerator

gen = ReportGenerator()

# Generate (or return cached) report for a completed session
report = gen.generate(session_id=42)
print(report.profile.employment_status)   # "employed"
print(report.report_en[:100])             # first 100 chars of English report
print(report.recommendations_ar)          # Arabic collector notes

# Read back without regenerating
existing = gen.get_session_report(session_id=42)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Callable, Optional

_logger = logging.getLogger(__name__)

from crewai import Agent, Crew, Task
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database.connection import SessionLocal
from backend.database.models import (
    QualityReview,
    SurveyReportRecord,
    SurveyResponse,
    SurveySession,
)
from backend.llm import TaskType, get_llm

load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Question IDs that map directly to UAE LFS employment fields (all paths)
_FIELD_QUESTION_IDS = {
    # All paths
    "employment_status", "education_level",
    "ai_preference", "data_confidence",
    # Employed path
    "employment_nature", "employment_sector",
    "job_title", "job_duties", "industry",
    "hours_per_week", "employment_type", "monthly_wage_range",
    # Unemployed path
    "job_search_active", "available_for_work", "unemployment_duration",
    "last_job_title", "reason_left_job",
    # Outside LF path
    "outside_lf_reason",
}


# ---------------------------------------------------------------------------
# Fallback report templates (bilingual, keyed by quality status)
# ---------------------------------------------------------------------------

_FALLBACK_REPORT_EN: dict[str, str] = {
    "pass": (
        "The respondent has successfully completed the Labour Force Survey. "
        "Employment data has been collected and validated. "
        "The ISCO-08 classification was assigned based on the reported occupation. "
        "All required fields are present and consistent."
    ),
    "fail": (
        "The respondent completed the Labour Force Survey; however, some responses "
        "require attention. One or more data quality checks did not pass. "
        "The employment profile has been recorded with the available information. "
        "Please review the flagged items before including this record in analysis."
    ),
    "escalated": (
        "This survey session has been escalated for human review. "
        "Significant data quality issues were detected that prevent automated validation. "
        "The employment profile is incomplete or contains inconsistent responses. "
        "A supervisor must review this session before the data can be used."
    ),
    "unknown": (
        "The Labour Force Survey session has been completed. "
        "Employment data has been recorded as provided by the respondent. "
        "Quality assessment was not available for this session."
    ),
}

_FALLBACK_REPORT_AR: dict[str, str] = {
    "pass": (
        "أتمّ المشارك استبيان مسح القوى العاملة بنجاح. "
        "تم جمع بيانات التوظيف والتحقق منها. "
        "تم تعيين تصنيف ISCO-08 بناءً على المهنة المُبلَّغ عنها. "
        "جميع الحقول المطلوبة موجودة ومتسقة."
    ),
    "fail": (
        "أتمّ المشارك استبيان مسح القوى العاملة، غير أن بعض الإجابات تستدعي المراجعة. "
        "لم تجتز إحدى فحوصات جودة البيانات أو أكثر. "
        "تم تسجيل ملف التوظيف بالمعلومات المتاحة. "
        "يُرجى مراجعة البنود المُحدَّدة قبل إدراج هذا السجل في التحليل."
    ),
    "escalated": (
        "تمت إحالة هذه الجلسة للمراجعة البشرية. "
        "رُصدت مشكلات جسيمة في جودة البيانات تحول دون إجراء التحقق الآلي. "
        "ملف التوظيف غير مكتمل أو يحتوي على إجابات متضاربة. "
        "يجب على المشرف مراجعة هذه الجلسة قبل استخدام البيانات."
    ),
    "unknown": (
        "اكتملت جلسة استبيان مسح القوى العاملة. "
        "تم تسجيل بيانات التوظيف كما أفاد بها المشارك. "
        "لم يكن تقييم الجودة متاحًا لهذه الجلسة."
    ),
}

_FALLBACK_REC_EN: dict[str, str] = {
    "pass": "No immediate follow-up required. Data quality is satisfactory.",
    "fail": (
        "• Review flagged responses before using this record in analysis.\n"
        "• Consider re-contacting the respondent to clarify inconsistent answers."
    ),
    "escalated": (
        "• Escalate to a supervisor for manual review before releasing this data.\n"
        "• Do not include this session in statistical outputs until resolved.\n"
        "• Document the escalation reason in the case management system."
    ),
    "unknown": "Verify data completeness before including in analysis outputs.",
}

_FALLBACK_REC_AR: dict[str, str] = {
    "pass": "لا يلزم أي متابعة فورية. جودة البيانات مُرضية.",
    "fail": (
        "• راجع الإجابات المُحدَّدة قبل استخدام هذا السجل في التحليل.\n"
        "• فكر في إعادة التواصل مع المشارك لتوضيح الإجابات غير المتسقة."
    ),
    "escalated": (
        "• أحل الأمر إلى مشرف للمراجعة اليدوية قبل نشر هذه البيانات.\n"
        "• لا تُدرج هذه الجلسة في المخرجات الإحصائية حتى يتم حل المشكلة.\n"
        "• وثّق سبب الإحالة في نظام إدارة الحالات."
    ),
    "unknown": "تحقق من اكتمال البيانات قبل إدراجها في مخرجات التحليل.",
}


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_REPORT_PROMPT = """\
You are an expert Labour Force Survey (LFS) data analyst generating the
official final report for one respondent's completed survey session.
This survey follows ILO ICLS-19 standards.

Employment profile collected:
  Employment status   : {employment_status}
  Education level     : {education_level}
  Field of study      : {field_of_study}
  Employment nature   : {employment_nature}
  Employment sector   : {employment_sector}
  Job title           : {job_title}
  Industry / sector   : {industry}
  ISCO-08 code        : {isco_code}
  ISCO confidence     : {isco_confidence}
  Hours per week      : {hours_per_week}
  Employment type     : {employment_type}
  Monthly wage range  : {monthly_wage_range}
  Job search active   : {job_search_active}
  Available for work  : {available_for_work}
  Unemployment dur.   : {unemployment_duration}
  Last job title      : {last_job_title}
  Reason left job     : {reason_left_job}
  Outside LF reason   : {outside_lf_reason}
  AI preference       : {ai_preference}
  Data confidence     : {data_confidence}

Data quality assessment:
  Score   : {quality_score}
  Status  : {quality_status}
  Flagged : {flagged_count} response(s)

Return ONLY a valid JSON object — no markdown fences, no extra text:
{{
  "report_en": "<3-paragraph professional English summary. Para 1: respondent \
employment situation and demographics. Para 2: ISCO-08 classification, \
industry, and wage profile. Para 3: data quality evaluation and implications \
for analysis, including respondent feedback on the AI survey experience.>",
  "report_ar": "<same 3-paragraph summary in Modern Standard Arabic (فصحى)>",
  "recommendations_en": "<2-3 concise bullet points for the data collector: \
follow-up actions, clarifications needed, or quality notes. \
Write 'No follow-up required.' if quality is high and data is complete.>",
  "recommendations_ar": "<same recommendations in Arabic>"
}}
"""


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class EmploymentProfile(BaseModel):
    """Extracted employment data from all responses in a UAE LFS survey session."""

    # ── All paths (Section B + C1 + K) ───────────────────────────────────────
    employment_status:    Optional[str]   = None
    education_level:      Optional[str]   = None   # B5: ISCED 2011 attainment level
    field_of_study:       Optional[str]   = None   # B5a: major field (ISCED-F 2013)
    ai_preference:        Optional[str]   = None   # K3: prefer AI / human / no pref
    data_confidence:      Optional[str]   = None   # K4: confidence in data privacy

    # ── Employed path (Section C + D + E) ────────────────────────────────────
    employment_nature:    Optional[str]   = None   # C3: paid_employee/employer/self_employed/family_worker
    employment_sector:    Optional[str]   = None   # C4: government/private/semi_government/ngo
    job_title:            Optional[str]   = None   # C5
    industry:             Optional[str]   = None   # C6
    isco_code:            Optional[str]   = None   # best ISCO code by confidence
    isco_confidence:      Optional[float] = Field(default=None, ge=0.0, le=1.0)
    hours_per_week:       Optional[str]   = None   # D2: usual hours
    employment_type:      Optional[str]   = None   # D6: full_time/part_time/seasonal/casual
    monthly_wage_range:   Optional[str]   = None   # E1: AED bracket

    # ── Unemployed path (Section F + G) ──────────────────────────────────────
    job_search_active:    Optional[str]   = None   # F1: yes/no
    available_for_work:   Optional[str]   = None   # F3: yes/no
    unemployment_duration: Optional[str]  = None   # F4: duration string
    last_job_title:       Optional[str]   = None   # G1: most recent occupation
    reason_left_job:      Optional[str]   = None   # G3: reason

    # ── Outside labour force path (Section F) ────────────────────────────────
    outside_lf_reason:    Optional[str]   = None   # F6: reason not seeking


class SurveyReport(BaseModel):
    """Complete generated report for one survey session."""

    report_id:          int
    session_id:         int
    language:           str           # "en" | "ar"
    profile:            EmploymentProfile
    quality_score:      Optional[float] = None
    quality_status:     Optional[str]   = None
    flagged_count:      int = 0
    report_en:          str
    report_ar:          str
    recommendations_en: str
    recommendations_ar: str
    generated_at:       str           # ISO 8601 UTC
    semantic_coherence:  Optional[dict] = None   # Cross-standard ISCO↔ISIC↔ISCED coherence
    isic_classification: Optional[dict] = None   # Full ISIC Rev.4 4-level hierarchy
    isced_classification: Optional[dict] = None  # Full ISCED 2011 4-digit programme category


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Generates and persists bilingual LFS survey reports.

    Parameters
    ----------
    session_factory : Callable[[], Session] | None
        SQLAlchemy session factory.  Defaults to the production
        ``SessionLocal``.  Inject a test factory for unit tests.
    context_memory : ContextMemory | None
        Reserved for future context enrichment; unused in current pipeline.
    """

    def __init__(
        self,
        session_factory: Optional[Callable[[], Session]] = None,
        context_memory=None,
    ) -> None:
        self._session_factory = session_factory or SessionLocal
        self._agent_available = False

        try:
            self._llm = get_llm(TaskType.CRITICAL)   # Claude 3.5 Sonnet, temp 0.0
            self._agent = Agent(
                role="LFS Survey Report Specialist",
                goal=(
                    "Generate accurate, professional bilingual (English and Arabic) "
                    "reports for completed Labour Force Survey sessions, summarising "
                    "the respondent's employment profile, ISCO-08 classification, "
                    "and data quality status for national statistics office analysts."
                ),
                backstory=(
                    "You are a senior analyst at a national statistics office with "
                    "deep expertise in the ISCO-08 occupation classification system "
                    "and labour-force survey methodology. You produce clear, precise "
                    "reports in both English and Modern Standard Arabic that help "
                    "data managers assess response quality and prepare final datasets."
                ),
                llm=self._llm,
                verbose=False,
                allow_delegation=False,
            )
            self._agent_available = True
        except Exception as exc:
            _logger.warning(
                "ReportGenerator: LLM unavailable (%s). "
                "Will use template-based fallback reports.",
                exc,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        session_id: int,
        regenerate: bool = False,
    ) -> SurveyReport:
        """
        Generate (or return cached) the final report for a completed session.

        Parameters
        ----------
        session_id : int
            DB primary key of the SurveySession.  Must be completed.
        regenerate : bool
            If True, a new LLM report is always generated and persisted
            even if one already exists.

        Returns
        -------
        SurveyReport

        Raises
        ------
        ValueError
            If the session does not exist or is not yet completed.
        """
        db: Session = self._session_factory()
        try:
            session = db.query(SurveySession).filter(
                SurveySession.id == session_id,
                SurveySession.deleted_at.is_(None),
            ).first()
            if session is None:
                raise ValueError(f"Session {session_id} not found.")
            if session.status != "completed":
                raise ValueError(
                    f"Session {session_id} is not completed "
                    f"(status={session.status!r}). "
                    "Generate a report only after the session is completed."
                )

            # Return cached report unless regeneration is requested
            if not regenerate:
                existing = self._load_existing(db, session_id)
                if existing:
                    return self._record_to_report(existing, session)

            # Load all session data
            responses = (
                db.query(SurveyResponse)
                .filter(SurveyResponse.session_id == session_id)
                .all()
            )
            quality = self._load_quality_review(db, session_id)

            # Build structured profile
            profile = self._build_profile(responses)

            # ── Semantic cross-classification coherence ──────────────────────
            # Computed from the profile's ISCO code, ISIC industry, ISCED level.
            # Must not raise — enrichment never blocks report generation.
            semantic_coherence: Optional[dict] = None
            isic_classification: Optional[dict] = None
            isced_classification: Optional[dict] = None
            if profile.isco_code:
                try:
                    from backend.agents.semantic_relation import get_semantic_relation_engine
                    from backend.agents.isic_classifier import ISICClassifier
                    from backend.agents.isced_classifier import ISCEDClassifier
                    import dataclasses
                    _sr = get_semantic_relation_engine(use_llm=False)
                    # Classify industry → ISIC (full 4-digit)
                    _isic_section: Optional[str] = None
                    if profile.industry:
                        try:
                            _ic = ISICClassifier().classify(profile.industry)
                            _isic_section = _ic.section
                            isic_classification = {
                                "industry_text":  profile.industry,
                                "section":        _ic.section,
                                "section_title":  _ic.section_title,
                                "division_code":  _ic.division_code,
                                "division_title": _ic.division_title,
                                "group_code":     _ic.group_code,
                                "group_title":    _ic.group_title,
                                "class_code":     _ic.class_code,
                                "class_title":    _ic.class_title,
                                "confidence":     _ic.confidence,
                                "method":         _ic.method,
                            }
                        except Exception:
                            pass
                    # Classify education → ISCED (full 4-digit)
                    # Combine level + field_of_study so ISCED-F gets a meaningful subject
                    _isced_level: Optional[int] = None
                    _edu_text = " ".join(filter(None, [profile.education_level, profile.field_of_study]))
                    if _edu_text:
                        try:
                            _ec = ISCEDClassifier().classify(_edu_text)
                            _isced_level = _ec.level
                            isced_classification = {
                                "education_text":  _edu_text,
                                "level":           _ec.level,
                                "level_title":     _ec.level_title,
                                "broad_code":      _ec.broad_code,
                                "broad_title":     _ec.broad_title,
                                "narrow_code":     _ec.narrow_code,
                                "narrow_title":    _ec.narrow_title,
                                "detailed_code":   _ec.detailed_code,
                                "detailed_title":  _ec.detailed_title,
                                "confidence":      _ec.confidence,
                                "method":          _ec.method,
                            }
                        except Exception:
                            pass
                    sc = _sr.analyse(
                        isco_code    = profile.isco_code,
                        isic_section = _isic_section,
                        isced_level  = _isced_level,
                        job_title    = str(profile.job_title or profile.last_job_title or ""),
                        language     = session.language or "en",
                    )
                    semantic_coherence = {
                        k: (
                            [dataclasses.asdict(v) for v in val]
                            if isinstance(val, list) else val
                        )
                        for k, val in dataclasses.asdict(sc).items()
                    }
                except Exception:
                    pass

            # Generate bilingual narrative via LLM (or template fallback)
            if not self._agent_available:
                parsed = self._build_profile_fallback(profile, quality)
            else:
                raw = self._generate_narrative(profile, quality, session.language)
                parsed = self._parse_narrative(raw, quality, profile)

            # Persist and return
            record = self._persist(db, session_id, session.language, profile, quality, parsed)
            report = self._record_to_report(record, session)
            report.semantic_coherence = semantic_coherence
            report.isic_classification = isic_classification
            report.isced_classification = isced_classification
            return report
        finally:
            db.close()

    def get_session_report(self, session_id: int) -> Optional[SurveyReport]:
        """
        Return the most recent report for a session without generating a new one.

        Returns None if no report has been generated yet.
        """
        db: Session = self._session_factory()
        try:
            session = db.query(SurveySession).filter(
                SurveySession.id == session_id,
                SurveySession.deleted_at.is_(None),
            ).first()
            if session is None:
                return None
            existing = self._load_existing(db, session_id)
            if existing is None:
                return None
            # Re-generate with regenerate=True to pick up semantic coherence
            return self.generate(session_id, regenerate=False)
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_existing(
        self, db: Session, session_id: int
    ) -> Optional[SurveyReportRecord]:
        """Return the most recent SurveyReportRecord for this session, or None."""
        return (
            db.query(SurveyReportRecord)
            .filter(SurveyReportRecord.session_id == session_id)
            .order_by(SurveyReportRecord.generated_at.desc())
            .first()
        )

    def _load_quality_review(
        self, db: Session, session_id: int
    ) -> Optional[QualityReview]:
        """Return the most recent QualityReview for this session, or None."""
        return (
            db.query(QualityReview)
            .filter(QualityReview.session_id == session_id)
            .order_by(QualityReview.created_at.desc())
            .first()
        )

    def _build_profile(self, responses: list[SurveyResponse]) -> EmploymentProfile:
        """Aggregate survey responses into a structured EmploymentProfile."""
        by_field: dict[str, list[SurveyResponse]] = {}
        for r in responses:
            by_field.setdefault(r.question_id, []).append(r)

        def _pick(field: str) -> Optional[str]:
            """Return the most recent answer for a field, or None."""
            items = by_field.get(field, [])
            return items[-1].answer if items else None

        # Best ISCO: highest confidence_score; covers both employed (job_title)
        # and unemployed (last_job_title) classification rows.
        isco_responses = [
            r for r in responses
            if r.isco_code and r.confidence_score is not None
        ]
        best = (
            max(isco_responses, key=lambda r: r.confidence_score)
            if isco_responses
            else None
        )

        return EmploymentProfile(
            # All paths
            employment_status=_pick("employment_status"),
            education_level=_pick("education_level"),
            field_of_study=_pick("field_of_study"),
            ai_preference=_pick("ai_preference"),
            data_confidence=_pick("data_confidence"),
            # Employed path
            employment_nature=_pick("employment_nature"),
            employment_sector=_pick("employment_sector"),
            job_title=_pick("job_title"),
            industry=_pick("industry"),
            isco_code=best.isco_code if best else None,
            isco_confidence=best.confidence_score if best else None,
            hours_per_week=_pick("hours_per_week"),
            employment_type=_pick("employment_type"),
            monthly_wage_range=_pick("monthly_wage_range"),
            # Unemployed path
            job_search_active=_pick("job_search_active"),
            available_for_work=_pick("available_for_work"),
            unemployment_duration=_pick("unemployment_duration"),
            last_job_title=_pick("last_job_title"),
            reason_left_job=_pick("reason_left_job"),
            # Outside LF path
            outside_lf_reason=_pick("outside_lf_reason"),
        )

    def _generate_narrative(
        self,
        profile: EmploymentProfile,
        quality: Optional[QualityReview],
        language: str,
    ) -> str:
        """Call the LLM to produce the bilingual JSON narrative."""

        def _fmt(value: Optional[object], fallback: str = "Not provided") -> str:
            if value is None:
                return fallback
            if isinstance(value, float):
                return f"{value:.0%}"
            return str(value)

        if quality is None:
            status_key = "unknown"
        elif quality.escalated:
            status_key = "escalated"
        elif quality.passed:
            status_key = "pass"
        else:
            status_key = "fail"

        # Determine best occupation title (employed=job_title, unemployed=last_job_title)
        occ_title = profile.job_title or profile.last_job_title

        prompt = _REPORT_PROMPT.format(
            employment_status=_fmt(profile.employment_status),
            education_level=_fmt(profile.education_level),
            field_of_study=_fmt(profile.field_of_study),
            employment_nature=_fmt(profile.employment_nature),
            employment_sector=_fmt(profile.employment_sector),
            job_title=_fmt(occ_title),
            industry=_fmt(profile.industry),
            isco_code=_fmt(profile.isco_code, "Not classified"),
            isco_confidence=_fmt(profile.isco_confidence, "N/A"),
            hours_per_week=_fmt(profile.hours_per_week),
            employment_type=_fmt(profile.employment_type),
            monthly_wage_range=_fmt(profile.monthly_wage_range),
            job_search_active=_fmt(profile.job_search_active),
            available_for_work=_fmt(profile.available_for_work),
            unemployment_duration=_fmt(profile.unemployment_duration),
            last_job_title=_fmt(profile.last_job_title),
            reason_left_job=_fmt(profile.reason_left_job),
            outside_lf_reason=_fmt(profile.outside_lf_reason),
            ai_preference=_fmt(profile.ai_preference),
            data_confidence=_fmt(profile.data_confidence),
            quality_score=_fmt(
                quality.quality_score if quality else None, "Not assessed"
            ),
            quality_status=status_key.upper() if quality else "Not assessed",
            flagged_count=quality.flagged_count if quality else 0,
        )

        task = Task(
            description=prompt,
            expected_output=(
                "JSON object with keys: report_en, report_ar, "
                "recommendations_en, recommendations_ar"
            ),
            agent=self._agent,
        )
        crew = Crew(agents=[self._agent], tasks=[task], verbose=False)
        try:
            return str(crew.kickoff()).strip()
        except Exception as exc:
            _logger.warning(
                "ReportGenerator LLM call failed (%s). Using template fallback.", exc
            )
            # Return empty string — _parse_narrative will detect failure and use templates
            return ""

    def _parse_narrative(
        self,
        raw: str,
        quality: Optional[QualityReview],
        profile: Optional[EmploymentProfile] = None,
    ) -> dict:
        """
        Parse the LLM JSON response.  Falls back to profile-enriched templates
        on any parsing failure (including empty string from a failed LLM call).
        """
        # Strip markdown fences
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw, flags=re.DOTALL).strip()

        def _extract(text: str) -> Optional[dict]:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
            return None

        data = _extract(clean)
        if data and all(
            k in data for k in (
                "report_en", "report_ar",
                "recommendations_en", "recommendations_ar",
            )
        ):
            return {
                "report_en":          str(data["report_en"]).strip(),
                "report_ar":          str(data["report_ar"]).strip(),
                "recommendations_en": str(data["recommendations_en"]).strip(),
                "recommendations_ar": str(data["recommendations_ar"]).strip(),
            }

        # Use profile-enriched fallback when profile data is available
        if profile is not None:
            return self._build_profile_fallback(profile, quality)

        status_key = "unknown"
        if quality:
            if quality.escalated:
                status_key = "escalated"
            elif quality.passed:
                status_key = "pass"
            else:
                status_key = "fail"
        return self._build_fallback(status_key)

    def _build_fallback(self, status_key: str) -> dict:
        """Return deterministic bilingual report text for the given quality status."""
        k = status_key if status_key in _FALLBACK_REPORT_EN else "unknown"
        return {
            "report_en":          _FALLBACK_REPORT_EN[k],
            "report_ar":          _FALLBACK_REPORT_AR[k],
            "recommendations_en": _FALLBACK_REC_EN[k],
            "recommendations_ar": _FALLBACK_REC_AR[k],
        }

    def _build_profile_fallback(
        self,
        profile: EmploymentProfile,
        quality: Optional[QualityReview],
    ) -> dict:
        """
        Build a rich bilingual report from the profile data without calling the LLM.
        Used when the LLM is unavailable (no API key, exhausted credits, etc.).
        """
        def _v(val: Optional[str], fallback: str = "not provided") -> str:
            return val.replace("_", " ") if val else fallback

        status_key = "unknown"
        if quality:
            if quality.escalated:
                status_key = "escalated"
            elif quality.passed:
                status_key = "pass"
            else:
                status_key = "fail"

        status_labels_ar = {
            "employed": "موظف", "unemployed": "عاطل عن العمل",
            "not_in_labour_force": "خارج سوق العمل",
        }
        emp_status = profile.employment_status or ""

        # ── English report ────────────────────────────────────────────────────
        lines_en = [f"The respondent is currently {_v(emp_status)}."]

        if profile.education_level:
            _edu_line = _v(profile.education_level)
            if profile.field_of_study:
                _edu_line += f", field of study: {profile.field_of_study}"
            lines_en.append(f"Education: {_edu_line}.")

        # Employed path
        if emp_status == "employed":
            if profile.employment_nature:
                lines_en.append(f"Employment arrangement: {_v(profile.employment_nature)}.")
            if profile.employment_sector:
                lines_en.append(f"Sector: {_v(profile.employment_sector)}.")
            if profile.job_title:
                lines_en.append(f"Job title: {profile.job_title}.")
            if profile.industry:
                lines_en.append(f"Industry: {_v(profile.industry)}.")
            if profile.hours_per_week:
                lines_en.append(
                    f"Usually works {profile.hours_per_week} hours/week "
                    f"({_v(profile.employment_type, 'arrangement not specified')})."
                )
            if profile.monthly_wage_range:
                lines_en.append(f"Monthly salary range: {_v(profile.monthly_wage_range)} AED.")

        # Unemployed path
        elif emp_status == "unemployed":
            if profile.last_job_title and profile.last_job_title != "never_worked":
                lines_en.append(f"Most recent job: {profile.last_job_title}.")
            if profile.reason_left_job:
                lines_en.append(f"Reason for leaving: {_v(profile.reason_left_job)}.")
            if profile.job_search_active:
                lines_en.append(f"Actively seeking work: {profile.job_search_active}.")
            if profile.unemployment_duration:
                lines_en.append(f"Duration of job search: {profile.unemployment_duration}.")
            if profile.available_for_work:
                lines_en.append(f"Available to start within 2 weeks: {profile.available_for_work}.")

        # Outside LF path
        elif emp_status == "not_in_labour_force":
            if profile.outside_lf_reason:
                lines_en.append(f"Reason not seeking work: {_v(profile.outside_lf_reason)}.")

        # ISCO classification (employed + unemployed with prior work experience)
        if profile.isco_code:
            pct = f"{profile.isco_confidence:.0%}" if profile.isco_confidence else "N/A"
            lines_en.append(f"ISCO-08 classification: {profile.isco_code} (confidence {pct}).")
        else:
            lines_en.append("No ISCO-08 classification was assigned for this session.")

        # Feedback
        if profile.ai_preference:
            lines_en.append(f"AI interviewer preference: {_v(profile.ai_preference)}.")
        if profile.data_confidence:
            lines_en.append(f"Data privacy confidence: {_v(profile.data_confidence)}.")

        lines_en.append(_FALLBACK_REPORT_EN.get(status_key, _FALLBACK_REPORT_EN["unknown"]))
        report_en = " ".join(lines_en)

        # ── Arabic report ─────────────────────────────────────────────────────
        employment_ar = status_labels_ar.get(emp_status, _v(emp_status))
        lines_ar = [f"حالة المشارك الوظيفية: {employment_ar}."]

        if profile.education_level:
            _edu_ar = _v(profile.education_level)
            if profile.field_of_study:
                _edu_ar += f"، مجال التخصص: {profile.field_of_study}"
            lines_ar.append(f"المستوى التعليمي: {_edu_ar}.")

        if emp_status == "employed":
            if profile.employment_nature:
                lines_ar.append(f"طبيعة العمل: {_v(profile.employment_nature)}.")
            if profile.employment_sector:
                lines_ar.append(f"القطاع: {_v(profile.employment_sector)}.")
            if profile.job_title:
                lines_ar.append(f"المسمى الوظيفي: {profile.job_title}.")
            if profile.industry:
                lines_ar.append(f"القطاع الاقتصادي: {_v(profile.industry)}.")
            if profile.hours_per_week:
                lines_ar.append(
                    f"يعمل عادةً {profile.hours_per_week} ساعة أسبوعيًا "
                    f"({_v(profile.employment_type, 'غير محدد')})."
                )
            if profile.monthly_wage_range:
                lines_ar.append(f"نطاق الراتب الشهري: {_v(profile.monthly_wage_range)} درهم.")
        elif emp_status == "unemployed":
            if profile.last_job_title and profile.last_job_title != "never_worked":
                lines_ar.append(f"آخر وظيفة: {profile.last_job_title}.")
            if profile.reason_left_job:
                lines_ar.append(f"سبب ترك العمل: {_v(profile.reason_left_job)}.")
            if profile.job_search_active:
                lines_ar.append(f"يبحث عن عمل بنشاط: {profile.job_search_active}.")
            if profile.unemployment_duration:
                lines_ar.append(f"مدة البحث عن عمل: {profile.unemployment_duration}.")
        elif emp_status == "not_in_labour_force":
            if profile.outside_lf_reason:
                lines_ar.append(f"سبب عدم البحث عن عمل: {_v(profile.outside_lf_reason)}.")

        if profile.isco_code:
            pct = f"{profile.isco_confidence:.0%}" if profile.isco_confidence else "غير متاح"
            lines_ar.append(f"تصنيف ISCO-08: {profile.isco_code} (الثقة {pct}).")
        else:
            lines_ar.append("لم يُعيَّن تصنيف ISCO-08 لهذه الجلسة.")

        if profile.ai_preference:
            lines_ar.append(f"تفضيل المحاور: {_v(profile.ai_preference)}.")
        if profile.data_confidence:
            lines_ar.append(f"الثقة بسرية البيانات: {_v(profile.data_confidence)}.")

        lines_ar.append(_FALLBACK_REPORT_AR.get(status_key, _FALLBACK_REPORT_AR["unknown"]))
        report_ar = " ".join(lines_ar)

        return {
            "report_en":          report_en,
            "report_ar":          report_ar,
            "recommendations_en": _FALLBACK_REC_EN.get(status_key, _FALLBACK_REC_EN["unknown"]),
            "recommendations_ar": _FALLBACK_REC_AR.get(status_key, _FALLBACK_REC_AR["unknown"]),
        }

    def _persist(
        self,
        db: Session,
        session_id: int,
        language: str,
        profile: EmploymentProfile,
        quality: Optional[QualityReview],
        parsed: dict,
    ) -> SurveyReportRecord:
        """Write the report to the database and return the new row."""
        record = SurveyReportRecord(
            session_id=session_id,
            language=language,
            profile_json=profile.model_dump_json(),
            quality_score=quality.quality_score if quality else None,
            quality_status=(
                ("escalated" if quality.escalated else ("pass" if quality.passed else "fail"))
                if quality else None
            ),
            flagged_count=quality.flagged_count if quality else 0,
            report_en=parsed["report_en"],
            report_ar=parsed["report_ar"],
            recommendations_en=parsed["recommendations_en"],
            recommendations_ar=parsed["recommendations_ar"],
            generated_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    def _record_to_report(
        self, record: SurveyReportRecord, session: SurveySession
    ) -> SurveyReport:
        """Convert a SurveyReportRecord ORM row into a SurveyReport Pydantic model."""
        try:
            profile = EmploymentProfile.model_validate_json(record.profile_json)
        except Exception:
            profile = EmploymentProfile()

        return SurveyReport(
            report_id=record.id,
            session_id=record.session_id,
            language=session.language,
            profile=profile,
            quality_score=record.quality_score,
            quality_status=record.quality_status,
            flagged_count=record.flagged_count,
            report_en=record.report_en,
            report_ar=record.report_ar,
            recommendations_en=record.recommendations_en,
            recommendations_ar=record.recommendations_ar,
            generated_at=record.generated_at.isoformat(),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[ReportGenerator] = None


def get_report_generator() -> ReportGenerator:
    """Return the process-wide ReportGenerator singleton."""
    global _instance
    if _instance is None:
        _instance = ReportGenerator()
    return _instance
