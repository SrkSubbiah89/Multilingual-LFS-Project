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
import re
from datetime import datetime, timezone
from typing import Callable, Optional

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

# Question IDs that map directly to LFS employment fields
_FIELD_QUESTION_IDS = {
    "employment_status",
    "job_title",
    "industry",
    "hours_per_week",
    "employment_type",
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

Employment profile collected:
  Employment status : {employment_status}
  Job title         : {job_title}
  ISCO-08 code      : {isco_code}
  ISCO confidence   : {isco_confidence}
  Industry / sector : {industry}
  Hours per week    : {hours_per_week}
  Employment type   : {employment_type}

Data quality assessment:
  Score   : {quality_score}
  Status  : {quality_status}
  Flagged : {flagged_count} response(s)

Return ONLY a valid JSON object — no markdown fences, no extra text:
{{
  "report_en": "<3-paragraph professional English summary. Para 1: respondent \
employment situation. Para 2: ISCO-08 classification and confidence. \
Para 3: data quality evaluation and implications for analysis.>",
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
    """Extracted employment data from all responses in a survey session."""

    employment_status: Optional[str] = None
    job_title:         Optional[str] = None
    isco_code:         Optional[str] = None      # best ISCO code by confidence
    isco_confidence:   Optional[float] = Field(default=None, ge=0.0, le=1.0)
    industry:          Optional[str] = None
    hours_per_week:    Optional[str] = None
    employment_type:   Optional[str] = None


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
                SurveySession.id == session_id
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

            # Generate bilingual narrative via LLM
            raw = self._generate_narrative(profile, quality, session.language)
            parsed = self._parse_narrative(raw, quality)

            # Persist and return
            record = self._persist(db, session_id, session.language, profile, quality, parsed)
            return self._record_to_report(record, session)
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
                SurveySession.id == session_id
            ).first()
            if session is None:
                return None
            existing = self._load_existing(db, session_id)
            if existing is None:
                return None
            return self._record_to_report(existing, session)
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

        # Best ISCO: highest confidence_score where isco_code is present
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
            employment_status=_pick("employment_status"),
            job_title=_pick("job_title"),
            isco_code=best.isco_code if best else None,
            isco_confidence=best.confidence_score if best else None,
            industry=_pick("industry"),
            hours_per_week=_pick("hours_per_week"),
            employment_type=_pick("employment_type"),
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

        prompt = _REPORT_PROMPT.format(
            employment_status=_fmt(profile.employment_status),
            job_title=_fmt(profile.job_title),
            isco_code=_fmt(profile.isco_code, "Not classified"),
            isco_confidence=_fmt(profile.isco_confidence, "N/A"),
            industry=_fmt(profile.industry),
            hours_per_week=_fmt(profile.hours_per_week),
            employment_type=_fmt(profile.employment_type),
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
        return str(crew.kickoff()).strip()

    def _parse_narrative(
        self,
        raw: str,
        quality: Optional[QualityReview],
    ) -> dict:
        """
        Parse the LLM JSON response.  Falls back to deterministic templates
        on any parsing failure.
        """
        status_key = "unknown"
        if quality:
            if quality.escalated:
                status_key = "escalated"
            elif quality.passed:
                status_key = "pass"
            else:
                status_key = "fail"

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
