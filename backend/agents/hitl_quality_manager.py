"""
backend/agents/hitl_quality_manager.py

Human-in-the-loop (HITL) quality manager for the LFS survey system.

Responsibilities
----------------
1. Real-time quality monitoring — assess every survey session's ISCO
   classifications, coverage, and confidence scores as soon as a session
   is ready for review.

2. Low-confidence flagging — any SurveyResponse whose ISCO confidence
   falls below the threshold (default 0.60) or that carries no ISCO code
   at all is flagged for human attention.

3. Quality score calculation — an overall score in [0, 1] combines average
   ISCO confidence, ISCO coverage, and the proportion of low-confidence
   responses into a single numeric indicator.

4. Escalation — sessions whose quality score drops below the escalation
   threshold (default 0.50) or that accumulate three or more flagged items
   are persisted as a ``QualityReview`` row with ``escalated=True``, ready
   for a human supervisor.

5. Bilingual quality reports — Claude 3.5 Sonnet (TaskType.CRITICAL)
   writes English + Arabic pass/fail summaries; deterministic string
   templates serve as a fallback when the LLM is unavailable.

Quality score formula
---------------------
  avg_conf  = mean ISCO confidence of responses that carry a score
  isco_cov  = fraction of responses that carry any ISCO code
  low_ratio = fraction of scored responses below the confidence threshold

  quality_score = (
      0.50 × avg_conf
    + 0.30 × isco_cov
    + 0.20 × (1 − low_ratio)
  )

Status thresholds
-----------------
  quality_score ≥ 0.70                  → PASS
  quality_score ≥ 0.50 and < 0.70       → FAIL
  quality_score < 0.50 or ≥ 3 flags     → ESCALATED (human review required)

Usage
-----
from backend.agents.hitl_quality_manager import HITLQualityManager

mgr    = HITLQualityManager()
report = mgr.review_session(session_id=42)
print(report.status)         # ReviewStatus.PASS | FAIL | ESCALATED
print(report.quality_score)  # e.g. 0.82
print(report.report_en)      # "Session 42 passed quality review …"

pending = mgr.get_pending_reviews()
for p in pending:
    mgr.resolve_review(
        p.review_id,
        reviewer_notes="Reviewed manually — classification confirmed.",
        reviewed_by=1,
    )
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from crewai import Agent, Crew, Task
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database.connection import SessionLocal
from backend.database.models import QualityReview, SurveyResponse
from backend.llm import TaskType, get_llm

load_dotenv()


# ---------------------------------------------------------------------------
# Thresholds and score weights
# ---------------------------------------------------------------------------

_LOW_CONFIDENCE_THRESHOLD    = 0.60   # ISCO confidence below this → flag
_MIN_PASS_QUALITY_SCORE      = 0.70   # quality score below this → FAIL
_ESCALATION_THRESHOLD        = 0.50   # quality score below this → ESCALATE
_MAX_FLAGS_BEFORE_ESCALATION = 3      # ≥ this many flagged items → ESCALATE

_WEIGHT_CONFIDENCE       = 0.50   # weight for avg_confidence in score
_WEIGHT_COVERAGE         = 0.30   # weight for isco_coverage
_WEIGHT_LOW_CONF_PENALTY = 0.20   # weight for (1 − low_conf_ratio)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ReviewStatus(str, Enum):
    """Overall outcome of a session quality review."""

    PASS      = "pass"       # quality score ≥ 0.70, few flags
    FAIL      = "fail"       # quality score 0.50–0.69
    ESCALATED = "escalated"  # quality score < 0.50 or too many flags


class FlagReason(str, Enum):
    """Reason a response was flagged for human review."""

    LOW_CONFIDENCE_ISCO = "low_confidence_isco"   # score < threshold
    MISSING_ISCO        = "missing_isco"           # no ISCO code at all


# ---------------------------------------------------------------------------
# Output models (Pydantic)
# ---------------------------------------------------------------------------

class FlaggedItem(BaseModel):
    """One response that requires human attention."""

    response_id: int
    question_id: str
    reason:      FlagReason
    detail:      str
    confidence:  Optional[float] = None


class QualityMetrics(BaseModel):
    """Aggregated quality signals for one survey session."""

    session_id:           int
    total_responses:      int
    responses_with_isco:  int                           # carry a non-empty ISCO code
    responses_with_score: int                           # carry a confidence_score
    avg_confidence:       float = Field(ge=0.0, le=1.0) # mean of available scores
    isco_coverage:        float = Field(ge=0.0, le=1.0) # with_isco / total
    low_confidence_count: int                           # scores < threshold
    missing_isco_count:   int                           # total − with_isco
    flagged_count:        int                           # len(flagged_items)


class QualityReport(BaseModel):
    """Full quality assessment returned by ``review_session()``."""

    review_id:         int
    session_id:        int
    quality_score:     float = Field(ge=0.0, le=1.0)
    status:            ReviewStatus
    metrics:           QualityMetrics
    flagged_items:     list[FlaggedItem]
    escalated:         bool
    escalation_reason: Optional[str]  = None
    report_en:         str
    report_ar:         str
    generated_at:      str


class PendingReview(BaseModel):
    """Lightweight view of an escalated, unresolved ``QualityReview`` row."""

    review_id:         int
    session_id:        int
    quality_score:     float
    passed:            bool
    escalated:         bool
    escalation_reason: Optional[str]  = None
    flagged_count:     int
    created_at:        str
    reviewed_at:       Optional[str]  = None
    reviewer_notes:    Optional[str]  = None


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_REPORT_INSTRUCTIONS = """\
You are a Labour Force Survey (LFS) Quality Assurance Manager at a national \
statistics office. Produce a concise bilingual quality assessment report for \
one survey session.

Return ONLY a valid JSON object — no markdown fences, no extra text:
{
  "report_en": "<two to three sentence quality report in English>",
  "report_ar": "<two to three sentence quality report in Arabic>"
}

Each report must:
  1. State the overall status (PASS / FAIL / ESCALATED) and quality score.
  2. Mention the ISCO coverage and average classification confidence.
  3. If flagged items exist, briefly describe the nature of the flags.
  4. State whether human review is required and, if so, why.

Arabic must be correct Modern Standard Arabic (فصحى).
Be factual. Do not invent figures not present in the statistics provided.
"""


# ---------------------------------------------------------------------------
# Deterministic fallback report templates
# ---------------------------------------------------------------------------

_FALLBACK_PASS_EN = (
    "Session {session_id} passed quality review with a score of {score:.0%}. "
    "{total} response(s) assessed; ISCO coverage {coverage:.0%}, "
    "average confidence {avg_conf:.0%}. "
    "No human review is required."
)
_FALLBACK_PASS_AR = (
    "اجتازت الجلسة {session_id} مراجعة الجودة بدرجة {score:.0%}. "
    "تم تقييم {total} إجابة؛ تغطية التصنيف المهني {coverage:.0%}، "
    "متوسط الثقة {avg_conf:.0%}. "
    "لا تلزم مراجعة بشرية."
)
_FALLBACK_FAIL_EN = (
    "Session {session_id} failed quality review with a score of {score:.0%}. "
    "{total} response(s) assessed; {flagged} flagged "
    "(ISCO coverage {coverage:.0%}, avg confidence {avg_conf:.0%}). "
    "Flagged responses must be reviewed before session data is accepted."
)
_FALLBACK_FAIL_AR = (
    "فشلت الجلسة {session_id} في مراجعة الجودة بدرجة {score:.0%}. "
    "تم تقييم {total} إجابة؛ تم وضع علامة على {flagged} منها "
    "(تغطية التصنيف المهني {coverage:.0%}، متوسط الثقة {avg_conf:.0%}). "
    "يجب مراجعة الإجابات المُعلَّمة قبل قبول بيانات الجلسة."
)
_FALLBACK_ESCALATED_EN = (
    "Session {session_id} has been escalated for supervisor review "
    "(quality score: {score:.0%}). "
    "{total} response(s) assessed; {flagged} flagged "
    "(ISCO coverage {coverage:.0%}, avg confidence {avg_conf:.0%}). "
    "Immediate human review is required."
)
_FALLBACK_ESCALATED_AR = (
    "تم تصعيد الجلسة {session_id} لمراجعة المشرف "
    "(درجة الجودة: {score:.0%}). "
    "تم تقييم {total} إجابة؛ تم وضع علامة على {flagged} منها "
    "(تغطية التصنيف المهني {coverage:.0%}، متوسط الثقة {avg_conf:.0%}). "
    "تلزم المراجعة البشرية الفورية."
)

_FALLBACK_TEMPLATES: dict[ReviewStatus, tuple[str, str]] = {
    ReviewStatus.PASS:      (_FALLBACK_PASS_EN,      _FALLBACK_PASS_AR),
    ReviewStatus.FAIL:      (_FALLBACK_FAIL_EN,      _FALLBACK_FAIL_AR),
    ReviewStatus.ESCALATED: (_FALLBACK_ESCALATED_EN, _FALLBACK_ESCALATED_AR),
}


# ---------------------------------------------------------------------------
# HITLQualityManager
# ---------------------------------------------------------------------------

class HITLQualityManager:
    """
    Human-in-the-loop quality manager for LFS survey sessions.

    Two-stage design
    ----------------
    Stage 1 — Deterministic metrics (always runs, no API calls):
        Load SurveyResponse rows → compute ISCO coverage, average confidence,
        flag low-confidence / missing-ISCO responses → derive quality score
        and status.

    Stage 2 — Claude 3.5 Sonnet (TaskType.CRITICAL) generates bilingual
        quality report prose.  Falls back to template strings on failure.

    Parameters
    ----------
    session_factory : callable | None
        Zero-argument callable that returns a SQLAlchemy ``Session``.
        Defaults to ``SessionLocal``.  Inject a test factory to avoid
        touching PostgreSQL in unit tests.
    low_confidence_threshold : float
        Confidence floor below which an ISCO classification is flagged
        (default ``0.60``).
    """

    def __init__(
        self,
        session_factory:          Optional[Callable[[], Session]] = None,
        low_confidence_threshold: float = _LOW_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._sf               = session_factory or SessionLocal
        self._low_conf_thresh  = low_confidence_threshold
        self._llm              = get_llm(TaskType.CRITICAL)   # Claude 3.5 Sonnet

        self._agent = Agent(
            role="LFS Survey Quality Assurance Manager",
            goal=(
                "Assess the quality of Labour Force Survey sessions by examining "
                "ISCO classification confidence scores and coverage, identifying "
                "responses that require human review, and producing bilingual "
                "pass/fail quality reports that guide supervisors."
            ),
            backstory=(
                "You are a senior quality assurance manager at a national statistics "
                "office. You oversee data quality for the Labour Force Survey and are "
                "responsible for flagging unreliable AI classifications, escalating "
                "problematic sessions to human supervisors, and ensuring that every "
                "published data point meets the office's accuracy standards."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review_session(self, session_id: int) -> QualityReport:
        """
        Run a full quality review for one survey session.

        Loads all ``SurveyResponse`` rows, computes quality metrics, flags
        low-confidence ISCO classifications and missing codes, derives the
        overall quality score and pass/fail/escalated status, persists a
        ``QualityReview`` row, and produces a bilingual report.

        Parameters
        ----------
        session_id : int
            Primary key of the ``SurveySession`` to review.

        Returns
        -------
        QualityReport
            Complete quality assessment including score, status, flagged
            items, metrics, and bilingual prose report.
        """
        responses = self._load_responses(session_id)
        flagged   = self._flag_items(responses)
        metrics   = self._compute_metrics(session_id, responses, flagged)
        score     = self._compute_quality_score(metrics)
        status    = self._determine_status(score, flagged)
        reason    = (
            self._escalation_reason(score, flagged)
            if status == ReviewStatus.ESCALATED
            else None
        )

        try:
            report_en, report_ar = self._generate_report_text(
                metrics, flagged, status, score
            )
        except Exception:
            report_en, report_ar = self._build_fallback_report(
                metrics, flagged, status, score
            )

        row = self._write_review(
            session_id=session_id,
            quality_score=score,
            passed=(status == ReviewStatus.PASS),
            flagged=flagged,
            escalated=(status == ReviewStatus.ESCALATED),
            escalation_reason=reason,
        )

        return QualityReport(
            review_id=row.id,
            session_id=session_id,
            quality_score=score,
            status=status,
            metrics=metrics,
            flagged_items=flagged,
            escalated=(status == ReviewStatus.ESCALATED),
            escalation_reason=reason,
            report_en=report_en,
            report_ar=report_ar,
            generated_at=datetime.utcnow().isoformat(),
        )

    def get_pending_reviews(self) -> list[PendingReview]:
        """
        Return all escalated ``QualityReview`` rows that have not yet been
        resolved by a human supervisor.

        Results are ordered oldest-first so the most time-critical items
        appear at the top.

        Returns
        -------
        list[PendingReview]
            Unresolved escalations awaiting human action.
        """
        db = self._sf()
        try:
            rows = (
                db.query(QualityReview)
                .filter(
                    QualityReview.escalated   == True,   # noqa: E712
                    QualityReview.reviewed_at == None,   # noqa: E711
                )
                .order_by(QualityReview.created_at.asc())
                .all()
            )
            return [self._to_pending(r) for r in rows]
        finally:
            db.close()

    def resolve_review(
        self,
        review_id:      int,
        reviewer_notes: str,
        reviewed_by:    Optional[int] = None,
    ) -> PendingReview:
        """
        Mark a ``QualityReview`` as resolved by a human supervisor.

        Parameters
        ----------
        review_id : int
            Primary key of the ``QualityReview`` row.
        reviewer_notes : str
            Mandatory supervisor notes (kept for audit trail).
        reviewed_by : int | None
            User ID of the supervisor, if available.

        Returns
        -------
        PendingReview
            Updated review record.

        Raises
        ------
        ValueError
            If no ``QualityReview`` with the given ID exists.
        """
        db = self._sf()
        try:
            row = (
                db.query(QualityReview)
                .filter(QualityReview.id == review_id)
                .first()
            )
            if row is None:
                raise ValueError(f"QualityReview id={review_id} not found.")
            row.reviewed_at    = datetime.utcnow()
            row.reviewer_notes = reviewer_notes
            row.reviewed_by    = reviewed_by
            db.commit()
            db.refresh(row)
            return self._to_pending(row)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Private — data loading
    # ------------------------------------------------------------------

    def _load_responses(self, session_id: int) -> list:
        """Return all SurveyResponse rows for a session, ordered by id."""
        db = self._sf()
        try:
            return (
                db.query(SurveyResponse)
                .filter(SurveyResponse.session_id == session_id)
                .order_by(SurveyResponse.id.asc())
                .all()
            )
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Private — quality computation
    # ------------------------------------------------------------------

    def _flag_items(self, responses: list) -> list[FlaggedItem]:
        """
        Inspect each response and return items that need human attention.

        Flagging rules
        --------------
        * No ISCO code               → ``FlagReason.MISSING_ISCO``
        * Confidence < threshold     → ``FlagReason.LOW_CONFIDENCE_ISCO``
        * ISCO code present but no confidence score → not flagged
        """
        flagged: list[FlaggedItem] = []
        for r in responses:
            has_isco = bool(r.isco_code and str(r.isco_code).strip())
            if not has_isco:
                flagged.append(FlaggedItem(
                    response_id=r.id,
                    question_id=r.question_id,
                    reason=FlagReason.MISSING_ISCO,
                    detail=(
                        f"Response '{r.question_id}' has no ISCO classification."
                    ),
                    confidence=r.confidence_score,
                ))
            elif (
                r.confidence_score is not None
                and r.confidence_score < self._low_conf_thresh
            ):
                flagged.append(FlaggedItem(
                    response_id=r.id,
                    question_id=r.question_id,
                    reason=FlagReason.LOW_CONFIDENCE_ISCO,
                    detail=(
                        f"ISCO confidence {r.confidence_score:.2f} is below "
                        f"the threshold of {self._low_conf_thresh:.2f}."
                    ),
                    confidence=r.confidence_score,
                ))
        return flagged

    def _compute_metrics(
        self,
        session_id: int,
        responses:  list,
        flagged:    list[FlaggedItem],
    ) -> QualityMetrics:
        """Aggregate response-level quality signals into a ``QualityMetrics`` object."""
        total     = len(responses)
        with_isco = sum(
            1 for r in responses
            if r.isco_code and str(r.isco_code).strip()
        )
        scores    = [
            r.confidence_score for r in responses
            if r.confidence_score is not None
        ]
        avg_conf  = sum(scores) / len(scores) if scores else 0.0
        isco_cov  = with_isco / total if total > 0 else 0.0
        low_conf  = sum(1 for s in scores if s < self._low_conf_thresh)
        missing   = total - with_isco

        return QualityMetrics(
            session_id=session_id,
            total_responses=total,
            responses_with_isco=with_isco,
            responses_with_score=len(scores),
            avg_confidence=round(min(max(avg_conf, 0.0), 1.0), 4),
            isco_coverage=round(min(max(isco_cov, 0.0), 1.0), 4),
            low_confidence_count=low_conf,
            missing_isco_count=missing,
            flagged_count=len(flagged),
        )

    def _compute_quality_score(self, metrics: QualityMetrics) -> float:
        """
        Derive the overall session quality score.

        Returns 0.0 immediately when the session has no responses.

        Formula
        -------
        low_ratio = low_confidence_count / max(responses_with_score, 1)
        score     = (
            0.50 × avg_confidence
          + 0.30 × isco_coverage
          + 0.20 × (1 − low_ratio)
        )
        """
        if metrics.total_responses == 0:
            return 0.0
        low_ratio = metrics.low_confidence_count / max(metrics.responses_with_score, 1)
        score = (
            _WEIGHT_CONFIDENCE       * metrics.avg_confidence
            + _WEIGHT_COVERAGE       * metrics.isco_coverage
            + _WEIGHT_LOW_CONF_PENALTY * (1.0 - low_ratio)
        )
        return round(min(max(score, 0.0), 1.0), 4)

    def _determine_status(
        self, quality_score: float, flagged: list[FlaggedItem]
    ) -> ReviewStatus:
        """Map ``(quality_score, flagged_count)`` to a ``ReviewStatus``."""
        if (
            quality_score < _ESCALATION_THRESHOLD
            or len(flagged) >= _MAX_FLAGS_BEFORE_ESCALATION
        ):
            return ReviewStatus.ESCALATED
        if quality_score < _MIN_PASS_QUALITY_SCORE:
            return ReviewStatus.FAIL
        return ReviewStatus.PASS

    @staticmethod
    def _escalation_reason(
        quality_score: float, flagged: list[FlaggedItem]
    ) -> str:
        """Produce a one-sentence human-readable escalation rationale."""
        score_low  = quality_score < _ESCALATION_THRESHOLD
        flags_high = len(flagged) >= _MAX_FLAGS_BEFORE_ESCALATION
        if score_low and flags_high:
            return (
                f"Quality score {quality_score:.2f} is below the escalation "
                f"threshold ({_ESCALATION_THRESHOLD}) and {len(flagged)} "
                f"flagged item(s) meet or exceed the limit "
                f"({_MAX_FLAGS_BEFORE_ESCALATION})."
            )
        if score_low:
            return (
                f"Quality score {quality_score:.2f} is below the escalation "
                f"threshold of {_ESCALATION_THRESHOLD}."
            )
        return (
            f"{len(flagged)} flagged item(s) meet or exceed the escalation "
            f"limit of {_MAX_FLAGS_BEFORE_ESCALATION}."
        )

    # ------------------------------------------------------------------
    # Private — DB persistence
    # ------------------------------------------------------------------

    def _write_review(
        self,
        session_id:        int,
        quality_score:     float,
        passed:            bool,
        flagged:           list[FlaggedItem],
        escalated:         bool,
        escalation_reason: Optional[str],
    ) -> QualityReview:
        """Persist a ``QualityReview`` row and return it with a populated ``id``."""
        items_json = json.dumps([item.model_dump() for item in flagged])
        row = QualityReview(
            session_id=session_id,
            quality_score=quality_score,
            passed=passed,
            flagged_count=len(flagged),
            flagged_items_json=items_json,
            escalated=escalated,
            escalation_reason=escalation_reason,
            created_at=datetime.utcnow(),
        )
        db = self._sf()
        try:
            db.add(row)
            db.commit()
            db.refresh(row)
            return row
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Private — report generation
    # ------------------------------------------------------------------

    def _generate_report_text(
        self,
        metrics: QualityMetrics,
        flagged: list[FlaggedItem],
        status:  ReviewStatus,
        score:   float,
    ) -> tuple[str, str]:
        """Ask Claude 3.5 Sonnet to generate the bilingual quality report."""
        flag_block = (
            "\n".join(
                f"  - {f.reason.value}: {f.detail}"
                for f in flagged
            )
            or "  (none)"
        )
        task = Task(
            description=(
                f"{_REPORT_INSTRUCTIONS}\n\n"
                f"Session ID         : {metrics.session_id}\n"
                f"Status             : {status.value.upper()}\n"
                f"Quality score      : {score:.0%}\n"
                f"Total responses    : {metrics.total_responses}\n"
                f"ISCO coverage      : {metrics.isco_coverage:.0%}\n"
                f"Average confidence : {metrics.avg_confidence:.0%}\n"
                f"Low-confidence hits: {metrics.low_confidence_count}\n"
                f"Missing ISCO codes : {metrics.missing_isco_count}\n"
                f"Flagged items      :\n{flag_block}"
            ),
            expected_output='JSON: {"report_en": "<string>", "report_ar": "<string>"}',
            agent=self._agent,
        )

        crew = Crew(agents=[self._agent], tasks=[task], verbose=False)
        raw  = str(crew.kickoff()).strip()
        return self._parse_report_response(raw, metrics, flagged, status, score)

    def _parse_report_response(
        self,
        raw:     str,
        metrics: QualityMetrics,
        flagged: list[FlaggedItem],
        status:  ReviewStatus,
        score:   float,
    ) -> tuple[str, str]:
        """Parse the LLM JSON; fall back to deterministic templates on failure."""
        clean = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL
        ).strip()

        data: dict = {}
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", clean, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        en = str(data.get("report_en", "")).strip()
        ar = str(data.get("report_ar", "")).strip()

        if not en or not ar:
            en, ar = self._build_fallback_report(metrics, flagged, status, score)
        return en, ar

    @staticmethod
    def _build_fallback_report(
        metrics: QualityMetrics,
        flagged: list[FlaggedItem],
        status:  ReviewStatus,
        score:   float,
    ) -> tuple[str, str]:
        """Produce a deterministic bilingual report from pre-defined templates."""
        tmpl_en, tmpl_ar = _FALLBACK_TEMPLATES[status]
        ctx = dict(
            session_id=metrics.session_id,
            score=score,
            total=metrics.total_responses,
            flagged=len(flagged),
            coverage=metrics.isco_coverage,
            avg_conf=metrics.avg_confidence,
        )
        return tmpl_en.format(**ctx), tmpl_ar.format(**ctx)

    # ------------------------------------------------------------------
    # Private — ORM-to-Pydantic converter
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pending(row: QualityReview) -> PendingReview:
        return PendingReview(
            review_id=row.id,
            session_id=row.session_id,
            quality_score=row.quality_score,
            passed=row.passed,
            escalated=row.escalated,
            escalation_reason=row.escalation_reason,
            flagged_count=row.flagged_count,
            created_at=row.created_at.isoformat(),
            reviewed_at=row.reviewed_at.isoformat() if row.reviewed_at else None,
            reviewer_notes=row.reviewer_notes,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[HITLQualityManager] = None


def get_hitl_quality_manager() -> HITLQualityManager:
    """
    Return the module-level ``HITLQualityManager`` singleton.

    Creates the instance on the first call (connects to PostgreSQL and
    loads the LLM client + CrewAI agent).  Subsequent calls return the
    cached instance.
    """
    global _instance
    if _instance is None:
        _instance = HITLQualityManager()
    return _instance
