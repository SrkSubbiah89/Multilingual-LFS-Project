"""
backend/agents/person_register.py

Person Register pre-fill service for the LFS survey.

Purpose
-------
When a respondent starts a new survey session the system looks up the
PersonRegister table for their most recent active record.  Known fields are
returned as a ``PreFillData`` object which ConversationManager injects into
the initial ConversationContext so the agent can skip (or just confirm)
already-known answers — targeting the 40-50% question-reduction goal from
the implementation gap analysis.

Integration
-----------
Called from ``survey_routes.py`` at session creation time::

    from backend.agents.person_register import PersonRegisterService
    pre_fill = PersonRegisterService().get_prefill(user_id, db)
    # pre_fill.as_collected_data() → dict suitable for ctx.collected_data

The ConversationManager FSM picks up pre-filled collected_data and
advances directly to the first *unanswered* question.

Usage
-----
from backend.agents.person_register import PersonRegisterService

svc     = PersonRegisterService()
prefill = svc.get_prefill(user_id=42, db=db_session)
if prefill:
    print(prefill.fields_available)          # ["employment_status", "job_title"]
    print(prefill.as_collected_data())       # {"employment_status": "employed", ...}
    print(prefill.reduction_pct)             # 0.4  (40% question reduction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session

from backend.database.models import PersonRegister

log = logging.getLogger(__name__)

# Ordered survey fields — same order as ConversationManager._extract_fields
_SURVEY_FIELDS = [
    "employment_status",
    "job_title",
    "industry",
    "hours_per_week",
    "employment_type",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PreFillData:
    """Pre-filled survey answers from the Person Register."""

    user_id:            int
    reference_period:   str
    employment_status:  Optional[str]   = None
    job_title:          Optional[str]   = None
    industry:           Optional[str]   = None
    hours_per_week:     Optional[float] = None
    employment_type:    Optional[str]   = None
    isco_code:          Optional[str]   = None
    isic_code:          Optional[str]   = None
    isced_level:        Optional[int]   = None
    source:             str             = "survey_round"
    prefilled_fields:   list[str]       = field(default_factory=list)

    @property
    def fields_available(self) -> list[str]:
        """Ordered list of survey fields that have a pre-filled value."""
        return [f for f in _SURVEY_FIELDS if getattr(self, f, None) is not None]

    @property
    def reduction_pct(self) -> float:
        """Estimated fraction of survey questions that can be skipped."""
        n = len(self.fields_available)
        return round(n / len(_SURVEY_FIELDS), 2)

    def as_collected_data(self) -> dict:
        """
        Return a ``collected_data`` dict that ConversationManager can merge
        into a new ConversationContext.  Only non-None survey fields are
        included.  Values are stringified so they match the FSM's raw-text
        extraction results.
        """
        data: dict = {}
        for f in _SURVEY_FIELDS:
            val = getattr(self, f, None)
            if val is not None:
                data[f] = str(val) if not isinstance(val, str) else val
        return data


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class PersonRegisterService:
    """Retrieve and apply Person Register pre-fill data for a survey session."""

    def get_prefill(self, user_id: int, db: Session) -> Optional[PreFillData]:
        """
        Return the most recent active PersonRegister record for *user_id*, or
        ``None`` if no record exists.

        Parameters
        ----------
        user_id:
            Primary key of the authenticated user.
        db:
            SQLAlchemy session (injected by FastAPI Depends).
        """
        record: Optional[PersonRegister] = (
            db.query(PersonRegister)
            .filter(
                PersonRegister.user_id == user_id,
                PersonRegister.is_active.is_(True),
            )
            .order_by(PersonRegister.updated_at.desc())
            .first()
        )

        if record is None:
            return None

        prefill = PreFillData(
            user_id=record.user_id,
            reference_period=record.reference_period,
            employment_status=record.employment_status,
            job_title=record.job_title,
            industry=record.industry,
            hours_per_week=record.hours_per_week,
            employment_type=record.employment_type,
            isco_code=record.isco_code,
            isic_code=record.isic_code,
            isced_level=record.isced_level,
            source=record.source or "survey_round",
        )

        log.info(
            "Person Register pre-fill for user %d: %d/%d fields available "
            "(~%.0f%% question reduction) from period %s",
            user_id,
            len(prefill.fields_available),
            len(_SURVEY_FIELDS),
            prefill.reduction_pct * 100,
            record.reference_period,
        )

        return prefill

    def update_from_session(
        self,
        user_id: int,
        reference_period: str,
        collected_data: dict,
        isco_code: Optional[str],
        isic_code: Optional[str],
        isced_level: Optional[int],
        db: Session,
    ) -> PersonRegister:
        """
        Upsert a PersonRegister row from a completed survey session.

        Called by survey_routes at session completion to keep the register
        current for the next survey round.
        """
        record = (
            db.query(PersonRegister)
            .filter(
                PersonRegister.user_id == user_id,
                PersonRegister.reference_period == reference_period,
            )
            .first()
        )

        if record is None:
            record = PersonRegister(
                user_id=user_id,
                reference_period=reference_period,
                source="survey_round",
            )
            db.add(record)

        record.employment_status = collected_data.get("employment_status")
        record.job_title         = collected_data.get("job_title")
        record.industry          = collected_data.get("industry")
        record.employment_type   = collected_data.get("employment_type")

        hours_raw = collected_data.get("hours_per_week")
        if hours_raw is not None:
            try:
                record.hours_per_week = float(hours_raw)
            except (ValueError, TypeError):
                pass

        if isco_code:
            record.isco_code = isco_code
        if isic_code:
            record.isic_code = isic_code
        if isced_level is not None:
            record.isced_level = isced_level

        record.is_active = True
        db.commit()
        db.refresh(record)

        log.info(
            "Person Register updated for user %d / period %s: %s",
            user_id,
            reference_period,
            {k: getattr(record, k) for k in _SURVEY_FIELDS},
        )
        return record
