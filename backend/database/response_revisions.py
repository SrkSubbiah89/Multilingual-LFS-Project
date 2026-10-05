"""Active answer revisions and report cache invalidation."""
from datetime import datetime

from backend.database.models import HITLQueue, SurveyReportRecord, SurveyResponse


def active_responses(db, session_id):
    return db.query(SurveyResponse).filter(
        SurveyResponse.session_id == session_id,
        SurveyResponse.deleted_at.is_(None),
    ).order_by(SurveyResponse.id.asc()).all()


def invalidate_reports(db, session_id):
    db.query(SurveyReportRecord).filter(
        SurveyReportRecord.session_id == session_id,
        SurveyReportRecord.invalidated_at.is_(None),
    ).update({SurveyReportRecord.invalidated_at: datetime.utcnow()}, synchronize_session="fetch")


def retire_response(db, row):
    row.deleted_at = datetime.utcnow()
    for item in db.query(HITLQueue).filter(
        HITLQueue.response_id == row.id, HITLQueue.status == "pending",
    ).all():
        if item.status != "pending":
            continue
        item.status = "rejected"
        item.reviewer_notes = "Classification superseded by a newer answer revision."
        item.reviewed_at = row.deleted_at


def save_response_revision(db, session_id, question_id, answer, *,
                           isco_code=None, confidence_score=None):
    """Keep one active revision, preserving replaced answers and decisions."""
    # Production sessions disable autoflush; include newly queued records.
    db.flush()
    previous = db.query(SurveyResponse).filter(
        SurveyResponse.session_id == session_id,
        SurveyResponse.question_id == question_id,
        SurveyResponse.deleted_at.is_(None),
    ).order_by(SurveyResponse.id.asc()).all()
    latest = previous[-1] if previous else None
    if latest and latest.answer == str(answer) and latest.isco_code == isco_code and latest.confidence_score == confidence_score:
        return latest
    for row in previous:
        retire_response(db, row)
    row = SurveyResponse(
        session_id=session_id, question_id=question_id, answer=str(answer),
        isco_code=isco_code, confidence_score=confidence_score,
        supersedes_id=latest.id if latest else None,
    )
    db.add(row)
    db.flush()
    invalidate_reports(db, session_id)
    return row
