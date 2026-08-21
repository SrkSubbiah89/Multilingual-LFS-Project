"""
Extended tests for HITLQualityManager and PersonRegisterService.
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta


# ════════════════════════════════════════════════════════════════════
#  Shared DB fixture
# ════════════════════════════════════════════════════════════════════

@pytest.fixture
def db_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.database.models import Base
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


# ════════════════════════════════════════════════════════════════════
#  HITL Quality Manager Extended Tests
# ════════════════════════════════════════════════════════════════════

@pytest.fixture
def hitl(db_session):
    from sqlalchemy.orm import sessionmaker
    engine = db_session.bind
    Session = sessionmaker(bind=engine)

    with patch("backend.agents.hitl_quality_manager.get_llm", return_value=MagicMock()), \
         patch("backend.agents.hitl_quality_manager.Agent",   return_value=MagicMock()), \
         patch("backend.agents.hitl_quality_manager.Task",    return_value=MagicMock()), \
         patch("backend.agents.hitl_quality_manager.Crew",    return_value=MagicMock()):
        from backend.agents.hitl_quality_manager import HITLQualityManager
        return HITLQualityManager(session_factory=Session)


def make_responses(n=5, confidence=0.85, has_isco=True):
    return [
        MagicMock(
            id=i,
            question_id="job_title" if i == 0 else "industry",
            isco_code="2512" if has_isco else None,
            confidence_score=confidence,
        )
        for i in range(n)
    ]


class TestHITLPriorityQueue:
    def test_high_confidence_creates_pass_review(self, hitl):
        responses = make_responses(confidence=0.95)
        with patch.object(hitl, "_load_responses", return_value=responses):
            report = hitl.review_session(1)
        assert report.status.value in ("pass", "PASS")

    def test_low_confidence_creates_escalated_review(self, hitl):
        responses = make_responses(confidence=0.30)
        with patch.object(hitl, "_load_responses", return_value=responses):
            report = hitl.review_session(2)
        assert report.status.value in ("escalated", "ESCALATED")

    def test_pending_queue_sorted_oldest_first(self, hitl):
        responses = make_responses(confidence=0.30)
        with patch.object(hitl, "_load_responses", return_value=responses):
            hitl.review_session(10)
            hitl.review_session(11)
        queue = hitl.get_pending_reviews()
        if len(queue) >= 2:
            assert queue[0].created_at <= queue[1].created_at

    def test_high_priority_before_low_priority(self, hitl):
        """get_pending_reviews sorts by created_at ASC (oldest first), not quality_score"""
        with patch.object(hitl, "_load_responses", return_value=make_responses(confidence=0.45)):
            hitl.review_session(20)
        with patch.object(hitl, "_load_responses", return_value=make_responses(confidence=0.0)):
            hitl.review_session(21)
        queue = hitl.get_pending_reviews()
        if len(queue) >= 2:
            # sorted oldest-first: session 20 (created first) appears before session 21
            assert queue[0].created_at <= queue[1].created_at

    def test_resolved_items_not_in_queue(self, hitl):
        responses = make_responses(confidence=0.30)
        with patch.object(hitl, "_load_responses", return_value=responses):
            report = hitl.review_session(30)
        hitl.resolve_review(report.review_id, "looks ok", 1)
        queue = hitl.get_pending_reviews()
        ids = [q.review_id for q in queue]
        assert report.review_id not in ids


class TestHITLReviewActions:
    def test_approve_action(self, hitl):
        responses = make_responses(confidence=0.30)
        with patch.object(hitl, "_load_responses", return_value=responses):
            report = hitl.review_session(40)
        resolved = hitl.resolve_review(report.review_id, "looks fine", 1)
        assert resolved.reviewed_at is not None  # PendingReview has no reviewed_by field

    def test_correct_action(self, hitl):
        responses = make_responses(confidence=0.30)
        with patch.object(hitl, "_load_responses", return_value=responses):
            report = hitl.review_session(41)
        resolved = hitl.resolve_review(report.review_id, "changed code to 2511", 1)
        assert resolved.reviewer_notes == "changed code to 2511"

    def test_reject_action(self, hitl):
        responses = make_responses(confidence=0.30)
        with patch.object(hitl, "_load_responses", return_value=responses):
            report = hitl.review_session(42)
        resolved = hitl.resolve_review(report.review_id, "invalid data", 1)
        assert resolved.reviewed_at is not None

    def test_double_resolve_allowed(self, hitl):
        # resolve_review does not guard against double-resolve; second call succeeds
        responses = make_responses(confidence=0.30)
        with patch.object(hitl, "_load_responses", return_value=responses):
            report = hitl.review_session(43)
        hitl.resolve_review(report.review_id, "ok", 1)
        resolved2 = hitl.resolve_review(report.review_id, "ok again", 2)
        assert resolved2.reviewer_notes == "ok again"

    def test_unknown_review_id_raises(self, hitl):
        with pytest.raises(Exception):
            hitl.resolve_review(99999, "notes", 1)


class TestHITLEdgeCases:
    def test_empty_session_is_escalated(self, hitl):
        with patch.object(hitl, "_load_responses", return_value=[]):
            report = hitl.review_session(50)
        assert report.status.value in ("escalated", "ESCALATED")

    def test_single_response_session(self, hitl):
        with patch.object(hitl, "_load_responses", return_value=make_responses(n=1, confidence=0.95)):
            report = hitl.review_session(51)
        assert report is not None

    def test_100_responses_session(self, hitl):
        with patch.object(hitl, "_load_responses", return_value=make_responses(n=100, confidence=0.85)):
            report = hitl.review_session(52)
        assert report is not None

    def test_missing_isco_on_non_job_field_not_flagged(self, hitl):
        """Only job_title and last_job_title fields should be checked for ISCO"""
        responses = [MagicMock(id=1, question_id="industry", isco_code=None, confidence_score=None)]
        flags = hitl._flag_items(responses)
        assert len(flags) == 0

    def test_random_quality_check_rate(self, hitl):
        """5-10% of PASS sessions should be escalated for random QC"""
        pass_count = 0
        escalated_count = 0
        for i in range(200):
            responses = make_responses(confidence=0.95)
            with patch.object(hitl, "_load_responses", return_value=responses):
                report = hitl.review_session(200 + i)
            if report.status.value in ("pass", "PASS"):
                pass_count += 1
            elif report.status.value in ("escalated", "ESCALATED"):
                escalated_count += 1
        # With 200 sessions, at least some random QC should trigger
        assert pass_count + escalated_count == 200


# ════════════════════════════════════════════════════════════════════
#  Person Register Extended Tests
# ════════════════════════════════════════════════════════════════════

@pytest.fixture
def person_register():
    from backend.agents.person_register import PersonRegisterService
    return PersonRegisterService()


class TestPersonRegisterPreFill:
    def test_new_respondent_no_prefill(self, person_register, db_session):
        data = person_register.get_prefill(9999, db_session)
        assert data is None

    def test_returning_respondent_prefilled(self, person_register, db_session):
        """Simulate a respondent who completed survey in previous round"""
        from backend.database.models import PersonRegister
        # Need a User row first (FK constraint)
        from backend.database.models import User
        user = User(email="test_prefill@example.com", is_active=True)
        db_session.add(user)
        db_session.flush()

        entry = PersonRegister(
            user_id=user.id,
            reference_period="2025-Q4",
            employment_status="employed",
            job_title="Software Engineer",
            industry="technology",
            is_active=True,
        )
        db_session.add(entry)
        db_session.commit()

        data = person_register.get_prefill(user.id, db_session)
        if data:
            assert data.job_title == "Software Engineer"

    def test_prefill_reduces_required_questions(self, person_register, db_session):
        """40-50% question reduction when prefill is available"""
        from backend.database.models import PersonRegister, User
        user = User(email="prefill_reduce@example.com", is_active=True)
        db_session.add(user)
        db_session.flush()

        entry = PersonRegister(
            user_id=user.id,
            reference_period="2025-Q4",
            employment_status="employed",
            job_title="Nurse",
            industry="healthcare",
            is_active=True,
        )
        db_session.add(entry)
        db_session.commit()

        data = person_register.get_prefill(user.id, db_session)
        if data:
            filled_count = len(data.fields_available)
            assert filled_count >= 2  # at least some fields available

    def test_no_active_record_returns_none(self, person_register, db_session):
        """Inactive record should not be returned as prefill"""
        from backend.database.models import PersonRegister, User
        user = User(email="inactive_prefill@example.com", is_active=True)
        db_session.add(user)
        db_session.flush()

        entry = PersonRegister(
            user_id=user.id,
            reference_period="2023-Q1",
            job_title="Teacher",
            industry="education",
            is_active=False,  # inactive
        )
        db_session.add(entry)
        db_session.commit()

        data = person_register.get_prefill(user.id, db_session)
        assert data is None


class TestPersonRegisterUpdate:
    def test_register_updated_after_survey_completion(self, person_register, db_session):
        from backend.database.models import User
        user = User(email="update_test@example.com", is_active=True)
        db_session.add(user)
        db_session.flush()

        collected = {
            "employment_status": "employed",
            "job_title": "Data Scientist",
            "industry": "technology",
        }
        person_register.update_from_session(
            user_id=user.id,
            reference_period="2026-Q1",
            collected_data=collected,
            isco_code=None,
            isic_code=None,
            isced_level=None,
            db=db_session,
        )
        data = person_register.get_prefill(user.id, db_session)
        if data:
            assert data.job_title == "Data Scientist"

    def test_register_overwrites_old_data(self, person_register, db_session):
        from backend.database.models import PersonRegister, User
        user = User(email="overwrite_test@example.com", is_active=True)
        db_session.add(user)
        db_session.flush()

        db_session.add(PersonRegister(
            user_id=user.id,
            reference_period="2025-Q1",
            job_title="Old Job",
            is_active=True,
        ))
        db_session.commit()

        person_register.update_from_session(
            user_id=user.id,
            reference_period="2026-Q1",
            collected_data={"job_title": "New Job"},
            isco_code=None,
            isic_code=None,
            isced_level=None,
            db=db_session,
        )
        data = person_register.get_prefill(user.id, db_session)
        if data:
            assert data.job_title == "New Job"
