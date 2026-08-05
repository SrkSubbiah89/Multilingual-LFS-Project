"""
Tests for backend/agents/person_register.py

Uses SQLite in-memory via the same test-session pattern as test_audit_logger.
No Redis, no Qdrant, no Ollama required.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.connection import Base
from backend.database.models import User, PersonRegister
from backend.agents.person_register import (
    PersonRegisterService,
    PreFillData,
    _SURVEY_FIELDS,
)

# ---------------------------------------------------------------------------
# In-memory SQLite fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    return eng


@pytest.fixture
def db(engine):
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def user(db):
    u = User(email="test@example.com")
    db.add(u)
    db.commit()
    db.refresh(u)
    yield u
    db.query(PersonRegister).filter(PersonRegister.user_id == u.id).delete()
    db.delete(u)
    db.commit()


@pytest.fixture
def svc():
    return PersonRegisterService()


# ---------------------------------------------------------------------------
# 1. get_prefill — no record
# ---------------------------------------------------------------------------

def test_no_record_returns_none(svc, db, user):
    result = svc.get_prefill(user.id, db)
    assert result is None


# ---------------------------------------------------------------------------
# 2. update_from_session + get_prefill roundtrip
# ---------------------------------------------------------------------------

def test_update_then_prefill_roundtrip(svc, db, user):
    collected = {
        "employment_status": "employed",
        "job_title":         "Software Developer",
        "industry":          "Technology",
        "hours_per_week":    "40",
        "employment_type":   "full_time",
    }
    svc.update_from_session(
        user_id=user.id,
        reference_period="2025-Q1",
        collected_data=collected,
        isco_code="2512",
        isic_code="62",
        isced_level=6,
        db=db,
    )

    prefill = svc.get_prefill(user.id, db)
    assert prefill is not None
    assert prefill.employment_status == "employed"
    assert prefill.job_title == "Software Developer"
    assert prefill.industry == "Technology"
    assert prefill.hours_per_week == 40.0
    assert prefill.employment_type == "full_time"
    assert prefill.isco_code == "2512"
    assert prefill.isic_code == "62"
    assert prefill.isced_level == 6


def test_as_collected_data_has_all_fields(svc, db, user):
    collected = {
        "employment_status": "employed",
        "job_title":         "Nurse",
        "industry":          "Healthcare",
        "hours_per_week":    "36",
        "employment_type":   "part_time",
    }
    svc.update_from_session(
        user_id=user.id,
        reference_period="2025-Q2",
        collected_data=collected,
        isco_code="2221",
        isic_code="86",
        isced_level=6,
        db=db,
    )

    prefill = svc.get_prefill(user.id, db)
    data = prefill.as_collected_data()
    for field in _SURVEY_FIELDS:
        assert field in data


# ---------------------------------------------------------------------------
# 3. PreFillData properties
# ---------------------------------------------------------------------------

def test_fields_available_returns_non_none_fields():
    pf = PreFillData(
        user_id=1, reference_period="2025-Q1",
        employment_status="employed", job_title="Engineer",
    )
    assert "employment_status" in pf.fields_available
    assert "job_title" in pf.fields_available
    assert "industry" not in pf.fields_available


def test_reduction_pct_full_prefill():
    pf = PreFillData(
        user_id=1, reference_period="2025-Q1",
        employment_status="employed",
        job_title="Accountant",
        industry="Finance",
        hours_per_week=40.0,
        employment_type="full_time",
    )
    assert pf.reduction_pct == 1.0


def test_reduction_pct_partial_prefill():
    pf = PreFillData(
        user_id=1, reference_period="2025-Q1",
        employment_status="employed",
        job_title="Teacher",
    )
    # 2 out of 5 fields
    assert abs(pf.reduction_pct - 0.4) < 0.01


def test_reduction_pct_empty():
    pf = PreFillData(user_id=1, reference_period="2025-Q1")
    assert pf.reduction_pct == 0.0


def test_as_collected_data_excludes_none_fields():
    pf = PreFillData(
        user_id=1, reference_period="2025-Q1",
        employment_status="unemployed",
    )
    data = pf.as_collected_data()
    assert "employment_status" in data
    assert "job_title" not in data


# ---------------------------------------------------------------------------
# 4. Upsert behaviour
# ---------------------------------------------------------------------------

def test_second_update_overwrites_first(svc, db, user):
    for title in ("Cook", "Chef"):
        svc.update_from_session(
            user_id=user.id,
            reference_period="2025-Q3",
            collected_data={"job_title": title, "employment_status": "employed",
                            "industry": "Food", "hours_per_week": "40",
                            "employment_type": "full_time"},
            isco_code="5120",
            isic_code="56",
            isced_level=3,
            db=db,
        )

    prefill = svc.get_prefill(user.id, db)
    assert prefill.job_title == "Chef"


def test_inactive_record_not_returned(svc, db, user):
    record = PersonRegister(
        user_id=user.id,
        reference_period="2020-Q1",
        employment_status="employed",
        job_title="Old Job",
        is_active=False,
    )
    db.add(record)
    db.commit()

    # Should not return the inactive record (assuming no active record for this user)
    # Clean slate: deactivate all
    db.query(PersonRegister).filter(
        PersonRegister.user_id == user.id
    ).update({"is_active": False})
    db.commit()

    result = svc.get_prefill(user.id, db)
    assert result is None
