"""Serve actual routes on port 8001 against task-owned browser-smoke services.

AI outputs and email delivery are fixtures. PostgreSQL, Redis, authentication,
conversation routing, revisions, authorization and reports use application code.
Never loads .env or opens the respondent database. Stop after browser checks.
"""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PRIVATE = ROOT / "Software/browser_smoke"

from scripts.manage_browser_smoke_services import CONTAINER_PORTS, IMAGES, OWNER_LABEL, inspect_owned


def validate_service_urls(state):
    """Bind both connection URLs to verified task-owned containers before writes."""
    from sqlalchemy.engine import make_url
    from sqlalchemy.exc import ArgumentError
    from urllib.parse import urlsplit

    if (
        state.get("status") != "ready"
        or set(state.get("services", {})) != set(IMAGES)
        or state.get("ownership_label") != OWNER_LABEL
        or not state.get("owner")
    ):
        raise RuntimeError("Complete ready browser-smoke service state is required.")

    ports = {}
    for role, service in state["services"].items():
        record = inspect_owned(service, state["owner"], role)
        if record is None or not record["State"]["Running"]:
            raise RuntimeError("A verified browser-smoke service is not running.")
        binding = record["NetworkSettings"]["Ports"][CONTAINER_PORTS[role]][0]
        ports[role] = int(binding["HostPort"])

    try:
        database = make_url(state["DATABASE_URL"])
        database_valid = (
            database.drivername in ("postgresql", "postgresql+psycopg2")
            and database.host == "127.0.0.1"
            and database.port == ports["postgres"]
            and database.database == "browser_smoke_db"
            and database.username == "browser_smoke_user"
            and not database.query
        )
    except (ArgumentError, KeyError, TypeError, ValueError):
        # SQLAlchemy parsing errors can contain their input URL. Keep private
        # credentials out of terminal tracebacks, including chained exceptions.
        raise RuntimeError("Invalid browser-smoke database URL.") from None
    if not database_valid:
        raise RuntimeError("Database URL does not target the verified browser-smoke container.")

    try:
        storage = urlsplit(state["REDIS_URL"])
        storage_valid = (
            storage.scheme == "redis"
            and storage.hostname == "127.0.0.1"
            and storage.port == ports["redis"]
            and storage.path == "/0"
            and storage.username is None
            and storage.password is None
            and not storage.query
            and not storage.fragment
        )
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("Invalid browser-smoke Redis URL.") from None
    if not storage_valid:
        raise RuntimeError("Redis URL does not target the verified browser-smoke container.")


def main():
    state = json.loads((PRIVATE / "services.json").read_text(encoding="utf-8"))
    validate_service_urls(state)
    import dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False
    os.environ.update({
        "DATABASE_URL": state["DATABASE_URL"], "REDIS_URL": state["REDIS_URL"],
        "JWT_SECRET": "isolated-browser-smoke-secret-at-least-thirty-two-characters",
        "APP_ENV": "development", "LFS_FAST_MODE": "true",
        "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "", "GEMINI_API_KEY": "",
        "GROQ_API_KEY": "", "OPENROUTER_API_KEY": "", "SENDGRID_API_KEY": "",
        "GMAIL_USER": "", "GMAIL_APP_PASSWORD": "", "TWILIO_ACCOUNT_SID": "",
        "TWILIO_AUTH_TOKEN": "", "CREWAI_TRACING_ENABLED": "false", "OTEL_SDK_DISABLED": "true",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "OLLAMA_BASE_URL": "http://127.0.0.1:1", "QDRANT_URL": "http://127.0.0.1:1",
        "CORS_ORIGINS": "http://127.0.0.1:3000,http://localhost:3000",
    })
    from alembic import command
    from alembic.config import Config
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")

    from backend.main import app
    from backend.api import survey_routes as routes
    from backend.auth import email_otp
    from backend.auth.jwt_handler import create_access_token
    from backend.agents.context_memory import ContextMemory
    from backend.agents.conversation_manager import ConversationContext, ConversationManager, ConversationState
    from backend.agents.hitl_quality_manager import HITLQualityManager
    from backend.agents.report_generator import ReportGenerator
    from backend.database.connection import SessionLocal
    from backend.database.models import User, SurveySession, HITLQueue, SurveyReportRecord
    from backend.database.response_revisions import save_response_revision, active_responses
    import redis

    email_otp.generate_otp = lambda: "123456"
    email_otp.send_otp_email = lambda recipient, code: True
    manager = object.__new__(ConversationManager)
    manager._agent_available = False
    manager._llm_extract_correction = lambda *args, **kwargs: False
    memory = object.__new__(ContextMemory)
    memory._redis = redis.Redis.from_url(state["REDIS_URL"], decode_responses=True)
    memory._ttl = 86400
    routes._get_agents = lambda: (manager, MagicMock())
    routes._get_context_memory = lambda: memory
    for getter in ("_get_audit_logger", "_get_emotional_intelligence", "_get_validation_agent", "_get_person_register_svc"):
        setattr(routes, getter, lambda: MagicMock())

    def classify_occupation(text, *args, **kwargs):
        nurse = "nurs" in text.lower()
        return SimpleNamespace(
            primary=SimpleNamespace(code="2221" if nurse else "2512", title_en="Nursing professionals" if nurse else "Software developers",
                                    title_ar="", confidence=0.95 if nurse else 0.4),
            method="browser_fixture", hitl_required=not nurse, stage_confidences={},
            hierarchy_path=[], alternatives=[], reasoning="Deterministic browser fixture; not an accuracy measurement.",
        )
    isco = SimpleNamespace(classify=classify_occupation)
    isic = SimpleNamespace(classify=lambda *args, **kwargs: SimpleNamespace(section="Q", section_title="Human health",
        division_code="86", division_title="Human health", group_code="", group_title="", class_code="", class_title="",
        confidence=0.95, method="browser_fixture"))
    isced = SimpleNamespace(classify=lambda text, **kwargs: SimpleNamespace(level=6 if "bachelor" in text.lower() else 3,
        level_title="Bachelor" if "bachelor" in text.lower() else "Secondary", broad_code="", broad_title="", narrow_code="",
        narrow_title="", detailed_code="", detailed_title="", confidence=0.95, method="browser_fixture"))
    routes._get_isco_classifier = lambda: isco
    routes._get_isic_classifier = lambda: isic
    routes._get_isced_classifier = lambda: isced
    routes._get_nationality_classifier = lambda: SimpleNamespace(classify=lambda *args, **kwargs: SimpleNamespace(method="unknown"))
    import backend.agents.isic_classifier as isic_module
    import backend.agents.isced_classifier as isced_module
    isic_module.ISICClassifier = lambda: isic
    isced_module.ISCEDClassifier = lambda: isced
    quality = object.__new__(HITLQualityManager)
    quality._sf = SessionLocal
    quality._low_conf_thresh = 0.70
    quality._generate_report_text = lambda *args: ("Browser quality fixture", "Browser quality fixture")
    routes._get_hitl_quality_manager = lambda: quality
    reports = object.__new__(ReportGenerator)
    reports._session_factory = SessionLocal
    reports._agent_available = False
    routes.get_report_generator = lambda: reports

    def collected(education="secondary", title="Engineer"):
        data = {"employment_status": "employed", "education_level": education, "employment_nature": "paid_employee",
            "secondary_job": "no", "platform_work": "no", "job_title": title, "job_duties": "I provide care",
            "industry": "healthcare", "nationality": "Indian", "gender": "female", "age_group": "25-34",
            "monthly_wage_range": "under_5000", "hours_per_week": "40"}
        if education == "bachelor":
            data["field_of_study"] = "Medicine"
        for field in manager._get_field_order(data):
            data.setdefault(field, "provided")
        return data

    with SessionLocal() as db:
        if db.query(User).count() != 0:
            raise RuntimeError("Use fresh smoke services; do not reuse respondents or old fixtures.")
        ordinary = User(email="smoke-ordinary@example.com")
        reviewer = User(email="smoke-reviewer@example.com")
        login = User(email="smoke-login@example.com")
        db.add_all((ordinary, reviewer, login))
        db.flush()
        os.environ["HITL_REVIEWER_USER_IDS"] = str(reviewer.id)
        fresh = SurveySession(user_id=ordinary.id, language="en", status="in_progress")
        validation = SurveySession(user_id=ordinary.id, language="en", status="in_progress")
        completed = SurveySession(user_id=ordinary.id, language="en", status="completed", completed_at=datetime.utcnow())
        db.add_all((fresh, validation, completed))
        db.flush()
        for session, data in ((validation, collected()), (completed, collected("bachelor", "Nurse"))):
            for field, value in data.items():
                save_response_revision(db, session.id, field, value,
                    isco_code="2512" if field == "job_title" else None,
                    confidence_score=0.4 if field == "job_title" else None)
            if session.id == validation.id:
                routes._save_context(ConversationContext(session_id=session.id, language="en", state=ConversationState.VALIDATING,
                    collected_data=data, history=[{"role": "assistant", "content": "Please confirm or correct your answers."}]))
        occupation = next(row for row in active_responses(db, completed.id) if row.question_id == "job_title")
        queue = HITLQueue(session_id=completed.id, response_id=occupation.id, raw_text="Nurse", ai_code="2512", ai_confidence=0.4,
            ai_reasoning="Deliberately incorrect low-confidence fixture for supervisor correction.", priority="HIGH", status="pending",
            created_at=datetime.utcnow())
        db.add(queue)
        db.flush()
        profile = reports._build_profile(active_responses(db, completed.id))
        db.add(SurveyReportRecord(session_id=completed.id, language="en", profile_json=profile.model_dump_json(),
            report_en="Synthetic browser workflow report.", report_ar="Synthetic browser workflow report.",
            recommendations_en="Human review required.", recommendations_ar="Human review required.",
            quality_status="escalated", quality_score=0.4, generated_at=datetime.utcnow()))
        db.commit()
        email_otp.store_otp(db, login.id, "123456")
        fixtures = {"frontend_url": "http://127.0.0.1:3000", "api_url": "http://127.0.0.1:8001",
            "ordinary": {"token": create_access_token(ordinary.id), "email": ordinary.email, "session_id": fresh.id},
            "reviewer": {"token": create_access_token(reviewer.id), "email": reviewer.email},
            "login": {"email": login.email, "otp": "123456"}, "validation_session_id": validation.id,
            "report_session_id": completed.id, "escalation_id": queue.id, "review_code": "2221",
            "isolation": "Disposable PostgreSQL/Redis; fixture AI and delivery; actual application routes."}
        (PRIVATE / "fixtures.json").write_text(json.dumps(fixtures, indent=2) + "\n", encoding="utf-8")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, lifespan="off")


if __name__ == "__main__":
    main()
