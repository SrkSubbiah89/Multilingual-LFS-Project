"""Run review regressions with test-only credentials/database and no startup warmup."""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update({
    "DATABASE_URL": "sqlite:///:memory:",
    "JWT_SECRET": "review-tests-only-secret-with-at-least-thirty-two-characters",
    "APP_ENV": "development",
    "LFS_FAST_MODE": "false",
    "REDIS_URL": "redis://127.0.0.1:1",
    "REDIS_HOST": "127.0.0.1", "REDIS_PORT": "1",
    "OLLAMA_BASE_URL": "http://127.0.0.1:1",
    "ANTHROPIC_API_KEY": "test-only-placeholder",
    "OPENAI_API_KEY": "", "GROQ_API_KEY": "", "GEMINI_API_KEY": "",
    "OPENROUTER_API_KEY": "", "SENDGRID_API_KEY": "",
    "TWILIO_ACCOUNT_SID": "", "TWILIO_AUTH_TOKEN": "",
    "CREWAI_TRACING_ENABLED": "false", "OTEL_SDK_DISABLED": "true",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HITL_REVIEWER_USER_IDS": "",
    "ENABLE_SURVEY_CLASSIFICATION_CREW": "false",
    "LFS_LOCAL_ONLY": "false",
})
# An inherited cloud or laptop Qdrant setting must not replace the host/port
# client the unit tests mock, and must not redirect deadline tests away from
# their local fixture server.
os.environ.pop("QDRANT_URL", None)
os.environ.pop("QDRANT_API_KEY", None)
import dotenv
from dotenv import load_dotenv as _real_load_dotenv

# Refuse the project .env so a request-path reload cannot retarget the suite at
# the development database. A test that points find_dotenv at its own file
# still loads that file; nothing else is accepted.
_PROJECT_ENV = (Path(__file__).resolve().parents[1] / ".env").resolve()


def isolated_load_dotenv(*args, **kwargs):
    explicit = kwargs.get("dotenv_path", args[0] if args else None)
    if explicit:
        candidate = Path(str(explicit))
    else:
        from dotenv.main import find_dotenv
        found = find_dotenv(usecwd=kwargs.get("usecwd", False))
        if not found:
            return False
        candidate = Path(found)
    try:
        resolved = candidate.resolve()
    except OSError:
        return False
    if resolved == _PROJECT_ENV:
        return False
    return _real_load_dotenv(*args, **kwargs)


dotenv.load_dotenv = isolated_load_dotenv

import socket
_connect = socket.socket.connect


def isolated_connect(sock, address):
    # Permit ephemeral local mock servers; deny production ports and remote hosts.
    if isinstance(address, tuple) and (address[0] not in ("127.0.0.1", "localhost", "::1") or address[1] in (5432, 6379, 6333, 6334, 11434, 15432, 16379, 25432, 26379, 8000, 18000)):
        raise ConnectionRefusedError("External service access disabled in review tests")
    return _connect(sock, address)


socket.socket.connect = isolated_connect

import sqlalchemy
from sqlalchemy.pool import QueuePool
_create_engine = sqlalchemy.create_engine


def isolated_engine(url, **kwargs):
    if str(url) == "sqlite:///:memory:" and "max_overflow" in kwargs and "poolclass" not in kwargs:
        kwargs["poolclass"] = QueuePool
    return _create_engine(url, **kwargs)


sqlalchemy.create_engine = isolated_engine

from backend.main import app
from backend.auth import email_otp
email_otp._get_rl_redis = lambda: None


@asynccontextmanager
async def no_warmup(app):
    yield


app.router.lifespan_context = no_warmup

import pytest
from unittest.mock import MagicMock


class ReviewIsolation:
    @pytest.fixture(autouse=True)
    def isolated_route_memory(self, monkeypatch, request):
        # This module already isolates Redis and provider initialization itself.
        # Replacing its production memory object would invalidate outage tests.
        if "isolated_unit_services" in request.fixturenames:
            yield
            return
        from backend.api import survey_routes
        from backend.agents.context_memory import ContextMemory

        class TestRedis:
            def __init__(self):
                self.values = {}

            def get(self, key):
                return self.values.get(key)

            def set(self, key, value, **kwargs):
                self.values[key] = value
                return True

            def delete(self, key):
                return bool(self.values.pop(key, None))

            def lock(self, *args, **kwargs):
                return MagicMock(acquire=MagicMock(return_value=True))

        memory = object.__new__(ContextMemory)
        memory._redis = TestRedis()
        memory._ttl = 86400
        survey_routes._contexts.clear()
        monkeypatch.setattr(survey_routes, "_get_context_memory", lambda: memory)
        yield
        survey_routes._contexts.clear()


raise SystemExit(pytest.main(sys.argv[1:], plugins=[ReviewIsolation()]))
