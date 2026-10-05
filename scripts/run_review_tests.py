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
})
import dotenv
dotenv.load_dotenv = lambda *args, **kwargs: False

import socket
_connect = socket.socket.connect


def isolated_connect(sock, address):
    # Permit ephemeral local mock servers; deny production ports and remote hosts.
    if isinstance(address, tuple) and (address[0] not in ("127.0.0.1", "localhost", "::1") or address[1] in (5432, 6379, 6333, 6334, 11434)):
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
    def isolated_route_memory(self, monkeypatch):
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
