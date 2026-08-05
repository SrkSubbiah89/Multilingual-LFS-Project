import logging
import os
import uuid

# Suppress TensorFlow noise BEFORE any import that triggers TF/Keras loading
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tf_keras").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy import text
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.api.auth_routes import router as auth_router
from backend.api.survey_routes import router as survey_router, _FAST_MODE as _ROUTE_FAST_MODE
from backend.database.connection import SessionLocal

_startup_logger = logging.getLogger("lfs.startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm ML classifiers on startup; flush connections on shutdown."""
    try:
        from backend.api.survey_routes import (
            _get_isco_classifier,
            _get_isic_classifier,
            _get_isced_classifier,
            _get_nationality_classifier,
        )
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _get_isco_classifier)
        await loop.run_in_executor(None, _get_isic_classifier)
        await loop.run_in_executor(None, _get_isced_classifier)
        await loop.run_in_executor(None, _get_nationality_classifier)
        _startup_logger.warning("All classifiers pre-warmed. fast_mode=%s", _ROUTE_FAST_MODE)
    except Exception as exc:
        _startup_logger.warning("Classifier pre-warm failed (non-fatal): %s", exc)

    yield  # server runs here

    # ── Graceful shutdown ───────────────────────────────────────────────────
    _startup_logger.warning("Shutting down — flushing connections.")
    try:
        from backend.api.survey_routes import _context_memory
        if _context_memory is not None:
            _context_memory._redis.close()
    except Exception:
        pass
    try:
        from backend.database.connection import engine
        engine.dispose()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CORS — explicit allowlist (not wildcard)
# ---------------------------------------------------------------------------
_CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if o.strip()
]

app = FastAPI(title="LFS Conversational AI", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)


# ---------------------------------------------------------------------------
# X-Request-ID middleware — correlation ID for distributed tracing
# ---------------------------------------------------------------------------

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Reads X-Request-ID from incoming requests (or generates a UUID v4).
    Echoes the ID in the response header so callers can correlate logs.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# Security headers middleware — CSP, clickjacking, MIME sniffing, referrer
# ---------------------------------------------------------------------------

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Injects defensive HTTP security headers on every response.

    CSP allows same-origin scripts + styles only; tighten further once a
    nonce-based inline script strategy is in place for Next.js.
    """
    _CSP = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "   # Next.js needs inline scripts
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = self._CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        return response


app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)


# ---------------------------------------------------------------------------
# Rate limiting — 30 requests per minute per IP on /message endpoint
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return HTTPException(
        status_code=429,
        detail="Too many requests. Max 30 per minute.",
    )


app.include_router(auth_router)
app.include_router(survey_router)


@app.get("/health", tags=["health"])
def health_check():
    """
    Returns service-level health for Redis, Qdrant, and Ollama in addition to
    the basic status flag.  Each sub-service reports "ok" or an error string.
    The overall HTTP status is always 200 — callers check individual fields.
    """
    import urllib.request
    import urllib.error

    services: dict[str, str] = {}

    # Redis
    try:
        import redis as _redis_lib
        _r = _redis_lib.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"), socket_timeout=2
        )
        _r.ping()
        services["redis"] = "ok"
    except Exception as _e:
        services["redis"] = f"error: {_e}"

    # Qdrant
    try:
        _qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        req = urllib.request.urlopen(f"{_qdrant_url}/healthz", timeout=2)
        services["qdrant"] = "ok" if req.status == 200 else f"status {req.status}"
    except Exception as _e:
        services["qdrant"] = f"error: {_e}"

    # Ollama
    try:
        _ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        req = urllib.request.urlopen(f"{_ollama_url}/api/tags", timeout=2)
        services["ollama"] = "ok" if req.status == 200 else f"status {req.status}"
    except Exception as _e:
        services["ollama"] = f"error: {_e}"

    return {
        "status": "ok",
        "fast_mode": _ROUTE_FAST_MODE,
        "lfs_fast_mode_env": os.getenv("LFS_FAST_MODE"),
        "services": services,
    }


@app.get("/debug/isco/{job_title}", tags=["debug"])
def debug_isco(job_title: str):
    import time, traceback
    try:
        from backend.api.survey_routes import _get_isco_classifier
        t = time.perf_counter()
        clf = _get_isco_classifier().classify(job_title, use_llm=False)
        ms = int((time.perf_counter() - t) * 1000)
        return {"code": clf.primary.code, "title": clf.primary.title_en, "conf": clf.primary.confidence, "ms": ms}
    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()}


@app.get("/ready", tags=["health"])
def readiness_check():
    """Returns 200 only when the database is reachable. Fails fast on DB error."""
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database not ready: {e}")

    redis_status = "unknown"
    try:
        import redis as _redis_lib
        _r = _redis_lib.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"), socket_timeout=2
        )
        _r.ping()
        redis_status = "connected"
    except Exception as _re:
        redis_status = f"unavailable: {_re}"

    return {"status": "ready", "database": "connected", "redis": redis_status}
