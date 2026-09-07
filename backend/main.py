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

    # Pre-warm the free-text correction model AND the 3 main per-path JSON
    # schemas, 2026-09-02.
    #
    # Two distinct costs were live-measured, not assumed: (1) loading
    # OLLAMA_CORRECTION_MODEL (default qwen2.5:3b) into Ollama's memory --
    # this model runs CPU-only on this machine (confirmed via GET /api/ps
    # showing size_vram: 0) and was only ever loaded lazily on a
    # respondent's first correction attempt, a real, reported slow response
    # that measured as a genuine cold start. (2) Separately, and this was
    # the bigger surprise: even with the model already warm, the FIRST
    # schema-constrained call for a given respondent's field-path shape
    # (correction_schema_for()'s valid_fields/enum, which differs by
    # employment_status path) still took ~60-75s -- confirmed directly by
    # timing a warm-model call against a brand new schema shape (74.4s)
    # right after two ~4s calls against an already-seen shape. Grammar
    # compilation for a JSON-Schema-constrained decode is apparently
    # cached per schema shape, not just per model. Warming the model alone
    # (a trivial "ping" call, tried first) only fixes cost (1).
    #
    # Fixed by pre-warming the model AND all 3 real employment_status paths'
    # schemas (employed / unemployed / not_in_labour_force) at boot, using
    # ConversationManager.correction_schema_for() -- the exact same method
    # the real correction call uses, so there's no risk of the warm-up
    # schema drifting from the real one. This does NOT cover every possible
    # schema shape (field_of_study/emiratization_program conditionals shift
    # it slightly), so a first correction can still occasionally be slower
    # than a cache hit -- but the 3 base paths cover the large majority of
    # real respondents. Trade-off, made deliberately: this adds real time
    # to server startup (3 sequential ~60-75s cold compiles the first time
    # this ever runs against a given Ollama installation); acceptable
    # because startup happens once per server run, off the critical path of
    # any real user, while a slow first correction is directly experienced
    # by every single respondent who makes one.
    try:
        import json
        import urllib.request
        from backend.agents.conversation_manager import ConversationManager as _CM

        _warm_model = os.getenv("OLLAMA_CORRECTION_MODEL", "qwen2.5:3b")
        _warm_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        _warm_shapes = {
            "employed": {"employment_status": "employed", "education_level": "bachelor"},
            "unemployed": {"employment_status": "unemployed", "education_level": "bachelor"},
            "not_in_labour_force": {"employment_status": "not_in_labour_force", "education_level": "bachelor"},
        }

        def _warm_ollama(schema: dict | None) -> None:
            # "format" defaults to the string "json" (not omitted, not
            # null) when there's no schema yet -- matching exactly what
            # _call_ollama_json does for its own non-schema calls, since
            # that's the behavior already confirmed working today, rather
            # than testing an untried `null`/omitted value here.
            payload = json.dumps({
                "model": _warm_model,
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
                # 30m -> 24h, 2026-09-04: matches the same change and the
                # same reasoning in conversation_manager.py's
                # _call_ollama_json -- keep this pre-warm's TTL in sync with
                # the real call's, since the whole point of pre-warming at
                # boot is to bridge the gap until that call's own keep_alive
                # takes over.
                "keep_alive": "24h",
                "format": schema if schema is not None else "json",
                "options": {"num_predict": 1},
            }).encode()
            req = urllib.request.Request(
                f"{_warm_url}/api/chat", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=120):
                pass

        import asyncio as _asyncio
        loop2 = _asyncio.get_event_loop()
        # First call has no schema -- loads the model weights themselves
        # (cost (1) above) before any schema-specific compile is attempted.
        await loop2.run_in_executor(None, _warm_ollama, None)
        for _path_name, _shape in _warm_shapes.items():
            _, _, _schema = _CM.correction_schema_for(_shape)
            await loop2.run_in_executor(None, _warm_ollama, _schema)
            _startup_logger.warning("Correction schema pre-warmed for path: %s", _path_name)
        _startup_logger.warning("Correction model (%s) fully pre-warmed (model + 3 path schemas).", _warm_model)
    except Exception as exc:
        _startup_logger.warning("Correction model pre-warm failed (non-fatal): %s", exc)

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
# Rate limiting — 30 requests per minute per IP on /message endpoint.
#
# REMOVED 2026-09-02: this used to set up a slowapi Limiter + a
# RateLimitExceeded exception handler here, but neither was ever wired to
# anything real -- no route ever used slowapi's @limiter.limit(...)
# decorator, so RateLimitExceeded was never once raised, and
# survey_routes.py's own rate-limit dependency called a `.hit()` method
# slowapi.Limiter doesn't actually have (a real, separately-fixed bug --
# see survey_routes.py's _check_rate_limit for the full writeup). This
# entire block was dead weight that looked like real protection but did
# nothing. The actual enforcement now lives entirely in
# survey_routes.py's _check_rate_limit (reusing email_otp.py's proven,
# Redis-backed check_rate_limit()), which raises HTTPException(429)
# directly rather than relying on a FastAPI exception handler.
# ---------------------------------------------------------------------------

app.include_router(auth_router)
app.include_router(survey_router)


@app.get("/", tags=["health"])
def root():
    """
    This is the backend API only — there is no web page here. The actual
    survey app is the separate frontend (Next.js) service; see /docs for
    the API reference.
    """
    return {
        "service": "LFS Conversational AI backend API",
        "note": "This is an API server, not a web page. The survey app itself is served separately by the frontend.",
        "docs": "/docs",
        "health": "/health",
    }


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
