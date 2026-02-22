import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

APP_ENV = os.getenv("APP_ENV", "development")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LFS Conversational AI",
    version="0.1.0",
    description=(
        "Backend API for the Multilingual Labour Force Survey (LFS) "
        "conversational AI system. Handles user authentication via email OTP "
        "and manages multilingual survey sessions."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

if APP_ENV == "development":
    origins = ["*"]
else:
    origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    import backend.database.models  # noqa: F401 — registers models with metadata
    from backend.database.connection import Base, engine

    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified/created.")
    except Exception as exc:
        logger.warning("Database not available at startup: %s", exc)

    logger.info("Environment : %s", APP_ENV)
    logger.info("Listening on: http://%s:%s", APP_HOST, APP_PORT)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from backend.api.auth_routes import router as auth_router      # noqa: E402
from backend.api.survey_routes import router as survey_router  # noqa: E402

app.include_router(auth_router)
app.include_router(survey_router)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok", "version": "0.1.0"}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=APP_ENV == "development",
    )
