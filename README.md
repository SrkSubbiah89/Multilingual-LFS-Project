# Multilingual LFS Conversational AI

An AI-powered Labour Force Survey (LFS) system that conducts employment interviews in **English and Arabic**, classifies job titles to [ISCO-08](https://www.ilo.org/public/english/bureau/stat/isco/isco08/) codes, and detects code-switching between languages in real time.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser                                                        │
│  Next.js 14  (login / OTP → chat interface, EN + AR RTL)       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP (REST)
┌───────────────────────────▼─────────────────────────────────────┐
│  FastAPI backend                                                │
│                                                                 │
│  ① LanguageProcessor  — langdetect + Unicode script analysis   │
│     • language detection (en / ar / other)                     │
│     • code-switch detection & segment labelling                │
│     • NER via CrewAI Agent (GPT-4o-mini)                       │
│                                                                 │
│  ② ConversationManager — 5-state FSM (CrewAI + GPT-4o-mini)   │
│     GREETING → COLLECTING → CLARIFYING → VALIDATING → DONE    │
│                                                                 │
│  ③ ISCOClassifier — two-stage pipeline                         │
│     • Stage 1: semantic search  (multilingual-e5-large)        │
│     • Stage 2: LLM re-ranking   (Claude 3.5 Sonnet)           │
│       (skipped when similarity ≥ 0.92)                         │
└──────┬─────────────────────────────┬───────────────────────────┘
       │                             │
┌──────▼──────┐          ┌──────────▼──────────────┐
│  PostgreSQL │          │  Qdrant vector DB        │
│  Users      │          │  113 ISCO-08 entries     │
│  Sessions   │          │  multilingual-e5-large   │
│  Responses  │          │  embeddings (1024-dim)   │
└─────────────┘          └─────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, React 18, Tailwind CSS, Noto Sans Arabic |
| Backend | FastAPI, SQLAlchemy, Alembic |
| AI Agents | CrewAI, GPT-4o-mini (general), Claude 3.5 Sonnet (classification) |
| Embeddings | `intfloat/multilingual-e5-large` (sentence-transformers) |
| Vector DB | Qdrant |
| Auth | Email OTP (SendGrid) → JWT (HS256) |
| Database | PostgreSQL 15 |
| Cache | Redis 7 |

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for the one-command start)
- API keys: **OpenAI**, **Anthropic**, **SendGrid**

For local development without Docker you also need Python 3.11+ and Node 20+.

---

## Quick Start (Docker)

```bash
# 1. Clone
git clone https://github.com/SrkSubbiah89/Multilingual-LFS-Project.git
cd Multilingual-LFS-Project

# 2. Configure
cp .env.example .env
# Edit .env — fill in the four required secrets (see table below)

# 3. Start everything
docker compose -f docker/docker-compose.yml up --build
```

| URL | Service |
|---|---|
| http://localhost:3000 | Frontend (login / chat) |
| http://localhost:8000 | Backend API |
| http://localhost:8000/docs | Interactive API docs (Swagger) |
| http://localhost:6333 | Qdrant dashboard |

> **First boot note:** The `multilingual-e5-large` model (~2 GB) is downloaded on
> the first backend start. Subsequent starts use the `model_cache` Docker volume.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

### Required

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Powers GPT-4o-mini (NER + conversation) |
| `ANTHROPIC_API_KEY` | Powers Claude 3.5 Sonnet (ISCO classification) |
| `JWT_SECRET` | Random secret for signing JWTs (e.g. `openssl rand -hex 32`) |
| `SENDGRID_API_KEY` | OTP email delivery |
| `SENDGRID_FROM_EMAIL` | Verified sender address in SendGrid |

### Optional / has defaults

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | postgres://… | Overridden automatically in Docker |
| `QDRANT_HOST` | `localhost` | Overridden automatically in Docker |
| `REDIS_URL` | `redis://localhost:6379` | Overridden automatically in Docker |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Token lifetime |

---

## Local Development (without Docker)

```bash
# ── Infrastructure (Postgres + Qdrant + Redis) ──
docker compose -f docker/docker-compose.yml up -d postgres qdrant redis

# ── Backend ─────────────────────────────────────
pip install -r requirements.txt tf-keras
cp .env.example .env        # fill in secrets

# Create DB tables
python -c "from backend.database.connection import Base, engine; Base.metadata.create_all(bind=engine)"

uvicorn backend.main:app --reload
# → http://localhost:8000

# ── Frontend ─────────────────────────────────────
cd frontend
cp .env.local.example .env.local   # set NEXT_PUBLIC_API_URL if needed
npm install
npm run dev
# → http://localhost:3000
```

---

## API Reference

### Health

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Returns `{"status": "ok"}` |

### Auth

| Method | Endpoint | Body | Description |
|---|---|---|---|
| POST | `/auth/request-otp` | `{"email": "..."}` | Send 6-digit OTP via email |
| POST | `/auth/verify-otp` | `{"email": "...", "code": "..."}` | Verify OTP → JWT |
| POST | `/auth/logout` | — | Client-side logout |

### Survey Sessions

| Method | Endpoint | Body | Description |
|---|---|---|---|
| POST | `/survey/sessions` | `{"language": "en"}` | Create session |
| GET | `/survey/sessions` | — | List user's sessions |
| GET | `/survey/sessions/{id}` | — | Get session |
| PATCH | `/survey/sessions/{id}/complete` | — | Mark complete |
| DELETE | `/survey/sessions/{id}` | — | Delete session |

### Conversational Turn

```
POST /survey/sessions/{id}/message
Authorization: Bearer <token>

Body:  { "message": "I work as a software engineer full time" }
```

```jsonc
// Response
{
  "reply": "Thank you! Which industry or sector do you work in?",
  "state": "collecting_info",
  "detected_language": "en",
  "is_code_switched": false,
  "entities": [
    { "text": "software engineer", "label": "JOB_TITLE", "language": "en" }
  ],
  "isco_classifications": [
    {
      "job_title": "software engineer",
      "primary_code": "2512",
      "primary_title_en": "Software Developers",
      "primary_title_ar": "مطورو البرمجيات",
      "confidence": 0.9134,
      "method": "semantic"
    }
  ],
  "session_completed": false
}
```

### Survey Responses (manual override)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/survey/sessions/{id}/responses` | Submit a raw response |
| GET | `/survey/sessions/{id}/responses` | List responses for a session |
| PATCH | `/survey/sessions/{id}/responses/{rid}` | Update ISCO code / confidence |

---

## Conversation Flow

```
User connects
     │
     ▼
 GREETING ──────────────────────────────────────────────┐
     │  (any message)                                    │
     ▼                                                   │
 COLLECTING_INFO                                         │
     │  ambiguous answer                                 │ terminal
     ├──────────────► CLARIFYING ──► COLLECTING_INFO     │
     │                                                   │
     │  all 5 fields collected                           │
     ▼                                                   │
 VALIDATING                                              │
     │  confirmed            not confirmed               │
     ├─────────────────────────────► COLLECTING_INFO     │
     │                                                   │
     ▼                                                   │
 COMPLETING ◄──────────────────────────────────────────┘
     │  session persisted to DB, context cleared
```

Required fields: `employment_status`, `job_title`, `industry`, `hours_per_week`, `employment_type`

---

## Running Tests

```bash
# From repo root
pip install -r requirements.txt tf-keras
pytest backend/tests/ -v
```

212 tests across 6 files — all pass without a running backend or LLM API keys (all external calls are mocked).

---

## Project Structure

```
.
├── Dockerfile                  # Backend image
├── .env.example                # Environment variable template
├── requirements.txt
├── alembic.ini
│
├── backend/
│   ├── main.py                 # FastAPI app + CORS
│   ├── agents/
│   │   ├── language_processor.py   # Lang detection, NER
│   │   ├── conversation_manager.py # FSM survey conductor
│   │   └── isco_classifier.py      # Two-stage ISCO-08 classifier
│   ├── api/
│   │   ├── auth_routes.py      # OTP + JWT endpoints
│   │   └── survey_routes.py    # Session + message endpoints
│   ├── auth/
│   │   ├── email_otp.py        # OTP generation & delivery
│   │   └── jwt_handler.py      # JWT creation & verification
│   ├── database/
│   │   ├── models.py           # SQLAlchemy models
│   │   └── connection.py       # DB engine & session
│   ├── llm/
│   │   └── llm_client.py       # Shared LLM factory (GPT / Claude)
│   ├── rag/
│   │   └── vector_store.py     # Qdrant + multilingual-e5-large
│   └── tests/                  # 212 pytest tests
│
├── frontend/
│   ├── Dockerfile              # Next.js multi-stage image
│   ├── pages/
│   │   ├── index.js            # Login (email → OTP)
│   │   └── chat.js             # Survey conversation
│   ├── components/
│   │   ├── api.js              # Fetch wrapper
│   │   └── LanguageToggle.js   # EN / AR switcher
│   └── styles/globals.css
│
└── docker/
    └── docker-compose.yml      # All 5 services
```
