# Multilingual LFS Conversational AI

Backend API for a Multilingual Labour Force Survey (LFS) conversational AI system. Handles user authentication via email OTP and manages multilingual survey sessions.

## Features

- Email OTP authentication with 10-minute expiry and single-use enforcement
- JWT-based session tokens (HS256)
- FastAPI with automatic OpenAPI docs
- PostgreSQL via SQLAlchemy ORM
- Docker Compose for local infrastructure (Postgres, Redis, Qdrant)

## Project Structure

```
.
├── backend/
│   ├── main.py               # FastAPI app entry point
│   ├── agents/               # Conversational AI agents
│   ├── api/
│   │   ├── auth_routes.py    # /auth/request-otp, /auth/verify-otp, /auth/logout
│   │   └── survey_routes.py  # Survey session endpoints
│   ├── auth/
│   │   ├── email_otp.py      # OTP generation, storage, and email delivery
│   │   └── jwt_handler.py    # JWT creation and verification
│   ├── database/
│   │   ├── models.py         # SQLAlchemy models (User, OTPCode, SurveySession)
│   │   ├── connection.py     # DB engine and session factory
│   │   └── migrations/       # Alembic migration scripts
│   ├── llm/                  # LLM integration
│   ├── rag/                  # Retrieval-Augmented Generation pipeline
│   └── tests/
└── docker/
    └── docker-compose.yml    # Postgres, Redis, Qdrant
```

## Prerequisites

- Python 3.11+
- Docker Desktop
- A SendGrid account (for OTP email delivery)

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/SrkSubbiah89/Multilingual-LFS-Project.git
cd Multilingual-LFS-Project
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `JWT_SECRET` | Secret key for signing JWTs |
| `SENDGRID_API_KEY` | SendGrid API key for OTP emails |
| `SENDGRID_FROM_EMAIL` | Verified sender email address |

### 3. Start infrastructure

```bash
docker compose -f docker/docker-compose.yml up -d postgres
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Create database tables

```bash
python -c "import backend.database.models; from backend.database.connection import Base, engine; Base.metadata.create_all(bind=engine)"
```

### 6. Run the server

```bash
python -m uvicorn backend.main:app --reload
```

API is available at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

## API Endpoints

### Health

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Returns `{"status": "ok"}` |

### Auth

| Method | Endpoint | Body | Description |
|---|---|---|---|
| POST | `/auth/request-otp` | `{"email": "..."}` | Send a 6-digit OTP to the given email |
| POST | `/auth/verify-otp` | `{"email": "...", "code": "..."}` | Verify OTP and receive a JWT |
| POST | `/auth/logout` | — | Client-side logout (discard JWT) |

## Running Tests

```bash
pytest backend/tests/
```
