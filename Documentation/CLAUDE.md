> **ARCHIVED — SUPERSEDED (2026-08-02).** This is the original project-kickoff
> prompt used to bootstrap the codebase in March 2026. The folder layout,
> agent file names (`agents/agent_01_auth.py` … `agent_10_audit.py`), and the
> `hierarchical_rag/` module described below were **never built this way** —
> the system evolved into a different, flatter structure under
> `backend/agents/`, `backend/rag/`, `backend/api/`, etc. **It also does not
> live at the project root**, so Claude Code does not auto-load it.
>
> For the actual current architecture, endpoints, agent list, database
> schema, and file layout, see **`README.md`** at the repo root — it is kept
> up to date and verified against the live codebase. Keep this file only as
> a historical record of the original plan; do not use it to navigate or
> modify the current code.

---

# CLAUDE.md — Master Instructions for LFS AI System
# M.Tech Thesis: Multilingual Conversational AI for Labour Force Surveys
# IIIT Kottayam 2026 | Supervisor: Dr. Goutam Mali
#
# HOW THIS FILE WORKS:
# Claude Code reads CLAUDE.md automatically at the start of every session.
# Keep this file in the PROJECT ROOT. Never delete it.

---

## PROJECT SUMMARY

You are helping build a **10-agent CrewAI system** that conducts Labour Force
Surveys in 5 languages (Arabic, English, Urdu, Hindi, Tagalog). The system
replaces traditional face-to-face interviews with an AI chat interface.

**Core job of the system:**
A person types their job in any language → system assigns the correct
ISCO-08 occupation code → confidence score decides auto-accept or human review.

---

## WHAT IS ALREADY BUILT — NEVER REWRITE THESE

```
hierarchical_rag/rag_agent.py          ← COMPLETE. 4-stage ISCO/ISIC/ISCED pipeline.
hierarchical_rag/survey_integration.py ← COMPLETE. Agent 2↔5 bridge + HITL handler.
hierarchical_rag/evaluation.py         ← COMPLETE. BM25 vs Flat vs Hierarchical comparison.
hierarchical_rag/__init__.py           ← COMPLETE. Package exports.
demo_pipeline.py                       ← COMPLETE. End-to-end mock test.
```

**Rules for these files:**
- Read them before writing anything that touches them
- Import from them; do not duplicate their logic
- If you need to change them, ask first and explain why

---

## KEY CLASSES TO KNOW (memorise these)

### From hierarchical_rag/rag_agent.py

| Class / Function | What it does | How to call |
|---|---|---|
| `HierarchicalRAGPipeline` | Top-level orchestrator | `pipeline = HierarchicalRAGPipeline()` |
| `pipeline.classify_isco_only()` | Fast real-time ISCO coding | `result = pipeline.classify_isco_only("software developer", language="ar")` |
| `pipeline.classify()` | Full ISCO + ISIC + ISCED | `result = pipeline.classify(job, industry, education)` |
| `ClassificationResult` | Output dataclass | `result.code`, `result.confidence`, `result.hitl_required` |
| `KnowledgeBaseLoader` | Loads JSON into Qdrant | `loader.load_all(data_dir="./knowledge_bases/")` |
| `HITL_CONFIDENCE_THRESHOLD` | = 0.70 | If `conf < 0.70` → escalate to human |

### From hierarchical_rag/survey_integration.py

| Class / Function | What it does | Add to |
|---|---|---|
| `SurveyClassificationBridge` | Connects Agent 2 output to Agent 5 | `crew_orchestrator.py` |
| `SurveyClassificationTool` | CrewAI tool for Agent 2 | `agents/agent_02_conversation.py` tools list |
| `PersonRegisterVerifyTool` | Verify-then-ask logic | `agents/agent_04_register.py` tools list |
| `HITLEscalationHandler` | Builds reviewer package | Called inside `agents/agent_09_hitl.py` |

---

## FOLDER STRUCTURE — CREATE EXACTLY THIS

```
lfs_ai_system/
├── CLAUDE.md                          ← This file (keep here always)
├── .env                               ← API keys (never commit)
├── config.py                          ← All settings loaded from .env
├── crew_orchestrator.py               ← Main entry point — runs all 10 agents
├── demo_pipeline.py                   ← Already written — test this first
├── requirements.txt                   ← Already written
│
├── hierarchical_rag/                  ← ALREADY COMPLETE — do not modify
│   ├── __init__.py
│   ├── rag_agent.py
│   ├── survey_integration.py
│   └── evaluation.py
│
├── agents/                            ← One file per agent
│   ├── agent_01_auth.py
│   ├── agent_02_conversation.py
│   ├── agent_03_language.py
│   ├── agent_04_register.py
│   ├── agent_05_rag.py               ← Thin wrapper — imports from hierarchical_rag
│   ├── agent_06_emotion.py
│   ├── agent_07_validation.py
│   ├── agent_08_memory.py
│   ├── agent_09_hitl.py
│   └── agent_10_audit.py
│
├── knowledge_bases/                   ← JSON files for ISCO-08 etc.
│   ├── load_knowledge_bases.py        ← Run once to index into Qdrant
│   ├── isco08_major.json
│   ├── isco08_submajor.json
│   ├── isco08_minor.json
│   └── isco08_unit.json
│
├── api/                               ← FastAPI backend
│   ├── main.py
│   ├── routes/
│   │   ├── survey.py
│   │   └── hitl.py
│   └── models/
│       └── schemas.py
│
└── tests/
    ├── test_rag.py
    ├── test_agents.py
    └── run_evaluation.py
```

---

## ENVIRONMENT VARIABLES

Always load from `.env` using `python-dotenv`. Never hardcode keys.

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
QDRANT_URL=http://localhost:6333
POSTGRES_URL=postgresql://lfs_user:lfs_pass@localhost:5432/lfs_db
REDIS_URL=redis://localhost:6379
HITL_CONFIDENCE_THRESHOLD=0.70
TOP_K_CANDIDATES=5
DEBUG=true
```

### config.py pattern (always use this):
```python
from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY            = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY         = os.getenv("ANTHROPIC_API_KEY", "")
QDRANT_URL                = os.getenv("QDRANT_URL", "http://localhost:6333")
HITL_CONFIDENCE_THRESHOLD = float(os.getenv("HITL_CONFIDENCE_THRESHOLD", "0.70"))
```

---

## AGENT SPECIFICATIONS — BUILD IN THIS ORDER

### Agent 1 — Authentication & Privacy Manager
```
File:   agents/agent_01_auth.py
LLM:    ChatOpenAI(model="gpt-4o-mini", temperature=0)
Tools:  VerifyUAEPassTool, SendOTPTool, PrivacyNoticeTool
Output: {"authenticated": bool, "respondent_id": str, "language": str, "consent": bool}
Mock:   Tools return fake data for local testing (no real UAE Pass API)
Test:   python -c "from agents.agent_01_auth import AuthenticationAgent; print('OK')"
```

### Agent 2 — Survey Conversation Manager  ← MOST IMPORTANT
```
File:   agents/agent_02_conversation.py
LLM:    ChatOpenAI(model="gpt-4o", temperature=0.3)
Tools:  SurveyClassificationTool (import from hierarchical_rag.survey_integration)
Logic:  Follow question order A→B→C→D→E→F→G→H→I→J→K
        After Q.C5: call SurveyClassificationTool with job title
        Show respondent: ISCO code + confidence + 3 options if medium conf
        Store all answers in dict: {"C1": answer, "C2": answer, ...}
Output: complete survey_responses dict
```

### Agent 3 — Multilingual Language Processor
```
File:   agents/agent_03_language.py
LLM:    ChatOpenAI(model="gpt-4o-mini", temperature=0)
Tools:  LanguageDetectionTool, CodeSwitchDetectorTool, DialectNormalizerTool, NERExtractorTool
Langs:  ar, ar-gulf, en, ur, hi, tl
Output: {"language": str, "normalized_text": str, "entities": dict, "code_switches": list}
Note:   Output of Agent 3 becomes RAG query input for Agent 5
```

### Agent 4 — Person Register Integration
```
File:   agents/agent_04_register.py
LLM:    ChatOpenAI(model="gpt-4o", temperature=0)
Tools:  PersonRegisterVerifyTool (import from hierarchical_rag.survey_integration)
Logic:  verify-then-ask: check register → only ask what changed
Output: {"pre_filled": dict, "questions_to_skip": list, "burden_reduction_pct": float}
Mock:   Create 10 fake person records in a dict for testing
```

### Agent 5 — Multi-Classification RAG Expert
```
File:   agents/agent_05_rag.py
NOTE:   This is a THIN WRAPPER only. The real logic is in hierarchical_rag/rag_agent.py
Code:   from hierarchical_rag.rag_agent import build_classification_agent, HierarchicalRAGPipeline
        pipeline = HierarchicalRAGPipeline()
        agent = build_classification_agent(pipeline.vs_manager, pipeline.llm)
Do NOT rewrite the RAG logic. Just import and expose it.
```

### Agent 6 — Emotional Intelligence Monitor
```
File:   agents/agent_06_emotion.py
LLM:    ChatOpenAI(model="gpt-4o", temperature=0.4)
Tools:  EmotionDetectionTool → classify: neutral/frustrated/confused/distressed
Logic:  Agent 2 sends each respondent message through this agent
        If distressed → escalate to Agent 9 regardless of ISCO confidence
Output: {"emotion": str, "confidence": float, "intervention_needed": bool, "suggested_tone": str}
```

### Agent 7 — Data Validation Specialist
```
File:   agents/agent_07_validation.py
LLM:    ChatOpenAI(model="gpt-4o-mini", temperature=0)
Checks:
  - C1=No (not employed) but D1>0 (hours worked) → contradiction
  - Age<22 but ISCED=8 (PhD) → unlikely, flag for review
  - D1>80 hours/week → flag as outlier
  - Salary > typical for stated sector → soft flag
Output: {"valid": bool, "flags": list[str], "severity": "none"|"soft"|"hard"}
```

### Agent 8 — Context Memory Manager
```
File:   agents/agent_08_memory.py
LLM:    ChatOpenAI(model="gpt-4o-mini", temperature=0)
Short:  Redis — current session answers (key: session:{respondent_id})
Long:   PostgreSQL — historical records (table: respondent_history)
Logic:  On start: check if incomplete session exists → offer to resume
        After each section: save progress to Redis
        On complete: persist to PostgreSQL, clear Redis
```

### Agent 9 — HITL Quality Manager
```
File:   agents/agent_09_hitl.py
LLM:    ChatOpenAI(model="gpt-4o", temperature=0)
Import: HITLEscalationHandler from hierarchical_rag.survey_integration
Triggers:
  - result.hitl_required == True (conf < 0.70)
  - validation flags >= 2
  - emotion == "distressed"
  - random QC sample: 8% of all surveys
Output: {"escalation_id": str, "priority": "HIGH"|"MEDIUM"|"LOW",
         "reviewer_package": dict, "estimated_review_mins": int}
Store:  PostgreSQL table: hitl_queue
```

### Agent 10 — Audit & Compliance Logger
```
File:   agents/agent_10_audit.py
LLM:    ChatOpenAI(model="gpt-4o-mini", temperature=0)
Rule:   INSERT ONLY — never UPDATE or DELETE audit records
Fields: event_id, respondent_id, agent_name, action, input_data,
        output_data, confidence, timestamp, session_id
Table:  PostgreSQL: audit_log
API:    Expose log_event(agent_name, action, data) for other agents to call
```

---

## THE MAIN CREW ORCHESTRATOR

```python
# crew_orchestrator.py — exact structure to follow

from crewai import Crew, Process, Task
from agents.agent_01_auth import AuthenticationAgent
from agents.agent_02_conversation import SurveyConversationAgent
# ... all 10 agents

def run_survey(respondent_id: str) -> dict:
    """Main entry point. Call this to run one complete LFS survey."""

    # 1. Init agents
    auth  = AuthenticationAgent().agent
    convo = SurveyConversationAgent().agent
    # ...

    # 2. Define tasks in order
    task_auth = Task(
        description="Authenticate respondent {respondent_id} and get informed consent.",
        agent=auth,
        expected_output='JSON only: {"authenticated": bool, "respondent_id": str, "language": str, "consent": bool}',
    )
    # task_register, task_interview, task_validate, task_hitl, task_audit

    # 3. Build crew
    crew = Crew(
        agents=[auth, convo, ...],
        tasks=[task_auth, ...],
        process=Process.hierarchical,
        manager_agent=convo,      # Agent 2 is the manager
        verbose=True,
    )

    return crew.kickoff(inputs={"respondent_id": respondent_id})
```

---

## CODING RULES — FOLLOW EVERY TIME

### 1. Always return JSON from tasks
```python
# Every task expected_output must say "JSON only" to prevent LLM prose
expected_output='JSON only: {"key": value, ...}'
```

### 2. Always use allow_delegation=False on worker agents
```python
agent = Agent(
    role="...",
    allow_delegation=False,   # ← REQUIRED on all agents except Agent 2
    ...
)
```

### 3. Always load API keys from config.py
```python
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY
# Never: api_key="sk-hardcoded"
```

### 4. Always test after creating each file
```python
# Minimum test pattern after every new agent file:
python -c "from agents.agent_01_auth import AuthenticationAgent; a = AuthenticationAgent(); print(a.agent.role)"
# Must print the role string without error
```

### 5. Always use temperature=0 for classification agents
```python
# Agents 1, 3, 4, 5, 7, 8, 10 → temperature=0   (deterministic)
# Agents 2, 6, 9              → temperature=0.2-0.4 (slight creativity ok)
```

### 6. Never duplicate logic that exists in hierarchical_rag/
```python
# WRONG:
def my_own_isco_lookup(job): ...   # duplicates rag_agent.py

# RIGHT:
from hierarchical_rag.rag_agent import HierarchicalRAGPipeline
pipeline = HierarchicalRAGPipeline()
result = pipeline.classify_isco_only(job)
```

---

## SERVICES — START BEFORE RUNNING ANY CODE

```bash
# Terminal 1: Start Qdrant
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# Terminal 2: Start PostgreSQL
docker run -d -p 5432:5432 --name postgres \
  -e POSTGRES_DB=lfs_db \
  -e POSTGRES_USER=lfs_user \
  -e POSTGRES_PASSWORD=lfs_pass \
  postgres:15

# Terminal 3: Start Redis
docker run -d -p 6379:6379 --name redis redis:alpine

# Verify all running:
docker ps | grep -E "qdrant|postgres|redis"
```

---

## TESTING SEQUENCE — RUN IN ORDER

### Step 1 — Verify existing code works
```bash
python demo_pipeline.py
# Expected: ISCO 2512 for "software developer", conf > 0.90, hitl_required=False
```

### Step 2 — Test RAG directly
```python
from hierarchical_rag.rag_agent import HierarchicalRAGPipeline

pipeline = HierarchicalRAGPipeline(use_local_embeddings=False)
result = pipeline.classify_isco_only("software developer", language="en")
assert result.code == "2512"
assert result.confidence > 0.70
assert result.hitl_required == False

result_ar = pipeline.classify_isco_only("مبرمج حاسوب", language="ar")
assert result_ar.code.startswith("25")

result_low = pipeline.classify_isco_only("I do various things")
assert result_low.hitl_required == True
print("All RAG tests passed ✓")
```

### Step 3 — Run evaluation comparison (thesis Chapter 6 data)
```bash
python tests/run_evaluation.py
# Must print table: BM25 vs FlatVector vs HierarchicalRAG
# HierarchicalRAG must show highest Top-1, Top-3, Kappa
```

### Step 4 — Test each agent as you build
```bash
python -c "from agents.agent_01_auth import AuthenticationAgent; print('Agent 1 OK')"
python -c "from agents.agent_02_conversation import SurveyConversationAgent; print('Agent 2 OK')"
# ... repeat for each agent
```

### Step 5 — Run full crew
```bash
python crew_orchestrator.py
# Should run a complete mock survey for test respondent "TEST-001"
```

---

## FASTAPI ENDPOINTS TO BUILD

```
POST /survey/start              → {session_id, first_question, language}
POST /survey/respond            → {session_id, answer} → {next_question, progress_pct, isco_result?}
GET  /survey/status/{id}        → {section, question_num, answers_count, complete}
GET  /survey/result/{id}        → {isco_code, isic_code, isced_code, confidence, hitl_required}
GET  /hitl/queue                → [{escalation_id, priority, respondent_id, created_at}]
POST /hitl/review               → {escalation_id, code, action} → {saved: bool}
GET  /health                    → {status: "ok", version: "1.0.0"}
WS   /survey/ws/{session_id}    → real-time chat socket
```

Run with: `uvicorn api.main:app --reload --port 8000`

---

## KNOWLEDGE BASE JSON FORMAT

Every JSON file must use this exact structure:

```json
[
  {
    "code": "2512",
    "parent_code": "251",
    "label_en": "Software and Applications Developers",
    "label_ar": "مطورو البرمجيات والتطبيقات",
    "description_en": "Design, develop, and test software systems.",
    "description_ar": "تصميم وتطوير واختبار أنظمة البرمجيات.",
    "examples": ["Software developer", "App developer", "مبرمج"]
  }
]
```

Note: `parent_code` is required for submajor/minor/unit files. Not needed for major.

---

## COMMON ERRORS AND FIXES

| Error | Cause | Fix |
|---|---|---|
| `Connection refused :6333` | Qdrant not running | `docker start qdrant` |
| `AuthenticationError OpenAI` | Key not set | Check `.env`, run `export OPENAI_API_KEY=sk-...` |
| `Confidence always 0.0` | Qdrant empty | Run `python knowledge_bases/load_knowledge_bases.py` |
| `ImportError: crewai` | Not installed | `pip install crewai==0.28.0` |
| `JSON decode error from crew` | LLM returned prose | Add "JSON only:" to task `expected_output` |
| `Agent delegation loop` | `allow_delegation=True` | Set `allow_delegation=False` on all worker agents |
| `Redis connection refused` | Redis not running | `docker start redis` |
| `ClassificationResult missing field` | Stale import | Re-run: `from hierarchical_rag.rag_agent import ClassificationResult` |

---

## WHAT TO SHOW AT EACH THESIS REVIEW

| Review | Minimum demo required | Command |
|---|---|---|
| **First Review** ✓ Done | Architecture slides | — |
| **Second Review** | RAG working + evaluation table | `python tests/run_evaluation.py` |
| **Third Review** | All 10 agents + FastAPI running | `python crew_orchestrator.py` |
| **Final Defense** | Live demo: chat → ISCO → HITL | `uvicorn api.main:app` |

---

## HOW TO ASK CLAUDE CODE FOR HELP

Use these patterns for best results:

**To build a new agent:**
```
Build agents/agent_01_auth.py following the spec in CLAUDE.md section "Agent 1".
After writing it, test with:
  python -c "from agents.agent_01_auth import AuthenticationAgent; print(AuthenticationAgent().agent.role)"
Show me the output.
```

**To debug an error:**
```
I got this error: [paste full traceback]
The relevant file is agents/agent_02_conversation.py
Read CLAUDE.md first, then fix the error without changing the imports from hierarchical_rag/
```

**To run the evaluation:**
```
Create tests/run_evaluation.py using RAGEvaluator from hierarchical_rag/evaluation.py.
Generate 50 synthetic test cases, run all 3 systems (BM25, Flat, Hierarchical),
print a formatted comparison table. Run it and show output.
```

**To connect two agents:**
```
Connect Agent 2 output to Agent 5 using SurveyClassificationBridge
from hierarchical_rag/survey_integration.py.
Read both files first. Then show me the connection code only — no rewrites.
```

---
*Last updated: March 2026 | Read CLAUDE.md at the start of every Claude Code session.*
