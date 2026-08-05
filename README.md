# Multilingual LFS Conversational AI

> **Last Updated: June 2026** · 1,178 tests passing · 11 DB tables · 25 test files

An AI-powered **Labour Force Survey (LFS)** system that conducts employment interviews in **English, Arabic (MSA + Gulf dialect), Urdu, Hindi, and Tagalog**, classifies job titles to [ISCO-08](https://www.ilo.org/public/english/bureau/stat/isco/isco08/) codes (4-digit unit groups), classifies industries to [ISIC Rev.4](https://unstats.un.org/unsd/publication/seriesm/seriesm_4rev4e.pdf) (full 4-level hierarchy: Section → Division → Group → **4-digit Class**), classifies education field of specialisation to [ISCED-F 2013](https://uis.unesco.org/en/topic/international-standard-classification-education-isced) (Broad → Narrow → **4-digit Detailed field**) plus attainment level to [ISCED 2011](https://uis.unesco.org/en/topic/international-standard-classification-education-isced) (levels 0–8), and implements the complete **UAE Labour Force Survey questionnaire** (Sections A–K, 56 fields, ILO ICLS-19 standards) with dynamic skip logic across three employment paths.

A key thesis contribution is the **Semantic Relation Engine** — a three-way crosswalk that cross-validates ISCO-08, ISIC Rev.4, and ISCED 2011 classifications against each other using ILO correspondence tables, producing a **SemanticCoherence score (0–1)** that adjusts ISCO confidence and triggers HITL escalation on HIGH-severity violations.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Tech Stack](#tech-stack)
3. [Semantic Relation Engine (Thesis Contribution)](#semantic-relation-engine-thesis-contribution)
4. [UAE LFS Questionnaire — Complete Field Reference](#uae-lfs-questionnaire--complete-field-reference)
5. [System Flowcharts](#system-flowcharts)
6. [Skip Logic Gates](#skip-logic-gates)
7. [Security Features](#security-features)
8. [Quick Start (Docker)](#quick-start-docker)
9. [Environment Variables](#environment-variables)
10. [Local Development](#local-development-without-docker)
11. [API Reference](#api-reference)
12. [Running Tests](#running-tests)
13. [Project Structure](#project-structure)
14. [Database Schema](#database-schema-11-tables)
15. [Implementation Status](#implementation-status)

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Browser                                                                   │
│  Next.js 14  (login / OTP → chat interface, EN + AR RTL)                  │
│  Supervisor Review Dashboard (HITL queue)                                  │
└────────────────────────┬───────────────────────────────────────────────────┘
                         │ HTTP (REST / JSON)  X-Request-ID correlation header
┌────────────────────────▼───────────────────────────────────────────────────┐
│  FastAPI Backend                                                           │
│                                                                            │
│  ① LanguageProcessor   — langdetect + Unicode script analysis             │
│     • 6 language codes: en / ar / ar-gulf / ur / hi / tl                  │
│     • Gulf Arabic normalisation (~30 dialect→MSA token replacements)      │
│     • Code-switch detection & per-segment labelling                       │
│     • NER via CrewAI Agent (Ollama / llama3.2, Claude 3.5 fallback)       │
│                                                                            │
│  ② ConversationManager — 5-state FSM (CrewAI + Ollama)                   │
│     GREETING → COLLECTING_INFO ↔ CLARIFYING → VALIDATING → COMPLETING    │
│     56-field UAE LFS questionnaire, 3 employment paths, dynamic skip logic│
│                                                                            │
│  ③ ISCOClassifier — four-stage hierarchical RAG pipeline                  │
│     • Stage 1: major group   (1-digit)  semantic search                   │
│     • Stage 2: sub-major     (2-digit)  parent-filtered search            │
│     • Stage 3: minor group   (3-digit)  parent-filtered search            │
│     • Stage 4: unit group    (4-digit)  parent-filtered + LLM re-ranking  │
│       (LLM skipped when top similarity ≥ 0.92)                            │
│     • HITL escalation when confidence < 0.70                              │
│     • Weighted confidence: 0.10×s1 + 0.20×s2 + 0.20×s3 + 0.50×s4        │
│                                                                            │
│  ④ ISICClassifier   — ISIC Rev.4 full 4-level hierarchy (keyword + LLM)  │
│     • Section (A–U) → Division (2-digit) → Group (3-digit) → Class (4-digit)│
│     • e.g. J → 62 → 620 → 6201 "Computer programming activities"         │
│  ⑤ ISCEDClassifier  — dual ISCED classification (keyword-only)            │
│     • ISCED 2011 attainment level (0–8)                                   │
│     • ISCED-F 2013 field of specialisation (4-digit detailed code)        │
│       Broad (2-digit) → Narrow (3-digit) → Detailed (4-digit)             │
│       e.g. 06 → 061 → 0613 "Software and applications development"        │
│  ⑥ NationalityClassifier — UN M49 + ISO 3166-1 alpha-3 (50 countries)   │
│  ⑦ ValidationAgent  — 10 cross-answer rules R01–R10 (ILO ICLS-19)       │
│  ⑧ PersonRegister   — pre-fill from previous rounds (40–50% fewer Qs)    │
│  ⑨ HITLQualityManager — automated quality scoring + escalation queue     │
│  ⑩ AuditLogger      — immutable GDPR audit trail (10-year retention)     │
│  ⑪ ReportGenerator  — bilingual EN+AR employment report                  │
│  ⑫ EmotionalIntelligence — abandonment-risk detection                    │
│  ⑬ SemanticRelationEngine — ISCO↔ISIC↔ISCED three-way crosswalk          │
│     • SemanticCoherence score 0–1 (ISIC 55% + ISCED 45% weight)          │
│     • Adjusts ISCO confidence: +10% coherent / −20% strong mismatch      │
│     • HIGH-severity violations trigger HITL escalation                    │
│     • ILO ISCO-ISIC table (Geneva 2012) + UNESCO ISCED 2011 Table 7       │
│  ⑭ SurveyOrchestrator — top-level agent coordinator                      │
└──────┬──────────────────────────────┬──────────────────────────────────────┘
       │                              │
┌──────▼──────┐           ┌──────────▼──────────────────────────────────────┐
│  PostgreSQL │           │  Qdrant vector DB                               │
│  Users      │           │  4 hierarchical ISCO-08 collections:            │
│  Sessions   │           │    • isco08_major_groups      (10 groups)       │
│  Responses  │           │    • isco08_submajor_groups   (43 groups)       │
│  HITLQueue  │           │    • isco08_minor_groups     (131 groups)       │
│  QualityRev │           │    • isco08_unit_groups      (441 groups)       │
│  PersonReg  │           │  multilingual-e5-large embeddings               │
│  AuditLogs  │           │  (1024-dim, handles en/ar/ur/hi/tl)             │
│  SurveyRpts │           └─────────────────────────────────────────────────┘
│  (11 tables)│
└──────┬──────┘
       │
┌──────▼──────────────┐    ┌─────────────────────────────────────────────────┐
│  Redis 7            │    │  LLM Routing (llm_client.py)                    │
│  Session context    │    │  TaskType.GENERAL → Ollama / llama3.2           │
│  TTL: 24 h          │    │    temp 0.3 — NER, conversation, summaries      │
│  Key: lfs:session:* │    │    graceful fallback → Claude 3.5 Sonnet        │
└─────────────────────┘    │  TaskType.CRITICAL → Claude 3.5 Sonnet          │
                           │    temp 0.0 — ISCO re-rank, validation, reports │
                           └─────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, React 18, Tailwind CSS, Noto Sans Arabic (RTL support) |
| Backend | FastAPI, SQLAlchemy, Alembic |
| AI Agents | CrewAI, Ollama / llama3.2 (general tasks), Claude 3.5 Sonnet (critical tasks) |
| Semantic Crosswalk | Deterministic ILO/UNESCO lookup tables; no LLM for core logic (speed + auditability) |
| Embeddings | `intfloat/multilingual-e5-large` (sentence-transformers, 1024-dim) |
| Vector DB | Qdrant (4 hierarchical collections, 441 ISCO-08 unit groups) |
| Auth | Email OTP (Gmail SMTP / SendGrid fallback) → JWT (HS256) + sliding-window rate limiting |
| Database | PostgreSQL 15 (11 tables, soft-delete on users/sessions/responses) |
| Cache | Redis 7 (conversation context, TTL 24 h) |
| Security | CORS allowlist, X-Request-ID middleware, OTP brute-force lockout |

---

## Classification Standards

The system applies four complementary ILO/UNESCO classification standards to every survey response:

| Standard | Scope | Output depth | Example |
|---|---|---|---|
| **ISCO-08** | Occupation (job title + duties) | 4-digit unit group | `2512` — Software Developers |
| **ISIC Rev.4** | Industry (employer's business) | Section → Division → Group → **4-digit Class** | `J` → `62` → `620` → `6201` Computer Programming |
| **ISCED 2011** | Education attainment level | Level 0–8 | Level `6` — Bachelor's or equivalent |
| **ISCED-F 2013** | Education field of specialisation | Broad → Narrow → **4-digit Detailed** | `06` → `061` → `0613` Software & Applications Dev |

> ISCED 2011 (level) and ISCED-F 2013 (field) are produced simultaneously by `ISCEDClassifier` in a single two-pass keyword scan. The Semantic Relation Engine uses the ISCED 2011 *level* for its crosswalk with ISCO-08 and ISIC Rev.4.

---

## Semantic Relation Engine (Thesis Contribution)

The **Semantic Relation Engine** (`backend/agents/semantic_relation.py`) is the novel contribution of this thesis. It cross-validates the three international classification outputs — ISCO-08 (occupation), ISIC Rev.4 (industry, 4-digit class), ISCED 2011 (attainment level, from the dual ISCED classifier) — against each other to detect inconsistencies that a single-standard classifier would miss.

### How it works

```
Given one respondent's classifications:
  ISCO code  2211   →  Major group 2 "Professionals"
  ISIC section  Q   →  "Human Health & Social Work"
  ISCED level   7   →  "Master's or equivalent"

Step 1 — ISCO↔ISIC check (ILO Geneva 2012 table)
  Major group 2 is compatible with sections: J, M, Q, P, K, L, R, N
  Section Q is in that list → COMPATIBLE ✓

Step 2 — ISCO↔ISCED check (UNESCO ISCED 2011 Op. Manual Table 7)
  Major group 2 expected ISCED range: min=6 typical=7 max=8
  Level 7 is within [6, 8] → COMPATIBLE ✓

Step 3 — Weighted coherence score
  isic_score  = 1.0  (no violation)
  isced_score = 1.0  (no violation)
  final_score = 0.55 × 1.0 + 0.45 × 1.0 = 1.00

Step 4 — Confidence adjustment
  score >= 0.90  →  +10% to ISCO confidence
```

### Violation severity

| Severity | Condition | Confidence Δ | HITL |
|---|---|---|---|
| None | score ≥ 0.90 | +10% | No |
| LOW | score ≥ 0.70 | +5% | No |
| MODERATE | score ≥ 0.50 | −5% | No |
| HIGH | score < 0.50 | −20% | Yes → HITLQueue |

### Sub-major group rules

Stricter rules apply for specific sub-major groups, e.g.:
- Sub-major `22` (Health Professionals): ISCED ≥ 7 required
- Sub-major `25` (ICT Professionals): ISIC section J preferred
- Sub-major `11` (Chief Executives): ISCED ≥ 6 required

### Demo (10 test cases)

```bash
python -m backend.evaluation.semantic_demo
```

Output shows colour-coded results: green = coherent, yellow = moderate mismatch, red = high violation.

### API response

The `semantic_coherence` field is included in every `/survey/sessions/{id}/message` response when ISCO + ISIC + ISCED data are available:

```jsonc
"semantic_coherence": {
  "isco_code": "2211",
  "isic_section": "Q",
  "isced_level": 7,
  "score": 1.0,
  "is_coherent": true,
  "isco_isic_compatible": true,
  "isco_isced_compatible": true,
  "violations": [],
  "confidence_adjustment": 0.10,
  "explanation_en": "Strong semantic alignment: occupation (Professionals), industry (Human Health & Social Work), and education (Master's Equivalent) are fully consistent.",
  "explanation_ar": "...",
  "major_group": "2",
  "major_label": "Professionals",
  "isic_label": "Human Health & Social Work",
  "isced_label": "Master's or equivalent"
}
```

The report page (`/report`) displays a **Cross-Standard Coherence** panel with score bar, compatibility flags, and violation list in EN or AR based on the user's language preference.

---

## UAE LFS Questionnaire — Complete Field Reference

The system implements the **complete UAE Labour Force Survey** (Sections A–K) following ILO ICLS-19 standards. Questions adapt dynamically based on employment status via skip logic.

### Section A — Authentication / Identification
> Handled automatically by the auth system (OTP + JWT). No conversational questions.

---

### Section B — Demographics (All Paths)

| # | Field | Question (EN) | Question (AR) | Options |
|---|---|---|---|---|
| B1 | `gender` | What is your gender? | ما هو جنسك؟ | Male / Female / Prefer not to say |
| B3 | `nationality` | What is your nationality? | ما جنسيتك؟ | UAE National / Expat — Arab / Expat — Non-Arab / Prefer not to say |
| B4 | `marital_status` | What is your marital status? | ما حالتك الاجتماعية؟ | Single / Married / Divorced / Widowed |
| B5 | `education_level` | What is your highest level of education completed? | ما أعلى مستوى تعليمي أتممته؟ | No formal / Primary / Lower secondary / Upper secondary / Diploma / Bachelor / Master / PhD / Vocational |
| B5a | `field_of_study` | What was your main field of study? *(if Bachelor/Master/PhD)* | ما تخصصك الدراسي الرئيسي؟ | Free text |
| B7 | `vocational_training` | Have you received any vocational or technical training in the past 12 months? | هل تلقيت تدريبًا مهنيًا أو تقنيًا خلال الـ 12 شهرًا الماضية؟ | Yes / No |
| B8 | `emirate` | Which emirate do you currently reside in? | في أي إمارة تقيم حاليًا؟ | Abu Dhabi / Dubai / Sharjah / Ajman / Umm Al Quwain / Ras Al Khaimah / Fujairah |
| B9 | `uae_residence_duration` | How long have you been residing in the UAE? | منذ متى وأنت مقيم في الإمارات؟ | Born here / < 1 yr / 1–5 yrs / 6–10 yrs / > 10 yrs |

---

### Section C — Employment Status & Current Job (Employed Path)

| # | Field | Question (EN) | Question (AR) | Options / Type |
|---|---|---|---|---|
| C1 | `employment_status` | Are you currently employed, unemployed, or not in the labour force? | هل أنت حاليًا موظف، عاطل عن العمل، أم خارج سوق العمل؟ | Employed / Unemployed / Not in labour force |
| C3 | `employment_nature` | What best describes your employment situation? | ما الذي يصف وضعك الوظيفي بشكل أفضل؟ | Paid employee / Self-employed / Employer / Unpaid family worker |
| C4 | `employment_sector` | Do you work in the government or private sector? | هل تعمل في القطاع الحكومي أم الخاص؟ | Government / Private / Semi-government / NGO / International org |
| C5 | `job_title` | What is your current job title or occupation? | ما مسماك الوظيفي الحالي؟ | Free text → ISCO-08 classified |
| C5a | `job_duties` | Briefly describe your main duties and responsibilities. | صف باختصار مهامك ومسؤولياتك الرئيسية. | Free text |
| C6 | `industry` | What industry or business sector do you work in? | في أي صناعة أو قطاع أعمال تعمل؟ | Free text → ISIC Rev.4 classified |

---

### Section D — Hours & Working Arrangements (Employed Path)

| # | Field | Question (EN) | Question (AR) | Options / Type |
|---|---|---|---|---|
| D1 | `actual_hours_worked` | How many hours did you actually work last week? | كم ساعة عملت فعليًا الأسبوع الماضي؟ | Free text (number) |
| D2 | `hours_per_week` | How many hours per week do you usually work in your main job? | كم ساعة في الأسبوع تعمل عادةً في وظيفتك الرئيسية؟ | Free text (number) |
| D3 | `secondary_job` | Do you have any secondary or additional jobs besides your main job? | هل لديك وظيفة ثانية أو إضافية؟ | Yes / No |
| D4 | `secondary_job_hours` | How many hours per week do you work in your secondary job? *(if D3=yes)* | كم ساعة في الأسبوع تعمل في وظيفتك الثانية؟ | Free text (number) |
| D5 | `underemployment` | Would you like to work more hours than you currently do? | هل تودّ العمل ساعات أكثر مما تعمله حاليًا؟ | Yes / No |
| D6 | `employment_type` | What is your employment arrangement? | ما ترتيب عملك؟ | Full-time / Part-time / Seasonal / Casual |
| D7 | `contract_type` | Do you have a written employment contract? *(if paid_employee)* | هل لديك عقد عمل مكتوب؟ | Permanent / Fixed-term / Temporary / No contract |
| D8 | `remote_work` | Does your job allow working from home or remotely? | هل وظيفتك تتيح العمل من المنزل أو عن بُعد؟ | Always / Sometimes / Never |

---

### Section E — Wages & Benefits (Employed — Paid Employees Only)

> **E1 is skipped for employers and self-employed** — they complete Section E via business income questions instead.

| # | Field | Question (EN) | Question (AR) | Options |
|---|---|---|---|---|
| E1 | `monthly_wage_range` | What is your approximate monthly salary range? (AED) *(paid_employee only)* | ما هو نطاق راتبك الشهري التقريبي؟ (بالدرهم) | < 5,000 / 5,000–10,000 / 10,001–20,000 / 20,001–50,000 / > 50,000 / Prefer not to say |
| E2 | `salary_allowances` | Do you receive any allowances (housing, transport, etc.) in addition to your basic salary? | هل تحصل على بدلات إضافية؟ | Yes / No / Prefer not to say |
| E3 | `bonuses` | Did you receive any bonuses or performance-related pay in the past 12 months? | هل حصلت على مكافآت خلال الـ 12 شهرًا الماضية؟ | Yes / No / Prefer not to say |
| E4 | `health_insurance` | Does your employer provide health insurance? | هل يوفر صاحب العمل تأمينًا صحيًا؟ | Yes (full) / Yes (partial) / No |
| E5 | `pension_scheme` | Are you enrolled in a pension or end-of-service benefits scheme? | هل أنت مسجل في نظام تقاعد أو مكافأة نهاية الخدمة؟ | Yes (GPSSA) / Yes (DEWS/private) / No |

---

### Section F — Unemployment & Outside Labour Force

#### Unemployed Path

| # | Field | Question (EN) | Question (AR) | Options |
|---|---|---|---|---|
| F1 | `job_search_active` | Over the past four weeks, have you been actively looking for a job? | خلال الأسابيع الأربعة الماضية، هل كنت تبحث بنشاط عن عمل؟ | Yes / No |
| F2 | `job_search_methods` | What methods have you used to search for work? *(if F1=yes)* | ما الأساليب التي استخدمتها للبحث عن عمل؟ | Free text |
| F3 | `available_for_work` | If a suitable job were offered today, would you be available to start within two weeks? | إذا عُرضت وظيفة مناسبة اليوم، هل ستكون متاحًا للبدء خلال أسبوعين؟ | Yes / No |
| F4 | `unemployment_duration` | How long have you been looking for work? | منذ متى وأنت تبحث عن عمل؟ | Free text |
| F5 | `desired_job_type` | What type of job or occupation are you looking for? | ما نوع الوظيفة أو المهنة التي تبحث عنها؟ | Full-time / Part-time / Any |
| F7 | `ever_worked` | Have you ever worked before? | هل عملت من قبل؟ | Yes, within 1 yr / Yes, 1–3 yrs ago / Yes, > 3 yrs ago / Never worked |

> **ILO ICLS-19 re-routing rule:** If F1=No AND F3=No → respondent is automatically reclassified as **not in labour force** and redirected to the Outside LF path (F6 question).

#### Outside Labour Force Path

| # | Field | Question (EN) | Question (AR) | Options |
|---|---|---|---|---|
| F6 | `outside_lf_reason` | What is the main reason you are not looking for work? | ما السبب الرئيسي لعدم بحثك عن عمل؟ | Housework / Student / Retired / Disabled / Discouraged / Other |
| F7 | `ever_worked` | Have you ever worked before? | هل عملت من قبل؟ | Yes, within 1 yr / Yes, 1–3 yrs ago / Yes, > 3 yrs ago / Never worked |

---

### Section G — Previous Employment (All Paths — if ever_worked ≠ never_worked)

| # | Field | Question (EN) | Question (AR) | Options / Type |
|---|---|---|---|---|
| G1 | `last_job_title` | What was your most recent job title? | ما كان مسماك الوظيفي الأخير؟ | Free text → ISCO-08 classified |
| G2 | `last_job_sector` | Which sector was your last employer in? | في أي قطاع كان صاحب العمل الأخير؟ | Government / Private / Semi-government / NGO |
| G3 | `reason_left_job` | What was the main reason you left your last job? *(unemployed path only)* | ما السبب الرئيسي لتركك آخر وظيفة؟ | Laid off / Contract ended / Resigned / Business closed / Family reasons / Other |
| G4 | `highest_previous_salary` | What was your highest monthly salary in your previous employment? | ما كان أعلى راتب شهري في وظيفتك السابقة؟ | < 5,000 / 5,000–10,000 / 10,001–20,000 / 20,001–50,000 / > 50,000 / Prefer not to say |

---

### Section H — Skills & Training (All Paths)

| # | Field | Question (EN) | Question (AR) | Options / Type |
|---|---|---|---|---|
| H1 | `main_skills` | What are your main professional skills? | ما مهاراتك المهنية الرئيسية؟ | Free text |
| H2 | `qualification_match` | Does your current job match your qualifications and skills? *(employed only)* | هل وظيفتك الحالية تتوافق مع مؤهلاتك ومهاراتك؟ | Well matched / Over-qualified / Under-qualified |
| H3 | `training_participation` | Have you participated in any training or skills development in the past year? | هل شاركت في أي تدريب أو تطوير مهارات خلال العام الماضي؟ | Yes / No |
| H4 | `emiratization_program` | Have you participated in any Emiratisation programme? *(UAE nationals only)* | هل شاركت في برنامج تعزيز توظيف المواطنين؟ | Yes / No / Currently enrolled |
| H5 | `labour_market_barriers` | What main barriers do you face in finding or keeping employment? | ما العوائق الرئيسية التي تواجهها في العثور على عمل أو الاحتفاظ به؟ | Free text |

---

### Section I — Digital Economy & Platform Work (All Paths)

| # | Field | Question (EN) | Question (AR) | Options |
|---|---|---|---|---|
| I1 | `platform_work` | Do you work through online platforms (Uber, Fiverr, Noon, etc.)? | هل تعمل عبر منصات رقمية؟ | Yes (primary income) / Yes (supplementary) / No |
| I2 | `platform_names` | Which platforms do you work through? *(if I1=yes)* | ما المنصات التي تعمل عبرها؟ | Free text |
| I3 | `platform_hours` | How many hours per week do you work through platforms? *(if I1=yes)* | كم ساعة في الأسبوع تعمل عبر المنصات؟ | Free text |
| I4 | `online_business` | Do you run an online business, sell products online, or provide freelance services? | هل تدير عملًا تجاريًا إلكترونيًا أو تبيع منتجات عبر الإنترنت؟ | Yes / No |

---

### Section J — Job Satisfaction & Quality of Work (Employed Path)

| # | Field | Question (EN) | Question (AR) | Options |
|---|---|---|---|---|
| J1 | `job_satisfaction` | Overall, how satisfied are you with your current job? | بشكل عام، ما مدى رضاك عن وظيفتك الحالية؟ | Very satisfied / Satisfied / Neutral / Dissatisfied / Very dissatisfied |
| J2 | `work_safety` | Do you feel your workplace is safe and healthy? | هل تشعر أن بيئة عملك آمنة وصحية؟ | Yes / Mostly / No |
| J3 | `workplace_issues` | Have you experienced any workplace issues in the past year? (harassment, discrimination, etc.) | هل واجهت أي مشكلات في مكان العمل خلال العام الماضي؟ | Free text |
| J4 | `work_life_balance` | How would you rate your work-life balance? | كيف تقيّم توازنك بين العمل والحياة؟ | Excellent / Good / Fair / Poor |

---

### Section K — Survey Feedback (All Paths)

| # | Field | Question (EN) | Question (AR) | Options |
|---|---|---|---|---|
| K1 | `question_clarity` | How clear and easy to understand were the survey questions? | ما مدى وضوح أسئلة الاستبيان وسهولة فهمها؟ | Very clear / Clear / Somewhat unclear / Very unclear |
| K2 | `difficulty_answering` | Were there any questions you found particularly difficult to answer? | هل كانت هناك أسئلة وجدتها صعبة الإجابة؟ | Free text |
| K3 | `ai_preference` | Would you prefer to be interviewed by an AI system or a human interviewer for future surveys? | هل تفضل إجراء المقابلة مع نظام ذكاء اصطناعي أم مع باحث بشري في الاستبيانات المستقبلية؟ | AI / Human / No preference |
| K4 | `data_confidence` | How confident are you that your answers accurately reflect your situation? | ما مدى ثقتك في أن إجاباتك تعكس وضعك الفعلي بدقة؟ | Very confident / Confident / Somewhat / Not confident |
| K5 | `survey_comments` | Any additional comments or suggestions about this survey? | هل لديك أي تعليقات أو اقتراحات إضافية؟ | Free text |

---

### Field Count by Path

| Path | Core Fields | Conditional Fields | Max Total |
|---|---|---|---|
| **Employed** | 38 | +7 (field_of_study, contract_type, monthly_wage_range, secondary_job_hours, emiratization, platform details ×2) | **45** |
| **Unemployed** | 23 | +9 (field_of_study, job_search_methods, last_job_title/sector, reason_left_job, prev_salary, emiratization, platform details ×2) | **32** |
| **Outside LF** | 20 | +7 (field_of_study, last_job_title/sector, prev_salary, emiratization, platform details ×2) | **27** |
| **Shared base** | 18 | — | 18 (all paths) |

> Totals verified against `ConversationManager._get_field_order()` output for each path with every conditional gate triggered (2026-08-02). The union of all fields across all three paths is exactly **56** distinct fields, matching the questionnaire's Section A–K field count.

---

## System Flowcharts

### 1. Conversation State Machine (FSM)

```
User opens chat
      │
      ▼
 Language Detection (LanguageProcessor)
 en / ar / ar-gulf / ur / hi / tl
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ConversationManager FSM                          │
│                                                                     │
│  ┌───────────┐  any message    ┌─────────────────────────────────┐ │
│  │  GREETING │ ──────────────► │       COLLECTING_INFO           │ │
│  │           │                 │  ask next unanswered field       │ │
│  │ warm intro│                 │  one question at a time          │ │
│  │ + C1 Qn   │                 │  ordered by _get_field_order()  │ │
│  └───────────┘                 └──────────┬────────────┬──────────┘ │
│                                 ambiguous │            │ all fields  │
│                                 answer    ▼            │ collected   │
│                         ┌──────────────────┐          │             │
│                         │    CLARIFYING    │          ▼             │
│                         │ re-ask specific  │    ┌──────────────┐   │
│                         │ field only       │    │  VALIDATING  │   │
│                         └────────┬─────────┘    │ read back all│   │
│                          answer  │              │ answers      │   │
│                          accepted│              └──────┬───────┘   │
│                                  │         confirmed   │ correction  │
│                                  └──────────────────►  │  wanted    │
│                                  ◄──────────────────── │ ──────────►│
│                                                         │            │
│                                                         ▼            │
│                                                  ┌───────────┐      │
│                                                  │ COMPLETING│      │
│                                                  │ (terminal)│      │
│                                                  │ farewell  │      │
│                                                  └───────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2. Field Collection — Three Employment Paths

```
                         C1: employment_status?
                         │
          ┌──────────────┼──────────────────────┐
          │              │                      │
       employed      unemployed          not_in_labour_force
          │              │                      │
          ▼              ▼                      ▼
  ┌───────────┐   ┌────────────┐        ┌─────────────┐
  │  EMPLOYED │   │ UNEMPLOYED │        │  OUTSIDE LF │
  │    PATH   │   │    PATH    │        │    PATH     │
  └─────┬─────┘   └─────┬──────┘        └──────┬──────┘
        │               │                      │
  ─── DEMOGRAPHICS (B1–B9, all paths) ─────────────────
  gender → nationality → marital_status → emirate
  → uae_residence_duration → vocational_training
  [B5a field_of_study — only if education ∈ bachelor/master/phd]
        │               │                      │
        ▼               ▼                      │
  C3 employment_nature  F1 job_search_active   F6 outside_lf_reason
  C4 employment_sector     │                   F7 ever_worked
  C5 job_title          F2 job_search_methods  │  └─ if ever_worked:
  C5a job_duties           └─ only if F1=yes      G1 last_job_title
  C6 industry           F3 available_for_work     G2 last_job_sector
  D1 actual_hours          │                      G4 prev_salary
  D2 hours_per_week        │ ⚡ ILO gate:          │
  D3 secondary_job         │ F1=no AND F3=no?      │
  └─ D4 secondary_hours    │ → reclassify to ──────┘
     only if D3=yes        │   not_in_labour_force
  D5 underemployment    F4 unemployment_duration
  D6 employment_type    F5 desired_job_type
  └─ D7 contract_type   F7 ever_worked
     only if paid_employee  └─ if ever_worked:
  D8 remote_work           G1 last_job_title
  └─ E1 monthly_wage_range G2 last_job_sector
     ⚡ only if paid_employee G3 reason_left_job
  E2 salary_allowances     G4 prev_salary
  E3 bonuses
  E4 health_insurance
  E5 pension_scheme
  H2 qualification_match
  J1 job_satisfaction
  J2 work_safety
  J3 workplace_issues
  J4 work_life_balance
        │               │                      │
        └───────────────┴──────────────────────┘
                        │
          ─── SHARED TAIL (H, I, K — all paths) ─────────────
          H1 main_skills → H3 training_participation
          └─ H4 emiratization_program [only if nationality=uae_national]
          H5 labour_market_barriers
          I1 platform_work
          └─ I2 platform_names [only if I1=yes_primary/supplementary]
          └─ I3 platform_hours [only if I1=yes_primary/supplementary]
          I4 online_business
          K1 question_clarity → K2 difficulty_answering
          K3 ai_preference → K4 data_confidence → K5 survey_comments
                        │
                        ▼
                   VALIDATING → COMPLETING
```

### 3. Infrastructure & Agent Pipeline

```
Frontend (Next.js 14)
  chat.js               supervisor_review.js
  ├── Pill buttons       ├── HITL queue list
  │   39 categorical     │   approve/correct/reject
  │   15 free-text       └── AI reasoning expandable
  └── RTL Arabic
          │ POST /survey/sessions/{id}/message
          │ GET|POST /survey/hitl/*
          ▼
    FastAPI Backend  (X-Request-ID middleware)
          │
          ├─► LanguageProcessor ─────────────────► Ollama (GENERAL)
          │     detect lang + NER                   llama3.2, temp 0.3
          │                                          ↓ if unreachable
          ├─► ConversationManager ──────────────►  Claude 3.5 Sonnet
          │     FSM + field extraction              (fallback or CRITICAL)
          │     heuristic regex extraction
          │     Redis context persistence (ContextMemory)
          │
          ├─► ISCOClassifier ──────────────────►  Qdrant (4 collections)
          │     4-stage hierarchical RAG            hierarchical search
          │     LLM re-rank if confidence < 0.92 ► Claude 3.5 Sonnet
          │     HITL flag if confidence < 0.70  ► HITLQueue (DB)
          │
          ├─► ISICClassifier ──────────────────►  keyword → LLM
          │     ISIC Rev.4 4-level: Section→Division→Group→Class (4-digit)
          │
          ├─► ISCEDClassifier ─────────────────►  keyword only (two-pass)
          │     ISCED 2011 attainment level (0–8)
          │     + ISCED-F 2013 field of specialisation (4-digit detailed)
          │
          ├─► NationalityClassifier ───────────►  keyword + alias lookup
          │     UN M49 + ISO 3166-1 alpha-3
          │     50 countries, EN+AR+UR/HI/TL aliases
          │
          ├─► SemanticRelationEngine ──────────►  deterministic crosswalk
          │     ISCO↔ISIC↔ISCED coherence score
          │     ±5–20% confidence adjustment
          │
          ├─► ValidationAgent ─────────────────►  Claude 3.5 Sonnet
          │     10 ILO ICLS-19 rules (R01–R10)
          │     runs in VALIDATING state
          │
          ├─► EmotionalIntelligence ───────────►  Ollama
          │     abandonment-risk detection
          │     culturally adapted support messages
          │
          ├─► ReportGenerator ─────────────────►  Claude 3.5 Sonnet
          │     bilingual EN+AR report
          │
          └─► AuditLogger ─────────────────────►  PostgreSQL
                GDPR trail + subject access          immutable append-only
                SESSION_STARTED / MESSAGE_SENT
                ISCO decisions, HITL escalations
```

---

## Skip Logic Gates

All conditional fields are evaluated at call time by `_get_field_order(collected_data)`. The field list is recomputed after every user turn, enabling real-time path switching.

| Gate | Trigger Condition | Effect |
|---|---|---|
| **B5a field_of_study** | `education_level ∈ {bachelor, master, phd}` | Insert field_of_study after education_level |
| **D4 secondary_job_hours** | `secondary_job == "yes"` | Insert secondary_job_hours after secondary_job |
| **D7 contract_type** | `employment_nature == "paid_employee"` | Insert contract_type after employment_type |
| **E1 wage gate** ⚡ | `employment_nature == "paid_employee"` | Insert monthly_wage_range — **skipped for employers/self-employed** |
| **F2 job_search_methods** | `job_search_active == "yes"` | Insert job_search_methods after job_search_active |
| **F3 ILO re-route** ⚡ | `job_search_active == "no"` AND `available_for_work == "no"` | Reclassify `employment_status → not_in_labour_force`; switch to outside-LF path |
| **G-section gate** | `ever_worked != "never_worked"` | Insert last_job_title, last_job_sector, (reason_left_job for unemployed), highest_previous_salary |
| **H4 emiratization** | `nationality == "uae_national"` | Insert emiratization_program after training_participation |
| **I2/I3 platform details** | `platform_work ∈ {yes_primary, yes_supplementary}` | Insert platform_names + platform_hours |
| **HITL escalation** | ISCO confidence < 0.70 | Flag to HITLQueue; supervisor review required |
| **LLM re-rank skip** | ISCO top-1 similarity ≥ 0.92 | Skip Claude re-ranking call (cost/latency saving) |

---

## Security Features

The backend implements layered security controls across authentication, API access, and data lifecycle.

### Rate Limiting (sliding-window, in-process)

| Endpoint | Limit | Window | Key |
|---|---|---|---|
| `POST /auth/request-otp` | 5 requests | 10 minutes | Client IP (X-Forwarded-For aware) |
| `POST /auth/verify-otp` | 10 requests | 10 minutes | Email address |
| `POST /survey/sessions/{id}/message` | 30 requests | 1 minute | Client IP (slowapi `Limiter`) |

- OTP endpoints: `threading.Lock` + `defaultdict(list)` sliding-window in `backend/auth/email_otp.py`
- Message endpoint: `slowapi.Limiter` (`app.state.limiter`, `key_func=get_remote_address`) enforced via the `_check_rate_limit` dependency in `backend/api/survey_routes.py`
- Returns HTTP 429 when limit exceeded
- Rate limiter state is in-process; deploy behind a shared Redis rate limiter for multi-worker setups

### Security Headers

Every response carries a `SecurityHeadersMiddleware`-injected header set (`backend/main.py`):

| Header | Value |
|---|---|
| `Content-Security-Policy` | `default-src 'self'` (same-origin scripts/styles only) |
| `X-Content-Type-Options` | `nosniff` |
| `X-Frame-Options` | `DENY` |
| `Referrer-Policy` | `strict-origin-when-cross-origin` |
| `Permissions-Policy` | `geolocation=(), microphone=(), camera=()` |

### OTP Brute-Force Lockout

- Each OTP code tracks its own failed attempt counter (`_otp_attempts` dict)
- After **5 consecutive wrong codes**, the OTP is permanently locked (no further attempts accepted)
- Respondent must request a fresh OTP

### CORS Allowlist

```python
# Set via CORS_ORIGINS environment variable
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
```

- Allowed methods: `GET, POST, PUT, PATCH, DELETE, OPTIONS`
- Allowed headers: `Content-Type, Authorization, X-Request-ID`
- `allow_credentials=True` — origins are still restricted to the explicit `CORS_ORIGINS` allowlist (never `*`)

### X-Request-ID Correlation

Every request is assigned a UUID `X-Request-ID` (taken from the incoming header if present, generated otherwise). The ID is echoed in every response header, enabling end-to-end tracing through logs.

### Message Validation

- `message` body field: `min_length=1, max_length=2000` — rejects empty or excessively long inputs before any agent processing

### Soft-Delete

Sessions, users, and responses are never physically deleted. A `deleted_at` timestamp is set instead:

| Model | Behaviour |
|---|---|
| `User` | `deleted_at` set; filtered from auth lookups |
| `SurveySession` | `deleted_at` set; hidden from list/get/message endpoints |
| `SurveyResponse` | Retained with parent session for GDPR audit trail |

### Graceful Shutdown

The FastAPI lifespan handler closes the Redis connection and disposes the SQLAlchemy engine on SIGTERM/SIGINT, preventing connection leaks in containerised deployments.

---

## Quick Start (Windows — Recommended)

```bat
REM From the repo root — starts all services automatically
start.bat
```

`start.bat` handles everything: Docker (Postgres + Redis + Qdrant), Ollama model pull, ISCO data load, Alembic migrations, backend and frontend in separate windows.

| URL | Service |
|---|---|
| http://localhost:3000 | Frontend (login / chat / report) |
| http://localhost:3000/supervisor_review | Supervisor HITL dashboard |
| http://localhost:8000 | Backend API |
| http://localhost:8000/docs | Interactive API docs (Swagger) |
| http://localhost:11434 | Ollama local LLM |
| http://localhost:6333/dashboard | Qdrant vector DB dashboard |

---

## Quick Start (Docker)

```bash
# 1. Clone
git clone https://github.com/SrkSubbiah89/Multilingual-LFS-Project.git
cd Multilingual-LFS-Project

# 2. Configure
cp .env.example .env
# Edit .env — fill in required secrets (see table below)

# 3. Start everything
docker compose -f docker/docker-compose.yml up --build
```

> **First boot note:** The `multilingual-e5-large` model (~2 GB) is downloaded on
> the first backend start. Subsequent starts use the `model_cache` Docker volume.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

### Required

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Powers Claude 3.5 Sonnet (ISCO re-rank, validation, reports) |
| `JWT_SECRET` | Random secret for signing JWTs (e.g. `openssl rand -hex 32`) |
| `GMAIL_USER` | Gmail address for OTP email delivery (primary) |
| `GMAIL_APP_PASSWORD` | 16-char Gmail App Password (Google Account → Security → App passwords) |

> **SendGrid fallback:** Set `SENDGRID_API_KEY` + `SENDGRID_FROM_EMAIL` if Gmail is not available. The system uses Gmail first; if `GMAIL_APP_PASSWORD` is not set, SendGrid is used automatically.

### Optional / has defaults

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model for general tasks (NER, conversation) |
| `DATABASE_URL` | postgres://… | Overridden automatically in Docker |
| `QDRANT_HOST` | `localhost` | Overridden automatically in Docker |
| `REDIS_URL` | `redis://localhost:6379` | Overridden automatically in Docker |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Token lifetime |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated list of allowed frontend origins |
| `LFS_FAST_MODE` | `false` | When `true`/`1`/`yes`: skips LLM calls for collecting_info/clarifying turns and NER, and skips ISCO LLM re-ranking — deterministic stub responses for fast local dev/demo without Ollama/Claude latency |

> **Note:** `OPENAI_API_KEY` is not required. General LLM tasks run on local Ollama with automatic fallback to Claude 3.5 Sonnet if Ollama is unreachable.

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

# Apply Alembic migrations (HITL + evaluation tables)
alembic upgrade head

# Load full ISCO-08 hierarchy into Qdrant (run once)
python -m backend.rag.load_full_isco

uvicorn backend.main:app --reload
# → http://localhost:8000

# ── Frontend ─────────────────────────────────────
cd frontend
cp .env.local.example .env.local   # set NEXT_PUBLIC_API_URL if needed
npm install
npm run dev
# → http://localhost:3000

# ── Ollama (local LLM) ───────────────────────────
ollama pull llama3.2
ollama serve
# → http://localhost:11434
```

---

## API Reference

### Health & Readiness

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Probes Redis, Qdrant, Ollama (2 s timeout each); returns `{"status": "ok", "services": {...}}` |
| GET | `/ready` | Lightweight readiness check — returns 200 when DB engine is contactable |

```jsonc
// GET /health — example response
{
  "status": "ok",
  "services": {
    "redis":  "ok",
    "qdrant": "ok",
    "ollama": "ok"
  }
}
```

### Auth

> Rate limits: `/auth/request-otp` — 5 req / 10 min per IP · `/auth/verify-otp` — 10 req / 10 min per email

| Method | Endpoint | Body | Description |
|---|---|---|---|
| POST | `/auth/request-otp` | `{"email": "..."}` | Send 6-digit OTP via email |
| POST | `/auth/verify-otp` | `{"email": "...", "code": "..."}` | Verify OTP → JWT (locked after 5 wrong attempts) |
| POST | `/auth/logout` | — | Client-side logout |

### Survey Sessions

| Method | Endpoint | Body | Description |
|---|---|---|---|
| POST | `/survey/sessions` | `{"language": "en"}` | Create session (pre-fills from Person Register if available) |
| GET | `/survey/sessions` | — | List user's active sessions (soft-deleted excluded) |
| GET | `/survey/sessions/{id}` | — | Get session |
| PATCH | `/survey/sessions/{id}/complete` | — | Mark complete |
| DELETE | `/survey/sessions/{id}` | — | Soft-delete session (responses retained for GDPR audit) |

### Conversational Turn

```
POST /survey/sessions/{id}/message
Authorization: Bearer <token>

Body:  { "message": "I work as a software engineer full time" }
       (1–2000 characters)
```

```jsonc
// Response
{
  "reply": "Thank you! Which industry or sector do you work in?",
  "state": "collecting_info",
  "next_field": "industry",
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
      "hitl_required": false,
      "method": "hierarchical_semantic",
      "stage_confidences": { "stage1": 0.91, "stage2": 0.88, "stage3": 0.85, "stage4": 0.91 }
    }
  ],
  "isic_classification": {
    "industry_text": "technology company, software products",
    "section": "J",
    "section_title": "Information and Communication",
    "division_code": "62",
    "division_title": "Computer programming, consultancy and related activities",
    "group_code": "620",
    "group_title": "Computer programming, consultancy and related activities",
    "class_code": "6201",
    "class_title": "Computer programming activities",
    "confidence": 0.95,
    "method": "keyword"
  },
  "isced_classification": {
    "education_text": "bachelor computer science",
    "level": 6,
    "level_title": "Bachelor's or equivalent",
    "broad_code": "06",
    "broad_title": "Information and Communication Technologies",
    "narrow_code": "061",
    "narrow_title": "Information and Communication Technologies",
    "detailed_code": "0613",
    "detailed_title": "Software and applications development and analysis",
    "confidence": 0.92,
    "method": "keyword"
  },
  "nationality_classification": {
    "raw_text": "Indian",
    "m49_code": "356",
    "iso_alpha3": "IND",
    "country_en": "India",
    "country_ar": "الهند",
    "region_en": "Southern Asia",
    "nationality_en": "Indian",
    "nationality_ar": "هندي",
    "confidence": 0.90,
    "method": "alias"
  },
  "semantic_coherence": {
    "isco_code": "2512",
    "isic_section": "J",
    "isced_level": 6,
    "score": 1.0,
    "is_coherent": true,
    "isco_isic_compatible": true,
    "isco_isced_compatible": true,
    "violations": [],
    "confidence_adjustment": 0.10,
    "explanation_en": "Strong semantic alignment: occupation (Professionals), industry (Information & Communication), and education (Bachelor's Equivalent) are fully consistent.",
    "major_group": "2",
    "major_label": "Professionals",
    "isic_label": "Information and Communication",
    "isced_label": "Bachelor's or equivalent"
  },
  "emotional_state": "neutral",
  "emotional_support_message": null,
  "validation_issues": [],
  "is_data_valid": null,
  "session_completed": false,
  "latency_ms": 420
}
```

**New fields added (May 2026):**

| Field | Type | Description |
|---|---|---|
| `emotional_state` | `string \| null` | Detected emotional state: `neutral`, `stressed`, `hesitant`, `disengaged` |
| `emotional_support_message` | `string \| null` | Culturally adapted support message if abandonment risk detected |
| `validation_issues` | `list[string]` | ILO rule violations surfaced during VALIDATING state |
| `is_data_valid` | `bool \| null` | `true` = all R01–R10 rules passed; `null` when not in VALIDATING state |

### Survey Responses (manual override)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/survey/sessions/{id}/responses` | Submit a raw response |
| GET | `/survey/sessions/{id}/responses` | List responses for a session |
| PATCH | `/survey/sessions/{id}/responses/{rid}` | Update ISCO code / confidence |

### Report

| Method | Endpoint | Description |
|---|---|---|
| GET | `/survey/sessions/{id}/report` | Generate bilingual EN+AR employment report |
| GET | `/survey/sessions/{id}/report?regenerate=true` | Force fresh LLM generation |

### HITL Supervisor Review

| Method | Endpoint | Description |
|---|---|---|
| GET | `/survey/hitl/queue?status_filter=pending` | List escalated items (HIGH priority first) |
| POST | `/survey/hitl/review` | Submit approve / correct / reject decision |

```jsonc
// POST /survey/hitl/review body
{
  "escalation_id": 42,
  "action": "correct",   // "approve" | "correct" | "reject"
  "code": "2514",        // corrected ISCO code (required when action == "correct")
  "notes": "Title is closer to Systems Analyst than Software Developer"
}
```

---

## Running Tests

```bash
# From repo root
pip install -r requirements.txt tf-keras
pytest backend/tests/ -v
```

- **1,178 tests** across **25 test files** — all passing as of June 2026
- Zero live infrastructure required — all external calls (DB, Redis, Qdrant, LLM APIs) are mocked or use in-memory fakes (SQLite, FakeRedis)
- Load/stress tests are marked `@pytest.mark.slow` and excluded by default via `pytest.ini`; run them explicitly with `pytest -m slow`

```bash
# Run load tests explicitly (requires a running backend at localhost:8000)
pytest backend/tests/load_test.py -m slow -v
```

> **Teardown noise:** Harmless `ValueError: I/O operation on closed file` from crewai/colorama atexit hook. Exit code is still 0 — not a test failure.

### Evaluation Framework

```bash
# Compare BM25 / Flat vector / Hierarchical RAG on 100 synthetic test cases
python -m backend.evaluation.evaluate --system all --top-k 3

# Arabic test cases
python -m backend.evaluation.evaluate --system all --arabic

# Semantic Relation Engine demo (10 cases, colour output)
python -m backend.evaluation.semantic_demo
```

Results are written to `backend/evaluation/results.csv`, `results_arabic.csv`, `results_approach.csv`.

### External Validation Dataset — WISCO (Phase II)

`backend/evaluation/wisco/` parses the [WISCO](https://doi.org/10.5281/zenodo.8262593)
(World database of ISCO Occupations) dataset for external ISCO-08 classifier validation across
the system's 5 target languages. See `Documentation/Phase_2/Week_1/` for full provenance,
structure inspection, and integrity-check documentation.

```bash
python backend/evaluation/wisco/inspect_wisco.py   # structural report
python backend/evaluation/wisco/analyze_wisco.py   # language/code/industry analysis
python backend/evaluation/wisco/parse_wisco.py     # produces wisco_raw_parsed.json
```

**Attribution (CC-BY-4.0, both required):**
- General use: Tijdens, K.G. (2023). *WISCO occupations_ISCO08_5dgt_55languages_4000titles_with_mapping_surveycodings_20230425*. Netherlands, WageIndicator Foundation/Surveycodings. DOI: [10.5281/zenodo.8262593](https://doi.org/10.5281/zenodo.8262593)
- If `OCC>>INDUSTRY` (NACE 2004 occupation→industry predictions) is used: Belloni, M., Tijdens, K.G. (2017). *Occupation > industry predictions for measuring industry in surveys*, Deliverable 8.11 of the SERISS project, EU Horizon 2020 GA No. 654221. DOI: [10.13140/RG.2.2.31328.02566](https://doi.org/10.13140/RG.2.2.31328.02566)

---

## Project Structure

```
.
├── Dockerfile                  # Backend image
├── .env.example                # Environment variable template
├── requirements.txt
├── alembic.ini
├── pytest.ini                  # Excludes @slow tests by default
├── start.bat                   # One-command Windows dev startup
│
├── backend/
│   ├── main.py                 # FastAPI app + CORS + X-Request-ID middleware + lifespan
│   ├── agents/
│   │   ├── language_processor.py    # Lang detection, Gulf normalisation, NER (6 codes)
│   │   ├── conversation_manager.py  # 5-state FSM, 56 fields, 3 paths, skip logic
│   │   ├── isco_classifier.py       # Four-stage hierarchical ISCO-08 classifier
│   │   ├── isic_classifier.py       # ISIC Rev.4 4-level industry classification (Section→Division→Group→Class)
│   │   ├── isced_classifier.py      # ISCED 2011 attainment level + ISCED-F 2013 field of specialisation (4-digit)
│   │   ├── nationality_classifier.py # UN M49 + ISO 3166-1 alpha-3 (50 countries, EN+AR+UR/HI/TL)
│   │   ├── semantic_relation.py     # ISCO↔ISIC↔ISCED three-way crosswalk (thesis contribution)
│   │   ├── person_register.py       # Person Register pre-fill service
│   │   ├── validation_agent.py      # Cross-answer consistency rules (R01–R10)
│   │   ├── hitl_quality_manager.py  # Quality scoring + escalation queue
│   │   ├── audit_logger.py          # GDPR-compliant immutable audit trail
│   │   ├── emotional_intelligence.py # Survey abandonment detection + support messages
│   │   ├── report_generator.py      # Bilingual EN+AR employment report
│   │   ├── context_memory.py        # Redis-backed session memory (TTL 24 h)
│   │   ├── rag_expert.py            # Hierarchical RAG search helper
│   │   └── survey_orchestrator.py   # Top-level agent coordinator
│   ├── api/
│   │   ├── auth_routes.py      # OTP + JWT endpoints (per-IP + per-email rate limiting)
│   │   └── survey_routes.py    # Session + message + HITL endpoints
│   ├── auth/
│   │   ├── email_otp.py        # OTP generation, delivery (SendGrid), sliding-window rate limiter, brute-force lockout
│   │   └── jwt_handler.py      # JWT creation & verification (HS256)
│   ├── database/
│   │   ├── models.py           # SQLAlchemy models (11 tables, soft-delete columns)
│   │   ├── connection.py       # DB engine & session
│   │   └── migrations/         # Alembic migration versions
│   │       └── versions/
│   │           ├── 001_initial_schema.py
│   │           ├── 976b9b9c96d4_add_hitlqueue_evaluation_tables.py
│   │           └── f514fcb81c72_add_deleted_at_soft_delete_columns.py
│   ├── llm/
│   │   └── llm_client.py       # LLM factory: Ollama (GENERAL) / Claude (CRITICAL)
│   ├── rag/
│   │   ├── vector_store.py     # Qdrant flat search + multilingual-e5-large
│   │   ├── hierarchical_store.py  # 4-stage hierarchical ISCO RAG (singleton)
│   │   └── load_full_isco.py   # Populate ISCO-08 unit groups into Qdrant — currently loads 441;
│   │   │                       # the true ISCO-08 standard has 436 (confirmed against WISCO,
│   │   │                       # see Documentation/Phase_2/Week_1/); 19 of the 441 are non-standard
│   │   │                       # codes and 14 real unit groups are missing — known defect, not yet fixed
│   ├── evaluation/
│   │   ├── evaluate.py         # BM25 / Flat / Hierarchical 3-system comparison (100 synthetic cases)
│   │   ├── run_comparison.py   # Batch runner for all three systems
│   │   ├── semantic_demo.py    # 10-case three-way crosswalk demo (colour output)
│   │   ├── results.csv         # English evaluation results
│   │   ├── results_arabic.csv  # Arabic evaluation results
│   │   ├── results_approach.csv # Per-approach comparison summary
│   │   └── wisco/              # Phase II Module A: WISCO external validation dataset pipeline
│   │       ├── inspect_wisco.py / analyze_wisco.py / parse_wisco.py
│   │       ├── requirements.lock.txt
│   │       └── data/raw|interim|processed/
│   └── tests/                  # 25 test files (+ conftest + slow-marked load_test), 1,178 tests, zero live infra required
│       ├── conftest.py              # Shared fixtures: in-memory DB, auth client, rate limiter reset
│       ├── test_auth_routes.py
│       ├── test_auth_and_api_extended.py
│       ├── test_survey_routes.py
│       ├── test_conversation_manager.py
│       ├── test_conversation_manager_extended.py
│       ├── test_language_processor.py
│       ├── test_language_processor_extended.py
│       ├── test_isco_classifier.py
│       ├── test_isco_classifier_extended.py
│       ├── test_isic_classifier.py
│       ├── test_isced_classifier.py
│       ├── test_person_register.py
│       ├── test_hitl_quality_manager.py
│       ├── test_hitl_and_e2e_extended.py
│       ├── test_validation_agent.py
│       ├── test_validation_agent_extended.py
│       ├── test_context_memory.py
│       ├── test_audit_logger.py
│       ├── test_emotional_intelligence.py
│       ├── test_rag_expert.py
│       ├── test_vector_store.py
│       ├── test_survey_orchestrator.py
│       ├── test_report_generator.py
│       ├── test_evaluation.py
│       └── load_test.py             # @pytest.mark.slow — excluded by default
│
├── frontend/
│   ├── Dockerfile              # Next.js multi-stage image
│   ├── pages/
│   │   ├── _document.js          # Custom HTML document (fonts, RTL meta)
│   │   ├── _app.js               # Global layout
│   │   ├── index.js              # Login (email → OTP → JWT, auto-fill in dev)
│   │   ├── chat.js               # Survey conversation + quick-reply pills (39 fields)
│   │   ├── report.js             # Bilingual employment report + coherence panel
│   │   └── supervisor_review.js  # HITL supervisor review dashboard
│   ├── components/
│   │   ├── api.js              # Fetch wrapper (auth, sessions, HITL queue)
│   │   └── LanguageToggle.js   # EN / AR / UR / HI / TL switcher
│   └── styles/globals.css
│
├── docker/
│   └── docker-compose.yml      # 5 services: backend, frontend, postgres, qdrant, redis
│
└── Documentation/
    ├── CLAUDE.md               # Archived — original project-kickoff prompt, superseded by this README
    ├── Test_Suite_Report.md    # Full per-test breakdown, regenerated from live pytest runs
    ├── Phase_1_Summary/        # Phase 1 (this system) status snapshot + doc audit
    ├── Phase_2/                # Phase II (thesis pilot study) — one folder per week
    │   └── Week_1/             # Module A (WISCO) + Module E (ethics) artefacts
    ├── Implementation/         # Gap-analysis docx versions
    └── Questionaries/
        ├── UAE_LFS_Questionnaire_Complete.docx
        └── questionnaire_text.txt
```

---

## Database Schema (11 Tables)

| Table | Soft-delete | Purpose |
|---|---|---|
| `users` | `deleted_at` | Registered respondents (email, OTP, JWT) |
| `otp_codes` | — | Time-limited 6-digit OTP codes |
| `survey_sessions` | `deleted_at` | Survey session per user per round |
| `survey_responses` | `deleted_at` | Individual field answers + ISCO codes (retained after session soft-delete for GDPR) |
| `audit_logs` | — | Immutable GDPR audit trail (agent decisions) |
| `data_access_logs` | — | Per-row PII access log (GDPR Art. 15) |
| `agent_decision_logs` | — | Full agent reasoning traces |
| `quality_reviews` | — | HITLQualityManager scoring records |
| `hitl_queue` | — | Low-confidence ISCO items pending supervisor review |
| `survey_report_records` | — | Cached bilingual employment reports |
| `person_register` | — | Pre-fill data from previous survey rounds |

> Soft-deleted rows are filtered from all application queries but retained for audit and GDPR compliance. Physical deletion is handled by the AuditLogger purge process after the configured retention period.

---

## Implementation Status

All thesis requirements fully implemented as of June 2026:

| Component | Status | Detail |
|---|---|---|
| Email OTP + JWT Auth | Done | Gmail SMTP (SendGrid fallback) + HS256 JWT, auto-fill in dev mode |
| Language Detection (6 codes) | Done | en / ar / ar-gulf / ur / hi / tl; Devanagari fast-path |
| Gulf Arabic Normalisation | Done | ~30 dialect→MSA token replacements before NER + embedding |
| Code-Switch Detection | Done | Arabic+Latin and Devanagari+Latin mixing detection |
| NER — 5 languages | Done | CrewAI agent, JOB_TITLE / INDUSTRY / LOCATION / EDUCATION |
| Conversation FSM (5 states) | Done | 56 fields, 3 employment paths, 11 conditional skip gates |
| UAE LFS Questionnaire (A–K) | Done | All sections implemented; dynamic skip logic per ILO ICLS-19 |
| ILO ICLS-19 E1 wage gate | Done | monthly_wage_range skipped for employer/self-employed |
| ILO ICLS-19 F3 re-routing | Done | F1=no + F3=no → auto-reclassify to not_in_labour_force |
| ISCO-08 Knowledge Base | Done | 441 unit groups across 4 Qdrant collections |
| Hierarchical RAG (4-stage) | Done | Major→Sub-major→Minor→Unit with parent_code filtering |
| Per-stage Confidence Scoring | Done | Weighted: 0.10×s1 + 0.20×s2 + 0.20×s3 + 0.50×s4 |
| LLM Re-ranking | Done | Claude 3.5 Sonnet; skipped when top-1 similarity ≥ 0.92 |
| HITL Escalation (< 0.70) | Done | HITLQueue DB + priority ordering (HIGH first) |
| Supervisor Review Dashboard | Done | Approve / correct / reject with inline form |
| Evaluation Framework | Done | BM25 / Flat / Hierarchical 3-system, 100 synthetic cases |
| Validation Agent (R01–R10) | Done | ILO ICLS-19 cross-answer consistency rules; wired into VALIDATING FSM state |
| ISCO-08 Keyword Pre-filter | Done | Major-group anchor prevents semantic drift |
| ISCO-08 Synonym Enrichment | Done | 50+ unit-group descriptions enriched with synonyms for better recall |
| ISIC Rev.4 Classification (4-digit) | Done | Full Section→Division→Group→Class hierarchy; keyword + LLM; 100+ class entries; EN+AR |
| ISCED-F 2013 Field of Specialisation (4-digit) | Done | ISCED-F 2013 Broad→Narrow→Detailed (0011–1041); keyword two-pass; 11 broad fields, 60+ detailed codes; EN+AR |
| ISCED 2011 Attainment Level | Done | Levels 0–8, combined with ISCED-F in single classifier; EN+AR |
| UN M49 Nationality Classification | Done | 50 countries, ISO 3166-1 alpha-3 + UN M49 codes, EN+AR+UR/HI/TL aliases; "United Kingdom"/"South Africa" alias collision fixed |
| Person Register Pre-fill | Done | 40–50% question reduction from previous round |
| Emotional Intelligence Monitor | Done | Abandonment-risk detection + culturally adapted responses; wired into every message turn |
| Audit / GDPR Compliance | Done | Immutable trail, Art. 15 subject-access, 10-year retention; SESSION_STARTED + MESSAGE_SENT events |
| Redis Context Persistence | Done | ContextMemory.load_session / save_session / delete_session around every turn |
| Report Generator | Done | Bilingual EN+AR employment profile + recommendations |
| Quick-reply Pill Buttons | Done | 39 categorical fields × 5 languages in frontend |
| RTL Arabic Support | Done | Noto Sans Arabic, full RTL layout in chat + report |
| **Semantic Relation Engine** | **Done** | ISCO↔ISIC↔ISCED three-way crosswalk; SemanticCoherence score; confidence adjustment ±5–20%; HITL on HIGH violations |
| Cross-Standard Coherence API | Done | `semantic_coherence` field in every `/survey/sessions/{id}/message` response |
| Coherence Report Panel | Done | Score bar, ISIC/ISCED compatibility flags, violation list (EN+AR) on report page |
| Semantic Demo Script | Done | `python -m backend.evaluation.semantic_demo` — 10 cases, colour output for presentation |
| **Rate Limiting** | **Done** | Sliding-window per-IP (OTP request) + per-email (OTP verify), in-process; plus `slowapi` 30 req/min per-IP limiter on `/message` |
| **OTP Brute-force Lockout** | **Done** | Locks OTP after 5 consecutive wrong codes; prevents credential stuffing |
| **Security Headers Middleware** | **Done** | CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy on every response |
| **LFS_FAST_MODE** | **Done** | Deterministic stub responses + skipped LLM/NER/re-ranking calls for fast local dev without live Ollama/Claude |
| **CORS Allowlist** | **Done** | Explicit origin/method/header allowlist from `CORS_ORIGINS` env var; no wildcard |
| **X-Request-ID Middleware** | **Done** | UUID correlation header echoed in every response for end-to-end tracing |
| **Soft-Delete** | **Done** | `deleted_at` on User, SurveySession, SurveyResponse; enforced in all query paths including report_generator, list_responses, update_response, HITL review |
| **Message Length Validation** | **Done** | 1–2000 char `min_length`/`max_length` on message body; rejects garbage before agent pipeline |
| **Graceful Shutdown** | **Done** | Lifespan handler closes Redis + disposes DB engine on SIGTERM |
| **N+1 Query Elimination** | **Done** | `joinedload(SurveySession.responses)` in create_session pre-fill path |
| **EmotionalIntelligence Wiring** | **Done** | Runs on every message > 10 chars; populates `emotional_state` + `emotional_support_message` in response |
| **ValidationAgent Wiring** | **Done** | Runs when FSM enters VALIDATING; populates `validation_issues` + `is_data_valid` in response |
| **HITL Auto-enqueue** | **Done** | `db.flush()` + `HITLQueue` insert when `clf.hitl_required=True`; AuditLogger records decision |
| **Load Test Suite** | **Done** | `@pytest.mark.slow` test in `load_test.py`; 5 users, 2 workers, ≥80% success rate assertion |
| **Test Suite** | **Done** | 1,178 tests, 25 files, zero live infrastructure; `pytest.ini` excludes slow tests by default |
| **Nationality Quick-Options** | **Done** | Pills updated to top 8 UAE nationalities (Emirati/Indian/Pakistani/Filipino/Bangladeshi/Egyptian/British/Other) |
| **Correction Acknowledgment** | **Done** | `_dev_stub_response` now shows "Got it, I've updated that" when correction applied in VALIDATING state |
