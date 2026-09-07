"""
backend/agents/conversation_manager.py

CrewAI-based Conversation Manager agent that drives a UAE Labour Force Survey
through a five-state finite state machine.

States
------
greeting        → welcome the respondent and confirm language preference
collecting_info → ask LFS survey questions one at a time
clarifying      → probe ambiguous or incomplete answers
validating      → read back and confirm collected answers
completing      → thank respondent and close the session

UAE LFS Questionnaire Coverage (ILO ICLS-19 standards) — Sections A-K
-----------------------------------------------------------------------
All paths   : employment_status (C1), education_level (B5),
              gender (B1), nationality (B3), marital_status (B4),
              emirate (B8), uae_residence_duration (B9),
              vocational_training (B7),
              main_skills (H1), training_participation (H3),
              labour_market_barriers (H5),
              platform_work (I1), online_business (I4),
              question_clarity (K1), ai_preference (K3), data_confidence (K4)

Employed    : employment_nature (C3), employment_sector (C4),
              job_title (C5), job_duties (C5a), industry (C6),
              actual_hours_worked (D1), hours_per_week (D2),
              secondary_job (D3), underemployment (D5),
              employment_type (D6),
              contract_type (D7) — if paid_employee,
              remote_work (D8),
              monthly_wage_range (E1) — if paid_employee, salary_allowances (E2),
              health_insurance (E4),
              qualification_match (H2),
              job_satisfaction (J1), work_life_balance (J4)

Unemployed  : job_search_active (F1),
              job_search_methods (F2) — if job_search_active=yes,
              available_for_work (F3) — if no+no → reclassify as not_in_labour_force, unemployment_duration (F4),
              desired_job_type (F5), ever_worked (F7),
              last_job_title (G1) — if ever_worked,
              reason_left_job (G3) — if ever_worked

Outside LF  : outside_lf_reason (F6), ever_worked (F7),
              last_job_title (G1) — if ever_worked

Supported languages: English ("en"), Arabic ("ar"), Urdu ("ur"),
                     Hindi ("hi"), Filipino/Tagalog ("tl")
LLM: Llama 3.2 via Ollama (with Claude 3.5 Sonnet fallback)
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from crewai import Agent, Crew, Task

from backend.llm import TaskType, get_llm

_logger = logging.getLogger(__name__)
_APP_ENV = os.getenv("APP_ENV", "development").lower()
# Set LFS_FAST_MODE=true to skip LLM for collecting_info/clarifying turns and
# use the deterministic rule-based stub instead.  Reduces per-turn latency from
# ~120 s to <1 s on CPU hardware.  Recommended for demos and testing.
_FAST_MODE = os.getenv("LFS_FAST_MODE", "false").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# FSM states
# ---------------------------------------------------------------------------

class ConversationState(str, Enum):
    GREETING = "greeting"
    COLLECTING_INFO = "collecting_info"
    CLARIFYING = "clarifying"
    VALIDATING = "validating"
    COMPLETING = "completing"


# ---------------------------------------------------------------------------
# Conversation context
# ---------------------------------------------------------------------------

@dataclass
class ConversationContext:
    """Holds all mutable state for a single survey session."""
    session_id: int
    language: str = "en"                        # "en" or "ar"
    state: ConversationState = ConversationState.GREETING
    collected_data: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)   # {"role": ..., "content": ...}
    clarification_target: Optional[str] = None          # field currently being clarified
    clarification_count: int = 0                        # how many times the same field was re-asked
    is_returning: bool = False                           # True when pre-filled from a previous session
    correction_applied: bool = False                     # True for one turn after a VALIDATING correction
    corrected_fields: set = field(default_factory=set)   # field keys changed by the correction just applied (one turn)
    correction_rejected_field: Optional[str] = None      # field key whose parsed correction failed the sanity check (one turn)
    correction_no_target: bool = False                    # True for one turn when the respondent said "no"/wants a correction but named no field at all


# ---------------------------------------------------------------------------
# Language-specific prompts
# ---------------------------------------------------------------------------

# Maps any backend language code → the prompt-dict key used for _SYSTEM_BASE /
# _STATE_INSTRUCTIONS.  Languages not in {"en","ar"} fall back to English
# prompts; the actual response language is enforced via _LANG_RESPONSE_INSTRUCTION.
_LANG_PROMPT_KEY: dict[str, str] = {
    "en":       "en",
    "ar":       "ar",
    "ar-gulf":  "ar",
    "ur":       "en",
    "hi":       "en",
    "tl":       "en",
    "other":    "en",
}

# Instruction appended to the task description so the LLM replies in the right language.
_LANG_RESPONSE_INSTRUCTION: dict[str, str] = {
    "en":      "Respond in English.",
    "ar":      "أجب باللغة العربية الفصحى.",
    "ar-gulf": "أجب باللغة العربية (اللهجة الخليجية مقبولة).",
    "ur":      "IMPORTANT: Respond entirely in Urdu (اردو). Do not switch to English.",
    "hi":      "IMPORTANT: Respond entirely in Hindi (हिन्दी). Do not switch to English.",
    "tl":      "IMPORTANT: Respond entirely in Filipino/Tagalog. Do not switch to English.",
    "other":   "Respond in English.",
}

_SYSTEM_BASE: dict[str, str] = {
    "en": (
        "You are a professional, empathetic survey interviewer for the "
        "Labour Force Survey (LFS). Your role is to collect accurate "
        "employment information from respondents in a natural, conversational "
        "way. Always be polite, clear, and patient. Ask one question at a time."
    ),
    "ar": (
        "أنت محاور مهني ومتعاطف لمسح القوى العاملة الإماراتي (LFS). دورك هو جمع "
        "معلومات دقيقة حول التوظيف من المشاركين بطريقة طبيعية وتفاعلية. "
        "كن دائمًا مؤدبًا وواضحًا وصبورًا. اطرح سؤالًا واحدًا في كل مرة."
    ),
}

_STATE_INSTRUCTIONS: dict[ConversationState, dict[str, str]] = {
    ConversationState.GREETING: {
        "en": (
            "Greet the respondent warmly and introduce yourself as the UAE LFS "
            "survey assistant. Briefly explain the purpose of the survey "
            "(understanding employment and labour market conditions in the UAE). "
            "Ask whether they prefer to continue in English or Arabic, then "
            "invite them to begin when ready."
        ),
        "ar": (
            "رحّب بالمشارك بدفء وقدّم نفسك كمساعد مسح القوى العاملة الإماراتي. "
            "اشرح بإيجاز الغرض من المسح (فهم أوضاع التوظيف وسوق العمل في الإمارات). "
            "ابدأ المقابلة باللغة العربية بشكل واضح ودعهم يستعدون للبدء."
        ),
    },
    ConversationState.COLLECTING_INFO: {
        "en": (
            "Collect the respondent's employment information by asking survey questions "
            "one at a time, in order, skipping any already answered. "
            "The survey covers employment status, education, occupation details, "
            "industry, hours, wages, and feedback questions. "
            "Acknowledge each answer warmly before moving to the next question. "
            "If an answer seems vague or incomplete, flag it mentally for clarification "
            "but do not interrupt the flow unless necessary."
        ),
        "ar": (
            "اجمع معلومات التوظيف من المشارك بطرح أسئلة المسح واحدًا تلو الآخر، "
            "بالترتيب، مع تخطي ما تمت الإجابة عنه. "
            "يشمل المسح: حالة التوظيف، التعليم، تفاصيل المهنة، القطاع، الساعات، "
            "الأجور، وأسئلة التغذية الراجعة. "
            "أقرّ بكل إجابة بدفء قبل الانتقال إلى السؤال التالي."
        ),
    },
    ConversationState.CLARIFYING: {
        "en": (
            "The respondent's previous answer needs clarification. "
            "Ask one focused, polite follow-up question to get the specific "
            "detail that is missing or unclear. Do not ask multiple questions "
            "at once. Once you have the clarification, confirm your understanding "
            "before continuing."
        ),
        "ar": (
            "إجابة المشارك السابقة تحتاج إلى توضيح. "
            "اطرح سؤالًا متابعًا واحدًا محددًا وبلطف للحصول على التفصيل "
            "الناقص أو غير الواضح. لا تطرح أسئلة متعددة في وقت واحد. "
            "بمجرد الحصول على التوضيح، أكّد فهمك قبل المتابعة."
        ),
    },
    ConversationState.VALIDATING: {
        "en": (
            "All required information has been collected. Read back a clear, "
            "concise summary of the respondent's answers and ask them to confirm "
            "that everything is correct. If they want to correct anything, "
            "acknowledge the correction and note which fields need updating."
        ),
        "ar": (
            "تم جمع جميع المعلومات المطلوبة. اقرأ ملخصًا واضحًا وموجزًا "
            "لإجابات المشارك واسألهم للتأكيد من صحة جميع البيانات. "
            "إذا أرادوا تصحيح أي شيء، أقرّ بالتصحيح ودوّن الحقول التي تحتاج تحديثًا."
        ),
    },
    ConversationState.COMPLETING: {
        "en": (
            "The survey is complete and all answers have been confirmed. "
            "Thank the respondent sincerely for their time and participation. "
            "Mention that their responses contribute to important labour market "
            "research and policy decisions. Close the conversation warmly and "
            "professionally."
        ),
        "ar": (
            "اكتمل المسح وتم تأكيد جميع الإجابات. "
            "اشكر المشارك بصدق على وقته ومشاركته الكريمة. "
            "أذكر أن إجاباتهم تساهم في أبحاث سوق العمل الإماراتي المهمة وصنع القرار. "
            "أنهِ المحادثة بدفء واحترافية."
        ),
    },
}

# Phrases that express uncertainty — catch-alls should NOT store these as answers
_UNCERTAINTY_PHRASES = frozenset({
    "not sure", "i'm not sure", "i m not sure", "not certain", "i don't know",
    "i dont know", "idk", "no idea", "unsure", "maybe", "i'm unsure",
    "لا أعرف", "لست متأكد", "لست متأكدة", "ربما", "ما أعرف",
})

# Employment-status answer words that must never be stored as job title / industry
_STATUS_WORDS = frozenset({
    "employed", "unemployed", "working", "not in the labour force",
    "not in labour force", "not in labor force", "retired", "student", "housewife",
    "موظف", "عاطل", "أعمل", "متقاعد", "طالب", "ربة منزل", "خارج سوق العمل",
})

_CONFIRMATIONS = {
    "en": frozenset({
        "yes", "right", "confirm", "that's right", "that's correct",
        "looks good", "all good", "perfect", "exactly", "yep", "yup",
        "sure", "absolutely",
    }),
    "ar": frozenset({
        "نعم", "صحيح", "موافق", "تأكيد", "هذا صحيح", "كل شيء صحيح", "ممتاز",
        "بالضبط", "أجل", "طبعًا",
    }),
}

_CORRECTIONS = {
    "en": frozenset({
        "wrong", "incorrect", "change", "fix", "update", "mistake", "error",
        "no", "not right", "not correct", "actually", "wait",
        # Additional correction signals
        "correct", "should be", "must be", "need to change", "want to change",
        "needs to be", "it should", "that's wrong", "thats wrong",
    }),
    "ar": frozenset({
        "خطأ", "غلط", "تغيير", "تعديل", "تصحيح", "لا", "ليس صحيحًا", "في الواقع", "انتظر",
        "غير", "غيّر", "عدّل", "عدل", "صحح",
    }),
}

# ── Comprehensive field schema: valid values + hints for every survey field ───
# Used in the LLM correction prompt so the model knows accepted values.
_CORRECTION_FIELD_SCHEMA: dict[str, dict] = {
    # ── Section B: Demographics ──────────────────────────────────────────────
    "employment_status":      {"label": "Employment Status",              "values": "employed | unemployed | not_in_labour_force"},
    "education_level":        {"label": "Education Level",                "values": "no_formal | primary | intermediate | secondary | diploma | bachelor | master | phd"},
    "gender":                 {"label": "Gender",                         "values": "male | female | prefer_not_to_say"},
    "nationality":            {"label": "Nationality",                    "values": "free text — plain country name, e.g. Indian, Pakistani, Emirati, British"},
    "marital_status":         {"label": "Marital Status",                 "values": "single | married | divorced | widowed"},
    "emirate":                {"label": "Emirate of Residence",           "values": "abu_dhabi | dubai | sharjah | ajman | umm_al_quwain | ras_al_khaimah | fujairah"},
    "uae_residence_duration": {"label": "Duration of UAE Residence",      "values": "born_in_uae | less_than_1_year | 1_to_5_years | 5_to_10_years | more_than_10_years"},
    "field_of_study":         {"label": "Main Field of Study",            "values": "free text — e.g. Engineering, Medicine, Business, Education"},
    "vocational_training":    {"label": "Vocational Training (12 mo.)",   "values": "yes | no"},
    # ── Section C/D/E: Employment ────────────────────────────────────────────
    "employment_nature":      {"label": "Employment Nature",              "values": "paid_employee | employer | self_employed | contributing_family_member"},
    "employment_sector":      {"label": "Employment Sector",              "values": "government | private | semi_government | ngo"},
    "job_title":              {"label": "Job Title",                      "values": "free text — e.g. Software Engineer, Teacher, Nurse, Driver"},
    "job_duties":             {"label": "Main Job Duties",                "values": "free text — describe main tasks"},
    "industry":               {"label": "Industry / Economic Activity",   "values": "free text — e.g. Construction, Healthcare, Education, Finance"},
    "actual_hours_worked":    {"label": "Actual Hours Worked Last Week",  "values": "numeric — e.g. 40"},
    "hours_per_week":         {"label": "Usual Hours per Week",           "values": "numeric — e.g. 36"},
    "secondary_job":          {"label": "Secondary Job",                  "values": "yes | no"},
    "secondary_job_hours":    {"label": "Secondary Job Hours/Week",       "values": "numeric — e.g. 10"},
    "underemployment":        {"label": "Hours Preference",               "values": "overemployed | underemployed | satisfied"},
    "employment_type":        {"label": "Employment Type",                "values": "full_time | part_time | seasonal | temporary"},
    "contract_type":          {"label": "Contract Type",                  "values": "permanent | temporary | no_contract"},
    "remote_work":            {"label": "Remote Work Arrangement",        "values": "fully_remote | partially | on_site"},
    "monthly_wage_range":     {"label": "Monthly Salary Range (AED)",     "values": "under_5000 | 5000_to_10000 | 10000_to_20000 | 20000_to_30000 | above_30000"},
    "salary_allowances":      {"label": "Salary Allowances",              "values": "housing | transport | education | medical | performance | none (comma-separated if multiple)"},
    "bonuses":                {"label": "Bonuses/Incentives (12 mo.)",    "values": "yes_performance | yes_annual | yes_other | no"},
    "health_insurance":       {"label": "Health Insurance Coverage",      "values": "full | partial | none"},
    "pension_scheme":         {"label": "Pension / Gratuity Scheme",      "values": "private_scheme | government_scheme | none"},
    # ── Section F/G: Unemployment / Outside LF ──────────────────────────────
    "job_search_active":      {"label": "Actively Searching for Work",    "values": "yes | no"},
    "job_search_methods":     {"label": "Job Search Methods",             "values": "free text — e.g. online portals, recruitment agencies, networking"},
    "available_for_work":     {"label": "Available to Start (2 weeks)",   "values": "yes | no"},
    "unemployment_duration":  {"label": "Duration of Unemployment",       "values": "less_than_1_month | 1_to_6_months | 6_to_12_months | more_than_1_year"},
    "desired_job_type":       {"label": "Type of Work Sought",            "values": "full_time | part_time | any"},
    "ever_worked":            {"label": "Previous Work Experience",       "values": "yes | no"},
    "outside_lf_reason":      {"label": "Reason for Not Seeking Work",    "values": "studying | housework | retired | health_condition | other"},
    "last_job_title":         {"label": "Last Job Title",                 "values": "free text — e.g. Accountant, Sales Manager"},
    "reason_left_job":        {"label": "Reason for Leaving Last Job",    "values": "resigned | dismissed | contract_ended | business_closed | retirement | other"},
    "last_job_sector":        {"label": "Sector of Last Job",             "values": "government | private | semi_government"},
    "highest_previous_salary":{"label": "Highest Previous Salary (AED)", "values": "under_5000 | 5000_to_10000 | 10000_to_20000 | 20000_to_30000 | above_30000"},
    # ── Section H: Skills & Training ────────────────────────────────────────
    "main_skills":            {"label": "Main Work-Related Skills",       "values": "free text — e.g. programming, accounting, teaching, carpentry"},
    "qualification_match":    {"label": "Qualification Match with Job",   "values": "well_matched | over_qualified | under_qualified"},
    "training_participation": {"label": "Training Participation (12 mo.)","values": "yes | no"},
    "labour_market_barriers": {"label": "Labour Market Barriers",         "values": "free text — e.g. salary expectations, language, discrimination"},
    "emiratization_program":  {"label": "Emiratization Program",          "values": "yes | no"},
    # ── Section I: Digital Work ──────────────────────────────────────────────
    "platform_work":          {"label": "Platform / Gig Work",            "values": "yes | no"},
    "platform_names":         {"label": "Platforms Used for Work",        "values": "free text — e.g. Upwork, Fiverr, Careem, Noon"},
    "platform_hours":         {"label": "Platform Hours per Week",        "values": "numeric — e.g. 15"},
    "online_business":        {"label": "Online Business / E-Commerce",   "values": "yes_registered | yes_informal | no"},
    # ── Section J: Wellbeing ─────────────────────────────────────────────────
    "job_satisfaction":       {"label": "Job Satisfaction (1–5)",         "values": "1 | 2 | 3 | 4 | 5"},
    "work_safety":            {"label": "Work Environment Safety",        "values": "always | usually | sometimes | rarely | never"},
    "workplace_issues":       {"label": "Workplace Issues",               "values": "none | harassment | discrimination | wage_theft | unsafe_conditions | other"},
    "work_life_balance":      {"label": "Work-Life Balance",              "values": "yes | no"},
    # ── Section K: Feedback ─────────────────────────────────────────────────
    "question_clarity":       {"label": "Question Clarity (1–5)",         "values": "1 | 2 | 3 | 4 | 5"},
    "difficulty_answering":   {"label": "Difficulty Answering Questions", "values": "yes | no"},
    "survey_comments":        {"label": "Survey Comments",                "values": "free text"},
    "ai_preference":          {"label": "AI vs Human Interviewer Pref.",  "values": "prefer_ai | prefer_human | no_preference"},
    "data_confidence":        {"label": "Confidence in Data Privacy",     "values": "very_confident | somewhat_confident | not_confident"},
}

# Arabic labels for canonical stored enum values (coded fields only — free-text
# fields like job_title/industry/nationality are left as-is since respondents
# already type those in their own language). Used by _ACK_AR so acknowledgment
# sentences don't leak raw English/snake_case tokens (e.g. "employed") into an
# otherwise-Arabic sentence. Falls back to the pre-existing `v.replace('_',' ')`
# behaviour for anything not listed here, so coverage gaps degrade to the prior
# (imperfect but non-broken) behaviour rather than a KeyError.
_AR_ENUM_LABELS: dict[str, str] = {
    "yes": "نعم", "no": "لا",
    "employed": "موظف", "unemployed": "عاطل عن العمل", "not_in_labour_force": "خارج القوى العاملة",
    "no_formal": "بدون تعليم رسمي", "primary": "ابتدائي", "intermediate": "متوسط",
    "secondary": "ثانوي", "diploma": "دبلوم", "bachelor": "بكالوريوس",
    "master": "ماجستير", "phd": "دكتوراه",
    "male": "ذكر", "female": "أنثى", "prefer_not_to_say": "تفضل عدم الإفصاح",
    "single": "أعزب", "married": "متزوج", "divorced": "مطلق", "widowed": "أرمل",
    "abu_dhabi": "أبوظبي", "dubai": "دبي", "sharjah": "الشارقة", "ajman": "عجمان",
    "umm_al_quwain": "أم القيوين", "ras_al_khaimah": "رأس الخيمة", "fujairah": "الفجيرة",
    "born_in_uae": "مولود في الإمارات", "less_than_1_year": "أقل من سنة",
    "1_to_5_years": "من 1 إلى 5 سنوات", "5_to_10_years": "من 5 إلى 10 سنوات",
    "more_than_10_years": "أكثر من 10 سنوات",
    "paid_employee": "موظف بأجر", "employer": "صاحب عمل", "self_employed": "عمل حر",
    "contributing_family_member": "عامل أسري مساهم",
    "government": "حكومي", "private": "خاص", "semi_government": "شبه حكومي", "ngo": "منظمة غير ربحية",
    "overemployed": "ساعات عمل زائدة", "underemployed": "ساعات عمل ناقصة", "satisfied": "راضٍ عن الساعات",
    "full_time": "دوام كامل", "part_time": "دوام جزئي", "seasonal": "موسمي", "temporary": "مؤقت",
    "permanent": "دائم", "no_contract": "بدون عقد",
    "fully_remote": "عن بُعد بالكامل", "partially": "جزئيًا عن بُعد", "on_site": "من الموقع",
    "under_5000": "أقل من 5000", "5000_to_10000": "من 5000 إلى 10000",
    "10000_to_20000": "من 10000 إلى 20000", "20000_to_30000": "من 20000 إلى 30000",
    "above_30000": "أكثر من 30000",
    "housing": "سكن", "transport": "مواصلات", "education": "تعليم", "medical": "طبي",
    "performance": "أداء", "none": "لا شيء",
    "yes_performance": "نعم — أداء", "yes_annual": "نعم — سنوية", "yes_other": "نعم — أخرى",
    "full": "كامل", "partial": "جزئي",
    "private_scheme": "نظام خاص", "government_scheme": "نظام حكومي",
    "less_than_1_month": "أقل من شهر", "1_to_6_months": "من 1 إلى 6 أشهر",
    "6_to_12_months": "من 6 إلى 12 شهرًا", "more_than_1_year": "أكثر من سنة",
    "any": "أي نوع",
    "studying": "الدراسة", "housework": "أعمال منزلية", "retired": "متقاعد",
    "health_condition": "حالة صحية", "other": "أخرى",
    "resigned": "استقالة", "dismissed": "إنهاء خدمة", "contract_ended": "انتهاء العقد",
    "business_closed": "إغلاق العمل", "retirement": "التقاعد",
    "well_matched": "متوافق جيدًا", "over_qualified": "مؤهل أعلى من الوظيفة",
    "under_qualified": "مؤهل أقل من الوظيفة",
    "yes_registered": "نعم — مسجل", "yes_informal": "نعم — غير رسمي",
    "always": "دائمًا", "usually": "غالبًا", "sometimes": "أحيانًا", "rarely": "نادرًا", "never": "أبدًا",
    "harassment": "تحرش", "discrimination": "تمييز", "wage_theft": "عدم دفع الأجور كاملة",
    "unsafe_conditions": "ظروف عمل غير آمنة",
    "prefer_ai": "الذكاء الاصطناعي", "prefer_human": "شخص حقيقي", "no_preference": "لا تفضيل",
    "very_confident": "واثق جدًا", "somewhat_confident": "واثق نوعًا ما", "not_confident": "غير واثق",
}


def _ar_val(v: object) -> str:
    """Arabic label for a canonical stored enum value; safe fallback for free text."""
    s = str(v)
    return _AR_ENUM_LABELS.get(s, s.replace("_", " "))


# Generic (non-field-specific) acknowledgment for Urdu/Hindi/Tagalog, added
# 2026-08-29 alongside the new follow-up-question translations. Deliberately
# does NOT attempt per-field phrasing or value translation the way _ACK_EN/
# _ACK_AR do — building 56 field-specific, value-aware templates per language
# (plus their own enum-value translation tables) was out of scope for this
# pass. A generic acknowledgment closes the more serious bug (raw English/
# snake_case values leaking into a non-English sentence, and mid-conversation
# language-switching back to English) without overstating translation
# coverage that doesn't exist yet.
def _generic_ack_ur(_v: object) -> str:
    return "ٹھیک ہے — نوٹ کر لیا گیا۔"


def _generic_ack_hi(_v: object) -> str:
    return "ठीक है — नोट कर लिया गया।"


def _generic_ack_tl(_v: object) -> str:
    return "Sige — naitala na."


# Value alias map: normalises common spoken variants → canonical stored value.
# Covers EN + AR + UR + HI + TL surface forms.
_VALUE_ALIASES: dict[str, dict[str, str]] = {
    "employment_status": {
        "employed": "employed", "working": "employed", "have a job": "employed",
        "i work": "employed", "يعمل": "employed", "موظف": "employed",
        "unemployed": "unemployed", "not working": "unemployed", "jobless": "unemployed",
        "looking for work": "unemployed", "عاطل": "unemployed", "بدون عمل": "unemployed",
        "not in labour force": "not_in_labour_force", "outside labour force": "not_in_labour_force",
        "housewife": "not_in_labour_force", "student": "not_in_labour_force",
        "retired": "not_in_labour_force", "متقاعد": "not_in_labour_force",
        "ربة منزل": "not_in_labour_force",
    },
    "gender": {
        "male": "male", "man": "male", "m": "male", "ذكر": "male", "رجل": "male",
        "female": "female", "woman": "female", "f": "female", "أنثى": "female", "امرأة": "female",
        "prefer not to say": "prefer_not_to_say", "rather not say": "prefer_not_to_say",
    },
    "marital_status": {
        "single": "single", "unmarried": "single", "أعزب": "single", "غير متزوج": "single",
        "married": "married", "متزوج": "married",
        "divorced": "divorced", "مطلق": "divorced",
        "widowed": "widowed", "widow": "widowed", "أرمل": "widowed",
    },
    "emirate": {
        "abu dhabi": "abu_dhabi", "abudhabi": "abu_dhabi", "أبوظبي": "abu_dhabi",
        "dubai": "dubai", "دبي": "dubai",
        "sharjah": "sharjah", "الشارقة": "sharjah",
        "ajman": "ajman", "عجمان": "ajman",
        "umm al quwain": "umm_al_quwain", "أم القيوين": "umm_al_quwain",
        "ras al khaimah": "ras_al_khaimah", "رأس الخيمة": "ras_al_khaimah",
        "fujairah": "fujairah", "الفجيرة": "fujairah",
    },
    "uae_residence_duration": {
        "born in uae": "born_in_uae", "born here": "born_in_uae",
        "less than 1 year": "less_than_1_year", "less than a year": "less_than_1_year",
        "1 to 5": "1_to_5_years", "1-5 years": "1_to_5_years",
        "5 to 10": "5_to_10_years", "5-10 years": "5_to_10_years",
        "more than 10": "more_than_10_years", "over 10": "more_than_10_years",
    },
    "employment_nature": {
        "paid employee": "paid_employee", "employee": "paid_employee", "موظف بأجر": "paid_employee",
        "employer": "employer", "صاحب عمل": "employer",
        "self employed": "self_employed", "freelancer": "self_employed", "عمل حر": "self_employed",
        "family worker": "contributing_family_member", "unpaid": "contributing_family_member",
    },
    "employment_sector": {
        "government": "government", "govt": "government", "public": "government", "حكومي": "government",
        "private": "private", "خاص": "private",
        "semi government": "semi_government", "semi-government": "semi_government", "شبه حكومي": "semi_government",
        "ngo": "ngo", "nonprofit": "ngo", "منظمة": "ngo",
    },
    "employment_type": {
        "full time": "full_time", "fulltime": "full_time", "دوام كامل": "full_time",
        "part time": "part_time", "parttime": "part_time", "دوام جزئي": "part_time",
        "seasonal": "seasonal", "موسمي": "seasonal",
        "temporary": "temporary", "temp": "temporary", "مؤقت": "temporary",
    },
    "contract_type": {
        "permanent": "permanent", "دائم": "permanent",
        "temporary": "temporary", "مؤقت": "temporary",
        "no contract": "no_contract", "without contract": "no_contract", "بدون عقد": "no_contract",
    },
    "remote_work": {
        "fully remote": "fully_remote", "remote": "fully_remote", "work from home": "fully_remote",
        "partial": "partially", "hybrid": "partially", "جزئي": "partially",
        "on site": "on_site", "onsite": "on_site", "office": "on_site", "في الموقع": "on_site",
    },
    "monthly_wage_range": {
        "under 5000": "under_5000", "less than 5000": "under_5000", "below 5000": "under_5000",
        "5000 to 10000": "5000_to_10000", "5k to 10k": "5000_to_10000", "5000-10000": "5000_to_10000",
        "10000 to 20000": "10000_to_20000", "10k to 20k": "10000_to_20000", "10000-20000": "10000_to_20000",
        "20000 to 30000": "20000_to_30000", "20k to 30k": "20000_to_30000", "20000-30000": "20000_to_30000",
        "above 30000": "above_30000", "more than 30000": "above_30000", "over 30k": "above_30000",
    },
    "education_level": {
        "no formal": "no_formal", "illiterate": "no_formal", "no education": "no_formal",
        "primary": "primary", "elementary": "primary", "ابتدائي": "primary",
        "intermediate": "intermediate", "middle school": "intermediate", "إعدادي": "intermediate",
        "secondary": "secondary", "high school": "secondary", "ثانوي": "secondary",
        "diploma": "diploma", "دبلوم": "diploma",
        "bachelor": "bachelor", "bachelors": "bachelor", "degree": "bachelor", "بكالوريوس": "bachelor",
        "master": "master", "masters": "master", "msc": "master", "ماجستير": "master",
        "phd": "phd", "doctorate": "phd", "doctoral": "phd", "دكتوراه": "phd",
    },
    "outside_lf_reason": {
        "studying": "studying", "student": "studying", "in school": "studying",
        "housework": "housework", "housewife": "housework", "homemaker": "housework",
        "retired": "retired", "pension": "retired",
        "health": "health_condition", "illness": "health_condition", "disability": "health_condition",
        "other": "other",
    },
    "ai_preference": {
        "prefer ai": "prefer_ai", "ai": "prefer_ai", "robot": "prefer_ai", "chatbot": "prefer_ai",
        "prefer human": "prefer_human", "human": "prefer_human", "person": "prefer_human",
        "no preference": "no_preference", "either": "no_preference", "both": "no_preference",
    },
    "data_confidence": {
        "very confident": "very_confident", "confident": "very_confident", "sure": "very_confident",
        "somewhat confident": "somewhat_confident", "somewhat": "somewhat_confident", "ok": "somewhat_confident",
        "not confident": "not_confident", "worried": "not_confident", "concerned": "not_confident",
    },
    "job_search_active":  {"yes": "yes", "no": "no", "نعم": "yes", "لا": "no"},
    "available_for_work": {"yes": "yes", "no": "no", "نعم": "yes", "لا": "no"},
    "secondary_job":      {"yes": "yes", "no": "no", "نعم": "yes", "لا": "no"},
    "vocational_training":{"yes": "yes", "no": "no", "نعم": "yes", "لا": "no"},
    "ever_worked":        {"yes": "yes", "no": "no", "نعم": "yes", "لا": "no"},
    "platform_work":      {"yes": "yes", "no": "no", "نعم": "yes", "لا": "no"},
    "work_life_balance":  {"yes": "yes", "no": "no", "نعم": "yes", "لا": "no"},
    "difficulty_answering":{"yes": "yes", "no": "no", "نعم": "yes", "لا": "no"},
    "training_participation":{"yes": "yes", "no": "no", "نعم": "yes", "لا": "no"},
    "emiratization_program":{"yes": "yes", "no": "no", "نعم": "yes", "لا": "no"},
}

# Maps surface words the respondent might say → internal field key.
# Used by _extract_correction() regex fallback.
_FIELD_ALIASES: dict[str, str] = {
    # ── English ──────────────────────────────────────────────────────────────
    "employment status": "employment_status", "work status": "employment_status",
    "job status": "employment_status", "am i employed": "employment_status",
    "education level": "education_level", "education": "education_level",
    "qualification": "education_level", "degree": "education_level", "study level": "education_level",
    "gender": "gender", "sex": "gender",
    "nationality": "nationality", "country": "nationality", "country of origin": "nationality",
    "citizenship": "nationality", "passport": "nationality",
    "marital status": "marital_status", "marital": "marital_status", "relationship status": "marital_status",
    "emirate": "emirate", "residence": "emirate", "city": "emirate", "location": "emirate",
    "uae residence": "uae_residence_duration", "residence duration": "uae_residence_duration",
    "how long in uae": "uae_residence_duration", "years in uae": "uae_residence_duration",
    "field of study": "field_of_study", "major": "field_of_study", "specialization": "field_of_study",
    "vocational training": "vocational_training", "training": "vocational_training",
    "employment nature": "employment_nature", "nature of employment": "employment_nature",
    "work nature": "employment_nature", "type of worker": "employment_nature",
    "employment sector": "employment_sector", "sector": "employment_sector", "work sector": "employment_sector",
    "job title": "job_title", "title": "job_title", "occupation": "job_title",
    "position": "job_title", "role": "job_title", "profession": "job_title",
    "job duties": "job_duties", "duties": "job_duties", "responsibilities": "job_duties",
    "tasks": "job_duties", "what i do": "job_duties",
    "industry": "industry", "company type": "industry", "business type": "industry",
    "actual hours": "actual_hours_worked", "hours last week": "actual_hours_worked",
    "hours per week": "hours_per_week", "working hours": "hours_per_week", "hours": "hours_per_week",
    "weekly hours": "hours_per_week",
    "secondary job": "secondary_job", "second job": "secondary_job", "other job": "secondary_job",
    "underemployment": "underemployment", "hours preference": "underemployment",
    "employment type": "employment_type", "job type": "employment_type",
    "full time or part time": "employment_type",
    "contract type": "contract_type", "contract": "contract_type",
    "remote work": "remote_work", "work arrangement": "remote_work", "wfh": "remote_work",
    "monthly wage": "monthly_wage_range", "salary": "monthly_wage_range",
    "wage": "monthly_wage_range", "income": "monthly_wage_range", "pay": "monthly_wage_range",
    "monthly salary": "monthly_wage_range",
    "allowances": "salary_allowances", "salary allowances": "salary_allowances",
    "bonuses": "bonuses", "bonus": "bonuses", "incentives": "bonuses",
    "health insurance": "health_insurance", "insurance": "health_insurance", "medical": "health_insurance",
    "pension": "pension_scheme", "gratuity": "pension_scheme", "end of service": "pension_scheme",
    "job search": "job_search_active", "searching for work": "job_search_active",
    "looking for job": "job_search_active", "job search active": "job_search_active",
    "available for work": "available_for_work", "availability": "available_for_work",
    "unemployment duration": "unemployment_duration", "how long unemployed": "unemployment_duration",
    "duration unemployed": "unemployment_duration",
    "desired job": "desired_job_type", "type of job sought": "desired_job_type",
    "ever worked": "ever_worked", "previous work": "ever_worked", "work experience": "ever_worked",
    "outside lf reason": "outside_lf_reason", "why not working": "outside_lf_reason",
    "reason not working": "outside_lf_reason",
    "last job title": "last_job_title", "previous job": "last_job_title",
    "last job": "last_job_title", "former job": "last_job_title",
    "reason left job": "reason_left_job", "why left": "reason_left_job",
    "reason for leaving": "reason_left_job",
    "last job sector": "last_job_sector", "previous sector": "last_job_sector",
    "previous salary": "highest_previous_salary", "last salary": "highest_previous_salary",
    "skills": "main_skills", "main skills": "main_skills", "abilities": "main_skills",
    "qualification match": "qualification_match", "job match": "qualification_match",
    "training participation": "training_participation", "courses": "training_participation",
    "labour market barriers": "labour_market_barriers", "barriers": "labour_market_barriers",
    "challenges": "labour_market_barriers",
    "emiratization": "emiratization_program", "nafis": "emiratization_program",
    "platform work": "platform_work", "gig work": "platform_work", "gig": "platform_work",
    "platforms": "platform_names", "platform names": "platform_names",
    "platform hours": "platform_hours", "gig hours": "platform_hours",
    "online business": "online_business", "e-commerce": "online_business",
    "job satisfaction": "job_satisfaction", "satisfaction": "job_satisfaction",
    "work safety": "work_safety", "safety": "work_safety",
    "workplace issues": "workplace_issues", "harassment": "workplace_issues",
    "work life balance": "work_life_balance", "balance": "work_life_balance",
    "question clarity": "question_clarity", "clarity": "question_clarity",
    "difficulty": "difficulty_answering", "hard to answer": "difficulty_answering",
    "comments": "survey_comments", "feedback": "survey_comments", "suggestions": "survey_comments",
    "ai preference": "ai_preference", "interviewer preference": "ai_preference",
    "data confidence": "data_confidence", "privacy confidence": "data_confidence",
    # ── Arabic ───────────────────────────────────────────────────────────────
    "الجنسية": "nationality", "بلد الأصل": "nationality", "الجنس": "gender",
    "الحالة الاجتماعية": "marital_status", "مستوى التعليم": "education_level",
    "التعليم": "education_level", "الشهادة": "education_level",
    "الإمارة": "emirate", "مدينة الإقامة": "emirate",
    "مدة الإقامة": "uae_residence_duration", "حالة التوظيف": "employment_status",
    "وضع العمل": "employment_status", "المسمى الوظيفي": "job_title",
    "الوظيفة": "job_title", "المهنة": "job_title", "المهام": "job_duties",
    "الواجبات": "job_duties", "القطاع": "employment_sector",
    "طبيعة العمل": "employment_nature", "نوع العمل": "employment_type",
    "الراتب": "monthly_wage_range", "الأجر": "monthly_wage_range",
    "الدخل": "monthly_wage_range", "الصناعة": "industry", "القطاع الاقتصادي": "industry",
    "ساعات العمل": "hours_per_week", "نوع العقد": "contract_type",
    "العمل عن بعد": "remote_work", "التأمين الصحي": "health_insurance",
    "المهارات": "main_skills", "سبب ترك العمل": "reason_left_job",
    "آخر وظيفة": "last_job_title", "الراتب السابق": "highest_previous_salary",
}

# Regex patterns for corrections in English and Arabic.
# Pattern A — verb-first:  "change the nationality to Indian"
_CORRECTION_PATTERNS_EN = re.compile(
    r"(?:change|update|fix|correct|set|make)\s+(?:the\s+|my\s+)?(.+?)\s+to\s+(.+)",
    re.IGNORECASE,
)
# Pattern B — field-first: "nationality need to change to Indian" / "nationality should be Indian"
_CORRECTION_PATTERNS_EN_FIELD_FIRST = re.compile(
    r"(?:the\s+|my\s+)?(.+?)\s+(?:need(?:s)?\s+to\s+(?:change\s+to|be(?:\s+changed\s+to)?|correct\s+to)|should\s+be|must\s+be|is\s+wrong[,.]?\s*(?:its?\s+)?(?:should\s+be|is))\s+(.+)",
    re.IGNORECASE,
)
_CORRECTION_PATTERNS_AR = re.compile(
    r"(?:غيّر|غير|عدّل|عدل|صحح|اجعل)\s+(.+?)\s+(?:إلى|الى)\s+(.+)",
)

# ── Correction value sanity-checking ────────────────────────────────────────
# The field-first pattern's ".+?" capture is loose enough to over-match on
# natural phrasing like "I should be listed as Senior Engineer" -> captures
# "listed as Senior Engineer" instead of "Senior Engineer". A failed parse is
# cheap (the respondent is asked again); a silent mis-parse is a data defect
# — "listed as Senior Engineer" stored as a job title goes on to be
# confidently mis-classified by ISCO. So free-text fields get filler-prefix
# stripping plus a sanity check; anything that still looks like a sentence
# fragment rather than a value is rejected rather than stored.
_CORRECTION_FILLER_PREFIXES = (
    "listed as", "recorded as", "written as", "down as", "put as", "marked as", "shown as",
)
_CORRECTION_LEADING_VERBS = frozenset({
    "is", "are", "was", "were", "should", "must", "need", "needs", "want", "wants",
    "change", "changed", "correct", "corrected", "update", "updated", "fix", "fixed",
    "listed", "recorded", "written", "put", "marked", "shown", "be", "becomes",
})
# Free-text fields where a correction value should read like a short phrase
# (a job title, an industry, a nationality) rather than a full sentence.
# Numeric/enum fields aren't in scope here — they're validated by
# _canonicalize_correction_value's alias lookup instead.
_CORRECTION_SANITY_CHECK_FIELDS = frozenset({
    "job_title", "last_job_title", "job_duties", "industry", "field_of_study", "nationality",
})
_CORRECTION_MAX_TOKENS = 8  # generous ceiling for a short free-text answer


def _sanity_check_correction_value(field_key: str, raw_value: str) -> Optional[str]:
    """Strip filler prefixes from a captured correction value and reject it if
    what remains still looks like a sentence fragment rather than a value.

    Returns the cleaned value if acceptable, or None if it should be
    rejected (caller should re-prompt rather than store the raw match).
    """
    if field_key not in _CORRECTION_SANITY_CHECK_FIELDS:
        return raw_value.strip() or None

    value = raw_value.strip()
    lowered = value.lower()
    for prefix in _CORRECTION_FILLER_PREFIXES:
        if lowered.startswith(prefix + " "):
            value = value[len(prefix):].strip()
            lowered = value.lower()
            break

    if not value:
        return None
    tokens = value.split()
    if len(tokens) > _CORRECTION_MAX_TOKENS:
        return None
    if tokens[0].lower().strip(".,!?") in _CORRECTION_LEADING_VERBS:
        return None
    return value


# ---------------------------------------------------------------------------
# ConversationManager
# ---------------------------------------------------------------------------

class ConversationManager:
    """
    Manages a UAE LFS survey conversation through a five-state FSM using a
    CrewAI agent backed by Llama 3.2 (Ollama) or Claude 3.5 Sonnet.

    The survey covers all UAE LFS sections A-K following ILO ICLS-19 standards,
    with dynamic question paths based on employment status.

    Usage
    -----
    manager = ConversationManager()
    ctx = manager.new_context(session_id=42, language="en")
    response = manager.process_message(ctx, "Hello, I'd like to start.")
    """

    def __init__(self) -> None:
        self._agent_available = False
        try:
            self._llm = get_llm(TaskType.GENERAL)
            self._agent = Agent(
                role="Conversation Manager",
                goal=(
                    "Guide respondents through the Labour Force Survey accurately "
                    "and empathetically, collecting complete and unambiguous "
                    "employment data in English or Arabic."
                ),
                backstory=(
                    "You are a seasoned LFS survey interviewer trained by the UAE "
                    "Federal Competitiveness and Statistics Centre. You understand "
                    "that precise employment data drives government policy and you "
                    "are skilled at keeping conversations focused, natural, and "
                    "culturally sensitive across both English and Arabic-speaking "
                    "respondents in the UAE."
                ),
                llm=self._llm,
                verbose=False,
                allow_delegation=False,
            )
            self._agent_available = True
        except Exception as exc:
            if _APP_ENV == "development":
                _logger.warning(
                    "ConversationManager: no LLM available (%s). "
                    "Falling back to rule-based dev stub. "
                    "Start Ollama or set a valid ANTHROPIC_API_KEY to use AI responses.",
                    exc,
                )
            else:
                raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_context(self, session_id: int, language: str = "en") -> ConversationContext:
        """Create a fresh ConversationContext for a new survey session."""
        lang = language if language in ("en", "ar", "ar-gulf", "ur", "hi", "tl") else "en"
        return ConversationContext(session_id=session_id, language=lang)

    def process_message(
        self,
        ctx: ConversationContext,
        user_message: str,
    ) -> str:
        """
        Process one conversational turn.

        1. Append user message to history
        2. Build a state-appropriate CrewAI Task
        3. Run the agent and get its response
        4. Append response to history
        5. Evaluate FSM transition
        6. Return the agent response string
        """
        ctx.history.append({"role": "user", "content": user_message})
        prev_state = ctx.state

        # ── Step 1: update FSM state + extract fields from user message ──────
        # Must happen BEFORE response generation so the response reflects the
        # updated collected_data and state (otherwise the just-answered question
        # gets asked again in the same turn).
        self._transition(ctx, user_message, agent_response="")

        # ── Step 2: generate the response based on the UPDATED state ─────────
        # GREETING turn always uses the fixed intro so respondents get a warm,
        # structured welcome + first question regardless of LLM availability.
        if prev_state == ConversationState.GREETING:
            response = self._greeting_with_first_question(ctx)
        elif not self._agent_available or _FAST_MODE:
            response = self._dev_stub_response(ctx)
        else:
            try:
                task = self._build_task(ctx)
                crew = Crew(agents=[self._agent], tasks=[task], verbose=False)
                result = crew.kickoff()
                response = str(result).strip()
            except Exception as exc:
                _logger.warning("LLM call failed: %s. Using dev stub response.", exc)
                response = self._dev_stub_response(ctx)

        ctx.history.append({"role": "assistant", "content": response})

        # When we just transitioned to COMPLETING, generate the farewell in the
        # same turn so the respondent sees a proper closing message.
        if ctx.state == ConversationState.COMPLETING and prev_state != ConversationState.COMPLETING:
            if not self._agent_available or _FAST_MODE:
                farewell = self._dev_stub_response(ctx)
            else:
                try:
                    farewell_task = self._build_task(ctx)
                    farewell_crew = Crew(agents=[self._agent], tasks=[farewell_task], verbose=False)
                    farewell = str(farewell_crew.kickoff()).strip()
                except Exception:
                    farewell = self._dev_stub_response(ctx)
            ctx.history.append({"role": "assistant", "content": farewell})
            return farewell

        return response

    # ------------------------------------------------------------------
    # Field-order and required-fields (dynamic, based on employment path)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_field_order(collected_data: dict) -> list[str]:
        """
        Return the ordered list of fields to collect based on employment status.
        Implements full ILO ICLS-19 skip logic: conditional fields are inserted
        only when their prerequisite answer is available and matches the condition.

        Sections covered:
          B: Demographics (all paths) + B6 field_of_study (if edu >= bachelor)
          C: Employment details (employed)
          D: Hours & conditions (employed); D4 secondary_job_hours if D3=yes;
             D7 contract_type if C3=paid_employee
          E: Wages & benefits (employed)
          F: Unemployment & job search; F2 only if F1=yes
          G: Previous employment; G1/G2/G3/G4 only if ever worked
          H: Skills & training (all paths); H4 emiratization if UAE national
          I: Digital work (all paths); I2/I3 if I1≠no
          J: Quality of work (employed)
          K: Feedback (all paths)
        """
        status = collected_data.get("employment_status", "")
        edu = collected_data.get("education_level", "")

        # Base: always asked regardless of path
        base: list[str] = ["employment_status", "education_level"]
        # B6: field of study only for bachelor-level and above
        if edu in ("bachelor", "master", "phd"):
            base.append("field_of_study")

        # ── Shared demographics block (all employed/unemployed/OLF paths) ──────
        def _demographics() -> list[str]:
            return [
                "gender",                  # B1
                "nationality",             # B3
                "marital_status",          # B4
                "emirate",                 # B8
                "uae_residence_duration",  # B9
                "vocational_training",     # B7
            ]

        # ── Shared skills/digital/feedback block ─────────────────────────────
        def _skills_digital_feedback(include_emiratization: bool) -> list[str]:
            fields: list[str] = [
                "main_skills",             # H1
                "training_participation",  # H3
            ]
            # H4: emiratization only for UAE nationals
            if include_emiratization or collected_data.get("nationality") == "uae_national":
                fields.append("emiratization_program")
            fields += [
                "labour_market_barriers",  # H5
                "platform_work",           # I1
            ]
            # I2/I3: platform details only if respondent uses platforms
            if collected_data.get("platform_work") in ("yes_primary", "yes_supplementary"):
                fields += ["platform_names", "platform_hours"]
            fields += [
                "online_business",         # I4
            ]
            return fields

        def _feedback() -> list[str]:
            return [
                "question_clarity",    # K1
                "difficulty_answering",# K2
                "ai_preference",       # K3
                "data_confidence",     # K4
                "survey_comments",     # K5 (optional)
            ]

        # ── EMPLOYED path ──────────────────────────────────────────────────────
        if status == "employed":
            fields = base + _demographics() + [
                "employment_nature",    # C3
                "employment_sector",    # C4
                "job_title",            # C5
                "job_duties",           # C5a
                "industry",             # C6
                "actual_hours_worked",  # D1
                "hours_per_week",       # D2
                "secondary_job",        # D3
            ]
            # D4: secondary job hours only if has a secondary job
            if collected_data.get("secondary_job") == "yes":
                fields.append("secondary_job_hours")
            fields += [
                "underemployment",      # D5
                "employment_type",      # D6
            ]
            # D7: contract type only for paid employees
            if collected_data.get("employment_nature") == "paid_employee":
                fields.append("contract_type")
            fields.append("remote_work")       # D8
            # E1: wage range only for paid employees (employers/self-employed report income differently)
            if collected_data.get("employment_nature") == "paid_employee":
                fields.append("monthly_wage_range")  # E1
            fields += [
                "salary_allowances",    # E2
                "bonuses",              # E3
                "health_insurance",     # E4
                "pension_scheme",       # E5
                "qualification_match",  # H2 (employed only)
            ]
            fields += _skills_digital_feedback(include_emiratization=False)
            fields += [
                "job_satisfaction",     # J1
                "work_safety",          # J2
                "workplace_issues",     # J3
                "work_life_balance",    # J4
            ]
            fields += _feedback()
            return fields

        # ── UNEMPLOYED path ────────────────────────────────────────────────────
        elif status == "unemployed":
            fields = base + _demographics() + [
                "job_search_active",    # F1
            ]
            # F2: search methods only if actively searching
            if collected_data.get("job_search_active") == "yes":
                fields.append("job_search_methods")
            fields += [
                "available_for_work",   # F3
                "unemployment_duration",# F4
                "desired_job_type",     # F5
                "ever_worked",          # F7
            ]
            # G1/G2/G3/G4: previous employment only if has worked before
            if "ever_worked" in collected_data and collected_data.get("ever_worked") != "never_worked":
                fields += [
                    "last_job_title",           # G1
                    "last_job_sector",          # G2
                    "reason_left_job",          # G3
                    "highest_previous_salary",  # G4
                ]
            fields += _skills_digital_feedback(include_emiratization=False)
            fields += _feedback()
            return fields

        # ── OUTSIDE LABOUR FORCE path ──────────────────────────────────────────
        elif status == "not_in_labour_force":
            fields = base + _demographics() + [
                "outside_lf_reason",    # F6
                "ever_worked",          # F7
            ]
            # G1/G2/G4: previous employment only if has worked before
            if "ever_worked" in collected_data and collected_data.get("ever_worked") != "never_worked":
                fields += [
                    "last_job_title",           # G1
                    "last_job_sector",          # G2
                    "highest_previous_salary",  # G4
                ]
            fields += _skills_digital_feedback(include_emiratization=False)
            fields += _feedback()
            return fields

        else:
            # Status not yet determined — only ask the base questions
            return base

    @staticmethod
    def _get_required_fields(collected_data: dict) -> frozenset:
        """Return the set of fields that must be collected before VALIDATING."""
        status = collected_data.get("employment_status", "")
        base = frozenset({"employment_status", "education_level"})

        if status == "employed":
            required = base | frozenset({
                "gender", "nationality", "marital_status", "emirate",
                "employment_nature", "employment_sector",
                "job_title", "job_duties", "industry",
                "hours_per_week", "employment_type",
                "main_skills", "platform_work",
                "job_satisfaction",
                "question_clarity", "ai_preference", "data_confidence",
            })
            # E1 only required for paid employees
            if collected_data.get("employment_nature") == "paid_employee":
                required = required | frozenset({"monthly_wage_range"})
            return required
        elif status == "unemployed":
            return base | frozenset({
                "gender", "nationality", "marital_status", "emirate",
                "job_search_active", "available_for_work", "unemployment_duration",
                "ever_worked",
                "main_skills", "platform_work",
                "question_clarity", "ai_preference", "data_confidence",
            })
        elif status == "not_in_labour_force":
            return base | frozenset({
                "gender", "nationality", "marital_status", "emirate",
                "outside_lf_reason", "ever_worked",
                "main_skills", "platform_work",
                "question_clarity", "ai_preference", "data_confidence",
            })
        else:
            return base

    # ------------------------------------------------------------------
    # Exact question texts (used by both LLM task builder and dev stub)
    # ------------------------------------------------------------------

    _EXACT_QUESTIONS_EN: dict[str, str] = {
        # ── Core (all paths) ──────────────────────────────────────────
        "employment_status": (
            "What is your current employment status?\n"
            "Please choose one: Employed / Unemployed / Not in the labour force"
        ),
        "education_level": (
            "What is your highest level of education completed?\n"
            "Choose one: No formal education / Primary / Secondary / Diploma / "
            "Bachelor's degree / Master's degree / PhD or higher"
        ),
        "ai_preference": (
            "Finally, do you prefer conducting surveys through an AI assistant "
            "like this one compared to a human interviewer?\n"
            "(Prefer AI / Prefer human / No preference)"
        ),
        "data_confidence": (
            "How confident are you that your data is kept private and "
            "confidential in this survey?\n"
            "(Very confident / Somewhat confident / Not confident)"
        ),

        # ── Employed path ─────────────────────────────────────────────
        "employment_nature": (
            "What best describes your employment arrangement?\n"
            "Choose one: Paid employee / Employer (you hire others) / "
            "Self-employed (no employees) / Contributing family worker"
        ),
        "employment_sector": (
            "Which sector does your employer belong to?\n"
            "Choose one: Government / Private / Semi-government / Non-profit / NGO"
        ),
        "job_title": "What is your job title?",
        "job_duties": (
            "Briefly describe your main tasks and duties in that role.\n"
            "(e.g. 'Analyse data, create reports, build dashboards' or "
            "'Treat patients, prescribe medication')"
        ),
        "industry": (
            "Which industry or sector do you work in?\n"
            "(e.g. technology, healthcare, airlines, government, finance, education)"
        ),
        "hours_per_week": "How many hours do you usually work per week?",
        "employment_type": (
            "What is your employment arrangement?\n"
            "Choose one: Full-time / Part-time / Seasonal / Casual"
        ),
        "monthly_wage_range": (
            "What is your approximate monthly salary range? (in AED)\n"
            "Choose one: Less than 5,000 / 5,000–10,000 / 10,001–20,000 / "
            "20,001–50,000 / More than 50,000 / Prefer not to say"
        ),

        # ── Unemployed path ───────────────────────────────────────────
        "job_search_active": (
            "Over the past four weeks, have you been actively looking for a job?\n"
            "(Yes / No)"
        ),
        "available_for_work": (
            "If a suitable job were offered today, would you be available to "
            "start work within the next two weeks?\n"
            "(Yes / No)"
        ),
        "unemployment_duration": (
            "How long have you been looking for work?\n"
            "(e.g. '2 months', '6 weeks', 'about a year')"
        ),
        "last_job_title": (
            "What was your most recent job title?\n"
            "(If you have never worked before, please say 'never worked'.)"
        ),
        "reason_left_job": (
            "Why did you leave or lose your last job?\n"
            "Choose one: Made redundant / Resigned / Business closed / "
            "Contract ended / Other"
        ),

        # ── Outside labour force path ─────────────────────────────────
        "outside_lf_reason": (
            "What is the main reason you are not currently seeking work?\n"
            "Choose one: Retired / Student / Homemaker / Discouraged worker / "
            "Illness or disability / Other"
        ),

        # ── Demographics (all paths) ───────────────────────────────────────────
        "gender": (
            "What is your gender?\n"
            "Choose one: Male / Female / Prefer not to say"
        ),
        "nationality": (
            "What is your nationality or country of origin?\n"
            "(e.g. Indian, Pakistani, Filipino, Emirati, Egyptian, British, American)"
        ),
        "marital_status": (
            "What is your marital status?\n"
            "Choose one: Single / Married / Divorced / Widowed"
        ),
        "emirate": (
            "Which Emirate do you currently reside in?\n"
            "Choose one: Abu Dhabi / Dubai / Sharjah / Ajman / Umm Al Quwain / Ras Al Khaimah / Fujairah"
        ),
        "uae_residence_duration": (
            "How long have you been residing in the UAE?\n"
            "Choose one: Born in UAE / Less than 1 year / 1\u20134 years / 5\u20139 years / 10\u201319 years / 20+ years"
        ),
        "vocational_training": (
            "Have you completed any vocational training or professional certifications in the last 12 months?\n"
            "(Yes / No)"
        ),

        # ── Employed — additional hours & conditions ───────────────────────────
        "actual_hours_worked": (
            "How many hours did you actually work in your main job during the past week?\n"
            "(Enter a number, e.g. '40')"
        ),
        "secondary_job": (
            "Do you have any other job or business in addition to your main job?\n"
            "(Yes / No)"
        ),
        "underemployment": (
            "Do you want to work more hours than you currently do and are you available for additional work?\n"
            "Choose one: Yes \u2014 I want more hours / No / Already overemployed"
        ),
        "contract_type": (
            "Is your employment contract permanent or temporary?\n"
            "Choose one: Permanent / Fixed-term (less than 1 year) / Fixed-term (1\u20133 years) / Probation period / No written contract"
        ),
        "remote_work": (
            "Do you primarily work from home (remote/telework)?\n"
            "Choose one: Always / Mostly / Partially / Never"
        ),

        # ── Employed — wages & benefits ────────────────────────────────────────
        "salary_allowances": (
            "Does your salary include any of the following allowances?\n"
            "Choose all that apply: Housing / Transport / Food / Schooling / None"
        ),
        "health_insurance": (
            "Does your employer provide health insurance?\n"
            "Choose one: Full coverage / Partial coverage / No / I pay for my own"
        ),

        # ── Unemployed — extended ──────────────────────────────────────────────
        "job_search_methods": (
            "What methods have you used to search for employment? (please describe)\n"
            "(e.g. online job portals, personal network, employment agencies, MOHRE, social media)"
        ),
        "desired_job_type": (
            "What type of work are you looking for?\n"
            "Choose one: Same as my previous occupation / A different occupation / This will be my first job"
        ),
        "ever_worked": (
            "Have you ever worked before?\n"
            "Choose one: Yes \u2014 my last job was in the UAE / Yes \u2014 my last job was outside the UAE / No \u2014 I have never worked"
        ),

        # ── Skills & Training (all paths) ──────────────────────────────────────
        "main_skills": (
            "What are your main work-related skills? (choose up to 3)\n"
            "Options: Digital/IT / Management / Finance / Engineering / Healthcare / Education / Trades/Technical / Hospitality/Service / Sales/Marketing / Other"
        ),
        "qualification_match": (
            "Do you feel your qualifications match your current job requirements?\n"
            "Choose one: Overqualified / Well matched / Underqualified"
        ),
        "training_participation": (
            "Did you participate in any vocational or professional training in the past 12 months?\n"
            "Choose one: Yes \u2014 employer-funded / Yes \u2014 self-funded / Yes \u2014 government program / No"
        ),
        "labour_market_barriers": (
            "What are the main barriers you face in the labour market? (choose all that apply)\n"
            "Options: Language barrier / Lack of experience / Qualification mismatch / Salary expectations / Discrimination / Location / No barriers / Other"
        ),

        # ── Digital Work (all paths) ───────────────────────────────────────────
        "platform_work": (
            "Do you work through digital platforms (e.g. Uber, Careem, Talabat, Freelancer, Upwork)?\n"
            "Choose one: Yes \u2014 as my primary income / Yes \u2014 as supplementary income / No"
        ),
        "online_business": (
            "Do you own or manage any online business or e-commerce activity?\n"
            "Choose one: Yes \u2014 registered business / Yes \u2014 informal / No"
        ),

        # ── Quality of Work (employed only) ───────────────────────────────────
        "job_satisfaction": (
            "Overall, how satisfied are you with your current job?\n"
            "Rate 1\u20135: 1 = Very dissatisfied / 2 = Dissatisfied / 3 = Neutral / 4 = Satisfied / 5 = Very satisfied"
        ),
        "work_life_balance": (
            "Do you feel you have an appropriate work-life balance?\n"
            "Choose one: Yes / Somewhat / No"
        ),

        # ── Education — conditional ────────────────────────────────────────────
        "field_of_study": (
            "What was your main field of study?\n"
            "(e.g. Engineering, Business, Medicine, Computer Science, Arts)"
        ),

        # ── Employed — secondary job hours ────────────────────────────────────
        "secondary_job_hours": (
            "How many hours did you work in your secondary job(s) last week?\n"
            "(Enter a number, e.g. '10')"
        ),

        # ── Employed — additional benefits ────────────────────────────────────
        "bonuses": (
            "Did you receive any bonuses or incentives in the past 12 months?\n"
            "Choose one: Yes — annual bonus / Yes — performance bonus / Yes — other / No"
        ),
        "pension_scheme": (
            "Are you enrolled in a pension fund or end-of-service gratuity scheme?\n"
            "Choose one: Yes — GPSSA (UAE National) / Yes — DIFC/ADGM scheme / "
            "Yes — employer private scheme / No / Not sure"
        ),

        # ── Emiratization (UAE nationals only) ────────────────────────────────
        "emiratization_program": (
            "Are you registered in any Emiratization program (NAFIS, Tawteen, Absher)?\n"
            "Choose one: Yes — NAFIS / Yes — other government program / No"
        ),

        # ── Digital Work — conditional ─────────────────────────────────────────
        "platform_names": (
            "Which platform(s) do you work through?\n"
            "Choose all that apply: Ride-hailing (Uber/Careem) / Food delivery (Talabat/Deliveroo) / "
            "Freelance platforms / Professional services / E-commerce / Other"
        ),
        "platform_hours": (
            "How many hours per week do you spend working through these platforms?\n"
            "(Enter a number, e.g. '20')"
        ),

        # ── Previous employment (if ever worked) ──────────────────────────────
        "last_job_sector": (
            "Which sector was your last job in?\n"
            "Choose one: Government / Private / Semi-government / Non-profit / NGO / Self-employed"
        ),
        "highest_previous_salary": (
            "What was the highest monthly salary you received in your previous employment? (in AED)\n"
            "Choose one: Less than 5,000 / 5,000–10,000 / 10,001–20,000 / "
            "20,001–50,000 / More than 50,000 / Prefer not to say"
        ),

        # ── Quality of Work — additional (employed only) ──────────────────────
        "work_safety": (
            "Do you work in a safe and healthy work environment?\n"
            "Choose one: Always / Mostly / Sometimes / Rarely / Never"
        ),
        "workplace_issues": (
            "Have you experienced any of the following at your workplace in the past 12 months?\n"
            "Choose all that apply: Harassment / Discrimination / Wage theft / Contract violation / None"
        ),

        # ── Feedback (all paths) ───────────────────────────────────────────────
        "question_clarity": (
            "How would you rate the clarity of the questions in this interview?\n"
            "Rate 1\u20135: 1 = Very unclear / 5 = Very clear"
        ),
        "difficulty_answering": (
            "Did you encounter difficulty answering any specific questions?\n"
            "(No / Yes — please specify which ones)"
        ),
        "survey_comments": (
            "Do you have any comments or suggestions to improve this survey?\n"
            "(This is optional — feel free to share or just say 'no comments')"
        ),
    }

    _EXACT_QUESTIONS_AR: dict[str, str] = {
        # ── Core (all paths) ──────────────────────────────────────────
        "employment_status": (
            "ما هي حالة توظيفك الحالية؟\n"
            "اختر أحد الخيارات: موظف / عاطل عن العمل / خارج سوق العمل"
        ),
        "education_level": (
            "ما أعلى مستوى تعليمي أتممته؟\n"
            "اختر أحد الخيارات: بدون تعليم رسمي / ابتدائي / ثانوي / دبلوم / "
            "بكالوريوس / ماجستير / دكتوراه"
        ),
        "ai_preference": (
            "أخيرًا، هل تفضل إجراء الاستبيانات عبر مساعد ذكاء اصطناعي "
            "مثل هذا بدلًا من محاور بشري؟\n"
            "(أفضل الذكاء الاصطناعي / أفضل المحاور البشري / لا فرق)"
        ),
        "data_confidence": (
            "ما مدى ثقتك بأن بياناتك ستظل خاصة وسرية في هذا الاستبيان؟\n"
            "(واثق جدًا / واثق نسبيًا / غير واثق)"
        ),

        # ── Employed path ─────────────────────────────────────────────
        "employment_nature": (
            "ما الذي يصف ترتيب عملك بشكل أفضل؟\n"
            "اختر أحد الخيارات: موظف براتب / صاحب عمل (توظّف آخرين) / "
            "عمل حر (بدون موظفين) / عامل عائلي مساهم"
        ),
        "employment_sector": (
            "إلى أي قطاع ينتمي صاحب عملك؟\n"
            "اختر أحد الخيارات: حكومي / خاص / شبه حكومي / غير ربحي / منظمة غير حكومية"
        ),
        "job_title": "ما هو مسماك الوظيفي؟",
        "job_duties": (
            "صف باختصار مهامك ومسؤولياتك الرئيسية في هذا الدور.\n"
            "(مثال: 'تحليل البيانات، إعداد التقارير، بناء لوحات المعلومات')"
        ),
        "industry": (
            "في أي قطاع أو صناعة تعمل؟\n"
            "(مثال: تقنية، رعاية صحية، طيران، حكومة، تعليم، مالية)"
        ),
        "hours_per_week": "كم ساعة تعمل عادةً في الأسبوع؟",
        "employment_type": (
            "ما ترتيب عملك؟\n"
            "اختر أحد الخيارات: دوام كامل / دوام جزئي / موسمي / عَرَضي"
        ),
        "monthly_wage_range": (
            "ما هو نطاق راتبك الشهري التقريبي؟ (بالدرهم الإماراتي)\n"
            "اختر: أقل من 5,000 / 5,000–10,000 / 10,001–20,000 / "
            "20,001–50,000 / أكثر من 50,000 / أفضل عدم الإفصاح"
        ),

        # ── Unemployed path ───────────────────────────────────────────
        "job_search_active": (
            "خلال الأسابيع الأربعة الماضية، هل كنت تبحث بنشاط عن عمل؟\n"
            "(نعم / لا)"
        ),
        "available_for_work": (
            "إذا عُرضت عليك وظيفة مناسبة اليوم، هل ستكون متاحًا للبدء "
            "خلال الأسبوعين القادمين؟\n"
            "(نعم / لا)"
        ),
        "unemployment_duration": (
            "منذ متى وأنت تبحث عن عمل؟\n"
            "(مثال: 'شهرين', '6 أسابيع', 'حوالي سنة')"
        ),
        "last_job_title": (
            "ما كان مسماك الوظيفي في آخر وظيفة لك؟\n"
            "(إذا لم تعمل من قبل، قل 'لم أعمل من قبل'.)"
        ),
        "reason_left_job": (
            "لماذا تركت أو فقدت وظيفتك الأخيرة؟\n"
            "اختر: فائض عن الحاجة / استقالة / إغلاق المنشأة / انتهاء العقد / أخرى"
        ),

        # ── Outside labour force path ─────────────────────────────────
        "outside_lf_reason": (
            "ما السبب الرئيسي لعدم بحثك عن عمل حاليًا؟\n"
            "اختر: متقاعد / طالب / ربة منزل / يأس من إيجاد عمل / مرض أو إعاقة / أخرى"
        ),

        # ── Demographics (all paths) ───────────────────────────────────────────
        "gender": (
            "ما هو جنسك؟\n"
            "اختر: ذكر / أنثى / أفضل عدم الإفصاح"
        ),
        "nationality": (
            "ما هي جنسيتك أو بلدك الأصلي؟\n"
            "(مثال: هندي، باكستاني، فلبيني، إماراتي، مصري، بريطاني، أمريكي)"
        ),
        "marital_status": (
            "ما هي حالتك الاجتماعية؟\n"
            "اختر: أعزب / متزوج / مطلق / أرمل"
        ),
        "emirate": (
            "في أي إمارة تقيم حاليًا؟\n"
            "اختر: أبوظبي / دبي / الشارقة / عجمان / أم القيوين / رأس الخيمة / الفجيرة"
        ),
        "uae_residence_duration": (
            "منذ متى وأنت مقيم في الإمارات؟\n"
            "اختر: مولود في الإمارات / أقل من سنة / 1-4 سنوات / 5-9 سنوات / 10-19 سنة / 20 سنة فأكثر"
        ),
        "vocational_training": (
            "هل أتممت أي تدريب مهني أو شهادات مهنية في الـ 12 شهراً الماضية؟\n"
            "(نعم / لا)"
        ),

        # ── Employed — additional hours & conditions ───────────────────────────
        "actual_hours_worked": (
            "كم ساعة عملت فعلياً في وظيفتك الرئيسية خلال الأسبوع الماضي؟\n"
            "(أدخل رقمًا، مثال: '40')"
        ),
        "secondary_job": (
            "هل لديك وظيفة أو عمل إضافي آخر بالإضافة إلى وظيفتك الرئيسية؟\n"
            "(نعم / لا)"
        ),
        "underemployment": (
            "هل تريد العمل لساعات أكثر وأنت متاح للعمل الإضافي؟\n"
            "اختر: نعم — أريد ساعات أكثر / لا / أعمل أكثر من اللازم"
        ),
        "contract_type": (
            "هل عقدك دائم أم مؤقت؟\n"
            "اختر: دائم / عقد محدد المدة (أقل من سنة) / عقد محدد المدة (1-3 سنوات) / فترة تجريبية / بدون عقد مكتوب"
        ),
        "remote_work": (
            "هل تعمل من المنزل بصفة رئيسية؟\n"
            "اختر: دائمًا / في الغالب / جزئيًا / أبدًا"
        ),

        # ── Employed — wages & benefits ────────────────────────────────────────
        "salary_allowances": (
            "هل يشمل راتبك أيًا من البدلات التالية؟\n"
            "اختر كل ما ينطبق: بدل سكن / بدل مواصلات / بدل طعام / بدل تعليم / لا شيء"
        ),
        "health_insurance": (
            "هل يوفر لك صاحب العمل تأمينًا صحيًا؟\n"
            "اختر: تغطية كاملة / تغطية جزئية / لا / أدفع من راتبي"
        ),

        # ── Unemployed — extended ──────────────────────────────────────────────
        "job_search_methods": (
            "ما الأساليب التي استخدمتها للبحث عن عمل؟ (صف ما استخدمته)\n"
            "(مثال: بوابات التوظيف، شبكة علاقات، وكالات توظيف، وزارة الموارد البشرية، وسائل التواصل الاجتماعي)"
        ),
        "desired_job_type": (
            "ما نوع العمل الذي تبحث عنه؟\n"
            "اختر: نفس مهنتي السابقة / مهنة مختلفة / أبحث عن أول وظيفة لي"
        ),
        "ever_worked": (
            "هل عملت من قبل؟\n"
            "اختر: نعم — آخر عمل في الإمارات / نعم — آخر عمل خارج الإمارات / لا — لم أعمل أبداً"
        ),

        # ── Skills & Training (all paths) ──────────────────────────────────────
        "main_skills": (
            "ما هي مهاراتك الأساسية المتعلقة بالعمل؟ (اختر حتى 3)\n"
            "الخيارات: رقمية / إدارة / مالية / هندسة / صحة / تعليم / مهني / ضيافة / مبيعات / أخرى"
        ),
        "qualification_match": (
            "هل تشعر أن مؤهلاتك تتناسب مع متطلبات وظيفتك الحالية؟\n"
            "اختر: مؤهل أكثر من اللازم / مطابق تماماً / مؤهل أقل من اللازم"
        ),
        "training_participation": (
            "هل شاركت في أي تدريب مهني في الـ 12 شهراً الماضية؟\n"
            "اختر: نعم — ممول من صاحب العمل / نعم — ممول ذاتيًا / نعم — برنامج حكومي / لا"
        ),
        "labour_market_barriers": (
            "ما العوائق الرئيسية التي تواجهها في سوق العمل؟ (اختر كل ما ينطبق)\n"
            "الخيارات: حاجز اللغة / نقص الخبرة / عدم تطابق المؤهلات / توقعات الراتب / تمييز / الموقع الجغرافي / لا عوائق / أخرى"
        ),

        # ── Digital Work (all paths) ───────────────────────────────────────────
        "platform_work": (
            "هل تعمل من خلال منصات رقمية (مثل أوبر، كريم، طلبات، فريلانسر)؟\n"
            "اختر: نعم — دخل رئيسي / نعم — دخل إضافي / لا"
        ),
        "online_business": (
            "هل تمتلك أو تدير أي نشاط تجاري إلكتروني؟\n"
            "اختر: نعم — نشاط مسجل / نعم — غير رسمي / لا"
        ),

        # ── Quality of Work (employed only) ───────────────────────────────────
        "job_satisfaction": (
            "بشكل عام، كيف تقيّم مستوى رضاك عن وظيفتك الحالية؟\n"
            "قيّم من 1 إلى 5: 1 = غير راضٍ جداً / 5 = راضٍ جداً"
        ),
        "work_life_balance": (
            "هل تشعر بأن لديك توازنًا مناسبًا بين العمل والحياة الشخصية؟\n"
            "اختر: نعم / نوعًا ما / لا"
        ),

        # ── Education — conditional ────────────────────────────────────────────
        "field_of_study": (
            "ما كان مجال تخصصك الدراسي الرئيسي؟\n"
            "(مثال: هندسة، إدارة الأعمال، طب، علوم الحاسوب، آداب)"
        ),

        # ── Employed — secondary job hours ────────────────────────────────────
        "secondary_job_hours": (
            "كم ساعة عملت في وظيفتك الإضافية الأسبوع الماضي؟\n"
            "(أدخل رقمًا، مثال: '10')"
        ),

        # ── Employed — additional benefits ────────────────────────────────────
        "bonuses": (
            "هل تلقيت أي مكافآت أو حوافز في الـ 12 شهراً الماضية؟\n"
            "اختر: نعم — مكافأة سنوية / نعم — حافز أداء / نعم — أخرى / لا"
        ),
        "pension_scheme": (
            "هل أنت مشترك في صندوق التقاعد أو ضمان نهاية الخدمة؟\n"
            "اختر: نعم — هيئة المعاشات (إماراتي) / نعم — نظام DIFC/ADGM / "
            "نعم — خطة خاصة من صاحب العمل / لا / غير متأكد"
        ),

        # ── Emiratization (UAE nationals only) ────────────────────────────────
        "emiratization_program": (
            "هل أنت مسجل في أي برنامج توطين (نافس، توطين، أبشر)؟\n"
            "اختر: نعم — نافس / نعم — برنامج حكومي آخر / لا"
        ),

        # ── Digital Work — conditional ─────────────────────────────────────────
        "platform_names": (
            "ما المنصة أو المنصات التي تعمل من خلالها؟\n"
            "اختر كل ما ينطبق: توصيل (أوبر/كريم) / توصيل طعام (طلبات/ديليفرو) / "
            "منصات العمل الحر / خدمات مهنية / تجارة إلكترونية / أخرى"
        ),
        "platform_hours": (
            "كم ساعة أسبوعياً تقضي في العمل عبر هذه المنصات؟\n"
            "(أدخل رقمًا، مثال: '20')"
        ),

        # ── Previous employment (if ever worked) ──────────────────────────────
        "last_job_sector": (
            "ما كان قطاع عملك الأخير؟\n"
            "اختر: حكومي / خاص / شبه حكومي / غير ربحي / منظمة غير حكومية / عمل حر"
        ),
        "highest_previous_salary": (
            "ما أعلى راتب شهري حصلت عليه في وظيفتك السابقة؟ (بالدرهم الإماراتي)\n"
            "اختر: أقل من 5,000 / 5,000–10,000 / 10,001–20,000 / "
            "20,001–50,000 / أكثر من 50,000 / أفضل عدم الإفصاح"
        ),

        # ── Quality of Work — additional (employed only) ──────────────────────
        "work_safety": (
            "هل تعمل في بيئة عمل آمنة وصحية؟\n"
            "اختر: دائمًا / في الغالب / أحيانًا / نادرًا / أبدًا"
        ),
        "workplace_issues": (
            "هل تعرضت لأي من الأمور التالية في مكان عملك في الـ 12 شهراً الماضية؟\n"
            "اختر كل ما ينطبق: تحرش / تمييز / سرقة الأجور / انتهاك العقد / لا شيء مما ذكر"
        ),

        # ── Feedback (all paths) ───────────────────────────────────────────────
        "question_clarity": (
            "كيف تقيّم وضوح أسئلة هذه المقابلة؟\n"
            "قيّم من 1 إلى 5: 1 = غير واضح / 5 = واضح جداً"
        ),
        "difficulty_answering": (
            "هل واجهت صعوبة في الإجابة على أسئلة بعينها؟\n"
            "(لا / نعم — حدد أي الأسئلة)"
        ),
        "survey_comments": (
            "هل لديك أي تعليقات أو اقتراحات لتحسين هذا المسح؟\n"
            "(اختياري — شاركنا أو قل 'لا تعليقات')"
        ),
    }

    # ── Urdu / Hindi / Tagalog follow-up questions ─────────────────────────────
    # Added 2026-08-29 to close a real gap: only the greeting
    # (_greeting_with_first_question) was translated for these three languages;
    # every follow-up question fell back to _EXACT_QUESTIONS_EN. These 56
    # entries are machine-drafted translations (by this assistant), matching
    # _EXACT_QUESTIONS_EN's structure and options exactly. They have NOT been
    # reviewed by a native speaker of Urdu, Hindi, or Tagalog — treat as a
    # first draft that should be checked before relying on it for a real
    # demo or production use, same discipline as every other unverified claim
    # in this project.
    _EXACT_QUESTIONS_UR: dict[str, str] = {
        "employment_status": (
            "آپ کی موجودہ ملازمت کی صورتحال کیا ہے؟\n"
            "ایک انتخاب کریں: ملازم / بے روزگار / لیبر فورس سے باہر"
        ),
        "education_level": (
            "آپ کی حاصل کردہ تعلیم کی اعلیٰ ترین سطح کیا ہے؟\n"
            "ایک انتخاب کریں: کوئی باضابطہ تعلیم نہیں / ابتدائی / ثانوی / ڈپلومہ / "
            "بیچلر ڈگری / ماسٹر ڈگری / پی ایچ ڈی یا اس سے اعلیٰ"
        ),
        "ai_preference": (
            "آخر میں، کیا آپ اس طرح کے AI اسسٹنٹ کے ذریعے سروے کروانے کو انسانی "
            "انٹرویو لینے والے کے مقابلے میں ترجیح دیتے ہیں؟\n"
            "(AI کو ترجیح / انسان کو ترجیح / کوئی ترجیح نہیں)"
        ),
        "data_confidence": (
            "آپ کو کتنا یقین ہے کہ اس سروے میں آپ کا ڈیٹا نجی اور خفیہ رکھا جائے گا؟\n"
            "(بہت پراعتماد / کسی حد تک پراعتماد / پراعتماد نہیں)"
        ),
        "employment_nature": (
            "آپ کے روزگار کے انتظام کو بہترین طور پر کیا بیان کرتا ہے؟\n"
            "ایک انتخاب کریں: تنخواہ دار ملازم / آجر (آپ دوسروں کو ملازم رکھتے ہیں) / "
            "خود روزگار (کوئی ملازم نہیں) / خاندانی کاروبار میں معاون کارکن"
        ),
        "employment_sector": (
            "آپ کا آجر کس شعبے سے تعلق رکھتا ہے؟\n"
            "ایک انتخاب کریں: سرکاری / نجی / نیم سرکاری / غیر منافع بخش / این جی او"
        ),
        "job_title": "آپ کا جاب ٹائٹل کیا ہے؟",
        "job_duties": (
            "براہ کرم مختصراً اپنے اس کردار میں اپنے بنیادی کام اور فرائض بیان کریں۔\n"
            "(مثلاً 'ڈیٹا کا تجزیہ کرنا، رپورٹس بنانا، ڈیش بورڈز تیار کرنا' یا "
            "'مریضوں کا علاج کرنا، دوا تجویز کرنا')"
        ),
        "industry": (
            "آپ کس صنعت یا شعبے میں کام کرتے ہیں؟\n"
            "(مثلاً ٹیکنالوجی، صحت کی دیکھ بھال، ایئر لائنز، حکومت، مالیات، تعلیم)"
        ),
        "hours_per_week": "آپ عام طور پر ہفتے میں کتنے گھنٹے کام کرتے ہیں؟",
        "employment_type": (
            "آپ کا روزگار کا انتظام کیا ہے؟\n"
            "ایک انتخاب کریں: کل وقتی / جز وقتی / موسمی / عارضی"
        ),
        "monthly_wage_range": (
            "آپ کی تقریباً ماہانہ تنخواہ کی حد کیا ہے؟ (درہم میں)\n"
            "ایک انتخاب کریں: 5,000 سے کم / 5,000–10,000 / 10,001–20,000 / "
            "20,001–50,000 / 50,000 سے زیادہ / بتانا نہیں چاہتے"
        ),
        "job_search_active": (
            "گزشتہ چار ہفتوں کے دوران، کیا آپ فعال طور پر نوکری تلاش کر رہے ہیں؟\n"
            "(ہاں / نہیں)"
        ),
        "available_for_work": (
            "اگر آج کوئی مناسب نوکری پیش کی جائے تو کیا آپ اگلے دو ہفتوں میں کام "
            "شروع کرنے کے لیے دستیاب ہوں گے؟\n"
            "(ہاں / نہیں)"
        ),
        "unemployment_duration": (
            "آپ کتنے عرصے سے کام تلاش کر رہے ہیں؟\n"
            "(مثلاً '2 مہینے'، '6 ہفتے'، 'تقریباً ایک سال')"
        ),
        "last_job_title": (
            "آپ کی سب سے حالیہ جاب ٹائٹل کیا تھی؟\n"
            "(اگر آپ نے کبھی کام نہیں کیا تو براہ کرم 'کبھی کام نہیں کیا' کہیں۔)"
        ),
        "reason_left_job": (
            "آپ نے اپنی آخری نوکری کیوں چھوڑی یا کھو دی؟\n"
            "ایک انتخاب کریں: نوکری ختم کر دی گئی / استعفیٰ دیا / کاروبار بند ہو گیا / "
            "معاہدہ ختم ہوا / دیگر"
        ),
        "outside_lf_reason": (
            "آپ فی الحال کام تلاش نہ کرنے کی بنیادی وجہ کیا ہے؟\n"
            "ایک انتخاب کریں: ریٹائرڈ / طالب علم / گھریلو خاتون/خاوند / دلبرداشتہ کارکن / "
            "بیماری یا معذوری / دیگر"
        ),
        "gender": (
            "آپ کی صنف کیا ہے؟\n"
            "ایک انتخاب کریں: مرد / عورت / بتانا نہیں چاہتے"
        ),
        "nationality": (
            "آپ کی قومیت یا اصل ملک کیا ہے؟\n"
            "(مثلاً بھارتی، پاکستانی، فلپائنی، اماراتی، مصری، برطانوی، امریکی)"
        ),
        "marital_status": (
            "آپ کی ازدواجی حیثیت کیا ہے؟\n"
            "ایک انتخاب کریں: غیر شادی شدہ / شادی شدہ / طلاق یافتہ / بیوہ"
        ),
        "emirate": (
            "آپ فی الحال کس امارات میں رہائش پذیر ہیں؟\n"
            "ایک انتخاب کریں: ابوظہبی / دبئی / شارجہ / عجمان / ام القوین / "
            "رأس الخیمہ / فجیرہ"
        ),
        "uae_residence_duration": (
            "آپ کتنے عرصے سے متحدہ عرب امارات میں مقیم ہیں؟\n"
            "ایک انتخاب کریں: امارات میں پیدا ہوئے / 1 سال سے کم / 1–4 سال / "
            "5–9 سال / 10–19 سال / 20 سال یا زیادہ"
        ),
        "vocational_training": (
            "کیا آپ نے گزشتہ 12 مہینوں میں کوئی پیشہ ورانہ تربیت یا سرٹیفیکیشن "
            "مکمل کی ہے؟\n"
            "(ہاں / نہیں)"
        ),
        "actual_hours_worked": (
            "گزشتہ ہفتے آپ نے اپنی بنیادی نوکری میں اصل میں کتنے گھنٹے کام کیا؟\n"
            "(ایک عدد درج کریں، مثلاً '40')"
        ),
        "secondary_job": (
            "کیا آپ کی اپنی بنیادی نوکری کے علاوہ کوئی اور نوکری یا کاروبار ہے؟\n"
            "(ہاں / نہیں)"
        ),
        "underemployment": (
            "کیا آپ اپنے موجودہ گھنٹوں سے زیادہ کام کرنا چاہتے ہیں اور کیا آپ اضافی "
            "کام کے لیے دستیاب ہیں؟\n"
            "ایک انتخاب کریں: ہاں — مجھے زیادہ گھنٹے چاہئیں / نہیں / پہلے ہی ضرورت "
            "سے زیادہ کام کر رہا ہوں"
        ),
        "contract_type": (
            "کیا آپ کا ملازمتی معاہدہ مستقل ہے یا عارضی؟\n"
            "ایک انتخاب کریں: مستقل / مقررہ مدت (1 سال سے کم) / مقررہ مدت (1–3 سال) / "
            "آزمائشی مدت / کوئی تحریری معاہدہ نہیں"
        ),
        "remote_work": (
            "کیا آپ بنیادی طور پر گھر سے کام کرتے ہیں (ریموٹ/ٹیلی ورک)؟\n"
            "ایک انتخاب کریں: ہمیشہ / زیادہ تر / جزوی طور پر / کبھی نہیں"
        ),
        "salary_allowances": (
            "کیا آپ کی تنخواہ میں مندرجہ ذیل میں سے کوئی الاؤنس شامل ہے؟\n"
            "جو لاگو ہو منتخب کریں: رہائش / آمد و رفت / کھانا / تعلیم / کوئی نہیں"
        ),
        "health_insurance": (
            "کیا آپ کا آجر صحت کی انشورنس فراہم کرتا ہے؟\n"
            "ایک انتخاب کریں: مکمل کوریج / جزوی کوریج / نہیں / میں خود ادائیگی کرتا ہوں"
        ),
        "job_search_methods": (
            "آپ نے روزگار تلاش کرنے کے لیے کون سے طریقے استعمال کیے ہیں؟ "
            "(براہ کرم بیان کریں)\n"
            "(مثلاً آن لائن جاب پورٹلز، ذاتی نیٹ ورک، روزگار ایجنسیاں، MOHRE، سوشل میڈیا)"
        ),
        "desired_job_type": (
            "آپ کس قسم کا کام تلاش کر رہے ہیں؟\n"
            "ایک انتخاب کریں: میرے پچھلے پیشے جیسا / ایک مختلف پیشہ / یہ میری پہلی "
            "نوکری ہوگی"
        ),
        "ever_worked": (
            "کیا آپ نے کبھی کام کیا ہے؟\n"
            "ایک انتخاب کریں: ہاں — میری آخری نوکری متحدہ عرب امارات میں تھی / "
            "ہاں — میری آخری نوکری متحدہ عرب امارات سے باہر تھی / "
            "نہیں — میں نے کبھی کام نہیں کیا"
        ),
        "main_skills": (
            "آپ کی بنیادی کام سے متعلق مہارتیں کیا ہیں؟ (زیادہ سے زیادہ 3 منتخب کریں)\n"
            "آپشنز: ڈیجیٹل/آئی ٹی / انتظام / مالیات / انجینئرنگ / صحت کی دیکھ بھال / "
            "تعلیم / دستکاری/تکنیکی / مہمان نوازی/خدمت / سیلز/مارکیٹنگ / دیگر"
        ),
        "qualification_match": (
            "کیا آپ محسوس کرتے ہیں کہ آپ کی قابلیت آپ کی موجودہ نوکری کی ضروریات سے "
            "مماثل ہے؟\n"
            "ایک انتخاب کریں: زیادہ قابل / اچھی طرح مماثل / کم قابل"
        ),
        "training_participation": (
            "کیا آپ نے گزشتہ 12 مہینوں میں کسی پیشہ ورانہ یا تکنیکی تربیت میں حصہ لیا؟\n"
            "ایک انتخاب کریں: ہاں — آجر کی طرف سے فنڈڈ / ہاں — خود فنڈڈ / "
            "ہاں — حکومتی پروگرام / نہیں"
        ),
        "labour_market_barriers": (
            "لیبر مارکیٹ میں آپ کو کن اہم رکاوٹوں کا سامنا ہے؟ (جو لاگو ہوں منتخب کریں)\n"
            "آپشنز: زبان کی رکاوٹ / تجربے کی کمی / قابلیت کا عدم مطابقت / تنخواہ کی "
            "توقعات / امتیازی سلوک / مقام / کوئی رکاوٹ نہیں / دیگر"
        ),
        "platform_work": (
            "کیا آپ ڈیجیٹل پلیٹ فارمز کے ذریعے کام کرتے ہیں "
            "(مثلاً Uber، Careem، Talabat، Freelancer، Upwork)؟\n"
            "ایک انتخاب کریں: ہاں — بطور بنیادی آمدنی / ہاں — بطور اضافی آمدنی / نہیں"
        ),
        "online_business": (
            "کیا آپ کے پاس کوئی آن لائن کاروبار یا ای کامرس سرگرمی ہے یا آپ اسے چلاتے ہیں؟\n"
            "ایک انتخاب کریں: ہاں — رجسٹرڈ کاروبار / ہاں — غیر رسمی / نہیں"
        ),
        "job_satisfaction": (
            "مجموعی طور پر، آپ اپنی موجودہ نوکری سے کتنے مطمئن ہیں؟\n"
            "1–5 کی درجہ بندی کریں: 1 = بہت غیر مطمئن / 2 = غیر مطمئن / 3 = غیر جانبدار / "
            "4 = مطمئن / 5 = بہت مطمئن"
        ),
        "work_life_balance": (
            "کیا آپ محسوس کرتے ہیں کہ آپ کا کام اور زندگی میں مناسب توازن ہے؟\n"
            "ایک انتخاب کریں: ہاں / کسی حد تک / نہیں"
        ),
        "field_of_study": (
            "آپ کا بنیادی مضمون تعلیم کیا تھا؟\n"
            "(مثلاً انجینئرنگ، بزنس، طب، کمپیوٹر سائنس، آرٹس)"
        ),
        "secondary_job_hours": (
            "گزشتہ ہفتے آپ نے اپنی ثانوی نوکری/نوکریوں میں کتنے گھنٹے کام کیا؟\n"
            "(ایک عدد درج کریں، مثلاً '10')"
        ),
        "bonuses": (
            "کیا آپ کو گزشتہ 12 مہینوں میں کوئی بونس یا مراعات ملی؟\n"
            "ایک انتخاب کریں: ہاں — سالانہ بونس / ہاں — کارکردگی بونس / ہاں — دیگر / نہیں"
        ),
        "pension_scheme": (
            "کیا آپ کسی پنشن فنڈ یا ملازمت کے اختتام پر گریجویٹی اسکیم میں شامل ہیں؟\n"
            "ایک انتخاب کریں: ہاں — GPSSA (اماراتی شہری) / ہاں — DIFC/ADGM اسکیم / "
            "ہاں — آجر کی نجی اسکیم / نہیں / یقین نہیں"
        ),
        "emiratization_program": (
            "کیا آپ کسی ایمرٹائزیشن پروگرام (NAFIS، توطین، ابشر) میں رجسٹرڈ ہیں؟\n"
            "ایک انتخاب کریں: ہاں — NAFIS / ہاں — دیگر حکومتی پروگرام / نہیں"
        ),
        "platform_names": (
            "آپ کن پلیٹ فارمز کے ذریعے کام کرتے ہیں؟\n"
            "جو لاگو ہوں منتخب کریں: رائیڈ ہیلنگ (Uber/Careem) / فوڈ ڈیلیوری "
            "(Talabat/Deliveroo) / فری لانس پلیٹ فارمز / پیشہ ورانہ خدمات / "
            "ای کامرس / دیگر"
        ),
        "platform_hours": (
            "آپ ہفتے میں ان پلیٹ فارمز کے ذریعے کام کرنے میں کتنے گھنٹے صرف کرتے ہیں؟\n"
            "(ایک عدد درج کریں، مثلاً '20')"
        ),
        "last_job_sector": (
            "آپ کی آخری نوکری کس شعبے میں تھی؟\n"
            "ایک انتخاب کریں: سرکاری / نجی / نیم سرکاری / غیر منافع بخش / این جی او / "
            "خود روزگار"
        ),
        "highest_previous_salary": (
            "آپ کی پچھلی ملازمت میں سب سے زیادہ ماہانہ تنخواہ کیا تھی؟ (درہم میں)\n"
            "ایک انتخاب کریں: 5,000 سے کم / 5,000–10,000 / 10,001–20,000 / "
            "20,001–50,000 / 50,000 سے زیادہ / بتانا نہیں چاہتے"
        ),
        "work_safety": (
            "کیا آپ ایک محفوظ اور صحت مند کام کے ماحول میں کام کرتے ہیں؟\n"
            "ایک انتخاب کریں: ہمیشہ / زیادہ تر / کبھی کبھار / شاذ و نادر / کبھی نہیں"
        ),
        "workplace_issues": (
            "کیا آپ نے گزشتہ 12 مہینوں میں اپنے کام کی جگہ پر مندرجہ ذیل میں سے کسی "
            "کا سامنا کیا؟\n"
            "جو لاگو ہوں منتخب کریں: ہراسانی / امتیازی سلوک / اجرت کی چوری / "
            "معاہدے کی خلاف ورزی / کوئی نہیں"
        ),
        "question_clarity": (
            "آپ اس انٹرویو کے سوالات کی وضاحت کو کیسے درجہ دیں گے؟\n"
            "1–5 کی درجہ بندی کریں: 1 = بہت غیر واضح / 5 = بہت واضح"
        ),
        "difficulty_answering": (
            "کیا آپ کو کسی مخصوص سوال کا جواب دینے میں دشواری ہوئی؟\n"
            "(نہیں / ہاں — براہ کرم بتائیں کون سے)"
        ),
        "survey_comments": (
            "کیا آپ کے پاس اس سروے کو بہتر بنانے کے لیے کوئی تبصرے یا تجاویز ہیں؟\n"
            "(یہ اختیاری ہے — بلا جھجھک شیئر کریں یا صرف 'کوئی تبصرہ نہیں' کہہ دیں)"
        ),
    }
    _EXACT_QUESTIONS_HI: dict[str, str] = {
        "employment_status": (
            "आपकी वर्तमान रोजगार स्थिति क्या है?\n"
            "एक चुनें: नियोजित / बेरोजगार / श्रम शक्ति से बाहर"
        ),
        "education_level": (
            "आपकी पूरी की गई उच्चतम शिक्षा का स्तर क्या है?\n"
            "एक चुनें: कोई औपचारिक शिक्षा नहीं / प्राथमिक / माध्यमिक / डिप्लोमा / "
            "स्नातक डिग्री / स्नातकोत्तर डिग्री / पीएचडी या उच्चतर"
        ),
        "ai_preference": (
            "अंत में, क्या आप इस तरह के AI सहायक के माध्यम से सर्वेक्षण कराना किसी "
            "मानव साक्षात्कारकर्ता की तुलना में पसंद करते हैं?\n"
            "(AI पसंद है / मानव पसंद है / कोई प्राथमिकता नहीं)"
        ),
        "data_confidence": (
            "आपको कितना विश्वास है कि इस सर्वेक्षण में आपका डेटा निजी और गोपनीय "
            "रखा जाएगा?\n"
            "(बहुत आश्वस्त / कुछ हद तक आश्वस्त / आश्वस्त नहीं)"
        ),
        "employment_nature": (
            "आपकी रोजगार व्यवस्था का सबसे अच्छा वर्णन क्या है?\n"
            "एक चुनें: वेतनभोगी कर्मचारी / नियोक्ता (आप दूसरों को नियुक्त करते हैं) / "
            "स्वरोजगार (कोई कर्मचारी नहीं) / पारिवारिक व्यवसाय में सहयोगी कार्यकर्ता"
        ),
        "employment_sector": (
            "आपका नियोक्ता किस क्षेत्र से संबंधित है?\n"
            "एक चुनें: सरकारी / निजी / अर्ध-सरकारी / गैर-लाभकारी / एनजीओ"
        ),
        "job_title": "आपका पद (जॉब टाइटल) क्या है?",
        "job_duties": (
            "कृपया अपनी इस भूमिका में अपने मुख्य कार्यों और कर्तव्यों का संक्षेप में "
            "वर्णन करें।\n"
            "(जैसे 'डेटा का विश्लेषण करना, रिपोर्ट बनाना, डैशबोर्ड तैयार करना' या "
            "'मरीजों का इलाज करना, दवा लिखना')"
        ),
        "industry": (
            "आप किस उद्योग या क्षेत्र में काम करते हैं?\n"
            "(जैसे प्रौद्योगिकी, स्वास्थ्य सेवा, एयरलाइंस, सरकार, वित्त, शिक्षा)"
        ),
        "hours_per_week": "आप आमतौर पर प्रति सप्ताह कितने घंटे काम करते हैं?",
        "employment_type": (
            "आपकी रोजगार व्यवस्था क्या है?\n"
            "एक चुनें: पूर्णकालिक / अंशकालिक / मौसमी / अस्थायी"
        ),
        "monthly_wage_range": (
            "आपकी लगभग मासिक वेतन सीमा क्या है? (AED में)\n"
            "एक चुनें: 5,000 से कम / 5,000–10,000 / 10,001–20,000 / 20,001–50,000 / "
            "50,000 से अधिक / बताना नहीं चाहते"
        ),
        "job_search_active": (
            "पिछले चार हफ्तों में, क्या आप सक्रिय रूप से नौकरी की तलाश कर रहे हैं?\n"
            "(हाँ / नहीं)"
        ),
        "available_for_work": (
            "यदि आज कोई उपयुक्त नौकरी मिले, तो क्या आप अगले दो हफ्तों में काम शुरू "
            "करने के लिए उपलब्ध होंगे?\n"
            "(हाँ / नहीं)"
        ),
        "unemployment_duration": (
            "आप कब से काम की तलाश कर रहे हैं?\n"
            "(जैसे '2 महीने', '6 सप्ताह', 'लगभग एक साल')"
        ),
        "last_job_title": (
            "आपकी सबसे हाल की नौकरी का पद क्या था?\n"
            "(यदि आपने कभी काम नहीं किया है, तो कृपया 'कभी काम नहीं किया' कहें।)"
        ),
        "reason_left_job": (
            "आपने अपनी पिछली नौकरी क्यों छोड़ी या खोई?\n"
            "एक चुनें: छंटनी हुई / इस्तीफा दिया / व्यवसाय बंद हो गया / अनुबंध समाप्त "
            "हुआ / अन्य"
        ),
        "outside_lf_reason": (
            "आप वर्तमान में काम की तलाश न करने का मुख्य कारण क्या है?\n"
            "एक चुनें: सेवानिवृत्त / छात्र / गृहिणी/गृहस्थ / निराश कार्यकर्ता / "
            "बीमारी या विकलांगता / अन्य"
        ),
        "gender": (
            "आपका लिंग क्या है?\n"
            "एक चुनें: पुरुष / महिला / बताना नहीं चाहते"
        ),
        "nationality": (
            "आपकी राष्ट्रीयता या मूल देश क्या है?\n"
            "(जैसे भारतीय, पाकिस्तानी, फिलिपिनो, अमीराती, मिस्री, ब्रिटिश, अमेरिकी)"
        ),
        "marital_status": (
            "आपकी वैवाहिक स्थिति क्या है?\n"
            "एक चुनें: अविवाहित / विवाहित / तलाकशुदा / विधवा/विधुर"
        ),
        "emirate": (
            "आप वर्तमान में किस अमीरात में रहते हैं?\n"
            "एक चुनें: अबू धाबी / दुबई / शारजाह / अजमान / उम्म अल क़ुवैन / "
            "रास अल खैमाह / फुजैराह"
        ),
        "uae_residence_duration": (
            "आप कब से यूएई में रह रहे हैं?\n"
            "एक चुनें: यूएई में जन्मे / 1 वर्ष से कम / 1–4 वर्ष / 5–9 वर्ष / "
            "10–19 वर्ष / 20+ वर्ष"
        ),
        "vocational_training": (
            "क्या आपने पिछले 12 महीनों में कोई व्यावसायिक प्रशिक्षण या पेशेवर "
            "प्रमाणन पूरा किया है?\n"
            "(हाँ / नहीं)"
        ),
        "actual_hours_worked": (
            "पिछले सप्ताह आपने अपनी मुख्य नौकरी में वास्तव में कितने घंटे काम किया?\n"
            "(एक संख्या दर्ज करें, जैसे '40')"
        ),
        "secondary_job": (
            "क्या आपकी मुख्य नौकरी के अलावा कोई अन्य नौकरी या व्यवसाय है?\n"
            "(हाँ / नहीं)"
        ),
        "underemployment": (
            "क्या आप अपने वर्तमान घंटों से अधिक काम करना चाहते हैं और क्या आप "
            "अतिरिक्त काम के लिए उपलब्ध हैं?\n"
            "एक चुनें: हाँ — मुझे अधिक घंटे चाहिए / नहीं / पहले से ही अधिक काम कर रहे हैं"
        ),
        "contract_type": (
            "क्या आपका रोजगार अनुबंध स्थायी है या अस्थायी?\n"
            "एक चुनें: स्थायी / निश्चित अवधि (1 वर्ष से कम) / निश्चित अवधि (1–3 वर्ष) / "
            "परिवीक्षा अवधि / कोई लिखित अनुबंध नहीं"
        ),
        "remote_work": (
            "क्या आप मुख्य रूप से घर से काम करते हैं (रिमोट/टेलीवर्क)?\n"
            "एक चुनें: हमेशा / अधिकतर / आंशिक रूप से / कभी नहीं"
        ),
        "salary_allowances": (
            "क्या आपके वेतन में निम्नलिखित में से कोई भत्ता शामिल है?\n"
            "जो लागू हों चुनें: आवास / परिवहन / भोजन / शिक्षा / कोई नहीं"
        ),
        "health_insurance": (
            "क्या आपका नियोक्ता स्वास्थ्य बीमा प्रदान करता है?\n"
            "एक चुनें: पूर्ण कवरेज / आंशिक कवरेज / नहीं / मैं स्वयं भुगतान करता/करती हूँ"
        ),
        "job_search_methods": (
            "आपने रोजगार खोजने के लिए किन तरीकों का उपयोग किया है? (कृपया वर्णन करें)\n"
            "(जैसे ऑनलाइन जॉब पोर्टल, व्यक्तिगत नेटवर्क, रोजगार एजेंसियां, MOHRE, "
            "सोशल मीडिया)"
        ),
        "desired_job_type": (
            "आप किस प्रकार का काम खोज रहे हैं?\n"
            "एक चुनें: मेरे पिछले पेशे जैसा / एक अलग पेशा / यह मेरी पहली नौकरी होगी"
        ),
        "ever_worked": (
            "क्या आपने कभी काम किया है?\n"
            "एक चुनें: हाँ — मेरी पिछली नौकरी यूएई में थी / हाँ — मेरी पिछली नौकरी "
            "यूएई के बाहर थी / नहीं — मैंने कभी काम नहीं किया"
        ),
        "main_skills": (
            "आपकी मुख्य कार्य-संबंधी कौशल क्या हैं? (अधिकतम 3 चुनें)\n"
            "विकल्प: डिजिटल/आईटी / प्रबंधन / वित्त / इंजीनियरिंग / स्वास्थ्य सेवा / "
            "शिक्षा / व्यापार/तकनीकी / आतिथ्य/सेवा / बिक्री/विपणन / अन्य"
        ),
        "qualification_match": (
            "क्या आपको लगता है कि आपकी योग्यताएं आपकी वर्तमान नौकरी की आवश्यकताओं "
            "से मेल खाती हैं?\n"
            "एक चुनें: अधिक योग्य / अच्छी तरह मेल खाता है / कम योग्य"
        ),
        "training_participation": (
            "क्या आपने पिछले 12 महीनों में किसी व्यावसायिक या पेशेवर प्रशिक्षण में "
            "भाग लिया?\n"
            "एक चुनें: हाँ — नियोक्ता द्वारा वित्त पोषित / हाँ — स्व-वित्त पोषित / "
            "हाँ — सरकारी कार्यक्रम / नहीं"
        ),
        "labour_market_barriers": (
            "श्रम बाजार में आपको किन मुख्य बाधाओं का सामना करना पड़ता है? "
            "(जो लागू हों चुनें)\n"
            "विकल्प: भाषा की बाधा / अनुभव की कमी / योग्यता में असंगति / वेतन "
            "अपेक्षाएं / भेदभाव / स्थान / कोई बाधा नहीं / अन्य"
        ),
        "platform_work": (
            "क्या आप डिजिटल प्लेटफॉर्म के माध्यम से काम करते हैं "
            "(जैसे Uber, Careem, Talabat, Freelancer, Upwork)?\n"
            "एक चुनें: हाँ — मुख्य आय के रूप में / हाँ — अतिरिक्त आय के रूप में / नहीं"
        ),
        "online_business": (
            "क्या आप कोई ऑनलाइन व्यवसाय या ई-कॉमर्स गतिविधि के मालिक हैं या उसका "
            "प्रबंधन करते हैं?\n"
            "एक चुनें: हाँ — पंजीकृत व्यवसाय / हाँ — अनौपचारिक / नहीं"
        ),
        "job_satisfaction": (
            "कुल मिलाकर, आप अपनी वर्तमान नौकरी से कितने संतुष्ट हैं?\n"
            "1–5 पर रेट करें: 1 = बहुत असंतुष्ट / 2 = असंतुष्ट / 3 = तटस्थ / "
            "4 = संतुष्ट / 5 = बहुत संतुष्ट"
        ),
        "work_life_balance": (
            "क्या आपको लगता है कि आपके पास कार्य-जीवन का उचित संतुलन है?\n"
            "एक चुनें: हाँ / कुछ हद तक / नहीं"
        ),
        "field_of_study": (
            "आपका मुख्य अध्ययन क्षेत्र क्या था?\n"
            "(जैसे इंजीनियरिंग, व्यवसाय, चिकित्सा, कंप्यूटर विज्ञान, कला)"
        ),
        "secondary_job_hours": (
            "पिछले सप्ताह आपने अपनी द्वितीयक नौकरी(यों) में कितने घंटे काम किया?\n"
            "(एक संख्या दर्ज करें, जैसे '10')"
        ),
        "bonuses": (
            "क्या आपको पिछले 12 महीनों में कोई बोनस या प्रोत्साहन मिला?\n"
            "एक चुनें: हाँ — वार्षिक बोनस / हाँ — प्रदर्शन बोनस / हाँ — अन्य / नहीं"
        ),
        "pension_scheme": (
            "क्या आप किसी पेंशन फंड या सेवा-समाप्ति उपदान योजना में नामांकित हैं?\n"
            "एक चुनें: हाँ — GPSSA (यूएई नागरिक) / हाँ — DIFC/ADGM योजना / "
            "हाँ — नियोक्ता की निजी योजना / नहीं / निश्चित नहीं"
        ),
        "emiratization_program": (
            "क्या आप किसी एमिराटाइजेशन कार्यक्रम (NAFIS, तवतीन, अब्शर) में "
            "पंजीकृत हैं?\n"
            "एक चुनें: हाँ — NAFIS / हाँ — अन्य सरकारी कार्यक्रम / नहीं"
        ),
        "platform_names": (
            "आप किन प्लेटफॉर्म के माध्यम से काम करते हैं?\n"
            "जो लागू हों चुनें: राइड-हेलिंग (Uber/Careem) / फूड डिलीवरी "
            "(Talabat/Deliveroo) / फ्रीलांस प्लेटफॉर्म / पेशेवर सेवाएं / "
            "ई-कॉमर्स / अन्य"
        ),
        "platform_hours": (
            "आप प्रति सप्ताह इन प्लेटफॉर्म के माध्यम से काम करने में कितने घंटे "
            "बिताते हैं?\n"
            "(एक संख्या दर्ज करें, जैसे '20')"
        ),
        "last_job_sector": (
            "आपकी पिछली नौकरी किस क्षेत्र में थी?\n"
            "एक चुनें: सरकारी / निजी / अर्ध-सरकारी / गैर-लाभकारी / एनजीओ / स्वरोजगार"
        ),
        "highest_previous_salary": (
            "आपकी पिछली नौकरी में सबसे अधिक मासिक वेतन क्या था? (AED में)\n"
            "एक चुनें: 5,000 से कम / 5,000–10,000 / 10,001–20,000 / 20,001–50,000 / "
            "50,000 से अधिक / बताना नहीं चाहते"
        ),
        "work_safety": (
            "क्या आप एक सुरक्षित और स्वस्थ कार्य वातावरण में काम करते हैं?\n"
            "एक चुनें: हमेशा / अधिकतर / कभी-कभी / शायद ही कभी / कभी नहीं"
        ),
        "workplace_issues": (
            "क्या आपने पिछले 12 महीनों में अपने कार्यस्थल पर निम्नलिखित में से किसी "
            "का अनुभव किया?\n"
            "जो लागू हों चुनें: उत्पीड़न / भेदभाव / वेतन की चोरी / अनुबंध का "
            "उल्लंघन / कोई नहीं"
        ),
        "question_clarity": (
            "आप इस साक्षात्कार में प्रश्नों की स्पष्टता को कैसे आंकेंगे?\n"
            "1–5 पर रेट करें: 1 = बहुत अस्पष्ट / 5 = बहुत स्पष्ट"
        ),
        "difficulty_answering": (
            "क्या आपको किसी विशिष्ट प्रश्न का उत्तर देने में कठिनाई हुई?\n"
            "(नहीं / हाँ — कृपया बताएं कौन से)"
        ),
        "survey_comments": (
            "क्या इस सर्वेक्षण को बेहतर बनाने के लिए आपके पास कोई टिप्पणी या "
            "सुझाव है?\n"
            "(यह वैकल्पिक है — बेझिझक साझा करें या केवल 'कोई टिप्पणी नहीं' कहें)"
        ),
    }
    _EXACT_QUESTIONS_TL: dict[str, str] = {
        "employment_status": (
            "Ano ang kasalukuyan mong katayuan sa trabaho?\n"
            "Pumili ng isa: Nagtatrabaho / Walang trabaho / Wala sa labor force"
        ),
        "education_level": (
            "Ano ang pinakamataas na antas ng edukasyon na natapos mo?\n"
            "Pumili ng isa: Walang pormal na edukasyon / Elementarya / Sekondarya / "
            "Diploma / Bachelor's degree / Master's degree / PhD o mas mataas"
        ),
        "ai_preference": (
            "Panghuli, mas gusto mo bang gawin ang survey sa pamamagitan ng isang "
            "AI assistant tulad nito kumpara sa isang taong interviewer?\n"
            "(Mas gusto ang AI / Mas gusto ang tao / Walang preference)"
        ),
        "data_confidence": (
            "Gaano ka kumpiyansa na ang iyong data ay itatago nang pribado at "
            "kumpidensyal sa survey na ito?\n"
            "(Lubos na kumpiyansa / Medyo kumpiyansa / Hindi kumpiyansa)"
        ),
        "employment_nature": (
            "Ano ang pinakamahusay na naglalarawan sa iyong kaayusan sa trabaho?\n"
            "Pumili ng isa: Sinasahurang empleyado / Employer (nagpapatrabaho ka "
            "sa iba) / Nagsasariling negosyo (walang empleyado) / Kasapi sa "
            "negosyo ng pamilya"
        ),
        "employment_sector": (
            "Anong sektor kabilang ang iyong employer?\n"
            "Pumili ng isa: Gobyerno / Pribado / Semi-gobyerno / Non-profit / NGO"
        ),
        "job_title": "Ano ang iyong job title?",
        "job_duties": (
            "Maikling ilarawan ang iyong mga pangunahing gawain at tungkulin sa "
            "papel na iyon.\n"
            "(hal. 'Suriin ang datos, gumawa ng mga ulat, bumuo ng mga dashboard' "
            "o 'Gamutin ang mga pasyente, magreseta ng gamot')"
        ),
        "industry": (
            "Anong industriya o sektor ka nagtatrabaho?\n"
            "(hal. teknolohiya, healthcare, mga airline, gobyerno, pananalapi, "
            "edukasyon)"
        ),
        "hours_per_week": "Ilang oras ka karaniwang nagtatrabaho bawat linggo?",
        "employment_type": (
            "Ano ang iyong kaayusan sa trabaho?\n"
            "Pumili ng isa: Full-time / Part-time / Pana-panahon / Kaswal"
        ),
        "monthly_wage_range": (
            "Ano ang tinatayang buwanang saklaw ng suweldo mo? (sa AED)\n"
            "Pumili ng isa: Mas mababa sa 5,000 / 5,000–10,000 / 10,001–20,000 / "
            "20,001–50,000 / Higit sa 50,000 / Mas gustong hindi sabihin"
        ),
        "job_search_active": (
            "Sa nakaraang apat na linggo, aktibo ka bang naghahanap ng trabaho?\n"
            "(Oo / Hindi)"
        ),
        "available_for_work": (
            "Kung may angkop na trabahong ialok ngayon, magagawa mo bang "
            "magsimula sa loob ng susunod na dalawang linggo?\n"
            "(Oo / Hindi)"
        ),
        "unemployment_duration": (
            "Gaano katagal ka nang naghahanap ng trabaho?\n"
            "(hal. '2 buwan', '6 na linggo', 'mga isang taon')"
        ),
        "last_job_title": (
            "Ano ang iyong pinakahuling job title?\n"
            "(Kung hindi ka pa kailanman nagtrabaho, sabihin lang na 'hindi pa "
            "kailanman nagtrabaho'.)"
        ),
        "reason_left_job": (
            "Bakit mo iniwan o nawala ang iyong huling trabaho?\n"
            "Pumili ng isa: Na-redundant / Nagbitiw / Nagsara ang negosyo / "
            "Natapos ang kontrata / Iba pa"
        ),
        "outside_lf_reason": (
            "Ano ang pangunahing dahilan kung bakit hindi ka kasalukuyang "
            "naghahanap ng trabaho?\n"
            "Pumili ng isa: Retirado / Estudyante / Nasa bahay (homemaker) / "
            "Nawalan ng pag-asang manghanap / Sakit o kapansanan / Iba pa"
        ),
        "gender": (
            "Ano ang iyong kasarian?\n"
            "Pumili ng isa: Lalaki / Babae / Mas gustong hindi sabihin"
        ),
        "nationality": (
            "Ano ang iyong nasyonalidad o bansang pinagmulan?\n"
            "(hal. Indian, Pakistani, Pilipino, Emirati, Egyptian, British, American)"
        ),
        "marital_status": (
            "Ano ang iyong katayuan sa kasal?\n"
            "Pumili ng isa: Walang asawa / May asawa / Diborsyado / Balo"
        ),
        "emirate": (
            "Aling Emirate ka kasalukuyang naninirahan?\n"
            "Pumili ng isa: Abu Dhabi / Dubai / Sharjah / Ajman / Umm Al Quwain / "
            "Ras Al Khaimah / Fujairah"
        ),
        "uae_residence_duration": (
            "Gaano katagal ka nang naninirahan sa UAE?\n"
            "Pumili ng isa: Ipinanganak sa UAE / Mas mababa sa 1 taon / 1–4 na "
            "taon / 5–9 na taon / 10–19 na taon / 20+ na taon"
        ),
        "vocational_training": (
            "Nakumpleto mo ba ang anumang bokasyonal na pagsasanay o propesyonal "
            "na sertipikasyon sa nakaraang 12 buwan?\n"
            "(Oo / Hindi)"
        ),
        "actual_hours_worked": (
            "Ilang oras ka talagang nagtrabaho sa iyong pangunahing trabaho "
            "noong nakaraang linggo?\n"
            "(Maglagay ng numero, hal. '40')"
        ),
        "secondary_job": (
            "May iba ka bang trabaho o negosyo bukod sa iyong pangunahing trabaho?\n"
            "(Oo / Hindi)"
        ),
        "underemployment": (
            "Gusto mo bang magtrabaho nang mas mahabang oras kaysa sa kasalukuyan "
            "at magagawa mo bang tumanggap ng dagdag na trabaho?\n"
            "Pumili ng isa: Oo — gusto ko ng mas maraming oras / Hindi / Sobra na "
            "ang trabaho ko"
        ),
        "contract_type": (
            "Permanente ba o pansamantala ang iyong kontrata sa trabaho?\n"
            "Pumili ng isa: Permanente / May takdang panahon (mas mababa sa 1 "
            "taon) / May takdang panahon (1–3 taon) / Panahon ng probasyon / "
            "Walang nakasulat na kontrata"
        ),
        "remote_work": (
            "Pangunahin ka bang nagtatrabaho mula sa bahay (remote/telework)?\n"
            "Pumili ng isa: Palagi / Kadalasan / Bahagya / Hindi kailanman"
        ),
        "salary_allowances": (
            "Kasama ba sa iyong suweldo ang alinman sa mga sumusunod na allowance?\n"
            "Pumili ng lahat na naaangkop: Pabahay / Transportasyon / Pagkain / "
            "Edukasyon / Wala"
        ),
        "health_insurance": (
            "Nagbibigay ba ang iyong employer ng health insurance?\n"
            "Pumili ng isa: Buong coverage / Bahagyang coverage / Hindi / Ako "
            "mismo ang nagbabayad"
        ),
        "job_search_methods": (
            "Anong mga paraan ang ginamit mo sa paghahanap ng trabaho? "
            "(mangyaring ilarawan)\n"
            "(hal. online job portals, personal network, mga ahensya ng trabaho, "
            "MOHRE, social media)"
        ),
        "desired_job_type": (
            "Anong uri ng trabaho ang hinahanap mo?\n"
            "Pumili ng isa: Kapareho ng aking dating trabaho / Ibang trabaho / "
            "Ito ang magiging unang trabaho ko"
        ),
        "ever_worked": (
            "Nagtrabaho ka na ba kailanman?\n"
            "Pumili ng isa: Oo — ang huli kong trabaho ay sa UAE / Oo — ang huli "
            "kong trabaho ay sa labas ng UAE / Hindi — hindi pa ako kailanman "
            "nagtrabaho"
        ),
        "main_skills": (
            "Ano ang iyong mga pangunahing kasanayan na may kaugnayan sa "
            "trabaho? (pumili ng hanggang 3)\n"
            "Mga pagpipilian: Digital/IT / Pamamahala / Pananalapi / Engineering / "
            "Healthcare / Edukasyon / Kalakal/Teknikal / Hospitality/Serbisyo / "
            "Sales/Marketing / Iba pa"
        ),
        "qualification_match": (
            "Sa tingin mo ba ang iyong mga kwalipikasyon ay tumutugma sa mga "
            "kinakailangan ng iyong kasalukuyang trabaho?\n"
            "Pumili ng isa: Overqualified / Tumutugmang mabuti / Underqualified"
        ),
        "training_participation": (
            "Nakisali ka ba sa anumang bokasyonal o propesyonal na pagsasanay sa "
            "nakaraang 12 buwan?\n"
            "Pumili ng isa: Oo — pinondohan ng employer / Oo — sariling pondo / "
            "Oo — programa ng gobyerno / Hindi"
        ),
        "labour_market_barriers": (
            "Ano ang mga pangunahing hadlang na kinakaharap mo sa labor market? "
            "(pumili ng lahat na naaangkop)\n"
            "Mga pagpipilian: Hadlang sa wika / Kakulangan sa karanasan / Hindi "
            "pagtutugma ng kwalipikasyon / Inaasahang suweldo / Diskriminasyon / "
            "Lokasyon / Walang hadlang / Iba pa"
        ),
        "platform_work": (
            "Nagtatrabaho ka ba sa pamamagitan ng mga digital platform "
            "(hal. Uber, Careem, Talabat, Freelancer, Upwork)?\n"
            "Pumili ng isa: Oo — bilang pangunahing kita / Oo — bilang karagdagang "
            "kita / Hindi"
        ),
        "online_business": (
            "May pagmamay-ari ka ba o namamahala ng anumang online business o "
            "e-commerce activity?\n"
            "Pumili ng isa: Oo — rehistradong negosyo / Oo — impormal / Hindi"
        ),
        "job_satisfaction": (
            "Sa kabuuan, gaano ka nasisiyahan sa iyong kasalukuyang trabaho?\n"
            "Mag-rate mula 1–5: 1 = Lubos na hindi nasisiyahan / 2 = Hindi "
            "nasisiyahan / 3 = Neutral / 4 = Nasisiyahan / 5 = Lubos na nasisiyahan"
        ),
        "work_life_balance": (
            "Sa tingin mo ba mayroon kang angkop na balanse sa trabaho at buhay?\n"
            "Pumili ng isa: Oo / Medyo / Hindi"
        ),
        "field_of_study": (
            "Ano ang iyong pangunahing larangan ng pag-aaral?\n"
            "(hal. Engineering, Business, Medisina, Computer Science, Arts)"
        ),
        "secondary_job_hours": (
            "Ilang oras ka nagtrabaho sa iyong pangalawang trabaho noong "
            "nakaraang linggo?\n"
            "(Maglagay ng numero, hal. '10')"
        ),
        "bonuses": (
            "Nakatanggap ka ba ng anumang bonus o insentibo sa nakaraang 12 buwan?\n"
            "Pumili ng isa: Oo — taunang bonus / Oo — bonus sa performance / "
            "Oo — iba pa / Hindi"
        ),
        "pension_scheme": (
            "Ikaw ba ay naka-enroll sa isang pension fund o end-of-service "
            "gratuity scheme?\n"
            "Pumili ng isa: Oo — GPSSA (UAE National) / Oo — DIFC/ADGM scheme / "
            "Oo — pribadong scheme ng employer / Hindi / Hindi sigurado"
        ),
        "emiratization_program": (
            "Naka-rehistro ka ba sa anumang Emiratization program (NAFIS, "
            "Tawteen, Absher)?\n"
            "Pumili ng isa: Oo — NAFIS / Oo — ibang programa ng gobyerno / Hindi"
        ),
        "platform_names": (
            "Anong mga platform ang ginagamit mo sa pagtatrabaho?\n"
            "Pumili ng lahat na naaangkop: Ride-hailing (Uber/Careem) / "
            "Paghahatid ng pagkain (Talabat/Deliveroo) / Freelance platforms / "
            "Propesyonal na serbisyo / E-commerce / Iba pa"
        ),
        "platform_hours": (
            "Ilang oras kada linggo ang ginugugol mo sa pagtatrabaho sa mga "
            "platform na ito?\n"
            "(Maglagay ng numero, hal. '20')"
        ),
        "last_job_sector": (
            "Anong sektor ang iyong huling trabaho?\n"
            "Pumili ng isa: Gobyerno / Pribado / Semi-gobyerno / Non-profit / "
            "NGO / Nagsasariling negosyo"
        ),
        "highest_previous_salary": (
            "Ano ang pinakamataas na buwanang suweldo na natanggap mo sa iyong "
            "nakaraang trabaho? (sa AED)\n"
            "Pumili ng isa: Mas mababa sa 5,000 / 5,000–10,000 / 10,001–20,000 / "
            "20,001–50,000 / Higit sa 50,000 / Mas gustong hindi sabihin"
        ),
        "work_safety": (
            "Nagtatrabaho ka ba sa ligtas at malusog na kapaligiran sa trabaho?\n"
            "Pumili ng isa: Palagi / Kadalasan / Minsan / Bihira / Hindi kailanman"
        ),
        "workplace_issues": (
            "Naranasan mo ba ang alinman sa mga sumusunod sa iyong lugar ng "
            "trabaho sa nakaraang 12 buwan?\n"
            "Pumili ng lahat na naaangkop: Panliligalig / Diskriminasyon / "
            "Pagnanakaw sa sahod / Paglabag sa kontrata / Wala"
        ),
        "question_clarity": (
            "Paano mo ira-rate ang kalinawan ng mga tanong sa interview na ito?\n"
            "Mag-rate mula 1–5: 1 = Hindi malinaw / 5 = Napakalinaw"
        ),
        "difficulty_answering": (
            "Naranasan mo ba ang kahirapan sa pagsagot sa alinmang partikular na "
            "tanong?\n"
            "(Hindi / Oo — mangyaring tukuyin kung alin)"
        ),
        "survey_comments": (
            "Mayroon ka bang mga komento o mungkahi upang mapabuti ang survey "
            "na ito?\n"
            "(Opsyonal ito — malayang magbahagi o sabihin lang na 'walang komento')"
        ),
    }

    # Acknowledgment templates for each collected field
    _ACK_EN: dict[str, object] = {
        "employment_status":      lambda v: f"Got it — you are currently {v.replace('_', ' ')}.",
        "education_level":        lambda v: f"Thank you — education level noted as {v.replace('_', ' ')}.",
        "employment_nature":      lambda v: f"Understood — you are a {v.replace('_', ' ')}.",
        "employment_sector":      lambda v: f"Noted — you work in the {v.replace('_', ' ')} sector.",
        "job_title":              lambda v: f"Thank you — your job title: {v[:60]}.",
        "job_duties":             lambda v: "Understood — I've noted your main tasks.",
        "industry":               lambda v: f"Noted — industry: {v.replace('_', ' ')}.",
        "hours_per_week":         lambda v: f"Got it — {v} hours per week.",
        "employment_type":        lambda v: f"Noted — {v.replace('_', ' ')}.",
        "monthly_wage_range":     lambda v: "Thank you — salary range noted.",
        "job_search_active":      lambda v: f"Got it — job search: {v}.",
        "available_for_work":     lambda v: f"Understood — availability: {v}.",
        "unemployment_duration":  lambda v: f"Noted — searching for work for {v}.",
        "last_job_title":         lambda v: f"Thank you — last job: {v[:60]}.",
        "reason_left_job":        lambda v: f"Noted — reason: {v.replace('_', ' ')}.",
        "outside_lf_reason":      lambda v: f"Understood — reason: {v.replace('_', ' ')}.",
        "ai_preference":          lambda v: f"Thank you — preference: {v.replace('_', ' ')}.",
        "data_confidence":        lambda v: "Thank you for your feedback.",
        # ── Demographics ──────────────────────────────────────────────────────
        "gender":                 lambda v: f"Noted — gender: {v}.",
        "nationality":            lambda v: f"Noted — nationality: {v}.",
        "marital_status":         lambda v: f"Noted — marital status: {v}.",
        "emirate":                lambda v: f"Got it — residing in {v}.",
        "uae_residence_duration": lambda v: f"Noted — UAE residence: {v}.",
        "vocational_training":    lambda v: f"Got it — vocational training: {v}.",
        # ── Employed extras ───────────────────────────────────────────────────
        "actual_hours_worked":    lambda v: f"Got it — {v} hours worked last week.",
        "secondary_job":          lambda v: f"Noted — secondary job: {v}.",
        "underemployment":        lambda v: f"Noted — hours preference: {v}.",
        "contract_type":          lambda v: f"Noted — contract: {v.replace('_', ' ')}.",
        "remote_work":            lambda v: f"Noted — remote work: {v}.",
        "salary_allowances":      lambda v: "Thank you — allowances noted.",
        "health_insurance":       lambda v: f"Noted — health insurance: {v.replace('_', ' ')}.",
        # ── Unemployed extras ─────────────────────────────────────────────────
        "job_search_methods":     lambda v: "Thank you — job search methods noted.",
        "desired_job_type":       lambda v: f"Noted — looking for: {v.replace('_', ' ')}.",
        "ever_worked":            lambda v: f"Noted — work history: {v.replace('_', ' ')}.",
        # ── Skills & Training ─────────────────────────────────────────────────
        "main_skills":            lambda v: "Thank you — skills noted.",
        "qualification_match":    lambda v: f"Noted — qualification match: {v.replace('_', ' ')}.",
        "training_participation": lambda v: f"Noted — training: {v.replace('_', ' ')}.",
        "labour_market_barriers": lambda v: "Thank you — barriers noted.",
        # ── Digital Work ──────────────────────────────────────────────────────
        "platform_work":          lambda v: f"Noted — platform work: {v.replace('_', ' ')}.",
        "online_business":        lambda v: f"Noted — online business: {v.replace('_', ' ')}.",
        # ── Quality & Feedback ────────────────────────────────────────────────
        "job_satisfaction":           lambda v: f"Thank you — satisfaction score: {v}.",
        "work_safety":                lambda v: f"Noted — work safety: {v}.",
        "workplace_issues":           lambda v: "Thank you — workplace experience noted.",
        "work_life_balance":          lambda v: f"Noted — work-life balance: {v}.",
        "question_clarity":           lambda v: f"Thank you — clarity score: {v}.",
        "difficulty_answering":       lambda v: "Thank you — noted.",
        "survey_comments":            lambda v: "Thank you for your feedback.",
        # ── Education conditional ─────────────────────────────────────────────
        "field_of_study":             lambda v: f"Noted — field of study: {v[:60]}.",
        # ── Employed extras ───────────────────────────────────────────────────
        "secondary_job_hours":        lambda v: f"Got it — {v} hours in secondary job.",
        "bonuses":                    lambda v: f"Noted — bonuses: {v.replace('_', ' ')}.",
        "pension_scheme":             lambda v: f"Noted — pension/gratuity: {v.replace('_', ' ')}.",
        # ── Emiratization ─────────────────────────────────────────────────────
        "emiratization_program":      lambda v: f"Noted — Emiratization: {v.replace('_', ' ')}.",
        # ── Digital conditional ───────────────────────────────────────────────
        "platform_names":             lambda v: "Thank you — platforms noted.",
        "platform_hours":             lambda v: f"Got it — {v} hours on platforms per week.",
        # ── Previous employment ───────────────────────────────────────────────
        "last_job_sector":            lambda v: f"Noted — last job sector: {v.replace('_', ' ')}.",
        "highest_previous_salary":    lambda v: "Thank you — previous salary noted.",
    }
    _ACK_AR: dict[str, object] = {
        "employment_status":      lambda v: f"حسنًا — حالتك الوظيفية: {_ar_val(v)}.",
        "education_level":        lambda v: f"شكرًا — المستوى التعليمي: {_ar_val(v)}.",
        "employment_nature":      lambda v: f"مفهوم — أنت {_ar_val(v)}.",
        "employment_sector":      lambda v: f"تم التسجيل — قطاع: {_ar_val(v)}.",
        "job_title":              lambda v: f"شكرًا — المسمى الوظيفي: {v[:60]}.",
        "job_duties":             lambda v: "مفهوم — تم تسجيل مهامك الرئيسية.",
        "industry":               lambda v: f"مفهوم — الصناعة: {v}.",
        "hours_per_week":         lambda v: f"ممتاز — {v} ساعة في الأسبوع.",
        "employment_type":        lambda v: f"تمام — نوع التوظيف: {_ar_val(v)}.",
        "monthly_wage_range":     lambda v: "شكرًا — تم تسجيل نطاق الراتب.",
        "job_search_active":      lambda v: f"حسنًا — البحث عن عمل: {_ar_val(v)}.",
        "available_for_work":     lambda v: f"مفهوم — التوفر: {_ar_val(v)}.",
        "unemployment_duration":  lambda v: f"تم التسجيل — تبحث منذ {_ar_val(v)}.",
        "last_job_title":         lambda v: f"شكرًا — آخر وظيفة: {v[:60]}.",
        "reason_left_job":        lambda v: f"تم التسجيل — السبب: {_ar_val(v)}.",
        "outside_lf_reason":      lambda v: f"مفهوم — السبب: {_ar_val(v)}.",
        "ai_preference":          lambda v: f"شكرًا — التفضيل: {_ar_val(v)}.",
        "data_confidence":        lambda v: "شكرًا على ملاحظاتك.",
        # ── Demographics ──────────────────────────────────────────────────────
        "gender":                 lambda v: f"تم التسجيل — الجنس: {_ar_val(v)}.",
        "nationality":            lambda v: f"تم التسجيل — الجنسية: {v}.",
        "marital_status":         lambda v: f"تم التسجيل — الحالة الاجتماعية: {_ar_val(v)}.",
        "emirate":                lambda v: f"حسنًا — تقيم في: {_ar_val(v)}.",
        "uae_residence_duration": lambda v: f"تم التسجيل — مدة الإقامة في الإمارات: {_ar_val(v)}.",
        "vocational_training":    lambda v: f"حسنًا — التدريب المهني: {_ar_val(v)}.",
        # ── Employed extras ───────────────────────────────────────────────────
        "actual_hours_worked":    lambda v: f"حسنًا — عملت {v} ساعات الأسبوع الماضي.",
        "secondary_job":          lambda v: f"تم التسجيل — وظيفة إضافية: {_ar_val(v)}.",
        "underemployment":        lambda v: f"تم التسجيل — تفضيل الساعات: {_ar_val(v)}.",
        "contract_type":          lambda v: f"تم التسجيل — نوع العقد: {_ar_val(v)}.",
        "remote_work":            lambda v: f"تم التسجيل — العمل عن بُعد: {_ar_val(v)}.",
        "salary_allowances":      lambda v: "شكرًا — تم تسجيل البدلات.",
        "health_insurance":       lambda v: f"تم التسجيل — التأمين الصحي: {_ar_val(v)}.",
        # ── Unemployed extras ─────────────────────────────────────────────────
        "job_search_methods":     lambda v: "شكرًا — تم تسجيل طرق البحث عن عمل.",
        "desired_job_type":       lambda v: f"تم التسجيل — نوع العمل المطلوب: {_ar_val(v)}.",
        "ever_worked":            lambda v: f"تم التسجيل — تاريخ العمل: {_ar_val(v)}.",
        # ── Skills & Training ─────────────────────────────────────────────────
        "main_skills":            lambda v: "شكرًا — تم تسجيل المهارات.",
        "qualification_match":    lambda v: f"تم التسجيل — تطابق المؤهلات: {_ar_val(v)}.",
        "training_participation": lambda v: f"تم التسجيل — التدريب: {_ar_val(v)}.",
        "labour_market_barriers": lambda v: "شكرًا — تم تسجيل العوائق.",
        # ── Digital Work ──────────────────────────────────────────────────────
        "platform_work":          lambda v: f"تم التسجيل — العمل عبر المنصات: {_ar_val(v)}.",
        "online_business":        lambda v: f"تم التسجيل — النشاط التجاري الإلكتروني: {_ar_val(v)}.",
        # ── Quality & Feedback ────────────────────────────────────────────────
        "job_satisfaction":           lambda v: f"شكرًا — درجة الرضا الوظيفي: {v}.",
        "work_safety":                lambda v: f"تم التسجيل — سلامة بيئة العمل: {_ar_val(v)}.",
        "workplace_issues":           lambda v: "شكرًا — تم تسجيل تجربة بيئة العمل.",
        "work_life_balance":          lambda v: f"تم التسجيل — التوازن بين العمل والحياة: {_ar_val(v)}.",
        "question_clarity":           lambda v: f"شكرًا — درجة وضوح الأسئلة: {v}.",
        "difficulty_answering":       lambda v: "شكرًا — تم التسجيل.",
        "survey_comments":            lambda v: "شكرًا على تعليقاتك.",
        # ── Education conditional ─────────────────────────────────────────────
        "field_of_study":             lambda v: f"تم التسجيل — مجال الدراسة: {v[:60]}.",
        # ── Employed extras ───────────────────────────────────────────────────
        "secondary_job_hours":        lambda v: f"حسنًا — {v} ساعات في الوظيفة الإضافية.",
        "bonuses":                    lambda v: f"تم التسجيل — المكافآت: {_ar_val(v)}.",
        "pension_scheme":             lambda v: f"تم التسجيل — صندوق التقاعد: {_ar_val(v)}.",
        # ── Emiratization ─────────────────────────────────────────────────────
        "emiratization_program":      lambda v: f"تم التسجيل — التوطين: {_ar_val(v)}.",
        # ── Digital conditional ───────────────────────────────────────────────
        "platform_names":             lambda v: "شكرًا — تم تسجيل المنصات.",
        "platform_hours":             lambda v: f"حسنًا — {v} ساعات أسبوعيًا عبر المنصات.",
        # ── Previous employment ───────────────────────────────────────────────
        "last_job_sector":            lambda v: f"تم التسجيل — قطاع آخر وظيفة: {_ar_val(v)}.",
        "highest_previous_salary":    lambda v: "شكرًا — تم تسجيل الراتب السابق.",
    }
    # Generic Urdu/Hindi/Tagalog acknowledgments — see _generic_ack_ur/hi/tl's
    # docstring-comment above for why these are generic rather than
    # field-specific like _ACK_EN/_ACK_AR.
    _ACK_UR: dict[str, object] = dict.fromkeys(_EXACT_QUESTIONS_UR.keys(), _generic_ack_ur)
    _ACK_HI: dict[str, object] = dict.fromkeys(_EXACT_QUESTIONS_HI.keys(), _generic_ack_hi)
    _ACK_TL: dict[str, object] = dict.fromkeys(_EXACT_QUESTIONS_TL.keys(), _generic_ack_tl)

    # Profession keywords for job title extraction
    _PROFESSION_KEYWORDS_EN = [
        "engineer", "manager", "director", "teacher", "professor",
        "doctor", "nurse", "driver", "analyst", "developer",
        "accountant", "officer", "architect", "consultant",
        "pilot", "captain", "technician", "operator", "supervisor",
        "coordinator", "specialist", "assistant", "administrator",
        "chef", "cook", "lawyer", "attorney", "designer", "researcher",
        "scientist", "programmer", "developer", "lecturer", "principal",
    ]
    _PROFESSION_KEYWORDS_AR = [
        "مهندس", "مدير", "معلم", "أستاذ", "طبيب", "ممرض",
        "سائق", "محلل", "مطور", "محاسب", "مسؤول", "مهندس معماري",
        "طيار", "فني", "مشغل", "مشرف", "منسق", "متخصص", "مساعد",
        "طاهٍ", "طاهي", "محامٍ", "مصمم", "باحث", "عالم",
    ]

    # Human-readable labels for summary and clarification
    _FIELD_LABELS_EN: dict[str, str] = {
        "employment_status":     "Employment Status",
        "education_level":       "Education Level",
        "employment_nature":     "Employment Arrangement (paid employee/employer/self-employed)",
        "employment_sector":     "Employment Sector (government/private/semi-government)",
        "job_title":             "Job Title",
        "job_duties":            "Main Tasks & Duties",
        "industry":              "Industry / Sector",
        "hours_per_week":        "Usual Hours per Week",
        "employment_type":       "Employment Type (full-time/part-time/seasonal)",
        "monthly_wage_range":    "Monthly Salary Range (AED)",
        "job_search_active":     "Actively Looking for Work (yes/no)",
        "available_for_work":    "Available to Start Within 2 Weeks (yes/no)",
        "unemployment_duration": "Duration of Job Search",
        "last_job_title":        "Most Recent Job Title",
        "reason_left_job":       "Reason for Leaving Last Job",
        "outside_lf_reason":     "Reason for Not Seeking Work",
        "ai_preference":         "Interviewer Preference (AI vs Human)",
        "data_confidence":       "Confidence in Data Privacy",
        # ── Demographics ──────────────────────────────────────────────────────
        "gender":                "Gender",
        "nationality":           "Nationality",
        "marital_status":        "Marital Status",
        "emirate":               "Emirate of Residence",
        "uae_residence_duration":"Duration of UAE Residence",
        "vocational_training":   "Vocational Training / Certifications (last 12 months)",
        # ── Employed extras ───────────────────────────────────────────────────
        "actual_hours_worked":   "Actual Hours Worked Last Week",
        "secondary_job":         "Secondary Job",
        "underemployment":       "Underemployment / Hours Preference",
        "contract_type":         "Contract Type",
        "remote_work":           "Remote Work Arrangement",
        "salary_allowances":     "Salary Allowances",
        "health_insurance":      "Health Insurance Coverage",
        # ── Unemployed extras ─────────────────────────────────────────────────
        "job_search_methods":    "Job Search Methods Used",
        "desired_job_type":      "Type of Work Sought",
        "ever_worked":           "Previous Work Experience",
        # ── Skills & Training ─────────────────────────────────────────────────
        "main_skills":           "Main Work-Related Skills",
        "qualification_match":   "Qualification Match with Job",
        "training_participation":"Training Participation (last 12 months)",
        "labour_market_barriers":"Labour Market Barriers",
        # ── Digital Work ──────────────────────────────────────────────────────
        "platform_work":         "Platform / Gig Work",
        "online_business":       "Online Business / E-Commerce",
        # ── Quality & Feedback ────────────────────────────────────────────────
        "job_satisfaction":          "Job Satisfaction (1–5)",
        "work_safety":               "Work Environment Safety",
        "workplace_issues":          "Workplace Issues (harassment/discrimination/etc.)",
        "work_life_balance":         "Work-Life Balance",
        "question_clarity":          "Question Clarity Rating (1–5)",
        "difficulty_answering":      "Difficulty Answering Specific Questions",
        "survey_comments":           "Survey Comments / Suggestions",
        # ── Education conditional ─────────────────────────────────────────────
        "field_of_study":            "Main Field of Study",
        # ── Employed extras ───────────────────────────────────────────────────
        "secondary_job_hours":       "Hours Worked in Secondary Job Last Week",
        "bonuses":                   "Bonuses / Incentives (last 12 months)",
        "pension_scheme":            "Pension / End-of-Service Gratuity Scheme",
        # ── Emiratization ─────────────────────────────────────────────────────
        "emiratization_program":     "Emiratization Program Registration",
        # ── Digital conditional ───────────────────────────────────────────────
        "platform_names":            "Digital Platforms Used for Work",
        "platform_hours":            "Hours per Week on Digital Platforms",
        # ── Previous employment ───────────────────────────────────────────────
        "last_job_sector":           "Sector of Last Job",
        "highest_previous_salary":   "Highest Previous Monthly Salary (AED)",
    }
    _FIELD_LABELS_AR: dict[str, str] = {
        "employment_status":     "حالة التوظيف",
        "education_level":       "المستوى التعليمي",
        "employment_nature":     "طبيعة العمل (موظف/صاحب عمل/عمل حر)",
        "employment_sector":     "قطاع التوظيف (حكومي/خاص/شبه حكومي)",
        "job_title":             "المسمى الوظيفي",
        "job_duties":            "المهام الرئيسية",
        "industry":              "القطاع / الصناعة",
        "hours_per_week":        "ساعات العمل الأسبوعية المعتادة",
        "employment_type":       "نوع التوظيف (دوام كامل/جزئي/موسمي)",
        "monthly_wage_range":    "نطاق الراتب الشهري (درهم)",
        "job_search_active":     "البحث النشط عن عمل (نعم/لا)",
        "available_for_work":    "التوفر للعمل خلال أسبوعين (نعم/لا)",
        "unemployment_duration": "مدة البحث عن عمل",
        "last_job_title":        "آخر مسمى وظيفي",
        "reason_left_job":       "سبب ترك آخر وظيفة",
        "outside_lf_reason":     "سبب عدم البحث عن عمل",
        "ai_preference":         "تفضيل المحاور (ذكاء اصطناعي أم بشري)",
        "data_confidence":       "الثقة بسرية البيانات",
        # ── Demographics ──────────────────────────────────────────────────────
        "gender":                "الجنس",
        "nationality":           "الجنسية",
        "marital_status":        "الحالة الاجتماعية",
        "emirate":               "إمارة الإقامة",
        "uae_residence_duration":"مدة الإقامة في الإمارات",
        "vocational_training":   "التدريب المهني / الشهادات (آخر 12 شهراً)",
        # ── Employed extras ───────────────────────────────────────────────────
        "actual_hours_worked":   "ساعات العمل الفعلية الأسبوع الماضي",
        "secondary_job":         "وظيفة ثانوية",
        "underemployment":       "التشغيل الناقص / تفضيل الساعات",
        "contract_type":         "نوع العقد",
        "remote_work":           "ترتيب العمل عن بُعد",
        "salary_allowances":     "بدلات الراتب",
        "health_insurance":      "التأمين الصحي",
        # ── Unemployed extras ─────────────────────────────────────────────────
        "job_search_methods":    "طرق البحث عن عمل",
        "desired_job_type":      "نوع العمل المطلوب",
        "ever_worked":           "سبق له العمل",
        # ── Skills & Training ─────────────────────────────────────────────────
        "main_skills":           "المهارات الأساسية المتعلقة بالعمل",
        "qualification_match":   "تطابق المؤهلات مع الوظيفة",
        "training_participation":"المشاركة في التدريب (آخر 12 شهراً)",
        "labour_market_barriers":"عوائق سوق العمل",
        # ── Digital Work ──────────────────────────────────────────────────────
        "platform_work":         "العمل عبر المنصات الرقمية",
        "online_business":       "النشاط التجاري الإلكتروني",
        # ── Quality & Feedback ────────────────────────────────────────────────
        "job_satisfaction":          "الرضا الوظيفي (1-5)",
        "work_safety":               "سلامة بيئة العمل",
        "workplace_issues":          "مشكلات بيئة العمل (تحرش/تمييز/إلخ)",
        "work_life_balance":         "التوازن بين العمل والحياة",
        "question_clarity":          "وضوح الأسئلة (1-5)",
        "difficulty_answering":      "صعوبة الإجابة على أسئلة بعينها",
        "survey_comments":           "تعليقات واقتراحات على المسح",
        # ── Education conditional ─────────────────────────────────────────────
        "field_of_study":            "مجال الدراسة الرئيسي",
        # ── Employed extras ───────────────────────────────────────────────────
        "secondary_job_hours":       "ساعات العمل في الوظيفة الثانوية الأسبوع الماضي",
        "bonuses":                   "المكافآت والحوافز (آخر 12 شهراً)",
        "pension_scheme":            "صندوق التقاعد / ضمان نهاية الخدمة",
        # ── Emiratization ─────────────────────────────────────────────────────
        "emiratization_program":     "التسجيل في برامج التوطين",
        # ── Digital conditional ───────────────────────────────────────────────
        "platform_names":            "المنصات الرقمية المستخدمة للعمل",
        "platform_hours":            "ساعات العمل الأسبوعية عبر المنصات",
        # ── Previous employment ───────────────────────────────────────────────
        "last_job_sector":           "قطاع آخر وظيفة",
        "highest_previous_salary":   "أعلى راتب شهري سابق (درهم)",
    }
    # Urdu/Hindi/Tagalog field labels, added 2026-08-29 alongside the
    # CLARIFYING/VALIDATING/COMPLETING translations below — machine-drafted,
    # not yet reviewed by a native speaker.
    _FIELD_LABELS_UR: dict[str, str] = {
        "employment_status":     "ملازمت کی صورتحال",
        "education_level":       "تعلیم کی سطح",
        "employment_nature":     "روزگار کا انتظام (تنخواہ دار/آجر/خود روزگار)",
        "employment_sector":     "روزگار کا شعبہ (سرکاری/نجی/نیم سرکاری)",
        "job_title":             "جاب ٹائٹل",
        "job_duties":            "بنیادی کام اور فرائض",
        "industry":              "صنعت / شعبہ",
        "hours_per_week":        "ہفتہ وار معمول کے گھنٹے",
        "employment_type":       "روزگار کی قسم (کل وقتی/جز وقتی/موسمی)",
        "monthly_wage_range":    "ماہانہ تنخواہ کی حد (درہم)",
        "job_search_active":     "فعال طور پر کام تلاش کرنا (ہاں/نہیں)",
        "available_for_work":    "2 ہفتوں میں کام کے لیے دستیاب (ہاں/نہیں)",
        "unemployment_duration": "کام کی تلاش کی مدت",
        "last_job_title":        "آخری جاب ٹائٹل",
        "reason_left_job":       "آخری نوکری چھوڑنے کی وجہ",
        "outside_lf_reason":     "کام تلاش نہ کرنے کی وجہ",
        "ai_preference":         "انٹرویو لینے والے کی ترجیح (AI بمقابلہ انسان)",
        "data_confidence":       "ڈیٹا کی رازداری پر اعتماد",
        "gender":                "صنف",
        "nationality":           "قومیت",
        "marital_status":        "ازدواجی حیثیت",
        "emirate":               "رہائشی امارات",
        "uae_residence_duration":"متحدہ عرب امارات میں قیام کی مدت",
        "vocational_training":   "پیشہ ورانہ تربیت / سرٹیفیکیشن (گزشتہ 12 ماہ)",
        "actual_hours_worked":   "گزشتہ ہفتے اصل کام کیے گئے گھنٹے",
        "secondary_job":         "ثانوی نوکری",
        "underemployment":       "کم روزگاری / گھنٹوں کی ترجیح",
        "contract_type":         "معاہدے کی قسم",
        "remote_work":           "ریموٹ ورک کا انتظام",
        "salary_allowances":     "تنخواہ کے الاؤنسز",
        "health_insurance":      "صحت کی انشورنس کوریج",
        "job_search_methods":    "کام تلاش کرنے کے طریقے",
        "desired_job_type":      "مطلوبہ کام کی قسم",
        "ever_worked":           "پچھلا کام کا تجربہ",
        "main_skills":           "بنیادی کام سے متعلق مہارتیں",
        "qualification_match":   "قابلیت اور نوکری کی مطابقت",
        "training_participation":"تربیت میں شرکت (گزشتہ 12 ماہ)",
        "labour_market_barriers":"لیبر مارکیٹ کی رکاوٹیں",
        "platform_work":         "پلیٹ فارم / گِگ ورک",
        "online_business":       "آن لائن کاروبار / ای کامرس",
        "job_satisfaction":          "ملازمت سے اطمینان (1-5)",
        "work_safety":               "کام کی جگہ کی حفاظت",
        "workplace_issues":          "کام کی جگہ کے مسائل (ہراسانی/امتیازی سلوک/وغیرہ)",
        "work_life_balance":         "کام اور زندگی کا توازن",
        "question_clarity":          "سوالات کی وضاحت کی درجہ بندی (1-5)",
        "difficulty_answering":      "مخصوص سوالات کا جواب دینے میں دشواری",
        "survey_comments":           "سروے کے تبصرے / تجاویز",
        "field_of_study":            "بنیادی مضمون تعلیم",
        "secondary_job_hours":       "گزشتہ ہفتے ثانوی نوکری میں کام کیے گئے گھنٹے",
        "bonuses":                   "بونس / مراعات (گزشتہ 12 ماہ)",
        "pension_scheme":            "پنشن / سروس کے اختتام پر گریجویٹی اسکیم",
        "emiratization_program":     "ایمرٹائزیشن پروگرام میں رجسٹریشن",
        "platform_names":            "کام کے لیے استعمال ہونے والے ڈیجیٹل پلیٹ فارمز",
        "platform_hours":            "ڈیجیٹل پلیٹ فارمز پر ہفتہ وار گھنٹے",
        "last_job_sector":           "آخری نوکری کا شعبہ",
        "highest_previous_salary":   "سب سے زیادہ پچھلی ماہانہ تنخواہ (درہم)",
    }
    _FIELD_LABELS_HI: dict[str, str] = {
        "employment_status":     "रोजगार स्थिति",
        "education_level":       "शिक्षा स्तर",
        "employment_nature":     "रोजगार व्यवस्था (वेतनभोगी/नियोक्ता/स्वरोजगार)",
        "employment_sector":     "रोजगार क्षेत्र (सरकारी/निजी/अर्ध-सरकारी)",
        "job_title":             "पद (जॉब टाइटल)",
        "job_duties":            "मुख्य कार्य और कर्तव्य",
        "industry":              "उद्योग / क्षेत्र",
        "hours_per_week":        "सामान्य साप्ताहिक घंटे",
        "employment_type":       "रोजगार प्रकार (पूर्णकालिक/अंशकालिक/मौसमी)",
        "monthly_wage_range":    "मासिक वेतन सीमा (AED)",
        "job_search_active":     "सक्रिय रूप से काम खोजना (हाँ/नहीं)",
        "available_for_work":    "2 सप्ताह में काम के लिए उपलब्ध (हाँ/नहीं)",
        "unemployment_duration": "काम खोजने की अवधि",
        "last_job_title":        "अंतिम पद",
        "reason_left_job":       "अंतिम नौकरी छोड़ने का कारण",
        "outside_lf_reason":     "काम न खोजने का कारण",
        "ai_preference":         "साक्षात्कारकर्ता प्राथमिकता (AI बनाम मानव)",
        "data_confidence":       "डेटा गोपनीयता में विश्वास",
        "gender":                "लिंग",
        "nationality":           "राष्ट्रीयता",
        "marital_status":        "वैवाहिक स्थिति",
        "emirate":               "निवास अमीरात",
        "uae_residence_duration":"यूएई में निवास अवधि",
        "vocational_training":   "व्यावसायिक प्रशिक्षण / प्रमाणन (पिछले 12 महीने)",
        "actual_hours_worked":   "पिछले सप्ताह वास्तव में काम किए गए घंटे",
        "secondary_job":         "द्वितीयक नौकरी",
        "underemployment":       "अल्परोजगार / घंटों की प्राथमिकता",
        "contract_type":         "अनुबंध प्रकार",
        "remote_work":           "रिमोट वर्क व्यवस्था",
        "salary_allowances":     "वेतन भत्ते",
        "health_insurance":      "स्वास्थ्य बीमा कवरेज",
        "job_search_methods":    "काम खोजने के तरीके",
        "desired_job_type":      "वांछित कार्य प्रकार",
        "ever_worked":           "पिछला कार्य अनुभव",
        "main_skills":           "मुख्य कार्य-संबंधी कौशल",
        "qualification_match":   "नौकरी के साथ योग्यता मिलान",
        "training_participation":"प्रशिक्षण में भागीदारी (पिछले 12 महीने)",
        "labour_market_barriers":"श्रम बाजार बाधाएं",
        "platform_work":         "प्लेटफ़ॉर्म / गिग वर्क",
        "online_business":       "ऑनलाइन व्यवसाय / ई-कॉमर्स",
        "job_satisfaction":          "नौकरी संतुष्टि (1-5)",
        "work_safety":               "कार्यस्थल सुरक्षा",
        "workplace_issues":          "कार्यस्थल के मुद्दे (उत्पीड़न/भेदभाव/आदि)",
        "work_life_balance":         "कार्य-जीवन संतुलन",
        "question_clarity":          "प्रश्न स्पष्टता रेटिंग (1-5)",
        "difficulty_answering":      "विशिष्ट प्रश्नों का उत्तर देने में कठिनाई",
        "survey_comments":           "सर्वेक्षण टिप्पणियां / सुझाव",
        "field_of_study":            "मुख्य अध्ययन क्षेत्र",
        "secondary_job_hours":       "पिछले सप्ताह द्वितीयक नौकरी में काम किए गए घंटे",
        "bonuses":                   "बोनस / प्रोत्साहन (पिछले 12 महीने)",
        "pension_scheme":            "पेंशन / सेवा-समाप्ति उपदान योजना",
        "emiratization_program":     "एमिराटाइजेशन कार्यक्रम पंजीकरण",
        "platform_names":            "काम के लिए उपयोग किए गए डिजिटल प्लेटफ़ॉर्म",
        "platform_hours":            "डिजिटल प्लेटफ़ॉर्म पर साप्ताहिक घंटे",
        "last_job_sector":           "अंतिम नौकरी क्षेत्र",
        "highest_previous_salary":   "उच्चतम पिछला मासिक वेतन (AED)",
    }
    _FIELD_LABELS_TL: dict[str, str] = {
        "employment_status":     "Katayuan sa Trabaho",
        "education_level":       "Antas ng Edukasyon",
        "employment_nature":     "Kaayusan sa Trabaho (sinasahurang empleyado/employer/nagsasarili)",
        "employment_sector":     "Sektor ng Trabaho (gobyerno/pribado/semi-gobyerno)",
        "job_title":             "Job Title",
        "job_duties":            "Mga Pangunahing Gawain at Tungkulin",
        "industry":              "Industriya / Sektor",
        "hours_per_week":        "Karaniwang Oras Kada Linggo",
        "employment_type":       "Uri ng Trabaho (full-time/part-time/pana-panahon)",
        "monthly_wage_range":    "Saklaw ng Buwanang Suweldo (AED)",
        "job_search_active":     "Aktibong Naghahanap ng Trabaho (Oo/Hindi)",
        "available_for_work":    "Available Magsimula sa Loob ng 2 Linggo (Oo/Hindi)",
        "unemployment_duration": "Tagal ng Paghahanap ng Trabaho",
        "last_job_title":        "Pinakahuling Job Title",
        "reason_left_job":       "Dahilan ng Pag-alis sa Huling Trabaho",
        "outside_lf_reason":     "Dahilan ng Hindi Paghahanap ng Trabaho",
        "ai_preference":         "Kagustuhan sa Interviewer (AI kumpara sa Tao)",
        "data_confidence":       "Kumpiyansa sa Privacy ng Data",
        "gender":                "Kasarian",
        "nationality":           "Nasyonalidad",
        "marital_status":        "Katayuan sa Kasal",
        "emirate":               "Emirate ng Paninirahan",
        "uae_residence_duration":"Tagal ng Paninirahan sa UAE",
        "vocational_training":   "Bokasyonal na Pagsasanay / Sertipikasyon (nakaraang 12 buwan)",
        "actual_hours_worked":   "Aktwal na Oras na Nagtrabaho Noong Nakaraang Linggo",
        "secondary_job":         "Pangalawang Trabaho",
        "underemployment":       "Underemployment / Kagustuhan sa Oras",
        "contract_type":         "Uri ng Kontrata",
        "remote_work":           "Kaayusan sa Remote Work",
        "salary_allowances":     "Mga Allowance sa Suweldo",
        "health_insurance":      "Coverage ng Health Insurance",
        "job_search_methods":    "Mga Paraan ng Paghahanap ng Trabaho",
        "desired_job_type":      "Uri ng Trabahong Hinahanap",
        "ever_worked":           "Naunang Karanasan sa Trabaho",
        "main_skills":           "Mga Pangunahing Kasanayang Kaugnay sa Trabaho",
        "qualification_match":   "Pagtutugma ng Kwalipikasyon sa Trabaho",
        "training_participation":"Paglahok sa Pagsasanay (nakaraang 12 buwan)",
        "labour_market_barriers":"Mga Hadlang sa Labor Market",
        "platform_work":         "Platform / Gig Work",
        "online_business":       "Online Business / E-Commerce",
        "job_satisfaction":          "Kasiyahan sa Trabaho (1-5)",
        "work_safety":               "Kaligtasan sa Lugar ng Trabaho",
        "workplace_issues":          "Mga Isyu sa Lugar ng Trabaho (panliligalig/diskriminasyon/atbp.)",
        "work_life_balance":         "Balanse ng Trabaho at Buhay",
        "question_clarity":          "Rating ng Kalinawan ng Tanong (1-5)",
        "difficulty_answering":      "Kahirapan sa Pagsagot ng mga Tiyak na Tanong",
        "survey_comments":           "Mga Komento / Mungkahi sa Survey",
        "field_of_study":            "Pangunahing Larangan ng Pag-aaral",
        "secondary_job_hours":       "Oras na Nagtrabaho sa Pangalawang Trabaho Noong Nakaraang Linggo",
        "bonuses":                   "Bonus / Insentibo (nakaraang 12 buwan)",
        "pension_scheme":            "Pension / End-of-Service Gratuity Scheme",
        "emiratization_program":     "Rehistro sa Emiratization Program",
        "platform_names":            "Mga Digital Platform na Ginamit sa Trabaho",
        "platform_hours":            "Oras Kada Linggo sa Digital Platforms",
        "last_job_sector":           "Sektor ng Huling Trabaho",
        "highest_previous_salary":   "Pinakamataas na Naunang Buwanang Suweldo (AED)",
    }

    # ------------------------------------------------------------------
    # Dev stub (used when no LLM is available in development mode)
    # ------------------------------------------------------------------

    # Human-readable labels for pre-filled field display
    _FIELD_LABELS: dict[str, str] = {
        "employment_status":  "Employment Status",
        "education_level":    "Education Level",
        "gender":             "Gender",
        "nationality":        "Nationality",
        "marital_status":     "Marital Status",
        "emirate":            "Emirate",
        "uae_residence_duration": "UAE Residence",
        "vocational_training":"Vocational Training",
        "employment_nature":  "Employment Nature",
        "employment_sector":  "Sector",
        "job_title":          "Job Title",
        "job_duties":         "Job Duties",
        "industry":           "Industry",
        "hours_per_week":     "Weekly Hours",
        "employment_type":    "Employment Type",
        "monthly_wage_range": "Monthly Wage",
        "education_level":    "Education Level",
    }

    def _returning_user_summary(self, collected_data: dict) -> str:
        """Build a bullet-list summary of pre-filled fields."""
        lines = []
        for field, label in self._FIELD_LABELS.items():
            val = collected_data.get(field)
            if val:
                lines.append(f"  • {label}: {val}")
        return "\n".join(lines[:10])  # cap at 10 fields

    def _greeting_with_first_question(self, ctx: "ConversationContext") -> str:
        """Return intro message + first survey question in a single turn."""
        lang = ctx.language
        is_returning = ctx.is_returning and bool(ctx.collected_data)

        if is_returning:
            summary = self._returning_user_summary(ctx.collected_data)
            field_order = self._get_field_order(ctx.collected_data)
            next_unanswered = next(
                (f for f in field_order if f not in ctx.collected_data), None
            )
            next_q_label = self._FIELD_LABELS.get(next_unanswered, next_unanswered) if next_unanswered else None

            if lang in ("ar", "ar-gulf"):
                msg = (
                    "مرحبًا بعودتك! وجدت بياناتك من مسحنا السابق:\n"
                    f"{summary}\n\n"
                    "هل تغيّر أي شيء منذ آخر مسح؟ "
                    "سأبدأ بالأسئلة التي لم تُجب عليها بعد."
                )
            elif lang == "ur":
                msg = (
                    "خوش آمدید! مجھے آپ کے پچھلے سروے سے آپ کی معلومات مل گئی:\n"
                    f"{summary}\n\n"
                    "کیا آخری سروے کے بعد سے کچھ بدلا ہے؟ "
                    "میں ابھی ان سوالوں سے شروع کروں گا جن کے جوابات ابھی تک نہیں ملے۔"
                )
            elif lang == "hi":
                msg = (
                    "वापसी पर स्वागत है! मुझे आपके पिछले सर्वेक्षण से आपकी जानकारी मिली:\n"
                    f"{summary}\n\n"
                    "क्या पिछले सर्वेक्षण के बाद से कुछ बदला है? "
                    "मैं उन प्रश्नों से शुरू करूँगा जिनके उत्तर अभी तक नहीं मिले।"
                )
            elif lang == "tl":
                msg = (
                    "Maligayang pagbabalik! Nahanap ko ang iyong impormasyon mula sa nakaraang survey:\n"
                    f"{summary}\n\n"
                    "May nagbago ba mula noong huling survey? "
                    "Magsisimula ako sa mga tanong na hindi pa nasasagot."
                )
            else:
                next_q_hint = f"\n\nLet's continue with: {next_q_label}." if next_q_label else ""
                msg = (
                    "Welcome back! I found your information from your previous survey:\n"
                    f"{summary}\n\n"
                    "Has anything changed since your last survey? "
                    "I'll pick up from where we left off and ask only about fields not yet answered."
                    f"{next_q_hint}"
                )
            return msg

        # ── First-time user greeting ─────────────────────────────────────────
        if lang in ("ar", "ar-gulf"):
            return (
                "مرحبًا! أنا مساعد مسح القوى العاملة. "
                "سأطرح عليك بعض الأسئلة حول وضعك الوظيفي والتعليمي "
                "— لن يستغرق ذلك سوى بضع دقائق.\n\n"
                "للبدء: ما هي حالة توظيفك الحالية؟\n"
                "(موظف / عاطل عن العمل / خارج سوق العمل)"
            )
        if lang == "ur":
            return (
                "السلام علیکم! میں آپ کا لیبر فورس سروے اسسٹنٹ ہوں۔ "
                "میں آپ سے روزگار اور تعلیمی پس منظر کے بارے میں چند سوالات پوچھوں گا "
                "— یہ عام طور پر 3–5 منٹ لیتا ہے۔\n\n"
                "شروع کرنے کے لیے: آپ کی موجودہ روزگار کی حیثیت کیا ہے؟\n"
                "(ملازم / بے روزگار / افرادی قوت سے باہر)"
            )
        if lang == "hi":
            return (
                "नमस्ते! मैं आपका श्रम बल सर्वेक्षण सहायक हूँ। "
                "मैं आपसे रोज़गार और शैक्षिक पृष्ठभूमि के बारे में कुछ प्रश्न पूछूँगा "
                "— इसमें आमतौर पर 3–5 मिनट लगते हैं।\n\n"
                "शुरू करने के लिए: आपकी वर्तभान रोज़गार स्थिति क्या है?\n"
                "(नियोजित / बेरोजगार / श्रम बल से बाहर)"
            )
        if lang == "tl":
            return (
                "Kumusta! Ako ang iyong Labour Force Survey assistant. "
                "Magtatanong ako ng ilang katanungan tungkol sa iyong trabaho at "
                "educational background — karaniwang tumatagal ng 3–5 minuto.\n\n"
                "Para magsimula: ano ang iyong kasalukuyang katayuan sa trabaho?\n"
                "(Employed / Unemployed / Not in the labour force)"
            )
        return (
            "Hello! I'm your Labour Force Survey assistant. "
            "I'll ask you a few questions about your employment and educational "
            "background — this usually takes about 3–5 minutes.\n\n"
            "To begin: what is your current employment status?\n"
            "(Employed / Unemployed / Not in the labour force)"
        )

    def _questions_and_ack_for_lang(self, lang: str) -> tuple[dict[str, str], dict[str, object]]:
        """
        Return (question dict, acknowledgment dict) for the given language code.

        Covers all 5 supported languages for follow-up questions (added
        2026-08-29 — previously only ar/ar-gulf had a translated question
        dict here, so ur/hi/tl silently fell back to English after the
        first, greeting-only message). Acknowledgments for ur/hi/tl are
        generic (see _generic_ack_ur/hi/tl) rather than field-specific like
        en/ar — a smaller, disclosed scope, not a bug.
        """
        if lang in ("ar", "ar-gulf"):
            return self._EXACT_QUESTIONS_AR, self._ACK_AR
        if lang == "ur":
            return self._EXACT_QUESTIONS_UR, self._ACK_UR
        if lang == "hi":
            return self._EXACT_QUESTIONS_HI, self._ACK_HI
        if lang == "tl":
            return self._EXACT_QUESTIONS_TL, self._ACK_TL
        return self._EXACT_QUESTIONS_EN, self._ACK_EN

    # Static (non-field-specific) UI strings for CLARIFYING/VALIDATING/
    # COMPLETING, keyed by language then by string id. Added 2026-08-29
    # alongside the field-label translations above, closing the last
    # disclosed gap: these states previously fell back to English for
    # ur/hi/tl even after the follow-up questions were translated.
    # ur/hi/tl entries are machine-drafted and not yet reviewed by a native
    # speaker, same caveat as everywhere else in this pass.
    _STATIC_UI: dict[str, dict[str, str]] = {
        "ar": {
            "collecting_done": "شكرًا! دعني أراجع إجاباتك معك.",
            "clarify_repeat": "آسف على الالتباس! أحتاج فقط إلى معرفة: {guidance}",
            "clarify_first": "لم أفهم إجابتك بشكل صحيح. أحتاج إلى معرفة: {guidance}",
            "no_data": "(لا بيانات)",
            "correction_rejected": "عذرًا، لم أفهم القيمة الجديدة بوضوح. هل يمكنك إعادة ذكر: {label}؟",
            "correction_applied": "تم التحديث! إليك الملخص المحدّث:\n{summary}\n\nهل جميع المعلومات صحيحة الآن؟ (نعم / لا، أريد تصحيح شيء)",
            "correction_no_target": "لا مشكلة — ما الذي تريد تصحيحه؟ اذكر الحقل والقيمة الصحيحة، مثال: \"مستوى التعليم يجب أن يكون بكالوريوس\".",
            "validating_summary": "إليك ملخص ما جمعناه:\n{summary}\n\nهل جميع المعلومات صحيحة؟ (نعم / لا، أريد تصحيح شيء)",
            "completing": "شكرًا جزيلًا على وقتك ومشاركتك في مسح القوى العاملة! إجاباتك ستساهم في أبحاث سوق العمل وصنع القرار. نتمنى لك يومًا سعيدًا!",
            "fallback": "شكرًا على إجابتك.",
        },
        "ur": {
            "collecting_done": "شکریہ! اب مجھے آپ کے جوابات کا جائزہ لینے دیں۔",
            "clarify_repeat": "الجھن کے لیے معذرت! مجھے صرف یہ جاننا ہے: {guidance}",
            "clarify_first": "مجھے آپ کا جواب ٹھیک سے سمجھ نہیں آیا۔ مجھے یہ جاننا ہے: {guidance}",
            "no_data": "(کوئی ڈیٹا نہیں)",
            "correction_rejected": "معذرت، مجھے نئی قیمت واضح طور پر سمجھ نہیں آئی۔ کیا آپ دوبارہ بتا سکتے ہیں: {label}؟",
            "correction_applied": "ٹھیک ہے، میں نے اسے اپ ڈیٹ کر دیا۔ یہ رہا آپ کا اپ ڈیٹ شدہ خلاصہ:\n{summary}\n\nکیا اب سب کچھ درست ہے؟ (ہاں / نہیں، میں کچھ درست کرنا چاہتا ہوں)",
            "correction_no_target": "کوئی مسئلہ نہیں — آپ کیا درست کرنا چاہتے ہیں؟ فیلڈ اور درست قیمت بتائیں، مثلاً: \"تعلیم کی سطح بیچلر ہونی چاہیے\"۔",
            "validating_summary": "یہ رہا ہمارا جمع کردہ خلاصہ:\n{summary}\n\nکیا سب معلومات درست ہیں؟ (ہاں / نہیں، میں کچھ درست کرنا چاہتا ہوں)",
            "completing": "لیبر فورس سروے میں آپ کے وقت اور شرکت کے لیے بہت شکریہ! آپ کے جوابات لیبر مارکیٹ کی تحقیق اور پالیسی فیصلوں میں مددگار ثابت ہوں گے۔ آپ کا دن اچھا گزرے!",
            "fallback": "آپ کے جواب کا شکریہ۔",
        },
        "hi": {
            "collecting_done": "धन्यवाद! अब मुझे आपके उत्तरों की समीक्षा करने दें।",
            "clarify_repeat": "भ्रम के लिए क्षमा करें! मुझे बस यह जानना है: {guidance}",
            "clarify_first": "मुझे आपका उत्तर ठीक से समझ नहीं आया। मुझे यह जानना है: {guidance}",
            "no_data": "(कोई डेटा नहीं)",
            "correction_rejected": "क्षमा करें, मुझे नया मान स्पष्ट रूप से समझ नहीं आया। क्या आप फिर से बता सकते हैं: {label}?",
            "correction_applied": "ठीक है, मैंने इसे अपडेट कर दिया है। यह रहा आपका अपडेटेड सारांश:\n{summary}\n\nक्या अब सब कुछ सही है? (हाँ / नहीं, मैं कुछ ठीक करना चाहता/चाहती हूँ)",
            "correction_no_target": "कोई बात नहीं — आप क्या ठीक करना चाहते हैं? फ़ील्ड और सही मान बताएं, जैसे: \"शिक्षा स्तर स्नातक होना चाहिए\"।",
            "validating_summary": "यह रहा हमारा एकत्रित सारांश:\n{summary}\n\nक्या सारी जानकारी सही है? (हाँ / नहीं, मैं कुछ ठीक करना चाहता/चाहती हूँ)",
            "completing": "श्रम बल सर्वेक्षण में आपके समय और भागीदारी के लिए बहुत धन्यवाद! आपके उत्तर महत्वपूर्ण श्रम बाजार अनुसंधान और नीति निर्णयों में योगदान देंगे। आपका दिन शुभ हो!",
            "fallback": "आपके उत्तर के लिए धन्यवाद।",
        },
        "tl": {
            "collecting_done": "Salamat! Hayaan mo akong suriin ang iyong mga sagot.",
            "clarify_repeat": "Paumanhin sa kalituhan! Kailangan ko lang malaman: {guidance}",
            "clarify_first": "Hindi ko naunawaan nang tama ang iyong sagot. Kailangan kong malaman: {guidance}",
            "no_data": "(walang datos)",
            "correction_rejected": "Paumanhin, hindi ko naunawaan nang malinaw ang bagong value. Maaari mo bang ulitin: {label}?",
            "correction_applied": "Sige, na-update ko na iyon para sa iyo. Narito ang iyong na-update na buod:\n{summary}\n\nTama na ba ang lahat ngayon? (oo / hindi, gusto kong itama ang isang bagay)",
            "correction_no_target": "Walang problema — ano ang gusto mong itama? Sabihin ang field at ang tamang value, hal. \"dapat bachelor ang antas ng edukasyon\".",
            "validating_summary": "Narito ang buod ng nakolekta natin:\n{summary}\n\nTama ba ang lahat? (oo / hindi, gusto kong itama ang isang bagay)",
            "completing": "Maraming salamat sa iyong oras at pakikilahok sa Labour Force Survey! Ang iyong mga sagot ay makakatulong sa mahahalagang pananaliksik sa labor market at mga desisyon sa patakaran. Magandang araw sa iyo!",
            "fallback": "Salamat sa iyong sagot.",
        },
        "en": {
            "collecting_done": "Great, I've collected all the information I need. Let me review it with you.",
            "clarify_repeat": "I apologise for the confusion! I just need to know: {guidance}",
            "clarify_first": "I didn't quite catch that. I need to know: {guidance}",
            "no_data": "(no data)",
            "correction_rejected": "Sorry, I didn't quite catch that clearly. Could you give me your {label} again?",
            "correction_applied": "Got it, I've updated that for you. Here's your updated summary:\n{summary}\n\nIs everything correct now? (yes / no, I'd like to correct something)",
            "correction_no_target": "No problem — what would you like to correct? Tell me the field and the correct value, e.g. \"education level should be bachelor\".",
            "validating_summary": "Here's a summary of what I've collected:\n{summary}\n\nIs everything correct? (yes / no, I'd like to correct something)",
            "completing": "Thank you so much for your time and participation in the Labour Force Survey! Your responses will contribute to important labour market research and policy decisions. Have a wonderful day!",
            "fallback": "Thank you for your response.",
        },
    }

    def _lang_key(self, lang: str) -> str:
        """Normalise a raw language code to one of the 5 supported UI keys."""
        if lang in ("ar", "ar-gulf"):
            return "ar"
        if lang in ("ur", "hi", "tl"):
            return lang
        return "en"

    def _field_labels_for_lang(self, lang: str) -> dict[str, str]:
        lk = self._lang_key(lang)
        return {
            "ar": self._FIELD_LABELS_AR,
            "ur": self._FIELD_LABELS_UR,
            "hi": self._FIELD_LABELS_HI,
            "tl": self._FIELD_LABELS_TL,
            "en": self._FIELD_LABELS_EN,
        }[lk]

    def _dev_stub_response(self, ctx: ConversationContext) -> str:
        """Rule-based fallback used when no LLM is configured (dev mode only)."""
        lang = ctx.language
        state = ctx.state
        lk = self._lang_key(lang)
        ui = self._STATIC_UI[lk]
        questions, ack_map = self._questions_and_ack_for_lang(lang)

        if state == ConversationState.GREETING:
            return self._greeting_with_first_question(ctx)

        if state == ConversationState.COLLECTING_INFO:
            field_order = self._get_field_order(ctx.collected_data)
            answered = set(ctx.collected_data.keys())

            for i, fld in enumerate(field_order):
                if fld not in answered:
                    question = questions.get(fld, f"Please provide: {fld}")
                    # Acknowledge the field answered just before this one
                    prev_field = field_order[i - 1] if i > 0 else None
                    ack = ""
                    if prev_field and prev_field in ctx.collected_data:
                        fn = ack_map.get(prev_field)
                        if fn:
                            ack = fn(str(ctx.collected_data[prev_field])) + "\n\n"
                    return ack + question

            return ui["collecting_done"]

        if state == ConversationState.CLARIFYING:
            fld = ctx.clarification_target
            count = ctx.clarification_count
            labels = self._field_labels_for_lang(lang)
            guidance = labels.get(fld, fld or "your previous answer")
            if count and count >= 2:
                return ui["clarify_repeat"].format(guidance=guidance)
            return ui["clarify_first"].format(guidance=guidance)

        if state == ConversationState.VALIDATING:
            # Self-healing stray-field purge, added 2026-09-02: a real, live
            # session reported by a user was still showing a bogus "Job
            # Title" field (job_title) alongside the real "Most Recent Job
            # Title" (last_job_title) for an UNEMPLOYED respondent, even
            # after the underlying wrong-field-injection bug in
            # _llm_extract_correction had already been fixed and deployed.
            # Root cause: that session was corrupted by the OLD, buggy code
            # in an earlier turn, before the fix existed -- the fix stops
            # NEW corruption, but nothing ever cleaned up data already sitting
            # in ctx.collected_data from before the fix. This purge is
            # deliberately unconditional and defensive: it runs every time
            # the VALIDATING summary is built and drops any key that isn't
            # part of _get_field_order()'s current output for this
            # respondent's data, regardless of how it got there (this exact
            # historical bug, a future bug, or a legitimate employment_status
            # change mid-session leaving the old path's fields behind) --
            # rather than only fixing the one known cause.
            _real_fields = set(self._get_field_order(ctx.collected_data))
            _stray = set(ctx.collected_data.keys()) - _real_fields
            for _k in _stray:
                _logger.info("Purging stray field from collected_data (not in respondent's real path): %s", _k)
                del ctx.collected_data[_k]

            labels = self._field_labels_for_lang(lang)
            # Fixed 2026-09-02: this previously printed the raw stored value
            # verbatim for every language except Arabic -- confirmed live,
            # e.g. "Interviewer Preference (AI vs Human): prefer_human" and
            # "Confidence in Data Privacy: somewhat_confident" shown with the
            # underscore un-replaced, in English. The turn-by-turn
            # acknowledgments (_ACK_EN) already did `v.replace('_', ' ')` for
            # exactly these fields; this final review summary just never got
            # the same treatment. ur/hi/tl get the same underscore-to-space
            # fallback as English (no per-language enum-value tables exist
            # for those three, same disclosed scope as _ACK_UR/HI/TL above).
            def _fmt_val(v: object) -> str:
                if lk == "ar":
                    return _ar_val(v)
                s = str(v)
                return s.replace("_", " ") if "_" in s else s

            lines = [
                f"  • {labels.get(k, k)}: {_fmt_val(v)}"
                for k, v in ctx.collected_data.items()
            ]
            summary = "\n".join(lines) or ui["no_data"]
            if ctx.correction_rejected_field:
                fld = ctx.correction_rejected_field
                ctx.correction_rejected_field = None
                label = labels.get(fld, fld)
                return ui["correction_rejected"].format(label=label)
            if ctx.correction_applied:
                ctx.correction_applied = False
                return ui["correction_applied"].format(summary=summary)
            if ctx.correction_no_target:
                ctx.correction_no_target = False
                return ui["correction_no_target"]
            return ui["validating_summary"].format(summary=summary)

        if state == ConversationState.COMPLETING:
            return ui["completing"]

        return ui["fallback"]

    # ------------------------------------------------------------------
    # Task construction
    # ------------------------------------------------------------------

    def _get_next_question_instruction(self, ctx: ConversationContext) -> str:
        """
        Return an explicit instruction telling the LLM exactly which question to ask
        next. This prevents the LLM from asking questions out of order or rephrasing
        them ambiguously.
        """
        lang = ctx.language
        field_order = self._get_field_order(ctx.collected_data)
        next_field = next(
            (f for f in field_order if f not in ctx.collected_data), None
        )
        if next_field is None:
            return ""

        qs, ack_map = self._questions_and_ack_for_lang(lang)
        question_text = qs.get(next_field, "")
        if not question_text:
            return ""

        idx = field_order.index(next_field)
        prev_field = field_order[idx - 1] if idx > 0 else None

        ack_instruction = ""
        if prev_field and prev_field in ctx.collected_data:
            fn = ack_map.get(prev_field)
            if fn:
                ack_example = fn(str(ctx.collected_data[prev_field]))
                ack_instruction = (
                    f"\nFirst give a brief warm acknowledgement of their previous answer "
                    f"(something like: \"{ack_example}\"), then ask the next question."
                )

        return (
            f"\n\nNEXT QUESTION (MANDATORY): You MUST ask about '{next_field}' and "
            f"ONLY about '{next_field}'. Do NOT ask any other question or deviate "
            f"from this order. Ask it exactly as follows:\n{question_text}"
            f"{ack_instruction}"
        )

    def _build_task(self, ctx: ConversationContext) -> Task:
        """Build a CrewAI Task for the current FSM state and language."""
        lang = ctx.language
        pk = _LANG_PROMPT_KEY.get(lang, "en")   # prompt-dict key

        clarify_note = ""
        if ctx.state == ConversationState.CLARIFYING and ctx.clarification_target:
            labels = self._FIELD_LABELS_EN if pk == "en" else self._FIELD_LABELS_AR
            field_label = labels.get(ctx.clarification_target, ctx.clarification_target)
            clarify_note = (
                f"\n\nCLARIFICATION NEEDED: The respondent's last answer did not clearly "
                f"address '{field_label}'. Politely tell them their answer was unclear, "
                f"explain what you need, and re-ask specifically about this field with an example."
            )

        next_q_note = ""
        if ctx.state == ConversationState.COLLECTING_INFO:
            next_q_note = self._get_next_question_instruction(ctx)

        correction_note = ""
        if ctx.state == ConversationState.VALIDATING and ctx.correction_rejected_field:
            field_label = self._FIELD_LABELS_AR.get(ctx.correction_rejected_field) if pk == "ar" \
                else self._FIELD_LABELS_EN.get(ctx.correction_rejected_field, ctx.correction_rejected_field)
            correction_note = (
                f"\n\nCORRECTION NOT UNDERSTOOD: The respondent tried to correct '{field_label}' "
                "but the new value couldn't be confidently parsed out of their message (it looked "
                "like a sentence fragment rather than an answer). Do NOT guess a value and do NOT "
                f"claim anything was updated. Apologise briefly and ask specifically: "
                f"what should '{field_label}' be?"
            )
            ctx.correction_rejected_field = None
        elif ctx.state == ConversationState.VALIDATING and ctx.correction_applied:
            correction_note = (
                "\n\nCORRECTION APPLIED: The respondent just corrected one or more answers. "
                "The COLLECTED DATA above already reflects the updated values. "
                "Acknowledge the correction naturally (e.g. 'I've updated that for you.'), "
                "then read back the FULL updated summary of all answers and ask the respondent "
                "to confirm everything is now correct."
            )
        elif ctx.state == ConversationState.VALIDATING and ctx.correction_no_target:
            correction_note = (
                "\n\nCORRECTION REQUESTED, NO FIELD NAMED: The respondent said they want to "
                "correct something (e.g. just 'no') but did not say what. Do NOT re-show the "
                "full summary again. Instead, briefly acknowledge and ask specifically which "
                "field they'd like to change, with a short example of how to phrase it."
            )
            ctx.correction_no_target = False

        lang_instruction = _LANG_RESPONSE_INSTRUCTION.get(lang, "Respond in English.")

        description = "\n\n".join([
            _SYSTEM_BASE[pk],
            f"CURRENT STATE: {ctx.state.value}",
            f"STATE INSTRUCTIONS:\n{_STATE_INSTRUCTIONS[ctx.state][pk]}{clarify_note}{next_q_note}{correction_note}",
            f"COLLECTED DATA SO FAR:\n{self._format_collected(ctx.collected_data, pk)}",
            f"CONVERSATION HISTORY (latest {min(len(ctx.history), 10)} turns):\n"
            f"{self._format_history(ctx.history, pk)}",
            (
                "Respond naturally as the survey interviewer for this state. "
                "Output ONLY what you say to the respondent — no internal notes, "
                "no JSON, no meta-commentary.\n\n"
                f"{lang_instruction}"
            ),
        ])
        # Clear the one-shot correction flag now that it has been consumed by the prompt
        ctx.correction_applied = False

        expected_output = (
            "A natural, conversational interviewer reply in English."
            if pk == "en"
            else "ردٌّ طبيعي وتفاعلي من المحاور باللغة العربية."
        )

        return Task(
            description=description,
            expected_output=expected_output,
            agent=self._agent,
        )

    # ------------------------------------------------------------------
    # FSM transitions
    # ------------------------------------------------------------------

    def _transition(
        self,
        ctx: ConversationContext,
        user_message: str,
        agent_response: str,
    ) -> None:
        """Evaluate and apply FSM state transition after each turn."""
        state = ctx.state

        if state == ConversationState.GREETING:
            # Always advance to collecting after the greeting exchange
            ctx.state = ConversationState.COLLECTING_INFO

        elif state == ConversationState.COLLECTING_INFO:
            # Capture field order BEFORE extraction so we know what was being asked
            field_order = self._get_field_order(ctx.collected_data)
            current_field = next(
                (f for f in field_order if f not in ctx.collected_data), None
            )

            self._extract_fields(ctx, user_message)

            # Re-compute field order after extraction — status may have just been set,
            # which expands the field order for the correct employment path.
            field_order = self._get_field_order(ctx.collected_data)
            required = self._get_required_fields(ctx.collected_data)

            if current_field and current_field not in ctx.collected_data:
                # User's answer didn't address the field we asked about → clarify
                ctx.clarification_target = current_field
                ctx.state = ConversationState.CLARIFYING
            elif required.issubset(ctx.collected_data.keys()):
                ctx.state = ConversationState.VALIDATING

        elif state == ConversationState.CLARIFYING:
            target = ctx.clarification_target
            keys_before = set(ctx.collected_data)
            self._extract_fields(ctx, user_message)
            answered = (
                (target and target in ctx.collected_data)
                or (not target and set(ctx.collected_data) != keys_before)
            )
            if answered:
                ctx.clarification_count = 0
                ctx.clarification_target = None
                ctx.state = ConversationState.COLLECTING_INFO
            else:
                ctx.clarification_count += 1
                if ctx.clarification_count >= 3 and target and len(user_message.strip()) >= 3:
                    ctx.collected_data[target] = user_message.strip()
                    ctx.clarification_count = 0
                    ctx.clarification_target = None
                    ctx.state = ConversationState.COLLECTING_INFO

        elif state == ConversationState.VALIDATING:
            if self._is_confirmed(user_message, ctx.language):
                ctx.state = ConversationState.COMPLETING
            elif self._wants_correction(user_message, ctx.language):
                # Apply correction: regex-first (instant), LLM fallback for complex cases.
                # Track actual success — previously `correction_applied` was set
                # unconditionally here regardless of whether either extractor found
                # anything to change, so a respondent whose correction wasn't
                # understood (or who hit an LLM timeout/outage) was still told
                # "I've updated that for you" while nothing changed. Confirmed live:
                # with both the Ollama and Anthropic correction backends failing,
                # this produced a silent loop where corrections never took effect.
                ctx.corrected_fields = set()
                ctx.correction_rejected_field = None
                ctx.correction_no_target = False
                regex_ok = self._extract_correction(ctx, user_message)
                llm_ok = False if regex_ok else self._llm_extract_correction(ctx, user_message)
                correction_ok = regex_ok or llm_ok
                # Stay in VALIDATING if all required fields are still satisfied —
                # the response will immediately re-read the corrected summary.
                # Drop to COLLECTING_INFO only if a field is now missing (e.g. the
                # correction changed employment_status and opened a new field path).
                required = self._get_required_fields(ctx.collected_data)
                if correction_ok and not required.issubset(ctx.collected_data.keys()):
                    ctx.state = ConversationState.COLLECTING_INFO
                elif correction_ok:
                    # Remain VALIDATING; signal _build_task to re-read updated summary
                    ctx.correction_applied = True
                elif not ctx.correction_rejected_field:
                    # Bare "no" (or similar) with no field named at all -- distinct
                    # from correction_rejected_field, where a field WAS identified
                    # but its new value couldn't be parsed. Previously this fell
                    # through with no signal set at all, so the reply just re-showed
                    # the identical validation summary with no indication the "no"
                    # was understood -- a real, reproduced dead-end loop (respondent
                    # says "no", gets back the exact same message, with nothing
                    # telling them to name what's wrong).
                    ctx.correction_no_target = True
                # else: correction_rejected_field was set by the extractor -- a
                # field WAS identified but its value failed the sanity check;
                # that already has its own dedicated reply branch.

        # COMPLETING is terminal

    # ------------------------------------------------------------------
    # Heuristic field extraction
    # ------------------------------------------------------------------

    def _extract_fields(self, ctx: ConversationContext, text: str) -> None:
        """
        Extract employment field values from free-text user input using
        keyword heuristics. Stores raw values for downstream processing.

        IMPORTANT: `next_field_before` is captured ONCE at the top — before any
        writes — and used to guard all catch-alls. This prevents a single answer
        from cascading into multiple fields within one call.
        """
        lower = text.lower()
        data = ctx.collected_data
        status = data.get("employment_status", "")

        # Snapshot of next unanswered field BEFORE we mutate data.
        field_order = self._get_field_order(data)
        next_field_before = next(
            (f for f in field_order if f not in data), None
        )

        # ── Employment status ─────────────────────────────────────────────────
        # Order matters: most-specific phrases checked first.
        if "employment_status" not in data:
            if any(w in lower for w in (
                "not in the labour force", "not in labour force",
                "not in labor force", "خارج سوق العمل",
            )):
                data["employment_status"] = "not_in_labour_force"
            elif any(w in lower for w in (
                "unemployed", "looking for work", "عاطل", "أبحث عن عمل",
            )):
                data["employment_status"] = "unemployed"
            elif any(w in lower for w in (
                "employed", "working", "موظف", "أعمل", "أنا أعمل",
            )):
                data["employment_status"] = "employed"
            elif any(w in lower for w in (
                "retired", "student", "housewife", "متقاعد", "طالب", "ربة منزل",
            )):
                data["employment_status"] = "not_in_labour_force"

        # ── Education level (B5) ──────────────────────────────────────────────
        if "education_level" not in data:
            if re.search(r"phd|doctorate|دكتوراه", lower):
                data["education_level"] = "phd"
            elif re.search(r"master|postgrad|ماجستير", lower):
                data["education_level"] = "master"
            elif re.search(r"bachelor|degree|bsc|ba\b|beng|bcom|uni(?:versity)?|college|بكالوريوس|جامعة", lower):
                data["education_level"] = "bachelor"
            elif re.search(r"diploma|دبلوم", lower):
                data["education_level"] = "diploma"
            elif re.search(r"secondary|high.?school|ثانوي|ثانوية", lower):
                data["education_level"] = "secondary"
            elif re.search(r"primary|elementary|ابتدائي", lower):
                data["education_level"] = "primary"
            elif re.search(r"no formal|none|no education|بدون|لا يوجد تعليم", lower):
                data["education_level"] = "no_formal"
            elif next_field_before == "education_level":
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["education_level"] = raw

        # ── Fields only relevant when employment_status is known ─────────────
        # Avoids premature extraction before routing is determined.

        # ── Employment nature (C3) ────────────────────────────────────────────
        if "employment_nature" not in data and status == "employed":
            if re.search(r"employ(?:er|s others)|own.?business.*hire|صاحب.?عمل|لدي موظفون", lower):
                data["employment_nature"] = "employer"
            elif re.search(r"self.?employ|freelanc|own.?account|عمل.?حر|لحسابي.?الخاص", lower):
                data["employment_nature"] = "self_employed"
            elif re.search(r"family.?work|عامل.?عائلي|مساهم.?عائلي", lower):
                data["employment_nature"] = "family_worker"
            elif re.search(r"paid.?employ|employ(?:ee)?|salary|salar|براتب|موظف.?براتب", lower):
                data["employment_nature"] = "paid_employee"
            elif next_field_before == "employment_nature":
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["employment_nature"] = raw

        # ── Employment sector (C4) ────────────────────────────────────────────
        if "employment_sector" not in data and status == "employed":
            if re.search(r"semi.?gov|شبه.?حكومي", lower):
                data["employment_sector"] = "semi_government"
            elif re.search(r"non.?profit|ngo|charity|جمعية|غير.?ربحي", lower):
                data["employment_sector"] = "ngo"
            elif re.search(r"government|ministry|public.?sector|civil.?service|حكومي|وزارة|قطاع.?عام", lower):
                data["employment_sector"] = "government"
            elif re.search(r"private|خاص|قطاع.?خاص", lower):
                data["employment_sector"] = "private"
            elif next_field_before == "employment_sector":
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["employment_sector"] = raw

        # ── Job title (C5) ────────────────────────────────────────────────────
        # Only extract when we are specifically asking for job_title.
        # Unrestricted keyword matching caused field_of_study answers like
        # "Computer Science and Engineering" to be mis-stored as job_title.
        if "job_title" not in data and status in ("employed", "") and next_field_before == "job_title":
            all_keywords = self._PROFESSION_KEYWORDS_EN + self._PROFESSION_KEYWORDS_AR
            raw = text.strip()
            raw_lower = raw.lower()
            if (
                any(kw in lower for kw in all_keywords)
                and raw_lower not in _STATUS_WORDS
            ):
                data["job_title"] = raw
            elif (
                len(raw) >= 3
                and not re.search(r"^\d+$", raw.strip())
                and raw_lower not in _UNCERTAINTY_PHRASES
                and raw_lower not in _STATUS_WORDS
            ):
                data["job_title"] = raw

        # ── Job duties (C5a) ──────────────────────────────────────────────────
        if "job_duties" not in data and next_field_before == "job_duties":
            raw = text.strip()
            if (
                len(raw) >= 3
                and raw.lower() not in _UNCERTAINTY_PHRASES
                and raw.lower() not in _STATUS_WORDS
            ):
                data["job_duties"] = raw

        # ── Industry / sector (C6) ────────────────────────────────────────────
        if "industry" not in data and status in ("employed", ""):
            sector_map: dict[str, list[str]] = {
                "technology": ["software", "tech", "it ", "information technology", "تقنية", "برمجة", "معلومات"],
                "healthcare": ["hospital", "clinic", "health", "medical", "pharmaceutical", "صحة", "مستشفى", "عيادة", "دواء"],
                "education": ["school", "university", "college", "teach", "تعليم", "مدرسة", "جامعة"],
                "construction": ["construction", "build", "infrastructure", "real estate", "بناء", "مقاولات", "تشييد", "عقارات"],
                "retail_trade": ["shop", "store", "retail", "trade", "تجزئة", "تجارة", "محل"],
                "government": ["government", "ministry", "public sector", "civil service",
                               "حكومة", "وزارة", "قطاع عام", "خدمة مدنية"],
                "finance": ["bank", "finance", "insurance", "investment", "بنك", "مالية", "تأمين", "استثمار"],
                "manufacturing": ["factory", "manufactur", "production", "industrial", "مصنع", "إنتاج", "تصنيع"],
                "transportation": ["airline", "aviation", "airport", "transport", "logistics",
                                   "shipping", "cargo", "طيران", "مطار", "نقل", "شحن", "لوجستيك"],
                "hospitality": ["hotel", "restaurant", "tourism", "travel", "catering", "فندق", "سياحة", "مطعم", "ضيافة"],
                "agriculture": ["farm", "agriculture", "food production", "زراعة", "مزرعة"],
                "energy": ["oil", "gas", "energy", "petrol", "electricity", "renewable", "نفط", "غاز", "طاقة", "كهرباء"],
                "telecom": ["telecom", "communication", "network", "اتصالات", "شبكات"],
                "media": ["media", "journalism", "publishing", "advertising", "إعلام", "صحافة", "إعلانات"],
            }
            for sector, keywords in sector_map.items():
                if any(kw in lower for kw in keywords):
                    data["industry"] = sector
                    break
            if "industry" not in data and next_field_before == "industry":
                raw = text.strip()
                if (
                    len(raw) >= 3
                    and raw.lower() not in _UNCERTAINTY_PHRASES
                    and raw.lower() not in _STATUS_WORDS
                ):
                    data["industry"] = raw

        # ── Hours per week (D2) ───────────────────────────────────────────────
        if "hours_per_week" not in data:
            match = re.search(r"(\d+)\s*(?:to|[-–])\s*(\d+)\s*(?:hours?|hrs?|ساعات?|ساعة)", lower)
            if match:
                data["hours_per_week"] = str((int(match.group(1)) + int(match.group(2))) // 2)
            else:
                match = re.search(r"(\d+)\s*(?:hours?|hrs?|ساعات?|ساعة)", lower)
                if match:
                    data["hours_per_week"] = match.group(1)
                elif next_field_before == "hours_per_week":
                    bare = re.search(r"\b(\d{1,2})\b", lower)
                    if bare:
                        val = int(bare.group(1))
                        if 1 <= val <= 99:
                            data["hours_per_week"] = str(val)

        # ── Employment type / work arrangement (D6) ───────────────────────────
        if "employment_type" not in data:
            if re.search(r"full[\s-]?time|permanent|دوام.?كامل|دائم", lower):
                data["employment_type"] = "full_time"
            elif re.search(r"part[\s-]?time|دوام.?جزئي", lower):
                data["employment_type"] = "part_time"
            elif re.search(r"seasonal|موسمي", lower):
                data["employment_type"] = "seasonal"
            elif re.search(r"casual|عَرَضي|عرضي", lower):
                data["employment_type"] = "casual"
            elif re.search(r"self[\s-]?employ|freelanc|عمل.?حر|مستقل", lower):
                data["employment_type"] = "self_employed"
            elif re.search(r"contract(?:or|ing)?|gig|temp(?:orary)?|متعاقد", lower):
                data["employment_type"] = "contractor"

        # ── Monthly wage range (E1) ───────────────────────────────────────────
        if "monthly_wage_range" not in data and status == "employed":
            if re.search(r"prefer.?not|don.?t.?say|rather.?not|لا أريد|أفضل.?عدم", lower):
                data["monthly_wage_range"] = "prefer_not_to_say"
            else:
                # Try to extract a number and bucket it
                wage_m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*k?\b", lower)
                if wage_m:
                    raw_num = wage_m.group(1).replace(",", "")
                    val = float(raw_num)
                    if "k" in lower[wage_m.start():wage_m.end() + 1]:
                        val *= 1000
                    if val < 5000:
                        data["monthly_wage_range"] = "under_5000"
                    elif val <= 10000:
                        data["monthly_wage_range"] = "5000_10000"
                    elif val <= 20000:
                        data["monthly_wage_range"] = "10001_20000"
                    elif val <= 50000:
                        data["monthly_wage_range"] = "20001_50000"
                    else:
                        data["monthly_wage_range"] = "over_50000"
                elif re.search(r"less\s*than\s*5|under\s*5|below\s*5|أقل.?من.?5", lower):
                    data["monthly_wage_range"] = "under_5000"
                elif re.search(r"more\s*than\s*50|over\s*50|above\s*50|أكثر.?من.?50", lower):
                    data["monthly_wage_range"] = "over_50000"
                elif next_field_before == "monthly_wage_range":
                    raw = text.strip()
                    if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                        data["monthly_wage_range"] = raw

        # ── Job search active (F1) ────────────────────────────────────────────
        if "job_search_active" not in data and status == "unemployed":
            if next_field_before == "job_search_active" or re.search(r"search|looking|seeking|أبحث|تبحث", lower):
                if re.search(r"\byes\b|نعم|أبحث|بنشاط|بالفعل", lower):
                    data["job_search_active"] = "yes"
                elif re.search(r"\bno\b|لا\b|لم|لست|لا أبحث", lower):
                    data["job_search_active"] = "no"

        # ── Available for work (F3) ───────────────────────────────────────────
        if "available_for_work" not in data and status == "unemployed":
            if next_field_before == "available_for_work":
                if re.search(r"\byes\b|نعم|متاح|يمكن|أستطيع", lower):
                    data["available_for_work"] = "yes"
                elif re.search(r"\bno\b|لا\b|غير.?متاح|لا.?يمكن", lower):
                    data["available_for_work"] = "no"

        # ── ILO ICLS-19 F3 re-routing ─────────────────────────────────────────
        # Per ILO definition: if not available for work AND not actively searching
        # → reclassify as "not in labour force" (outside LF path).
        # Triggered when both F1 and F3 are resolved.
        if (
            "available_for_work" in data
            and data.get("available_for_work") == "no"
            and data.get("job_search_active") == "no"
            and data.get("employment_status") == "unemployed"
        ):
            data["employment_status"] = "not_in_labour_force"

        # ── Unemployment duration (F4) ────────────────────────────────────────
        if "unemployment_duration" not in data and status == "unemployed":
            dur_m = re.search(r"(\d+\+?)\s*(week|month|year|أسبوع|شهر|سنة|أشهر|أسابيع|سنوات)", lower)
            if dur_m:
                data["unemployment_duration"] = dur_m.group(0)
            elif next_field_before == "unemployment_duration":
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["unemployment_duration"] = raw

        # ── Last job title (G1) ───────────────────────────────────────────────
        if "last_job_title" not in data and status in ("unemployed", "not_in_labour_force"):
            if re.search(r"never.?work|لم.?أعمل|لا.?خبرة.?عمل", lower):
                data["last_job_title"] = "never_worked"
            else:
                all_keywords = self._PROFESSION_KEYWORDS_EN + self._PROFESSION_KEYWORDS_AR
                raw = text.strip()
                if (
                    any(kw in lower for kw in all_keywords)
                    and raw.lower() not in _STATUS_WORDS
                ):
                    data["last_job_title"] = raw
                elif next_field_before == "last_job_title":
                    if (
                        len(raw) >= 3
                        and raw.lower() not in _UNCERTAINTY_PHRASES
                        and raw.lower() not in _STATUS_WORDS
                    ):
                        data["last_job_title"] = raw

        # ── Reason left job (G3) ──────────────────────────────────────────────
        if "reason_left_job" not in data and status == "unemployed":
            if re.search(r"redundant|laid.?off|retrench|فائض|اختزال|تسريح", lower):
                data["reason_left_job"] = "redundant"
            elif re.search(r"resign|quit|left.?voluntar|استقال|استقالة", lower):
                data["reason_left_job"] = "resigned"
            elif re.search(r"business.?clos|compan.?clos|shut.?down|إغلاق|المنشأة.?أغلقت", lower):
                data["reason_left_job"] = "business_closed"
            elif re.search(r"contract.?end|end.?contract|expired|انتهاء.?العقد|انتهى.?العقد", lower):
                data["reason_left_job"] = "contract_ended"
            elif next_field_before == "reason_left_job":
                raw = text.strip()
                if len(raw) >= 3 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["reason_left_job"] = raw

        # ── Outside LF reason (F6) ────────────────────────────────────────────
        if "outside_lf_reason" not in data and status == "not_in_labour_force":
            if re.search(r"retir|متقاعد", lower):
                data["outside_lf_reason"] = "retired"
            elif re.search(r"student|study|studying|طالب|دراسة|أدرس", lower):
                data["outside_lf_reason"] = "student"
            elif re.search(r"homemaker|housewife|househusband|family.?caregiv|ربة.?منزل|مربة.?منزل", lower):
                data["outside_lf_reason"] = "homemaker"
            elif re.search(r"discourag|give.?up|gave.?up|يأس|محبط|يئس", lower):
                data["outside_lf_reason"] = "discouraged"
            elif re.search(r"ill|sick|disab|health.?reason|مريض|إعاقة|عجز|مرض", lower):
                data["outside_lf_reason"] = "illness"
            elif next_field_before == "outside_lf_reason":
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["outside_lf_reason"] = raw

        # ── Gender (B1) ───────────────────────────────────────────────────────────
        if "gender" not in data and next_field_before == "gender":
            if re.search(r"\bmale\b|ذكر", lower):
                data["gender"] = "male"
            elif re.search(r"\bfemale\b|أنثى", lower):
                data["gender"] = "female"
            elif re.search(r"prefer.?not|لا أريد الإفصاح", lower):
                data["gender"] = "prefer_not_to_say"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["gender"] = raw

        # ── Nationality (B3) ──────────────────────────────────────────────────────
        # Store the raw country name. The NationalityClassifier in survey_routes.py
        # maps it to UN M49 / ISO 3166-1 alpha-3 codes.
        if "nationality" not in data and next_field_before == "nationality":
            raw = text.strip()
            if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                data["nationality"] = raw

        # ── Marital status (B4) ───────────────────────────────────────────────────
        if "marital_status" not in data and next_field_before == "marital_status":
            if re.search(r"single|أعزب|أعزبة", lower):
                data["marital_status"] = "single"
            elif re.search(r"married|متزوج", lower):
                data["marital_status"] = "married"
            elif re.search(r"divorced|مطلق", lower):
                data["marital_status"] = "divorced"
            elif re.search(r"widowed|أرمل", lower):
                data["marital_status"] = "widowed"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["marital_status"] = raw

        # ── Emirate (B8) ──────────────────────────────────────────────────────────
        if "emirate" not in data and next_field_before == "emirate":
            if re.search(r"abu.?dhabi|أبوظبي|ابوظبي", lower):
                data["emirate"] = "abu_dhabi"
            elif re.search(r"\bdubai\b|دبي", lower):
                data["emirate"] = "dubai"
            elif re.search(r"sharjah|الشارقة", lower):
                data["emirate"] = "sharjah"
            elif re.search(r"ajman|عجمان", lower):
                data["emirate"] = "ajman"
            elif re.search(r"umm.?al.?quwain|أم القيوين", lower):
                data["emirate"] = "umm_al_quwain"
            elif re.search(r"ras.?al.?khaimah|رأس الخيمة", lower):
                data["emirate"] = "ras_al_khaimah"
            elif re.search(r"fujairah|الفجيرة", lower):
                data["emirate"] = "fujairah"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["emirate"] = raw

        # ── UAE residence duration (B9) ───────────────────────────────────────────
        if "uae_residence_duration" not in data and next_field_before == "uae_residence_duration":
            if re.search(r"born.?(?:in|here)|مولود", lower):
                data["uae_residence_duration"] = "born_in_uae"
            elif re.search(r"less.?than.?1.?year|أقل.?من.?سنة|less.?than.?one", lower):
                data["uae_residence_duration"] = "less_than_1_year"
            elif re.search(r"20\+|twenty|عشرون|20.?سنة.?فأكثر", lower):
                data["uae_residence_duration"] = "20_plus"
            elif re.search(r"10.?(?:to|[-\u2013])\s*19|عشر|10-19", lower):
                data["uae_residence_duration"] = "10_to_19"
            elif re.search(r"5.?(?:to|[-\u2013])\s*9|خمس|5-9", lower):
                data["uae_residence_duration"] = "5_to_9"
            elif re.search(r"1.?(?:to|[-\u2013])\s*4|1-4", lower):
                data["uae_residence_duration"] = "1_to_4"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["uae_residence_duration"] = raw

        # ── Vocational training (B7) ──────────────────────────────────────────────
        if "vocational_training" not in data and next_field_before == "vocational_training":
            if re.search(r"\byes\b|نعم", lower):
                data["vocational_training"] = "yes"
            elif re.search(r"\bno\b|لا\b", lower):
                data["vocational_training"] = "no"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["vocational_training"] = raw

        # ── Actual hours worked (D1) ──────────────────────────────────────────────
        if "actual_hours_worked" not in data and status == "employed":
            if next_field_before == "actual_hours_worked":
                match = re.search(r"\b(\d{1,3})\b", lower)
                if match:
                    val = int(match.group(1))
                    if 0 <= val <= 168:
                        data["actual_hours_worked"] = str(val)
                elif text.strip().lower() not in _UNCERTAINTY_PHRASES:
                    data["actual_hours_worked"] = text.strip()

        # ── Secondary job (D3) ────────────────────────────────────────────────────
        if "secondary_job" not in data and status == "employed" and next_field_before == "secondary_job":
            if re.search(r"\byes\b|نعم|have.?another|additional.?job|وظيفة.?إضافية", lower):
                data["secondary_job"] = "yes"
            elif re.search(r"\bno\b|لا\b|no.?other|don.?t.?have.?another", lower):
                data["secondary_job"] = "no"

        # ── Underemployment (D5) ──────────────────────────────────────────────────
        if "underemployment" not in data and status == "employed" and next_field_before == "underemployment":
            if re.search(r"overemploy|too.?many.?hours|أعمل.?أكثر", lower):
                data["underemployment"] = "overemployed"
            elif re.search(r"\byes\b|نعم|want.?more|more.?hours|أريد.?أكثر", lower):
                data["underemployment"] = "yes_want_more"
            elif re.search(r"\bno\b|لا\b|satisfied|content|مكتفٍ", lower):
                data["underemployment"] = "no"

        # ── Contract type (D7) ────────────────────────────────────────────────────
        if "contract_type" not in data and status == "employed" and next_field_before == "contract_type":
            if re.search(r"permanent|دائم", lower):
                data["contract_type"] = "permanent"
            elif re.search(r"probation|تجريبي", lower):
                data["contract_type"] = "probation"
            elif re.search(r"no.?written|بدون.?عقد", lower):
                data["contract_type"] = "no_written_contract"
            elif re.search(r"1.?to.?3|1-3.?year|سنة.?إلى.?3", lower):
                data["contract_type"] = "fixed_1_3_years"
            elif re.search(r"fixed|temporary|limited|محدد.?المدة|مؤقت", lower):
                data["contract_type"] = "fixed_less_1_year"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["contract_type"] = raw

        # ── Remote work (D8) ──────────────────────────────────────────────────────
        if "remote_work" not in data and status == "employed" and next_field_before == "remote_work":
            if re.search(r"\balways\b|fully.?remote|دائمًا|عن.?بُعد.?دائمًا", lower):
                data["remote_work"] = "always"
            elif re.search(r"\bmostly\b|mainly.?remote|في.?الغالب", lower):
                data["remote_work"] = "mostly"
            elif re.search(r"partial|hybrid|جزئيًا", lower):
                data["remote_work"] = "partially"
            elif re.search(r"\bnever\b|on.?site|office|أبدًا|في.?المكتب", lower):
                data["remote_work"] = "never"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["remote_work"] = raw

        # ── Salary allowances (E2) ────────────────────────────────────────────────
        if "salary_allowances" not in data and status == "employed" and next_field_before == "salary_allowances":
            raw = text.strip()
            if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                data["salary_allowances"] = raw

        # ── Health insurance (E4) ─────────────────────────────────────────────────
        if "health_insurance" not in data and status == "employed" and next_field_before == "health_insurance":
            if re.search(r"full|complete|كاملة", lower):
                data["health_insurance"] = "full"
            elif re.search(r"partial|جزئية", lower):
                data["health_insurance"] = "partial"
            elif re.search(r"self.?pay|pay.?for.?own|my.?own|أدفع.?من.?راتبي", lower):
                data["health_insurance"] = "self_paid"
            elif re.search(r"\bno\b|لا\b|don.?t.?have|none|لا.?يوجد", lower):
                data["health_insurance"] = "no"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["health_insurance"] = raw

        # ── Job search methods (F2) ───────────────────────────────────────────────
        if "job_search_methods" not in data and status == "unemployed" and next_field_before == "job_search_methods":
            raw = text.strip()
            if len(raw) >= 3 and raw.lower() not in _UNCERTAINTY_PHRASES:
                data["job_search_methods"] = raw

        # ── Desired job type (F5) ─────────────────────────────────────────────────
        if "desired_job_type" not in data and status == "unemployed" and next_field_before == "desired_job_type":
            if re.search(r"same|previous|نفس.?المهنة|نفس.?السابق", lower):
                data["desired_job_type"] = "same_as_previous"
            elif re.search(r"first.?job|never.?worked|أول.?وظيفة", lower):
                data["desired_job_type"] = "first_job"
            elif re.search(r"different|change|other|مختلفة|تغيير", lower):
                data["desired_job_type"] = "different"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["desired_job_type"] = raw

        # ── Ever worked (F7) ──────────────────────────────────────────────────────
        if "ever_worked" not in data and status in ("unemployed", "not_in_labour_force") and next_field_before == "ever_worked":
            if re.search(r"never.?work|لم.?أعمل.?أبدًا|never.?had.?a.?job", lower):
                data["ever_worked"] = "never_worked"
            elif re.search(r"outside.?uae|abroad|outside.?emirates|خارج.?الإمارات", lower):
                data["ever_worked"] = "yes_outside_uae"
            elif re.search(r"\byes\b|in.?uae|في.?الإمارات|worked.?here|نعم", lower):
                data["ever_worked"] = "yes_in_uae"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["ever_worked"] = raw

        # ── Main skills (H1) ──────────────────────────────────────────────────────
        if "main_skills" not in data and next_field_before == "main_skills":
            raw = text.strip()
            if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                data["main_skills"] = raw

        # ── Qualification match (H2) ──────────────────────────────────────────────
        if "qualification_match" not in data and status == "employed" and next_field_before == "qualification_match":
            if re.search(r"over.?qualif|too.?qualif|مؤهل.?أكثر", lower):
                data["qualification_match"] = "overqualified"
            elif re.search(r"under.?qualif|not.?enough.?qualif|مؤهل.?أقل", lower):
                data["qualification_match"] = "underqualified"
            elif re.search(r"well.?match|perfect.?fit|good.?fit|مطابق|مناسب", lower):
                data["qualification_match"] = "well_matched"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["qualification_match"] = raw

        # ── Training participation (H3) ───────────────────────────────────────────
        if "training_participation" not in data and next_field_before == "training_participation":
            if re.search(r"employer.?fund|company.?fund|employer.?paid|ممول.?من.?صاحب", lower):
                data["training_participation"] = "yes_employer_funded"
            elif re.search(r"government|govt|حكومي|برنامج.?حكومي", lower):
                data["training_participation"] = "yes_govt_program"
            elif re.search(r"self.?fund|own.?expense|ممول.?ذاتيًا|من.?مالي", lower):
                data["training_participation"] = "yes_self_funded"
            elif re.search(r"\byes\b|نعم|completed.?training|attended", lower):
                data["training_participation"] = "yes_self_funded"
            elif re.search(r"\bno\b|لا\b|didn.?t|لم.?أشارك", lower):
                data["training_participation"] = "no"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["training_participation"] = raw

        # ── Labour market barriers (H5) ───────────────────────────────────────────
        if "labour_market_barriers" not in data and next_field_before == "labour_market_barriers":
            raw = text.strip()
            if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                data["labour_market_barriers"] = raw

        # ── Platform work (I1) ────────────────────────────────────────────────────
        if "platform_work" not in data and next_field_before == "platform_work":
            if re.search(r"primary|main.?income|دخل.?رئيسي", lower):
                data["platform_work"] = "yes_primary"
            elif re.search(r"supplement|additional|extra.?income|دخل.?إضافي", lower):
                data["platform_work"] = "yes_supplementary"
            elif re.search(r"\byes\b|نعم|do.?work.?through|من.?خلال.?منصة", lower):
                data["platform_work"] = "yes_supplementary"
            elif re.search(r"\bno\b|لا\b|don.?t.?use|لا.?أعمل.?من.?خلال", lower):
                data["platform_work"] = "no"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["platform_work"] = raw

        # ── Online business (I4) ──────────────────────────────────────────────────
        if "online_business" not in data and next_field_before == "online_business":
            if re.search(r"registered|formal|مسجل", lower):
                data["online_business"] = "yes_registered"
            elif re.search(r"informal|unofficial|غير.?رسمي", lower):
                data["online_business"] = "yes_informal"
            elif re.search(r"\byes\b|نعم|have.?online|own.?online|لدي", lower):
                data["online_business"] = "yes_informal"
            elif re.search(r"\bno\b|لا\b|don.?t.?have|ليس.?لدي", lower):
                data["online_business"] = "no"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["online_business"] = raw

        # ── Job satisfaction (J1) ─────────────────────────────────────────────────
        if "job_satisfaction" not in data and status == "employed" and next_field_before == "job_satisfaction":
            if re.search(r"very.?satisf|very.?happy|راضٍ.?جداً", lower):
                data["job_satisfaction"] = "5"
            elif re.search(r"\b4\b|satisf|happy|راضٍ|سعيد", lower):
                data["job_satisfaction"] = "4"
            elif re.search(r"\b3\b|neutral|okay|ok|محايد|مقبول", lower):
                data["job_satisfaction"] = "3"
            elif re.search(r"\b2\b|dissatisf|unhappy|غير.?راضٍ", lower):
                data["job_satisfaction"] = "2"
            elif re.search(r"very.?dissatisf|very.?unhappy|غير.?راضٍ.?جداً", lower):
                data["job_satisfaction"] = "1"
            else:
                bare = re.search(r"\b([1-5])\b", lower)
                if bare:
                    data["job_satisfaction"] = bare.group(1)
                elif text.strip() and text.strip().lower() not in _UNCERTAINTY_PHRASES:
                    data["job_satisfaction"] = text.strip()

        # ── Work-life balance (J4) ────────────────────────────────────────────────
        if "work_life_balance" not in data and status == "employed" and next_field_before == "work_life_balance":
            if re.search(r"somewhat|sort.?of|kind.?of|not.?always|نوعًا.?ما|أحيانًا", lower):
                data["work_life_balance"] = "somewhat"
            elif re.search(r"\byes\b|نعم|good|great|fine|نعم.?لدي", lower):
                data["work_life_balance"] = "yes"
            elif re.search(r"\bno\b|لا\b|poor|bad|difficult|لا.?يوجد", lower):
                data["work_life_balance"] = "no"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["work_life_balance"] = raw

        # ── Question clarity (K1) ─────────────────────────────────────────────────
        if "question_clarity" not in data and next_field_before == "question_clarity":
            bare = re.search(r"\b([1-5])\b", lower)
            if bare:
                data["question_clarity"] = bare.group(1)
            elif re.search(r"very.?clear|perfect|واضح.?جداً", lower):
                data["question_clarity"] = "5"
            elif re.search(r"unclear|confus|غير.?واضح", lower):
                data["question_clarity"] = "2"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["question_clarity"] = raw

        # ── Field of study (B6) — conditional on education >= bachelor ───────────
        if "field_of_study" not in data and next_field_before == "field_of_study":
            raw = text.strip()
            if len(raw) >= 3 and raw.lower() not in _UNCERTAINTY_PHRASES:
                data["field_of_study"] = raw

        # ── Secondary job hours (D4) — conditional on secondary_job = yes ────────
        if "secondary_job_hours" not in data and status == "employed" and next_field_before == "secondary_job_hours":
            match = re.search(r"\b(\d{1,3})\b", lower)
            if match:
                val = int(match.group(1))
                if 0 <= val <= 99:
                    data["secondary_job_hours"] = str(val)
            elif text.strip() and text.strip().lower() not in _UNCERTAINTY_PHRASES:
                data["secondary_job_hours"] = text.strip()

        # ── Bonuses (E3) ──────────────────────────────────────────────────────────
        if "bonuses" not in data and status == "employed" and next_field_before == "bonuses":
            if re.search(r"annual|سنوي|year.?end|end.?of.?year", lower):
                data["bonuses"] = "yes_annual"
            elif re.search(r"performance|أداء|kpi|target", lower):
                data["bonuses"] = "yes_performance"
            elif re.search(r"\bno\b|لا\b|none|لا.?مكافآت|لم.?أتلقَّ", lower):
                data["bonuses"] = "no"
            elif re.search(r"\byes\b|نعم|received|تلقيت", lower):
                data["bonuses"] = "yes_other"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["bonuses"] = raw

        # ── Pension scheme (E5) ───────────────────────────────────────────────────
        if "pension_scheme" not in data and status == "employed" and next_field_before == "pension_scheme":
            if re.search(r"gpssa|هيئة.?المعاشات|معاشات.?إماراتي", lower):
                data["pension_scheme"] = "gpssa"
            elif re.search(r"difc|adgm", lower):
                data["pension_scheme"] = "difc_adgm"
            elif re.search(r"private|خاصة|employer.?scheme|خطة.?صاحب", lower):
                data["pension_scheme"] = "private_scheme"
            elif re.search(r"not.?sure|غير.?متأكد", lower):
                data["pension_scheme"] = "not_sure"
            elif re.search(r"\bno\b|لا\b|not.?enrolled|غير.?مشترك", lower):
                data["pension_scheme"] = "no"
            elif re.search(r"\byes\b|نعم", lower):
                data["pension_scheme"] = "yes_other"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["pension_scheme"] = raw

        # ── Emiratization program (H4) — UAE nationals only ───────────────────────
        if "emiratization_program" not in data and next_field_before == "emiratization_program":
            if re.search(r"nafis|نافس", lower):
                data["emiratization_program"] = "nafis"
            elif re.search(r"tawteen|تو.?طين|absher|أبشر|other.?gov|حكومي.?آخر", lower):
                data["emiratization_program"] = "other_govt"
            elif re.search(r"\bno\b|لا\b|not.?applic|not.?register|غير.?مسجل|لا.?ينطبق", lower):
                data["emiratization_program"] = "no"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["emiratization_program"] = raw

        # ── Platform names (I2) — conditional on platform_work ≠ no ─────────────
        if "platform_names" not in data and next_field_before == "platform_names":
            raw = text.strip()
            if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                data["platform_names"] = raw

        # ── Platform hours (I3) — conditional on platform_work ≠ no ─────────────
        if "platform_hours" not in data and next_field_before == "platform_hours":
            match = re.search(r"\b(\d{1,3})\b", lower)
            if match:
                val = int(match.group(1))
                if 0 <= val <= 99:
                    data["platform_hours"] = str(val)
            elif text.strip() and text.strip().lower() not in _UNCERTAINTY_PHRASES:
                data["platform_hours"] = text.strip()

        # ── Last job sector (G2) ──────────────────────────────────────────────────
        if "last_job_sector" not in data and status in ("unemployed", "not_in_labour_force") and next_field_before == "last_job_sector":
            if re.search(r"semi.?gov|شبه.?حكومي", lower):
                data["last_job_sector"] = "semi_government"
            elif re.search(r"non.?profit|ngo|غير.?ربحي", lower):
                data["last_job_sector"] = "ngo"
            elif re.search(r"government|ministry|حكومي|وزارة", lower):
                data["last_job_sector"] = "government"
            elif re.search(r"self.?employ|freelanc|عمل.?حر", lower):
                data["last_job_sector"] = "self_employed"
            elif re.search(r"private|خاص", lower):
                data["last_job_sector"] = "private"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["last_job_sector"] = raw

        # ── Highest previous salary (G4) ──────────────────────────────────────────
        if "highest_previous_salary" not in data and status in ("unemployed", "not_in_labour_force") and next_field_before == "highest_previous_salary":
            if re.search(r"prefer.?not|don.?t.?say|أفضل.?عدم|لا.?أريد", lower):
                data["highest_previous_salary"] = "prefer_not_to_say"
            else:
                wage_m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*k?\b", lower)
                if wage_m:
                    raw_num = wage_m.group(1).replace(",", "")
                    val = float(raw_num)
                    if "k" in lower[wage_m.start():wage_m.end() + 1]:
                        val *= 1000
                    if val < 5000:
                        data["highest_previous_salary"] = "under_5000"
                    elif val <= 10000:
                        data["highest_previous_salary"] = "5000_10000"
                    elif val <= 20000:
                        data["highest_previous_salary"] = "10001_20000"
                    elif val <= 50000:
                        data["highest_previous_salary"] = "20001_50000"
                    else:
                        data["highest_previous_salary"] = "over_50000"
                elif next_field_before == "highest_previous_salary":
                    raw = text.strip()
                    if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                        data["highest_previous_salary"] = raw

        # ── Work safety (J2) ──────────────────────────────────────────────────────
        if "work_safety" not in data and status == "employed" and next_field_before == "work_safety":
            if re.search(r"\balways\b|دائمًا|very.?safe|آمن.?جداً", lower):
                data["work_safety"] = "always"
            elif re.search(r"\bmostly\b|في.?الغالب|generally.?safe", lower):
                data["work_safety"] = "mostly"
            elif re.search(r"\bsometimes\b|أحيانًا|partially", lower):
                data["work_safety"] = "sometimes"
            elif re.search(r"\brarely\b|نادرًا|not.?often", lower):
                data["work_safety"] = "rarely"
            elif re.search(r"\bnever\b|أبدًا|not.?safe|unsafe|غير.?آمن", lower):
                data["work_safety"] = "never"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["work_safety"] = raw

        # ── Workplace issues (J3) ─────────────────────────────────────────────────
        if "workplace_issues" not in data and status == "employed" and next_field_before == "workplace_issues":
            raw = text.strip()
            if re.search(r"\bnone\b|لا.?شيء|nothing|no.?issues|لم.?أتعرض", lower):
                data["workplace_issues"] = "none"
            elif len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                data["workplace_issues"] = raw

        # ── Difficulty answering (K2) ─────────────────────────────────────────────
        if "difficulty_answering" not in data and next_field_before == "difficulty_answering":
            if re.search(r"\bno\b|لا\b|none|no.?difficulty|لم.?أواجه", lower):
                data["difficulty_answering"] = "no"
            elif re.search(r"\byes\b|نعم|had.?difficulty|واجهت", lower):
                data["difficulty_answering"] = text.strip()
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["difficulty_answering"] = raw

        # ── Survey comments (K5) — optional, always accept ───────────────────────
        if "survey_comments" not in data and next_field_before == "survey_comments":
            raw = text.strip()
            if len(raw) >= 2:
                data["survey_comments"] = raw

        # ── AI preference (K3) ────────────────────────────────────────────────
        if "ai_preference" not in data and next_field_before == "ai_preference":
            if re.search(r"prefer.?ai|ai.?prefer|like.?ai|prefer.?this|prefer.?digital|أفضل.?الذكاء|ذكاء.?اصطناعي", lower):
                data["ai_preference"] = "prefer_ai"
            elif re.search(r"prefer.?human|human.?prefer|prefer.?person|أفضل.?المحاور.?البشري|بشري", lower):
                data["ai_preference"] = "prefer_human"
            elif re.search(r"no.?prefer|no.?diff|doesn.?t.?matter|either|لا.?فرق|سواء|محايد", lower):
                data["ai_preference"] = "no_preference"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["ai_preference"] = raw

        # ── Data confidence (K4) ──────────────────────────────────────────────
        if "data_confidence" not in data and next_field_before == "data_confidence":
            if re.search(r"very.?conf|fully.?conf|definitely|complete.?conf|واثق.?جدًا|ثقة.?كاملة|تمامًا", lower):
                data["data_confidence"] = "very_confident"
            elif re.search(r"somewhat|fairly|sort.?of|kind.?of|مقبول|واثق.?نسبيًا|نسبيًا|إلى.?حد.?ما", lower):
                data["data_confidence"] = "somewhat_confident"
            elif re.search(r"not.?conf|don.?t.?trust|not.?sure|غير.?واثق|لا.?ثقة|لست.?متأكد", lower):
                data["data_confidence"] = "not_confident"
            else:
                raw = text.strip()
                if len(raw) >= 2 and raw.lower() not in _UNCERTAINTY_PHRASES:
                    data["data_confidence"] = raw

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_ambiguous(self, text: str) -> bool:
        """Return True if the user input is too short or too vague to extract data from."""
        stripped = text.strip()
        if len(stripped) < 5:
            return True
        vague = {"yes", "no", "ok", "okay", "sure", "fine", "نعم", "لا", "حسنًا", "موافق", "طيب"}
        return stripped.lower() in vague

    @staticmethod
    def _is_confirmed(text: str, language: str) -> bool:
        """Return True if the text expresses confirmation of the validation summary."""
        lower = text.lower().strip()
        confirmations = _CONFIRMATIONS.get(language, _CONFIRMATIONS["en"])
        return any(
            re.search(r"\b" + re.escape(c) + r"\b", lower)
            for c in confirmations
        )

    @staticmethod
    def _wants_correction(text: str, language: str) -> bool:
        """Return True if the text indicates the respondent wants to correct something."""
        lower = text.lower().strip()
        corrections = _CORRECTIONS.get(language, _CORRECTIONS["en"])
        return any(
            re.search(r"\b" + re.escape(c) + r"\b", lower)
            for c in corrections
        )

    @staticmethod
    def correction_schema_for(collected_data: dict) -> tuple[set[str], str, dict]:
        """
        Shared by _llm_extract_correction and main.py's startup pre-warm
        (added 2026-09-02, refactored out at the same time so both use the
        exact same logic rather than a duplicated copy): given a
        respondent's collected_data, returns (valid_fields, schema_block,
        correction_schema) -- the path-restricted field set (see
        _llm_extract_correction's docstring for why job_title-style
        cross-path leakage and employment_status are excluded), the
        human-readable schema text embedded in the LLM prompt, and the
        JSON-Schema dict passed to Ollama for grammar-constrained decoding.
        """
        path_fields = set(ConversationManager._get_field_order(collected_data))
        valid_fields = set(_CORRECTION_FIELD_SCHEMA.keys()) & path_fields
        valid_fields.discard("employment_status")

        schema_block = "\n".join(
            f"  {k} ({v['label']}): {v['values']}"
            for k, v in _CORRECTION_FIELD_SCHEMA.items()
            if k in valid_fields
        )
        correction_schema = {
            "type": "object",
            "properties": {
                "corrections": {
                    "type": "array",
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string", "enum": sorted(valid_fields)},
                            "value": {"type": "string"},
                        },
                        "required": ["field", "value"],
                    },
                },
            },
            "required": ["corrections"],
        }
        return valid_fields, schema_block, correction_schema

    def _llm_extract_correction(self, ctx: "ConversationContext", text: str) -> bool:
        """Parse any free-text correction and overwrite collected_data via direct LLM call.

        Uses the Ollama REST API directly (no CrewAI overhead) with format=json to
        guarantee structured output and a hard 100-second timeout (covers a
        real measured cold-start worst case of ~78s against the actual
        default model -- see _CORRECTION_TIMEOUT's own comment).  Falls back to
        Anthropic API if Ollama is unreachable.  Returns True if ≥1 field updated.
        Handles all 60+ fields, all 5 languages, implicit + multi-field corrections.

        `valid_fields` is restricted to the respondent's actual field path (via
        _get_field_order), not the full 56-field schema -- fixed 2026-09-02
        after a real, reproduced bug: an unemployed respondent's correction
        naming "Most Recent Job Title" (last_job_title, a real field on the
        unemployed path) was written to "job_title" instead -- a field that
        only exists on the EMPLOYED path and should never appear for this
        respondent at all. Restricting both the prompt's shown schema and the
        post-parse validation to the real path prevents the LLM from ever
        having "job_title" as an option to confuse with "last_job_title" for
        an unemployed respondent, rather than just making the mix-up less
        likely.
        """
        # employment_status is deliberately excluded from the LLM's
        # guessable set by correction_schema_for() -- a real, reproduced
        # misfire on this exact garbled input had the LLM silently flip
        # employment_status "unemployed" -> "employed" (the message
        # mentions "working", which is enough for a small local model to
        # misread as an employment-status claim) while making zero attempt
        # on the field actually named. This is the single highest-
        # consequence field in the whole schema -- it redetermines the
        # respondent's entire remaining question path -- so it gets a
        # stricter bar than "the LLM guessed it": _extract_correction's
        # regex/alias path (which already runs first, unconditionally,
        # before this method is even called) is the only way
        # employment_status gets corrected from here on. An explicit
        # "change employment status to employed" still works exactly as
        # before; an indirect/ambiguous mention no longer risks a silent,
        # structurally significant misfire.
        valid_fields, schema_block, correction_schema = self.correction_schema_for(ctx.collected_data)

        current_data = json.dumps(
            {k: v for k, v in ctx.collected_data.items() if v},
            ensure_ascii=False,
        )

        prompt = (
            "You are a data-correction assistant for a UAE Labour Force Survey.\n"
            "The respondent has reviewed their survey summary and wants to correct answers.\n"
            "The message may be in English, Arabic, Urdu, Hindi, Filipino, or mixed.\n\n"
            f"CURRENT DATA:\n{current_data}\n\n"
            f"CORRECTION MESSAGE:\n\"{text}\"\n\n"
            "FIELD SCHEMA (key | label | accepted values):\n"
            f"{schema_block}\n\n"
            "RULES:\n"
            "- Identify every field being corrected, explicit OR implicit.\n"
            "  Examples: 'I am from India' → nationality=Indian, "
            "'actually part-time' → employment_type=part_time, "
            "'private sector' → employment_sector=private, "
            "'أنا هندي' → nationality=Indian, "
            "'Most Recent Job Title should be Statistician' → last_job_title=Statistician "
            "(the respondent is naming the FIELD SCHEMA's label text, e.g. 'Last Job Title', "
            "not typing the exact key -- match it to the field whose label they mean, "
            "don't skip it just because they didn't say the literal key).\n"
            "- For enum fields use the exact accepted value (snake_case).\n"
            "- For free-text fields (job_title, nationality, industry…) use the respondent's words.\n"
            # An explicit script-preservation instruction was tried here
            # (2026-09-02) and reverted the same day: it measurably regressed
            # Hindi/Urdu reliability with this specific small local model --
            # reproduced 2/2 confidently WRONG English words for Hindi
            # ("Sociologist" instead of "Statistician") and 2/2 silent no-ops
            # for Urdu, neither of which happened with this plainer
            # instruction (which correctly extracted "Statistician" for both
            # languages in the original, pre-change test). A correct English
            # translation the model can actually produce reliably beats an
            # aspirational instruction it can't follow consistently.
            "- CRITICAL: only include fields the correction message actually changes. "
            "Do NOT re-list fields from CURRENT DATA that are staying the same, even to "
            "confirm they're correct -- a real, reproduced bug (2026-09-02) is the model "
            "echoing back every field in CURRENT DATA as if all of them were being "
            "corrected, which produces oversized output that gets cut off mid-JSON. "
            "A single correction message usually changes 1-2 fields; correcting most or "
            "all fields at once is a sign the output is wrong, not that the respondent "
            "meant to.\n"
            "- Output ONLY valid JSON, no extra text:\n"
            '{"corrections":[{"field":"<key>","value":"<value>"}]}\n'
            "- If nothing is clearly corrected: {\"corrections\":[]}"
        )

        # correction_schema (grammar-constrained: `field` can only be one of
        # this respondent's actual valid_fields, array capped at 5 items) was
        # already built above by correction_schema_for() -- see that
        # method's docstring for why.
        #
        # CORRECTION_LLM_PROVIDER, added 2026-09-04 for the cloud-hosting
        # migration: defaults to "ollama" so local dev on the laptop is
        # completely unchanged. The hosted deployment (no local Ollama
        # available at all) sets CORRECTION_LLM_PROVIDER=groq. Both
        # _call_ollama_json and _call_groq_json return the same raw JSON
        # string, so every line of parsing below this point is identical
        # regardless of provider.
        provider = os.getenv("CORRECTION_LLM_PROVIDER", "ollama").strip().lower()
        if provider == "groq":
            raw = self._call_groq_json(prompt, schema=correction_schema) or self._call_anthropic_json(prompt)
        else:
            raw = self._call_ollama_json(prompt, schema=correction_schema) or self._call_anthropic_json(prompt)
        if not raw:
            return False

        try:
            # Strip markdown fences if present
            clean = raw.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```[a-z]*\n?", "", clean, flags=re.IGNORECASE)
                clean = clean.rstrip("`").strip()
            try:
                data = json.loads(clean)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", clean, re.DOTALL)
                data = json.loads(m.group()) if m else {}

            corrections = data.get("corrections", [])
            updated = False
            for c in corrections:
                f_key = (c.get("field") or "").strip()
                f_val = (c.get("value") or "").strip()
                if not f_key or not f_val or f_key not in valid_fields:
                    continue
                f_val = self._canonicalize_correction_value(f_key, f_val)
                cleaned = _sanity_check_correction_value(f_key, f_val)
                if cleaned is None:
                    ctx.correction_rejected_field = f_key
                    _logger.info("Correction rejected (failed sanity check): %s = %r", f_key, f_val)
                    continue
                ctx.collected_data[f_key] = cleaned
                ctx.corrected_fields.add(f_key)
                _logger.info("Correction applied: %s = %r", f_key, cleaned)
                updated = True
            return updated
        except Exception as exc:
            _logger.warning("_llm_extract_correction parse error: %s | raw=%r", exc, raw[:200])
            return False

    # ── Low-level LLM helpers (bypass CrewAI for speed-critical calls) ─────────

    # History of this constant, most recent correction first:
    #
    # 2026-09-02: the two factual claims below this line are stale as of
    # today's changes and are being corrected here rather than rewritten
    # from scratch, per this project's own "verify before restating"
    # discipline. (1) The prompt no longer embeds the full ~60-field
    # schema -- schema_block in _llm_extract_correction is now filtered to
    # `valid_fields` (the respondent's real field-order path, typically
    # ~15-25 fields depending on employed/unemployed/OLF path), so the real
    # prompt is meaningfully smaller than "~5,200 chars" for most
    # respondents. (2) This call no longer uses the shared OLLAMA_MODEL --
    # it has its own OLLAMA_CORRECTION_MODEL default (qwen2.5:3b), which is
    # NOT the "bare llama3.2, 3B" the 78s/4.7-6.1s figures below were
    # measured against. Real, repeated timings observed today against the
    # new qwen2.5:3b default + path-filtered schema ranged roughly 40-60s
    # per call across English/Arabic/Urdu/Hindi/Tagalog test corrections
    # (not re-measured as a formal cold/warm split like the entry below) --
    # comfortably inside the existing 100s budget, so the timeout value
    # itself is left unchanged; only the claims about what it was measured
    # against are corrected.
    #
    # 2026-08-19 entry (model-identity correction, now itself partly
    # superseded by the note above): this constant's comment previously
    # claimed it was measured against "llama3.2:1b", but the actual shipped
    # default at the time -- both llm_client.py's code fallback and
    # .env.example -- was bare "llama3.2", resolving to the 3B :latest
    # model (2.0GB), not the 1B model (1.3GB) the old 45s figure was tuned
    # against. Re-measured directly against that real default (bare
    # "llama3.2", 3B): 5 real sequential calls with that era's full-schema
    # prompt -- cold start (model not yet resident in Ollama) took 78.0s;
    # once warm, calls took 4.7-6.1s. The 45s figure was failing
    # consistently because it only ever budgeted for a warm call, not a
    # cold start. 100s applied a ~29% headroom margin over that measured
    # 78s cold-start worst case. Setting Ollama's keep_alive on this call to
    # keep a model resident between requests would reduce how often the
    # cold-start path is hit at all -- still a separate, unimplemented
    # improvement, not part of any fix so far.
    _CORRECTION_TIMEOUT = 100  # seconds

    def _call_ollama_json(self, prompt: str, schema: dict | None = None) -> str | None:
        """POST to Ollama /api/chat with format=json (or a constrained JSON Schema).
        Returns raw response string or None.

        `schema`, added 2026-09-02: Ollama >=0.5 supports passing a real JSON
        Schema as `format` instead of the bare string "json" -- the server
        does grammar-constrained decoding against it, so the model is
        structurally unable to emit tokens that violate the schema (e.g. an
        enum'd field name outside the allowed set, or more array items than
        maxItems permits). This installation is 0.33.2, well past that
        floor (confirmed via GET /api/version). Prompt instructions alone
        were tried first for the over-long-response bug this fixes and
        proved unreliable against this small local model -- constraining the
        grammar itself doesn't depend on the model choosing to comply.
        Optional and additive: omitted, this call behaves exactly as before
        (loose "format": "json").
        """
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # OLLAMA_CORRECTION_MODEL, added 2026-09-02: this call's own model
        # choice, deliberately separate from the shared OLLAMA_MODEL used by
        # llm_client.py's general TaskType routing (that default is left
        # alone -- Module J's own finding was explicitly "a measurement, not
        # a decision" for the shared routing). This one narrow call is
        # different: three repeated, reproduced tests against the exact
        # failing correction from this session's bug report showed
        # llama3.2 (the OLLAMA_MODEL default) never once correctly mapped
        # "Most Recent Job Title" to last_job_title -- either silently
        # flipping employment_status from a misread, or echoing 5 unrelated
        # unchanged fields as noise. The same 3 runs against qwen2.5:3b
        # (already installed locally, already the model Module J's own
        # ablation found consistently stronger, and the model this project's
        # corrective-RAG-retry work already uses successfully for a similarly
        # structured extraction task) got the correct field and value clean,
        # 3/3. Broadened the same day: re-tested across all 5 supported
        # languages (en/ar/ur/hi/tl), 2 real end-to-end runs each with
        # qwen2.5:3b as this call's model -- 10/10 correctly identified
        # last_job_title with the correct value and zero wrong-field
        # injection in any language. Deliberately does NOT fall through to
        # OLLAMA_MODEL if unset --
        # this repo's real .env pins OLLAMA_MODEL=llama3.2:1b (confirmed by
        # reading it directly), the exact model that failed the reproduction
        # above, so chaining through it would have silently defeated this
        # fix in the real deployment. Defaults straight to qwen2.5:3b
        # instead; still overridable via OLLAMA_CORRECTION_MODEL for an
        # environment that hasn't pulled it.
        model = os.getenv("OLLAMA_CORRECTION_MODEL", "qwen2.5:3b")
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": schema if schema is not None else "json",
            # num_predict raised 512 -> 1024, 2026-09-02: 512 was tight enough
            # that a real (buggy) full-dataset echo truncated mid-JSON rather
            # than failing cleanly. Schema-constrained calls (maxItems-capped)
            # shouldn't need this much headroom, but it's kept as a shared
            # safety margin for any future non-schema caller too.
            "options": {"temperature": 0.0, "num_predict": 1024},
            # keep_alive, added 2026-09-02, raised 30m -> 24h the same week
            # after a real, live-confirmed gap: 30m keeps the model resident
            # across one continuous survey session, but this demo's real
            # usage pattern is sporadic (a team testing it a few times a day
            # over several days), and the backend process itself stays up
            # across all of that -- so the 30m TTL was expiring long before
            # the next real correction attempt, even though the server never
            # restarted. Confirmed directly: GET /api/ps returned an empty
            # model list (nothing resident) after a ~2-day gap with the
            # server continuously up, reproducing the exact "slow again"
            # report. 24h means any correction used at least once a day
            # keeps it effectively always warm; a genuinely idle stretch
            # longer than that still frees the memory rather than pinning it
            # forever. Real, disclosed cost: this model is ~2.0GiB
            # (confirmed via its Ollama manifest size) on a machine this
            # project has already documented as memory-constrained (as low
            # as ~300MB free observed elsewhere in this project's own
            # history) -- keeping it resident for up to 24h at a stretch is
            # a real, deliberate trade of memory headroom for response-time
            # reliability, not a free win. See main.py's startup pre-warm
            # for the matching load-at-boot half of this fix (keep_alive
            # alone doesn't help if the model was never loaded in the first
            # place).
            "keep_alive": "24h",
        }).encode()
        try:
            req = urllib.request.Request(
                f"{ollama_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._CORRECTION_TIMEOUT) as resp:
                body = json.loads(resp.read())
                return body.get("message", {}).get("content", "")
        except Exception as exc:
            _logger.debug("Ollama correction call failed: %s", exc)
            return None

    def _call_groq_json(self, prompt: str, schema: dict | None = None) -> str | None:
        """POST to Groq's OpenAI-compatible chat completions endpoint, with
        real JSON-Schema strict-mode structured output when `schema` is
        given. Returns raw response string or None.

        Added as part of the cloud-hosting migration (2026-09-04): the
        hosted deployment has no local Ollama to call, so
        CORRECTION_LLM_PROVIDER=groq routes _llm_extract_correction here
        instead. Deliberately mirrors _call_ollama_json's own low-level,
        no-CrewAI-overhead style (raw urllib, no new dependency) rather than
        going through llm_client.py's CrewAI/LiteLLM `LLM` class -- that
        class is built for the general TaskType routing chain, not this
        method's speed-critical, schema-constrained use.

        Model is openai/gpt-oss-120b -- already the model this project's
        ISCO-08 reranking work (backend/llm/llm_client.py) uses successfully
        via Groq's free tier, and (confirmed via a live web search against
        Groq's own docs before writing this) the only model family Groq's
        strict-mode structured output currently supports.

        Groq's strict mode has stricter requirements than the JSON Schema
        `correction_schema_for()` builds for Ollama: every object needs
        `additionalProperties: false` and *every* property (not just the
        ones semantically required) listed in `required`. This function
        adapts the schema at the call site rather than changing
        correction_schema_for()'s own output, so the Ollama path (still the
        local-dev default) is completely unaffected.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None

        groq_schema = None
        if schema is not None:
            item_props = schema["properties"]["corrections"]["items"]["properties"]
            groq_schema = {
                "type": "json_schema",
                "json_schema": {
                    "name": "corrections_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "corrections": {
                                "type": "array",
                                "maxItems": schema["properties"]["corrections"]["maxItems"],
                                "items": {
                                    "type": "object",
                                    "properties": item_props,
                                    "required": list(item_props.keys()),
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["corrections"],
                        "additionalProperties": False,
                    },
                },
            }

        payload = json.dumps({
            "model": "openai/gpt-oss-120b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "response_format": groq_schema if groq_schema is not None else {"type": "json_object"},
        }).encode()
        try:
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._CORRECTION_TIMEOUT) as resp:
                body = json.loads(resp.read())
                return body["choices"][0]["message"]["content"]
        except Exception as exc:
            _logger.debug("Groq correction call failed: %s", exc)
            return None

    def _call_anthropic_json(self, prompt: str) -> str | None:
        """Call Anthropic Claude as fallback for correction extraction. Returns raw text or None."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        payload = json.dumps({
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 512,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._CORRECTION_TIMEOUT) as resp:
                body = json.loads(resp.read())
                return body["content"][0]["text"]
        except Exception as exc:
            _logger.debug("Anthropic correction fallback failed: %s", exc)
            return None

    @staticmethod
    def _canonicalize_correction_value(field: str, raw: str) -> str:
        """Normalise a correction value to its canonical stored form.

        Looks up the raw value (case-insensitive, strip) in _VALUE_ALIASES[field].
        Falls back to the raw value as-is for free-text fields.
        """
        aliases = _VALUE_ALIASES.get(field)
        if not aliases:
            return raw.strip()
        key = raw.strip().lower()
        return aliases.get(key, raw.strip())

    @staticmethod
    def _extract_correction(ctx: "ConversationContext", text: str) -> bool:
        """Parse 'change X to Y' patterns and directly overwrite collected_data.

        Called from the VALIDATING → COLLECTING_INFO transition so that inline
        corrections (e.g. "change the nationality to Indian") are applied even
        though _extract_fields guards all fields with 'if field not in data'.

        Populates ctx.corrected_fields with the keys actually changed, and
        ctx.correction_rejected_field if a match was found but its value
        failed the sanity check (see _sanity_check_correction_value) — the
        caller re-prompts for that field rather than silently storing a
        sentence fragment as the answer.

        Returns True if at least one field was updated.
        """
        updated = False
        _patterns = (
            _CORRECTION_PATTERNS_EN,
            _CORRECTION_PATTERNS_EN_FIELD_FIRST,
            _CORRECTION_PATTERNS_AR,
        )
        for pattern in _patterns:
            m = pattern.search(text)
            if not m:
                continue
            raw_field = m.group(1).strip().lower()
            raw_value = m.group(2).strip()
            if not raw_value:
                continue
            # Try longest-matching alias first (handles multi-word phrases)
            field_key = None
            for alias in sorted(_FIELD_ALIASES, key=len, reverse=True):
                if alias in raw_field:
                    field_key = _FIELD_ALIASES[alias]
                    break
            if not field_key:
                continue
            cleaned = _sanity_check_correction_value(field_key, raw_value)
            if cleaned is None:
                ctx.correction_rejected_field = field_key
                continue
            ctx.collected_data[field_key] = cleaned
            ctx.corrected_fields.add(field_key)
            updated = True
        return updated

    @staticmethod
    def _format_history(history: list[dict], lang: str) -> str:
        if not history:
            return "No messages yet." if lang == "en" else "لا توجد رسائل بعد."
        lines = []
        for msg in history[-10:]:
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    @staticmethod
    def _format_collected(data: dict, lang: str) -> str:
        if not data:
            return "Nothing collected yet." if lang == "en" else "لم يتم جمع أي بيانات بعد."
        return "\n".join(f"  {k}: {v}" for k, v in data.items())
