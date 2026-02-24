"""
backend/agents/conversation_manager.py

CrewAI-based Conversation Manager agent that drives an LFS survey
through a five-state finite state machine.

States
------
greeting        → welcome the respondent and confirm language preference
collecting_info → ask LFS survey questions one at a time
clarifying      → probe ambiguous or incomplete answers
validating      → read back and confirm collected answers
completing      → thank respondent and close the session

Supported languages: English ("en"), Arabic ("ar")
LLM: GPT-4o-mini (OpenAI)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from crewai import Agent, Crew, Task

from backend.llm import TaskType, get_llm


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


# ---------------------------------------------------------------------------
# Language-specific prompts
# ---------------------------------------------------------------------------

_SYSTEM_BASE: dict[str, str] = {
    "en": (
        "You are a professional, empathetic survey interviewer for the "
        "Labour Force Survey (LFS). Your role is to collect accurate "
        "employment information from respondents in a natural, conversational "
        "way. Always be polite, clear, and patient. Ask one question at a time."
    ),
    "ar": (
        "أنت محاور مهني ومتعاطف لمسح القوى العاملة (LFS). دورك هو جمع "
        "معلومات دقيقة حول التوظيف من المشاركين بطريقة طبيعية وتفاعلية. "
        "كن دائمًا مؤدبًا وواضحًا وصبورًا. اطرح سؤالًا واحدًا في كل مرة."
    ),
}

_STATE_INSTRUCTIONS: dict[ConversationState, dict[str, str]] = {
    ConversationState.GREETING: {
        "en": (
            "Greet the respondent warmly and introduce yourself as the LFS "
            "survey assistant. Briefly explain the purpose of the survey "
            "(understanding employment and labour market conditions). "
            "Ask whether they prefer to continue in English or Arabic, then "
            "invite them to begin when ready."
        ),
        "ar": (
            "رحّب بالمشارك بدفء وقدّم نفسك كمساعد مسح القوى العاملة. "
            "اشرح بإيجاز الغرض من المسح (فهم أوضاع التوظيف وسوق العمل). "
            "ابدأ المقابلة باللغة العربية بشكل واضح ودعهم يستعدون للبدء."
        ),
    },
    ConversationState.COLLECTING_INFO: {
        "en": (
            "Collect the respondent's employment information by asking the "
            "following questions one at a time, in order, skipping any already answered:\n"
            "1. Current employment status (employed / unemployed / not in labour force)\n"
            "2. Job title and main duties (if employed)\n"
            "3. Industry or sector of work\n"
            "4. Usual hours worked per week\n"
            "5. Employment type (full-time / part-time / self-employed)\n\n"
            "Acknowledge each answer warmly before moving to the next question. "
            "If an answer seems vague or incomplete, flag it mentally for clarification "
            "but do not interrupt the flow unless necessary."
        ),
        "ar": (
            "اجمع معلومات التوظيف من المشارك بطرح الأسئلة التالية واحدًا تلو الآخر، "
            "بالترتيب، مع تخطي ما تمت الإجابة عنه:\n"
            "1. حالة التوظيف الحالية (موظف / عاطل عن العمل / خارج سوق العمل)\n"
            "2. المسمى الوظيفي والمهام الرئيسية (إذا كان موظفًا)\n"
            "3. القطاع أو الصناعة\n"
            "4. ساعات العمل المعتادة أسبوعيًا\n"
            "5. نوع التوظيف (دوام كامل / دوام جزئي / عمل حر)\n\n"
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
            "أذكر أن إجاباتهم تساهم في أبحاث سوق العمل المهمة وصنع القرار. "
            "أنهِ المحادثة بدفء واحترافية."
        ),
    },
}

# Fields required before transitioning to VALIDATING
_REQUIRED_FIELDS = frozenset({
    "employment_status",
    "job_title",
    "industry",
    "hours_per_week",
    "employment_type",
})

# Words/phrases that signal confirmation in each language
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


# ---------------------------------------------------------------------------
# ConversationManager
# ---------------------------------------------------------------------------

class ConversationManager:
    """
    Manages an LFS survey conversation through a five-state FSM using a
    CrewAI agent backed by GPT-4o-mini.

    Usage
    -----
    manager = ConversationManager()
    ctx = manager.new_context(session_id=42, language="en")
    response = manager.process_message(ctx, "Hello, I'd like to start.")
    """

    def __init__(self) -> None:
        self._llm = get_llm(TaskType.GENERAL)
        self._agent = Agent(
            role="Conversation Manager",
            goal=(
                "Guide respondents through the Labour Force Survey accurately "
                "and empathetically, collecting complete and unambiguous "
                "employment data in English or Arabic."
            ),
            backstory=(
                "You are a seasoned LFS survey interviewer trained by a national "
                "statistics office. You understand that precise employment data "
                "drives government policy and you are skilled at keeping "
                "conversations focused, natural, and culturally sensitive "
                "across both English and Arabic-speaking respondents."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_context(self, session_id: int, language: str = "en") -> ConversationContext:
        """Create a fresh ConversationContext for a new survey session."""
        lang = language if language in ("en", "ar") else "en"
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

        task = self._build_task(ctx)
        crew = Crew(agents=[self._agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        response = str(result).strip()

        ctx.history.append({"role": "assistant", "content": response})
        self._transition(ctx, user_message, response)

        return response

    # ------------------------------------------------------------------
    # Task construction
    # ------------------------------------------------------------------

    def _build_task(self, ctx: ConversationContext) -> Task:
        """Build a CrewAI Task for the current FSM state and language."""
        lang = ctx.language

        description = "\n\n".join([
            _SYSTEM_BASE[lang],
            f"CURRENT STATE: {ctx.state.value}",
            f"STATE INSTRUCTIONS:\n{_STATE_INSTRUCTIONS[ctx.state][lang]}",
            f"COLLECTED DATA SO FAR:\n{self._format_collected(ctx.collected_data, lang)}",
            f"CONVERSATION HISTORY (latest {min(len(ctx.history), 10)} turns):\n"
            f"{self._format_history(ctx.history, lang)}",
            (
                "Respond naturally as the survey interviewer for this state. "
                "Output ONLY what you say to the respondent — no internal notes, "
                "no JSON, no meta-commentary."
            ),
        ])

        expected_output = (
            "A natural, conversational interviewer reply in English."
            if lang == "en"
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
            self._extract_fields(ctx, user_message)

            if self._is_ambiguous(user_message):
                ctx.state = ConversationState.CLARIFYING
            elif _REQUIRED_FIELDS.issubset(ctx.collected_data.keys()):
                ctx.state = ConversationState.VALIDATING

        elif state == ConversationState.CLARIFYING:
            self._extract_fields(ctx, user_message)
            ctx.state = ConversationState.COLLECTING_INFO

        elif state == ConversationState.VALIDATING:
            if self._is_confirmed(user_message, ctx.language):
                ctx.state = ConversationState.COMPLETING
            else:
                # Respondent wants corrections — return to collecting
                ctx.state = ConversationState.COLLECTING_INFO

        # COMPLETING is terminal

    # ------------------------------------------------------------------
    # Heuristic field extraction
    # ------------------------------------------------------------------

    def _extract_fields(self, ctx: ConversationContext, text: str) -> None:
        """
        Extract employment field values from free-text user input using
        keyword heuristics. Stores raw values for downstream ISCO classification.
        """
        lower = text.lower()
        data = ctx.collected_data

        # Employment status — check more specific terms first to avoid substring collisions
        # ("employed" is a substring of "unemployed", so check "unemployed" first)
        if "employment_status" not in data:
            if any(w in lower for w in ("unemployed", "looking for work", "عاطل", "أبحث عن عمل")):
                data["employment_status"] = "unemployed"
            elif any(w in lower for w in ("employed", "working", "موظف", "أعمل", "أنا أعمل")):
                data["employment_status"] = "employed"
            elif any(w in lower for w in ("retired", "student", "housewife", "متقاعد", "طالب", "ربة منزل")):
                data["employment_status"] = "not_in_labour_force"

        # Hours per week — match digits followed by hours/ساعات
        if "hours_per_week" not in data:
            match = re.search(r"(\d+)\s*(?:hours?|hrs?|ساعات?|ساعة)", lower)
            if match:
                data["hours_per_week"] = match.group(1)

        # Employment type
        if "employment_type" not in data:
            if re.search(r"full[\s-]?time|دوام كامل", lower):
                data["employment_type"] = "full_time"
            elif re.search(r"part[\s-]?time|دوام جزئي", lower):
                data["employment_type"] = "part_time"
            elif re.search(r"self[\s-]?employ|freelanc|عمل حر|مستقل", lower):
                data["employment_type"] = "self_employed"

        # Job title — store raw text when recognisable profession keywords appear
        if "job_title" not in data:
            profession_keywords = {
                "en": [
                    "engineer", "manager", "director", "teacher", "professor",
                    "doctor", "nurse", "driver", "analyst", "developer",
                    "accountant", "officer", "architect", "consultant",
                ],
                "ar": [
                    "مهندس", "مدير", "معلم", "أستاذ", "طبيب", "ممرض",
                    "سائق", "محلل", "مطور", "محاسب", "مسؤول", "مهندس معماري",
                ],
            }
            all_keywords = profession_keywords["en"] + profession_keywords["ar"]
            if any(kw in lower for kw in all_keywords):
                data["job_title"] = text.strip()

        # Industry / sector
        if "industry" not in data:
            sector_map: dict[str, list[str]] = {
                "technology": ["software", "tech", "it ", "technology", "تقنية", "برمجة", "معلومات"],
                "healthcare": ["hospital", "clinic", "health", "medical", "صحة", "مستشفى", "عيادة"],
                "education": ["school", "university", "college", "teach", "تعليم", "مدرسة", "جامعة"],
                "construction": ["construction", "build", "infrastructure", "بناء", "مقاولات", "تشييد"],
                "retail_trade": ["shop", "store", "retail", "trade", "تجزئة", "تجارة", "محل"],
                "government": ["government", "ministry", "public sector", "civil service",
                               "حكومة", "وزارة", "قطاع عام", "خدمة مدنية"],
                "finance": ["bank", "finance", "insurance", "investment", "بنك", "مالية", "تأمين"],
                "manufacturing": ["factory", "manufactur", "production", "مصنع", "إنتاج", "تصنيع"],
            }
            for sector, keywords in sector_map.items():
                if any(kw in lower for kw in keywords):
                    data["industry"] = sector
                    break

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
    def _format_history(history: list[dict], lang: str) -> str:
        if not history:
            return "No messages yet." if lang == "en" else "لا توجد رسائل بعد."
        lines = []
        for msg in history[-10:]:   # limit to last 10 turns for context window
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    @staticmethod
    def _format_collected(data: dict, lang: str) -> str:
        if not data:
            return "Nothing collected yet." if lang == "en" else "لم يتم جمع أي بيانات بعد."
        return "\n".join(f"  {k}: {v}" for k, v in data.items())
