"""
backend/agents/emotional_intelligence.py

Emotionally-aware survey assistant for the LFS conversational AI.

Responsibilities
----------------
1. Emotion detection   — Classifies respondent emotional state (stressed,
   confused, frustrated, neutral, engaged) using a two-stage pipeline:
     Stage 1  Fast rule-based keyword / punctuation matching — always runs.
     Stage 2  GPT-4o-mini CrewAI agent — refines Stage 1 and generates all
              bilingual text fields.

2. Adaptive prompts    — Generates empathetic, language-appropriate follow-up
   suggestions for the survey interviewer in both English and Arabic.

3. Support messages    — Produces short respondent-facing support messages that
   acknowledge emotional state and encourage continued participation.

4. Action recommendation — Maps detected state × intensity to one of four
   survey actions: continue | slow_down | pause | end.  The ``end``
   recommendation is reserved for high-intensity frustration; ``pause`` is
   used for high stress or significant confusion.

Emotional states
----------------
stressed   — anxious, overwhelmed, or under pressure
confused   — does not understand the question or the process
frustrated — annoyed, impatient, or actively disengaged
neutral    — calm, cooperative, no strong emotion signal
engaged    — enthusiastic, eager, or very cooperative

Survey actions
--------------
continue   — proceed normally (neutral / engaged respondents)
slow_down  — simplify language and slow the pace (mild stress / confusion)
pause      — offer a short break (high stress / significant confusion)
end        — recommend stopping the session (high frustration / distress)

LLM routing
-----------
GPT-4o-mini (TaskType.GENERAL) — fast and sufficient for tone analysis;
all emotional intelligence decisions are soft guidance, not data classification.

Usage
-----
from backend.agents.emotional_intelligence import EmotionalIntelligence

ei = EmotionalIntelligence()
result = ei.analyze("I don't understand any of this, help me!")
print(result.emotional_state)      # EmotionalState.CONFUSED
print(result.survey_action)        # SurveyAction.SLOW_DOWN
print(result.adapted_prompt_en)    # "I notice this might be unclear …"
print(result.adapted_prompt_ar)    # "أرى أن هذا قد يكون غير واضح …"
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Optional

from crewai import Agent, Crew, Task
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from backend.llm import TaskType, get_llm

load_dotenv()


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EmotionalState(str, Enum):
    """Respondent emotional state detected from message tone."""

    STRESSED   = "stressed"     # anxious, overwhelmed, pressured
    CONFUSED   = "confused"     # doesn't understand, lost, unclear
    FRUSTRATED = "frustrated"   # annoyed, impatient, disengaged
    NEUTRAL    = "neutral"      # calm, cooperative, no strong signal
    ENGAGED    = "engaged"      # enthusiastic, eager, positive


class SurveyAction(str, Enum):
    """Recommended action for the survey interviewer."""

    CONTINUE  = "continue"      # proceed normally
    SLOW_DOWN = "slow_down"     # use simpler language, more patience
    PAUSE     = "pause"         # offer a short break
    END       = "end"           # recommend stopping the session


# ---------------------------------------------------------------------------
# Output models (Pydantic)
# ---------------------------------------------------------------------------

class EmotionalSignal(BaseModel):
    """A linguistic or tonal feature that contributed to emotion detection."""

    signal_type: str                  # "keyword" | "punctuation" | "caps"
    text: str                         # triggering snippet from the message
    weight: float = Field(ge=0.0, le=1.0)


class EmotionalAnalysis(BaseModel):
    """Full result returned by ``EmotionalIntelligence.analyze()``."""

    raw_text:           str
    detected_language:  str                     # "en" | "ar" | "mixed"
    emotional_state:    EmotionalState
    confidence:         float = Field(ge=0.0, le=1.0)   # certainty of state
    intensity:          float = Field(ge=0.0, le=1.0)   # strength of emotion
    signals:            list[EmotionalSignal]   # what drove the detection
    adapted_prompt_en:  str     # empathetic interviewer follow-up (English)
    adapted_prompt_ar:  str     # empathetic interviewer follow-up (Arabic)
    support_message_en: str     # respondent-facing supportive message (English)
    support_message_ar: str     # respondent-facing supportive message (Arabic)
    survey_action:      SurveyAction
    action_reason:      str     # one-sentence rationale


# ---------------------------------------------------------------------------
# Rule-based signal patterns
# ---------------------------------------------------------------------------

# Each pattern is compiled with IGNORECASE so both Arabic and English are
# matched case-insensitively.  Arabic patterns work correctly because Python
# re supports full Unicode.

_STRESS_PATTERNS: list[str] = [
    # English
    r"\bstress(ed)?\b",          r"\boverwhelm(ed)?\b",
    r"\banxious\b",              r"\bworried\b",
    r"\bnervous\b",              r"\bpanic(k?ing|k?ed|ky)?\b",
    r"\btoo much\b",             r"\bcan'?t (take|handle|cope)\b",
    r"\bexhausted?\b",           r"\bburn(ed|t)? out\b",
    r"\bunder (a lot of )?pressure\b", r"\bscared\b",
    # Arabic
    r"متوتر",   r"مرهق",   r"قلق",   r"خائف",
    r"ضغط نفسي", r"منهك",  r"مضغوط", r"مرعوب",
]

_CONFUSION_PATTERNS: list[str] = [
    # English
    r"\bconfus(ed|ing)\b",       r"\bdon'?t understand\b",
    r"\bwhat do you mean\b",     r"\bnot sure (what|how|why)\b",
    r"\bunclear\b",              r"\bi'?m lost\b",
    r"\bi don'?t know\b",        r"\bhelp me understand\b",
    r"\bmakes? no sense\b",      r"\bdon'?t get it\b",
    # Arabic
    r"لا أفهم", r"لا أعرف", r"ماذا تقصد", r"غير واضح",
    r"محتار",   r"مش فاهم", r"مو واضح",   r"مبهم",
]

_FRUSTRATION_PATTERNS: list[str] = [
    # English
    r"\bfrustrat(ed|ing)\b",     r"\bann?oy(ed|ing)\b",
    r"\btired of (this|the|these)\b",
    r"\bstop (it|asking|this)\b", r"\benough\b",
    r"\bpointless\b",            r"\bwaste of (my |your )?time\b",
    r"\bthis is ridiculous\b",   r"\bleave me alone\b",
    r"\bI (hate|can'?t stand) this\b",
    # Arabic
    r"محبط",     r"مستاء", r"تعبت من", r"كفاية",
    r"اتركني",   r"ملل",   r"مزعج",    r"سخيف",
    r"ما فائدة",
]

_ENGAGEMENT_PATTERNS: list[str] = [
    # English
    r"\bhappy to (help|answer|continue|participate)\b",
    r"\bglad to\b",   r"\bof course\b",  r"\bcertainly\b",
    r"\bwith pleasure\b", r"\bexcited?\b", r"\binterested?\b",
    r"\bno problem\b", r"\bsounds good\b", r"\blook forward\b",
    # Arabic
    r"بكل سرور",  r"بالتأكيد", r"طبعًا",    r"ممتاز",
    r"لا مشكلة",  r"مع السرور", r"يسعدني",  r"أهلًا وسهلًا",
]

# Pre-compiled pattern sets keyed by emotional state
_COMPILED: dict[EmotionalState, list[re.Pattern]] = {
    EmotionalState.STRESSED:    [re.compile(p, re.IGNORECASE) for p in _STRESS_PATTERNS],
    EmotionalState.CONFUSED:    [re.compile(p, re.IGNORECASE) for p in _CONFUSION_PATTERNS],
    EmotionalState.FRUSTRATED:  [re.compile(p, re.IGNORECASE) for p in _FRUSTRATION_PATTERNS],
    EmotionalState.ENGAGED:     [re.compile(p, re.IGNORECASE) for p in _ENGAGEMENT_PATTERNS],
}

_KEYWORD_WEIGHT = 0.30   # score added per keyword match
_PUNCT_WEIGHT   = 0.20   # score added per punctuation signal
_CAPS_WEIGHT    = 0.25   # score added for all-caps words
_THRESHOLD      = 0.20   # minimum score to claim a non-neutral state


# ---------------------------------------------------------------------------
# Bilingual response templates (used when LLM fails or produces empty fields)
# ---------------------------------------------------------------------------

_ADAPTED_PROMPTS: dict[EmotionalState, dict[str, str]] = {
    EmotionalState.STRESSED: {
        "en": (
            "I understand this may feel like a lot right now. "
            "Let's take it slowly — there is absolutely no rush, "
            "and we can pause at any time."
        ),
        "ar": (
            "أفهم أن هذا قد يبدو كثيرًا في الوقت الحالي. "
            "دعنا نأخذها بهدوء — لا يوجد أي تسرع، "
            "ويمكننا التوقف في أي وقت."
        ),
    },
    EmotionalState.CONFUSED: {
        "en": (
            "I notice this might be unclear — let me rephrase the question. "
            "Please feel free to ask me to explain anything at any point."
        ),
        "ar": (
            "أرى أن هذا قد يكون غير واضح — دعني أعيد صياغة السؤال. "
            "لا تتردد في طلب توضيح أي شيء في أي وقت."
        ),
    },
    EmotionalState.FRUSTRATED: {
        "en": (
            "I hear you, and I truly appreciate your patience. "
            "Let's keep this as simple and brief as possible."
        ),
        "ar": (
            "أسمعك، وأقدّر صبرك حقًا. "
            "دعنا نجعل هذا بسيطًا وموجزًا قدر الإمكان."
        ),
    },
    EmotionalState.NEUTRAL: {
        "en": "Thank you for your response. Let's continue with the next question.",
        "ar": "شكرًا لإجابتك. لنكمل السؤال التالي.",
    },
    EmotionalState.ENGAGED: {
        "en": "Thank you — that is very helpful! Let's continue.",
        "ar": "شكرًا — هذا مفيد جدًا! لنكمل.",
    },
}

_SUPPORT_MESSAGES: dict[EmotionalState, dict[str, str]] = {
    EmotionalState.STRESSED: {
        "en": (
            "Your participation is entirely voluntary. "
            "You can stop or take a break at any time — we truly appreciate your time."
        ),
        "ar": (
            "مشاركتك طوعية تمامًا. "
            "يمكنك التوقف أو أخذ استراحة في أي وقت — نحن نقدّر وقتك حقًا."
        ),
    },
    EmotionalState.CONFUSED: {
        "en": (
            "There are no wrong answers here. "
            "Just share what you know, and I will guide you through each step."
        ),
        "ar": (
            "لا توجد إجابات خاطئة هنا. "
            "فقط شاركنا ما تعرفه وسأرشدك في كل خطوة."
        ),
    },
    EmotionalState.FRUSTRATED: {
        "en": (
            "Thank you for bearing with us. "
            "Your input genuinely contributes to important national employment research."
        ),
        "ar": (
            "شكرًا لتحملّك معنا. "
            "مساهمتك تُسهم فعلًا في أبحاث التوظيف الوطنية المهمة."
        ),
    },
    EmotionalState.NEUTRAL: {
        "en": "Your responses are helping us understand employment conditions in your region.",
        "ar": "إجاباتك تساعدنا على فهم أوضاع التوظيف في منطقتك.",
    },
    EmotionalState.ENGAGED: {
        "en": "Your detailed responses are extremely valuable for our research — thank you!",
        "ar": "إجاباتك التفصيلية ذات قيمة كبيرة لأبحاثنا — شكرًا لك!",
    },
}


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_EI_INSTRUCTIONS = """\
You are an emotionally intelligent survey interviewer assistant for the Labour \
Force Survey (LFS). Analyse the emotional tone of a respondent's message and \
generate empathetic, culturally sensitive guidance for the interviewer.

Return ONLY a valid JSON object — no markdown fences, no extra text:
{
  "emotional_state": "<one of: stressed, confused, frustrated, neutral, engaged>",
  "confidence": <float 0.0–1.0>,
  "intensity": <float 0.0–1.0; how strongly the emotion is expressed>,
  "adapted_prompt_en": "<empathetic interviewer follow-up in English (1–2 sentences)>",
  "adapted_prompt_ar": "<empathetic interviewer follow-up in Arabic (1–2 sentences)>",
  "support_message_en": "<short supportive message for the respondent in English>",
  "support_message_ar": "<short supportive message in Arabic>",
  "survey_action": "<one of: continue, slow_down, pause, end>",
  "action_reason": "<one sentence explaining why this action is recommended>"
}

Emotional state definitions:
  stressed   — respondent appears anxious, overwhelmed, or under pressure
  confused   — respondent does not understand a question or the process
  frustrated — respondent is annoyed, impatient, or disengaged
  neutral    — respondent is calm and cooperative
  engaged    — respondent is enthusiastic or very cooperative

Survey action guidelines:
  continue   — neutral / engaged; proceed normally
  slow_down  — mild stress, confusion, or early frustration; use simpler language
  pause      — high stress or significant confusion; offer a short break
  end        — high frustration or distress; recommend stopping respectfully

Language requirements:
  - Arabic must be correct Modern Standard Arabic (فصحى)
  - Tone must be warm, professional, and culturally appropriate
"""


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_ARABIC_ALPHA = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
_LATIN_ALPHA = re.compile(r"[A-Za-z]")


def _detect_script(text: str) -> str:
    """
    Return ``"ar"``, ``"en"``, or ``"mixed"`` based on character ratios.

    - ≥ 90% Arabic alphabetic characters → ``"ar"``
    - ≥ 10% Arabic and ≥ 10% Latin       → ``"mixed"``
    - Otherwise                           → ``"en"``
    """
    ar  = len(_ARABIC_ALPHA.findall(text))
    lat = len(_LATIN_ALPHA.findall(text))
    total = ar + lat
    if total == 0:
        return "en"
    ar_ratio = ar / total
    if ar_ratio >= 0.90:
        return "ar"
    if ar_ratio >= 0.10:
        return "mixed"
    return "en"


# ---------------------------------------------------------------------------
# EmotionalIntelligence
# ---------------------------------------------------------------------------

class EmotionalIntelligence:
    """
    Two-stage emotional-state detector with bilingual empathetic response generation.

    Stage 1 — Rule-based keyword matching (always runs, deterministic).
    Stage 2 — GPT-4o-mini CrewAI agent (refines Stage 1 and writes bilingual
              adapted prompts / support messages).  If the LLM call fails or
              returns unparseable output, Stage 1 results + bilingual templates
              are used as the fallback.
    """

    def __init__(self) -> None:
        self._llm = get_llm(TaskType.GENERAL)   # GPT-4o-mini, temp 0.3

        self._agent = Agent(
            role="Empathetic Survey Support Specialist",
            goal=(
                "Detect respondent emotional state from survey message tone and "
                "generate warm, culturally sensitive guidance that helps the "
                "interviewer adapt in real time to respondent needs — in both "
                "English and Arabic."
            ),
            backstory=(
                "You are a trained survey methodologist and cross-cultural "
                "communication expert at a national statistics office. You have "
                "extensive experience recognising stress, confusion, and "
                "disengagement in survey conversations and crafting empathetic "
                "responses that respect both Western and Arabic cultural norms."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str, language: str = "en") -> EmotionalAnalysis:
        """
        Analyse the emotional tone of one survey respondent message.

        Parameters
        ----------
        text : str
            Raw respondent message (English, Arabic, or code-switched).
        language : str
            Session survey language hint (``"en"`` or ``"ar"``).
            Used only when *text* is empty.

        Returns
        -------
        EmotionalAnalysis
            Detected state, confidence, intensity, rule-based signals, bilingual
            adapted prompts, support messages, and survey-action recommendation.
        """
        text = text.strip()
        if not text:
            return self._empty_result(language)

        detected_lang = _detect_script(text)

        # Stage 1: fast rule-based detection
        rb_state, rb_conf, rb_intensity, signals = self._detect_emotion_rules(text)

        # Stage 2: LLM refinement + bilingual text generation
        try:
            return self._analyze_with_llm(
                text, detected_lang, rb_state, rb_conf, rb_intensity, signals
            )
        except Exception:
            return self._build_fallback_analysis(
                text, detected_lang, rb_state, rb_conf, rb_intensity, signals
            )

    # ------------------------------------------------------------------
    # Stage 1: rule-based detection
    # ------------------------------------------------------------------

    def _detect_emotion_rules(
        self,
        text: str,
    ) -> tuple[EmotionalState, float, float, list[EmotionalSignal]]:
        """
        Score each emotional state using keyword and punctuation patterns.

        Returns
        -------
        (state, confidence, intensity, signals)
        """
        signals: list[EmotionalSignal] = []
        scores: dict[EmotionalState, float] = {
            EmotionalState.STRESSED:   0.0,
            EmotionalState.CONFUSED:   0.0,
            EmotionalState.FRUSTRATED: 0.0,
            EmotionalState.ENGAGED:    0.0,
        }

        # Keyword matching
        for state, patterns in _COMPILED.items():
            for pat in patterns:
                m = pat.search(text)
                if m:
                    scores[state] = min(scores[state] + _KEYWORD_WEIGHT, 1.0)
                    signals.append(EmotionalSignal(
                        signal_type="keyword",
                        text=m.group(),
                        weight=_KEYWORD_WEIGHT,
                    ))

        # Punctuation: 3+ consecutive exclamation marks → stressed + frustrated
        if re.search(r"!{3,}", text):
            scores[EmotionalState.STRESSED]   = min(
                scores[EmotionalState.STRESSED]   + _PUNCT_WEIGHT, 1.0
            )
            scores[EmotionalState.FRUSTRATED] = min(
                scores[EmotionalState.FRUSTRATED] + _PUNCT_WEIGHT, 1.0
            )
            signals.append(EmotionalSignal(
                signal_type="punctuation", text="!!!", weight=_PUNCT_WEIGHT
            ))

        # Punctuation: 3+ consecutive question marks → confused
        if re.search(r"\?{3,}", text):
            scores[EmotionalState.CONFUSED] = min(
                scores[EmotionalState.CONFUSED] + _PUNCT_WEIGHT, 1.0
            )
            signals.append(EmotionalSignal(
                signal_type="punctuation", text="???", weight=_PUNCT_WEIGHT
            ))

        # All-caps words: 2+ uppercase words (length > 2) → frustrated
        caps_words = [
            w for w in text.split()
            if len(w) > 2 and w.isupper() and w.isalpha()
        ]
        if len(caps_words) >= 2:
            scores[EmotionalState.FRUSTRATED] = min(
                scores[EmotionalState.FRUSTRATED] + _CAPS_WEIGHT, 1.0
            )
            signals.append(EmotionalSignal(
                signal_type="caps",
                text=" ".join(caps_words[:3]),
                weight=_CAPS_WEIGHT,
            ))

        # Determine winner
        best_state = max(scores, key=lambda s: scores[s])
        best_score = scores[best_state]

        if best_score < _THRESHOLD:
            return EmotionalState.NEUTRAL, 1.0, 0.0, signals

        confidence = min(best_score, 0.95)
        intensity  = min(best_score, 1.0)
        return best_state, confidence, intensity, signals

    # ------------------------------------------------------------------
    # Stage 2: LLM refinement
    # ------------------------------------------------------------------

    def _analyze_with_llm(
        self,
        text: str,
        detected_lang: str,
        rb_state: EmotionalState,
        rb_conf: float,
        rb_intensity: float,
        signals: list[EmotionalSignal],
    ) -> EmotionalAnalysis:
        """Build and run the CrewAI task; delegate parsing to ``_parse_llm_response``."""
        task = Task(
            description=(
                f"{_EI_INSTRUCTIONS}\n\n"
                f"Detected language   : {detected_lang}\n"
                f"Rule-based state    : {rb_state.value} "
                f"(confidence {rb_conf:.2f}, intensity {rb_intensity:.2f})\n\n"
                f'Respondent message:\n"""\n{text}\n"""'
            ),
            expected_output=(
                'JSON with keys: emotional_state, confidence, intensity, '
                'adapted_prompt_en, adapted_prompt_ar, support_message_en, '
                'support_message_ar, survey_action, action_reason'
            ),
            agent=self._agent,
        )

        crew = Crew(agents=[self._agent], tasks=[task], verbose=False)
        raw  = str(crew.kickoff()).strip()
        return self._parse_llm_response(
            raw, text, detected_lang, rb_state, rb_conf, rb_intensity, signals
        )

    # ------------------------------------------------------------------
    # LLM response parsing
    # ------------------------------------------------------------------

    def _parse_llm_response(
        self,
        raw: str,
        text: str,
        detected_lang: str,
        rb_state: EmotionalState,
        rb_conf: float,
        rb_intensity: float,
        signals: list[EmotionalSignal],
    ) -> EmotionalAnalysis:
        """
        Parse the LLM JSON and return an ``EmotionalAnalysis``.

        Falls back to rule-based values + bilingual templates on any failure.
        """
        clean = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL
        ).strip()

        data: dict = {}
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", clean, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        if not data:
            return self._build_fallback_analysis(
                text, detected_lang, rb_state, rb_conf, rb_intensity, signals
            )

        # ── emotional_state ──────────────────────────────────────────────
        try:
            state = EmotionalState(str(data.get("emotional_state", "")).lower().strip())
        except ValueError:
            state = rb_state

        # ── confidence / intensity ────────────────────────────────────────
        def _clamp(val, default: float) -> float:
            try:
                return max(0.0, min(1.0, float(val)))
            except (TypeError, ValueError):
                return default

        confidence = _clamp(data.get("confidence"), rb_conf)
        intensity  = _clamp(data.get("intensity"),  rb_intensity)

        # ── survey_action ────────────────────────────────────────────────
        try:
            action = SurveyAction(
                str(data.get("survey_action", "")).lower().strip()
            )
        except ValueError:
            action, _ = self._determine_action(state, intensity)

        action_reason = str(data.get("action_reason", "")).strip()
        if not action_reason:
            _, action_reason = self._determine_action(state, intensity)

        # ── bilingual text fields ─────────────────────────────────────────
        def _field(key: str, fallback: str) -> str:
            return str(data.get(key, "")).strip() or fallback

        adapted_en  = _field("adapted_prompt_en",  _ADAPTED_PROMPTS[state]["en"])
        adapted_ar  = _field("adapted_prompt_ar",  _ADAPTED_PROMPTS[state]["ar"])
        support_en  = _field("support_message_en", _SUPPORT_MESSAGES[state]["en"])
        support_ar  = _field("support_message_ar", _SUPPORT_MESSAGES[state]["ar"])

        return EmotionalAnalysis(
            raw_text=text,
            detected_language=detected_lang,
            emotional_state=state,
            confidence=confidence,
            intensity=intensity,
            signals=signals,
            adapted_prompt_en=adapted_en,
            adapted_prompt_ar=adapted_ar,
            support_message_en=support_en,
            support_message_ar=support_ar,
            survey_action=action,
            action_reason=action_reason,
        )

    # ------------------------------------------------------------------
    # Fallback builder (no LLM)
    # ------------------------------------------------------------------

    def _build_fallback_analysis(
        self,
        text: str,
        detected_lang: str,
        state: EmotionalState,
        confidence: float,
        intensity: float,
        signals: list[EmotionalSignal],
    ) -> EmotionalAnalysis:
        """
        Construct a deterministic ``EmotionalAnalysis`` using bilingual templates.

        Called when the LLM response is absent, malformed, or raises an exception.
        """
        action, reason = self._determine_action(state, intensity)
        return EmotionalAnalysis(
            raw_text=text,
            detected_language=detected_lang,
            emotional_state=state,
            confidence=confidence,
            intensity=intensity,
            signals=signals,
            adapted_prompt_en=_ADAPTED_PROMPTS[state]["en"],
            adapted_prompt_ar=_ADAPTED_PROMPTS[state]["ar"],
            support_message_en=_SUPPORT_MESSAGES[state]["en"],
            support_message_ar=_SUPPORT_MESSAGES[state]["ar"],
            survey_action=action,
            action_reason=reason,
        )

    # ------------------------------------------------------------------
    # Action mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_action(
        state: EmotionalState,
        intensity: float,
    ) -> tuple[SurveyAction, str]:
        """
        Map ``(state, intensity)`` to a ``(SurveyAction, reason)`` pair.

        Intensity threshold for elevated actions: ≥ 0.60.
        """
        if state in (EmotionalState.NEUTRAL, EmotionalState.ENGAGED):
            return (
                SurveyAction.CONTINUE,
                "Respondent is calm and cooperative; proceed normally.",
            )

        if state == EmotionalState.CONFUSED:
            if intensity >= 0.60:
                return (
                    SurveyAction.PAUSE,
                    "Significant confusion detected; offer a short break or clarification.",
                )
            return (
                SurveyAction.SLOW_DOWN,
                "Mild confusion detected; simplify the question wording.",
            )

        if state == EmotionalState.STRESSED:
            if intensity >= 0.60:
                return (
                    SurveyAction.PAUSE,
                    "High stress detected; offer a short break.",
                )
            return (
                SurveyAction.SLOW_DOWN,
                "Mild stress detected; proceed more gently and reassure the respondent.",
            )

        if state == EmotionalState.FRUSTRATED:
            if intensity >= 0.60:
                return (
                    SurveyAction.END,
                    "Significant frustration detected; recommend ending the session respectfully.",
                )
            return (
                SurveyAction.SLOW_DOWN,
                "Mild frustration detected; acknowledge feelings and simplify.",
            )

        return SurveyAction.CONTINUE, "No concerning emotional signals detected."

    # ------------------------------------------------------------------
    # Empty-text helper
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(language: str) -> EmotionalAnalysis:
        """Return a neutral ``EmotionalAnalysis`` for empty or whitespace input."""
        lang = language if language in ("en", "ar") else "en"
        return EmotionalAnalysis(
            raw_text="",
            detected_language=lang,
            emotional_state=EmotionalState.NEUTRAL,
            confidence=1.0,
            intensity=0.0,
            signals=[],
            adapted_prompt_en=_ADAPTED_PROMPTS[EmotionalState.NEUTRAL]["en"],
            adapted_prompt_ar=_ADAPTED_PROMPTS[EmotionalState.NEUTRAL]["ar"],
            support_message_en=_SUPPORT_MESSAGES[EmotionalState.NEUTRAL]["en"],
            support_message_ar=_SUPPORT_MESSAGES[EmotionalState.NEUTRAL]["ar"],
            survey_action=SurveyAction.CONTINUE,
            action_reason="Empty message; no emotional signal detected.",
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[EmotionalIntelligence] = None


def get_emotional_intelligence() -> EmotionalIntelligence:
    """
    Return the module-level ``EmotionalIntelligence`` singleton.

    Creates the instance on the first call (loads the LLM client and
    CrewAI agent).  Subsequent calls return the cached instance.
    """
    global _instance
    if _instance is None:
        _instance = EmotionalIntelligence()
    return _instance
