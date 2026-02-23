"""
backend/agents/language_processor.py

CrewAI-based Language Processor agent for the LFS survey.

Responsibilities
----------------
1. Language detection   – identifies whether a message is English, Arabic,
                          or code-switched (both scripts present) using
                          langdetect plus Unicode script-ratio analysis.
2. Code-switching       – segments the text into contiguous Arabic / Latin runs
                          and exposes per-segment language labels.
3. Named Entity Recognition – extracts LFS-relevant entities (job titles,
                               organisations, locations, industry sectors,
                               employment status, durations, hours) using a
                               GPT-4o-mini CrewAI agent that returns strict JSON.
4. Structured output    – all results are returned in a validated Pydantic model.

Supported scripts : Arabic (Unicode 0600-06FF + extended blocks) and Latin.
LLM              : GPT-4o-mini via CrewAI
"""

from __future__ import annotations

import json
import re
from typing import Optional

from crewai import Agent, Crew, Task
from langdetect import DetectorFactory, LangDetectException, detect_langs
from pydantic import BaseModel, Field

from backend.llm import TaskType, get_llm

# Make langdetect deterministic across runs
DetectorFactory.seed = 0


# ---------------------------------------------------------------------------
# Unicode script patterns
# ---------------------------------------------------------------------------

# Covers Arabic, Arabic Supplement, Arabic Extended-A, Arabic Presentation
# Forms-A and -B blocks.
_ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+"
)
_LATIN_RE = re.compile(r"[A-Za-z]+")

# Both scripts must each contribute at least this fraction of alphabetic
# characters to be considered code-switched.
_CODE_SWITCH_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# LFS-relevant entity labels
# ---------------------------------------------------------------------------

LFS_ENTITY_LABELS: list[str] = [
    "JOB_TITLE",          # e.g. "software engineer", "مهندس برمجيات"
    "ORGANIZATION",       # e.g. "Ministry of Finance", "وزارة المالية"
    "LOCATION",           # e.g. "Riyadh", "الرياض"
    "INDUSTRY",           # e.g. "healthcare", "قطاع الصحة"
    "EMPLOYMENT_STATUS",  # e.g. "unemployed", "عاطل عن العمل"
    "DURATION",           # e.g. "5 years", "٣ سنوات"
    "HOURS",              # e.g. "40 hours a week", "٤٠ ساعة أسبوعيًا"
    "PERSON",             # e.g. "Ahmed", "أحمد"
]

_LABELS_BLOCK = "\n".join(f"  - {lbl}" for lbl in LFS_ENTITY_LABELS)


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------

class CodeSegment(BaseModel):
    """A contiguous run of characters belonging to one script."""

    text: str
    script: str                            # "arabic" | "latin" | "other"
    detected_language: Optional[str] = None  # langdetect result for this segment


class Entity(BaseModel):
    """A single named entity extracted from the message."""

    text: str
    label: str                             # one of LFS_ENTITY_LABELS
    language: str                          # "en" | "ar"
    start: Optional[int] = None            # character offset in original text
    end: Optional[int] = None


class LanguageProcessorResult(BaseModel):
    """Full structured result returned by LanguageProcessor.process()."""

    raw_text: str
    detected_language: str                 # "en" | "ar" | "other"
    confidence: float = Field(ge=0.0, le=1.0)
    is_code_switched: bool
    arabic_ratio: float = Field(ge=0.0, le=1.0)
    latin_ratio: float = Field(ge=0.0, le=1.0)
    segments: list[CodeSegment]
    entities: list[Entity]


# ---------------------------------------------------------------------------
# NER task prompt
# ---------------------------------------------------------------------------

_NER_INSTRUCTIONS = f"""You are a multilingual Named Entity Recognition (NER) specialist
for a Labour Force Survey (LFS).

Extract entities from the survey message below. Only use these entity types:
{_LABELS_BLOCK}

Output rules (strictly enforced):
- Return ONLY a valid JSON array — no markdown fences, no explanation.
- Each element must be an object with exactly three keys:
    "text"     : the entity text exactly as it appears in the input
    "label"    : one of the types listed above (uppercase)
    "language" : "en" if the entity is English, "ar" if Arabic
- If no entities are found return an empty array: []
"""


# ---------------------------------------------------------------------------
# LanguageProcessor
# ---------------------------------------------------------------------------

class LanguageProcessor:
    """
    Detects language, handles code-switching, and extracts LFS entities.

    Usage
    -----
    processor = LanguageProcessor()
    result = processor.process("I work as a nurse in Riyadh hospital")
    print(result.detected_language)   # "en"
    print(result.entities[0].label)   # "JOB_TITLE"

    # Code-switched example
    result2 = processor.process("أنا software engineer في tech company بالرياض")
    print(result2.is_code_switched)   # True
    print(result2.arabic_ratio)       # ~0.40
    """

    def __init__(self) -> None:
        self._llm = get_llm(TaskType.CRITICAL)
        self._agent = Agent(
            role="Multilingual NER Specialist",
            goal=(
                "Extract all LFS-relevant named entities from survey messages "
                "written in English, Arabic, or a mixture of both. "
                "Return results as a precise, parseable JSON array."
            ),
            backstory=(
                "You are a computational linguist with deep expertise in Arabic "
                "and English NLP. You have processed thousands of Labour Force "
                "Survey responses and excel at identifying employment-related "
                "entities — job titles, organisations, industries, locations — "
                "across both scripts, including code-switched messages."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str) -> LanguageProcessorResult:
        """
        Run language detection, code-switch analysis, and NER on one message.

        Parameters
        ----------
        text : str
            Raw survey respondent message.

        Returns
        -------
        LanguageProcessorResult
            Structured output containing language metadata and extracted entities.
        """
        text = text.strip()
        if not text:
            return LanguageProcessorResult(
                raw_text=text,
                detected_language="other",
                confidence=0.0,
                is_code_switched=False,
                arabic_ratio=0.0,
                latin_ratio=0.0,
                segments=[],
                entities=[],
            )

        detected_lang, confidence = self._detect_language(text)
        is_code_switched, segments, ar_ratio, lat_ratio = self._segment_scripts(text)
        entities = self._run_ner(text, detected_lang, is_code_switched)

        return LanguageProcessorResult(
            raw_text=text,
            detected_language=detected_lang,
            confidence=round(confidence, 4),
            is_code_switched=is_code_switched,
            arabic_ratio=round(ar_ratio, 4),
            latin_ratio=round(lat_ratio, 4),
            segments=segments,
            entities=entities,
        )

    # ------------------------------------------------------------------
    # Step 1: language detection
    # ------------------------------------------------------------------

    def _detect_language(self, text: str) -> tuple[str, float]:
        """
        Detect the primary language of *text*.

        Strategy
        --------
        1. Run langdetect.detect_langs() for probabilistic detection.
        2. If the top result is "en" or "ar", accept it directly.
        3. If the top result is something else but "en"/"ar" has a secondary
           probability ≥ 0.30, prefer that.
        4. Fall back to Unicode script ratio analysis.

        Returns
        -------
        (language_code, confidence)  e.g. ("ar", 0.9999)
        """
        try:
            predictions = detect_langs(text)
        except LangDetectException:
            return self._script_fallback(text)

        prob_map = {p.lang: p.prob for p in predictions}
        top_lang = predictions[0].lang
        top_prob = predictions[0].prob

        if top_lang in ("ar", "en"):
            return top_lang, top_prob

        # Secondary preference for Arabic or English
        for lang in ("ar", "en"):
            if prob_map.get(lang, 0.0) >= 0.30:
                return lang, prob_map[lang]

        return self._script_fallback(text)

    def _script_fallback(self, text: str) -> tuple[str, float]:
        """Infer primary language from Unicode script proportions."""
        ar = sum(len(m.group()) for m in _ARABIC_RE.finditer(text))
        lat = sum(len(m.group()) for m in _LATIN_RE.finditer(text))
        total = ar + lat or 1
        ar_ratio = ar / total
        if ar_ratio >= 0.5:
            return "ar", round(ar_ratio, 4)
        if lat > 0:
            return "en", round(1.0 - ar_ratio, 4)
        return "other", 0.5

    # ------------------------------------------------------------------
    # Step 2: code-switching and script segmentation
    # ------------------------------------------------------------------

    def _segment_scripts(
        self, text: str
    ) -> tuple[bool, list[CodeSegment], float, float]:
        """
        Compute script ratios and split *text* into script-homogeneous runs.

        Returns
        -------
        is_code_switched : bool
            True when both Arabic and Latin each exceed _CODE_SWITCH_THRESHOLD.
        segments         : list[CodeSegment]
        arabic_ratio     : float  (fraction of alphabetic chars that are Arabic)
        latin_ratio      : float  (fraction of alphabetic chars that are Latin)
        """
        ar = sum(len(m.group()) for m in _ARABIC_RE.finditer(text))
        lat = sum(len(m.group()) for m in _LATIN_RE.finditer(text))
        total = ar + lat or 1

        ar_ratio = ar / total
        lat_ratio = lat / total

        is_cs = ar_ratio >= _CODE_SWITCH_THRESHOLD and lat_ratio >= _CODE_SWITCH_THRESHOLD
        segments = self._build_segments(text)

        return is_cs, segments, ar_ratio, lat_ratio

    def _build_segments(self, text: str) -> list[CodeSegment]:
        """
        Split *text* into contiguous script runs.

        Each character is labelled "arabic", "latin", or "other"
        (whitespace / punctuation / digits). "other" characters are absorbed
        into the current run so segments don't split on spaces.
        """
        if not text:
            return []

        def _script(ch: str) -> str:
            cp = ord(ch)
            if (
                0x0600 <= cp <= 0x06FF
                or 0x0750 <= cp <= 0x077F
                or 0x08A0 <= cp <= 0x08FF
                or 0xFB50 <= cp <= 0xFDFF
                or 0xFE70 <= cp <= 0xFEFF
            ):
                return "arabic"
            if ch.isalpha():   # catches all Latin + other alpha scripts
                return "latin"
            return "other"

        # Build raw (script, text) runs
        runs: list[tuple[str, str]] = []
        cur_script = _script(text[0])
        cur_buf = text[0]

        for ch in text[1:]:
            s = _script(ch)
            if s == "other" or s == cur_script:
                cur_buf += ch
            else:
                runs.append((cur_script, cur_buf))
                cur_script = s
                cur_buf = ch
        runs.append((cur_script, cur_buf))

        # Convert to CodeSegment, adding per-segment language detection for
        # segments that are long enough to be reliable.
        segments: list[CodeSegment] = []
        for script, seg_text in runs:
            stripped = seg_text.strip()
            if not stripped:
                continue
            lang = None
            if len(stripped) >= 4 and script in ("arabic", "latin"):
                lang, _ = self._detect_language(stripped)
            segments.append(CodeSegment(
                text=seg_text,
                script=script,
                detected_language=lang,
            ))

        return segments

    # ------------------------------------------------------------------
    # Step 3: NER via CrewAI
    # ------------------------------------------------------------------

    def _run_ner(
        self,
        text: str,
        language: str,
        is_code_switched: bool,
    ) -> list[Entity]:
        """
        Ask the CrewAI NER agent to extract entities and return parsed results.
        """
        if is_code_switched:
            lang_ctx = "The message contains both Arabic and English (code-switched)."
        elif language == "ar":
            lang_ctx = "The message is written in Arabic."
        else:
            lang_ctx = "The message is written in English."

        task = Task(
            description=(
                f"{_NER_INSTRUCTIONS}\n\n"
                f"Language context: {lang_ctx}\n\n"
                f'Survey message:\n"""\n{text}\n"""'
            ),
            expected_output=(
                "A JSON array of entity objects. "
                "Keys: text (str), label (str), language (str). "
                "Return [] if no entities are present."
            ),
            agent=self._agent,
        )

        crew = Crew(agents=[self._agent], tasks=[task], verbose=False)
        raw = str(crew.kickoff()).strip()

        return self._parse_entities(raw, text)

    def _parse_entities(self, raw: str, original_text: str) -> list[Entity]:
        """
        Parse the agent's output into a list of Entity objects.

        Handles common model quirks:
        - Strips markdown code fences (```json … ```)
        - Falls back to regex extraction if strict JSON.loads fails
        - Silently drops entries with unknown labels or bad structure
        """
        # Remove markdown fences if present
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()

        data: list = []
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Try to find a JSON array anywhere in the output
            match = re.search(r"\[.*\]", clean, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if not isinstance(data, list):
            return []

        entities: list[Entity] = []
        for item in data:
            if not isinstance(item, dict):
                continue

            entity_text = str(item.get("text", "")).strip()
            label = str(item.get("label", "")).upper().strip()
            lang = str(item.get("language", "en")).lower().strip()

            if not entity_text or label not in LFS_ENTITY_LABELS:
                continue
            if lang not in ("en", "ar"):
                lang = "en"

            # Best-effort character offsets in the original text
            start = original_text.find(entity_text)
            end = (start + len(entity_text)) if start != -1 else None
            start = start if start != -1 else None

            entities.append(Entity(
                text=entity_text,
                label=label,
                language=lang,
                start=start,
                end=end,
            ))

        return entities
