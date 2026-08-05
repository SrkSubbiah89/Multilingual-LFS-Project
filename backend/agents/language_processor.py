"""
backend/agents/language_processor.py

CrewAI-based Language Processor agent for the LFS survey.

Responsibilities
----------------
1. Language detection   – identifies English (en), Modern Standard Arabic (ar),
                          Gulf Arabic (ar-gulf), Urdu (ur), Hindi (hi), Tagalog
                          (tl), or code-switched messages.
2. Gulf Arabic normalisation – maps ~30 common Gulf dialect tokens to their MSA
                               equivalents before passing text downstream.
3. Code-switching       – segments the text into contiguous script runs and
                          exposes per-segment language labels.
4. Named Entity Recognition – extracts LFS-relevant entities (job titles,
                               organisations, locations, industry sectors,
                               employment status, durations, hours) using a
                               CrewAI agent that returns strict JSON.
5. Structured output    – all results are returned in a validated Pydantic model.

Supported languages : en, ar, ar-gulf, ur, hi, tl, other
Scripts detected    : Arabic (Unicode 0600-06FF + extended), Devanagari
                      (0900-097F), Latin
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from crewai import Agent, Crew, Task
from langdetect import DetectorFactory, LangDetectException, detect_langs
from pydantic import BaseModel, Field

from backend.llm import TaskType, get_llm

_logger = logging.getLogger(__name__)
_APP_ENV = os.getenv("APP_ENV", "development").lower()

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
# Devanagari block (used by Hindi; Urdu uses Arabic script)
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")
_LATIN_RE = re.compile(r"[A-Za-z]+")

# Both scripts must each contribute at least this fraction of alphabetic
# characters to be considered code-switched.
_CODE_SWITCH_THRESHOLD = 0.10

# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------

# Languages this processor can detect and return.
SUPPORTED_LANGUAGES: set[str] = {"en", "ar", "ar-gulf", "ur", "hi", "tl", "other"}

# langdetect codes → our canonical language codes
_LANG_MAP: dict[str, str] = {
    "en": "en",
    "ar": "ar",
    "ur": "ur",
    "hi": "hi",
    "tl": "tl",
}

# ---------------------------------------------------------------------------
# Gulf Arabic lexical markers + MSA normalization dictionary
# ---------------------------------------------------------------------------

# If ≥ GULF_MARKER_THRESHOLD fraction of detected Arabic words are Gulf
# dialect markers, we label the language "ar-gulf".
_GULF_MARKER_THRESHOLD = 0.10

# Gulf dialect word → MSA equivalent (used to normalise before LLM/NER)
_GULF_NORMALISE: dict[str, str] = {
    # work / employment
    "شغل":    "عمل",       # shaghl  → ʿamal   (work)
    "شغلة":   "وظيفة",     # shaghlah → wazifa  (job)
    "شاغل":   "عامل",      # shāghil → ʿāmil   (worker)
    "اشتغل":  "عمل",       # ishtghal → ʿamala  (worked)
    # quantity / degree
    "وايد":   "كثير",      # wāyid → kathīr    (a lot)
    "واجد":   "كثير",      # wājid → kathīr
    "ذرة":    "قليل",      # dhurra → qalīl    (a little)
    # time
    "الحين":  "الآن",      # al-ḥīn → al-ān    (now)
    "عقب":    "بعد",       # ʿugub → baʿd      (after)
    "بعدين":  "بعد ذلك",   # baʿdayn → later
    # place
    "هني":    "هنا",       # hini → hunā        (here)
    "هناك":   "هناك",
    # existence
    "ماكو":   "لا يوجد",   # māku → lā yūjad   (there is not)
    "أكو":    "يوجد",      # āku → yūjad        (there is)
    # money / salary
    "قديش":   "كم",        # gadaysh → kam      (how much)
    "كاش":    "نقدي",      # cash → naqdī
    "راتب":   "راتب",
    # status / condition
    "زين":    "جيد",       # zayn → jayyid      (good/ok)
    "عادل":   "مقبول",     # ʿādil → maqbūl     (acceptable)
    "صج":     "صحيح",      # ṣaj → ṣaḥīḥ        (true/right)
    "مو":     "ليس",       # mu → laysa          (not)
    "مب":     "ليس",       # mub → laysa
    # direction / instruction
    "شوف":    "انظر",      # shūf → unẓur       (look)
    "خل":     "دع",        # khall → daʿ         (let)
    # question words
    "اشلون":  "كيف",       # ashlūn → kayfa      (how)
    "شنو":    "ماذا",      # shinu → mādhā       (what)
    "شبيك":   "ما بك",     # shibīk → what's wrong with you
    "ليش":    "لماذا",     # laysh → limādhā     (why)
    # possession
    "مال":    "خاص بـ",    # māl → belonging to
    # conjunctions / particles
    "بس":     "فقط",       # bas → faqat         (just/only)
    "يعني":   "أي",        # yaʿni → ay          (meaning/i.e.)
    # ── Additional Gulf / Arabian Peninsula markers ──────────────────────
    # occupations / workplace (Saudi, UAE, Kuwaiti, Bahraini, Omani)
    "مراح":   "لن أذهب",  # marāḥ → lan adhhab   (I won't go; negated volitive)
    "ودي":    "أريد",      # widdī → urīdu        (I want)
    "أبي":    "أريد",      # abī → urīdu          (I want; Saudi)
    "ابغى":   "أريد",      # abghā → urīdu        (I want; Gulf)
    "بغيت":   "أردت",      # baghīt → aradt       (I wanted)
    "مابي":   "لا أريد",   # mābi → lā urīdu      (I don't want)
    "مودي":   "لا أريد",   # mōdi → lā urīdu
    "حق":     "لـ / عند",  # ḥagg → li/ʿinda      (for/at — possessive marker)
    "حقي":    "لي",        # ḥaggī → lī           (mine)
    "حقه":    "له",        # ḥaggah → lahu         (his)
    "حقها":   "لها",       # ḥaggahā → lahā        (hers)
    "تبي":    "تريد",      # tibī → turīdu         (you want / she wants)
    "يبي":    "يريد",      # yibī → yurīdu         (he wants)
    "دشداشة": "ثوب",       # dishdāsha → thawb    (traditional robe — cultural ref)
    "فيلا":   "فيلا",      # villa → villa         (unchanged, loanword)
    # time / frequency
    "دايم":   "دائماً",    # dāyim → dāʾiman      (always)
    "دوم":    "دائماً",    # dōm → dāʾiman
    "هالحين": "الآن",      # hāl-ḥīn → al-ān      (right now; variant)
    "توه":    "للتو",      # tawwah → lil-taw      (just now)
    "بكير":   "مبكراً",    # bakīr → mubakkiran   (early)
    "وقتين":  "مرتين",     # waqtayn → marratyn   (twice / two times)
    "زمان":   "منذ فترة",  # zamān → mundhu fatra  (a long time ago)
    # negation / confirmation
    "لا والله": "لا",       # lā wallah → no (emphatic denial)
    "أيوه":   "نعم",        # aywa → naʿam          (yes; Egyptian-Gulf shared)
    "ايه":    "نعم",        # ay → naʿam             (yes; variant)
    "مو صح":  "غير صحيح",  # mu ṣaḥ → ghayr ṣaḥīḥ  (not right)
    "ماصح":   "غير صحيح",
    "ماعدل":  "غير مقبول",
    # modal / conditional
    "لو":     "إذا",        # law → idhā              (if)
    "خوش":    "جيد",        # khōsh → jayyid          (good; Gulf Arabized Persian)
    "خوشة":   "جيدة",
    "ما عدل": "غير مقبول",  # mā ʿadal → unacceptable
    # workplace / occupation related (Gulf-specific)
    "كفيل":   "كفيل",       # kafīl → sponsor (kafala system, keep as-is)
    "إقامة":  "إقامة",      # iqāma → residence permit (keep as-is)
    "بدل":    "بدل",        # badal → allowance (keep as-is)
    "راس المال": "رأس المال", # normalise hamza
    "دوام":   "دوام",       # dawām → working hours / shift (already MSA but common Gulf usage)
    "استراحة": "استراحة",
    # quantity / comparative
    "أكثر شي": "أكثر شيء",  # most thing → most of all
    "أهون":   "أسهل",       # ahwan → ashal            (easier; Gulf comparative)
    "ثقيل":   "صعب",        # thaqīl → ṣaʿb            (heavy → difficult; fig.)
    "خفيف":   "سهل",        # khafīf → sahl             (light → easy; fig.)
    # location / direction
    "البر":   "البر / الخارج", # al-barr → outside/countryside
    "سوق":    "سوق",
    "المول":  "المجمع التجاري", # mall → shopping centre
    # contract / legal
    "عقد":    "عقد",         # ʿaqd → contract (same in MSA)
    "أجرة":   "أجر",         # ujra → ajr (wage/fee; variant spelling)
    "أجور":   "أجور",
}

# Flat set of Gulf markers for fast membership testing
_GULF_MARKERS: frozenset[str] = frozenset(_GULF_NORMALISE.keys())


# ---------------------------------------------------------------------------
# Tagalog (Filipino) occupation keyword dictionary
# Maps common TL occupation / work-related tokens to their English equivalents
# so the LLM NER prompt receives normalised text.
# Coverage targets the UAE labour force: domestic helpers, construction,
# healthcare aides, retail, drivers, service workers.
# ---------------------------------------------------------------------------
_TL_NORMALISE: dict[str, str] = {
    # occupations
    "guro":          "teacher",
    "titser":        "teacher",
    "nars":          "nurse",
    "doktor":        "doctor",
    "manggagamot":   "doctor",
    "abogado":       "lawyer",
    "inhinyero":     "engineer",
    "arkitekto":     "architect",
    "accountant":    "accountant",
    "drayber":       "driver",
    "driver":        "driver",
    "karpintero":    "carpenter",
    "plomero":       "plumber",
    "electrician":   "electrician",
    "kusinero":      "cook",
    "chef":          "chef",
    "waiter":        "waiter",
    "waitress":      "waitress",
    "cashier":       "cashier",
    "security":      "security guard",
    "guard":         "security guard",
    "bantay":        "security guard",
    "katulong":      "domestic helper",
    "kasambahay":    "domestic helper",
    "yaya":          "domestic helper",
    "maglalaba":     "laundry worker",
    "manglalaba":    "laundry worker",
    "tagapaglinis":  "cleaner",
    "janitor":       "janitor",
    "sales":         "sales worker",
    "tindera":       "sales worker",
    "tindero":       "sales worker",
    "OFW":           "overseas worker",
    "manggagawa":    "worker",
    "trabahador":    "worker",
    "empleyado":     "employee",
    "magsasaka":     "farmer",
    "mangingisda":   "fisherman",
    "mekaniko":      "mechanic",
    "welder":        "welder",
    "mason":         "mason",
    "construction":  "construction worker",
    "bodega":        "warehouse worker",
    "delivery":      "delivery worker",
    "messenger":     "messenger",
    "receptionist":  "receptionist",
    "secretary":     "secretary",
    "manager":       "manager",
    "supervisor":    "supervisor",
    "negosyante":    "businessman",
    "sariling negosyo": "self-employed",
    # employment status
    "employed":      "employed",
    "nawalan ng trabaho": "unemployed",
    "walang trabaho":    "unemployed",
    "naghahanap ng trabaho": "job seeking",
    "part-time":     "part-time",
    "full-time":     "full-time",
    "kontrata":      "contract",
    # sectors
    "ospital":       "hospital",
    "paaralan":      "school",
    "restaurant":    "restaurant",
    "kumpanya":      "company",
    "gobyerno":      "government",
    "pribado":       "private sector",
}

_TL_MARKERS: frozenset[str] = frozenset(_TL_NORMALISE.keys())


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
    language: str                          # "en"|"ar"|"ar-gulf"|"ur"|"hi"|"tl"
    start: Optional[int] = None            # character offset in original text
    end: Optional[int] = None


class LanguageProcessorResult(BaseModel):
    """Full structured result returned by LanguageProcessor.process()."""

    raw_text: str
    detected_language: str   # "en"|"ar"|"ar-gulf"|"ur"|"hi"|"tl"|"other"
    confidence: float = Field(ge=0.0, le=1.0)
    is_code_switched: bool
    arabic_ratio: float = Field(ge=0.0, le=1.0)
    latin_ratio: float = Field(ge=0.0, le=1.0)
    devanagari_ratio: float = Field(ge=0.0, le=1.0, default=0.0)
    segments: list[CodeSegment]
    entities: list[Entity]
    normalised_text: Optional[str] = None  # Gulf-Arabic-normalised version


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
    "language" : one of "en", "ar", "ar-gulf", "ur", "hi", "tl"
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
        self._agent_available = False
        try:
            self._llm = get_llm(TaskType.GENERAL)
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
            self._agent_available = True
        except Exception as exc:
            _logger.warning(
                "LanguageProcessor: no LLM available for NER (%s). "
                "Language detection will still work; NER will return empty entities.",
                exc,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str) -> LanguageProcessorResult:
        """
        Run language detection, Gulf normalisation, code-switch analysis, and
        NER on one survey message.

        Supported languages: en, ar, ar-gulf, ur, hi, tl, other.
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
                devanagari_ratio=0.0,
                segments=[],
                entities=[],
            )

        detected_lang, confidence = self._detect_language(text)

        # Gulf Arabic normalisation: produce a cleaned copy for NER / downstream
        normalised: Optional[str] = None
        if detected_lang in ("ar", "ar-gulf"):
            normalised = self._normalise_gulf_arabic(text)

        if detected_lang == "tl":
            normalised = self._normalise_tagalog(text)

        is_code_switched, segments, ar_ratio, lat_ratio, dev_ratio = (
            self._segment_scripts(text)
        )
        # NER runs on normalised text when available so the LLM sees MSA tokens
        ner_text = normalised if normalised else text
        entities = self._run_ner(ner_text, detected_lang, is_code_switched)

        return LanguageProcessorResult(
            raw_text=text,
            detected_language=detected_lang,
            confidence=round(confidence, 4),
            is_code_switched=is_code_switched,
            arabic_ratio=round(ar_ratio, 4),
            latin_ratio=round(lat_ratio, 4),
            devanagari_ratio=round(dev_ratio, 4),
            segments=segments,
            entities=entities,
            normalised_text=normalised,
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
        2. Accept en, ar, ur, hi, tl directly if top result.
        3. Secondary preference for any supported lang with prob ≥ 0.30.
        4. Fall back to Unicode script ratio analysis.
        5. Post-process Arabic to "ar-gulf" when Gulf markers are present.

        Returns
        -------
        (language_code, confidence)  e.g. ("ar-gulf", 0.92)
        """
        # --- Devanagari fast-path (langdetect sometimes misses pure Hindi) ---
        dev_chars = sum(len(m.group()) for m in _DEVANAGARI_RE.finditer(text))
        total_alpha = (
            dev_chars
            + sum(len(m.group()) for m in _ARABIC_RE.finditer(text))
            + sum(len(m.group()) for m in _LATIN_RE.finditer(text))
        ) or 1
        if dev_chars / total_alpha >= 0.50:
            return "hi", round(dev_chars / total_alpha, 4)

        try:
            predictions = detect_langs(text)
        except LangDetectException:
            return self._script_fallback(text)

        prob_map = {p.lang: p.prob for p in predictions}
        top_lang = predictions[0].lang
        top_prob = predictions[0].prob

        # Accept known supported languages from langdetect directly
        if top_lang in _LANG_MAP:
            lang = _LANG_MAP[top_lang]
            return self._apply_gulf_detection(lang, text), top_prob

        # Secondary preference for any supported language
        for ld_code, our_code in _LANG_MAP.items():
            if prob_map.get(ld_code, 0.0) >= 0.30:
                return self._apply_gulf_detection(our_code, text), prob_map[ld_code]

        return self._script_fallback(text)

    def _apply_gulf_detection(self, lang: str, text: str) -> str:
        """Upgrade 'ar' to 'ar-gulf' when Gulf lexical markers are present."""
        if lang != "ar":
            return lang
        tokens = set(re.findall(r"\w+", text, re.UNICODE))
        gulf_hits = tokens & _GULF_MARKERS
        if gulf_hits and len(gulf_hits) / max(len(tokens), 1) >= _GULF_MARKER_THRESHOLD:
            return "ar-gulf"
        return lang

    def _normalise_gulf_arabic(self, text: str) -> str:
        """
        Replace Gulf dialect tokens with MSA equivalents.

        Only whole-word replacements are performed (regex word boundary).
        Returns the original text unchanged when no Gulf tokens are found.
        """
        result = text
        changed = False
        for dialect, msa in _GULF_NORMALISE.items():
            pattern = rf"(?<!\w){re.escape(dialect)}(?!\w)"
            new_text = re.sub(pattern, msa, result)
            if new_text != result:
                changed = True
                result = new_text
        return result if changed else text

    @staticmethod
    def _normalise_tagalog(text: str) -> str:
        """Replace known Tagalog occupation tokens with English equivalents."""
        # Multi-word first (longest match)
        for tl, en in sorted(_TL_NORMALISE.items(), key=lambda x: -len(x[0])):
            if " " in tl and tl.lower() in text.lower():
                text = re.sub(re.escape(tl), en, text, flags=re.IGNORECASE)
        # Single token pass
        tokens = text.split()
        return " ".join(_TL_NORMALISE.get(tok.lower(), tok) for tok in tokens)

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
    ) -> tuple[bool, list[CodeSegment], float, float, float]:
        """
        Compute script ratios and split *text* into script-homogeneous runs.

        Returns
        -------
        is_code_switched : bool
            True when both Arabic and Latin each exceed _CODE_SWITCH_THRESHOLD.
        segments         : list[CodeSegment]
        arabic_ratio     : float
        latin_ratio      : float
        devanagari_ratio : float
        """
        ar  = sum(len(m.group()) for m in _ARABIC_RE.finditer(text))
        dev = sum(len(m.group()) for m in _DEVANAGARI_RE.finditer(text))
        lat = sum(len(m.group()) for m in _LATIN_RE.finditer(text))
        total = ar + dev + lat or 1

        ar_ratio  = ar  / total
        lat_ratio = lat / total
        dev_ratio = dev / total

        is_cs = (
            (ar_ratio  >= _CODE_SWITCH_THRESHOLD and lat_ratio >= _CODE_SWITCH_THRESHOLD)
            or (dev_ratio >= _CODE_SWITCH_THRESHOLD and lat_ratio >= _CODE_SWITCH_THRESHOLD)
        )
        segments = self._build_segments(text)

        return is_cs, segments, ar_ratio, lat_ratio, dev_ratio

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
            if 0x0900 <= cp <= 0x097F:
                return "devanagari"
            if ch.isalpha():   # catches Latin + other alpha scripts
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
        Returns an empty list if the agent is unavailable or the call fails.
        """
        if not self._agent_available:
            return []

        _LANG_LABELS = {
            "en": "English",
            "ar": "Modern Standard Arabic",
            "ar-gulf": "Gulf Arabic (dialect)",
            "ur": "Urdu",
            "hi": "Hindi",
            "tl": "Tagalog (Filipino)",
        }
        if is_code_switched:
            lang_ctx = "The message contains mixed scripts (code-switched)."
        else:
            label = _LANG_LABELS.get(language, language.upper())
            lang_ctx = f"The message is written in {label}."

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

        try:
            crew = Crew(agents=[self._agent], tasks=[task], verbose=False)
            raw = str(crew.kickoff()).strip()
            return self._parse_entities(raw, text)
        except Exception as exc:
            _logger.warning("NER agent call failed: %s. Returning empty entities.", exc)
            return []

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
            if lang not in SUPPORTED_LANGUAGES or lang == "other":
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
