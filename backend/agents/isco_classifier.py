"""
backend/agents/isco_classifier.py

Two-stage ISCO-08 occupation classifier for the LFS survey.

Pipeline
--------
Stage 1 – Semantic search (VectorStore)
    Queries Qdrant with multilingual-e5-large embeddings.
    Returns the top-5 ISCO-08 candidates by cosine similarity.
    Handles English, Arabic, and code-switched input natively.

Stage 2 – LLM re-ranking (Claude 3.5 Sonnet, TaskType.CRITICAL)
    A CrewAI agent picks the single best candidate and provides
    one-sentence reasoning.
    Skipped when the top semantic match is unambiguous (≥ 0.92).

The two-stage design keeps latency low for clear-cut titles while
preserving accuracy for ambiguous or informal descriptions.

Usage
-----
from backend.agents.isco_classifier import ISCOClassifier

clf    = ISCOClassifier()
result = clf.classify("software engineer")
result = clf.classify("مهندس برمجيات")
result = clf.classify("I fix broken pipes", context="construction sector")

print(result.primary.code)        # e.g. "2512"
print(result.primary.title_en)    # "Software Developers"
print(result.primary.confidence)  # 0.8741
print(result.method)              # "semantic" | "llm_ranked"
"""

from __future__ import annotations

import json
import re

from crewai import Agent, Crew, Task
from pydantic import BaseModel

from backend.llm import TaskType, get_llm
from backend.rag import OccupationMatch, get_vector_store


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Top semantic match at or above this score → skip LLM re-ranking.
_HIGH_CONFIDENCE_THRESHOLD = 0.92

# Below this floor the primary result should be treated as unreliable.
# Not enforced in code; exposed so callers can gate on it.
MIN_USABLE_CONFIDENCE = 0.35


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class ISCOClassification(BaseModel):
    """Structured result from ISCOClassifier.classify()."""

    query: str
    language: str                      # "en" | "ar" | "mixed" | "other"
    primary: OccupationMatch           # best-matching ISCO occupation
    candidates: list[OccupationMatch]  # all top-k from semantic search
    reasoning: str                     # explanation for primary selection
    method: str                        # "semantic" | "llm_ranked"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ISCOClassifier:
    """
    Classify a free-text job title to an ISCO-08 occupation code.

    Attributes
    ----------
    MIN_USABLE_CONFIDENCE : float
        Confidence floor below which the primary result may be unreliable.
        Callers can check ``result.primary.confidence < MIN_USABLE_CONFIDENCE``
        and ask the respondent to clarify.
    """

    MIN_USABLE_CONFIDENCE = MIN_USABLE_CONFIDENCE

    def __init__(self) -> None:
        self._store = get_vector_store()
        self._llm   = get_llm(TaskType.CRITICAL)   # Claude 3.5 Sonnet, temp 0.0
        self._agent = Agent(
            role="ISCO-08 Occupation Classification Specialist",
            goal=(
                "Select the single most accurate ISCO-08 occupation code for a "
                "given job title from a shortlist of semantic search candidates. "
                "Prefer the most specific (unit-group) code the title clearly supports."
            ),
            backstory=(
                "You are an expert in the International Standard Classification "
                "of Occupations (ISCO-08) and have classified thousands of job "
                "titles for national statistics offices. You understand formal and "
                "informal job descriptions in both English and Arabic, including "
                "code-switched text common in the Arab world's labour market."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        job_title: str,
        context: str = "",
        top_k: int = 5,
    ) -> ISCOClassification:
        """
        Classify a free-text job title to the best ISCO-08 occupation.

        Parameters
        ----------
        job_title : str
            Raw job title (English, Arabic, or code-switched).
        context : str
            Optional free-text context that aids disambiguation, e.g.
            "works in a hospital" or "construction sector, 10 years experience".
        top_k : int
            Number of semantic candidates to retrieve (default 5).

        Returns
        -------
        ISCOClassification
            Always returns a result.  Check ``primary.confidence`` against
            ``MIN_USABLE_CONFIDENCE`` if you need to gate on reliability.
        """
        job_title = job_title.strip()
        if not job_title:
            return self._empty_result(job_title)

        lang       = _detect_script(job_title)
        candidates = self._store.search(job_title, top_k=top_k)

        if not candidates:
            return self._empty_result(job_title, lang)

        # Fast path: unambiguous top match — skip LLM
        if candidates[0].confidence >= _HIGH_CONFIDENCE_THRESHOLD:
            return ISCOClassification(
                query=job_title,
                language=lang,
                primary=candidates[0],
                candidates=candidates,
                reasoning=(
                    f"Unambiguous semantic match "
                    f"(score {candidates[0].confidence:.2%})."
                ),
                method="semantic",
            )

        # Stage 2: LLM re-ranking
        primary, reasoning = self._llm_select(job_title, candidates, context, lang)
        return ISCOClassification(
            query=job_title,
            language=lang,
            primary=primary,
            candidates=candidates,
            reasoning=reasoning,
            method="llm_ranked",
        )

    # ------------------------------------------------------------------
    # Stage 2: LLM re-ranking
    # ------------------------------------------------------------------

    def _llm_select(
        self,
        job_title: str,
        candidates: list[OccupationMatch],
        context: str,
        lang: str,
    ) -> tuple[OccupationMatch, str]:
        """Ask Claude 3.5 Sonnet to pick the best candidate."""
        candidate_block = "\n".join(
            f"{i + 1}. [{c.code}] {c.title_en} / {c.title_ar}\n"
            f"   Level {c.level} | Semantic score: {c.confidence:.2%}\n"
            f"   {c.description}"
            for i, c in enumerate(candidates)
        )

        lang_note = {
            "ar":    "The job title is written in Arabic.",
            "mixed": "The job title is code-switched (Arabic and English).",
        }.get(lang, "The job title is written in English.")

        context_line = f"\nAdditional context: {context}" if context.strip() else ""

        task = Task(
            description=(
                "You are an ISCO-08 classification specialist for a national "
                "Labour Force Survey.\n\n"
                f'Job title: "{job_title}"\n'
                f"{lang_note}{context_line}\n\n"
                f"Candidates (from semantic search):\n{candidate_block}\n\n"
                "Select the single best ISCO-08 match. Prefer the most specific "
                "code (unit group over sub-major over major group) when the title "
                "clearly supports it.\n\n"
                "Return ONLY a valid JSON object — no markdown fences, no extra text:\n"
                '{"selected_code": "<isco_code>", "reasoning": "<one sentence>"}'
            ),
            expected_output=(
                'JSON: {"selected_code": "<code>", "reasoning": "<sentence>"}'
            ),
            agent=self._agent,
        )

        crew = Crew(agents=[self._agent], tasks=[task], verbose=False)
        raw  = str(crew.kickoff()).strip()

        return self._parse_llm_response(raw, candidates)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_llm_response(
        self,
        raw: str,
        candidates: list[OccupationMatch],
    ) -> tuple[OccupationMatch, str]:
        """
        Parse the LLM JSON response.

        Handles markdown fences and partial JSON.  Falls back to the top
        semantic candidate if the response cannot be parsed or contains an
        unrecognised code.
        """
        code_map = {c.code: c for c in candidates}

        # Strip markdown fences
        clean = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL
        ).strip()

        data: dict = {}
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Try to extract the first JSON object anywhere in the output
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        selected_code = str(data.get("selected_code", "")).strip()
        reasoning     = str(data.get("reasoning", "")).strip()

        if selected_code in code_map:
            return code_map[selected_code], reasoning or "Selected by LLM classifier."

        # Fallback: top semantic match
        return (
            candidates[0],
            "Fallback to top semantic match (LLM response could not be parsed).",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(job_title: str, lang: str = "other") -> ISCOClassification:
        """Return a safe sentinel when no candidates are available."""
        placeholder = OccupationMatch(
            code="",
            title_en="Unknown",
            title_ar="غير معروف",
            level=0,
            description="No matching ISCO-08 occupation found.",
            confidence=0.0,
        )
        return ISCOClassification(
            query=job_title,
            language=lang,
            primary=placeholder,
            candidates=[],
            reasoning="No candidates returned by the vector store.",
            method="semantic",
        )


# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------

_ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+"
)
_LATIN_RE = re.compile(r"[A-Za-z]+")


def _detect_script(text: str) -> str:
    """
    Return the dominant script of *text*.

    Returns
    -------
    "ar"    ≥ 90 % Arabic characters
    "en"    ≤ 10 % Arabic characters (mostly Latin)
    "mixed" between 10 % and 90 % Arabic
    "other" no alphabetic characters detected
    """
    ar    = sum(len(m.group()) for m in _ARABIC_RE.finditer(text))
    lat   = sum(len(m.group()) for m in _LATIN_RE.finditer(text))
    total = ar + lat
    if total == 0:
        return "other"
    ar_ratio = ar / total
    if ar_ratio >= 0.90:
        return "ar"
    if ar_ratio <= 0.10:
        return "en"
    return "mixed"
