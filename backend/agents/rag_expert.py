"""
backend/agents/rag_expert.py

RAG Expert agent for hierarchical ISCO-08 occupation retrieval.

Pipeline
--------
Stage 1 – Broad semantic search (VectorStore)
    Queries Qdrant with multilingual-e5-large embeddings and retrieves the top
    2×top_k candidates.  Handles English, Arabic, and code-switched input.

Stage 2 – Hierarchical expansion
    For every unit-group (level-4) result the parent sub-major group (level-2)
    and major group (level-1) are resolved from the ISCO-08 code map.
    Parents not already in the candidate list are added with a discounted
    confidence score, ensuring the response covers the full occupational
    hierarchy from broad to specific.

    Confidence discounts:
        sub-major parent  → unit-group confidence × 0.85
        major-group parent → unit-group confidence × 0.70

Stage 3 – Bilingual explanation (Claude 3.5 Sonnet via CrewAI)
    A single CrewAI task asks the LLM to produce one-sentence match
    explanations in both English and Arabic for each of the top-5 candidates,
    making the results directly usable in a bilingual LFS interview.

Usage
-----
from backend.agents.rag_expert import RAGExpert

expert = RAGExpert()

# English query
result = expert.retrieve("software engineer")

# Arabic query
result = expert.retrieve("مهندس برمجيات")

# Code-switched query with context
result = expert.retrieve(
    "أنا software developer بالبنك",
    context="financial sector, 5 years experience",
)

for cand in result.candidates:
    print(f"#{cand.rank}  [{cand.code}] {cand.title_en}  ({cand.confidence:.0%})")
    print(f"  Hierarchy: {cand.hierarchy.major_title_en}"
          + (f" → {cand.hierarchy.sub_major_title_en}" if cand.hierarchy.sub_major_code else ""))
    print(f"  EN: {cand.explanation_en}")
    print(f"  AR: {cand.explanation_ar}")
"""

from __future__ import annotations

import json
import re
from typing import Optional

from crewai import Agent, Crew, Task
from pydantic import BaseModel, Field

from backend.llm import TaskType, get_llm
from backend.rag import OccupationMatch, get_vector_store
from backend.rag.vector_store import _ISCO_DATA

# ---------------------------------------------------------------------------
# ISCO-08 code map — O(1) lookup for hierarchy traversal
# ---------------------------------------------------------------------------

_CODE_MAP: dict[str, dict] = {entry["code"]: entry for entry in _ISCO_DATA}


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class HierarchyInfo(BaseModel):
    """ISCO-08 hierarchy context for an occupation match."""

    major_code: str
    major_title_en: str
    major_title_ar: str
    sub_major_code: Optional[str] = None
    sub_major_title_en: Optional[str] = None
    sub_major_title_ar: Optional[str] = None


class OccupationCandidate(BaseModel):
    """
    A single ranked ISCO-08 candidate with hierarchical context and
    bilingual explanations.
    """

    rank: int                              # 1 = best match, 5 = least confident
    code: str
    title_en: str
    title_ar: str
    level: int                             # 1 = major, 2 = sub-major, 4 = unit
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    hierarchy: HierarchyInfo               # parent group path
    explanation_en: str                    # why this matches (English)
    explanation_ar: str                    # why this matches (Arabic / عربي)
    retrieval_stage: str                   # "semantic" | "hierarchical_expansion"


class RAGExpertResult(BaseModel):
    """Structured result returned by RAGExpert.retrieve()."""

    query: str
    language: str                          # "en" | "ar" | "mixed" | "other"
    candidates: list[OccupationCandidate]  # top 5, ranked 1 … 5
    total_retrieved: int                   # raw candidates before expansion
    method: str                            # "hierarchical" | "semantic_only"


# ---------------------------------------------------------------------------
# LLM prompt constants
# ---------------------------------------------------------------------------

_EXPLANATION_INSTRUCTIONS = """\
You are an ISCO-08 occupational classification expert for a bilingual \
(English / Arabic) Labour Force Survey.

Your task: for each occupation candidate listed below, write a \
single-sentence explanation in BOTH English and Arabic that describes why \
the occupation code is (or may be) relevant to the survey respondent's query.

Rules:
- Be specific — mention the query text and the occupation title.
- Use plain, non-technical language suitable for survey respondents.
- Arabic sentences must be grammatically correct Modern Standard Arabic.
- Return ONLY a valid JSON array — no markdown fences, no extra text.
- Array element format:
    {"code": "<isco_code>", "explanation_en": "<sentence>", "explanation_ar": "<sentence>"}
"""


# ---------------------------------------------------------------------------
# RAGExpert
# ---------------------------------------------------------------------------

class RAGExpert:
    """
    Hierarchical ISCO-08 occupation retrieval agent.

    Three-stage pipeline:
      1. Broad semantic search via VectorStore (multilingual-e5-large / Qdrant).
      2. Hierarchical expansion — parent major/sub-major groups are added for
         every unit-group hit that lacks an explicit parent in the result set.
      3. Bilingual explanation generation by Claude 3.5 Sonnet (CrewAI).

    The agent always returns up to five ranked `OccupationCandidate` objects,
    each carrying the full ISCO-08 hierarchy path and match explanations in
    English and Arabic.
    """

    def __init__(self) -> None:
        self._store = get_vector_store()
        self._llm   = get_llm(TaskType.CRITICAL)   # Claude 3.5 Sonnet, temp 0.0
        self._agent = Agent(
            role="ISCO-08 Hierarchical Classification Expert",
            goal=(
                "Analyse occupation candidates from an ISCO-08 hierarchical search "
                "and generate clear, accurate bilingual explanations in English and "
                "Arabic for each match, helping Labour Force Survey respondents "
                "understand why each occupation code applies to their job."
            ),
            backstory=(
                "You are a bilingual occupational classification expert with deep "
                "knowledge of the ISCO-08 International Standard Classification of "
                "Occupations. You work for a national statistics office conducting "
                "Labour Force Surveys in both English and Arabic. You excel at "
                "explaining occupational classifications in plain language, including "
                "for respondents who mix Arabic and English in their answers."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        context: str = "",
        top_k: int = 5,
    ) -> RAGExpertResult:
        """
        Retrieve the top-k ISCO-08 occupation candidates for a free-text query.

        Parameters
        ----------
        query : str
            Job title or description in English, Arabic, or code-switched text.
        context : str
            Optional extra context that aids disambiguation (e.g., sector,
            years of experience, employment type).
        top_k : int
            Number of ranked candidates to return (default 5, max 10).

        Returns
        -------
        RAGExpertResult
            Always returns a result object; `candidates` is empty only when
            the vector store returns no hits at all.
        """
        query = query.strip()
        if not query:
            return RAGExpertResult(
                query=query,
                language="other",
                candidates=[],
                total_retrieved=0,
                method="semantic_only",
            )

        top_k = max(1, min(top_k, 10))
        lang  = _detect_script(query)

        # Stage 1 — broad semantic search
        raw: list[OccupationMatch] = self._store.search(query, top_k=top_k * 2)
        total_retrieved = len(raw)

        if not raw:
            return RAGExpertResult(
                query=query,
                language=lang,
                candidates=[],
                total_retrieved=0,
                method="semantic_only",
            )

        # Stage 2 — hierarchical expansion
        enriched = self._expand_hierarchy(raw, top_k)

        has_expansion = any(stage == "hierarchical_expansion" for _, _, stage in enriched)
        method = "hierarchical" if has_expansion else "semantic_only"

        # Stage 3 — bilingual explanations via CrewAI
        candidates = self._generate_explanations(query, enriched, context, lang)

        return RAGExpertResult(
            query=query,
            language=lang,
            candidates=candidates,
            total_retrieved=total_retrieved,
            method=method,
        )

    # ------------------------------------------------------------------
    # Stage 2: hierarchical expansion
    # ------------------------------------------------------------------

    def _expand_hierarchy(
        self,
        raw: list[OccupationMatch],
        top_k: int,
    ) -> list[tuple[OccupationMatch, HierarchyInfo, str]]:
        """
        Enrich the semantic search results with parent-group context.

        For every unit-group (level-4) match in *raw* whose sub-major or
        major parent is not already represented, a synthetic parent entry is
        added with a discounted confidence score.

        Returns
        -------
        list of (OccupationMatch, HierarchyInfo, retrieval_stage) triples,
        sorted by confidence descending, capped at *top_k*.
        """
        seen: set[str] = set()
        enriched: list[tuple[OccupationMatch, HierarchyInfo, str]] = []

        # Add all raw semantic results
        for match in raw:
            if match.code in seen:
                continue
            seen.add(match.code)
            hier = self._build_hierarchy_info(match.code, match.level)
            enriched.append((match, hier, "semantic"))

        # Add missing parent groups for unit-group results
        expansions: list[tuple[OccupationMatch, HierarchyInfo, str]] = []
        for match, _, _ in enriched:
            if match.level != 4:
                continue

            # Sub-major parent (2-digit prefix, discount × 0.85)
            sub_code = match.code[:2]
            if sub_code not in seen and sub_code in _CODE_MAP:
                entry  = _CODE_MAP[sub_code]
                parent = OccupationMatch(
                    code=entry["code"],
                    title_en=entry["title_en"],
                    title_ar=entry["title_ar"],
                    level=entry["level"],
                    description=entry["description"],
                    confidence=round(match.confidence * 0.85, 4),
                )
                expansions.append(
                    (parent, self._build_hierarchy_info(sub_code, 2), "hierarchical_expansion")
                )
                seen.add(sub_code)

            # Major-group parent (1-digit prefix, discount × 0.70)
            maj_code = match.code[:1]
            if maj_code not in seen and maj_code in _CODE_MAP:
                entry  = _CODE_MAP[maj_code]
                parent = OccupationMatch(
                    code=entry["code"],
                    title_en=entry["title_en"],
                    title_ar=entry["title_ar"],
                    level=entry["level"],
                    description=entry["description"],
                    confidence=round(match.confidence * 0.70, 4),
                )
                expansions.append(
                    (parent, self._build_hierarchy_info(maj_code, 1), "hierarchical_expansion")
                )
                seen.add(maj_code)

        # Merge: semantic results first (higher confidence), then expansions
        all_candidates = enriched + expansions
        all_candidates.sort(key=lambda t: t[0].confidence, reverse=True)
        return all_candidates[:top_k]

    def _build_hierarchy_info(self, code: str, level: int) -> HierarchyInfo:
        """
        Resolve the ISCO-08 hierarchy path for a given code.

        Level 1  → only major group populated.
        Level 2  → major group + sub-major group populated.
        Level 4  → major group + sub-major group populated (minor/unit level).
        """
        _unknown_en = "Unknown"
        _unknown_ar = "غير معروف"

        if level == 1:
            e = _CODE_MAP.get(code, {})
            return HierarchyInfo(
                major_code=code,
                major_title_en=e.get("title_en", _unknown_en),
                major_title_ar=e.get("title_ar", _unknown_ar),
            )

        if level == 2:
            maj = _CODE_MAP.get(code[:1], {})
            sub = _CODE_MAP.get(code, {})
            return HierarchyInfo(
                major_code=code[:1],
                major_title_en=maj.get("title_en", _unknown_en),
                major_title_ar=maj.get("title_ar", _unknown_ar),
                sub_major_code=code,
                sub_major_title_en=sub.get("title_en", _unknown_en),
                sub_major_title_ar=sub.get("title_ar", _unknown_ar),
            )

        # level 4 (unit group)
        maj_code = code[:1]
        sub_code = code[:2]
        maj = _CODE_MAP.get(maj_code, {})
        sub = _CODE_MAP.get(sub_code, {})
        return HierarchyInfo(
            major_code=maj_code,
            major_title_en=maj.get("title_en", _unknown_en),
            major_title_ar=maj.get("title_ar", _unknown_ar),
            sub_major_code=sub_code,
            sub_major_title_en=sub.get("title_en", _unknown_en),
            sub_major_title_ar=sub.get("title_ar", _unknown_ar),
        )

    # ------------------------------------------------------------------
    # Stage 3: bilingual explanations via CrewAI
    # ------------------------------------------------------------------

    def _generate_explanations(
        self,
        query: str,
        enriched: list[tuple[OccupationMatch, HierarchyInfo, str]],
        context: str,
        lang: str,
    ) -> list[OccupationCandidate]:
        """
        Ask Claude 3.5 Sonnet to produce bilingual match explanations for
        each candidate, then merge with candidate metadata.
        """
        lang_note = {
            "ar":    "The query is written in Arabic.",
            "mixed": "The query is code-switched (Arabic and English mixed).",
        }.get(lang, "The query is written in English.")

        context_line = f"\nAdditional context: {context}" if context.strip() else ""

        candidate_block = "\n".join(
            f"{i + 1}. [{c.code}] {c.title_en} / {c.title_ar}\n"
            f"   Level {c.level} | Confidence {c.confidence:.2%}\n"
            f"   {c.description}\n"
            f"   Hierarchy: {h.major_title_en}"
            + (f" → {h.sub_major_title_en}" if h.sub_major_code else "")
            for i, (c, h, _) in enumerate(enriched)
        )

        task = Task(
            description=(
                f"{_EXPLANATION_INSTRUCTIONS}\n\n"
                f'Query: "{query}"\n'
                f"{lang_note}{context_line}\n\n"
                f"Candidates:\n{candidate_block}"
            ),
            expected_output=(
                'JSON array: [{"code": "...", "explanation_en": "...", '
                '"explanation_ar": "..."}, ...]'
            ),
            agent=self._agent,
        )

        crew     = Crew(agents=[self._agent], tasks=[task], verbose=False)
        raw_out  = str(crew.kickoff()).strip()

        return self._parse_explanations(raw_out, enriched)

    def _parse_explanations(
        self,
        raw: str,
        enriched: list[tuple[OccupationMatch, HierarchyInfo, str]],
    ) -> list[OccupationCandidate]:
        """
        Parse the LLM JSON output and merge with candidate metadata.

        Falls back to auto-generated explanations for any candidate whose
        code is missing from (or malformed in) the LLM response.
        """
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()

        data: list = []
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", clean, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        # Build code → (explanation_en, explanation_ar) map
        exp_map: dict[str, tuple[str, str]] = {}
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                code   = str(item.get("code", "")).strip()
                en_exp = str(item.get("explanation_en", "")).strip()
                ar_exp = str(item.get("explanation_ar", "")).strip()
                if code and en_exp:
                    exp_map[code] = (en_exp, ar_exp)

        results: list[OccupationCandidate] = []
        for rank, (match, hier, stage) in enumerate(enriched, start=1):
            fallback_en = (
                f"Matches '{match.title_en}' based on semantic similarity to the query."
            )
            fallback_ar = (
                f"يطابق '{match.title_ar}' بناءً على التشابه الدلالي مع الاستعلام."
            )
            exp_en, exp_ar = exp_map.get(match.code, (fallback_en, fallback_ar))

            results.append(OccupationCandidate(
                rank=rank,
                code=match.code,
                title_en=match.title_en,
                title_ar=match.title_ar,
                level=match.level,
                description=match.description,
                confidence=match.confidence,
                hierarchy=hier,
                explanation_en=exp_en,
                explanation_ar=exp_ar,
                retrieval_stage=stage,
            ))

        return results


# ---------------------------------------------------------------------------
# Script detection (mirrors isco_classifier._detect_script)
# ---------------------------------------------------------------------------

_ARABIC_CHARS = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+"
)
_LATIN_CHARS = re.compile(r"[A-Za-z]+")


def _detect_script(text: str) -> str:
    """
    Return the dominant script of *text*.

    Returns
    -------
    "ar"    if ≥ 90 % of alphabetic characters are Arabic
    "en"    if ≤ 10 % of alphabetic characters are Arabic
    "mixed" if between 10 % and 90 % Arabic
    "other" if no alphabetic characters detected
    """
    ar    = sum(len(m.group()) for m in _ARABIC_CHARS.finditer(text))
    lat   = sum(len(m.group()) for m in _LATIN_CHARS.finditer(text))
    total = ar + lat
    if total == 0:
        return "other"
    ratio = ar / total
    if ratio >= 0.90:
        return "ar"
    if ratio <= 0.10:
        return "en"
    return "mixed"
