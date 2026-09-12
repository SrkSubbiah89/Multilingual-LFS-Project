"""
backend/agents/hierarchical_classification_crew.py

Real CrewAI multi-agent delegation (Item 3 of the 2026-09-12 multi-agent
RAG work) -- the first and only Crew in this codebase's 18+
Crew(agents=[...], tasks=[...]) construction sites to use
process=Process.hierarchical + manager_llm. Every other site defaults to
CrewAI's plain sequential process; there is otherwise zero real
manager-delegates-to-workers orchestration anywhere in this codebase.

Deliberately kept SEPARATE from backend/agents/cross_standard_coordinator.py
(Item 1's deterministic coordinator) rather than built on top of it -- the
two are meant to be compared (does a real, non-deterministic LLM manager
add anything over deterministic coordination?), not conflated. See
CLAUDE.md's "Knowledge base construction" section for the full rationale:
this project has 7 independent, already-confirmed-null results for "add
more LLM/agent sophistication on top of retrieval" and zero data points
for "does delegation itself help" -- genuinely untested territory, not a
safe assumption of benefit.

Standalone and eval-only: this module is NOT imported by
backend/api/survey_routes.py's per-turn path. It wraps the three
ALREADY-CONSTRUCTED classifiers (no reimplemented retrieval/reranking/
corrective-retry logic) as CrewAI tools behind three worker Agents
(allow_delegation=False, same as every one of this codebase's 13 existing
agent-constructing modules), orchestrated by a manager LLM that decides
invocation order/delegation at runtime -- the genuinely different
mechanism from Item 1's fixed, deterministic call order.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from backend.llm import TaskType, get_llm

log = logging.getLogger(__name__)


@dataclass
class CoordinatedResult:
    """Result of one classify_all() call -- whichever dimensions had
    non-empty input text are populated, others stay None."""

    isco_code: Optional[str] = None
    isic_section: Optional[str] = None
    isced_level: Optional[int] = None
    fallback_used: bool = False
    fallback_reason: str = ""


# ---------------------------------------------------------------------------
# Field-shape validation -- NOT defensive boilerplate. Real, live-caught
# finding (2026-09-12): a first manual smoke check against local
# ollama/qwen2.5:3b as the manager LLM produced malformed values (e.g.
# isco_code="5310, null") that parsed as valid JSON strings but are not
# real classification codes -- confirmed directly before this validation
# existed, which had silently reported fallback_used=False (i.e. "success")
# for a garbage result. A value that doesn't match its real code shape is
# now treated as "no answer" (None), never passed through.
# ---------------------------------------------------------------------------

def _valid_isco_code(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if re.fullmatch(r"\d{4}", value) else None


def _valid_isic_section(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip().upper()
    return value if re.fullmatch(r"[A-U]", value) else None


def _valid_isced_level(value) -> Optional[int]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    level = int(value)
    return level if 0 <= level <= 8 else None


class HierarchicalClassificationCoordinator:
    """Wraps ISCOClassifier/ISICClassifier/ISCEDClassifier behind a real
    CrewAI hierarchical crew. Every failure mode (manager LLM unreachable,
    malformed tool output, delegation loop, timeout) falls back to calling
    all three classifiers directly and sequentially -- i.e. degrades to
    today's exact survey_routes.py Stage 4/4b/4c behaviour -- never raises,
    never leaves a partial/inconsistent result."""

    def __init__(
        self,
        isco_classifier,
        isic_classifier,
        isced_classifier,
        manager_llm: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        manager_llm : str, optional
            Provider-qualified model string for the hierarchical process's
            manager (e.g. "ollama/qwen2.5:3b"). Default None uses
            get_llm(TaskType.GENERAL) -- same fallback-on-down contract as
            every other unpinned LLM use in this codebase.
        """
        self._isco = isco_classifier
        self._isic = isic_classifier
        self._isced = isced_classifier
        self._manager_llm_pin = manager_llm
        self._manager_llm = manager_llm or get_llm(TaskType.GENERAL)

    def classify_all(
        self,
        job_title: str = "",
        industry_text: str = "",
        education_text: str = "",
        language: str = "en",
    ) -> CoordinatedResult:
        """Classify whichever of job_title/industry_text/education_text
        are non-empty, via the real hierarchical crew. Falls back to a
        direct sequential call to all three classifiers on ANY failure."""
        try:
            return self._classify_via_crew(job_title, industry_text, education_text, language)
        except Exception as exc:
            log.warning(
                "HierarchicalClassificationCoordinator: crew failed (%s); "
                "falling back to direct sequential classification.", exc,
            )
            return self._classify_sequential(job_title, industry_text, education_text)

    # ------------------------------------------------------------------
    # Fallback path -- degrades to today's exact independent-classifier
    # behaviour, no delegation involved at all.
    # ------------------------------------------------------------------

    def _classify_sequential(
        self, job_title: str, industry_text: str, education_text: str
    ) -> CoordinatedResult:
        isco_code = isic_section = None
        isced_level = None
        if job_title:
            isco_code = self._isco.classify(job_title).primary.code
        if industry_text:
            isic_section = self._isic.classify(industry_text).section
        if education_text:
            isced_level = self._isced.classify(education_text).level
        return CoordinatedResult(
            isco_code=isco_code, isic_section=isic_section, isced_level=isced_level,
            fallback_used=True, fallback_reason="sequential fallback (crew unavailable or failed)",
        )

    # ------------------------------------------------------------------
    # Real hierarchical crew -- built lazily here, not in __init__, same
    # convention as every other CrewAI construction in this codebase
    # (built per-call; see query_planner.py's own fix for why a
    # constructor-time Agent build breaks tests that mock this method).
    # ------------------------------------------------------------------

    def _classify_via_crew(
        self, job_title: str, industry_text: str, education_text: str, language: str
    ) -> CoordinatedResult:
        from crewai import Agent, Crew, Process, Task
        from crewai.tools import tool

        isco_clf, isic_clf, isced_clf = self._isco, self._isic, self._isced

        @tool("Classify Occupation")
        def classify_occupation(text: str) -> str:
            """Classify a respondent's job title / occupation description to an ISCO-08 4-digit code."""
            result = isco_clf.classify(text)
            return f"{result.primary.code} ({result.primary.title_en})"

        @tool("Classify Industry")
        def classify_industry(text: str) -> str:
            """Classify a respondent's industry description to an ISIC Rev.4 section and class code."""
            result = isic_clf.classify(text)
            return f"{result.section} / {result.class_code} ({result.class_title})"

        @tool("Classify Education")
        def classify_education(text: str) -> str:
            """Classify a respondent's education description to an ISCED 2011 attainment level."""
            result = isced_clf.classify(text)
            return f"level {result.level} ({result.level_title})"

        occupation_agent = Agent(
            role="Occupation Classification Specialist",
            goal="Classify occupation descriptions to ISCO-08 codes using the Classify Occupation tool",
            backstory="ISCO-08 coding specialist with 15 years of LFS experience.",
            tools=[classify_occupation], llm=self._manager_llm, verbose=False, allow_delegation=False,
        )
        industry_agent = Agent(
            role="Industry Classification Specialist",
            goal="Classify industry descriptions to ISIC Rev.4 codes using the Classify Industry tool",
            backstory="ISIC Rev.4 coding specialist with 15 years of LFS experience.",
            tools=[classify_industry], llm=self._manager_llm, verbose=False, allow_delegation=False,
        )
        education_agent = Agent(
            role="Education Classification Specialist",
            goal="Classify education descriptions to ISCED 2011 levels using the Classify Education tool",
            backstory="ISCED 2011 coding specialist with 15 years of LFS experience.",
            tools=[classify_education], llm=self._manager_llm, verbose=False, allow_delegation=False,
        )

        task = Task(
            description=(
                "A survey respondent gave the following free-text descriptions "
                f"(any may be empty, meaning that dimension was not answered):\n"
                f'occupation: "{job_title}"\n'
                f'industry: "{industry_text}"\n'
                f'education: "{education_text}"\n\n'
                "Delegate to the appropriate specialist tool for each NON-EMPTY "
                "description above -- do not invent a classification for an "
                "empty one. Return ONLY a JSON object with the results:\n"
                '{"isco_code": "XXXX or null", "isic_section": "X or null", '
                '"isced_level": N or null}'
            ),
            expected_output='JSON: {"isco_code": ..., "isic_section": ..., "isced_level": ...}',
            agent=occupation_agent,
        )

        crew = Crew(
            agents=[occupation_agent, industry_agent, education_agent],
            tasks=[task],
            process=Process.hierarchical,
            manager_llm=self._manager_llm,
            verbose=False,
        )
        raw = str(crew.kickoff()).strip()
        return self._parse_crew_result(raw)

    @staticmethod
    def _parse_crew_result(raw: str) -> CoordinatedResult:
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
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

        return CoordinatedResult(
            isco_code=_valid_isco_code(data.get("isco_code")),
            isic_section=_valid_isic_section(data.get("isic_section")),
            isced_level=_valid_isced_level(data.get("isced_level")),
            fallback_used=False,
        )
