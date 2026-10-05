"""
backend/agents/hierarchical_classification_crew.py

Standalone experimental CrewAI hierarchical delegation around the existing
ISCO/ISIC/ISCED classifiers. Live survey collaboration uses the bounded
sequential ``survey_classification_crew`` instead. The manager's JSON must
match specialist tool evidence; missing, malformed or invented results use
the direct fallback. Delegation itself has no established accuracy benefit.
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
    if isinstance(value, float) and not value.is_integer():
        return None
    level = int(value)
    return level if 0 <= level <= 8 else None


class HierarchicalClassificationCoordinator:
    """Wraps ISCOClassifier/ISICClassifier/ISCEDClassifier behind a real
    CrewAI hierarchical crew. Every failure mode (manager LLM unreachable,
    malformed tool output, delegation loop, timeout) falls back to calling
    unfinished classifiers directly and sequentially. Completed tool results
    are retained; unavailable dimensions remain None with fallback disclosed."""

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
        self._manager_llm = manager_llm

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
        completed = {}
        if not any((job_title, industry_text, education_text)):
            return CoordinatedResult()
        try:
            return self._classify_via_crew(job_title, industry_text, education_text, language,
                                           completed=completed)
        except Exception as exc:
            log.warning(
                "HierarchicalClassificationCoordinator: crew failed (%s); "
                "falling back to direct sequential classification.", exc,
            )
            return self._classify_sequential(job_title, industry_text, education_text,
                                             completed=completed)

    # ------------------------------------------------------------------
    # Fallback path -- degrades to today's exact independent-classifier
    # behaviour, no delegation involved at all.
    # ------------------------------------------------------------------

    def _classify_sequential(
        self, job_title: str, industry_text: str, education_text: str, completed=None
    ) -> CoordinatedResult:
        completed = completed if completed is not None else {}
        fallback_errors = []
        callbacks = (
            ("isco_code", job_title, lambda: _valid_isco_code(self._isco.classify(job_title).primary.code)),
            ("isic_section", industry_text, lambda: _valid_isic_section(self._isic.classify(industry_text).section)),
            ("isced_level", education_text, lambda: _valid_isced_level(self._isced.classify(education_text).level)),
        )
        for key, text, callback in callbacks:
            if not text or key in completed:
                continue
            try:
                completed[key] = callback()
            except Exception as exc:
                completed[key] = None
                fallback_errors.append(f"{key}: {type(exc).__name__}")
        return CoordinatedResult(
            isco_code=completed.get("isco_code") if job_title else None,
            isic_section=completed.get("isic_section") if industry_text else None,
            isced_level=completed.get("isced_level") if education_text else None,
            fallback_used=True,
            fallback_reason="sequential fallback (crew unavailable or failed)" +
                            ("; " + ", ".join(fallback_errors) if fallback_errors else ""),
        )

    # ------------------------------------------------------------------
    # Real hierarchical crew -- built lazily here, not in __init__, same
    # convention as every other CrewAI construction in this codebase
    # (built per-call; see query_planner.py's own fix for why a
    # constructor-time Agent build breaks tests that mock this method).
    # ------------------------------------------------------------------

    def _classify_via_crew(
        self, job_title: str, industry_text: str, education_text: str, language: str,
        completed=None,
    ) -> CoordinatedResult:
        from crewai import Agent, Crew, Process, Task
        from crewai.tools import tool

        isco_clf, isic_clf, isced_clf = self._isco, self._isic, self._isced
        completed = completed if completed is not None else {}
        manager_llm = self._manager_llm or get_llm(TaskType.GENERAL)

        @tool("Classify Occupation")
        def classify_occupation(text: str) -> str:
            """Classify a respondent's job title / occupation description to an ISCO-08 4-digit code."""
            if not job_title or text != job_title:
                raise ValueError("Occupation tool must use the exact non-empty survey input")
            if "isco_code" in completed:
                return str(completed["isco_code"])
            completed["isco_code"] = None
            result = isco_clf.classify(text)
            completed["isco_code"] = _valid_isco_code(result.primary.code)
            return f"{result.primary.code} ({result.primary.title_en})"

        @tool("Classify Industry")
        def classify_industry(text: str) -> str:
            """Classify a respondent's industry description to an ISIC Rev.4 section and class code."""
            if not industry_text or text != industry_text:
                raise ValueError("Industry tool must use the exact non-empty survey input")
            if "isic_section" in completed:
                return str(completed["isic_section"])
            completed["isic_section"] = None
            result = isic_clf.classify(text)
            completed["isic_section"] = _valid_isic_section(result.section)
            return f"{result.section} / {result.class_code} ({result.class_title})"

        @tool("Classify Education")
        def classify_education(text: str) -> str:
            """Classify a respondent's education description to an ISCED 2011 attainment level."""
            if not education_text or text != education_text:
                raise ValueError("Education tool must use the exact non-empty survey input")
            if "isced_level" in completed:
                return str(completed["isced_level"])
            completed["isced_level"] = None
            result = isced_clf.classify(text)
            completed["isced_level"] = _valid_isced_level(result.level)
            return f"level {result.level} ({result.level_title})"

        occupation_agent = Agent(
            role="Occupation Classification Specialist",
            goal="Classify occupation descriptions to ISCO-08 codes using the Classify Occupation tool",
            backstory="ISCO-08 coding specialist with 15 years of LFS experience.",
            tools=[classify_occupation], llm=manager_llm, verbose=False, allow_delegation=False,
            max_iter=4, max_retry_limit=0, max_execution_time=180,
        )
        industry_agent = Agent(
            role="Industry Classification Specialist",
            goal="Classify industry descriptions to ISIC Rev.4 codes using the Classify Industry tool",
            backstory="ISIC Rev.4 coding specialist with 15 years of LFS experience.",
            tools=[classify_industry], llm=manager_llm, verbose=False, allow_delegation=False,
            max_iter=4, max_retry_limit=0, max_execution_time=180,
        )
        education_agent = Agent(
            role="Education Classification Specialist",
            goal="Classify education descriptions to ISCED 2011 levels using the Classify Education tool",
            backstory="ISCED 2011 coding specialist with 15 years of LFS experience.",
            tools=[classify_education], llm=manager_llm, verbose=False, allow_delegation=False,
            max_iter=4, max_retry_limit=0, max_execution_time=180,
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
        )

        manager_agent = Agent(
            role="Classification Manager",
            goal="Delegate each non-empty survey input and preserve specialist tool classifications.",
            backstory="Coordinator for occupation, industry and education specialists.",
            llm=manager_llm, allow_delegation=True, verbose=False,
            max_iter=8, max_retry_limit=0, max_execution_time=240,
        )
        crew = Crew(
            agents=[occupation_agent, industry_agent, education_agent],
            tasks=[task],
            process=Process.hierarchical,
            manager_llm=manager_llm,
            manager_agent=manager_agent,
            verbose=False,
        )
        raw = str(crew.kickoff()).strip()
        parsed = self._parse_crew_result(raw)
        for key, text in (("isco_code", job_title), ("isic_section", industry_text),
                          ("isced_level", education_text)):
            if not text:
                setattr(parsed, key, None)
            elif key not in completed or completed[key] is None or getattr(parsed, key) != completed[key]:
                raise ValueError("Manager result lacks matching specialist tool evidence")
        return parsed

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
                except json.JSONDecodeError as exc:
                    raise ValueError("Malformed manager classification JSON") from exc
            else:
                raise ValueError("Missing manager classification JSON")
        if not isinstance(data, dict):
            raise ValueError("Manager classification JSON must be an object")

        return CoordinatedResult(
            isco_code=_valid_isco_code(data.get("isco_code")),
            isic_section=_valid_isic_section(data.get("isic_section")),
            isced_level=_valid_isced_level(data.get("isced_level")),
            fallback_used=False,
        )
