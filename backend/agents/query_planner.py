"""
backend/agents/query_planner.py

Multi-step agentic retrieval (Item 2 of the 2026-09-12 multi-agent RAG
work): decompose an ambiguous compound description into up to N distinct
sub-descriptions, so a classifier can retrieve for each and reconcile,
rather than retrieving once for the whole (possibly compound, possibly
vague) original text.

This generalizes the existing single-shot corrective-retry pattern
already proven in isco_classifier.py/isic_classifier.py/isced_classifier.py
(_maybe_corrective_retry / _maybe_corrective_retry_field: reformulate once,
re-retrieve, accept only on a strictly wider top1/top2 candidate-score
gap) from "one reformulation" to "up to N sub-queries, reconciled." It
does not replace corrective retry; classifiers that opt into
enable_query_planning treat the two as mutually exclusive in practice
(query planning takes precedence when both are enabled) -- see each
classifier's own enable_query_planning docstring.

Uses the exact same CrewAI Agent/Task/Crew construction pattern as
_llm_reformulate_query (allow_delegation=False, no process= -- sequential
default), so this is not a new orchestration style, only a new prompt
shape (N phrases instead of 1).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from crewai import Agent, Crew, Task

from backend.llm import TaskType, get_llm, get_llm_strict

log = logging.getLogger(__name__)
MAX_SUBQUERIES = 3


class QueryPlanner:
    """Decomposes ambiguous text into sub-queries and reconciles per-sub-query results."""

    def __init__(self, reranker_model: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        reranker_model : str, optional
            Pin the decomposition LLM to exactly this model via
            get_llm_strict(), mirroring every other reranker_model
            parameter in this codebase (ISCOClassifier, ISICClassifier,
            ISCEDClassifier). Default None uses get_llm(TaskType.GENERAL)
            -- Ollama, with Claude fallback if Ollama is unreachable.
        """
        if reranker_model:
            self._llm = get_llm_strict(reranker_model, temperature=0.3)
        else:
            self._llm = get_llm(TaskType.GENERAL)

    def decompose(self, text: str, dimension: str, max_subqueries: int = 3,
                  usage_observer: Optional[Callable[[Any, bool, Optional[str]], None]] = None) -> list[str]:
        """Return up to *max_subqueries* distinct sub-descriptions of *text*.

        Falls back to ``[text]`` (i.e. behaves like a single, unmodified
        query -- the same shape corrective retry's own single-reformulation
        path produces) on any LLM failure, empty response, or parse
        failure. Never raises.

        Parameters
        ----------
        dimension : str
            A short label for what's being classified (e.g. "occupation",
            "industry", "education field") -- included in the prompt only,
            purely to make the LLM's task concrete; not otherwise used.
        max_subqueries : int
            Clamped to 1--3. Duplicate phrases do not consume the limit
            or contribute repeated evidence during reconciliation.
        usage_observer : callable, optional
            Receives the Crew, response-received flag, and call error after
            a successful or failed LLM attempt; observer failures are ignored.
        """
        text = (text or "").strip()
        if not text:
            return [text]

        # Keep decomposition bounded even when called outside a classifier.
        # Invalid settings retain the default rather than defeating the
        # fallback contract or allowing an unbounded list of retrieval calls.
        if not isinstance(max_subqueries, int) or isinstance(max_subqueries, bool):
            max_subqueries = MAX_SUBQUERIES
        max_subqueries = max(1, min(max_subqueries, MAX_SUBQUERIES))

        call_error = None
        try:
            # Built lazily here, not in __init__ -- same convention as every
            # other CrewAI construction in this codebase (ISCOClassifier's
            # _llm_reformulate_query, ISICClassifier's _llm_rerank, etc.
            # build Agent/Task/Crew per-call, never once at construction
            # time), and specifically so a test that mocks decompose()
            # itself never has to also worry about a real Agent() pydantic
            # validation against self._llm.
            agent = Agent(
                role="Query decomposition specialist",
                goal="Split an ambiguous or compound respondent description into distinct, specific sub-descriptions",
                backstory=(
                    "Expert at recognising when a single free-text answer actually "
                    "describes more than one distinct thing (e.g. two occupations, "
                    "or an occupation plus an unrelated detail) and separating them "
                    "into individually-searchable phrases."
                ),
                llm=self._llm,
                verbose=False,
                allow_delegation=False,
            )
            task = Task(
                description=(
                    f"A survey respondent gave this free-text {dimension} description:\n\n"
                    f'"{text}"\n\n'
                    f"If this text genuinely describes more than one distinct {dimension} "
                    f"(e.g. two different jobs, or a job plus an unrelated detail), split it "
                    f"into up to {max_subqueries} separate, more specific search phrases, one "
                    f"per line. If it already describes just ONE thing, return that one thing "
                    f"as a single, more specific search phrase.\n\n"
                    "Return ONLY the phrases, one per line, no numbering, no extra text."
                ),
                expected_output="One search phrase per line, no numbering.",
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task], verbose=False)
            raw = str(crew.kickoff()).strip()
        except Exception as exc:
            call_error = f"{type(exc).__name__}: {exc}"
            log.warning("QueryPlanner: decomposition LLM call failed (%s); using original text.", exc)
            return [text]
        finally:
            if usage_observer is not None and "crew" in locals():
                try:
                    usage_observer(crew, "raw" in locals(), call_error)
                except Exception as exc:
                    log.debug("QueryPlanner: could not record decomposition usage: %s", exc)

        lines = []
        seen = set()
        for line in raw.splitlines():
            # Remove list markers, not leading digits inside a search phrase.
            phrase = re.sub(r"^\s*(?:[-*]\s+|\d+[.)]\s+)", "", line).strip()
            key = " ".join(phrase.split()).casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            lines.append(phrase)
            if len(lines) == max_subqueries:
                break
        return lines or [text]

    @staticmethod
    def reconcile(candidates: list[tuple[Any, float]]) -> tuple[Any, float]:
        """Reconcile per-sub-query (identifier, score) pairs into one winner.

        Disclosed, provisional rule (same class of judgement call as
        ISICClassifier's own ``_MIN_CANDIDATE_GAP = 0.15`` -- not
        independently measured, flag before citing in the manuscript):
        an identifier that is the top candidate for >=2 sub-queries beats
        a single higher-scoring outlier that only one sub-query produced.
        Among identifiers tied on "most repeated," the highest score
        wins; with no repeats at all, the single highest score wins.

        Raises ValueError on an empty list -- callers must not call this
        with nothing to reconcile.
        """
        if not candidates:
            raise ValueError("QueryPlanner.reconcile() requires at least one candidate")
        if len(candidates) == 1:
            return candidates[0]

        groups: dict[Any, list[float]] = {}
        for ident, score in candidates:
            groups.setdefault(ident, []).append(score)

        repeated = {k: v for k, v in groups.items() if len(v) >= 2}
        pool = repeated if repeated else groups
        best_ident = max(pool, key=lambda k: (len(pool[k]), max(pool[k])))
        return best_ident, max(pool[best_ident])
