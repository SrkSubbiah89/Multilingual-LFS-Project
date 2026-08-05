"""
backend/agents/isco_reranker_strict.py

Strict-output reranker for the future B3-Reliability ablation.

NOT wired into ISCOClassifier's main classify() flow, and NOT used by B0,
B1, or B2. Those all continue to use
isco_classifier.py's _llm_select_from_candidates()/_parse_llm_response(),
unchanged, which silently falls back to the top pooled candidate on any
parse failure. That fallback behaviour is exactly what B0/B1 numbers were
measured against, so it cannot change under them.

This module implements the alternative, stricter contract for a later,
separately-named experiment:
  - Output schema: {"selected_isco_code", "confidence_or_score", "reason"}
    (B0/B1/B2 use {"selected_code", "reasoning"} -- deliberately different
    field names so a caller can never accidentally mix the two schemas up).
  - selected_isco_code MUST be one of the candidate codes supplied, or the
    output is invalid.
  - On invalid JSON, an unrecognised code, or a timeout: retry exactly
    once. If the retry also fails, return an explicit ABSTENTION result
    (RerankDecision.abstained=True) -- never silently substitute the top
    candidate.

Usage (standalone, opt-in, for a future eval harness invocation):
    from backend.agents.isco_reranker_strict import StrictReranker
    reranker = StrictReranker(llm=some_llm)
    decision = reranker.select(job_title, candidates, lang="en")
    if decision.abstained:
        ...  # record as abstention, not as a normal prediction
    else:
        ...  # decision.selected_code, decision.confidence, decision.reason
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from crewai import Agent, Crew, Task

_logger = logging.getLogger(__name__)


@dataclass
class StrictCandidate:
    code: str
    label_en: str
    label_ar: str = ""
    score: float = 0.0


@dataclass
class RerankDecision:
    """Result of one StrictReranker.select() call. Exactly one of
    (selected_code set, abstained=True) is ever true -- never both, and
    never a selected_code that silently defaulted to the top candidate."""
    selected_code: Optional[str] = None
    confidence: Optional[float] = None
    reason: str = ""
    abstained: bool = False
    abstain_reason: str = ""
    attempts: int = 0
    invalid_output: bool = False
    timed_out: bool = False
    raw_outputs: list = field(default_factory=list)  # one entry per attempt, for logging
    latency_ms: float = 0.0


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class StrictReranker:
    """
    Wraps a CrewAI Agent/LLM with the strict output contract described in
    the module docstring. max_retries=1 means "one retry" -- two total
    attempts, matching the spec's "retry once only".
    """

    def __init__(self, agent: Agent, max_retries: int = 1) -> None:
        self._agent = agent
        self._max_retries = max(0, int(max_retries))

    def select(
        self,
        job_title: str,
        candidates: list[StrictCandidate],
        lang: str = "en",
        context: str = "",
    ) -> RerankDecision:
        if not candidates:
            return RerankDecision(abstained=True, abstain_reason="empty candidate list")

        code_map = {c.code: c for c in candidates}
        t_start = time.perf_counter()
        decision = RerankDecision()

        for attempt in range(self._max_retries + 1):
            decision.attempts = attempt + 1
            try:
                raw = self._call_llm(job_title, candidates, lang, context)
            except Exception as exc:  # noqa: BLE001 -- includes timeouts
                decision.raw_outputs.append(f"EXCEPTION: {type(exc).__name__}: {exc}")
                decision.timed_out = "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
                _logger.warning(
                    "StrictReranker attempt %d/%d failed: %s",
                    attempt + 1, self._max_retries + 1, exc,
                )
                continue

            decision.raw_outputs.append(raw)
            parsed_code, confidence, reason, ok = self._parse(raw, code_map)
            if ok:
                decision.selected_code = parsed_code
                decision.confidence = confidence
                decision.reason = reason
                decision.latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
                return decision

            decision.invalid_output = True
            _logger.warning(
                "StrictReranker attempt %d/%d produced invalid/unrecognised output: %r",
                attempt + 1, self._max_retries + 1, raw[:200],
            )

        # Every attempt (initial + retries) failed -- explicit abstention,
        # never a silent fallback to candidates[0].
        decision.abstained = True
        decision.abstain_reason = (
            "timeout on all attempts" if decision.timed_out and not decision.invalid_output
            else "invalid or unrecognised output on all attempts"
        )
        decision.latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return decision

    # ------------------------------------------------------------------

    def _call_llm(self, job_title: str, candidates: list[StrictCandidate], lang: str, context: str) -> str:
        candidate_block = "\n".join(
            f"{i + 1}. [{c.code}] {c.label_en} / {c.label_ar}\n   Semantic score: {c.score:.2%}"
            for i, c in enumerate(candidates)
        )
        lang_note = {
            "ar": "The job title is written in Arabic.",
            "mixed": "The job title is code-switched (Arabic and English).",
        }.get(lang, "The job title is written in English.")
        context_line = f"\nAdditional context: {context}" if context.strip() else ""

        task = Task(
            description=(
                "You are an ISCO-08 classification specialist for a national "
                "Labour Force Survey.\n\n"
                f'Job title: "{job_title}"\n'
                f"{lang_note}{context_line}\n\n"
                f"Candidates (from hierarchical semantic search):\n{candidate_block}\n\n"
                "Select the single best ISCO-08 match. Prefer the most specific "
                "code (unit group over sub-major over major group) when the title "
                "clearly supports it.\n\n"
                "Return ONLY a valid JSON object -- no markdown fences, no extra text:\n"
                '{"selected_isco_code": "<isco_code>", "confidence_or_score": <0-1 float>, "reason": "<one sentence>"}'
            ),
            expected_output=(
                'JSON: {"selected_isco_code": "<code>", "confidence_or_score": <float>, "reason": "<sentence>"}'
            ),
            agent=self._agent,
        )
        crew = Crew(agents=[self._agent], tasks=[task], verbose=False)
        return str(crew.kickoff()).strip()

    @staticmethod
    def _parse(
        raw: str, code_map: dict[str, StrictCandidate],
    ) -> tuple[Optional[str], Optional[float], str, bool]:
        """Returns (selected_code, confidence, reason, ok). ok=False means
        this attempt is invalid -- caller retries or abstains, NEVER
        substitutes a default here."""
        clean = _JSON_FENCE_RE.sub("", raw).strip()
        data: dict = {}
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            m = _JSON_OBJECT_RE.search(clean)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    return None, None, "", False
            else:
                return None, None, "", False

        selected = str(data.get("selected_isco_code", "")).strip()
        if selected not in code_map:
            return None, None, "", False  # unrecognised code -- invalid, not a fallback trigger

        reason = str(data.get("reason", "")).strip()
        conf_raw = data.get("confidence_or_score")
        try:
            confidence = float(conf_raw) if conf_raw is not None else None
        except (TypeError, ValueError):
            confidence = None

        return selected, confidence, reason, True
