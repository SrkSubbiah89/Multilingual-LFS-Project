"""
eval/legacy_decision_policy41/policy.py

Task 41's additive historical decision-policy compatibility component.

Preserves exactly, from `backend/agents/isco_classifier.py` @ LEGACY_SHA
(824fcf235ae2f8787706cf479a07620519c914de) -- direct source evidence is
quoted next to each preserved value below and in POLICY_MANIFEST:

  - exactly five ordered candidates (`top_k: int = 5`);
  - the 0.92 confidence threshold (`_HIGH_CONFIDENCE_THRESHOLD = 0.92`);
  - the fast semantic-only path when the top candidate meets/exceeds it
    (`if candidates[0].confidence >= _HIGH_CONFIDENCE_THRESHOLD: ...
    method="semantic"`);
  - exactly one reranker invocation on the low-confidence path (the
    historical `classify()` calls `self._llm_select(...)` exactly once
    per below-threshold classification);
  - the historical prompt field set (job title, language note, optional
    context line, and per-candidate code/title_en/title_ar/level/
    confidence-score/description, verbatim from `_llm_select`'s
    `candidate_block` construction);
  - candidate-only `selected_code` validation (`if selected_code in
    code_map: ...`);
  - the historical semantic-top fallback and its exact reasoning string
    (`"Fallback to top semantic match (LLM response could not be
    parsed)."`) for malformed/invalid/out-of-candidate responses; and
  - the `"semantic"` / `"llm_ranked"` method labels (the historical
    `classify()` sets `method="llm_ranked"` for the entire below-threshold
    branch regardless of whether `_parse_llm_response` succeeded or hit
    its own internal fallback -- this module preserves that exact
    convention: the label reflects which branch was taken, not whether
    the reranker's output was ultimately usable).

Necessarily different from the historical code (disclosed, not hidden):
the historical `_llm_select` had no exception handling around
`crew.kickoff()` itself -- an exception there would have propagated
unhandled. This module's `reranker` parameter is an arbitrary injected
callable (not a CrewAI `Crew`), so a single try/except around that one
call catches any exception and folds it into the same semantic-top
fallback used for a malformed response, still counting as exactly one
invocation. This is a deliberate compatibility-layer safety addition, not
a claim about historical behavior on an unhandled crew.kickoff() failure.

This module has zero project imports (no Qdrant, embedding model, CrewAI,
current classifier, WISCO, ISIC, ISCED, SRE, or evaluator import of any
kind) and makes no network call itself anywhere in its own code --
`reranker` is supplied entirely by the caller and invoked at most once.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

HISTORICAL_CANDIDATE_COUNT = 5
HISTORICAL_THRESHOLD = 0.92
HISTORICAL_MODEL = "anthropic/claude-3-5-sonnet-20241022"
HISTORICAL_TEMPERATURE = 0.0

METHOD_SEMANTIC = "semantic"
METHOD_LLM_RANKED = "llm_ranked"
ALLOWED_METHODS = frozenset({METHOD_SEMANTIC, METHOD_LLM_RANKED})

HISTORICAL_FALLBACK_REASONING = (
    "Fallback to top semantic match (LLM response could not be parsed)."
)


@dataclass(frozen=True)
class PolicyCandidate:
    """Caller-supplied candidate. Mirrors the fields the historical prompt
    reads off `OccupationMatch` -- this dataclass does not import that
    class (or anything from `backend.rag`) to keep this module free of any
    Qdrant-adjacent import."""

    code: str
    title_en: str
    title_ar: str
    level: int
    confidence: float
    description: str


@dataclass
class PolicyResult:
    primary: PolicyCandidate
    reasoning: str
    method: str
    reranker_invocations: int
    prompt_text: Optional[str]


def _lang_note(lang: str) -> str:
    return {
        "ar": "The job title is written in Arabic.",
        "mixed": "The job title is code-switched (Arabic and English).",
    }.get(lang, "The job title is written in English.")


def build_prompt_text(
    job_title: str, context: str, lang: str, candidates: list[PolicyCandidate]
) -> str:
    """Reproduces the historical `_llm_select` prompt construction (the
    candidate block, language note, optional context line, instructions,
    and exact JSON response contract) verbatim -- pure string building, no
    network call, no agent/LLM construction of any kind."""
    candidate_block = "\n".join(
        f"{i + 1}. [{c.code}] {c.title_en} / {c.title_ar}\n"
        f"   Level {c.level} | Semantic score: {c.confidence:.2%}\n"
        f"   {c.description}"
        for i, c in enumerate(candidates)
    )
    lang_note = _lang_note(lang)
    context_line = f"\nAdditional context: {context}" if context.strip() else ""
    return (
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
    )


def parse_reranker_response(
    raw: str, candidates: list[PolicyCandidate]
) -> tuple[PolicyCandidate, str]:
    """Reproduces the historical `_parse_llm_response` verbatim: strips
    markdown fences, tries a direct JSON decode, falls back to extracting
    the first `{...}` block, validates `selected_code` against the
    supplied candidates only, and falls back to the top candidate with
    the exact historical fallback reasoning string on any failure."""
    code_map = {c.code: c for c in candidates}

    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    data: dict = {}
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    selected_code = str(data.get("selected_code", "")).strip()
    reasoning = str(data.get("reasoning", "")).strip()

    if selected_code in code_map:
        return code_map[selected_code], reasoning or "Selected by LLM classifier."

    return candidates[0], HISTORICAL_FALLBACK_REASONING


def classify_with_policy(
    job_title: str,
    candidates: list[PolicyCandidate],
    context: str = "",
    lang: str = "en",
    reranker: Optional[Callable[[str], str]] = None,
) -> PolicyResult:
    """
    The additive historical decision-policy compatibility component.

    `reranker`, if the confidence threshold is missed, is called exactly
    once with the constructed prompt text and must return a raw string
    (mirroring the historical `str(crew.kickoff())`). This function never
    constructs a reranker itself and never calls one when the fast path is
    taken.
    """
    if len(candidates) != HISTORICAL_CANDIDATE_COUNT:
        raise ValueError(
            f"historical decision policy requires exactly "
            f"{HISTORICAL_CANDIDATE_COUNT} ordered candidates, got {len(candidates)}"
        )

    top = candidates[0]
    if top.confidence >= HISTORICAL_THRESHOLD:
        return PolicyResult(
            primary=top,
            reasoning=f"Unambiguous semantic match (score {top.confidence:.2%}).",
            method=METHOD_SEMANTIC,
            reranker_invocations=0,
            prompt_text=None,
        )

    if reranker is None:
        raise ValueError(
            "a reranker callable is required when the top candidate is below threshold"
        )

    prompt_text = build_prompt_text(job_title, context, lang, candidates)

    try:
        raw = reranker(prompt_text)
    except Exception:
        return PolicyResult(
            primary=top,
            reasoning=HISTORICAL_FALLBACK_REASONING,
            method=METHOD_LLM_RANKED,
            reranker_invocations=1,
            prompt_text=prompt_text,
        )

    primary, reasoning = parse_reranker_response(str(raw), candidates)
    return PolicyResult(
        primary=primary,
        reasoning=reasoning,
        method=METHOD_LLM_RANKED,
        reranker_invocations=1,
        prompt_text=prompt_text,
    )


POLICY_MANIFEST = [
    {
        "component": "candidate_count",
        "preserved_value": HISTORICAL_CANDIDATE_COUNT,
        "legacy_evidence": 'def classify(self, job_title: str, context: str = "", top_k: int = 5) -> ISCOClassification:',
    },
    {
        "component": "confidence_threshold",
        "preserved_value": HISTORICAL_THRESHOLD,
        "legacy_evidence": "_HIGH_CONFIDENCE_THRESHOLD = 0.92",
    },
    {
        "component": "fast_path_condition",
        "preserved_value": "candidates[0].confidence >= _HIGH_CONFIDENCE_THRESHOLD",
        "legacy_evidence": 'if candidates[0].confidence >= _HIGH_CONFIDENCE_THRESHOLD: ... method="semantic"',
    },
    {
        "component": "reranker_invocation_count",
        "preserved_value": 1,
        "legacy_evidence": "primary, reasoning = self._llm_select(job_title, candidates, context, lang)  -- called exactly once per below-threshold classify()",
    },
    {
        "component": "historical_model",
        "preserved_value": HISTORICAL_MODEL,
        "legacy_evidence": 'MODEL_CRITICAL = "anthropic/claude-3-5-sonnet-20241022"  (backend/llm/llm_client.py @ LEGACY_SHA)',
    },
    {
        "component": "historical_temperature",
        "preserved_value": HISTORICAL_TEMPERATURE,
        "legacy_evidence": "_TEMP_CRITICAL = 0.0  (backend/llm/llm_client.py @ LEGACY_SHA)",
    },
    {
        "component": "prompt_fields",
        "preserved_value": [
            "job_title", "language_note", "context_line", "candidate_code",
            "title_en", "title_ar", "level", "confidence_score", "description",
        ],
        "legacy_evidence": "candidate_block f-string + lang_note dict + context_line in _llm_select()",
    },
    {
        "component": "candidate_only_validation",
        "preserved_value": "selected_code in code_map",
        "legacy_evidence": "if selected_code in code_map: return code_map[selected_code], ...",
    },
    {
        "component": "method_labels",
        "preserved_value": sorted(ALLOWED_METHODS),
        "legacy_evidence": 'method="semantic" / method="llm_ranked" in ISCOClassification construction',
    },
    {
        "component": "fallback_behavior",
        "preserved_value": HISTORICAL_FALLBACK_REASONING,
        "legacy_evidence": 'return (candidates[0], "Fallback to top semantic match (LLM response could not be parsed).")',
    },
]
