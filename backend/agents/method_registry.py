"""
backend/agents/method_registry.py

Classifier / agent method registry for Conference I Reviewer #2 feedback
point 6 ("multiple LLM roles are unclear -- the specific task handled by
each model must be documented and measurable").

Produces one ``ClassifierMethodEntry`` row per (component, method) pair for
every classifier/agent named in the reviewer response: LanguageProcessor,
ConversationManager, ISCOClassifier, ISICClassifier, ISCEDClassifier,
SemanticRelationEngine, ValidationAgent, HITLQualityManager,
SurveyOrchestrator.

Every field below was hand-derived by reading the component's actual code
(dataclass/Pydantic model fields, ``get_llm(TaskType.*)`` calls, CrewAI
``Agent(role=...)`` strings, threshold constants) -- not inferred or
assumed. In particular:

  - ``affects_hitl_escalation`` reflects VERIFIED wiring, not intent. Grep
    confirms ``SurveyOrchestrator`` triggers ``HITLQualityManager.
    review_session()`` on session completion using ONLY
    ``HITLQualityManager``'s own independent thresholds
    (backend/agents/hitl_quality_manager.py) and ``ISCOClassifier``'s
    per-response confidence (via HITL_THRESHOLD=0.70 and the
    "low_confidence_isco" flag path). ``ValidationAgent.rule_violations``
    and ``SemanticRelationEngine``'s ``SemanticCoherence`` are surfaced on
    ``TurnResult`` but are NOT read anywhere near the escalation decision --
    so both are marked ``affects_hitl_escalation=False`` here. If a future
    change wires either of them into escalation, this registry (and
    ``test_method_registry.py``'s cross-check against
    ``survey_orchestrator.py``) must be updated together.
  - ``evaluated`` / ``evaluated_ref`` are honest: only ISCO has any
    evaluation history at all today (the ``eval/`` B2 harness), and even
    that is ISCO-only accuracy -- so every ISIC/ISCED/SRE/HITL row is
    ``evaluated=False`` until Section D/E of the Reviewer #2 work produces a
    real run manifest to point ``evaluated_ref`` at.
  - The two "not yet implemented" ISIC/ISCED-F hierarchical-retrieval rows
    have their ``input_fields``/``output_schema`` populated with the
    *intended* future schema (clearly marked "planned, not yet built" in
    ``fallback_behaviour``), not treated as unknown/null -- this is useful
    for the agent-role diagram export (Section I) and does not overstate
    current capability (``evaluated`` is False and the method is listed in
    ``classifier_methods.NOT_IMPLEMENTED_METHODS``).

Usage
-----
    python -m backend.agents.method_registry --out Documentation/Conference_I_Reviewer_2/generated/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from backend.agents.classifier_methods import (
    ISCED_RULE_KEYWORD,
    ISCEDF_HIERARCHICAL_RETRIEVAL,
    ISCO_HIERARCHICAL_RAG,
    ISIC_HIERARCHICAL_RETRIEVAL,
    ISIC_KEYWORD_LLM,
)


class ClassifierMethodEntry(BaseModel):
    """One row: a single (component, method) combination."""

    component: str
    method_id: str
    category: Literal["deterministic", "retrieval", "llm", "hybrid"]
    input_fields: list[str]
    output_schema: dict[str, str]          # field name -> type description
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    embedding_model: Optional[str] = None
    prompt_version: Optional[str] = None
    decoding_config: Optional[dict] = None
    fallback_behaviour: str
    evaluated: bool
    evaluated_ref: Optional[str] = None
    affects_hitl_escalation: bool


# ---------------------------------------------------------------------------
# Shared model-routing descriptions (see backend/llm/llm_client.py)
# ---------------------------------------------------------------------------

_GENERAL_MODEL = "ollama/llama3.2 (env OLLAMA_MODEL), falls back to anthropic/claude-3-5-sonnet-20241022 if Ollama unreachable"
_CRITICAL_MODEL = "anthropic/claude-3-5-sonnet-20241022"
_E5_EMBEDDING = "intfloat/multilingual-e5-small (384-dim)"

_NO_PROMPT_VERSIONING = (
    "Prompt text is not currently version-tagged in source; the prompt "
    "string itself (in the component's module) is the only record of what "
    "was sent to the LLM for any given commit."
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: list[ClassifierMethodEntry] = [

    # ── LanguageProcessor ────────────────────────────────────────────────
    ClassifierMethodEntry(
        component="LanguageProcessor",
        method_id="language_detection_rule_based",
        category="deterministic",
        input_fields=["raw_text"],
        output_schema={
            "detected_language": "str (en|ar|ar-gulf|ur|hi|tl|other)",
            "confidence": "float 0-1", "is_code_switched": "bool",
            "arabic_ratio": "float 0-1", "latin_ratio": "float 0-1",
            "devanagari_ratio": "float 0-1", "segments": "list[CodeSegment]",
            "normalised_text": "Optional[str]",
        },
        model_name=None, model_version=None, embedding_model=None,
        prompt_version=None, decoding_config=None,
        fallback_behaviour="Pure regex/script-ratio + seeded langdetect (DetectorFactory.seed=0); deterministic, no LLM, no failure mode.",
        evaluated=False, evaluated_ref=None,
        affects_hitl_escalation=False,
    ),
    ClassifierMethodEntry(
        component="LanguageProcessor",
        method_id="ner_llm",
        category="llm",
        input_fields=["raw_text"],
        output_schema={"entities": "list[Entity{text,label,language,start,end}]"},
        model_name=_GENERAL_MODEL, model_version=None, embedding_model=None,
        prompt_version=_NO_PROMPT_VERSIONING,
        decoding_config={"task_type": "GENERAL"},
        fallback_behaviour="Role 'Multilingual NER Specialist' (CrewAI Agent, get_llm(TaskType.GENERAL)). Degrades to entities=[] (not a hard failure) if the LLM is unavailable.",
        evaluated=False, evaluated_ref=None,
        affects_hitl_escalation=False,
    ),

    # ── ConversationManager ──────────────────────────────────────────────
    ClassifierMethodEntry(
        component="ConversationManager",
        method_id="fsm_dialogue_llm",
        category="llm",
        input_fields=["conversation_context", "collected_data", "current_state"],
        output_schema={"response_text": "str", "next_state": "ConversationState enum",
                        "collected_data": "dict (raw text only, no classification)"},
        model_name=_GENERAL_MODEL, model_version=None, embedding_model=None,
        prompt_version=_NO_PROMPT_VERSIONING,
        decoding_config={"task_type": "GENERAL"},
        fallback_behaviour="Role 'Conversation Manager' (CrewAI Agent, get_llm(TaskType.GENERAL)). Falls back to a deterministic _dev_stub_response() when the LLM is unavailable or LFS_FAST_MODE=true. Does NOT call any classifier directly -- only collects raw text; classification happens downstream in SurveyOrchestrator.",
        evaluated=False, evaluated_ref=None,
        affects_hitl_escalation=False,
    ),

    # ── ISCOClassifier ────────────────────────────────────────────────────
    ClassifierMethodEntry(
        component="ISCOClassifier",
        method_id=ISCO_HIERARCHICAL_RAG,
        category="hybrid",
        input_fields=["job_title", "language"],
        output_schema={"primary": "ISCOMatch{code,title_en,title_ar,confidence}",
                        "alternatives": "list[ISCOMatch]", "method": "str",
                        "stage_confidences": "dict{stage1..stage4}",
                        "hierarchy_path": "list[str]", "hitl_required": "bool",
                        "reasoning": "str"},
        model_name=_GENERAL_MODEL, model_version=None,
        embedding_model=_E5_EMBEDDING,
        prompt_version=_NO_PROMPT_VERSIONING,
        decoding_config={"task_type": "GENERAL", "high_confidence_skip_llm_threshold": 0.92},
        fallback_behaviour="4-stage hierarchical beam search (backend/rag/hierarchy_engine.py) with optional LLM re-ranking of the top pooled candidates. Falls back to flat isco_occupations search if any hierarchical Qdrant collection is missing or a stage returns 0 hits; falls back further to semantic-only selection (method suffix '_semantic') if the LLM is unavailable.",
        evaluated=True, evaluated_ref="eval/results/ (B0-B2 accuracy runs on test_set_full130.csv -- ISCO top-1/top-3 digit-level accuracy only; see EVALUATION_PROTOCOL.md)",
        affects_hitl_escalation=True,
    ),

    # ── ISICClassifier ────────────────────────────────────────────────────
    ClassifierMethodEntry(
        component="ISICClassifier",
        method_id=ISIC_KEYWORD_LLM,
        category="hybrid",
        input_fields=["text"],
        output_schema={"section": "str (A-U)", "section_title": "str",
                        "division_code": "str (2-digit)", "division_title": "str",
                        "group_code": "str (3-digit)", "group_title": "str",
                        "class_code": "str (4-digit)", "class_title": "str",
                        "confidence": "float 0-1", "method": "str (keyword|llm)",
                        "alternatives": "list[dict]", "raw_text": "Optional[str]"},
        model_name=_GENERAL_MODEL, model_version=None, embedding_model=None,
        prompt_version=_NO_PROMPT_VERSIONING,
        decoding_config={"task_type": "GENERAL", "keyword_threshold": 0.85, "top_k": 3},
        fallback_behaviour="Keyword lookup over a flat leaf-path table (_ISIC_DATA); LLM re-ranking (role 'ISIC Industry Classifier') only invoked when keyword confidence < 0.85. LLM failure falls back to best keyword match with confidence deflated by 0.8x. NOT hierarchical retrieval -- this is keyword lookup, not Section->Division->Group->Class RAG (see isic_hierarchical_retrieval below for the deferred hierarchical mode).",
        evaluated=False, evaluated_ref=None,
        affects_hitl_escalation=False,
    ),
    ClassifierMethodEntry(
        component="ISICClassifier",
        method_id=ISIC_HIERARCHICAL_RETRIEVAL,
        category="retrieval",
        input_fields=["text"],
        output_schema={
            "section": "str (planned)", "division_code": "str (planned)",
            "group_code": "str (planned)", "class_code": "str (planned)",
            "confidence": "float (planned)", "hierarchy_path": "list[str] (planned)",
            "stage_confidences": "dict (planned)", "top_candidates": "list (planned)",
        },
        model_name=None, model_version=None, embedding_model=None,
        prompt_version=None, decoding_config=None,
        fallback_behaviour="NOT YET IMPLEMENTED. No Qdrant collections, loader, or live retrieval exist for ISIC today. Calling ISICClassifier.classify(text, method='isic_hierarchical_retrieval') returns a structured not-implemented result (confidence=0.0, all hierarchy fields empty) rather than raising or silently running the keyword pipeline. See CLASSIFIER_METHOD_REGISTRY.md for the deferred build plan (Section B, full pass).",
        evaluated=False, evaluated_ref=None,
        affects_hitl_escalation=False,
    ),

    # ── ISCEDClassifier ───────────────────────────────────────────────────
    ClassifierMethodEntry(
        component="ISCEDClassifier",
        method_id=ISCED_RULE_KEYWORD,
        category="deterministic",
        input_fields=["text"],
        output_schema={"level": "int 0-8", "level_title": "str",
                        "broad_code": "str (2-digit)", "broad_title": "str",
                        "narrow_code": "str (3-digit)", "narrow_title": "str",
                        "detailed_code": "str (4-digit)", "detailed_title": "str",
                        "confidence": "float 0-1", "method": "str (keyword|rule)",
                        "raw_text": "Optional[str]"},
        model_name=None, model_version=None, embedding_model=None,
        prompt_version=None, decoding_config=None,
        fallback_behaviour="Pure token-overlap scoring against two independent tables (_ISCED_LEVELS for attainment level, _ISCED_FIELDS for ISCED-F field); no LLM, fully offline. Empty/no-match input defaults to level=3 (Upper secondary, confidence 0.3) and a default field entry (confidence 0.0). NOT hierarchical retrieval for the field dimension (see iscedf_hierarchical_retrieval below).",
        evaluated=False, evaluated_ref=None,
        affects_hitl_escalation=False,
    ),
    ClassifierMethodEntry(
        component="ISCEDClassifier",
        method_id=ISCEDF_HIERARCHICAL_RETRIEVAL,
        category="retrieval",
        input_fields=["text"],
        output_schema={
            "broad_code": "str (planned)", "narrow_code": "str (planned)",
            "detailed_code": "str (planned)", "confidence": "float (planned)",
            "hierarchy_path": "list[str] (planned)", "stage_confidences": "dict (planned)",
        },
        model_name=None, model_version=None, embedding_model=None,
        prompt_version=None, decoding_config=None,
        fallback_behaviour="NOT YET IMPLEMENTED. No Qdrant collections, loader, or live retrieval exist for ISCED-F today. Calling ISCEDClassifier.classify(text, method='iscedf_hierarchical_retrieval') returns a structured not-implemented result (confidence=0.0, level=-1 sentinel, all field codes empty) rather than raising or silently running the rule/keyword pipeline. ISCED 2011 attainment-LEVEL classification is intentionally kept separate (a level, not a field hierarchy) and is unaffected by this deferred item.",
        evaluated=False, evaluated_ref=None,
        affects_hitl_escalation=False,
    ),

    # ── SemanticRelationEngine ───────────────────────────────────────────
    ClassifierMethodEntry(
        component="SemanticRelationEngine",
        method_id="isco_isic_isced_crosswalk",
        category="hybrid",
        input_fields=["isco_code", "isic_section", "isced_level", "job_title", "language"],
        output_schema={"score": "float 0-1", "is_coherent": "bool",
                        "isco_isic_compatible": "bool", "isco_isced_compatible": "bool",
                        "violations": "list[SemanticViolation]", "inferred_isco": "Optional[str]",
                        "explanation_en": "str", "explanation_ar": "str",
                        "confidence_adjustment": "float (-0.20 to +0.10)",
                        "major_group": "str", "major_label": "str",
                        "isic_label": "Optional[str]", "isced_label": "Optional[str]"},
        model_name=_GENERAL_MODEL, model_version=None, embedding_model=None,
        prompt_version=_NO_PROMPT_VERSIONING,
        decoding_config={"task_type": "GENERAL", "use_llm_default": True},
        fallback_behaviour="Deterministic dict-lookup crosswalk (ILO ISCO-ISIC correspondence table, UNESCO ISCED 2011 Operational Manual Table 7) is the core, always-on logic -- no LLM required for scoring. Optional LLM re-inference (role 'ISCO-08 occupation coding specialist') only fires when NOT is_coherent AND isic_section+job_title are both present; LLM failure is caught and silently ignored (inferred_isco stays None). VERIFIED: SurveyOrchestrator stores this on TurnResult.semantic_coherence but does NOT read it near the HITL escalation decision -- purely informational in production today.",
        evaluated=False, evaluated_ref=None,
        affects_hitl_escalation=False,
    ),

    # ── ValidationAgent ───────────────────────────────────────────────────
    ClassifierMethodEntry(
        component="ValidationAgent",
        method_id="rule_based_r01_r10",
        category="deterministic",
        input_fields=["responses (employment_status, job_title, industry, hours_per_week, employment_type)", "language"],
        output_schema={"is_valid": "bool", "rule_violations": "list[RuleViolation{rule_id,field,severity,message_en,message_ar}]"},
        model_name=None, model_version=None, embedding_model=None,
        prompt_version=None, decoding_config=None,
        fallback_behaviour="10 fixed rules (R01-R10, e.g. hours-in-range, employment-status contradictions). Pure Python, no API calls, cannot fail. VERIFIED: rule_violations are surfaced on TurnResult/ValidationResult but are NOT read by HITLQualityManager anywhere -- this is a real, currently-unaddressed architectural gap, not a design choice.",
        evaluated=False, evaluated_ref=None,
        affects_hitl_escalation=False,
    ),
    ClassifierMethodEntry(
        component="ValidationAgent",
        method_id="semantic_check_llm",
        category="llm",
        input_fields=["responses", "language"],
        output_schema={"semantic_issues": "list[str]", "explanation_en": "str",
                        "explanation_ar": "str", "confidence": "float 0-1"},
        model_name=_CRITICAL_MODEL, model_version="claude-3-5-sonnet-20241022",
        embedding_model=None, prompt_version=_NO_PROMPT_VERSIONING,
        decoding_config={"task_type": "CRITICAL", "temperature": 0.0},
        fallback_behaviour="Role 'LFS Survey Data Quality Specialist' (CrewAI Agent, get_llm(TaskType.CRITICAL)). Only runs when Stage 1 (rule_based_r01_r10) found zero errors. Final confidence = 1.0 - 0.25*errors - 0.10*warnings - 0.10*semantic_issues, then multiplied by llm_confidence if no errors.",
        evaluated=False, evaluated_ref=None,
        affects_hitl_escalation=False,
    ),

    # ── HITLQualityManager ────────────────────────────────────────────────
    ClassifierMethodEntry(
        component="HITLQualityManager",
        method_id="quality_scoring_deterministic",
        category="deterministic",
        input_fields=["session_id", "session responses with ISCO confidence scores"],
        output_schema={"quality_score": "float 0-1", "status": "ReviewStatus(pass|fail|escalated)",
                        "metrics": "QualityMetrics", "flagged_items": "list[FlaggedItem]",
                        "escalated": "bool", "escalation_reason": "Optional[str]"},
        model_name=None, model_version=None, embedding_model=None,
        prompt_version=None, decoding_config=None,
        fallback_behaviour="Deterministic weighted score (confidence 0.50, coverage 0.30, low-conf-penalty 0.20) against its OWN independent thresholds (_LOW_CONFIDENCE_THRESHOLD=0.60, _MIN_PASS_QUALITY_SCORE=0.70, _ESCALATION_THRESHOLD=0.50, _MAX_FLAGS_BEFORE_ESCALATION=3) -- VERIFIED independent of ISCOClassifier.HITL_THRESHOLD=0.70 (different constant, different file, no cross-reference). Only question_id in {'job_title','last_job_title'} is eligible for a MISSING_ISCO flag. Cannot fail (no external calls).",
        evaluated=False, evaluated_ref=None,
        affects_hitl_escalation=True,
    ),
    ClassifierMethodEntry(
        component="HITLQualityManager",
        method_id="bilingual_report_llm",
        category="llm",
        input_fields=["quality_score", "metrics", "flagged_items"],
        output_schema={"report_en": "str", "report_ar": "str"},
        model_name=_CRITICAL_MODEL, model_version="claude-3-5-sonnet-20241022",
        embedding_model=None, prompt_version=_NO_PROMPT_VERSIONING,
        decoding_config={"task_type": "CRITICAL"},
        fallback_behaviour="Role 'LFS Survey Quality Assurance Manager' (CrewAI Agent, get_llm(TaskType.CRITICAL)). Generates prose report text only -- scoring/flagging/escalation (quality_scoring_deterministic above) is unaffected by this step's success or failure.",
        evaluated=False, evaluated_ref=None,
        affects_hitl_escalation=False,
    ),

    # ── SurveyOrchestrator ────────────────────────────────────────────────
    ClassifierMethodEntry(
        component="SurveyOrchestrator",
        method_id="turn_orchestration",
        category="deterministic",
        input_fields=["session_id", "turn responses"],
        output_schema={"TurnResult": "wraps classification/validation/semantic_coherence results for one survey turn"},
        model_name=None, model_version=None, embedding_model=None,
        prompt_version=None, decoding_config=None,
        fallback_behaviour="Pure Python wiring: calls ISCOClassifier/ISICClassifier/ISCEDClassifier, ValidationAgent, SemanticRelationEngine(use_llm=False), and triggers HITLQualityManager.review_session() on FSM completion. Does not itself call any LLM. VERIFIED: only ISCOClassifier's per-response confidence and HITLQualityManager's own scoring feed the escalation decision -- ValidationAgent and SemanticRelationEngine outputs are attached to TurnResult but not read by the escalation path.",
        evaluated=False, evaluated_ref=None,
        affects_hitl_escalation=True,
    ),
]


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_json(path: Path) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(REGISTRY),
        "entries": [e.model_dump() for e in REGISTRY],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_markdown(path: Path) -> None:
    lines = [
        "# Classifier Method Registry",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Generated by `python -m backend.agents.method_registry` -- do not hand-edit; ",
        "edit `backend/agents/method_registry.py`'s `REGISTRY` list instead.",
        "",
        "| Component | Method | Category | Model | Embedding | Evaluated | Affects HITL |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in REGISTRY:
        lines.append(
            f"| {e.component} | `{e.method_id}` | {e.category} | "
            f"{e.model_name or '—'} | {e.embedding_model or '—'} | "
            f"{'yes' if e.evaluated else 'no'} | {'yes' if e.affects_hitl_escalation else 'no'} |"
        )

    lines.append("")
    lines.append("## Detail")
    for e in REGISTRY:
        lines.append("")
        lines.append(f"### {e.component} — `{e.method_id}`")
        lines.append("")
        lines.append(f"- **Category**: {e.category}")
        lines.append(f"- **Input fields**: {', '.join(e.input_fields)}")
        lines.append(f"- **Output schema**: {', '.join(f'{k}: {v}' for k, v in e.output_schema.items())}")
        lines.append(f"- **Model**: {e.model_name or 'none'}" + (f" ({e.model_version})" if e.model_version else ""))
        lines.append(f"- **Embedding model**: {e.embedding_model or 'none'}")
        lines.append(f"- **Decoding config**: {e.decoding_config or 'none'}")
        lines.append(f"- **Evaluated**: {e.evaluated}" + (f" — {e.evaluated_ref}" if e.evaluated_ref else ""))
        lines.append(f"- **Affects HITL escalation**: {e.affects_hitl_escalation}")
        lines.append(f"- **Fallback behaviour**: {e.fallback_behaviour}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path,
                         help="Output directory; writes classifier_method_registry.json and .md")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "classifier_method_registry.json"
    md_path = args.out / "classifier_method_registry.md"
    export_json(json_path)
    export_markdown(md_path)
    print(f"Wrote {len(REGISTRY)} entries to:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
