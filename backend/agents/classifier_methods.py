"""
backend/agents/classifier_methods.py

Single source of truth for the classification-method label strings used by
Documentation/Conference_I_Reviewer_2's classifier method registry (Section A)
and by the ISIC/ISCED "not yet implemented" hierarchical-retrieval stubs
(Section B, deferred scope -- see Documentation/Conference_I_Reviewer_2/
CLASSIFIER_METHOD_REGISTRY.md for the full writeup of what "not yet
implemented" means here and what a future implementation would need).

These constants are additive labels for documentation/registry/evaluation
purposes. They do NOT rename any existing method string a classifier already
returns (e.g. ISCOClassifier's own "hierarchical_llm" / "hierarchical_semantic"
/ "flat_llm" / "flat_semantic" values, or ISICClassifier's "keyword" / "llm",
or ISCEDClassifier's "keyword" / "rule") -- changing those would be a breaking
change to existing API responses, which is explicitly out of scope.
"""

from __future__ import annotations

# ISCO-08: genuinely implemented, tested 4-stage hierarchical RAG
# (backend/rag/hierarchical_store.py + backend/rag/hierarchy_engine.py).
ISCO_HIERARCHICAL_RAG = "isco_hierarchical_rag"

# ISIC Rev.4: currently implemented method (keyword lookup over a flat
# leaf-path table, with optional CrewAI LLM re-ranking below the keyword
# confidence threshold) -- see backend/agents/isic_classifier.py.
ISIC_KEYWORD_LLM = "isic_keyword_llm"

# ISIC Rev.4: hierarchical retrieval (Section->Division->Group->Class,
# parent-filtered, analogous to ISCO's pipeline) is NOT YET IMPLEMENTED.
# No Qdrant collections, loader, or live retrieval exist for this today.
# Deferred to a later pass -- see Documentation/Conference_I_Reviewer_2/
# CLASSIFIER_METHOD_REGISTRY.md.
ISIC_HIERARCHICAL_RETRIEVAL = "isic_hierarchical_retrieval"

# ISCED 2011: currently implemented method (pure keyword/rule-based
# attainment-level classification, no LLM) -- see
# backend/agents/isced_classifier.py.
ISCED_RULE_KEYWORD = "isced_rule_keyword"

# ISCED-F 2013: hierarchy-aware field classification (Broad->Narrow->Detailed,
# parent-filtered) is NOT YET IMPLEMENTED. No Qdrant collections, loader, or
# live retrieval exist for this today. Deferred to a later pass.
ISCEDF_HIERARCHICAL_RETRIEVAL = "iscedf_hierarchical_retrieval"

# Method labels that exist ONLY as documented, tested "not implemented"
# stubs today. A classifier asked to run one of these returns a structured
# not-implemented result (see ISICClassifier.classify(method=...) /
# ISCEDClassifier.classify(method=...)) rather than silently falling back to
# its default behaviour or fabricating a result.
NOT_IMPLEMENTED_METHODS: frozenset[str] = frozenset({
    ISIC_HIERARCHICAL_RETRIEVAL,
    ISCEDF_HIERARCHICAL_RETRIEVAL,
})

NOT_IMPLEMENTED_REASON = (
    "Hierarchical retrieval for this standard is deferred scope: no Qdrant "
    "collections, loader, or live multi-stage retrieval exist yet. See "
    "Documentation/Conference_I_Reviewer_2/CLASSIFIER_METHOD_REGISTRY.md."
)
