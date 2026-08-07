"""
backend/agents/classifier_methods.py

Single source of truth for the classification-method label strings used by
Documentation/Conference_I_Reviewer_2's classifier method registry
(Section A) and by ISIC/ISCED's hierarchical-retrieval and explicit-fallback
labelling (Task 05 -- see Documentation/Conference_I_Reviewer_2/
ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md for the full writeup).

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
# parent-filtered, built on the same generic backend.rag.hierarchy_engine
# ISCO uses) -- code path is real and tested (backend/rag/
# standard_hierarchical_store.py), but NOT YET EVALUATED for accuracy, and
# only produces a live result once an operator has built the
# isic_rev4_* Qdrant collections (backend/rag/
# build_standard_hierarchical_collections.py --standard isic --execute).
# Until then -- or whenever the store returns no usable result -- classify()
# reports one of the ISIC_HIERARCHICAL_FALLBACK_* labels below instead, never
# this constant.
ISIC_HIERARCHICAL_RETRIEVAL = "isic_hierarchical_retrieval"

# ISIC Rev.4: explicit fallback labels used when the hierarchical-retrieval
# path (above) was requested but the collections were unavailable or the
# search returned nothing. The classifier falls back to its existing
# keyword/LLM pipeline but reports one of these -- never silently relabels a
# keyword/LLM result as isic_hierarchical_retrieval.
ISIC_HIERARCHICAL_FALLBACK_KEYWORD = "isic_hierarchical_fallback_keyword"
ISIC_HIERARCHICAL_FALLBACK_LLM = "isic_hierarchical_fallback_llm"

# ISCED 2011: currently implemented method (pure keyword/rule-based
# attainment-level classification, no LLM) -- see
# backend/agents/isced_classifier.py.
ISCED_RULE_KEYWORD = "isced_rule_keyword"

# ISCED-F 2013: hierarchy-aware field classification (Broad->Narrow->Detailed,
# parent-filtered), same real-but-unevaluated status as
# ISIC_HIERARCHICAL_RETRIEVAL above. ISCED 2011 attainment LEVEL is never
# part of this path -- it stays independently classified either way (see
# ISCEDClassifier.classify()'s docstring).
ISCEDF_HIERARCHICAL_RETRIEVAL = "iscedf_hierarchical_retrieval"

# ISCED-F 2013: explicit fallback label, same contract as the ISIC fallback
# labels above (ISCEDClassifier has no LLM path, so there is only one).
ISCEDF_HIERARCHICAL_FALLBACK_KEYWORD = "iscedf_hierarchical_fallback_keyword"
