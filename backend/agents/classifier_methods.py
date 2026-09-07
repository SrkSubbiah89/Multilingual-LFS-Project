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
# standard_hierarchical_store.py). The isic_rev4_* Qdrant collections were
# built and populated 2026-08-23 (backend/rag/
# build_standard_hierarchical_collections.py --standard isic --execute,
# 341 nodes: 21/68/118/134 sections/divisions/groups/classes) -- this path
# now genuinely fires live, confirmed directly (method="isic_hierarchical_
# retrieval", fallback_used=False on a real query). Accuracy is still NOT
# YET EVALUATED against a labelled test set -- that remains open. If the
# store ever returns no usable result (e.g. Qdrant unreachable), classify()
# reports one of the ISIC_HIERARCHICAL_FALLBACK_* labels below instead,
# never silently relabels a fallback result as this constant.
ISIC_HIERARCHICAL_RETRIEVAL = "isic_hierarchical_retrieval"

# ISIC Rev.4: explicit fallback labels used when the hierarchical-retrieval
# path (above) was requested but the collections were unavailable or the
# search returned nothing. The classifier falls back to its existing
# keyword/LLM pipeline but reports one of these -- never silently relabels a
# keyword/LLM result as isic_hierarchical_retrieval.
ISIC_HIERARCHICAL_FALLBACK_KEYWORD = "isic_hierarchical_fallback_keyword"
ISIC_HIERARCHICAL_FALLBACK_LLM = "isic_hierarchical_fallback_llm"

# ISIC Rev.4: FLAT retrieval (added 2026-08-25) -- single-collection direct
# search over the 134 leaf classes, no parent-chain beam traversal. This is
# the architecturally-identical counterpart to ISCO-08's own BEST-TESTED
# configuration (flat retrieval + rich catalogue text + multilingual-e5-large
# -- see CLAUDE.md's 40.95% headline result). Uses backend/rag/
# standard_hierarchical_store.py's StandardFlatStore(profile="enriched_e5large")
# -- **corrected 2026-08-27 (code review): this comment previously said
# "e5_large" and claimed ISIC's catalogue text needed no enrichment; both
# were true only until the enrichment work later the same day (2026-08-25)
# found the richness gap WAS real relative to ISCO-08's actual enriched
# state (not its original bug state) and built
# backend/rag/official_source_enrichment.py's real official-text
# enrichment for ISIC too -- see CLAUDE.md's "Knowledge base construction"
# log for the full finding. This comment was never updated when that
# landed; now corrected.** The 13 ISIC codes with no official-document
# match (NON_STANDARD_ISIC_CODES) are excluded from this collection
# entirely, not indexed with weaker text -- a live-tested magnet-effect
# regression (see CLAUDE.md) showed keeping them caused active
# misclassification. Whether this actually outperforms the keyword/LLM
# pipeline for ISIC beyond the synthetic-benchmark signal already measured
# (see CLAUDE.md's 82.85%-vs-13.14% synthetic result) is not yet confirmed
# on real respondent data (see
# Documentation/Conference_I_Reviewer_2/
# ISIC_ISCEDF_HIERARCHICAL_RETRIEVAL_IMPLEMENTATION.md). This gives ISIC the
# same IMPLEMENTATION ISCO-08 uses, not a WISCO-equivalent accuracy result.
ISIC_FLAT_RETRIEVAL = "isic_flat_retrieval"
ISIC_FLAT_FALLBACK_KEYWORD = "isic_flat_fallback_keyword"
ISIC_FLAT_FALLBACK_LLM = "isic_flat_fallback_llm"

# ISCED 2011: currently implemented method (pure keyword/rule-based
# attainment-level classification, no LLM) -- see
# backend/agents/isced_classifier.py.
ISCED_RULE_KEYWORD = "isced_rule_keyword"

# ISCED-F 2013: hierarchy-aware field classification (Broad->Narrow->Detailed,
# parent-filtered). The iscedf2013_* Qdrant collections were built and
# populated 2026-08-23 (99 nodes: 11/25/63 broad/narrow/detailed fields) --
# same now-live, still-unevaluated-for-accuracy status as
# ISIC_HIERARCHICAL_RETRIEVAL above, confirmed directly the same way. ISCED
# 2011 attainment LEVEL is never part of this path -- it stays independently
# classified either way (see ISCEDClassifier.classify()'s docstring).
ISCEDF_HIERARCHICAL_RETRIEVAL = "iscedf_hierarchical_retrieval"

# ISCED-F 2013: explicit fallback label, same contract as the ISIC fallback
# labels above (ISCEDClassifier has no LLM path, so there is only one).
ISCEDF_HIERARCHICAL_FALLBACK_KEYWORD = "iscedf_hierarchical_fallback_keyword"

# ISCED-F 2013: FLAT retrieval (added 2026-08-25) -- same rationale, same
# StandardFlatStore(profile="enriched_e5large") + real official-text
# enrichment, and same exclusion of NON_STANDARD_ISCEDF_CODES, as
# ISIC_FLAT_RETRIEVAL above (see its comment for the full, corrected
# writeup): single-collection direct search over the 61 indexed leaf
# detailed fields (63 minus 2 excluded non-standard codes), paired with
# multilingual-e5-large, mirroring ISCO-08's own best-tested flat+e5-large
# recipe. ISCED 2011 attainment LEVEL is never part of this path -- it
# stays independently classified either way, same as the hierarchical
# path above.
ISCEDF_FLAT_RETRIEVAL = "iscedf_flat_retrieval"
ISCEDF_FLAT_FALLBACK_KEYWORD = "iscedf_flat_fallback_keyword"
