"""Hermetic retrieval regressions: no encoder, live Qdrant, or labelled data."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from backend.agents.classifier_methods import (
    ISCO_HYBRID_RRF,
    ISCO_HYBRID_SOFT_HIERARCHY_RRF,
    ISCO_HYBRID_SPARSE_FALLBACK,
    ISCO_HYBRID_UNAVAILABLE,
)
from backend.rag.hybrid_isco import HybridISCORetriever, unicode_tokens
from backend.rag.official_isco08_catalogue import (
    DEFAULT_PROFILE, ENRICHED_PROFILE, OfficialCatalogueRecord, PROFILE_COLLECTION_NAMES,
)

HASH = "a" * 64


def catalogue(profile=DEFAULT_PROFILE):
    rows = [
        ("major", "2", "", "Professionals"),
        ("major", "5", "", "Service Workers"),
        ("submajor", "25", "2", "ICT Professionals"),
        ("submajor", "23", "2", "Teaching Professionals"),
        ("submajor", "51", "5", "Personal Service Workers"),
        ("minor", "251", "25", "Software Developers"),
        ("minor", "234", "23", "School Teachers"),
        ("minor", "512", "51", "Chefs"),
        ("unit", "2512", "251", "Software Developers"),
        ("unit", "2341", "234", "Primary School Teachers"),
        ("unit", "5120", "512", "Chefs"),
    ]
    return [OfficialCatalogueRecord(
        code=code, level=level, parent_code=parent, title_en=title,
        embedding_text=f"{code} {title}", profile=profile, source_catalogue_sha256=HASH,
    ) for level, code, parent, title in rows]


def dense_scores():
    # An intentionally wrong major winner must never prune software globally.
    return {
        "major": [("5", 0.96), ("2", 0.70)],
        "submajor": [("51", 0.95), ("25", 0.75), ("23", 0.60)],
        "minor": [("512", 0.94), ("251", 0.74), ("234", 0.50)],
        "unit": [("5120", 0.90), ("2512", 0.89), ("2341", 0.40)],
    }


def make_retriever(**kwargs):
    return HybridISCORetriever(catalogue(), profile=DEFAULT_PROFILE, **kwargs)


def test_actual_rrf_formula_uses_one_based_ranks_and_keeps_raw_cosine_separate():
    retriever = make_retriever(hierarchy_weight=0)
    trace = {}
    result = retriever.rank("software developers", dense_scores_by_level=dense_scores(), trace=trace)
    assert result.method == ISCO_HYBRID_RRF
    assert result.code == "2512"
    assert result.ranking_score == pytest.approx(1 / 62 + 1 / 61)
    candidate = result.top_candidates[0]
    assert candidate.dense_score == 0.89
    assert (candidate.dense_rank, candidate.lexical_rank) == (2, 1)
    assert candidate.fusion_score != candidate.dense_score
    assert trace["ranking_score_calibrated"] is False
    assert trace["rrf_rank_indexing"] == "one_based"
    assert trace["parameters"]["rrf_k"] == 60
    assert trace["lexical_fields"] == "title_only"
    assert not hasattr(result, "confidence")


def test_soft_hierarchy_recovers_global_leaf_from_wrong_major_without_hard_filters():
    result = make_retriever().rank("software", dense_scores_by_level=dense_scores())
    assert result.method == ISCO_HYBRID_SOFT_HIERARCHY_RRF
    assert result.code == "2512"
    assert result.hierarchy_path == ["2", "25", "251", "2512"]
    assert {candidate.code for candidate in result.top_candidates} == {"2512", "5120", "2341"}
    assert result.top_candidates[0].hierarchy_support == pytest.approx(1 / 62)


def test_soft_paths_do_not_change_candidate_membership_even_with_large_wrong_support():
    result = make_retriever(hierarchy_weight=100).rank(
        "software", dense_scores_by_level=dense_scores(), top_k=3,
    )
    assert {candidate.code for candidate in result.top_candidates} == {"2512", "5120", "2341"}


def test_positive_lexical_leaf_can_escape_dense_top_k_pool():
    result = make_retriever(candidate_k=1, hierarchy_weight=0).rank(
        "software", dense_scores_by_level=dense_scores(), top_k=3,
    )
    assert {candidate.code for candidate in result.top_candidates} == {"2512", "5120"}
    lexical_escape = next(candidate for candidate in result.top_candidates if candidate.code == "2512")
    assert lexical_escape.dense_rank is None
    assert lexical_escape.lexical_rank == 1
    assert lexical_escape.dense_score == 0.89


@pytest.mark.parametrize("query", ["مهندس برمجيات", "सॉफ्टवेयर डेवलपर", "سافٹ ویئر انجینئر", "zzzzunknown"])
def test_absent_lexical_overlap_preserves_dense_order_without_arbitrary_lexical_votes(query):
    retriever = make_retriever(hierarchy_weight=0)
    trace = {}
    result = retriever.rank(query, dense_scores_by_level=dense_scores(), trace=trace)
    assert result.code == "5120"
    assert retriever.lexical_scores(query) == []
    assert trace["lexical_ranking"] == []
    assert trace["lexical_evidence_present"] is False
    assert all(candidate.lexical_rank is None and candidate.lexical_score == 0 for candidate in result.top_candidates)


def test_unicode_tokenizer_preserves_combining_marks_and_presentation_forms():
    assert unicode_tokens("हिन्दी nurse") == ["हिन्दी", "nurse"]
    assert unicode_tokens("مُهندس سافٹ") == ["مُهندس", "سافٹ"]
    assert unicode_tokens("ＳＯＦＴＷＡＲＥ Developer") == ["software", "developer"]


def test_catalogue_fields_are_separate_and_title_weight_is_effective():
    records = catalogue(ENRICHED_PROFILE)
    records = [replace(record, embedding_text=record.embedding_text + ". software")
               if record.code == "2341" else record for record in records]
    retriever = HybridISCORetriever(records, profile=ENRICHED_PROFILE, hierarchy_weight=0)
    lexical = dict(retriever.lexical_scores("software"))
    assert lexical["2512"] > lexical["2341"] > 0
    title_only = HybridISCORetriever(records, profile=ENRICHED_PROFILE, body_weight=0)
    assert {code for code, _ in title_only.lexical_scores("software")} == {"2512"}
    trace = {}
    retriever.rank("software", dense_scores_by_level=dense_scores(), trace=trace)
    assert trace["lexical_fields"] == "title_and_official_body"


def test_repeating_query_terms_does_not_manufacture_lexical_evidence():
    retriever = make_retriever()
    assert retriever.lexical_scores("software") == retriever.lexical_scores("software software software")


def test_missing_ancestor_scores_preserve_leaf_results_and_report_partial_support():
    trace = {}
    result = make_retriever().rank("software", dense_scores_by_level={"unit": dense_scores()["unit"]}, trace=trace)
    assert result.code == "2512"
    assert result.fallback_used is False
    assert result.top_candidates[0].hierarchy_support == 0
    assert trace["missing_dense_levels"] == ["major", "submajor", "minor"]


def test_sparse_fallback_explicit_and_no_arbitrary_answer_without_any_evidence():
    retriever = make_retriever()
    sparse = retriever.rank("software", dense_scores_by_level={})
    assert sparse.code == "2512"
    assert sparse.method == ISCO_HYBRID_SPARSE_FALLBACK
    assert sparse.requested_method == ISCO_HYBRID_SOFT_HIERARCHY_RRF
    assert sparse.fallback_used is True
    assert sparse.fallback_reason == "unit_dense_evidence_unavailable"
    assert sparse.top_candidates[0].dense_score is None
    unavailable = retriever.rank("zzzzunknown", dense_scores_by_level={})
    assert unavailable.method == ISCO_HYBRID_UNAVAILABLE
    assert unavailable.code == "" and unavailable.top_candidates == []
    assert unavailable.ranking_score == 0


def test_duplicate_dense_codes_are_deduplicated_and_input_order_has_no_effect():
    retriever = make_retriever(hierarchy_weight=0)
    first = retriever.rank("software", dense_scores_by_level=dense_scores())
    shuffled = {level: list(reversed(scores)) for level, scores in dense_scores().items()}
    shuffled["unit"].extend([("2512", 0.10), ("2512", 0.89)])
    second = retriever.rank("software", dense_scores_by_level=shuffled)
    assert first == second


@pytest.mark.parametrize("scores", [
    {"unit": [("9999", 0.8)]},
    {"unit": [("25", 0.8)]},
    {"unit": [("2512", float("nan"))]},
    {"unit": [("2512", float("inf"))]},
    {"unit": [("2512", True)]},
    {"unit": [("2512", "0.8")]},
    {"alien": []},
])
def test_malformed_or_foreign_dense_evidence_is_rejected(scores):
    with pytest.raises(ValueError):
        make_retriever().rank("software", dense_scores_by_level=scores)


def test_numpy_float_scores_are_supported_for_offline_matrix_evaluation():
    import numpy as np
    result = make_retriever(hierarchy_weight=0).rank(
        "software", dense_scores_by_level={"unit": [("2512", np.float32(0.8))]},
    )
    assert result.top_candidates[0].dense_score == pytest.approx(0.8)


@pytest.mark.parametrize("kwargs", [
    {"profile": "legacy"}, {"profile": "unknown"}, {"candidate_k": 0},
    {"candidate_k": True}, {"rrf_k": 0}, {"lexical_weight": -1},
    {"hierarchy_weight": float("nan")}, {"title_weight": True},
    {"lexical_weight": 0, "dense_weight": 0}, {"title_weight": 0, "body_weight": 0},
    {"query_timeout_seconds": 0},
])
def test_invalid_configuration_fails_closed(kwargs):
    with pytest.raises(ValueError):
        HybridISCORetriever(catalogue(), **({"profile": DEFAULT_PROFILE} | kwargs))


@pytest.mark.parametrize("change", ["duplicate", "foreign_profile", "foreign_digest", "missing_parent", "wrong_parent_level"])
def test_mixed_or_broken_catalogue_provenance_is_rejected(change):
    records = catalogue()
    if change == "duplicate":
        records.append(records[-1])
    elif change == "foreign_profile":
        records[-1] = replace(records[-1], profile=ENRICHED_PROFILE)
    elif change == "foreign_digest":
        records[-1] = replace(records[-1], source_catalogue_sha256="b" * 64)
    elif change == "missing_parent":
        records[-1] = replace(records[-1], parent_code="999")
    else:
        records[-1] = replace(records[-1], parent_code="51")
    with pytest.raises(ValueError):
        HybridISCORetriever(records, profile=DEFAULT_PROFILE)


class ReadOnlyClient:
    """Only a query method exists: mutations/model loading cannot be hidden."""

    def __init__(self, *, fail=(), payload_changes=None):
        self.calls = []
        self.fail = set(fail)
        self.payload_changes = payload_changes or {}

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        level = next(level for level, collection in PROFILE_COLLECTION_NAMES[DEFAULT_PROFILE].items()
                     if collection == kwargs["collection_name"])
        if level in self.fail:
            raise TimeoutError("simulated unavailable Qdrant query")
        return SimpleNamespace(points=[
            SimpleNamespace(score=score, payload={
                "code": code, "profile": DEFAULT_PROFILE, "source_catalogue_sha256": HASH,
                "title_en": "Untrusted replacement title", **self.payload_changes,
            }) for code, score in dense_scores()[level][:kwargs["limit"]]
        ])


def test_search_uses_one_global_unit_query_and_unfiltered_ancestor_queries_with_existing_vector():
    client = ReadOnlyClient()
    retriever = make_retriever(client=client, embed_query=lambda query: pytest.fail("No extra model call"))
    trace = {}
    result = retriever.search("software", query_vector=[0.0] * 384, trace=trace)
    assert result.code == "2512" and result.label_en == "Software Developers"
    assert len(client.calls) == 4
    assert all(call["query_filter"] is None for call in client.calls)
    assert all(call["timeout"] == 10 for call in client.calls)
    assert trace["retrieval_failures"] == {}
    assert trace["source_catalogue_sha256"] == HASH
    assert trace["embedding_model"] == "intfloat/multilingual-e5-small"


def test_search_embeds_raw_multilingual_query_exactly_once_using_injected_adapter():
    seen = []
    def embed(query):
        seen.append(query)
        return [0.0] * 384
    client = ReadOnlyClient()
    make_retriever(client=client, embed_query=embed).search("مهندس برمجيات")
    assert seen == ["مهندس برمجيات"]


def test_rrf_without_soft_hierarchy_makes_only_one_qdrant_query():
    client = ReadOnlyClient()
    result = make_retriever(client=client, hierarchy_weight=0).search("software", query_vector=[0.0] * 384)
    assert result.method == ISCO_HYBRID_RRF
    assert len(client.calls) == 1


def test_failed_dense_query_uses_explicit_sparse_fallback_not_parent_only_fabrication():
    client = ReadOnlyClient(fail={"unit"})
    trace = {}
    result = make_retriever(client=client).search("software", query_vector=[0.0] * 384, trace=trace)
    assert result.method == ISCO_HYBRID_SPARSE_FALLBACK and result.code == "2512"
    assert result.top_candidates[0].hierarchy_support == 0
    assert trace["retrieval_failures"] == {"unit": "TimeoutError"}


def test_failed_ancestor_does_not_discard_valid_dense_units():
    client = ReadOnlyClient(fail={"major", "minor"})
    trace = {}
    result = make_retriever(client=client).search("software", query_vector=[0.0] * 384, trace=trace)
    assert result.method == ISCO_HYBRID_SOFT_HIERARCHY_RRF and not result.fallback_used
    assert trace["retrieval_failures"] == {"major": "TimeoutError", "minor": "TimeoutError"}


@pytest.mark.parametrize("payload_changes", [
    {"profile": ENRICHED_PROFILE}, {"source_catalogue_sha256": "b" * 64}, {"code": "9999"},
])
def test_foreign_qdrant_payload_cannot_enter_selected_catalogue(payload_changes):
    trace = {}
    result = make_retriever(client=ReadOnlyClient(payload_changes=payload_changes)).search(
        "software", query_vector=[0.0] * 384, trace=trace,
    )
    assert result.method == ISCO_HYBRID_SPARSE_FALLBACK
    assert set(trace["retrieval_failures"]) == {"unit", "major", "submajor", "minor"}


def test_invalid_embedding_dimension_falls_back_before_any_dense_call():
    client = ReadOnlyClient()
    trace = {}
    result = make_retriever(client=client).search("software", query_vector=[1.0], trace=trace)
    assert result.method == ISCO_HYBRID_SPARSE_FALLBACK
    assert client.calls == []
    assert trace["retrieval_failures"] == {"embedding": "ValueError"}


def test_search_without_client_uses_explicit_sparse_fallback():
    trace = {}
    result = make_retriever().search("software", trace=trace)
    assert result.method == ISCO_HYBRID_SPARSE_FALLBACK
    assert trace["retrieval_failures"] == {"client": "Unavailable"}


@pytest.mark.parametrize("query", ["", " ", None])
def test_blank_query_rejected_before_retrieval(query):
    retriever = make_retriever()
    with pytest.raises(ValueError):
        retriever.rank(query, dense_scores_by_level={})
    with pytest.raises(ValueError):
        retriever.search(query)
    with pytest.raises(ValueError):
        retriever.lexical_scores(query)
