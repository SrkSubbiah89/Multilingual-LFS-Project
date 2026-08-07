"""
Tests for backend/rag/hierarchy_engine.py -- the generic, stage-count-agnostic
beam-search engine extracted from HierarchicalISCOStore._hierarchical_search.

No live Qdrant/embedding model is involved: FakeQdrantClient stands in for
qdrant_client.QdrantClient, driven by a lookup table keyed by
(collection, parent_code) -> list of (code, label_en, label_ar, score) rows,
mirroring the (collection_name, query_filter, limit) shape the real
``query_points()`` call receives.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.rag.hierarchy_engine import (
    HierarchyBeamSearchEngine,
    SeedSpec,
    StageConfig,
    StageOverride,
)


class FakeQdrantClient:
    """table: dict[(collection, parent_code_or_None)] -> list[(code, label_en, label_ar, score)]"""

    def __init__(self, table):
        self.table = table
        self.calls = []  # (collection_name, parent_code, limit) per call, in order

    def query_points(self, collection_name, query, query_filter, limit, with_payload):
        parent_code = None
        if query_filter is not None:
            parent_code = query_filter.must[0].match.value
        self.calls.append((collection_name, parent_code, limit))
        rows = self.table.get((collection_name, parent_code), [])
        points = [
            SimpleNamespace(score=score, payload={"code": code, "label_en": label_en, "label_ar": label_ar})
            for code, label_en, label_ar, score in rows[:limit]
        ]
        return SimpleNamespace(points=points)


def make_engine(table, stages, hitl_threshold=0.70):
    return HierarchyBeamSearchEngine(FakeQdrantClient(table), stages, hitl_threshold)


TWO_STAGE = [
    StageConfig(name="root", collection="col_root", weight=0.4),
    StageConfig(name="leaf", collection="col_leaf", weight=0.6),
]

THREE_STAGE = [
    StageConfig(name="a", collection="col_a", weight=0.2),
    StageConfig(name="b", collection="col_b", weight=0.3),
    StageConfig(name="c", collection="col_c", weight=0.5),
]

FOUR_STAGE = [
    StageConfig(name="major", collection="col_major", weight=0.10),
    StageConfig(name="submajor", collection="col_submajor", weight=0.20),
    StageConfig(name="minor", collection="col_minor", weight=0.20),
    StageConfig(name="unit", collection="col_unit", weight=0.50),
]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_requires_at_least_two_stages():
    with pytest.raises(ValueError):
        HierarchyBeamSearchEngine(FakeQdrantClient({}), [StageConfig("only", "col", 1.0)], 0.70)


def test_requires_at_least_two_stages_empty_list():
    with pytest.raises(ValueError):
        HierarchyBeamSearchEngine(FakeQdrantClient({}), [], 0.70)


# ---------------------------------------------------------------------------
# Zero-hit fallthrough
# ---------------------------------------------------------------------------

def test_no_stage0_hits_returns_none():
    engine = make_engine({}, TWO_STAGE)
    assert engine.search(query_vec=[0.0], top_k=3) is None


def test_no_final_stage_hits_returns_none():
    table = {("col_root", None): [("r1", "Root 1", "", 0.9)]}
    engine = make_engine(table, TWO_STAGE)  # col_leaf has no rows for parent "r1"
    assert engine.search(query_vec=[0.0], top_k=3) is None


# ---------------------------------------------------------------------------
# Basic 2/3/4-stage search + weighted confidence arithmetic
# ---------------------------------------------------------------------------

def test_two_stage_search_basic():
    table = {
        ("col_root", None): [("r1", "Root 1", "", 0.9)],
        ("col_leaf", "r1"): [("r1-1", "Leaf 1", "", 0.8)],
    }
    engine = make_engine(table, TWO_STAGE)
    result = engine.search(query_vec=[0.0], top_k=3, beam=2)
    assert result is not None
    assert result.code == "r1-1"
    assert result.hierarchy_path == ["r1", "r1-1"]
    # confidence = 0.4*0.9 + 0.6*0.8 = 0.84
    assert result.confidence == pytest.approx(0.84)
    assert result.stage_confidences == {"stage1": 0.9, "stage2": 0.8}


def test_three_stage_search_weighted_confidence():
    table = {
        ("col_a", None): [("a1", "A1", "", 0.5)],
        ("col_b", "a1"): [("b1", "B1", "", 0.6)],
        ("col_c", "b1"): [("c1", "C1", "", 0.7)],
    }
    engine = make_engine(table, THREE_STAGE)
    result = engine.search(query_vec=[0.0], top_k=3, beam=2)
    assert result.code == "c1"
    assert result.hierarchy_path == ["a1", "b1", "c1"]
    # 0.2*0.5 + 0.3*0.6 + 0.5*0.7 = 0.63
    assert result.confidence == pytest.approx(0.63)


def test_four_stage_search_matches_isco_weights():
    table = {
        ("col_major", None): [("2", "Professionals", "", 0.85)],
        ("col_submajor", "2"): [("25", "ICT", "", 0.80)],
        ("col_minor", "25"): [("251", "Software", "", 0.75)],
        ("col_unit", "251"): [("2512", "Software Developer", "", 0.90)],
    }
    engine = make_engine(table, FOUR_STAGE)
    result = engine.search(query_vec=[0.0], top_k=5, beam=2)
    assert result.code == "2512"
    assert result.hierarchy_path == ["2", "25", "251", "2512"]
    expected = round(0.10 * 0.85 + 0.20 * 0.80 + 0.20 * 0.75 + 0.50 * 0.90, 4)
    assert result.confidence == pytest.approx(expected)


def test_confidence_clamped_to_one():
    table = {
        ("col_root", None): [("r1", "Root 1", "", 1.0)],
        ("col_leaf", "r1"): [("r1-1", "Leaf 1", "", 1.0)],
    }
    engine = make_engine(table, TWO_STAGE)
    result = engine.search(query_vec=[0.0], top_k=3)
    assert result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Beam width: multiple stage-0 branches, best final score wins
# ---------------------------------------------------------------------------

def test_beam_explores_multiple_branches_and_picks_best_final_score():
    table = {
        ("col_root", None): [("r1", "Root 1", "", 0.9), ("r2", "Root 2", "", 0.85)],
        ("col_leaf", "r1"): [("r1-1", "Leaf 1", "", 0.60)],
        ("col_leaf", "r2"): [("r2-1", "Leaf 2", "", 0.95)],  # weaker root, stronger leaf
    }
    engine = make_engine(table, TWO_STAGE)
    result = engine.search(query_vec=[0.0], top_k=3, beam=2)
    # r2 branch's final score (0.95) beats r1 branch's (0.60), even though
    # r1's own stage-0 score was higher -- selection is by final-stage score.
    assert result.code == "r2-1"
    assert result.hierarchy_path == ["r2", "r2-1"]


def test_beam_1_only_explores_top_branch():
    table = {
        ("col_root", None): [("r1", "Root 1", "", 0.9), ("r2", "Root 2", "", 0.85)],
        ("col_leaf", "r1"): [("r1-1", "Leaf 1", "", 0.60)],
        ("col_leaf", "r2"): [("r2-1", "Leaf 2", "", 0.95)],
    }
    engine = make_engine(table, TWO_STAGE)
    result = engine.search(query_vec=[0.0], top_k=3, beam=1)
    # beam=1 -> stage 0 query itself is limited to 1 result -> only r1 explored
    assert result.code == "r1-1"


# ---------------------------------------------------------------------------
# branch_collapse: global pool vs winning-branch-only
# ---------------------------------------------------------------------------

def test_branch_collapse_false_pools_candidates_across_branches():
    table = {
        ("col_root", None): [("r1", "Root 1", "", 0.9), ("r2", "Root 2", "", 0.85)],
        ("col_leaf", "r1"): [("r1-1", "Leaf 1", "", 0.800)],
        ("col_leaf", "r2"): [("r2-1", "Leaf 2", "", 0.792)],
    }
    engine = make_engine(table, TWO_STAGE)
    result = engine.search(query_vec=[0.0], top_k=3, beam=2, branch_collapse=False, reranker_candidates=5)
    codes = {c.code for c in result.top_candidates}
    # global pool includes the losing branch's competitive candidate
    assert codes == {"r1-1", "r2-1"}


def test_branch_collapse_true_discards_losing_branch_candidates():
    table = {
        ("col_root", None): [("r1", "Root 1", "", 0.9), ("r2", "Root 2", "", 0.85)],
        ("col_leaf", "r1"): [("r1-1", "Leaf 1", "", 0.800)],
        ("col_leaf", "r2"): [("r2-1", "Leaf 2", "", 0.792)],
    }
    engine = make_engine(table, TWO_STAGE)
    result = engine.search(query_vec=[0.0], top_k=3, beam=2, branch_collapse=True, reranker_candidates=5)
    codes = {c.code for c in result.top_candidates}
    assert codes == {"r1-1"}


def test_pool_dedupes_by_code_keeping_highest_score():
    table = {
        ("col_root", None): [("r1", "Root 1", "", 0.9), ("r2", "Root 2", "", 0.85)],
        ("col_leaf", "r1"): [("shared", "Shared", "", 0.70)],
        ("col_leaf", "r2"): [("shared", "Shared", "", 0.792)],
    }
    engine = make_engine(table, TWO_STAGE)
    result = engine.search(query_vec=[0.0], top_k=3, beam=2, branch_collapse=False)
    assert len(result.top_candidates) == 1
    assert result.top_candidates[0].score == pytest.approx(0.792)


def test_reranker_candidates_truncates_pool():
    table = {
        ("col_root", None): [("r1", "R1", "", 0.9)],
        ("col_leaf", "r1"): [
            ("u1", "U1", "", 0.9), ("u2", "U2", "", 0.8),
            ("u3", "U3", "", 0.7), ("u4", "U4", "", 0.6),
        ],
    }
    engine = make_engine(table, TWO_STAGE)
    result = engine.search(query_vec=[0.0], top_k=4, beam=2, reranker_candidates=2)
    assert len(result.top_candidates) == 2
    assert [c.code for c in result.top_candidates] == ["u1", "u2"]


# ---------------------------------------------------------------------------
# seed: bypasses stage-0 query entirely
# ---------------------------------------------------------------------------

def test_seed_bypasses_stage0_query():
    table = {("col_leaf", "seeded"): [("leaf1", "Leaf", "", 0.77)]}
    engine = make_engine(table, TWO_STAGE)
    seed = SeedSpec(candidates=[("seeded", 1.0)], source_label="keyword_map")
    result = engine.search(query_vec=[0.0], top_k=3, seed=seed)
    assert result.code == "leaf1"
    assert result.hierarchy_path == ["seeded", "leaf1"]
    # stage-0 collection was never queried
    assert all(call[0] != "col_root" for call in engine._client.calls)


def test_seed_with_no_candidates_returns_none():
    engine = make_engine({}, TWO_STAGE)
    seed = SeedSpec(candidates=[])
    assert engine.search(query_vec=[0.0], top_k=3, seed=seed) is None


def test_seed_populates_trace_stage1_source():
    table = {("col_leaf", "seeded"): [("leaf1", "Leaf", "", 0.77)]}
    engine = make_engine(table, TWO_STAGE)
    seed = SeedSpec(candidates=[("seeded", 1.0)], source_label="keyword_map",
                     candidate_label="(keyword hint, search skipped)")
    trace = {}
    engine.search(query_vec=[0.0], top_k=3, seed=seed, trace=trace)
    assert trace["stage1_source"] == "keyword_map"
    assert trace["stage1"] == [{"code": "seeded", "label_en": "(keyword hint, search skipped)", "score": 1.0}]
    assert trace["stage1_latency_ms"] == 0.0


# ---------------------------------------------------------------------------
# stage_overrides: replaces a stage's normal query with a callable
# ---------------------------------------------------------------------------

def test_stage_override_replaces_stage0_query():
    table = {("col_leaf", "override_code"): [("leaf1", "Leaf", "", 0.66)]}
    engine = make_engine(table, TWO_STAGE)
    override = StageOverride(
        fn=lambda qv, beam: [("override_code", 0.5)],
        source_label="leaf_vote",
        candidate_label="(override aggregate)",
    )
    trace = {}
    result = engine.search(query_vec=[0.0], top_k=3, stage_overrides={0: override}, trace=trace)
    assert result.code == "leaf1"
    assert result.hierarchy_path == ["override_code", "leaf1"]
    assert trace["stage1_source"] == "leaf_vote"
    assert all(call[0] != "col_root" for call in engine._client.calls)


def test_stage_override_empty_candidates_returns_none():
    engine = make_engine({}, TWO_STAGE)
    override = StageOverride(fn=lambda qv, beam: [])
    assert engine.search(query_vec=[0.0], top_k=3, stage_overrides={0: override}) is None


# ---------------------------------------------------------------------------
# trace: generic stage keys, latency accumulation, pool metadata
# ---------------------------------------------------------------------------

def test_trace_populates_generic_stage_keys_for_n_stages():
    table = {
        ("col_a", None): [("a1", "A1", "", 0.5)],
        ("col_b", "a1"): [("b1", "B1", "", 0.6)],
        ("col_c", "b1"): [("c1", "C1", "", 0.7)],
    }
    engine = make_engine(table, THREE_STAGE)
    trace = {}
    engine.search(query_vec=[0.0], top_k=3, trace=trace)
    for i in range(1, 4):
        assert f"stage{i}" in trace
        assert f"stage{i}_latency_ms" in trace
    assert trace["stage1_source"] == "semantic_retrieval"
    assert trace["reranker_candidate_pool_size"] == 1
    assert trace["reranker_candidate_branches"] == 1
    assert "stage3_pool" in trace  # final stage is stage3 for a 3-stage config


def test_trace_capture_pool_metadata_populates_enriched_pool():
    table = {
        ("col_root", None): [("r1", "R1", "", 0.9)],
        ("col_leaf", "r1"): [("u1", "U1", "", 0.8)],
    }
    engine = make_engine(table, TWO_STAGE)
    trace = {}
    engine.search(query_vec=[0.0], top_k=3, capture_pool_metadata=True, trace=trace)
    enriched = trace["stage2_pool_enriched"]
    assert len(enriched) == 1
    assert enriched[0]["code"] == "u1"
    assert enriched[0]["branch_id"] == "r1"
    assert enriched[0]["source_rank"] == 1
    assert enriched[0]["path"] == ["r1", "u1"]


def test_capture_pool_metadata_false_omits_enriched_key():
    table = {
        ("col_root", None): [("r1", "R1", "", 0.9)],
        ("col_leaf", "r1"): [("u1", "U1", "", 0.8)],
    }
    engine = make_engine(table, TWO_STAGE)
    trace = {}
    engine.search(query_vec=[0.0], top_k=3, capture_pool_metadata=False, trace=trace)
    assert "stage2_pool_enriched" not in trace


def test_no_trace_dict_does_not_raise():
    table = {
        ("col_root", None): [("r1", "R1", "", 0.9)],
        ("col_leaf", "r1"): [("u1", "U1", "", 0.8)],
    }
    engine = make_engine(table, TWO_STAGE)
    result = engine.search(query_vec=[0.0], top_k=3, trace=None)
    assert result is not None


# ---------------------------------------------------------------------------
# HITL threshold
# ---------------------------------------------------------------------------

def test_hitl_required_below_threshold():
    table = {
        ("col_root", None): [("r1", "R1", "", 0.3)],
        ("col_leaf", "r1"): [("u1", "U1", "", 0.3)],
    }
    engine = make_engine(table, TWO_STAGE, hitl_threshold=0.70)
    result = engine.search(query_vec=[0.0], top_k=3)
    assert result.hitl_required is True


def test_hitl_not_required_above_threshold():
    table = {
        ("col_root", None): [("r1", "R1", "", 0.95)],
        ("col_leaf", "r1"): [("u1", "U1", "", 0.95)],
    }
    engine = make_engine(table, TWO_STAGE, hitl_threshold=0.70)
    result = engine.search(query_vec=[0.0], top_k=3)
    assert result.hitl_required is False


def test_fallback_used_is_always_false_from_engine():
    # fallback_used is a caller-level concept (flat-collection fallback);
    # the engine itself never sets it True -- callers decide when to fall back.
    table = {
        ("col_root", None): [("r1", "R1", "", 0.9)],
        ("col_leaf", "r1"): [("u1", "U1", "", 0.8)],
    }
    engine = make_engine(table, TWO_STAGE)
    result = engine.search(query_vec=[0.0], top_k=3)
    assert result.fallback_used is False
