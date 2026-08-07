"""
Tests for backend/rag/hierarchy_nodes.py -- deterministic derivation of
Qdrant-indexable hierarchy nodes from the existing embedded ISIC/ISCED-F
classifier tables. Pure, offline, no Qdrant/embedding/network dependency.
"""

from __future__ import annotations

import pytest

from backend.rag.hierarchy_nodes import (
    HierarchyNode,
    HierarchyValidationError,
    ISCEDF_CODE_PATTERNS,
    ISCEDF_LEVELS,
    ISIC_CODE_PATTERNS,
    ISIC_LEVELS,
    _validate_common,
    derive_iscedf_nodes,
    derive_isic_nodes,
    validate_iscedf_nodes,
    validate_isic_nodes,
)


# ---------------------------------------------------------------------------
# 1. Deterministic ISIC node derivation -- exact counts
# ---------------------------------------------------------------------------

def test_derive_isic_nodes_exact_counts():
    nodes = derive_isic_nodes()
    assert len(nodes["sections"]) == 21
    assert len(nodes["divisions"]) == 68
    assert len(nodes["groups"]) == 118
    assert len(nodes["classes"]) == 134


def test_derive_isic_nodes_is_deterministic():
    a = derive_isic_nodes()
    b = derive_isic_nodes()
    assert {n.code for n in a["classes"]} == {n.code for n in b["classes"]}
    assert [n.code for n in a["sections"]] == [n.code for n in b["sections"]]


def test_derive_isic_nodes_validates_cleanly():
    # derive_isic_nodes() already calls validate_isic_nodes() internally --
    # this just re-asserts it doesn't raise and the result is well-formed.
    nodes = derive_isic_nodes()
    validate_isic_nodes(nodes)  # must not raise
    for level in ISIC_LEVELS:
        assert nodes[level], f"level {level!r} must not be empty"


# ---------------------------------------------------------------------------
# 2. Deterministic ISCED-F node derivation -- exact counts
# ---------------------------------------------------------------------------

def test_derive_iscedf_nodes_exact_counts():
    nodes = derive_iscedf_nodes()
    assert len(nodes["broad_fields"]) == 11
    assert len(nodes["narrow_fields"]) == 25
    assert len(nodes["detailed_fields"]) == 63


def test_derive_iscedf_nodes_is_deterministic():
    a = derive_iscedf_nodes()
    b = derive_iscedf_nodes()
    assert {n.code for n in a["detailed_fields"]} == {n.code for n in b["detailed_fields"]}


def test_derive_iscedf_nodes_validates_cleanly():
    nodes = derive_iscedf_nodes()
    validate_iscedf_nodes(nodes)  # must not raise
    for level in ISCEDF_LEVELS:
        assert nodes[level], f"level {level!r} must not be empty"


# ---------------------------------------------------------------------------
# 3. Fail-closed validation: malformed/conflicting/parent-missing records
# ---------------------------------------------------------------------------

_TWO_LEVEL = ("roots", "leaves")
_TWO_LEVEL_PATTERNS = {
    "roots": ISIC_CODE_PATTERNS["sections"],    # ^[A-Z]$
    "leaves": ISIC_CODE_PATTERNS["divisions"],  # ^\d{2}$
}


def _node(code, parent_code, label_en, level):
    return HierarchyNode(code=code, parent_code=parent_code, label_en=label_en, label_ar="", index_text=label_en, level=level)


def test_validate_common_rejects_empty_level():
    nodes = {"roots": [_node("A", "", "Root A", "root")], "leaves": []}
    with pytest.raises(HierarchyValidationError, match="zero derived nodes"):
        _validate_common(nodes, _TWO_LEVEL, _TWO_LEVEL_PATTERNS, standard="TEST")


def test_validate_common_rejects_malformed_code():
    nodes = {
        "roots": [_node("A", "", "Root A", "root")],
        "leaves": [_node("1", "A", "Bad code", "leaf")],  # pattern requires 2 digits
    }
    with pytest.raises(HierarchyValidationError, match="malformed code"):
        _validate_common(nodes, _TWO_LEVEL, _TWO_LEVEL_PATTERNS, standard="TEST")


def test_validate_common_rejects_duplicate_code():
    nodes = {
        "roots": [_node("A", "", "Root A", "root")],
        "leaves": [_node("01", "A", "Leaf 1", "leaf"), _node("01", "A", "Leaf 1 dup", "leaf")],
    }
    with pytest.raises(HierarchyValidationError, match="duplicate code"):
        _validate_common(nodes, _TWO_LEVEL, _TWO_LEVEL_PATTERNS, standard="TEST")


def test_validate_common_rejects_missing_parent():
    nodes = {
        "roots": [_node("A", "", "Root A", "root")],
        "leaves": [_node("01", "Z", "Orphan leaf", "leaf")],  # parent "Z" does not exist
    }
    with pytest.raises(HierarchyValidationError, match="does not exist in level"):
        _validate_common(nodes, _TWO_LEVEL, _TWO_LEVEL_PATTERNS, standard="TEST")


def test_validate_common_accepts_well_formed_hierarchy():
    nodes = {
        "roots": [_node("A", "", "Root A", "root")],
        "leaves": [_node("01", "A", "Leaf 1", "leaf"), _node("02", "A", "Leaf 2", "leaf")],
    }
    _validate_common(nodes, _TWO_LEVEL, _TWO_LEVEL_PATTERNS, standard="TEST")  # must not raise
