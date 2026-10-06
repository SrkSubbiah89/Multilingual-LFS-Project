"""Catalogue-only ridge algebra, query normalization and evidence guards."""
import numpy as np
import pytest

from backend.rag.catalogue_alignment import fit_catalogue_alignment, projected_parent_scores, combine_aligned_evidence


def basis(rows, dimension):
    return np.eye(dimension, dtype=np.float32)[:rows]


def test_ridge_fits_paired_catalogue_geometry_with_known_closed_form():
    small, large = basis(3, 384), basis(3, 1024)
    matrix = fit_catalogue_alignment(small, large, ridge=0.5)
    expected = np.zeros((384, 1024))
    expected[:3, :3] = np.eye(3) / 1.5
    np.testing.assert_allclose(matrix, expected)
    scores = projected_parent_scores(small, matrix, large)
    np.testing.assert_allclose(scores, np.eye(3), atol=1e-7)


def test_pair_permutation_does_not_change_catalogue_fit():
    small, large = basis(4, 384), basis(4, 1024)
    order = [2, 0, 3, 1]
    first = fit_catalogue_alignment(small, large, ridge=0.1)
    second = fit_catalogue_alignment(small[order], large[order], ridge=0.1)
    np.testing.assert_allclose(first, second)


def test_projected_query_scores_are_scale_invariant_after_normalization():
    matrix = fit_catalogue_alignment(basis(2, 384), basis(2, 1024), ridge=0.1)
    first = projected_parent_scores(basis(2, 384), matrix, basis(2, 1024))
    second = projected_parent_scores(basis(2, 384), matrix * 10, basis(2, 1024))
    np.testing.assert_allclose(first, second)


@pytest.mark.parametrize('ridge', [0, -1, True, '0.1', float('nan'), float('inf')])
def test_invalid_regularizers_do_not_produce_unreviewed_fit(ridge):
    with pytest.raises(ValueError, match='regularization'):
        fit_catalogue_alignment(basis(2, 384), basis(2, 1024), ridge=ridge)


@pytest.mark.parametrize('small,large', [
    (basis(2, 383), basis(2, 1024)), (basis(2, 384), basis(2, 1023)),
    (basis(2, 384), basis(3, 1024)), (basis(2, 384) * 2, basis(2, 1024)),
    (np.full((2, 384), float('nan')), basis(2, 1024)),
    (basis(2, 384), np.full((2, 1024), float('inf'))),
    (np.empty((0, 384)), np.empty((0, 1024))),
])
def test_incomplete_or_invalid_catalogue_pairs_are_rejected(small, large):
    with pytest.raises(ValueError):
        fit_catalogue_alignment(small, large, ridge=0.1)


@pytest.mark.parametrize('matrix', [np.zeros((384, 1024)), np.zeros((383, 1024)),
                                    np.full((384, 1024), float('nan'))])
def test_zero_or_invalid_projected_direction_is_rejected(matrix):
    with pytest.raises(ValueError):
        projected_parent_scores(basis(2, 384), matrix, basis(2, 1024))


def test_control_blend_is_bitwise_identical_to_frozen_half_child_half_small():
    child = np.array([[0.8384, 0.9073]], dtype=np.float32)
    small = np.array([[0.7319, 0.8728]], dtype=np.float32)
    actual = combine_aligned_evidence(child, small, [[-1, -1]], large_parent_weight=0)
    np.testing.assert_array_equal(actual, 0.5 * child + 0.5 * small)


@pytest.mark.parametrize('weight,expected', [(0, 0.5), (0.5, 0.65), (1, 0.8)])
def test_only_parent_half_is_blended_with_projected_large_geometry(weight, expected):
    scores = combine_aligned_evidence([[0.8]], [[0.2]], [[0.8]], large_parent_weight=weight)
    assert scores[0, 0] == pytest.approx(expected)


@pytest.mark.parametrize('weight', [-0.1, 1.1, True, '0.5', float('nan'), float('inf')])
def test_invalid_parent_blend_weights_are_rejected(weight):
    with pytest.raises(ValueError, match='blend weight'):
        combine_aligned_evidence([[0.8]], [[0.2]], [[0.8]], large_parent_weight=weight)


@pytest.mark.parametrize('child,small,large', [
    ([0.8], [0.2], [0.8]), ([[0.8]], [[0.2, 0.3]], [[0.8]]),
    ([[float('nan')]], [[0.2]], [[0.8]]), ([[0.8]], [[0.2]], [[float('inf')]]),
])
def test_malformed_parent_evidence_does_not_create_an_alignment_score(child, small, large):
    with pytest.raises(ValueError):
        combine_aligned_evidence(child, small, large, large_parent_weight=0.5)
