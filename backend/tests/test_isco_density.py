"""Reference-only correction must exclude self matches and reject bad evidence."""
from types import SimpleNamespace

import numpy as np
import pytest

from backend.rag.isco_density import correct_density, reference_density


def vector(x, y):
    result = np.zeros(384, dtype=np.float32)
    result[:2] = [x, y]
    return result / np.linalg.norm(result)


def test_reference_density_excludes_same_parent_and_preserves_unit_order():
    # Orthogonal title pairs would each score 1 on themselves; excluding the
    # same-parent reference leaves their cross-occupation score of zero.
    vectors = np.asarray([vector(1, 0), vector(0, 1)])
    fragments = [SimpleNamespace(code='1111', kind='title'),
                 SimpleNamespace(code='2222', kind='title')]
    bias = reference_density(parent_vectors=vectors, unit_codes=['1111', '2222'],
        fragment_vectors=vectors, fragments=fragments, neighbors=1, child_weight=0.5)
    np.testing.assert_allclose(bias, [0, 0], atol=1e-7)


def test_correction_can_change_rank_but_zero_strength_is_exact_control():
    scores = np.asarray([[0.8, 0.75]], dtype=np.float32)
    bias = np.asarray([0.7, 0.5], dtype=np.float32)
    np.testing.assert_array_equal(correct_density(scores, bias, 0), scores)
    assert np.argmax(correct_density(scores, bias, 1), axis=1).tolist() == [1]


@pytest.mark.parametrize('strength', [True, -0.1, 1.1, float('nan'), float('inf')])
def test_invalid_correction_strength_is_rejected(strength):
    with pytest.raises(ValueError):
        correct_density([[0.8, 0.75]], [0.7, 0.5], strength)


def test_duplicate_official_titles_are_rejected():
    vectors = np.asarray([vector(1, 0), vector(0, 1)])
    fragments = [SimpleNamespace(code='1111', kind='title'),
                 SimpleNamespace(code='1111', kind='title')]
    with pytest.raises(ValueError, match='Exactly one'):
        reference_density(parent_vectors=vectors, unit_codes=['1111', '2222'],
            fragment_vectors=vectors, fragments=fragments, neighbors=1, child_weight=0.5)


@pytest.mark.parametrize('scores,bias', [([[1, float('nan')]], [0, 0]),
                                      ([[1, 0]], [0]), ([1, 0], [0, 0])])
def test_nonfinite_or_misaligned_correction_evidence_is_rejected(scores, bias):
    with pytest.raises(ValueError):
        correct_density(scores, bias, 0.5)
