"""Ridge geometry alignment fitted only to paired official catalogue vectors.

Queries still use the verified small encoder. Projected queries compare with
actual stored 1024-D vectors; this is not inference through a large encoder.
"""
from __future__ import annotations
import math
import numpy as np


def normalized_vectors(values, dimension):
    vectors = np.asarray(values, dtype=np.float64)
    if (vectors.ndim != 2 or vectors.shape[1] != dimension or not len(vectors)
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=0.002, rtol=0)):
        raise ValueError('Catalogue/query vectors must be finite, normalized and dimensionally aligned')
    return vectors


def fit_catalogue_alignment(small_vectors, large_vectors, *, ridge):
    """Fit only explicit catalogue matrices; no query/text/label arguments."""
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float)) or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError('Ridge regularization must be finite and positive')
    small, large = normalized_vectors(small_vectors, 384), normalized_vectors(large_vectors, 1024)
    if len(small) != len(large):
        raise ValueError('Paired catalogue vector rows differ')
    matrix = np.linalg.solve(small.T @ small + ridge * np.eye(384), small.T @ large)
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite catalogue alignment')
    return matrix


def projected_parent_scores(query_vectors, matrix, large_vectors):
    queries, parents = normalized_vectors(query_vectors, 384), normalized_vectors(large_vectors, 1024)
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (384, 1024) or not np.isfinite(transform).all():
        raise ValueError('Invalid catalogue alignment matrix')
    projected = queries @ transform
    norms = np.linalg.norm(projected, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError('Projected query has no finite direction')
    return ((projected / norms[:, None]) @ parents.T).astype(np.float32)


def combine_aligned_evidence(child_scores, small_parent_scores, aligned_parent_scores, *, large_parent_weight):
    weight = large_parent_weight
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Large-parent blend weight must be finite in [0,1]')
    child, small, large = [np.asarray(value, dtype=np.float32) for value in
                           (child_scores, small_parent_scores, aligned_parent_scores)]
    if child.ndim != 2 or child.shape != small.shape or child.shape != large.shape or any(
            not np.isfinite(value).all() for value in (child, small, large)):
        raise ValueError('Aligned evidence must be finite with matching parent/query shapes')
    # Separate the exact frozen control so weight zero keeps its float32
    # operations and rankings, rather than introducing arithmetic drift.
    if weight == 0:
        return 0.5 * child + 0.5 * small
    return 0.5 * child + 0.5 * ((1 - weight) * small + weight * large)
