"""Catalogue-only density correction for occupation evidence scores.

Independent official title vectors supply reference queries. A parent's
cross-occupation popularity is measured from its nearest other official
titles, and subtracted from each query's score. No benchmark text/label is
used to build this correction. Parameters still require development selection.
"""
from __future__ import annotations
import math
import numpy as np
from backend.rag.parent_document_isco import aggregate_children, combine_parent_evidence


def reference_density(*, parent_vectors, unit_codes, fragment_vectors, fragments, neighbors, child_weight):
    codes = list(unit_codes)
    if isinstance(neighbors, bool) or not isinstance(neighbors, int) or not 1 <= neighbors < len(codes):
        raise ValueError('Neighbor count must be an integer smaller than the parent count')
    parent = np.asarray(parent_vectors, dtype=np.float32)
    child = np.asarray(fragment_vectors, dtype=np.float32)
    if parent.shape != (len(codes), 384) or child.shape != (len(fragments), 384):
        raise ValueError('Density correction requires aligned 384-dimensional catalogue vectors')
    if len(set(codes)) != len(codes) or not np.isfinite(parent).all() or not np.isfinite(child).all():
        raise ValueError('Catalogue vectors/codes must be finite and unique')
    if not np.allclose(np.linalg.norm(parent, axis=1), 1, atol=1e-3) or not np.allclose(np.linalg.norm(child, axis=1), 1, atol=1e-3):
        raise ValueError('Catalogue vectors must be normalized')
    titles = {fragment.code: index for index, fragment in enumerate(fragments) if fragment.kind == 'title'}
    if set(titles) != set(codes) or sum(fragment.kind == 'title' for fragment in fragments) != len(codes):
        raise ValueError('Exactly one independently sourced title is required per parent')
    references = child[[titles[code] for code in codes]]
    similarities = references @ child.T
    scores = combine_parent_evidence(
        aggregate_children(similarities, [fragment.code for fragment in fragments], codes, mode='max'),
        references @ parent.T, child_weight=child_weight)
    # A title must not contribute its own unit's near-duplicate self score.
    np.fill_diagonal(scores, -np.inf)
    bias = np.partition(scores, -neighbors, axis=0)[-neighbors:].mean(axis=0)
    if not np.isfinite(bias).all():
        raise ValueError('Nonfinite reference density')
    return bias.astype(np.float32)


def correct_density(scores, bias, strength):
    values, offsets = np.asarray(scores, dtype=np.float32), np.asarray(bias, dtype=np.float32)
    if (isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not math.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError('Density strength must be a finite number in [0,1]')
    if values.ndim != 2 or offsets.shape != (values.shape[1],) or not np.isfinite(values).all() or not np.isfinite(offsets).all():
        raise ValueError('Density scores and offsets must be finite and aligned')
    return values - strength * offsets
