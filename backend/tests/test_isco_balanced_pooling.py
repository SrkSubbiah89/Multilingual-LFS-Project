"""Global pooling algebra and malformed-evidence regressions; no live model."""
import math

import numpy as np
import pytest

from backend.rag.isco_balanced_pooling import aggregate_balanced_children as aggregate, validate_pooling_config


def test_maximum_control_preserves_global_code_membership_and_raw_cosines():
    scores = np.array([[0.2, 0.9, 0.4], [-0.6, -0.3, -0.8]])
    actual = aggregate(scores, ['2512', '2512', '5120'], ['title', 'example', 'title'],
                       ['5120', '2512'], config={'kind': 'max'})
    np.testing.assert_allclose(actual, [[0.4, 0.9], [-0.8, -0.3]])


def test_kind_balancing_votes_once_per_kind_and_renormalizes_missing_kinds():
    scores = [[0.2, 0.4, 0.6, 0.8, 0.9]]
    actual = aggregate(scores, ['2512'] * 4 + ['5120'],
                       ['title', 'definition', 'example', 'example', 'title'], ['2512', '5120'],
                       config={'kind': 'kind_balanced', 'weights': [0.5, 0.25, 0.25]})
    assert actual[0, 0] == pytest.approx(0.5 * 0.2 + 0.25 * 0.4 + 0.25 * 0.8)
    assert actual[0, 1] == pytest.approx(0.9)  # title has full weight when other kinds are absent


def test_many_duplicate_examples_cannot_create_extra_kind_votes():
    config = {'kind': 'kind_balanced', 'weights': [0.5, 0.25, 0.25]}
    first = aggregate([[0.2, 0.8]], ['2512', '2512'], ['title', 'example'], ['2512'], config=config)
    second = aggregate([[0.2] + [0.8] * 30], ['2512'] * 31, ['title'] + ['example'] * 30,
                        ['2512'], config=config)
    np.testing.assert_array_equal(first, second)


def test_logmeanexp_uses_child_count_normalization():
    actual = aggregate([[0.2, 0.8]], ['2512', '2512'], ['title', 'example'], ['2512'],
                       config={'kind': 'logmeanexp', 'temperature': 0.1})
    expected = 0.8 + 0.1 * math.log((math.exp(-6) + 1) / 2)
    assert actual[0, 0] == pytest.approx(expected)
    assert 0.2 < actual[0, 0] < 0.8


def test_repeating_all_children_does_not_change_count_normalized_score():
    config = {'kind': 'logmeanexp', 'temperature': 0.05}
    first = aggregate([[0.2, 0.8]], ['2512'] * 2, ['title', 'example'], ['2512'], config=config)
    second = aggregate([[0.2, 0.8] * 10], ['2512'] * 20, ['title', 'example'] * 10,
                        ['2512'], config=config)
    np.testing.assert_allclose(first, second, atol=1e-7)


@pytest.mark.parametrize('temperature', [1e-12, 0.02, 1, 1e6])
def test_soft_pooling_is_numerically_stable_and_bounded_by_children(temperature):
    actual = aggregate([[-1000, 1000], [-0.7, -0.2]], ['2512'] * 2, ['title', 'example'], ['2512'],
                       config={'kind': 'logmeanexp', 'temperature': temperature})
    assert np.isfinite(actual).all()
    assert -1000 <= actual[0, 0] <= 1000
    assert -0.700001 <= actual[1, 0] <= -0.199999


@pytest.mark.parametrize('config', [
    {'kind': 'max'}, {'kind': 'kind_balanced', 'weights': [0.5, 0.25, 0.25]},
    {'kind': 'logmeanexp', 'temperature': 0.02},
])
def test_child_permutation_never_changes_parent_results(config):
    scores = np.array([[0.2, 0.6, 0.8, 0.4]])
    codes, kinds, order = ['2512', '5120', '2512', '5120'], ['title', 'title', 'example', 'definition'], [3, 0, 2, 1]
    first = aggregate(scores, codes, kinds, ['2512', '5120'], config=config)
    second = aggregate(scores[:, order], [codes[i] for i in order], [kinds[i] for i in order],
                        ['2512', '5120'], config=config)
    np.testing.assert_allclose(first, second)


@pytest.mark.parametrize('config', [
    {}, {'kind': 'beam'}, {'kind': 'max', 'temperature': 0.1},
    {'kind': 'kind_balanced', 'weights': [0.5, 0.5]},
    {'kind': 'kind_balanced', 'weights': [0.5, 0.5, 0.5]},
    {'kind': 'kind_balanced', 'weights': [-0.1, 0.6, 0.5]},
    {'kind': 'kind_balanced', 'weights': [True, 0, 0]},
    {'kind': 'kind_balanced', 'weights': [float('nan'), 0, 1]},
    {'kind': 'logmeanexp', 'temperature': 0}, {'kind': 'logmeanexp', 'temperature': -0.1},
    {'kind': 'logmeanexp', 'temperature': True}, {'kind': 'logmeanexp', 'temperature': float('inf')},
    {'kind': 'logmeanexp', 'temperature': '0.1'},
])
def test_invalid_weights_or_temperatures_cannot_manufacture_evidence(config):
    with pytest.raises(ValueError):
        validate_pooling_config(config)


@pytest.mark.parametrize('scores,codes,kinds,parents', [
    ([0.1], ['2512'], ['title'], ['2512']),
    ([[0.1]], ['2512', '5120'], ['title', 'title'], ['2512', '5120']),
    ([[float('nan')]], ['2512'], ['title'], ['2512']),
    ([[float('inf')]], ['2512'], ['title'], ['2512']),
    ([[0.1]], ['2512'], ['title'], ['2512', '2512']),
    ([[0.1]], ['2512'], ['title'], ['5120']),
    ([[0.1]], ['2512'], ['title'], []),
    ([[0.1]], ['2512'], ['unverified'], ['2512']),
    ([[0.1]], ['2512'], [], ['2512']),
])
def test_incomplete_or_malformed_global_pool_is_rejected(scores, codes, kinds, parents):
    with pytest.raises(ValueError):
        aggregate(scores, codes, kinds, parents, config={'kind': 'max'})


def test_zero_weight_on_all_available_kinds_does_not_invent_a_fallback_vote():
    with pytest.raises(ValueError, match='positive-weight'):
        aggregate([[0.8]], ['2512'], ['example'], ['2512'],
                  config={'kind': 'kind_balanced', 'weights': [1, 0, 0]})
