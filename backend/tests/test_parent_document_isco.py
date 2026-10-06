"""Hermetic evidence and global-candidate regressions for parent-document RAG."""

import csv
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from backend.rag import parent_document_isco as retrieval


def basis(index):
    vector = np.zeros(384, dtype=np.float32)
    vector[index] = 1
    return vector


def toy_retriever(**overrides):
    # The title for 5120 is strong, but its definition is unrelated. The best
    # global leaf belongs to a different major group and must remain eligible.
    records = [SimpleNamespace(code=code, level='unit', title_en=title,
                               embedding_text='Official definition: ' + title)
               for code, title in [('2512', 'Software developers'), ('5120', 'Cooks')]]
    fragments = [retrieval.ISCOFragment('software-title', '2512', 'title', 'Software developers'),
                 retrieval.ISCOFragment('software-example', '2512', 'example', 'Computer programmer'),
                 retrieval.ISCOFragment('cook-title', '5120', 'title', 'Cooks')]
    arguments = dict(records=records, fragments=fragments,
                     fragment_vectors=np.stack([basis(1), basis(0), basis(0)]),
                     definition_vectors=np.stack([basis(0), basis(1)]),
                     unit_codes=['2512', '5120'], embed_query=lambda query: basis(0),
                     child_weight=0.5, aggregation='max')
    arguments.update(overrides)
    return retrieval.ParentDocumentISCORetriever(**arguments)


@pytest.mark.parametrize('mode,expected', [
    ('max', [[0.9, 0.4], [-0.2, -0.3]]),
    ('mean_top2', [[0.8, 0.4], [-0.4, -0.3]]),
])
def test_aggregation_preserves_every_parent_and_uses_actual_top_children(mode, expected):
    scores = np.array([[0.7, 0.4, 0.9, 0.1], [-0.2, -0.3, -0.6, -0.8]])
    actual = retrieval.aggregate_children(scores, ['2512', '5120', '2512', '2512'],
                                         ['2512', '5120'], mode=mode)
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (2, 2)


def test_child_order_does_not_change_parent_aggregation():
    scores = np.array([[0.2, 0.6, 0.7, 0.1]])
    permutation = [3, 0, 2, 1]
    codes = ['2512', '5120', '2512', '5120']
    for mode in ('max', 'mean_top2'):
        first = retrieval.aggregate_children(scores, codes, ['5120', '2512'], mode=mode)
        second = retrieval.aggregate_children(scores[:, permutation],
            [codes[index] for index in permutation], ['5120', '2512'], mode=mode)
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize('scores,children,parents,mode', [
    ([0.1, 0.2], ['2512', '5120'], ['2512', '5120'], 'max'),
    ([[0.1]], ['2512', '5120'], ['2512', '5120'], 'max'),
    ([[float('nan'), 0.2]], ['2512', '5120'], ['2512', '5120'], 'max'),
    ([[0.1, float('inf')]], ['2512', '5120'], ['2512', '5120'], 'max'),
    ([[0.1, 0.2]], ['2512', '9999'], ['2512', '5120'], 'max'),
    ([[0.1, 0.2]], ['2512', '5120'], ['2512', '2512'], 'max'),
    ([[0.1, 0.2]], ['2512', '5120'], ['2512', '5120'], 'greedy'),
])
def test_malformed_or_incomplete_child_evidence_is_rejected(scores, children, parents, mode):
    with pytest.raises(ValueError):
        retrieval.aggregate_children(scores, children, parents, mode=mode)


@pytest.mark.parametrize('weight', [-0.01, 1.01, float('nan'), float('inf'), True, '0.5', None])
def test_invalid_blend_weights_cannot_inflate_similarity(weight):
    with pytest.raises(ValueError, match='child_weight'):
        retrieval.combine_parent_evidence(np.array([[0.2]]), np.array([[0.8]]), child_weight=weight)


@pytest.mark.parametrize('weight,expected', [(0, 0.8), (0.5, 0.5), (1, 0.2)])
def test_blend_is_a_convex_combination_of_uncalibrated_evidence(weight, expected):
    actual = retrieval.combine_parent_evidence(np.array([[0.2]]), np.array([[0.8]]), child_weight=weight)
    assert actual[0, 0] == pytest.approx(expected)


@pytest.mark.parametrize('children,definitions', [
    ([[0.1]], [[0.1, 0.2]]), ([[float('nan')]], [[0.2]]), ([[0.1]], [[float('inf')]]),
])
def test_blend_rejects_misaligned_or_nonfinite_parent_matrices(children, definitions):
    with pytest.raises(ValueError, match='finite and aligned'):
        retrieval.combine_parent_evidence(children, definitions, child_weight=0.5)


def test_search_maps_best_child_to_full_parent_and_keeps_all_major_groups():
    result = toy_retriever().search('computer programmer', top_k=5)
    assert [row['code'] for row in result] == ['2512', '5120']
    first = result[0]
    assert first['score'] == 1
    assert first['definition_score'] == first['child_score'] == 1
    assert first['fragment_id'] == 'software-example'
    assert first['fragment_kind'] == 'example'
    assert first['fragment_text'] == 'Computer programmer'
    assert first['parent_definition'] == 'Official definition: Software developers'
    assert first['hierarchy_path'] == ['2', '25', '251', '2512']
    assert 'confidence' not in first


def test_equal_scores_break_ties_by_code_even_when_parent_order_is_reversed():
    result = toy_retriever(unit_codes=['5120', '2512'],
                           definition_vectors=np.stack([basis(0), basis(0)])).search('title')
    assert [row['code'] for row in result] == ['2512', '5120']


def test_search_passes_only_stripped_user_text_to_injected_encoder():
    seen = []
    def embed(query):
        seen.append(query)
        return basis(0)
    toy_retriever(embed_query=embed).search('  computer programmer  ')
    assert seen == ['computer programmer']


@pytest.mark.parametrize('query,top_k', [('', 5), ('  ', 5), (None, 5), ('title', 0),
                                       ('title', 437), ('title', True), ('title', 1.5)])
def test_search_rejects_blank_inputs_and_invalid_candidate_limits_before_embedding(query, top_k):
    def should_not_embed(query):
        pytest.fail('Invalid user input reached the encoder')
    with pytest.raises(ValueError, match='Nonblank query'):
        toy_retriever(embed_query=should_not_embed).search(query, top_k=top_k)


@pytest.mark.parametrize('vector', [np.zeros(384), np.ones(383), np.ones(384),
                                    np.full(384, float('nan')), np.full(384, float('inf'))])
def test_search_rejects_invalid_query_vectors(vector):
    with pytest.raises(ValueError, match='Query embedding'):
        toy_retriever(embed_query=lambda query: vector).search('title')


@pytest.mark.parametrize('field,value', [
    ('unit_codes', ['2512']),
    ('unit_codes', ['2512', '2512', '5120']),
    ('fragment_vectors', np.ones((3, 383))),
    ('definition_vectors', np.ones((2, 383))),
    ('fragment_vectors', np.ones((3, 384))),
    ('definition_vectors', np.zeros((2, 384))),
    ('fragment_vectors', np.full((3, 384), float('nan'))),
    ('definition_vectors', np.full((2, 384), float('inf'))),
    ('fragments', [retrieval.ISCOFragment('foreign', '9999', 'title', 'Foreign')]),
    ('aggregation', 'strict_hierarchy'),
    ('child_weight', True),
])
def test_retriever_rejects_incompatible_cached_evidence(field, value):
    with pytest.raises(ValueError):
        toy_retriever(**{field: value})


def test_catalogue_fragments_use_only_official_fields_and_stable_source_identity(tmp_path, monkeypatch):
    source = tmp_path / 'official.csv'
    definition = ' '.join(f'word{index}' for index in range(125))
    rows = [dict(level='unit', code='2512', definition=definition,
                 included_occupations='- Computer programmer\n• Software engineer\n- COMPUTER programmer\nUnmarked text'),
            dict(level='unit', code='5120', definition='', included_occupations='')]
    with source.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['level', 'code', 'definition', 'included_occupations'])
        writer.writeheader()
        writer.writerows(rows)
    records = [SimpleNamespace(code=code, level='unit', title_en=title)
               for code, title in [('5120', 'Cooks'), ('2512', 'Software developers')]]
    called = []
    def validated_catalogue(path, *, profile):
        called.append((path, profile))
        return records
    monkeypatch.setattr(retrieval, 'load_enriched_catalogue', validated_catalogue)
    actual_records, fragments = retrieval.catalogue_fragments(source)
    assert actual_records is records
    assert called == [(source, retrieval.ENRICHED_PROFILE)]
    software = [fragment for fragment in fragments if fragment.code == '2512']
    assert [fragment.kind for fragment in software] == ['title', 'definition', 'definition', 'example', 'example']
    assert [len(fragment.text.split()) for fragment in software[1:3]] == [120, 5]
    assert [fragment.text for fragment in software if fragment.kind == 'example'] == ['Computer programmer', 'Software engineer']
    assert fragments[-1].code == '5120' and fragments[-1].kind == 'title'
    for fragment in fragments:
        expected = hashlib.sha256(f'{fragment.code}\x1f{fragment.kind}\x1f{fragment.text}'.encode()).hexdigest()
        assert fragment.fragment_id == expected
    assert len({fragment.fragment_id for fragment in fragments}) == len(fragments)


def test_official_catalogue_validation_failure_is_never_replaced_by_unverified_rows(tmp_path, monkeypatch):
    def reject(path, *, profile):
        raise ValueError('Official source digest mismatch')
    monkeypatch.setattr(retrieval, 'load_enriched_catalogue', reject)
    with pytest.raises(ValueError, match='Official source digest mismatch'):
        retrieval.catalogue_fragments(tmp_path / 'not-opened.csv')
