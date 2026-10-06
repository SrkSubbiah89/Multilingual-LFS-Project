"""Catalogue-only fitting, exact control and pre-label provenance/freeze guards."""
import hashlib
import io
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from eval import compare_catalogue_aligned_isco as comparison


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def parent_vectors(dimension):
    matrix = np.zeros((436, dimension), dtype=np.float32)
    matrix[np.arange(436), np.arange(436) % 384] = 1
    return matrix


def store_large(path, small_metadata, codes):
    vectors = parent_vectors(1024)
    np.savez_compressed(path, vectors=vectors, unit_codes=np.asarray(codes))
    metadata = {
        'cache_sha256': comparison.common.sha256(path), 'dimension': 1024, 'records': 436,
        'declared_profile': comparison.ENRICHED_E5LARGE_PROFILE,
        'source_catalogue_sha256': small_metadata['source_catalogue_sha256'],
        'paired_small_cache_sha256': small_metadata['cache_sha256'],
        'vector_bytes_sha256': hashlib.sha256(vectors.tobytes()).hexdigest(),
        'encoder_execution_identity': 'Historical stored vectors; encoder revision/weights unverified',
        'encoder_revision': None, 'encoder_weights_sha256': None, 'benchmark_queries_or_labels_used': False,
    }
    write_json(path.with_suffix('.meta.json'), metadata)
    return vectors, metadata


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    paths = {name: tmp_path / f'{name}.npz' for name in ('queries', 'fragments', 'catalogue')}
    for name, path in paths.items():
        path.write_bytes(f'known-{name}-cache'.encode())
    digest = {name: comparison.common.sha256(path) for name, path in paths.items()}
    encoder = {'encoder_id': comparison.common.MODEL, 'encoder_revision': 'revision', 'weights_sha256': 'weights'}
    metadata = {
        'queries': {**encoder, 'cache_sha256': digest['queries'], 'gold_labels_used': False,
                    'query_prefix': 'query: ', 'normalize_embeddings': True, 'dimension': 384},
        'catalogue': {'cache_sha256': digest['catalogue'], 'source_catalogue_sha256': 'official-source',
                      'profile': comparison.parent.PROFILE},
        'fragments': {**encoder, 'cache_sha256': digest['fragments'], 'catalogue_cache_sha256': digest['catalogue'],
                      'source_catalogue_sha256': 'official-source', 'benchmark_text_indexed': False,
                      'normalize_embeddings': True, 'dimension': 384, 'passage_prefix': 'passage: ',
                      'minimum_reencoded_parent_cosine': 0.9999999, 'stored_parent_vectors_reencoded': 436},
    }
    for name, path in paths.items():
        write_json(path.with_suffix('.meta.json'), metadata[name])
    frozen_parent = {'selection_sha256': 'frozen-parent', 'config': {'child_weight': 0.5, 'aggregation': 'max'},
                     'fragment_cache_sha256': digest['fragments'], 'catalogue_cache_sha256': digest['catalogue'],
                     'query_encoder': encoder, 'development_case_ids': ['parent-dev']}
    monkeypatch.setattr(comparison.parent, 'load_frozen_selection', lambda path: frozen_parent)
    source = tmp_path / 'algorithm.py'
    source.write_text('unchanged implementation', encoding='utf-8')
    monkeypatch.setattr(comparison, 'SOURCE_FILES', [source])
    monkeypatch.setattr(comparison.common, 'common_provenance', lambda *args: {
        'n': 2, 'query_encoder': metadata['queries'], 'catalogue': metadata['catalogue']})
    codes, small = [str(1000 + index) for index in range(436)], parent_vectors(384)
    large_path = tmp_path / 'large.npz'
    large, metadata['large'] = store_large(large_path, metadata['catalogue'], codes)
    monkeypatch.setattr(comparison.common, 'load_catalogue', lambda *args: ({'unit': (codes, small)}, metadata['catalogue'], []))
    vectors = np.eye(384, dtype=np.float32)[:2]
    queries = [comparison.common.Query('alignment-dev-1', 'toy title', 'en'),
               comparison.common.Query('alignment-dev-2', 'toy arabic title', 'ar')]
    scores = vectors @ small.T
    inputs = [queries, ['1000', '1001'], codes, scores, {'max': scores.copy()}]
    monkeypatch.setattr(comparison.parent, 'inputs', lambda *args: tuple(inputs))
    monkeypatch.setattr(comparison.common, 'load_query_vectors', lambda *args: (vectors, metadata['queries']))
    # Algebra is covered separately; evaluator tests isolate sequencing, identities and row pairing.
    monkeypatch.setattr(comparison, 'fit_catalogue_alignment', lambda *args, **kwargs: np.eye(384, 1024))
    args = NS(command='select', output=tmp_path / 'development', parent_selection=tmp_path / 'parent_selection.json',
              selection=None, split=None, cases=tmp_path / 'development.csv', large=large_path, **paths)
    return args, metadata, frozen_parent, inputs, source, small, large


def selected_experiment(experiment):
    args, _, _, inputs, *_ = experiment
    comparison.run(args)
    choice = args.output / 'selected_config.json'
    args.command, args.selection, args.split = 'evaluate', choice, 'heldout'
    args.output = args.output.parent / 'evaluation'
    inputs[0] = [comparison.common.Query('new-1', 'new toy', 'en'),
                 comparison.common.Query('new-2', 'new toy arabic', 'ar')]
    return choice


def resign_choice(path, choice):
    choice.pop('selection_sha256', None)
    choice['selection_sha256'] = comparison.common.object_hash(choice)
    write_json(path, choice)


def forbid_labels(monkeypatch):
    monkeypatch.setattr(comparison.parent, 'inputs', lambda *args: pytest.fail('Labels accessed before provenance/freeze validation'))


def test_predeclared_seven_global_configs_keep_control_first():
    assert len(comparison.GRID) == 7
    assert comparison.GRID[0] == {'ridge': None, 'large_parent_weight': 0.0}
    assert len({comparison.common.object_hash(config) for config in comparison.GRID}) == 7
    for config in comparison.GRID:
        comparison.validate_alignment_config(config)


def test_selection_retains_control_on_exact_ties_and_records_actual_encoder(experiment):
    args, _, frozen_parent, *_ = experiment
    comparison.run(args)
    report = json.loads((args.output / 'report.json').read_text())
    assert len(report['parameter_search']) == 7 and report['selected'] == comparison.GRID[0]
    assert report['methods']['parent_document_rag'] == report['methods']['parent_document_catalogue_aligned']
    assert report['methods']['parent_document_rag']['top1_correct'] == 2
    assert report['large_encoder_inference'] is False
    assert report['benchmark_queries_or_labels_used_for_fit'] is False
    assert report['per_language_parameters'] is False
    assert report['large_snapshot_metadata']['encoder_revision'] is None
    assert report['large_snapshot_metadata']['encoder_weights_sha256'] is None
    assert report['rrf_k'] is None and report['rrf_rank_indexing'] is None
    choice = comparison.load_frozen_selection(args.output / 'selected_config.json', frozen_parent)
    assert choice['validation_labels_accessed_for_selection'] is False


def test_fitting_receives_only_catalogue_pairs_before_inputs(experiment, monkeypatch):
    args, _, _, inputs, _, small, large = experiment
    events = []

    def fit(actual_small, actual_large, *, ridge):
        assert actual_small is small
        np.testing.assert_array_equal(actual_large, large)
        events.append(('fit', ridge, len(actual_small)))
        return np.eye(384, 1024)

    def load_inputs(*args):
        assert events == [('fit', 0.01, 436), ('fit', 0.1, 436), ('fit', 1.0, 436)]
        events.append(('labels',))
        return tuple(inputs)

    monkeypatch.setattr(comparison, 'fit_catalogue_alignment', fit)
    monkeypatch.setattr(comparison.parent, 'inputs', load_inputs)
    comparison.run(args)
    assert events[-1] == ('labels',)


def test_evaluation_uses_frozen_choice_without_parameter_search_or_refitting_control(experiment, monkeypatch):
    selected_experiment(experiment)
    monkeypatch.setattr(comparison, 'fit_catalogue_alignment', lambda *args, **kwargs: pytest.fail('Control must not fit an unused matrix'))
    comparison.run(experiment[0])
    report = json.loads((experiment[0].output / 'report.json').read_text())
    assert report['split'] == 'heldout' and report['parameter_search'] is None
    assert report['selected'] == comparison.GRID[0]


@pytest.mark.parametrize('name', ['queries', 'fragments', 'catalogue', 'large'])
def test_changed_cache_bytes_fail_before_label_loading(experiment, monkeypatch, name):
    args = experiment[0]
    getattr(args, name).write_bytes(b'changed-cache')
    forbid_labels(monkeypatch)
    with pytest.raises(ValueError, match='hash changed|provenance differs'):
        comparison.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize('field,value', [
    ('source_catalogue_sha256', 'wrong-source'), ('paired_small_cache_sha256', 'wrong-cache'),
    ('benchmark_queries_or_labels_used', True), ('encoder_revision', 'unsupported-revision'),
    ('encoder_weights_sha256', 'unsupported-weights'), ('encoder_execution_identity', 'verified E5-large'),
    ('vector_bytes_sha256', 'different-vectors'), ('dimension', 384), ('records', 435),
])
def test_large_snapshot_provenance_rejects_invalid_or_unsupported_identity_before_labels(experiment, monkeypatch, field, value):
    args, metadata, *_ = experiment
    metadata['large'][field] = value
    write_json(args.large.with_suffix('.meta.json'), metadata['large'])
    forbid_labels(monkeypatch)
    with pytest.raises(ValueError, match='differs'):
        comparison.run(args)


def test_unpaired_official_rows_fail_before_any_catalogue_fit_or_labels(experiment, monkeypatch):
    args, metadata, _, _, _, _, large = experiment
    codes = [str(1000 + index) for index in range(436)][::-1]
    np.savez_compressed(args.large, vectors=large, unit_codes=np.asarray(codes))
    metadata['large']['cache_sha256'] = comparison.common.sha256(args.large)
    write_json(args.large.with_suffix('.meta.json'), metadata['large'])
    forbid_labels(monkeypatch)
    monkeypatch.setattr(comparison, 'fit_catalogue_alignment', lambda *args, **kwargs: pytest.fail('Unpaired rows fitted'))
    with pytest.raises(ValueError, match='not paired'):
        comparison.run(args)


@pytest.mark.parametrize('tamper', ['digest', 'source', 'report', 'parent_binding', 'cache_binding', 'large_metadata_binding', 'label_flag', 'query_encoder'])
def test_frozen_selection_tampering_fails_before_evaluation_labels(experiment, monkeypatch, tamper):
    choice_path = selected_experiment(experiment)
    args, _, _, _, source, *_ = experiment
    choice = json.loads(choice_path.read_text())
    if tamper == 'digest':
        choice['config'] = comparison.GRID[1]
        write_json(choice_path, choice)
    elif tamper == 'source':
        source.write_text('changed implementation')
    elif tamper == 'report':
        type(choice_path)(choice['development_report']).write_text('changed evidence')
    else:
        field = {'parent_binding': 'parent_selection_sha256', 'cache_binding': 'fragment_cache_sha256',
                 'large_metadata_binding': 'large_snapshot_metadata_sha256', 'label_flag': 'validation_labels_accessed_for_selection',
                 'query_encoder': 'query_encoder'}[tamper]
        choice[field] = True if tamper == 'label_flag' else {'encoder_id': 'changed'} if tamper == 'query_encoder' else 'changed'
        resign_choice(choice_path, choice)
    forbid_labels(monkeypatch)
    with pytest.raises(ValueError, match='changed|differs'):
        comparison.run(args)


@pytest.mark.parametrize('config', [
    {'ridge': None, 'large_parent_weight': False}, {'ridge': 0.1, 'large_parent_weight': True},
    {'ridge': True, 'large_parent_weight': 1.0}, {'ridge': 0.1, 'large_parent_weight': 0.75},
    {'ridge': 0.1, 'large_parent_weight': 1.0, 'language': 'en'},
])
def test_resigned_boolean_or_unlisted_configuration_fails_before_inputs(experiment, monkeypatch, config):
    choice_path = selected_experiment(experiment)
    choice = json.loads(choice_path.read_text())
    choice['config'] = config
    resign_choice(choice_path, choice)
    forbid_labels(monkeypatch)
    with pytest.raises(ValueError, match='configuration'):
        comparison.run(experiment[0])


@pytest.mark.parametrize('field,value', [('selected', comparison.GRID[1]), ('large_encoder_inference', True),
                                        ('benchmark_queries_or_labels_used_for_fit', True)])
def test_resigned_report_must_still_match_frozen_selection_and_provenance(experiment, monkeypatch, field, value):
    choice_path = selected_experiment(experiment)
    choice = json.loads(choice_path.read_text())
    report_path = type(choice_path)(choice['development_report'])
    report = json.loads(report_path.read_text())
    report[field] = value
    write_json(report_path, report)
    choice['development_report_sha256'] = comparison.common.sha256(report_path)
    resign_choice(choice_path, choice)
    forbid_labels(monkeypatch)
    with pytest.raises(ValueError, match='development evidence'):
        comparison.run(experiment[0])


@pytest.mark.parametrize('case_id', ['alignment-dev-1', 'parent-dev'])
def test_evaluation_cannot_reuse_alignment_or_parent_selection_cases(experiment, case_id):
    selected_experiment(experiment)
    experiment[3][0][0] = comparison.common.Query(case_id, 'overlap', 'en')
    with pytest.raises(ValueError, match='overlaps development'):
        comparison.run(experiment[0])


def test_original_parent_blend_cannot_be_reselected_before_inputs(experiment, monkeypatch):
    experiment[2]['config']['child_weight'] = 0.75
    forbid_labels(monkeypatch)
    with pytest.raises(ValueError, match='0.5/max parent control'):
        comparison.run(experiment[0])


def test_existing_output_is_never_overwritten(experiment, monkeypatch):
    experiment[0].output.mkdir()
    forbid_labels(monkeypatch)
    with pytest.raises(ValueError, match='cannot be overwritten'):
        comparison.run(experiment[0])


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    codes = [str(1000 + index) for index in range(436)]
    records = [NS(code=code, level='unit', parent_code=code[:3], title_en=f'Official {code}',
                  source_catalogue_sha256='source', embedding_text=f'Official independent description {code}') for code in codes]
    vectors = parent_vectors(1024)
    points = [{'id': index, 'payload': {**vars(record), 'profile': comparison.ENRICHED_E5LARGE_PROFILE},
               'vector': vectors[index].tolist()} for index, record in enumerate(records)]
    catalogue = {'unit': (codes, parent_vectors(384))}
    monkeypatch.setattr(comparison.common, 'load_catalogue', lambda *args: (catalogue, {'cache_sha256': 'small', 'source_catalogue_sha256': 'source'}, records))
    info = {'config': {'params': {'vectors': {'size': 1024, 'distance': 'Cosine'}}}, 'points_count': 436}
    requests = []

    def urlopen(request, **kwargs):
        if isinstance(request, str):
            result = info
        else:
            body = json.loads(request.data)
            requests.append(body)
            assert request.get_method() == 'POST' and request.full_url.endswith('/points/scroll')
            offset = 0 if 'offset' not in body else 256
            # Unsorted pages ensure ordering is based on exact official codes.
            result = {'points': list(reversed(points[offset:offset + 256])), 'next_page_offset': 256 if offset == 0 else None}
        return io.BytesIO(json.dumps({'result': result}).encode())

    monkeypatch.setattr(comparison, 'urlopen', urlopen)
    return tmp_path / 'large.npz', tmp_path / 'small.npz', points, info, requests


def test_snapshot_reads_all_pages_and_exactly_pairs_official_text_without_encoder_claim(snapshot):
    output, small, _, _, requests = snapshot
    comparison.snapshot_large(output, small)
    vectors, codes, metadata = comparison.load_large(output, {'cache_sha256': 'small', 'source_catalogue_sha256': 'source'})
    assert vectors.shape == (436, 1024) and codes == [str(1000 + index) for index in range(436)]
    assert len(requests) == 2 and requests[1]['offset'] == 256
    assert metadata['benchmark_queries_or_labels_used'] is False
    assert metadata['encoder_revision'] is None and metadata['encoder_weights_sha256'] is None


@pytest.mark.parametrize('field', ['code', 'level', 'parent_code', 'title_en', 'source_catalogue_sha256', 'embedding_text', 'profile'])
def test_snapshot_requires_exact_official_code_text_source_and_profile(snapshot, field):
    output, small, points, *_ = snapshot
    points[0]['payload'][field] = 'altered'
    with pytest.raises(ValueError, match='differ'):
        comparison.snapshot_large(output, small)
    assert not output.exists()


@pytest.mark.parametrize('vector', [[float('nan')] + [0] * 1023, [0] * 1024, [1] * 1023])
def test_snapshot_rejects_nonfinite_unnormalized_or_wrong_dimension_vectors(snapshot, vector):
    output, small, points, *_ = snapshot
    points[0]['vector'] = vector
    with pytest.raises(ValueError, match='vector|normalized|inhomogeneous'):
        comparison.snapshot_large(output, small)
    assert not output.exists()


def test_snapshot_never_overwrites_existing_cache(snapshot):
    output, small, *_ = snapshot
    output.write_bytes(b'preserve')
    with pytest.raises(ValueError, match='cannot be overwritten'):
        comparison.snapshot_large(output, small)
    assert output.read_bytes() == b'preserve'
