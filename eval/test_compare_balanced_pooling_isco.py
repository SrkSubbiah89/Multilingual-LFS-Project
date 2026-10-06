"""Cache/freeze checks precede label access; synthetic pooling controls only."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from eval import compare_balanced_pooling_isco as comparison
from backend.rag.parent_document_isco import ISCOFragment


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


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
    monkeypatch.setattr(comparison.common, 'common_provenance', lambda *args: {'n': 2})
    args = NS(command='select', output=tmp_path / 'development', parent_selection=tmp_path / 'parent_selection.json',
              selection=None, split=None, cases=tmp_path / 'development.csv', **paths)
    queries = [comparison.common.Query('pool-dev-1', 'toy title', 'en'),
               comparison.common.Query('pool-dev-2', 'toy title arabic', 'ar')]
    vector = np.zeros(384, dtype=np.float32)
    vector[0] = 1
    fragments = [ISCOFragment('title-a', '2512', 'title', 'Software'),
                 ISCOFragment('example-a', '2512', 'example', 'Programmer'),
                 ISCOFragment('title-b', '5120', 'title', 'Cook')]
    inputs = [queries, ['2512', '5120'], ['2512', '5120'], np.stack([vector, vector]),
              np.stack([vector, vector]), np.stack([vector, vector, vector]), fragments]
    monkeypatch.setattr(comparison, 'inputs', lambda *args: tuple(inputs))
    return args, metadata, frozen_parent, inputs, source


def selected_experiment(experiment):
    args, metadata, frozen_parent, inputs, source = experiment
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


def test_fixed_grid_is_bounded_global_and_retains_control_first():
    assert len(comparison.GRID) == 15
    assert comparison.GRID[0] == {'kind': 'max'}
    assert len({comparison.common.object_hash(config) for config in comparison.GRID}) == 15
    for config in comparison.GRID:
        comparison.validate_pooling_config(config)


def test_development_selection_keeps_control_on_exact_ties_and_freezes_evidence(experiment):
    args, _, frozen_parent, *_ = experiment
    comparison.run(args)
    report = json.loads((args.output / 'report.json').read_text())
    assert len(report['parameter_search']) == 15
    assert report['selected'] == {'kind': 'max'}
    assert report['parent_config'] == {'child_weight': 0.5, 'aggregation': 'max'}
    assert report['methods']['parent_document_rag'] == report['methods']['parent_document_balanced_pooling']
    assert report['runtime_mode'].startswith('offline cached official child')
    assert report['rrf_k'] is None and report['per_language_parameters'] is False
    selected = comparison.load_frozen_selection(args.output / 'selected_config.json', frozen_parent)
    assert selected['validation_labels_accessed_for_selection'] is False


def test_evaluation_uses_only_frozen_choice_without_new_parameter_search(experiment):
    selected_experiment(experiment)
    args = experiment[0]
    comparison.run(args)
    report = json.loads((args.output / 'report.json').read_text())
    assert report['split'] == 'heldout' and report['parameter_search'] is None
    assert report['selected'] == {'kind': 'max'}
    assert report['methods']['parent_document_rag']['top1_correct'] == 1
    assert report['methods']['parent_document_balanced_pooling']['n'] == 2


@pytest.mark.parametrize('name', ['queries', 'fragments', 'catalogue'])
def test_altered_cache_bytes_fail_before_labels_are_loaded(experiment, monkeypatch, name):
    args = experiment[0]
    getattr(args, name).write_bytes(b'changed-cache')
    monkeypatch.setattr(comparison, 'inputs', lambda *args: pytest.fail('Labels loaded before cache validation'))
    with pytest.raises(ValueError, match='cache content hash'):
        comparison.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize('field,value', [
    ('minimum_reencoded_parent_cosine', float('nan')), ('minimum_reencoded_parent_cosine', float('inf')),
    ('minimum_reencoded_parent_cosine', 1.01), ('minimum_reencoded_parent_cosine', 0.9),
    ('minimum_reencoded_parent_cosine', True), ('stored_parent_vectors_reencoded', 435),
])
def test_encoder_compatibility_evidence_must_be_complete_and_finite_before_labels(experiment, monkeypatch, field, value):
    args, metadata, *_ = experiment
    metadata['fragments'][field] = value
    write_json(args.fragments.with_suffix('.meta.json'), metadata['fragments'])
    monkeypatch.setattr(comparison, 'inputs', lambda *args: pytest.fail('Labels loaded before encoder compatibility validation'))
    with pytest.raises(ValueError, match='compatibility evidence'):
        comparison.run(args)


@pytest.mark.parametrize('name,field,value', [
    ('queries', 'gold_labels_used', True), ('queries', 'encoder_revision', 'changed'),
    ('queries', 'query_prefix', ''), ('fragments', 'benchmark_text_indexed', True),
    ('fragments', 'weights_sha256', 'changed'), ('fragments', 'source_catalogue_sha256', 'changed'),
    ('catalogue', 'profile', 'legacy'),
])
def test_cache_provenance_must_match_frozen_parent_before_labels(experiment, monkeypatch, name, field, value):
    args, metadata, *_ = experiment
    metadata[name][field] = value
    write_json(getattr(args, name).with_suffix('.meta.json'), metadata[name])
    monkeypatch.setattr(comparison, 'inputs', lambda *args: pytest.fail('Labels loaded before cache provenance validation'))
    with pytest.raises(ValueError, match='frozen parent comparison'):
        comparison.run(args)


@pytest.mark.parametrize('tamper', ['config', 'source', 'report', 'parent_binding', 'cache_binding', 'label_flag'])
def test_frozen_pooling_tampering_fails_before_evaluation_labels(experiment, monkeypatch, tamper):
    choice_path = selected_experiment(experiment)
    args, _, _, _, source = experiment
    choice = json.loads(choice_path.read_text())
    if tamper == 'config':
        choice['config'] = comparison.GRID[1]
        write_json(choice_path, choice)  # retain original digest
    elif tamper == 'source':
        source.write_text('changed implementation')
    elif tamper == 'report':
        report_path = type(choice_path)(choice['development_report'])
        report_path.write_text('changed development evidence')
    else:
        field = {'parent_binding': 'parent_selection_sha256', 'cache_binding': 'fragment_cache_sha256',
                 'label_flag': 'validation_labels_accessed_for_selection'}[tamper]
        choice[field] = True if tamper == 'label_flag' else 'changed'
        resign_choice(choice_path, choice)
    monkeypatch.setattr(comparison, 'inputs', lambda *args: pytest.fail('Evaluation labels loaded before freeze validation'))
    with pytest.raises(ValueError, match='changed|differs'):
        comparison.run(args)
    assert not args.output.exists()


def test_resigned_development_report_must_still_match_selected_configuration(experiment, monkeypatch):
    choice_path = selected_experiment(experiment)
    args = experiment[0]
    choice = json.loads(choice_path.read_text())
    report_path = type(choice_path)(choice['development_report'])
    report = json.loads(report_path.read_text())
    report['selected'] = comparison.GRID[1]
    write_json(report_path, report)
    choice['development_report_sha256'] = comparison.common.sha256(report_path)
    resign_choice(choice_path, choice)
    monkeypatch.setattr(comparison, 'inputs', lambda *args: pytest.fail('Evaluation labels loaded before report/config binding'))
    with pytest.raises(ValueError, match='development evidence'):
        comparison.run(args)


@pytest.mark.parametrize('case_id', ['pool-dev-1', 'parent-dev'])
def test_evaluation_cannot_reuse_either_pooling_or_parent_selection_cases(experiment, case_id):
    selected_experiment(experiment)
    experiment[3][0][0] = comparison.common.Query(case_id, 'overlap', 'en')
    with pytest.raises(ValueError, match='overlaps development'):
        comparison.run(experiment[0])


def test_parent_blend_cannot_be_reselected_inside_pooling_experiment(experiment, monkeypatch):
    args, _, frozen_parent, *_ = experiment
    frozen_parent['config']['child_weight'] = 0.75
    monkeypatch.setattr(comparison, 'inputs', lambda *args: pytest.fail('Labels loaded before parent control validation'))
    with pytest.raises(ValueError, match='0.5/max parent control'):
        comparison.run(args)


def test_existing_evidence_directory_is_never_overwritten(experiment, monkeypatch):
    args = experiment[0]
    args.output.mkdir()
    monkeypatch.setattr(comparison, 'inputs', lambda *args: pytest.fail('Existing evidence accessed'))
    with pytest.raises(ValueError, match='cannot be overwritten'):
        comparison.run(args)
