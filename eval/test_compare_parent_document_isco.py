"""Hermetic provenance, dev-selection and frozen-evaluation regressions."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from eval import compare_parent_document_isco as comparison
from backend.rag.parent_document_isco import ISCOFragment


def basis(index):
    vector = np.zeros(384, dtype=np.float32)
    vector[index] = 1
    return vector


@pytest.fixture
def frozen_choice(tmp_path, monkeypatch):
    algorithm = tmp_path / 'retriever.py'
    helper = tmp_path / 'helper.py'
    algorithm.write_text('fixed ranking implementation', encoding='utf-8')
    helper.write_text('fixed loading and metrics implementation', encoding='utf-8')
    monkeypatch.setattr(comparison, 'SOURCE_FILES', [algorithm, helper])
    source_hashes = {str(path): comparison.baseline.sha256(path) for path in comparison.SOURCE_FILES}
    config = {'child_weight': 0.5, 'aggregation': 'max'}
    report = tmp_path / 'development_report.json'
    report.write_text(json.dumps({'split': 'development', 'selected': config,
                                  'source_hashes': source_hashes}), encoding='utf-8')
    selection = {'config': config, 'development_case_ids': ['dev-1'], 'source_hashes': source_hashes,
                 'development_report': str(report), 'development_report_sha256': comparison.baseline.sha256(report),
                 'fragment_cache_sha256': 'fragment-cache', 'catalogue_cache_sha256': 'catalogue-cache',
                 'query_encoder': {'encoder_id': 'e5-small', 'encoder_revision': 'fixed-revision', 'weights_sha256': 'fixed-weights'},
                 'validation_labels_accessed_for_selection': False}
    selection['selection_sha256'] = comparison.baseline.object_hash(selection)
    path = tmp_path / 'selected.json'
    path.write_text(json.dumps(selection), encoding='utf-8')
    return path, selection, report, algorithm, helper


def resign_selection(path, selection):
    selection.pop('selection_sha256', None)
    selection['selection_sha256'] = comparison.baseline.object_hash(selection)
    path.write_text(json.dumps(selection), encoding='utf-8')


def test_frozen_choice_verifies_current_source_and_development_report(frozen_choice):
    path, selection, *_ = frozen_choice
    assert comparison.load_frozen_selection(path) == selection


@pytest.mark.parametrize('change', ['parameters', 'algorithm', 'helper', 'report'])
def test_frozen_choice_rejects_post_selection_tampering(frozen_choice, change):
    path, selection, report, algorithm, helper = frozen_choice
    if change == 'parameters':
        selection['config']['child_weight'] = 0.75
        path.write_text(json.dumps(selection), encoding='utf-8')
    elif change == 'report':
        report.write_text('changed development evidence', encoding='utf-8')
    else:
        {'algorithm': algorithm, 'helper': helper}[change].write_text('changed code', encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        comparison.load_frozen_selection(path)


@pytest.mark.parametrize('flag', [True, None, 0, 'false'])
def test_even_resigned_selection_must_assert_development_only_labels(frozen_choice, flag):
    path, selection, *_ = frozen_choice
    selection['validation_labels_accessed_for_selection'] = flag
    resign_selection(path, selection)
    with pytest.raises(ValueError, match='development-only'):
        comparison.load_frozen_selection(path)


@pytest.mark.parametrize('field,value', [
    ('split', 'validation'), ('selected', {'child_weight': 0.75, 'aggregation': 'max'}), ('source_hashes', {}),
])
def test_resigned_report_must_still_match_development_choice(frozen_choice, field, value):
    path, selection, report, *_ = frozen_choice
    evidence = json.loads(report.read_text(encoding='utf-8'))
    evidence[field] = value
    report.write_text(json.dumps(evidence), encoding='utf-8')
    selection['development_report_sha256'] = comparison.baseline.sha256(report)
    resign_selection(path, selection)
    with pytest.raises(ValueError, match='development evidence'):
        comparison.load_frozen_selection(path)


def arguments(tmp_path, *, command='evaluate', selection=None):
    return SimpleNamespace(command=command, selection=selection, split='heldout',
                           cases=tmp_path / 'cases.csv', queries=tmp_path / 'queries.npz',
                           fragments=tmp_path / 'fragments.npz', catalogue=tmp_path / 'catalogue.npz',
                           output=tmp_path / 'experiment')


def mocked_comparison_inputs(monkeypatch, *, case_id='heldout-1'):
    queries = [comparison.baseline.Query(case_id, 'computer programmer', 'en'),
               comparison.baseline.Query('heldout-2', 'cook', 'ar')]
    codes = ['2512', '5120']
    # One correct result is recovered by child passages while all parent codes
    # remain available. The second case is intentionally a remaining error.
    parent_scores = np.array([[0.1, 0.4], [0.6, 0.4]], dtype=np.float32)
    children = np.array([[0.7, 0.3], [0.7, 0.3]], dtype=np.float32)
    query_meta = {'encoder_id': 'e5-small', 'encoder_revision': 'fixed-revision', 'weights_sha256': 'fixed-weights'}
    catalogue_meta = {'cache_sha256': 'catalogue-cache'}
    fragment_meta = {'cache_sha256': 'fragment-cache'}
    inputs = (queries, ['2512', '5120'], codes, parent_scores,
              {'max': children, 'mean_top2': children.copy()}, query_meta, catalogue_meta, fragment_meta)
    monkeypatch.setattr(comparison, 'inputs', lambda *args: inputs)
    monkeypatch.setattr(comparison.baseline, 'common_provenance', lambda *args: {'n': len(queries)})
    return inputs


def test_bad_freeze_is_rejected_before_evaluation_cases_are_read(tmp_path, frozen_choice, monkeypatch):
    path, selection, *_ = frozen_choice
    selection['config']['child_weight'] = 0.75
    path.write_text(json.dumps(selection), encoding='utf-8')
    monkeypatch.setattr(comparison, 'inputs', lambda *args: pytest.fail('Evaluation labels were accessed before freeze verification'))
    args = arguments(tmp_path, selection=path)
    with pytest.raises(ValueError, match='Frozen parameters'):
        comparison.run(args)
    assert not args.output.exists()


def test_evaluation_runs_only_frozen_config_and_keeps_remaining_errors(tmp_path, frozen_choice, monkeypatch):
    path, selection, *_ = frozen_choice
    mocked_comparison_inputs(monkeypatch)
    args = arguments(tmp_path, selection=path)
    comparison.run(args)
    report = json.loads((args.output / 'report.json').read_text(encoding='utf-8'))
    assert report['selected'] == selection['config']
    assert report['parameter_search'] is None
    assert report['selection_sha256'] == selection['selection_sha256']
    assert report['methods']['dense_flat']['top1_correct'] == 0
    assert report['methods']['parent_document_rag']['top1_correct'] == 1
    assert report['methods']['parent_document_rag']['n'] == 2
    assert report['per_language']['en']['parent_document_rag']['top1_correct'] == 1
    assert report['per_language']['ar']['parent_document_rag']['top1_correct'] == 0


@pytest.mark.parametrize('mismatch', ['overlap', 'fragments', 'catalogue', 'encoder'])
def test_evaluation_rejects_overlap_or_changed_model_and_catalogue(tmp_path, frozen_choice, monkeypatch, mismatch):
    path, _, *_ = frozen_choice
    data = mocked_comparison_inputs(monkeypatch, case_id='dev-1' if mismatch == 'overlap' else 'heldout-1')
    if mismatch == 'fragments':
        data[7]['cache_sha256'] = 'changed-fragments'
    elif mismatch == 'catalogue':
        data[6]['cache_sha256'] = 'changed-catalogue'
    elif mismatch == 'encoder':
        data[5]['encoder_revision'] = 'changed-revision'
    args = arguments(tmp_path, selection=path)
    with pytest.raises(ValueError, match='overlaps|differs'):
        comparison.run(args)
    assert not args.output.exists()


def test_development_search_records_all_eight_choices_and_freezes_tie_break(tmp_path, frozen_choice, monkeypatch):
    mocked_comparison_inputs(monkeypatch)
    args = arguments(tmp_path, command='select')
    comparison.run(args)
    report = json.loads((args.output / 'report.json').read_text(encoding='utf-8'))
    assert len(report['parameter_search']) == 8
    assert {row['config']['child_weight'] for row in report['parameter_search']} == {0.25, 0.5, 0.75, 1}
    assert {row['config']['aggregation'] for row in report['parameter_search']} == {'max', 'mean_top2'}
    assert report['selected'] == {'child_weight': 0.5, 'aggregation': 'max'}
    selection = comparison.load_frozen_selection(args.output / 'selected_config.json')
    assert selection['validation_labels_accessed_for_selection'] is False
    assert selection['development_case_ids'] == ['heldout-1', 'heldout-2']
    assert selection['development_report_sha256'] == comparison.baseline.sha256(args.output / 'report.json')


def test_existing_experiment_directory_is_never_overwritten(tmp_path, monkeypatch):
    args = arguments(tmp_path, command='select')
    args.output.mkdir()
    monkeypatch.setattr(comparison, 'inputs', lambda *args: pytest.fail('Existing evidence directory was touched'))
    with pytest.raises(ValueError, match='will not be overwritten'):
        comparison.run(args)


@pytest.fixture
def fragment_input(tmp_path, monkeypatch):
    queries = [comparison.baseline.Query('case-1', 'computer programmer', 'en')]
    records = [SimpleNamespace(code='2512', level='unit'), SimpleNamespace(code='5120', level='unit')]
    fragments = [ISCOFragment('software-title', '2512', 'title', 'Software developers'),
                 ISCOFragment('software-example', '2512', 'example', 'Computer programmer'),
                 ISCOFragment('cook-title', '5120', 'title', 'Cooks')]
    path = tmp_path / 'fragments.npz'
    np.savez_compressed(path, fragment_vectors=np.stack([basis(1), basis(0), basis(1)]),
                        fragment_codes=np.asarray([fragment.code for fragment in fragments]),
                        fragment_ids=np.asarray([fragment.fragment_id for fragment in fragments]))
    query_meta = {'encoder_id': 'e5-small', 'encoder_revision': 'fixed-revision', 'weights_sha256': 'fixed-weights'}
    catalogue_meta = {'source_catalogue_sha256': 'official-source', 'cache_sha256': 'catalogue-cache'}
    fragment_meta = {'cache_sha256': comparison.baseline.sha256(path),
                     'fragment_map_sha256': comparison.baseline.object_hash([vars(fragment) for fragment in fragments]),
                     'source_catalogue_sha256': 'official-source', 'catalogue_cache_sha256': 'catalogue-cache', **query_meta}
    path.with_suffix('.meta.json').write_text(json.dumps(fragment_meta), encoding='utf-8')
    monkeypatch.setattr(comparison.baseline, 'read_cases', lambda path: (queries, ['2512']))
    monkeypatch.setattr(comparison.baseline, 'load_query_vectors', lambda *args: (basis(0)[None, :], query_meta))
    monkeypatch.setattr(comparison.baseline, 'load_catalogue',
                        lambda *args: ({'unit': (['2512', '5120'], np.stack([basis(0), basis(1)]))}, catalogue_meta, records))
    monkeypatch.setattr(comparison, 'catalogue_fragments', lambda path: (records, fragments))
    return path, fragment_meta, fragments


def test_fragment_vectors_remain_aligned_with_authoritative_parent_codes(fragment_input, tmp_path):
    path, _, _ = fragment_input
    data = comparison.inputs(tmp_path / 'cases.csv', tmp_path / 'queries.npz', path, tmp_path / 'catalogue.npz')
    np.testing.assert_array_equal(data[3], [[1, 0]])
    np.testing.assert_array_equal(data[4]['max'], [[1, 0]])
    np.testing.assert_array_equal(data[4]['mean_top2'], [[0.5, 0]])


@pytest.mark.parametrize('field', ['cache_sha256', 'fragment_map_sha256', 'source_catalogue_sha256',
                                   'catalogue_cache_sha256', 'encoder_id', 'encoder_revision', 'weights_sha256'])
def test_fragment_cache_rejects_incompatible_provenance(fragment_input, tmp_path, field):
    path, metadata, _ = fragment_input
    metadata[field] = 'tampered'
    path.with_suffix('.meta.json').write_text(json.dumps(metadata), encoding='utf-8')
    with pytest.raises(ValueError, match='provenance'):
        comparison.inputs(tmp_path / 'cases.csv', tmp_path / 'queries.npz', path, tmp_path / 'catalogue.npz')


def test_fragment_order_mismatch_is_rejected_even_after_cache_digest_update(fragment_input, tmp_path):
    path, metadata, fragments = fragment_input
    np.savez_compressed(path, fragment_vectors=np.stack([basis(1), basis(0), basis(1)]),
                        fragment_ids=np.asarray([fragment.fragment_id for fragment in reversed(fragments)]))
    metadata['cache_sha256'] = comparison.baseline.sha256(path)
    path.with_suffix('.meta.json').write_text(json.dumps(metadata), encoding='utf-8')
    with pytest.raises(ValueError, match='vector order'):
        comparison.inputs(tmp_path / 'cases.csv', tmp_path / 'queries.npz', path, tmp_path / 'catalogue.npz')


def test_rank_codes_preserves_canonical_code_order_on_similarity_ties():
    assert comparison.rank_codes(np.array([[0.5, 0.5, 0.1]]), ['0110', '2512', '5120']) == [['0110', '2512', '5120']]
