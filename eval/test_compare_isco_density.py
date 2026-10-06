"""Rehashed selections still have to agree with their frozen dev evidence."""
import json
import sys
from types import SimpleNamespace

import pytest

from eval import compare_isco_density as density


def frozen_files(tmp_path, monkeypatch):
    source = tmp_path / 'source.py'
    source.write_text('independent official evidence', encoding='utf-8')
    monkeypatch.setattr(density, 'SOURCE_FILES', [source])
    hashes = {str(source): density.common.sha256(source)}
    report = {'split': 'development', 'selected': {'neighbors': 5, 'strength': 0},
              'source_hashes': hashes, 'parent_selection_sha256': 'parent-frozen'}
    report_path = tmp_path / 'report.json'
    report_path.write_text(json.dumps(report), encoding='utf-8')
    selection = {'config': report['selected'], 'source_hashes': hashes,
        'parent_selection_sha256': 'parent-frozen', 'development_report': str(report_path),
        'development_report_sha256': density.common.sha256(report_path),
        'validation_labels_accessed_for_selection': False}
    return selection, report_path


def save_selection(tmp_path, selection):
    selection['selection_sha256'] = density.common.object_hash(selection)
    path = tmp_path / 'selected.json'
    path.write_text(json.dumps(selection), encoding='utf-8')
    return path


def test_rehashed_parameter_change_cannot_bypass_development_evidence(tmp_path, monkeypatch):
    selection, _ = frozen_files(tmp_path, monkeypatch)
    selection['config'] = {'neighbors': 20, 'strength': 1}
    with pytest.raises(ValueError, match='development evidence'):
        density.load_frozen_selection(save_selection(tmp_path, selection),
                                      {'selection_sha256': 'parent-frozen'})


def test_valid_selection_loads_and_changed_report_fails(tmp_path, monkeypatch):
    selection, report_path = frozen_files(tmp_path, monkeypatch)
    path = save_selection(tmp_path, selection)
    assert density.load_frozen_selection(path, {'selection_sha256': 'parent-frozen'})['config']['strength'] == 0
    report_path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='report changed'):
        density.load_frozen_selection(path, {'selection_sha256': 'parent-frozen'})


def test_different_parent_selection_is_rejected(tmp_path, monkeypatch):
    selection, _ = frozen_files(tmp_path, monkeypatch)
    path = save_selection(tmp_path, selection)
    with pytest.raises(ValueError, match='integrity'):
        density.load_frozen_selection(path, {'selection_sha256': 'another-parent'})


def evaluation_arguments(tmp_path, monkeypatch, selection):
    argv = ['density', 'evaluate']
    for name in ['cases', 'queries', 'fragments', 'catalogue', 'parent-selection', 'output']:
        argv.extend(['--' + name, str(tmp_path / name)])
    argv.extend(['--selection', str(selection), '--split', 'heldout'])
    monkeypatch.setattr(sys, 'argv', argv)


def test_bad_freeze_is_rejected_before_labels_are_accessed(tmp_path, monkeypatch):
    selection, _ = frozen_files(tmp_path, monkeypatch)
    selection['config'] = {'neighbors': 20, 'strength': 1}
    path = save_selection(tmp_path, selection)
    monkeypatch.setattr(density.parent, 'load_frozen_selection', lambda _: {
        'selection_sha256': 'parent-frozen', 'config': {'aggregation': 'max'}})
    monkeypatch.setattr(density.parent, 'inputs', lambda *a: pytest.fail('Evaluation labels accessed'))
    evaluation_arguments(tmp_path, monkeypatch, path)
    with pytest.raises(ValueError, match='development evidence'):
        density.main()
    assert not (tmp_path / 'output').exists()


def test_other_parent_aggregation_is_rejected_before_labels_are_accessed(tmp_path, monkeypatch):
    monkeypatch.setattr(density.parent, 'load_frozen_selection', lambda _: {
        'selection_sha256': 'parent-frozen', 'config': {'aggregation': 'mean_top2'}})
    monkeypatch.setattr(density.parent, 'inputs', lambda *a: pytest.fail('Evaluation labels accessed'))
    evaluation_arguments(tmp_path, monkeypatch, tmp_path / 'unused')
    with pytest.raises(ValueError, match='maximum-child'):
        density.main()


def test_evaluation_cannot_overlap_parent_training_when_density_used_subset(tmp_path, monkeypatch):
    selection, _ = frozen_files(tmp_path, monkeypatch)
    selection['development_case_ids'] = ['density-dev']
    path = save_selection(tmp_path, selection)
    frozen_parent = {'selection_sha256': 'parent-frozen', 'config': {'aggregation': 'max'},
        'fragment_cache_sha256': 'fragments', 'catalogue_cache_sha256': 'catalogue',
        'query_encoder': {}, 'development_case_ids': ['density-dev', 'parent-dev']}
    monkeypatch.setattr(density.parent, 'load_frozen_selection', lambda _: frozen_parent)
    monkeypatch.setattr(density.parent, 'inputs', lambda *a: (
        [SimpleNamespace(case_id='parent-dev')], ['1111'], ['1111'], None, None,
        {}, {'cache_sha256': 'catalogue'}, {'cache_sha256': 'fragments'}))
    evaluation_arguments(tmp_path, monkeypatch, path)
    with pytest.raises(ValueError, match='overlaps development'):
        density.main()
    assert not (tmp_path / 'output').exists()
