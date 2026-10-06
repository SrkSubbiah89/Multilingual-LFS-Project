"""Evidence integrity and read-only catalogue snapshot regressions."""
from contextlib import contextmanager
import io
import json

import numpy as np
import pytest

from eval import compare_hybrid_isco as comparison
from backend.rag.official_isco08_catalogue import OfficialCatalogueRecord, PROFILE_COLLECTION_NAMES


def toy_records():
    return [OfficialCatalogueRecord(code=code, level=level, parent_code=parent,
            title_en=title, embedding_text=code + ' ' + title,
            profile=comparison.PROFILES[0], source_catalogue_sha256='a' * 64)
            for level, code, parent, title in [
                ('major', '2', '', 'Professionals'), ('submajor', '25', '2', 'ICT professionals'),
                ('minor', '251', '25', 'Software professionals'),
                ('unit', '2511', '251', 'Systems analysts'), ('unit', '2512', '251', 'Software developers')]]


@pytest.mark.parametrize('corrupt_flat', [False, True])
def test_snapshot_roundtrip_checks_flat_and_unit_identity(tmp_path, monkeypatch, corrupt_flat):
    records = toy_records()
    monkeypatch.setattr(comparison, 'authoritative_records', lambda profile: records)
    monkeypatch.setattr(comparison, 'COUNTS', {'major': 1, 'submajor': 1, 'minor': 1, 'unit': 2})
    names = PROFILE_COLLECTION_NAMES[comparison.PROFILES[0]]
    vector = (np.ones(384, dtype=np.float32) / np.sqrt(384)).tolist()

    @contextmanager
    def fake_urlopen(request, timeout):
        collection = request.full_url.split('/collections/')[1].split('/')[0]
        level = next(level for level, name in names.items() if name == collection)
        body = json.loads(request.data)
        assert body['with_vector'] and body['with_payload']
        points = []
        for record in records:
            if record.level != ('unit' if level == 'flat' else level):
                continue
            actual = list(vector)
            if corrupt_flat and level == 'flat':
                actual[0] *= -1
            points.append({'id': record.code, 'payload': vars(record), 'vector': actual})
        yield io.BytesIO(json.dumps({'result': {'points': points, 'next_page_offset': None}}).encode())

    monkeypatch.setattr(comparison, 'urlopen', fake_urlopen)
    target = tmp_path / 'catalogue.npz'
    if corrupt_flat:
        with pytest.raises(ValueError, match='Flat and hierarchical unit'):
            comparison.snapshot_catalogue(comparison.PROFILES[0], target)
        assert not target.exists()
    else:
        comparison.snapshot_catalogue(comparison.PROFILES[0], target)
        catalogue, metadata, actual_records = comparison.load_catalogue(target, comparison.PROFILES[0])
        assert catalogue['unit'][0] == ['2511', '2512']
        assert catalogue['unit'][1].shape == (2, 384)
        assert actual_records == records
        assert metadata['operation'].startswith('read-only')


@pytest.mark.parametrize('tamper', ['selection', 'report', 'validation_flag'])
def test_frozen_selection_rejects_post_selection_changes(tmp_path, tamper):
    report = tmp_path / 'development.json'
    report.write_text('{"n": 2}', encoding='utf-8')
    selection = {'development_report': str(report), 'development_report_sha256': comparison.sha256(report),
                 'validation_labels_accessed_for_selection': False, 'selected': {'weight': 1}}
    selection['selection_sha256'] = comparison.object_hash(selection)
    path = tmp_path / 'selected.json'
    comparison.write_json_new(path, selection)
    assert comparison.load_frozen_selection(path)['selected'] == {'weight': 1}
    if tamper == 'report':
        report.write_text('{"n": 3}', encoding='utf-8')
    else:
        selection['selected']['weight'] = 2
        if tamper == 'validation_flag':
            selection['validation_labels_accessed_for_selection'] = True
            selection.pop('selection_sha256')
            selection['selection_sha256'] = comparison.object_hash(selection)
        path.write_text(json.dumps(selection), encoding='utf-8')
    with pytest.raises(ValueError):
        comparison.load_frozen_selection(path)


def test_metrics_count_abstention_as_an_error():
    result = comparison.metrics([[], ['2511', '2512'], ['2512']], ['2512', '2512', '2512'])
    assert result['top1_correct'] == 1
    assert result['top3_correct'] == 2
    assert result['abstentions'] == 1
    assert result['top1_accuracy'] == pytest.approx(1 / 3)


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_vectors_are_rejected(value):
    vectors = np.ones((1, 384), dtype=np.float32) / np.sqrt(384)
    vectors[0, 0] = value
    with pytest.raises(ValueError):
        comparison.valid_vectors(vectors, rows=1)
