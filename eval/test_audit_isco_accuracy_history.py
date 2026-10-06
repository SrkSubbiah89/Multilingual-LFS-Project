"""An accuracy difference must compare identical cases and preserve failures."""
import hashlib
import json
import pytest

from scripts.audit_isco_accuracy_history import align, insert, paired
from scripts import audit_isco_accuracy_history as audit


@pytest.mark.parametrize('other', [
    {'case-b': ('cook', 'en', '5120', '5120')},
    {'case-a': ('chef', 'en', '5120', '5120')},
    {'case-a': ('cook', 'ar', '5120', '5120')},
    {'case-a': ('cook', 'en', '5131', '5120')},
])
def test_case_or_input_difference_cannot_be_reported_as_retrieval_regression(other):
    with pytest.raises(ValueError):
        align({'case-a': ('cook', 'en', '5120', '5120')}, other)


def test_duplicate_case_cannot_inflate_accuracy():
    row = {'case_id': 'case-a', 'input_text': 'cook', 'input_language': 'en', 'gold_isco_4digit': '5120'}
    records = {}
    insert(records, row, '5120')
    with pytest.raises(ValueError, match='duplicate'):
        insert(records, row, '5120')


@pytest.mark.parametrize('gold,prediction', [('', ''), ('abc', '0110'), ('110', '0110'),
                                          ('0110', '110'), ('0110', 'ABCD')])
def test_invalid_gold_and_prediction_codes_cannot_count_as_correct(gold, prediction):
    row = {'case_id': 'case-a', 'input_text': 'title', 'input_language': 'en', 'gold_isco_4digit': gold}
    with pytest.raises(ValueError, match='four-digit ISCO'):
        insert({}, row, prediction)


def test_leading_zero_code_and_abstention_are_preserved():
    row = {'case_id': 'case-a', 'input_text': 'title', 'input_language': 'en', 'gold_isco_4digit': '0110'}
    records = {}
    insert(records, row, '')
    assert records['case-a'][2:] == ('0110', '')


def test_paired_summary_preserves_improvements_regressions_and_missing_predictions():
    reference = {str(i): ('title', 'en', '0110', prediction)
                 for i, prediction in enumerate(['0110', '0110', '1111', ''])}
    candidate = {str(i): ('title', 'en', '0110', prediction)
                 for i, prediction in enumerate(['0110', '', '0110', '1111'])}
    report = paired(reference, candidate)
    assert (report['both_correct'], report['reference_only_correct'],
            report['candidate_only_correct'], report['both_wrong']) == (1, 1, 1, 1)
    assert report['prediction_mismatches'] == 3
    assert report['difference_correct_candidate_minus_reference'] == 0


@pytest.fixture
def current_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    config_directory = tmp_path / 'backend/rag'
    config_directory.mkdir(parents=True)
    choice = {'config': {'child_weight': 0.5, 'aggregation': 'max'}, 'source_hashes': {'retrieval': 'fixed'},
              'fragment_cache_sha256': 'fragments', 'catalogue_cache_sha256': 'catalogue'}
    selection_hash = hashlib.sha256(json.dumps(choice, sort_keys=True, ensure_ascii=False,
                                   separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    choice['selection_sha256'] = selection_hash
    (config_directory / 'parent_isco_selection.json').write_text(json.dumps(choice), encoding='utf-8')
    config = {'encoder_id': 'verified-small', 'encoder_revision': 'fixed-revision',
              'encoder_weights_sha256': 'fixed-weights', 'selection_sha256': selection_hash,
              'profile': 'official', **choice['config']}
    (config_directory / 'parent_isco_config.json').write_text(json.dumps(config), encoding='utf-8')
    methods = {'dense_flat': {'case': ('title', 'en', '0110', '1111')},
               'parent_document_rag': {'case': ('title', 'en', '0110', '0110')}}
    report = {'split': 'heldout', 'selected': choice['config'], 'source_hashes': choice['source_hashes'],
              'selection_sha256': selection_hash, 'catalogue': {'profile': 'official'},
              'query_encoder': {'encoder_id': 'verified-small', 'encoder_revision': 'fixed-revision',
                                'weights_sha256': 'fixed-weights'},
              'methods': {name: audit.metrics(records) for name, records in methods.items()}}
    return tmp_path / 'report.json', report, methods


def test_current_report_is_bound_to_frozen_serving_identity(current_evidence):
    path, report, methods = current_evidence
    path.write_text(json.dumps(report), encoding='utf-8')
    identity = audit.bind_current_evidence(path, methods)
    assert identity['selection_sha256'] == report['selection_sha256']
    assert identity['comparison_report_sha256'] == audit.digest(path)
    assert identity['query_encoder']['weights_sha256'] == 'fixed-weights'


@pytest.mark.parametrize('mismatch', ['encoder', 'parameters', 'source', 'counts'])
def test_changed_current_identity_or_counts_fail_audit(current_evidence, mismatch):
    path, report, methods = current_evidence
    if mismatch == 'encoder':
        report['query_encoder']['weights_sha256'] = 'different-weights'
    elif mismatch == 'parameters':
        report['selected'] = {'child_weight': 0.75, 'aggregation': 'max'}
    elif mismatch == 'source':
        report['source_hashes'] = {'retrieval': 'different'}
    else:
        report['methods']['dense_flat']['top1_correct'] = 1
    path.write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError):
        audit.bind_current_evidence(path, methods)
