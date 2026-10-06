"""Reconcile historical and current ISCO results without running inference.

Only aggregates and source hashes are published. CSV input text and case IDs
remain in memory/local sources and are never copied to the audit report.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / 'eval/results/raw_runs/enriched_e5large_heldout_20260824/20260824T123941Z_flat.csv'
SMALL = ROOT / 'eval/results/raw_runs/enriched_catalogue_heldout_20260824/20260824T113739Z_flat.csv'
CORRECTION = ROOT / 'eval/results/corrections_20261004/raw_runs/enriched_e5large_heldout_20260824/20260824T123941Z_flat.provenance_correction.json'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def insert(records, row, prediction):
    key = row['case_id']
    if not key or key in records:
        raise ValueError('Missing or duplicate case ID in a method')
    gold = row['gold_isco_4digit']
    valid_code = lambda code: len(code) == 4 and all(char in '0123456789' for char in code)
    if not valid_code(gold) or (prediction and not valid_code(prediction)):
        raise ValueError('Gold must be a four-digit ISCO code; predictions must be blank or four digits')
    records[key] = (row['input_text'], row['input_language'], row['gold_isco_4digit'], prediction)


def legacy(path):
    records = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['error'] or row['reranker_fired'].lower() != 'false':
                raise ValueError('Historical run contains an error or an executed reranker')
            insert(records, row, row['pred_isco_4digit'])
    return records


def current(path):
    methods = {'dense_flat': {}, 'parent_document_rag': {}}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['method'] not in methods:
                raise ValueError('Unexpected method in current predictions')
            insert(methods[row['method']], row, row['prediction'])
    return methods


def align(reference, other):
    if reference.keys() != other.keys():
        raise ValueError('Case sets differ; comparison is not paired')
    if any(reference[key][:3] != other[key][:3] for key in reference):
        raise ValueError('Input text, language or gold code differs across runs')


def metrics(records):
    correct = sum(row[2] == row[3] for row in records.values())
    n = len(records)
    if not n:
        raise ValueError('Empty prediction run')
    return {'n': n, 'top1_correct': correct, 'top1_accuracy': correct / n}


def paired(reference, candidate):
    align(reference, candidate)
    both = sum(row[2] == row[3] and candidate[key][2] == candidate[key][3] for key, row in reference.items())
    reference_correct = metrics(reference)['top1_correct']
    candidate_correct = metrics(candidate)['top1_correct']
    return {'both_correct': both, 'reference_only_correct': reference_correct - both,
            'candidate_only_correct': candidate_correct - both,
            'both_wrong': len(reference) - reference_correct - candidate_correct + both,
            'prediction_mismatches': sum(reference[key][3] != candidate[key][3] for key in reference),
            'difference_correct_candidate_minus_reference': candidate_correct - reference_correct,
            'difference_percentage_points_candidate_minus_reference': 100 * (candidate_correct - reference_correct) / len(reference)}


def bind_current_evidence(comparison_report, methods):
    report = json.loads(Path(comparison_report).read_text(encoding='utf-8'))
    config_path = ROOT / 'backend/rag/parent_isco_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    selection_path = ROOT / 'backend/rag/parent_isco_selection.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    selection_hash = selection.pop('selection_sha256')
    calculated = hashlib.sha256(json.dumps(selection, sort_keys=True, ensure_ascii=False,
                                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    encoder = report['query_encoder']
    if (report.get('split') != 'heldout' or selection_hash != calculated
            or report['selection_sha256'] != selection_hash or config['selection_sha256'] != selection_hash
            or report['selected'] != selection['config']
            or report['selected'] != {'child_weight': config['child_weight'], 'aggregation': config['aggregation']}
            or report['source_hashes'] != selection['source_hashes']
            or encoder['encoder_id'] != config['encoder_id']
            or encoder['encoder_revision'] != config['encoder_revision']
            or encoder['weights_sha256'] != config['encoder_weights_sha256']
            or report['catalogue']['profile'] != config['profile']):
        raise ValueError('Current comparison identity differs from frozen serving configuration')
    for name, records in methods.items():
        measured = metrics(records)
        if any(report['methods'][name][key] != value for key, value in measured.items()):
            raise ValueError('Current prediction counts differ from their comparison report')
    return {'selection_sha256': selection_hash, 'selected': report['selected'],
            'query_encoder': encoder, 'catalogue_profile': config['profile'],
            'fragment_cache_sha256': selection['fragment_cache_sha256'],
            'catalogue_cache_sha256': selection['catalogue_cache_sha256'],
            'comparison_report_sha256': digest(comparison_report),
            'runtime_config_sha256': digest(config_path), 'selection_file_sha256': digest(selection_path)}


def audit(predictions, comparison_report):
    correction = json.loads(CORRECTION.read_text(encoding='utf-8'))
    historical_hash = digest(HISTORICAL)
    if (correction['source_sha256'] != historical_hash
            or correction['executed_embedding_model'] is not None
            or correction['provenance_status'] != 'historical_execution_identity_unresolved'):
        raise ValueError('Historical source or unresolved provenance correction changed')
    historical, small = legacy(HISTORICAL), legacy(SMALL)
    now = current(predictions)
    current_identity = bind_current_evidence(comparison_report, now)
    methods = {'historical_intended_e5large_encoder_unresolved': historical,
               'historical_enriched_dense_small': small, **now}
    for records in methods.values():
        align(historical, records)
    if len(historical) != 18747:
        raise ValueError('Expected the complete 18,747-case reference split')
    return {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'WISCO ISCO-08 v3 historically reused heldout split',
        'case_ids_input_text_languages_gold_codes_identical': True,
        'labour_force_survey_field_validation': False,
        'historical_encoder_provenance': correction,
        'current_frozen_reference_identity': current_identity,
        'methods': {name: metrics(records) for name, records in methods.items()},
        'per_language': {language: {name: metrics({key: row for key, row in records.items() if row[1] == language})
                         for name, records in methods.items()}
                         for language in sorted({row[1] for row in historical.values()})},
        'paired_comparisons': {
            'historical_small_to_current_dense': paired(small, now['dense_flat']),
            'current_dense_to_parent': paired(now['dense_flat'], now['parent_document_rag']),
            'historical_intended_large_to_parent': paired(historical, now['parent_document_rag'])},
        'source_sha256': {'historical_intended_large': historical_hash,
                         'historical_small': digest(SMALL), 'current_predictions': digest(predictions),
                         'historical_correction': digest(CORRECTION), 'audit_script': digest(__file__)},
        'interpretation': (f"{metrics(now['dense_flat'])['top1_accuracy']:.2%} is the same-profile dense baseline; "
            f"the configured parent method scored {metrics(now['parent_document_rag'])['top1_accuracy']:.2%}. "
            f"Historical {metrics(historical)['top1_accuracy']:.2%} has unresolved executed encoder provenance; "
            'no verified encoder comparison or field-accuracy claim.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--comparison-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Existing accuracy audit will not be overwritten')
    report = audit(args.predictions, args.comparison_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({'methods': report['methods'], 'paired': report['paired_comparisons']}))


if __name__ == '__main__':
    main()
