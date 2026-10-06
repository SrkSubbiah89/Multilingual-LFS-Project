"""Paired correctness and occupation-cluster uncertainty integrity regressions."""

import csv
import hashlib

import pytest

from eval.summarize_parent_isco_results import summarize


LANGUAGES = ('ar', 'en', 'hi', 'tl', 'ur')
FIELDS = ('case_id', 'input_language', 'input_text', 'gold_isco_4digit', 'method', 'prediction')


def pair(key, language, *, text=None, dense_correct=False, parent_correct=True):
    common = {'case_id': f'WISCO-{key}-{language}', 'input_language': language,
              'input_text': text or f'Official test title {key} {language}', 'gold_isco_4digit': '2512'}
    return [dict(common, method='dense_flat', prediction='2512' if dense_correct else '5120'),
            dict(common, method='parent_document_rag', prediction='2512' if parent_correct else '5120')]


def write_predictions(tmp_path, rows):
    path = tmp_path / 'predictions.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_all_five_language_variants_resample_as_one_occupation_family(tmp_path):
    rows = [row for language in LANGUAGES for row in pair('100', language)]
    path = write_predictions(tmp_path, rows)
    result = summarize(path, replicates=100, seed=7)
    assert result['n'] == 5 and result['clusters'] == 1
    assert result['parent_only_correct'] == 5
    assert result['dense_only_correct'] == result['both_correct'] == result['both_wrong'] == 0
    assert result['accuracy_difference'] == 1
    assert result['cluster_bootstrap_percentile95_difference'] == [1, 1]
    assert result['prediction_csv_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_shared_normalized_title_merges_whole_multilingual_families(tmp_path):
    rows = []
    for key, correct, shared in [('100', True, '  SOFTWARE   Developer  '),
                                  ('200', False, 'software developer')]:
        for language in LANGUAGES:
            rows.extend(pair(key, language, text=shared if language == 'en' else None,
                             dense_correct=not correct, parent_correct=correct))
    result = summarize(write_predictions(tmp_path, rows), replicates=100)
    assert result['n'] == 10 and result['clusters'] == 1
    assert result['parent_only_correct'] == result['dense_only_correct'] == 5
    assert result['accuracy_difference'] == 0
    assert result['cluster_bootstrap_percentile95_difference'] == [0, 0]


def test_shared_title_merges_are_transitive_across_different_languages(tmp_path):
    rows = []
    for key in ('100', '200', '300'):
        for language in LANGUAGES:
            text = None
            if key in ('100', '200') and language == 'en':
                text = 'First shared title'
            if key in ('200', '300') and language == 'ar':
                text = 'Second shared title'
            rows.extend(pair(key, language, text=text))
    result = summarize(write_predictions(tmp_path, rows), replicates=100)
    assert result['n'] == 15 and result['clusters'] == 1


def test_same_text_in_different_languages_does_not_merge_different_families(tmp_path):
    rows = pair('100', 'en', text='cook') + pair('200', 'ar', text='cook')
    result = summarize(write_predictions(tmp_path, rows), replicates=100)
    assert result['clusters'] == 2


def test_paired_table_includes_wins_losses_both_correct_and_both_wrong(tmp_path):
    rows = pair('100', 'en', dense_correct=False, parent_correct=True)
    rows += pair('200', 'en', dense_correct=True, parent_correct=False)
    rows += pair('300', 'en', dense_correct=True, parent_correct=True)
    rows += pair('400', 'en', dense_correct=False, parent_correct=False)
    result = summarize(write_predictions(tmp_path, rows), replicates=100)
    assert result['n'] == 4
    assert [result[key] for key in ('parent_only_correct', 'dense_only_correct', 'both_correct', 'both_wrong')] == [1, 1, 1, 1]
    assert result['accuracy_difference'] == 0


def test_gain_is_weighted_by_cases_rather_than_average_cluster_accuracy(tmp_path):
    # Five wins in one occupation family and two losses in another: +3/7.
    # Averaging each family's gain would incorrectly report zero.
    rows = [row for language in LANGUAGES for row in pair('100', language)]
    rows += [row for language in ('en', 'ar')
             for row in pair('200', language, dense_correct=True, parent_correct=False)]
    result = summarize(write_predictions(tmp_path, rows), replicates=1000, seed=123)
    assert result['clusters'] == 2 and result['n'] == 7
    assert result['accuracy_difference'] == pytest.approx(3 / 7)
    assert result['cluster_bootstrap_percentile95_difference'] == [-1, 1]


def test_cluster_bootstrap_is_deterministic_for_the_same_input_and_seed(tmp_path):
    rows = []
    for index in range(5):
        for language in LANGUAGES[:index + 1]:
            rows.extend(pair(str(index + 100), language, dense_correct=index % 2 == 0,
                             parent_correct=index % 3 == 0))
    path = write_predictions(tmp_path, rows)
    first = summarize(path, replicates=1000, seed=2026)
    second = summarize(path, replicates=1000, seed=2026)
    assert first == second
    assert first['clusters'] == 5
    assert first['replicates'] == 1000 and first['seed'] == 2026
    low, high = first['cluster_bootstrap_percentile95_difference']
    assert -1 <= low <= first['accuracy_difference'] <= high <= 1


@pytest.mark.parametrize('field,value', [('input_language', 'ar'), ('input_text', 'Different title'),
                                        ('gold_isco_4digit', '5120')])
def test_paired_methods_must_use_the_same_query_and_gold(tmp_path, field, value):
    rows = pair('100', 'en')
    rows[1][field] = value
    with pytest.raises(ValueError, match='Paired query/label differs'):
        summarize(write_predictions(tmp_path, rows), replicates=100)


@pytest.mark.parametrize('change', ['missing_parent', 'different_case', 'foreign_method'])
def test_missing_or_unpaired_method_rows_are_rejected(tmp_path, change):
    rows = pair('100', 'en')
    if change == 'missing_parent':
        rows.pop()
    elif change == 'different_case':
        rows[1]['case_id'] = 'WISCO-200-en'
    else:
        rows[1]['method'] = 'unmeasured_reranker'
    with pytest.raises(ValueError, match='Paired methods'):
        summarize(write_predictions(tmp_path, rows), replicates=100)


def test_duplicate_case_method_is_rejected(tmp_path):
    rows = pair('100', 'en')
    rows.append(rows[0].copy())
    with pytest.raises(ValueError, match='Duplicate method/case'):
        summarize(write_predictions(tmp_path, rows), replicates=100)


@pytest.mark.parametrize('identifier', ['OTHER-100-en', 'WISCO-100-ar', 'WISCO-100-en-extra', '100',
                                        'WISCO--en', 'WISCO-text-en'])
def test_family_identifier_must_match_language_and_wisco_shape(tmp_path, identifier):
    rows = pair('100', 'en')
    for row in rows:
        row['case_id'] = identifier
    with pytest.raises(ValueError, match='family identifier'):
        summarize(write_predictions(tmp_path, rows), replicates=100)


def test_empty_prediction_file_has_no_reportable_accuracy(tmp_path):
    with pytest.raises(ValueError, match='Nonempty cases'):
        summarize(write_predictions(tmp_path, []), replicates=100)


@pytest.mark.parametrize('replicates', [0, 10, 99, True, 100.5, '2000'])
def test_too_few_bootstrap_replicates_are_rejected(tmp_path, replicates):
    with pytest.raises(ValueError, match='[Aa]t least 100'):
        summarize(write_predictions(tmp_path, pair('100', 'en')), replicates=replicates)


def test_abstention_is_a_remaining_error_in_paired_correctness(tmp_path):
    rows = pair('100', 'en', dense_correct=True, parent_correct=False)
    rows[1]['prediction'] = ''
    result = summarize(write_predictions(tmp_path, rows), replicates=100)
    assert result['dense_only_correct'] == 1 and result['accuracy_difference'] == -1


@pytest.mark.parametrize('field,value', [('input_text', '   '), ('gold_isco_4digit', ''),
                                        ('gold_isco_4digit', '123'), ('gold_isco_4digit', '１２３４')])
def test_blank_query_or_malformed_gold_cannot_generate_a_gain(tmp_path, field, value):
    rows = pair('100', 'en')
    for row in rows:
        row[field] = value
    with pytest.raises(ValueError, match='family identifier'):
        summarize(write_predictions(tmp_path, rows), replicates=100)
