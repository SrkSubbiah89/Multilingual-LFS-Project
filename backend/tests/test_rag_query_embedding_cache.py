"""Embedding preparation keeps labels out of inference and rejects bad IDs."""
import csv

import pytest

from scripts.cache_rag_query_embeddings import load_queries


def write_queries(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_id', 'input_text', 'input_language', 'gold_isco_4digit'])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_query_inputs_and_order_are_independent_of_gold_labels(tmp_path):
    first = {'case_id': 'q-ar', 'input_text': 'مهندس برمجيات', 'input_language': 'ar', 'gold_isco_4digit': '9999'}
    second = {'case_id': 'q-hi', 'input_text': 'चिकित्सक', 'input_language': 'hi', 'gold_isco_4digit': '0000'}
    path = write_queries(tmp_path / 'queries.csv', [first, second])
    expected = [{key: row[key] for key in ('case_id', 'input_text', 'input_language')} for row in [first, second]]
    assert load_queries([path]) == expected
    first['gold_isco_4digit'], second['gold_isco_4digit'] = '2512', '2211'
    write_queries(path, [first, second])
    assert load_queries([path]) == expected


def test_duplicate_query_ids_across_split_files_are_rejected(tmp_path):
    row = {'case_id': 'same', 'input_text': 'nurse', 'input_language': 'en', 'gold_isco_4digit': '2221'}
    dev = write_queries(tmp_path / 'dev.csv', [row])
    validation = write_queries(tmp_path / 'validation.csv', [{**row, 'input_text': 'driver'}])
    with pytest.raises(ValueError, match='duplicate'):
        load_queries([dev, validation])


@pytest.mark.parametrize('query', ['', '   ', '\t'])
def test_empty_query_is_rejected_before_model_load(tmp_path, query):
    path = write_queries(tmp_path / 'queries.csv', [{'case_id': 'q', 'input_text': query,
                        'input_language': 'en', 'gold_isco_4digit': '2512'}])
    with pytest.raises(ValueError, match='Blank'):
        load_queries([path])


def test_missing_query_column_is_rejected(tmp_path):
    path = tmp_path / 'bad.csv'
    path.write_text('case_id,input_text,gold_isco_4digit\nq,nurse,2221\n', encoding='utf-8')
    with pytest.raises(ValueError, match='input_language'):
        load_queries([path])
