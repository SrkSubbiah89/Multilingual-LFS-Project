"""Read-only diagnostics must exercise the same duties adapter as surveys."""
from urllib.parse import quote

import pytest

from backend.agents.isco_classifier import ISCOClassification, ISCOMatch
from backend.api import survey_routes


@pytest.fixture
def observed_classifier(monkeypatch):
    calls = []

    class Classifier:
        def classify(self, query, **kwargs):
            calls.append((query, kwargs))
            return ISCOClassification(query=query, language=kwargs.get('language', 'en'),
                primary=ISCOMatch(code='2512', title_en='Software developers', title_ar='', confidence=0.7),
                alternatives=[], method='isco_parent_document_rag', hierarchy_path=['2', '25', '251', '2512'],
                hitl_required=True, reasoning='Synthetic diagnostic evidence')

    monkeypatch.setattr(survey_routes, '_get_isco_classifier', lambda: Classifier())
    return calls


def test_debug_forwards_native_duties_and_disables_llm(client, observed_classifier):
    title, duties = 'सहायक', 'सॉफ्टवेयर बनाता हूँ और प्रोग्राम लिखता हूँ'
    response = client.get('/debug/isco/' + quote(title, safe=''),
                          params={'duties': duties, 'language': 'hi', 'include_trace': 'true'})
    assert response.status_code == 200
    body = response.json()
    assert 'error' not in body
    assert len(observed_classifier) == 1
    query, options = observed_classifier[0]
    assert title in query and duties in query
    assert options['language'] == 'hi' and options['use_llm'] is False
    assert body['method'] == 'isco_parent_document_duties_rag'
    assert body['duties_used'] is True and body['hitl_required'] is True
    assert body['conf'] == 0.7


def test_title_only_debug_keeps_old_method_and_hides_optional_trace(client, observed_classifier):
    response = client.get('/debug/isco/software%20developer')
    body = response.json()
    assert 'error' not in body
    assert observed_classifier[0][0] == 'software developer'
    assert body['method'] == 'isco_parent_document_rag'
    assert body['duties_used'] is False and 'trace' not in body


def test_missing_duties_sentinel_does_not_change_title_only_reference(client, observed_classifier):
    body = client.get('/debug/isco/software%20developer', params={'duties': 'N/A'}).json()
    assert 'error' not in body
    assert observed_classifier[0][0] == 'software developer'
    assert body['method'] == 'isco_parent_document_rag' and body['duties_used'] is False
