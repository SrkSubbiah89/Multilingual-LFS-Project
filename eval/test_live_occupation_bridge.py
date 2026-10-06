"""The native CrewAI probe must forward explicit duties without live services."""
import io
import json
from urllib.parse import parse_qs, unquote, urlsplit

from scripts.verify_live_classification_crew import ReadOnlyOccupationBridge


def test_bridge_forwards_native_duties_and_language_with_read_only_retrieval(monkeypatch):
    observed = []

    def response(url, **kwargs):
        observed.append((url, kwargs))
        return io.BytesIO(json.dumps({'code': '2512', 'title': 'Software developers', 'conf': 0.71,
            'method': 'isco_parent_document_duties_rag', 'hitl_required': True}).encode())

    monkeypatch.setattr('urllib.request.urlopen', response)
    bridge = ReadOnlyOccupationBridge(duties='أكتب البرامج وأطور تطبيقات الحاسوب')
    result = bridge.classify('مساعد', language='ar', context='private sector', use_llm=True)
    target = urlsplit(observed[0][0])
    assert target.hostname == '127.0.0.1' and target.port == 8000
    assert unquote(target.path) == '/debug/isco/مساعد'
    assert parse_qs(target.query)['duties'] == ['أكتب البرامج وأطور تطبيقات الحاسوب']
    assert parse_qs(target.query)['language'] == ['ar']
    assert 'private sector' not in observed[0][0]
    assert result.primary.confidence == 0.71 and result.hitl_required is True
    assert result.method == 'isco_parent_document_duties_rag'
    assert bridge.calls[0]['explicit_duties_forwarded'] is True
    assert bridge.calls[0]['language_forwarded'] is True
    assert bridge.calls[0]['llm_reranking'] is False


def test_bridge_without_duties_preserves_title_only_method(monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: io.BytesIO(json.dumps({
        'code': '0110', 'title': 'Commissioned armed forces officers', 'conf': 0.73,
        'method': 'isco_parent_document_rag', 'hitl_required': True}).encode()))
    bridge = ReadOnlyOccupationBridge()
    result = bridge.classify('officer', language='en')
    assert result.primary.code == '0110' and result.method == 'isco_parent_document_rag'
    assert bridge.calls[0]['explicit_duties_forwarded'] is False
