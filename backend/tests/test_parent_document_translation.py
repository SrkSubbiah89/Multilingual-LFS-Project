"""Opt-in query translation on the parent-document path.

These tests are hermetic: no encoder, Qdrant instance or language model is
loaded. They pin the contract only -- default off, non-English queries
translated before embedding, English untouched, and any translation failure
degrading to the original text. They do NOT establish that translation
improves accuracy; no development, validation or heldout run exists for this
combination.
"""
from types import SimpleNamespace as NS

import pytest

from backend.agents.parent_document_isco_classifier import ParentDocumentISCOClassifier
from backend.tests.test_parent_document_isco_classifier import classifier


def _embedded(clf):
    """Capture the text that actually reaches the encoder."""
    seen = []
    original = clf.embed_query
    clf.embed_query = lambda query: (seen.append(query) or original(query))
    return seen


def test_default_is_off_and_query_reaches_encoder_untouched(monkeypatch):
    clf, _, _, _ = classifier()
    seen = _embedded(clf)
    monkeypatch.setattr(ParentDocumentISCOClassifier, '_translate_to_english',
                        lambda self, text, language: pytest.fail('translation must not run by default'))
    trace = {}
    clf.classify('مهندس برمجيات', language='ar', trace=trace)
    assert seen == ['مهندس برمجيات']
    assert 'translated_before_retrieval' not in trace


def test_non_english_query_is_translated_before_embedding():
    clf, _, _, _ = classifier()
    clf.translate_before_retrieval = True
    seen = _embedded(clf)
    clf._translate_to_english = lambda text, language: 'software engineer'
    trace = {}
    result = clf.classify('مهندس برمجيات', language='ar', trace=trace)
    assert seen == ['software engineer']
    assert trace['translated_before_retrieval'] is True
    assert trace['translation_source_language'] == 'ar'
    # The reported query stays the respondent's own text, not the translation.
    assert result.query == 'مهندس برمجيات'


def test_english_query_is_never_translated():
    clf, _, _, _ = classifier()
    clf.translate_before_retrieval = True
    seen = _embedded(clf)
    clf._translate_to_english = lambda text, language: pytest.fail('English must not be translated')
    trace = {}
    clf.classify('software engineer', language='en', trace=trace)
    assert seen == ['software engineer']
    assert 'translated_before_retrieval' not in trace


@pytest.mark.parametrize('returned', ['', '   '])
def test_empty_translation_falls_back_to_original_text(returned):
    clf, _, _, _ = classifier()
    clf.translate_before_retrieval = True
    seen = _embedded(clf)
    clf._translate_to_english = lambda text, language: returned
    clf.classify('مهندس برمجيات', language='ar', trace={})
    assert seen == ['مهندس برمجيات']


def test_translation_helper_uses_no_instance_state():
    """The borrowed ISCOClassifier method is called with ``None`` as self, so a
    future change that relies on instance state fails loudly here first."""
    from backend.agents import isco_classifier as module

    calls = {}

    class _LLM:
        def call(self, messages):
            calls['prompt'] = messages[0]['content']
            return '  software engineer  '

    module_get_llm_strict = module.get_llm_strict
    module.get_llm_strict = lambda model, temperature: _LLM()
    try:
        clf = object.__new__(ParentDocumentISCOClassifier)
        assert clf._translate_to_english('مهندس برمجيات', 'ar') == 'software engineer'
    finally:
        module.get_llm_strict = module_get_llm_strict
    assert 'مهندس برمجيات' in calls['prompt']


def test_translation_failure_degrades_to_original_text():
    from backend.agents import isco_classifier as module

    def _raise(model, temperature):
        raise RuntimeError('ollama unreachable')

    module_get_llm_strict = module.get_llm_strict
    module.get_llm_strict = _raise
    try:
        clf = object.__new__(ParentDocumentISCOClassifier)
        assert clf._translate_to_english('مهندس برمجيات', 'ar') == 'مهندس برمجيات'
    finally:
        module.get_llm_strict = module_get_llm_strict
