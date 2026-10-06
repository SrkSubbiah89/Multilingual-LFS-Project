"""Live retrieval rejects incomplete/unverified evidence and preserves scores."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from backend.agents.parent_document_isco_classifier import ParentDocumentISCOClassifier, METHOD


def classifier():
    clf = object.__new__(ParentDocumentISCOClassifier)
    clf.config = {'collection': 'children', 'child_weight': 0.5,
        'source_catalogue_sha256': 'source', 'encoder_weights_sha256': 'weights',
        'selection_sha256': 'selection', 'encoder_id': 'intfloat/multilingual-e5-small'}
    codes = [f'{i:04}' for i in range(1, 437)]
    clf.units = {code: NS(code=code, title_en='Title ' + code, level='unit',
                    parent_code=code[:3], profile='official_ilo2021_v1_enriched',
                    source_catalogue_sha256='source', embedding_text='Definition ' + code) for code in codes}
    clf.fragments = {code: NS(fragment_id=code, code=code, kind='example', text='Example ' + code) for code in codes}
    child_points = [NS(score=0.6, payload={'unit_code': code, 'fragment_id': code, 'kind': 'example',
                        'text': 'Example ' + code, 'source_catalogue_sha256': 'source',
                        'encoder_weights_sha256': 'weights', 'index_owner': 'Multilingual-LFS/parent-document/v1'}) for code in codes]
    parent_points = [NS(score=0.8, payload=vars(clf.units[code]).copy()) for code in codes]
    groups = [NS(id=code, hits=[point]) for code, point in zip(codes, child_points)]
    calls = []
    clf.client = NS(query_points_groups=lambda **kwargs: (calls.append(kwargs) or NS(groups=groups)),
                    query_points=lambda **kwargs: (calls.append(kwargs) or NS(points=parent_points)))
    clf.embed_query = lambda query: [1.0] + [0.0] * 383
    clf.parent_collection = 'parents'
    return clf, groups, parent_points, calls


def test_live_blend_covers_all_codes_preserves_scores_and_requires_review():
    clf, _, _, calls = classifier()
    trace = {}
    result = clf.classify('  occupation duties  ', language='ur', trace=trace, top_k=5)
    assert result.query == 'occupation duties' and result.language == 'ur'
    assert result.primary.code == '0001'  # deterministic code order for tied scores
    assert result.primary.confidence == pytest.approx(0.7)
    assert len(result.alternatives) == 4 and result.method == METHOD
    assert result.hitl_required is True and result.stage_confidences is None
    assert trace['candidates_scored'] == 436 and trace['score_calibrated'] is False
    assert trace['reranker_fired'] is False and trace['context_used'] is False
    assert all(call['limit'] == 436 and call['search_params'].exact for call in calls)
    assert calls[0]['group_by'] == 'unit_code' and calls[0]['group_size'] == 1


def test_child_evidence_can_move_a_code_above_stronger_parent_definition():
    clf, groups, parents, _ = classifier()
    groups[10].hits[0].score = 0.99
    parents[10].score = 0.7
    result = clf.classify('occupation')
    assert result.primary.code == '0011'
    assert result.primary.confidence == pytest.approx(0.845)


@pytest.mark.parametrize('kind', ['missing_child', 'missing_parent', 'duplicate_child', 'duplicate_parent', 'empty_group'])
def test_incomplete_or_duplicate_global_candidate_pool_is_rejected(kind):
    clf, groups, parents, _ = classifier()
    if kind == 'missing_child': groups.pop()
    if kind == 'missing_parent': parents.pop()
    if kind == 'duplicate_child': groups[-1] = groups[0]
    if kind == 'duplicate_parent': parents[-1] = parents[0]
    if kind == 'empty_group': groups[0].hits = []
    with pytest.raises((ValueError, RuntimeError)):
        clf.classify('occupation')


@pytest.mark.parametrize('field', ['unit_code', 'fragment_id', 'kind', 'text', 'source_catalogue_sha256', 'encoder_weights_sha256', 'index_owner'])
def test_unverified_child_provenance_is_rejected(field):
    clf, groups, _, _ = classifier()
    groups[0].hits[0].payload[field] = 'tampered'
    with pytest.raises(ValueError, match='Unverified child'):
        clf.classify('occupation')


@pytest.mark.parametrize('field', ['code', 'level', 'parent_code', 'title_en', 'profile', 'source_catalogue_sha256', 'embedding_text'])
def test_unverified_parent_provenance_is_rejected(field):
    clf, _, parents, _ = classifier()
    parents[0].payload[field] = 'tampered'
    with pytest.raises(ValueError, match='Unverified parent'):
        clf.classify('occupation')


@pytest.mark.parametrize('vector', [[0.0] * 384, [1.0] * 384, [float('nan')] * 384, [1.0] * 383])
def test_invalid_query_vectors_are_rejected_before_qdrant(vector):
    clf, _, _, calls = classifier()
    clf.embed_query = lambda query: vector
    with pytest.raises(ValueError, match='query embedding'):
        clf.classify('occupation')
    assert not calls


@pytest.mark.parametrize('score', [float('inf'), float('nan')])
def test_nonfinite_evidence_scores_are_rejected(score):
    clf, groups, _, _ = classifier()
    groups[0].hits[0].score = score
    with pytest.raises(ValueError, match='Nonfinite'):
        clf.classify('occupation')


@pytest.mark.parametrize('query,top_k', [('', 3), ('  ', 3), (None, 3), ('job', True), ('job', 0), ('job', 437)])
def test_invalid_query_arguments_never_reach_qdrant(query, top_k):
    clf, _, _, calls = classifier()
    with pytest.raises(ValueError):
        clf.classify(query, top_k=top_k)
    assert not calls


def test_live_factory_concurrent_initialization_loads_one_parent_encoder(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import Mock
    from backend.api import survey_routes as routes
    from backend.agents import parent_document_isco_classifier as module
    instance = NS(ready=True)
    construct = Mock(return_value=instance)
    monkeypatch.setenv('ISCO_RETRIEVAL_STRATEGY', 'parent_document')
    monkeypatch.setattr(routes, '_isco_classifier', None)
    monkeypatch.setattr(module, 'ParentDocumentISCOClassifier', construct)
    with ThreadPoolExecutor(max_workers=8) as executor:
        returned = list(executor.map(lambda _: routes._get_isco_classifier(), range(24)))
    assert all(value is instance for value in returned)
    construct.assert_called_once_with()


def test_live_factory_rejects_unknown_strategy(monkeypatch):
    from backend.api import survey_routes as routes
    monkeypatch.setenv('ISCO_RETRIEVAL_STRATEGY', 'typo')
    monkeypatch.setattr(routes, '_isco_classifier', None)
    with pytest.raises(ValueError, match='Unknown ISCO_RETRIEVAL_STRATEGY'):
        routes._get_isco_classifier()


@pytest.mark.parametrize('employment_status,field', [('employed', 'job_title'), ('unemployed', 'last_job_title')])
def test_completion_fallback_preserves_required_review_even_at_high_similarity(db, user, monkeypatch, employment_status, field):
    from backend.api import survey_routes as routes
    from backend.database.models import SurveySession, SurveyResponse, HITLQueue
    session = SurveySession(user_id=user.id, language='en')
    db.add(session)
    db.flush()
    response = SurveyResponse(session_id=session.id, question_id=field, answer='IT')
    db.add(response)
    db.commit()
    result = NS(primary=NS(code='2512', confidence=0.96), hitl_required=True,
                hierarchy_path=['2', '25', '251', '2512'], reasoning='Uncalibrated parent-document evidence')
    monkeypatch.setattr(routes, '_get_isco_classifier', lambda: NS(classify=lambda *args, **kwargs: result))
    collected = {'employment_status': employment_status, field: 'IT'}
    routes._ensure_isco_classification(db, session.id, collected)
    routes._ensure_isco_classification(db, session.id, collected)
    pending = db.query(HITLQueue).filter_by(session_id=session.id, response_id=response.id, status='pending').all()
    assert response.isco_code == '2512' and response.confidence_score == 0.96
    assert len(pending) == 1 and pending[0].ai_code == '2512'
    assert pending[0].raw_text == 'IT' and pending[0].ai_reasoning == result.reasoning


@pytest.fixture
def actual_frozen_config():
    import json
    from backend.agents import parent_document_isco_classifier as module
    # Load checked-in metadata only; no encoder, network, index or benchmark
    # cache is needed to prove that live parameters match the measured choice.
    return json.loads(module.CONFIG.read_text(encoding='utf-8'))


def test_checked_in_live_config_matches_exact_frozen_development_choice(actual_frozen_config):
    from backend.agents import parent_document_isco_classifier as module
    module._validate_selection(actual_frozen_config)


@pytest.mark.parametrize('field,value', [('child_weight', 0.75), ('aggregation', 'mean_top2'),
                                        ('selection_sha256', '0' * 64)])
def test_unmeasured_live_parameters_cannot_keep_frozen_selection_provenance(actual_frozen_config, field, value):
    from backend.agents import parent_document_isco_classifier as module
    actual_frozen_config[field] = value
    with pytest.raises(ValueError, match='Live parameters differ'):
        module._validate_selection(actual_frozen_config)


@pytest.mark.parametrize('field,value', [('encoder_id', 'different-encoder'),
                                        ('encoder_revision', 'different-revision'),
                                        ('encoder_weights_sha256', '0' * 64)])
def test_live_encoder_identity_must_match_frozen_query_encoder(actual_frozen_config, field, value):
    from backend.agents import parent_document_isco_classifier as module
    actual_frozen_config[field] = value
    with pytest.raises(ValueError, match='Live encoder differs'):
        module._validate_selection(actual_frozen_config)


def test_changed_frozen_snapshot_parameters_fail_digest_verification(tmp_path, actual_frozen_config):
    import json
    from backend.agents import parent_document_isco_classifier as module
    selection = json.loads(module.SELECTION.read_text(encoding='utf-8'))
    selection['config']['child_weight'] = 0.75
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(selection), encoding='utf-8')
    with pytest.raises(ValueError, match='Live parameters differ'):
        module._validate_selection(actual_frozen_config, path)


@pytest.mark.parametrize('flag', [True, None, 0, 'false'])
def test_even_resigned_live_selection_requires_development_only_label_flag(tmp_path, actual_frozen_config, flag):
    import json
    from backend.agents import parent_document_isco_classifier as module
    selection = json.loads(module.SELECTION.read_text(encoding='utf-8'))
    selection['validation_labels_accessed_for_selection'] = flag
    selection.pop('selection_sha256')
    selection['selection_sha256'] = module._digest(selection)
    actual_frozen_config['selection_sha256'] = selection['selection_sha256']
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(selection), encoding='utf-8')
    with pytest.raises(ValueError, match='Live parameters differ'):
        module._validate_selection(actual_frozen_config, path)
