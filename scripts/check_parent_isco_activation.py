"""Read-only local activation evidence; never logs environment credentials."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool

ROOT = Path(__file__).resolve().parents[1]
TABLES = ('users', 'otp_codes', 'survey_sessions', 'survey_responses', 'audit_logs',
          'data_access_logs', 'agent_decision_logs', 'quality_reviews', 'hitl_queue',
          'person_register', 'survey_report_records', 'alembic_version')


def get(path):
    with urlopen('http://127.0.0.1:8000' + path, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['before', 'after'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Existing runtime evidence must not be overwritten')
    env = dotenv_values(ROOT / '.env')
    url = make_url(env['DATABASE_URL'])
    if url.host not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Runtime database check is restricted to the authorized local database')
    engine = create_engine(url, poolclass=NullPool)
    try:
        with engine.begin() as connection:
            connection.execute(text('SET TRANSACTION READ ONLY'))
            counts = {table: connection.execute(text('SELECT COUNT(*) FROM ' + table)).scalar_one() for table in TABLES}
            revision = connection.execute(text('SELECT version_num FROM alembic_version')).scalar_one()
            supervisor_active = connection.execute(text('SELECT is_active FROM users WHERE id = 5')).scalar_one()
    finally:
        engine.dispose()
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'stage': args.stage,
        'operation': 'Read-only database counts/readiness; after-stage debug queries disable LLM reranking',
        'database_counts': counts, 'migration_revision': revision, 'supervisor_user_id': 5,
        'supervisor_active': supervisor_active,
        'runtime_configuration': {key: env.get(key) for key in ('ISCO_RETRIEVAL_STRATEGY', 'LFS_FAST_MODE',
             'LFS_LOCAL_ONLY', 'ENABLE_SURVEY_CLASSIFICATION_CREW', 'OLLAMA_MODEL', 'HITL_REVIEWER_USER_IDS')},
        'health': get('/health'), 'readiness': get('/ready')}
    if args.stage == 'after':
        queries = {'en': 'software developer', 'ar': 'مطور برمجيات', 'ur': 'سافٹ ویئر ڈویلپر',
                   'hi': 'सॉफ्टवेयर डेवलपर', 'tl': 'tagapagbuo ng software'}
        report['occupation_probes'] = {language: get('/debug/isco/' + quote(query, safe=''))
                                       for language, query in queries.items()}
        if any(result.get('method') != 'isco_parent_document_rag' or result.get('hitl_required') is not True
               for result in report['occupation_probes'].values()):
            raise RuntimeError('The live backend did not execute the configured parent-document method')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'stage': args.stage, 'ready': report['readiness'].get('status'),
                      'counts': counts, 'method_verified': args.stage == 'after'}))


if __name__ == '__main__':
    main()
