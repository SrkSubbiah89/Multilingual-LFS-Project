"""Development-split preview: does translating non-English queries to English
help the served fragment-to-parent retriever?

Motivation, stated before running: the served configuration's own per-language
results show English at 63.91% against Arabic 27.94%, Tagalog 29.85% and Urdu
31.60%, over an index built from English-sourced official ILO definitions and
examples. Translating the query before embedding is therefore a plausible
retrieval-side correction. It is also e5-small, so it carries none of the
memory constraints that block the larger encoder.

Scope and limits, fixed in advance:
  * DEVELOPMENT CASES ONLY. This script refuses to read the heldout split.
  * It is a PREVIEW, not a promotion. No production default changes on any
    outcome. A positive preview would justify a full development run and then
    a heldout confirmation with a gate declared before that run, exactly as
    the catalogue-alignment experiment did.
  * Both arms score the identical sampled cases with the identical classifier,
    so the only difference is the text that reaches the encoder.
  * This project has nine prior null results for additions of this kind. A
    tenth null is an expected and reportable outcome.

Two phases, run separately so the translation model's memory is released
before the encoder is loaded:

    py -3.11 eval/compare_translated_parent_isco.py translate --out <dir>
    ollama stop qwen2.5:3b
    py -3.11 eval/compare_translated_parent_isco.py classify --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from math import comb, log10
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# litellm tries to build a "standard logging object" on every call, which needs
# the optional `apscheduler` proxy extra. Without it each call raises and logs a
# ~25-line traceback: 400 calls buried the real progress output under 10,000
# lines. Non-fatal (litellm catches it and the completion still returns), but it
# makes the log useless and costs real time, so turn the callbacks off. This
# changes logging only, never the prompt, model or temperature.
try:  # pragma: no cover - environment-dependent
    import litellm

    litellm.success_callback = []
    litellm.failure_callback = []
    litellm._async_success_callback = []
    litellm._async_failure_callback = []
    litellm.turn_off_message_logging = True
except Exception:
    pass
DEV_CASES = ROOT / 'eval/local_benchmarks/wisco_isco08_v2_group_split/dev_run_eval_format.csv'
NON_ENGLISH = ('ar', 'hi', 'tl', 'ur')
PER_LANGUAGE = 100
SEED = 42

DESIGN = {
    'experiment': 'Query translation before fragment-to-parent retrieval',
    'status': 'development_preview_not_a_promotion',
    'split': 'development',
    'heldout_accessed': False,
    'sample': f'{PER_LANGUAGE} cases per language, stratified over {list(NON_ENGLISH)}, seed {SEED}',
    'arms': ['control: respondent text unchanged', 'candidate: text translated to English'],
    'metric': 'top-1 exact four-digit ISCO-08 match, paired on identical cases',
    'test': 'exact two-sided McNemar on discordant pairs',
    'production_change_on_any_outcome': False,
    'declared_before_running': True,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample() -> list[dict]:
    rows = list(csv.DictReader(DEV_CASES.open(encoding='utf-8-sig')))
    chosen: list[dict] = []
    for language in NON_ENGLISH:
        pool = [row for row in rows
                if row['input_language'] == language
                and row['input_text'].strip()
                and row['gold_isco_4digit'].strip()]
        pool.sort(key=lambda row: row['case_id'])
        rng = random.Random(f'{SEED}-{language}')
        chosen.extend(rng.sample(pool, min(PER_LANGUAGE, len(pool))))
    return chosen


def _exact_mcnemar(b: int, c: int) -> float | str:
    """Two-sided exact binomial probability, standard library only."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    total = sum(comb(n, i) for i in range(k + 1))
    exponent = log10(2) + log10(total) - n * log10(2)
    if exponent > -300:
        return min(1.0, 2 * total / 2 ** n)
    return f'1e{exponent:.2f}'


def _language_blocks(results: list[dict]) -> dict:
    """Per-language counts AND paired quadrants.

    The aggregate result can be null while individual languages move in
    opposite directions and cancel, which is exactly what the first run of
    this experiment showed. Recording only totals would have hidden that, so
    the paired quadrants are kept per language too.
    """
    blocks = {}
    for language in NON_ENGLISH:
        subset = [r for r in results if r['language'] == language]
        if not subset:
            continue
        control_only = sum(1 for r in subset if r['control_correct'] and not r['candidate_correct'])
        candidate_only = sum(1 for r in subset if r['candidate_correct'] and not r['control_correct'])
        control_correct = sum(1 for r in subset if r['control_correct'])
        candidate_correct = sum(1 for r in subset if r['candidate_correct'])
        blocks[language] = {
            'n': len(subset),
            'control_correct': control_correct,
            'candidate_correct': candidate_correct,
            'difference_correct': candidate_correct - control_correct,
            'control_only_correct': control_only,
            'candidate_only_correct': candidate_only,
            'exact_mcnemar_two_sided_p': _exact_mcnemar(control_only, candidate_only),
        }
    return blocks


def summarize(out: Path) -> None:
    """Rebuild results.json from an existing per_case.csv.

    Separate from `classify` so the aggregate report can be regenerated or
    extended without reloading the encoder and re-running retrieval, and so
    the published report is reproducible from the retained per-case file.
    """
    rows = list(csv.DictReader((out / 'per_case.csv').open(encoding='utf-8-sig')))
    results = [{**row,
                'control_correct': row['control_correct'] == 'True',
                'candidate_correct': row['candidate_correct'] == 'True'}
               for row in rows]
    _write_report(out, results)


def translate(out: Path) -> None:
    from backend.agents.isco_classifier import ISCOClassifier

    out.mkdir(parents=True, exist_ok=True)
    (out / 'design.json').write_text(json.dumps(DESIGN, indent=2) + '\n', encoding='utf-8')
    cases = _sample()
    written = []
    for index, row in enumerate(cases, start=1):
        english = ISCOClassifier._translate_to_english(None, row['input_text'], row['input_language'])
        english = (english or '').strip() or row['input_text']
        written.append({**row, 'translated_text': english,
                        'translation_changed': english != row['input_text']})
        if index % 25 == 0:
            print(f'  translated {index}/{len(cases)}', flush=True)
    target = out / 'translated_cases.csv'
    with target.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(written[0]))
        writer.writeheader()
        writer.writerows(written)
    unchanged = sum(1 for row in written if not row['translation_changed'])
    print(f'Wrote {len(written)} rows to {target}; {unchanged} unchanged by translation')


def classify(out: Path) -> None:
    from backend.agents.parent_document_isco_classifier import ParentDocumentISCOClassifier

    source = out / 'translated_cases.csv'
    cases = list(csv.DictReader(source.open(encoding='utf-8-sig')))
    classifier = ParentDocumentISCOClassifier()
    results = []
    for index, row in enumerate(cases, start=1):
        gold = row['gold_isco_4digit'].strip()
        control = classifier.classify(row['input_text'], language=row['input_language'], top_k=1)
        candidate = classifier.classify(row['translated_text'], language='en', top_k=1)
        results.append({
            'case_id': row['case_id'], 'language': row['input_language'], 'gold': gold,
            'control_code': control.primary.code, 'candidate_code': candidate.primary.code,
            'control_correct': control.primary.code == gold,
            'candidate_correct': candidate.primary.code == gold,
        })
        if index % 25 == 0:
            print(f'  classified {index}/{len(cases)}', flush=True)

    with (out / 'per_case.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    _write_report(out, results)


def _write_report(out: Path, results: list[dict]) -> None:
    both = sum(1 for r in results if r['control_correct'] and r['candidate_correct'])
    control_only = sum(1 for r in results if r['control_correct'] and not r['candidate_correct'])
    candidate_only = sum(1 for r in results if r['candidate_correct'] and not r['control_correct'])
    neither = len(results) - both - control_only - candidate_only
    control_correct = both + control_only
    candidate_correct = both + candidate_only

    by_language = _language_blocks(results)

    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'design': DESIGN,
        'source_cases_sha256': _sha256(DEV_CASES),
        'translated_cases_sha256': _sha256(out / 'translated_cases.csv'),
        'n': len(results),
        'control_correct': control_correct,
        'candidate_correct': candidate_correct,
        'control_accuracy_percent': 100 * control_correct / len(results),
        'candidate_accuracy_percent': 100 * candidate_correct / len(results),
        'difference_percentage_points': 100 * (candidate_correct - control_correct) / len(results),
        'paired': {'both_correct': both, 'control_only_correct': control_only,
                   'candidate_only_correct': candidate_only, 'neither_correct': neither},
        'exact_mcnemar_two_sided_p': _exact_mcnemar(control_only, candidate_only),
        'by_language': by_language,
        'interpretation_limits': [
            'Development split only; no heldout evidence and no promotion.',
            'Reused WISCO occupation titles, not Labour Force Survey field data.',
            'Multilingual variants share source-title groups; the paired test does not account for that clustering.',
            'Translation quality itself was not independently assessed.',
        ],
    }
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    print(f"n={report['n']}  control={control_correct} ({report['control_accuracy_percent']:.2f}%)  "
          f"translated={candidate_correct} ({report['candidate_accuracy_percent']:.2f}%)  "
          f"delta={report['difference_percentage_points']:+.2f}pp")
    print(f"paired b={control_only} c={candidate_only}  exact McNemar p={report['exact_mcnemar_two_sided_p']}")
    for language, block in by_language.items():
        print(f"  {language}: {block['control_correct']} -> {block['candidate_correct']} of {block['n']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('translate', 'classify', 'summarize'):
        command = commands.add_parser(name)
        command.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    {'translate': translate, 'classify': classify, 'summarize': summarize}[args.command](args.out)


if __name__ == '__main__':
    main()
