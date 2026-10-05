"""Dataset adapters for clean defense calibration and reference-answer evaluation."""
import ast
import csv
import json
import random
import unicodedata
from pathlib import Path
from exp11_ablation import fingerprint
from .protocol import norm

DATASETS = ('hotpotqa', 'nq', 'musique')
FORMATS = ('auto', 'hotpotqa', 'musique', 'qa-json', 'dpr-nq-tsv')


def question_key(text):
    return norm(unicodedata.normalize('NFKC', text)).rstrip('?').rstrip()


def benchmark_rows(path, dataset, format='auto'):
    if dataset not in DATASETS or format not in FORMATS:
        raise ValueError('Unsupported dataset or benchmark format')
    if format == 'auto':
        format = 'dpr-nq-tsv' if dataset == 'nq' and Path(path).suffix in ('.csv', '.tsv') else dataset if dataset in ('hotpotqa', 'musique') else 'qa-json'
    if format == 'dpr-nq-tsv':
        with open(path, encoding='utf-8', newline='') as f:
            source = []
            for i, fields in enumerate(csv.reader(f, delimiter='\t'), 1):
                if len(fields) != 2:
                    raise ValueError(f'NQ QA row {i}: expected question and answer-list columns')
                source.append({'question': fields[0], 'references': ast.literal_eval(fields[1])})
    else:
        with open(path, encoding='utf-8') as f:
            if Path(path).suffix == '.jsonl':
                source = [json.loads(line) for line in f if line.strip()]
            else:
                source = json.load(f)
        if not isinstance(source, list):
            raise ValueError('Expected a benchmark JSON list or JSONL records, not synthetic safe/attack contexts')
    rows = []
    ids, questions = {}, {}
    for i, item in enumerate(source, 1):
        if not isinstance(item, dict):
            raise ValueError(f'Benchmark row {i} is not an object')
        if format == 'musique':
            if 'answerable' in item and not isinstance(item['answerable'], bool):
                raise ValueError(f'MuSiQue row {i}: answerable must be boolean')
            if item.get('answerable') is False:
                continue
        q = item.get('question', item.get('query'))
        refs = item.get('references', item.get('answers', item.get('answer')))
        refs = [refs] if isinstance(refs, str) else refs
        if not isinstance(q, str) or not q.strip() or not isinstance(refs, list) or not refs or any(not isinstance(x, str) or not x.strip() for x in refs):
            raise ValueError(f'Benchmark row {i}: nonempty question and answer aliases required')
        if format == 'musique':
            aliases = item.get('answer_aliases', [])
            if not isinstance(aliases, list) or any(not isinstance(x, str) or not x.strip() for x in aliases):
                raise ValueError(f'MuSiQue row {i}: invalid answer_aliases')
            refs = refs + aliases
        key = question_key(q)
        ident = str(item.get('_id', item.get('id', 'benchmark:' + fingerprint(key))))
        if not ident or ident == 'None':
            raise ValueError(f'Benchmark row {i}: invalid ID')
        refs = sorted(set(refs))
        if ident in ids and ids[ident] != (key, refs):
            raise ValueError(f'Ambiguous benchmark ID: {ident}')
        if key in questions and questions[key] != refs:
            raise ValueError(f'Ambiguous reference join for question at row {i}')
        ids[ident] = (key, refs)
        questions[key] = refs
        rows.append({'id': ident, 'query': q, 'references': refs})
    return rows, format


def select_dataset_holdout(benchmark, queries, runs, n_calib=500, n_utility=200, seed=42, dataset='hotpotqa', format='auto'):
    if n_calib < 1 or n_utility < 1:
        raise ValueError('Positive calibration and utility sizes required')
    gold, resolved = benchmark_rows(benchmark, dataset, format)
    by_id = {r['id']: r for r in gold}
    by_question = {question_key(r['query']): r for r in gold}
    attacks = [r for v in runs.values() for r in v['data']['retrieval']['records']]
    excluded_ids = {str(r['id']) for r in attacks}
    excluded_q = {question_key(r['query']) for r in attacks}
    if queries:
        with open(queries, encoding='utf-8') as f:
            candidates = []
            seen_ids = set()
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                ident = str(item.get('_id', item.get('id')))
                q = item.get('text', item.get('question', item.get('query')))
                if ident in ('', 'None') or not isinstance(q, str) or not q.strip():
                    raise ValueError('Query file requires an ID and text/question/query')
                if ident in seen_ids: raise ValueError(f'Duplicate query ID: {ident}')
                seen_ids.add(ident)
                ref = by_id.get(ident)
                if ref and question_key(ref['query']) != question_key(q):
                    raise ValueError(f'Benchmark query join mismatch: {ident}')
                ref = ref or by_question.get(question_key(q))
                if ref: candidates.append({'id': ident, 'query': q, 'references': ref['references']})
    else:
        candidates = gold
    seen_q, rows = set(), []
    for r in candidates:
        key = question_key(r['query'])
        if r['id'] in excluded_ids or key in excluded_q or key in seen_q: continue
        rows.append(r); seen_q.add(key)
    print(f'[{dataset} references] {len(rows)} eligible matched disjoint queries ({resolved})', flush=True)
    if len(rows) < n_calib + n_utility:
        raise ValueError(f'Only {len(rows)} matched disjoint benchmark queries; need {n_calib+n_utility}. Supply benchmark answers; no fallback to generated references.')
    random.Random(seed).shuffle(rows)
    return [{**r, 'split': 'calibration' if i < n_calib else 'utility'} for i, r in enumerate(rows[:n_calib+n_utility])]


def reference_description(dataset, format):
    return f'Supplied {dataset} benchmark answer aliases (format={format}); calibration and utility queries exclude attack IDs and question text. Reference-file hash stored in holdout provenance.'
