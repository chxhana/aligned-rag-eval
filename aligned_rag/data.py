"""Explicit attack interchange and streaming corpus storage; standard library only."""
import csv
import gzip
import json
import sqlite3
import subprocess
from pathlib import Path

from exp11_ablation import fingerprint, save_json
from research_integrity import file_sha256


def read(path):
    return json.loads(Path(path).read_text())


def validate_manifest(data):
    if data.get('schema_version') != 1 or not data.get('dataset') or not data.get('attack'):
        raise ValueError('manifest requires schema_version=1, dataset and attack provenance')
    rows = data.get('records', [])
    if not rows:
        raise ValueError('empty candidate cohort')
    ids = set()
    for row in rows:
        for field in ('id', 'query', 'target', 'group_id'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'missing/non-string {field}')
        if row['id'] in ids:
            raise ValueError('duplicate candidate ID')
        ids.add(row['id'])
        if not isinstance(row.get('poisons'), list):
            raise ValueError('poisons must be a list; [] retains a failed construction')
        for text in row['poisons']:
            if not isinstance(text, str) or not text.strip():
                raise ValueError('empty/non-string injected passage')
        if not isinstance(row.get('references', []), list):
            raise ValueError('references must be a list')
        if any(not isinstance(x, str) or not x.strip() for x in row.get('references', [])):
            raise ValueError('invalid reference')
        if not isinstance(row.get('support_ids', []), list):
            raise ValueError('support_ids must be explicit corpus IDs')
    return data


def released_poisonedrag(repo, dataset='hotpotqa', adv_per_query=5, group_size=10,
                         qrels=None, references=None):
    """Exact released LM_targeted assembly: question + '.' + each released text.

    Groups preserve upstream main.py's simultaneous M-query poisoning. No new
    LLM construction is claimed; model-generated reference answers are marked.
    """
    repo = Path(repo)
    if adv_per_query < 1 or group_size < 1:
        raise ValueError('positive poison and group budgets required')
    artifact = repo / 'results' / 'adv_targeted_results' / f'{dataset}.json'
    source = read(artifact)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    supports = {}
    if qrels:
        with open(qrels) as f:
            for row in csv.DictReader(f, delimiter='\t'):
                if float(row['score']) > 0:
                    supports.setdefault(row['query-id'], []).append(row['corpus-id'])
    reference_map = {}
    if references:
        raw = read(references)
        if isinstance(raw, dict) and 'safe' in raw:
            raw = raw['safe']
        if not isinstance(raw, list):
            raise ValueError('reference file must be a JSON list or {safe:[...]}')
        for r in raw:
            query = r.get('question', r.get('query'))
            answer = r.get('answer')
            if query and answer:
                reference_map[query] = answer if isinstance(answer, list) else [answer]
    rows = []
    for i, (qid, r) in enumerate(source.items()):
        if len(r['adv_texts']) < adv_per_query:
            raise ValueError(f'{qid}: fewer than requested released poisons; refusing silent truncation')
        question = r['question']
        if references and question not in reference_map:
            raise ValueError(f'missing benchmark reference for {qid}; retain the fixed cohort')
        rows.append({'id': str(qid), 'query': question, 'target': r['incorrect answer'],
                     'references': reference_map.get(question, [r['correct answer']]),
                     'reference_source': 'supplied_benchmark' if references else 'released_model_generated',
                     'poisons': [question + '.' + t for t in r['adv_texts'][:adv_per_query]],
                     'support_ids': supports.get(str(qid), []),
                     'support_semantics': 'qrels_relevant_set_not_designated_split_A',
                     'group_id': str(i // group_size), 'construction_status': 'released_selected',
                     'construction_calls': None})
    return validate_manifest({'schema_version': 1, 'dataset': dataset,
        'attack': {'name': 'PoisonedRAG', 'variant': 'LM_targeted',
                   'source_url': 'https://github.com/sleeepeer/PoisonedRAG',
                   'source_revision': revision, 'artifact_sha256': file_sha256(artifact),
                   'attack_code_sha256': file_sha256(repo / 'src/attack.py'),
                   'evaluation': 'released_attack_artifacts_with_common_evaluator',
                   'source_generation_budget': 'not recorded in released artifact',
                   'adv_per_query': adv_per_query, 'simultaneous_queries_per_group': group_size,
                   'reference_file_sha256': file_sha256(references) if references else None},
        'records': rows})


def split_manifest(path, dataset, support_map=None):
    source = read(path)
    rows = []
    mapping = read(support_map) if support_map else {}
    for i, r in enumerate(source['attack']):
        context = r['context']
        if len(context) != 6:
            raise ValueError('split import requires the documented 3+3 layout')
        key = r.get('source_id', fingerprint([r['query'], r['false_answer']]))
        rows.append({'id': key, 'query': r['query'], 'target': r['false_answer'],
                     'references': [r['answer']], 'reference_source': 'source_artifact',
                     'poisons': [' '.join(context[3:])], 'support_ids': mapping.get(key, []),
                     'support_texts': [' '.join(context[:3])],
                     'support_semantics': 'designated_split_A',
                     'group_id': str(i), 'construction_status': 'selected',
                     'construction_calls': r.get('trials'),
                     'verification_stage': r.get('verification_stage', 'legacy_unknown')})
    return validate_manifest({'schema_version': 1, 'dataset': dataset,
        'attack': {'name': 'split_knowledge', 'artifact_sha256': file_sha256(path),
                   'source_metadata': source.get('meta', {})}, 'records': rows})


def corpus_rows(path, format):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8', newline='') as f:
        if format == 'dpr-tsv':
            for r in csv.DictReader(f, delimiter='\t'):
                yield str(r['id']), r.get('title', ''), r['text']
        else:
            for line in f:
                r = json.loads(line)
                yield str(r.get('_id', r.get('id'))), r.get('title', ''), r['text']


def import_corpus(path, destination, format='beir-jsonl', limit=0):
    """Create an indexed on-disk corpus without loading millions of passages."""
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=False)
    conn = sqlite3.connect(dest / 'corpus.sqlite')
    try:
        conn.execute('CREATE TABLE docs (rownum INTEGER PRIMARY KEY, id TEXT UNIQUE, title TEXT, text TEXT)')
        n = 0
        empty_text_rows = 0
        for ident, title, text in corpus_rows(path, format):
            if ident in ('', 'None') or not isinstance(text, str) or not isinstance(title, str):
                raise ValueError(f'invalid corpus row {n}: id={ident!r}, '
                                 f'title_type={type(title).__name__}, text_type={type(text).__name__}')
            empty_text_rows += int(not text.strip())
            conn.execute('INSERT INTO docs VALUES (?,?,?,?)', (n, ident, title, text))
            n += 1
            if n % 10000 == 0:
                conn.commit()
                print(f'[corpus] {n} passages', flush=True)
            if limit and n >= limit:
                break
        if not n:
            raise ValueError('empty corpus')
        conn.commit()
        save_json(dest / 'manifest.json', {'schema_version': 1, 'n': n, 'format': format,
                  'source': str(Path(path).resolve()), 'source_sha256': file_sha256(path),
                  'selection': 'full_source' if not limit else 'prefix', 'limit': limit,
                  'empty_text_rows': empty_text_rows, 'empty_text_policy': 'preserve',
                  'database_sha256': file_sha256(dest / 'corpus.sqlite')})
    finally:
        conn.close()


class Corpus:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = read(self.path / 'manifest.json')
        if file_sha256(self.path / 'corpus.sqlite') != self.meta['database_sha256']:
            raise ValueError('corpus database content changed after import')
        self.conn = sqlite3.connect(f'file:{self.path / "corpus.sqlite"}?mode=ro', uri=True)
        if self.conn.execute('SELECT COUNT(*) FROM docs').fetchone()[0] != self.meta['n']:
            raise ValueError('corpus count mismatch')

    def get(self, rownum):
        row = self.conn.execute('SELECT id,title,text FROM docs WHERE rownum=?', (int(rownum),)).fetchone()
        if row is None:
            raise ValueError(f'unknown row {rownum}')
        return dict(zip(('id', 'title', 'text'), row))

    def batches(self, start=0, size=256):
        cur = self.conn.execute('SELECT id,title,text FROM docs WHERE rownum>=? ORDER BY rownum', (start,))
        while True:
            rows = cur.fetchmany(size)
            if not rows:
                break
            yield [dict(zip(('id', 'title', 'text'), row)) for row in rows]
