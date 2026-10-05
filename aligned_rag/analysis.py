"""Nested events, paired comparisons, stratified review, and portable reporting."""
import copy
import json
import math
import random
from pathlib import Path

from answer_review import RUBRIC, parse_judgment, paired_test
from exp11_ablation import fingerprint, rate_summary, save_json
from research_integrity import paired_bootstrap, unified_qa, screened_target, scoring_metadata
from human_review_ui import write_review_html
from .protocol import answer_id


def case_labels(data, row, review=None, unified=False, screened=False):
    labels = {}
    refs = {}
    for name, docs in row['conditions'].items():
        key = answer_id(row, docs)
        answer = data['answers'][key]
        if review is None:
            from .protocol import legacy_labels
            labels[name] = screened_target(answer['output'], row['target']) if screened else legacy_labels(answer['output'], row['target'])['poisonedrag_substring']
            qa = unified_qa(answer['output'], row['references']) if unified else answer['qa']
            refs[name] = bool(qa and qa['exact_match'])
        else:
            judgment = review['judgments'].get(key)
            if not judgment or not judgment.get('reviewer') or str(judgment['reviewer']).startswith('model:'):
                raise ValueError(f'missing human reviewer: {key}')
            item = {'answer': answer['output'], 'target': row['target']}
            parsed, error = parse_judgment(json.dumps(judgment), item)
            if error or parsed['target_endorsed'] not in ('yes', 'no') or parsed['reference_correct'] not in ('yes', 'no'):
                raise ValueError(f'unresolved/invalid human judgment {key}: {error}')
            labels[name] = parsed['target_endorsed'] == 'yes'
            refs[name] = parsed['reference_correct'] == 'yes'
    return labels, refs


def evaluate(data, review=None, selected_ids=None, unified=False, screened=False):
    if review is not None and (unified or screened):
        raise ValueError('Human adjudication uses its own labels; do not mix scoring overrides')
    if not data.get('complete'):
        raise ValueError('generation is incomplete')
    if data.get('answers_sha256') != fingerprint(data['answers']):
        raise ValueError('answer record hash mismatch')
    rows = data['retrieval']['records']
    if review is not None:
        blank = copy.deepcopy(review)
        signature = blank.pop('packet_sha256', None)
        for judgment in blank['judgments'].values():
            for key in ('target_endorsed', 'reference_correct'): judgment[key] = None
            for key in ('target_evidence', 'reference_evidence', 'rationale', 'reviewer'): judgment[key] = ''
        if fingerprint(blank) != signature:
            raise ValueError('review inputs/selection changed after export')
        if review.get('source_answers_sha256') != fingerprint(data['answers']):
            raise ValueError('answer outputs changed since review export')
        if review.get('source_signature') != data['signature'] or review.get('kind') != 'human_review':
            raise ValueError('review belongs to another run or is not human adjudication')
        selected_ids = review['selected_query_ids']
    if selected_ids is not None:
        if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids) <= {r['id'] for r in rows}:
            raise ValueError('review subset IDs invalid')
        rows = [r for r in rows if r['id'] in selected_ids]
    if not rows:
        raise ValueError('empty evaluation subset')
    records = []
    for row in rows:
        t, g = case_labels(data, row, review, unified, screened)
        support_defined = 'background_only' in t
        strict = (row['retrieval']['co_retrieved'] and t['poisoned'] and
                  not t['background_only'] and not t['support_background'] and not t['poison_background']) if support_defined else None
        new = t['poisoned'] and not t['clean']
        records.append({'id': row['id'], 'group_id': row['group_id'], 'query': row['query'],
            'target': row['target'], 'retrieval': row['retrieval'],
            'clean_target': t['clean'], 'poisoned_target': t['poisoned'],
            'newly_induced': new, 'removed_target': t['clean'] and not t['poisoned'],
            'persistent_target': t['clean'] and t['poisoned'],
            'injection_dependent_new': new and not t['no_poison'],
            'own_injection_dependent_new': new and not t['no_own_poison'],
            'poison_only_sufficient': t['poison_only'] and row['retrieval']['any_poison_retrieved'],
            'single_poison_sufficient': any(v for k, v in t.items() if k.startswith('single_poison_')),
            'support_poison_strict': strict,
            'strict_newly_induced': strict and new if strict is not None else None,
            'strict_new_clean_reference': strict and new and g['clean'] if strict is not None else None,
            'support_semantics': row.get('support_semantics'), 'reference_source': row.get('reference_source'),
            'clean_reference_match': g['clean'], 'poisoned_reference_match': g['poisoned'],
            'target_outside_coretrieval': t['poisoned'] and not row['retrieval']['co_retrieved'] if support_defined else None})
    names = ('clean_target', 'poisoned_target', 'newly_induced', 'removed_target', 'persistent_target',
             'injection_dependent_new', 'own_injection_dependent_new', 'poison_only_sufficient',
             'single_poison_sufficient', 'support_poison_strict', 'strict_newly_induced',
             'strict_new_clean_reference', 'clean_reference_match', 'poisoned_reference_match',
             'target_outside_coretrieval')
    metrics = {}
    for name in names:
        values = [r[name] for r in records if r[name] is not None]
        metrics[name] = {**rate_summary(values), 'eligible_queries': len(values)}
    for name in ('persistent_target', 'injection_dependent_new', 'strict_newly_induced', 'single_poison_sufficient'):
        metrics[name + '/poisoned_targets'] = rate_summary([r[name] for r in records if r['poisoned_target'] and r[name] is not None])
    group_sizes = {}
    for r in records: group_sizes[r['group_id']] = group_sizes.get(r['group_id'], 0) + 1
    before, after = [r['clean_target'] for r in records], [r['poisoned_target'] for r in records]
    groups = list(group_sizes)
    rng = random.Random(42)
    aggregates = {k: [int(r['poisoned_target']) - int(r['clean_target']) for r in records if r['group_id'] == k] for k in groups}
    draws = []
    for _ in range(10000):
        chosen = [aggregates[rng.choice(groups)] for _ in groups]
        draws.append(sum(map(sum, chosen)) / sum(map(len, chosen)))
    draws.sort()
    return {'schema_version': 1, 'source_signature': data['signature'],
            'label_source': 'human_adjudicated_subset' if review else ('negation_screened_heuristic_NOT_endorsement' if screened else 'released_substring_diagnostic'),
            'scoring': scoring_metadata() if unified else {'version': 'saved_reference_scores'},
            'denominator': {'evaluated_queries': len(records), 'full_cohort': len(data['retrieval']['records']),
                            'selection': 'all' if selected_ids is None else 'explicit_subset',
                            'groups': len(groups)},
            'metrics': metrics, 'records': records,
            'paired_target_test': {**paired_test(before, after),
                'caution': 'query-level test is exploratory; shared injection groups may violate independence'},
            'group_bootstrap_target_difference': {'difference': (sum(after)-sum(before))/len(before),
                'ci95': [draws[249], draws[9749]], 'unit': 'injection_group', 'groups': len(groups),
                'draws': 10000, 'seed': 42},
            'interpretation': {'ordinary_single_passage_sufficiency': 'mechanism, not attack failure',
                'strict': 'support-set removal dependence; not a split-knowledge claim unless support_semantics says designated_split_A',
                'reference_match': 'use reference_source to distinguish benchmark from model-generated references',
                'multiplicity': 'exploratory unadjusted; no confirmatory system-ranking claim'}}


def export_review(data, out_dir, per_stratum=15, seed=42, all_queries=False, required_ids=None):
    initial = evaluate(data)
    rows = {r['id']: r for r in data['retrieval']['records']}
    strict_ids = {r['id'] for r in initial['records'] if r['support_poison_strict']}
    required_ids = set(required_ids or [])
    if not required_ids <= set(rows):
        raise ValueError('unknown required review query IDs')
    mandatory = strict_ids | required_ids
    strata = {'target_positive': [], 'target_negative': []}
    for r in initial['records']:
        if r['id'] not in mandatory:
            strata['target_positive' if r['poisoned_target'] else 'target_negative'].append(r['id'])
    rng = random.Random(seed); selected = set(mandatory); sampling = {}
    for name, ids in strata.items():
        sample = ids if all_queries else rng.sample(ids, min(per_stratum, len(ids)))
        selected.update(sample)
        sampling[name] = {'population_queries': len(ids), 'sampled_queries': len(sample),
                          'query_inclusion_probability': len(sample)/len(ids) if ids else None}
    packet = {'kind': 'human_review', 'source_signature': data['signature'],
              'source_answers_sha256': fingerprint(data['answers']), 'rubric': RUBRIC,
              'selected_query_ids': sorted(selected), 'sampling': sampling,
              'mandatory_strict_queries': sorted(strict_ids),
              'mandatory_requested_queries': sorted(required_ids),
              'mandatory_query_inclusion_probability': 1.0,
              'seed': seed, 'judgments': {}}
    evidence = []
    for qid in sorted(selected):
        row = rows[qid]
        evidence.append(row)
        for docs in row['conditions'].values():
            key = answer_id(row, docs); a = data['answers'][key]
            packet['judgments'][key] = {'question': row['query'], 'target': row['target'],
                'reference': ' / '.join(row.get('references', [])), 'answer': a['output'],
                'target_endorsed': None, 'reference_correct': None, 'target_evidence': '',
                'reference_evidence': '', 'rationale': '', 'reviewer': ''}
    ids = sorted(packet['judgments']); rng.shuffle(ids)
    packet['judgments'] = {k: packet['judgments'][k] for k in ids}
    packet['packet_sha256'] = fingerprint(packet)
    root = Path(out_dir); root.mkdir(parents=True, exist_ok=False)
    save_json(root / 'human_labels.json', packet)
    save_json(root / 'selection.json', {k: v for k, v in packet.items() if k != 'judgments'})
    save_json(root / 'evidence_unblinded.json', evidence)
    write_review_html(root / 'review.html', packet)
    print(f'[audit] {len(selected)} complete queries, {len(ids)} distinct outputs; '
          'mandatory cases plus stratified positive/negative queries')
    return packet


def compare_review(data, labels):
    reviewed = evaluate(data, labels)
    original = evaluate(data, selected_ids=labels['selected_query_ids'])
    agree = {}
    for field in ('clean_target', 'poisoned_target', 'support_poison_strict', 'strict_newly_induced'):
        pairs = [(a[field], b[field]) for a, b in zip(original['records'], reviewed['records']) if a[field] is not None]
        n = len(pairs)
        po = sum(a == b for a,b in pairs)/n if n else None
        pa = sum(a for a,b in pairs)/n if n else 0
        pb = sum(b for a,b in pairs)/n if n else 0
        pe = pa*pb + (1-pa)*(1-pb)
        agree[field] = {'n': n, 'agreement': po, 'kappa': (po-pe)/(1-pe) if n and pe != 1 else None,
                        'old_positive_new_negative': sum(a and not b for a,b in pairs),
                        'old_negative_new_positive': sum(not a and b for a,b in pairs)}
    answer_pairs = []
    for key, judgment in labels['judgments'].items():
        answer_pairs.append((data['answers'][key]['legacy']['poisonedrag_substring'],
                             judgment['target_endorsed'] == 'yes'))
    n = len(answer_pairs)
    output_agreement = {'n_distinct_outputs': n,
                        'agreement': sum(a == b for a,b in answer_pairs)/n if n else None,
                        'false_positive_old_labels': sum(a and not b for a,b in answer_pairs),
                        'false_negative_old_labels': sum(not a and b for a,b in answer_pairs),
                        'scope': 'descriptive on stratified sampled outputs, not population accuracy'}
    return {'output_agreement': output_agreement, 'original_full_cohort': evaluate(data), 'original_same_subset': original,
            'adjudicated_subset': reviewed, 'agreement_by_query': agree,
            'sampling': labels.get('sampling'),
            'scope': 'subset rates are conditional on the recorded sampling design; not replacements for full-cohort prevalence'}


def rescore_report(data):
    """No generation or source mutation. All three reports use the same cohort."""
    primary = evaluate(data, unified=True)
    primary['original_scoring_report'] = evaluate(data)
    primary['negation_screened_sensitivity'] = evaluate(data, unified=True, screened=True)
    primary['source_answers_sha256'] = data['answers_sha256']
    primary['answer_scores'] = {
        key: {'output_sha256': fingerprint(a['output']),
              'reference': unified_qa(a['output'], a['references']),
              'screened_target': screened_target(a['output'], a['target'])}
        for key, a in data['answers'].items()}
    primary['reference_summary'] = {}
    for condition in ('clean', 'poisoned'):
        scores = [primary['answer_scores'][answer_id(r, r['conditions'][condition])]['reference']
                  for r in data['retrieval']['records']]
        primary['reference_summary'][condition] = {
            field: sum(s[field] for s in scores) / len(scores)
            for field in ('exact_match', 'token_f1', 'raw_exact_match', 'raw_token_f1')}
    return primary
