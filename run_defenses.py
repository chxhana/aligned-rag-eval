#!/usr/bin/env python3
"""One defense runner for completed aligned HotpotQA, NQ, or MuSiQue runs."""
import argparse
import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys
from aligned_rag.defense_data import DATASETS, FORMATS
from aligned_rag.defense_suite import load_runs

STAGES = ('prepare', 'scores', 'filter', 'vanilla', 'robustrag_prompt_control',
          'robustrag_keyword', 'trustrag_filter_conflict', 'report', 'labels')


def commands(args, dataset, stamp):
    common = ['--runs', *args.runs, '--out', str(args.out), '--baselines', str(args.baselines)]
    prefix = [sys.executable, '-u', '-m', 'aligned_rag.defense_suite']
    prepared = ['--dataset', dataset, '--corpus', str(args.corpus), '--index', str(args.index),
                '--benchmark', str(args.benchmark), '--benchmark-format', args.benchmark_format,
                '--n-calib', str(args.n_calib), '--n-utility', str(args.n_utility), '--seed', str(args.seed)]
    if args.queries: prepared += ['--queries', str(args.queries)]
    result = []
    stages=STAGES[STAGES.index(args.start_stage):STAGES.index(args.stop_after)+1]
    if getattr(args,'detectors_only',False):
        stages=[s for s in stages if s in ('prepare','scores','report')]
    for stage in stages:
        if stage == 'prepare': cmd = prefix + [stage] + common + prepared
        elif stage in ('scores', 'filter'): cmd = prefix + [stage] + common
        elif stage in ('report', 'labels'):
            extra = ['--legacy-diagnostics', '--report-out', str(args.out/f'defense_report_{stamp}.json')] if stage == 'report' else ['--export-labels', str(args.out/f'defense_target_review_{stamp}.json')]
            if stage == 'report' and getattr(args,'unified_scoring',False): extra += ['--unified-scoring']
            cmd = prefix + [('detectors' if getattr(args,'detectors_only',False) else 'report')] + common + extra
        else: cmd = prefix + ['generate'] + common + ['--method', stage]
        result.append((stage, cmd))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('auto', *DATASETS), default='auto', help='Default: infer from completed attack manifest')
    p.add_argument('--runs', nargs='+', required=True, help='cell_name=completed_run_directory; same dataset, cohort, victim and retrieval depth')
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--benchmark', type=Path, required=True, help='Dataset benchmark answers, not synthetic safe/attack contexts')
    p.add_argument('--queries', type=Path, help='Optional BEIR query JSONL to join/restrict benchmark questions')
    p.add_argument('--benchmark-format', choices=FORMATS, default='auto')
    p.add_argument('--out', type=Path, required=True, help='Separate defense output directory for this matched set of runs')
    p.add_argument('--baselines', type=Path, default=Path('baselines'))
    p.add_argument('--gpu', help='CUDA_VISIBLE_DEVICES value; otherwise preserve environment, default 0')
    p.add_argument('--n-calib', type=int, default=500)
    p.add_argument('--n-utility', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--start-stage', choices=STAGES, default='prepare')
    p.add_argument('--stop-after', choices=STAGES, default='labels')
    p.add_argument('--unified-scoring',action='store_true',help='Apply the common reference extraction policy in the final report')
    p.add_argument('--detectors-only', action='store_true', help='Prepare, score, and report detectors; skip pipelines and utility generation')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    root = str(Path(__file__).resolve().parent)
    for field in ('corpus','index','benchmark','queries','out','baselines'):
        value=getattr(args,field)
        if value is not None: setattr(args,field,value.resolve())
    resolved_runs=[]
    for item in args.runs:
        if '=' not in item: p.error('--runs entries must be cell_name=directory')
        name,path=item.split('=',1)
        if not name or not path: p.error('--runs entries require a name and directory')
        resolved_runs.append(name+'='+str(Path(path).resolve()))
    args.runs=resolved_runs
    if STAGES.index(args.start_stage) > STAGES.index(args.stop_after): p.error('--start-stage is after --stop-after')
    if args.n_calib < 1 or args.n_utility < 1: p.error('Calibration and utility sizes must be positive')
    runs = load_runs(args.runs)
    datasets = {v['data']['retrieval'].get('manifest', {}).get('dataset') for v in runs.values()}
    if args.dataset == 'auto':
        if len(datasets) != 1 or next(iter(datasets)) not in DATASETS: p.error('Cannot infer one supported dataset; supply --dataset and valid aligned manifests')
        args.dataset = next(iter(datasets))
    if any(d is not None and d != args.dataset for d in datasets): p.error('Dataset does not match saved attack runs')
    for path in (args.corpus, args.index, args.benchmark, *([args.queries] if args.queries else [])):
        if not path.exists(): p.error(f'Missing input: {path}')
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    plan = commands(args, args.dataset, stamp)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu if args.gpu is not None else env.get('CUDA_VISIBLE_DEVICES', '0')
    env['TOKENIZERS_PARALLELISM'] = 'false'; env['PYTHONUNBUFFERED'] = '1'
    root = str(Path(__file__).resolve().parent)
    env['PYTHONPATH'] = root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    print(f'[dataset] {args.dataset}; GPU {env["CUDA_VISIBLE_DEVICES"]}; saved corpus index and attack outputs reused', flush=True)
    for stage, cmd in plan:
        print(f'[{stage}] {shlex.join(cmd)}', flush=True)
        if args.dry_run: continue
        logs = args.out/'logs'; logs.mkdir(parents=True, exist_ok=True)
        log_path = logs/f'{stage}_{stamp}.log'
        print(f'[log] {log_path}', flush=True)
        with log_path.open('x') as log:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=root)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True); log.write(line); log.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate(); process.wait(); raise
        if code: raise SystemExit(f'{stage} failed ({code}); inspect {log_path}. Resume with --start-stage {stage} after fixing the cause.')
    if not args.dry_run: print(f'[done] {args.out}; target reports are provisional substring diagnostics until endorsement review.')


if __name__ == '__main__': main()
