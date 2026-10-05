#!/usr/bin/env python3
"""BGE retrieval and quantized-70B victim checks. Print commands unless --execute."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import shlex
from aligned_rag.data import read
from exp11_ablation import fingerprint
from aligned_rag.protocol import official_prompt

CELLS={
    'p1_k5':'poisonedrag_hotpotqa_contriever_llama3_p1_k5',
    'p5_k5':'poisonedrag_hotpotqa_contriever_llama3',
    'split':'split_hotpotqa_contriever_llama3_full',
}
AWQ='hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4'
BGE='BAAI/bge-base-en-v1.5'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['bge','victim70b'],required=True)
    p.add_argument('--gpu',required=True,help='One GPU for BGE; two comma-separated GPUs for 70B')
    p.add_argument('--data-root',type=Path,default=Path.home()/'topo/aligned_rag_data')
    p.add_argument('--runs-root',type=Path,default=Path('pilot_runs'))
    p.add_argument('--benchmark',type=Path,help='Required for new BGE detector calibration')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--start',choices=['retrieval','generation','detectors'],default='retrieval')
    args=p.parse_args()
    if args.arm=='bge' and not args.benchmark:p.error('BGE needs --benchmark for clean calibration')
    if args.arm=='victim70b' and len(set(args.gpu.split(',')))!=2:p.error('70B arm requires two distinct GPU IDs')
    cli=[sys.executable,'-u','-m','aligned_rag.cli']
    env={**os.environ,'CUDA_VISIBLE_DEVICES':args.gpu,'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}
    def run(cmd):
        cmd=list(map(str,cmd));print(shlex.join(cmd),flush=True)
        if args.execute:subprocess.run(cmd,env=env,check=True)
    def pin(model):
        file=args.runs_root/('bge_revision.txt' if args.arm=='bge' else 'llama31_70b_awq_revision.txt')
        if file.exists():return file.read_text().strip()
        if not args.execute:return 'PIN_RESOLVED_ON_EXECUTE'
        from huggingface_hub import model_info
        sha=model_info(model).sha
        if not sha:raise ValueError('Could not resolve model revision')
        file.parent.mkdir(parents=True,exist_ok=True);file.write_text(sha+'\n');return sha
    revision=pin(BGE if args.arm=='bge' else AWQ)
    index=args.data_root/'bge_base_cosine512'
    encoding=['--encoder',BGE,'--encoder-revision',revision,'--score','cosine','--max-length','512']
    for dirname in CELLS.values():
        source=args.runs_root/dirname
        data=read(source/'generation/answers.json')
        ret=read(source/'retrieval/retrieval.json')
        if (not data['complete'] or data['answers_sha256']!=fingerprint(data['answers'])
            or data['retrieval']!=ret or not ret['complete']
            or ret['records_sha256']!=fingerprint(ret['records'])):
            raise ValueError(f'Incomplete or mismatched source artifacts: {source}')
    if args.arm=='bge' and args.start=='retrieval':
        run(cli+['encode','--corpus',args.data_root/'hotpotqa_sqlite','--index',index,'--batch-size','64']+encoding)
    destinations={}
    for cell,dirname in CELLS.items():
        source=args.runs_root/dirname
        data=read(source/'generation/answers.json')
        if not data['complete']:raise ValueError(f'Incomplete source {source}')
        if data['template']!=official_prompt('baselines/PoisonedRAG'):
            raise ValueError('Saved source prompt differs from current official template')
        dest=args.runs_root/(dirname+('_bge_cosine512' if args.arm=='bge' else '_llama31_70b_awq'))
        destinations[cell]=dest
        retrieval=source/'retrieval/retrieval.json'
        if args.arm=='bge':
            manifest=dest/'attacks.json'
            if args.execute:
                manifest.parent.mkdir(parents=True,exist_ok=True)
                if manifest.exists() and read(manifest)!=data['retrieval']['manifest']:
                    raise ValueError(f'Different manifest already exists: {manifest}')
                if not manifest.exists():manifest.write_text(json.dumps(data['retrieval']['manifest'],indent=2)+'\n')
            retrieval=dest/'retrieval/retrieval.json'
            if args.start=='retrieval':
                run(cli+['retrieve','--corpus',args.data_root/'hotpotqa_sqlite','--index',index,
                         '--attacks',manifest,'--top-k','5','--out-dir',dest/'retrieval']+encoding)
        if args.start!='detectors':
            gen=data['generator']
            model=AWQ if args.arm=='victim70b' else gen['model']
            rev=revision if args.arm=='victim70b' else gen['resolved_revision']
            if not rev:raise ValueError('Baseline victim revision must be known')
            run(cli+['generate','--retrieval',retrieval,'--victim',model,'--revision',rev,
                     '--backend','vllm' if args.arm=='victim70b' else gen['backend'],
                     '--tensor-parallel','2' if args.arm=='victim70b' else str(gen['tensor_parallel']),
                     '--batch-size','1','--max-new-tokens',str(gen['max_new_tokens']),
                     '--max-input-tokens',str(gen['max_input_tokens']),
                     '--poisonedrag-prompt-repo','baselines/PoisonedRAG','--out-dir',dest/'generation'])
            report=dest/'report_unified.json'
            if not report.exists():run(cli+['report','--answers',dest/'generation/answers.json','--unified-scoring','--out',report])
    for cells,source_def in [(['p1_k5','p5_k5'],'poisonedrag_defenses_p1_p5_k5'),(['split'],'split_hotpotqa_defenses_full')]:
        out=args.runs_root/(source_def+'_'+args.arm)
        runs=[f'{cell}={destinations[cell]}' for cell in cells]
        if args.arm=='bge':
            run([sys.executable,'-u','run_defenses.py','--dataset','hotpotqa','--gpu',args.gpu,
                 '--runs',*runs,'--corpus',args.data_root/'hotpotqa_sqlite','--index',index,
                 '--queries',args.data_root/'downloads/hotpotqa/queries.jsonl','--benchmark',args.benchmark,
                 '--detectors-only','--out',out])
        else:
            run([sys.executable,'-u','-m','aligned_rag.defense_suite','detectors','--runs',*runs,
                 '--score-source',args.runs_root/source_def,'--out',out,'--legacy-diagnostics'])
    if not args.execute:print('Plan only. Add --execute to run. No GPU work has been performed.')


if __name__=='__main__':main()
