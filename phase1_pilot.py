"""Plan or execute a PoisonedRAG pilot or encoding-control run. Dry-run is the default."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from exp11_ablation import save_json
from research_integrity import file_sha256

PIN = 'f660d72174f06b13fae5163ce656e7b235db858f'
REPO = 'https://github.com/sleeepeer/PoisonedRAG.git'


def build_steps(args):
    data, run = Path(args.data_root), Path(args.run_root)
    py = [sys.executable, '-u', '-m', 'aligned_rag.cli']
    dataset = getattr(args, 'dataset', 'hotpotqa')
    score = getattr(args, 'score', 'dot')
    length = getattr(args, 'max_length', 128)
    budget = getattr(args, 'adv_per_query', 5)
    top_k = getattr(args, 'top_k', 5)
    corpus = Path(args.corpus) if getattr(args, 'corpus', None) else data / (dataset + '_sqlite')
    raw = data / 'downloads' / dataset
    index = Path(args.index) if getattr(args, 'index', None) else data / f'contriever_{score}{length}'
    encoding = ['--score', score, '--max-length', str(length)]
    manifest = run / 'attacks.json'
    def step(name, argv):return (name, list(map(str, argv)))
    steps = [step('download', py + ['download-beir','--dataset',dataset,'--out-dir',data/'downloads']),
             step('corpus', py + ['import-corpus','--input',raw/'corpus.jsonl','--out-dir',corpus]),
             step('manifest', py + ['import-poisonedrag','--repo',args.repo,'--dataset',dataset,
                                   '--adv-per-query',str(budget),'--group-size','10','--qrels',raw/'qrels/test.tsv',
                                   '--out',manifest] + (['--references',args.references] if args.references else [])),
             step('encode', py + ['encode','--corpus',corpus,'--index',index,'--batch-size',str(args.encode_batch)] + encoding),
             step('retrieve',py + ['retrieve','--corpus',corpus,'--index',index,'--attacks',manifest,
                                  '--top-k',str(top_k),'--out-dir',run/'retrieval'] + encoding),
             step('generate',py + ['generate','--retrieval',run/'retrieval/retrieval.json',
                                   '--victim',args.victim,'--backend',args.backend,
                                   '--tensor-parallel',str(args.tensor_parallel),
                                   '--batch-size',str(args.generation_batch),
                                   '--poisonedrag-prompt-repo',args.repo,'--out-dir',run/'generation'] +
                  (['--revision',args.victim_revision] if args.victim_revision else [])),
             step('report',py + ['report','--answers',run/'generation/answers.json','--out',run/'pilot_report.json'] + (['--unified-scoring'] if getattr(args,'unified_scoring',False) else [])),
             step('audit',py + ['audit-export','--answers',run/'generation/answers.json',
                               '--queries-per-stratum','15','--out-dir',run/'audit']),
             step('figure',py + ['plot','--report',run/'pilot_report.json','--out-prefix',run/'figures/outcomes'])]
    return steps


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',required=True,help='large local/scratch disk for full corpus and embeddings')
    ap.add_argument('--run-root',required=True,help='new pilot output directory; keep each setup isolated')
    ap.add_argument('--dataset',choices=['hotpotqa','nq'],default='hotpotqa')
    ap.add_argument('--corpus',help='Existing corpus directory; preserve original database')
    ap.add_argument('--index',help='Explicit index directory; use a fresh path for each encoding control')
    ap.add_argument('--score',choices=['dot','cosine'],default='dot')
    ap.add_argument('--max-length',type=int,choices=[128,512],default=128,help='Corpus limit; query/poison limits are 512. Use 512 for equal lengths.')
    ap.add_argument('--adv-per-query',type=int,choices=[1,5],default=5)
    ap.add_argument('--top-k',type=int,choices=[5,10],default=5)
    ap.add_argument('--unified-scoring',action='store_true')
    ap.add_argument('--repo',default='baselines/PoisonedRAG')
    ap.add_argument('--references',help='optional benchmark reference JSON; otherwise upstream model references stay marked')
    ap.add_argument('--victim',default='meta-llama/Meta-Llama-3-8B-Instruct')
    ap.add_argument('--victim-revision')
    ap.add_argument('--backend',choices=['hf','vllm'],default='hf')
    ap.add_argument('--tensor-parallel',type=int,default=1)
    ap.add_argument('--encode-batch',type=int,default=128)
    ap.add_argument('--generation-batch',type=int,default=8)
    ap.add_argument('--from-step',choices=['download','corpus','manifest','encode','retrieve','generate','report','audit','figure'],default='download')
    ap.add_argument('--through-step',choices=['download','corpus','manifest','encode','retrieve','generate','report','audit','figure'],default='figure')
    ap.add_argument('--execute',action='store_true',help='actually download, index, and run generation; otherwise print the plan')
    args=ap.parse_args()
    steps=build_steps(args)
    names=[name for name,_ in steps]
    start,end=names.index(args.from_step),names.index(args.through_step)
    if start>end:ap.error('from-step follows through-step')
    steps=steps[start:end+1]
    for name,command in steps:print(f'[{name}] {shlex.join(command)}',flush=True)
    if not args.execute:
        print('\nPlan only. Add --execute on the GPU server. Run directories preserve human reviews and source artifacts.')
        return
    if not Path(args.repo).exists():
        raise SystemExit(f'Clone the pinned upstream repo first:\ngit clone {REPO} {args.repo}\n'
                         f'git -C {args.repo} checkout {PIN}')
    revision=subprocess.check_output(['git','-C',args.repo,'rev-parse','HEAD'],text=True).strip()
    if revision!=PIN:raise SystemExit(f'Expected upstream revision {PIN}; found {revision}. No repository changes made.')
    root=Path(args.run_root);root.mkdir(parents=True,exist_ok=True)
    plan={'steps':steps,'args':vars(args),'upstream_revision':revision,
          'python':sys.version,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
          'code':{str(p):file_sha256(p) for p in sorted(Path('aligned_rag').glob('*.py'))}}
    import time
    stamp=str(time.time_ns())
    save_json(root/f'invocation_{stamp}.json',plan)
    for name,command in steps:
        log=root/f'{name}_{stamp}.log'
        print(f'[{name}] running; log {log}',flush=True)
        with log.open('w') as f:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                     env={**os.environ,'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false'})
            for line in process.stdout:
                print(line,end='',flush=True);f.write(line);f.flush()
            status=process.wait()
        if status:raise SystemExit(f'{name} failed ({status}); remaining steps not run. Inspect {log}')
    print('[done] pilot artifacts generated. Legacy labels remain diagnostic until adjudication.')

if __name__=='__main__':main()
