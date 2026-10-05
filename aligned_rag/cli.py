"""Phase-one pilot commands. Heavy libraries are imported only by GPU commands."""
import argparse
import json
from pathlib import Path

from exp11_ablation import save_json
from .data import read, released_poisonedrag, split_manifest, import_corpus, validate_manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    subs = ap.add_subparsers(dest='command', required=True)
    dl = subs.add_parser('download-beir')
    dl.add_argument('--dataset', choices=['hotpotqa', 'nq'], default='hotpotqa')
    dl.add_argument('--out-dir', required=True)
    dpr = subs.add_parser('download-dpr')
    dpr.add_argument('--out', required=True)
    imp = subs.add_parser('import-poisonedrag')
    imp.add_argument('--repo', default='baselines/PoisonedRAG')
    imp.add_argument('--dataset', default='hotpotqa')
    imp.add_argument('--adv-per-query', type=int, default=5)
    imp.add_argument('--group-size', type=int, default=10)
    imp.add_argument('--qrels')
    imp.add_argument('--references')
    imp.add_argument('--out', required=True)
    split = subs.add_parser('import-split')
    split.add_argument('--attacks', required=True)
    split.add_argument('--dataset', required=True)
    split.add_argument('--support-map')
    split.add_argument('--out', required=True)
    corp = subs.add_parser('import-corpus')
    corp.add_argument('--input', required=True)
    corp.add_argument('--format', choices=['beir-jsonl', 'dpr-tsv'], default='beir-jsonl')
    corp.add_argument('--limit', type=int, default=0, help='0=full corpus; nonzero is explicitly labeled a prefix subset')
    corp.add_argument('--out-dir', required=True)
    for name in ('encode', 'retrieve'):
        p = subs.add_parser(name)
        p.add_argument('--corpus', required=True)
        p.add_argument('--index', required=True)
        p.add_argument('--encoder', default='facebook/contriever')
        p.add_argument('--encoder-revision')
        p.add_argument('--score', choices=['dot', 'cosine'], default='dot')
        p.add_argument('--max-length', type=int, default=128)
        p.add_argument('--device', default='cuda')
        p.add_argument('--batch-size', type=int, default=128 if name=='encode' else 16)
        if name == 'encode':
            p.add_argument('--text-only', action='store_true')
        else:
            p.add_argument('--attacks', required=True)
            p.add_argument('--top-k', type=int, default=5)
            p.add_argument('--block-size', type=int, default=50000)
            p.add_argument('--min-passages', type=int, default=1000000)
            p.add_argument('--out-dir', required=True)
    gen = subs.add_parser('generate')
    gen.add_argument('--retrieval', required=True)
    gen.add_argument('--victim', default='meta-llama/Meta-Llama-3-8B-Instruct')
    gen.add_argument('--revision')
    gen.add_argument('--backend', choices=['hf', 'vllm'], default='hf')
    gen.add_argument('--tensor-parallel', type=int, default=1)
    gen.add_argument('--batch-size', type=int, default=8)
    gen.add_argument('--max-new-tokens', type=int, default=150)
    gen.add_argument('--max-input-tokens', type=int, default=8192)
    gen.add_argument('--poisonedrag-prompt-repo', help='use released prompt_id=4 text; tokenizer chat wrapper remains victim-specific')
    gen.add_argument('--out-dir', required=True)
    report = subs.add_parser('report')
    report.add_argument('--answers', required=True)
    report.add_argument('--unified-scoring', action='store_true', help='Rescore saved answers; include raw scores and heuristic sensitivity without generation')
    report.add_argument('--out', required=True)
    audit = subs.add_parser('audit-export')
    audit.add_argument('--answers', required=True)
    audit.add_argument('--queries-per-stratum', type=int, default=15)
    audit.add_argument('--seed', type=int, default=42)
    audit.add_argument('--all', action='store_true')
    audit.add_argument('--out-dir', required=True)
    apply = subs.add_parser('audit-apply')
    apply.add_argument('--answers', required=True)
    apply.add_argument('--labels', required=True)
    apply.add_argument('--out', required=True)
    plot = subs.add_parser('plot')
    plot.add_argument('--report', required=True)
    plot.add_argument('--out-prefix', required=True)
    args = ap.parse_args()
    try:
        if args.command == 'download-beir':
            from .download import download_beir
            download_beir(args.dataset, args.out_dir)
        elif args.command == 'download-dpr':
            from .download import download_dpr
            download_dpr(args.out)
        elif args.command == 'import-poisonedrag':
            if Path(args.out).exists(): raise ValueError('output exists')
            data = released_poisonedrag(args.repo, args.dataset, args.adv_per_query, args.group_size,
                                       args.qrels, args.references)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            save_json(args.out, data)
            print(f'[import] {len(data["records"])} released cases; reference provenance retained')
        elif args.command == 'import-split':
            if Path(args.out).exists(): raise ValueError('output exists')
            data = split_manifest(args.attacks, args.dataset, args.support_map)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            save_json(args.out, data)
        elif args.command == 'import-corpus':
            if args.limit < 0: raise ValueError('limit must be nonnegative')
            import_corpus(args.input, args.out_dir, args.format, args.limit)
        elif args.command in ('encode', 'retrieve'):
            from .models import Encoder, ExactRetriever, encode_corpus
            if args.batch_size < 1 or args.max_length < 1: raise ValueError('positive batch and length required')
            encoder = Encoder(args.encoder, args.encoder_revision, args.device, args.score, args.max_length)
            if args.command == 'encode':
                encode_corpus(args.corpus, args.index, encoder, args.batch_size, not args.text_only)
            else:
                from .protocol import retrieve_manifest
                if args.block_size < 1: raise ValueError('positive block size required')
                retriever = ExactRetriever(args.corpus, args.index, encoder, args.block_size)
                retrieve_manifest(read(args.attacks), retriever, args.out_dir,
                                  args.top_k, args.min_passages, args.batch_size)
        elif args.command == 'generate':
            from .models import Generator
            from .protocol import generate_answers, official_prompt, DEFAULT_PROMPT
            generator = Generator(args.victim, args.revision, args.backend, args.tensor_parallel,
                                  args.max_new_tokens, args.max_input_tokens)
            template = official_prompt(args.poisonedrag_prompt_repo) if args.poisonedrag_prompt_repo else DEFAULT_PROMPT
            generate_answers(read(args.retrieval), generator, args.out_dir, template, args.batch_size)
        elif args.command in ('report', 'audit-apply'):
            from .analysis import evaluate, compare_review, rescore_report
            if Path(args.out).exists(): raise ValueError('report exists; choose a new filename')
            data = read(args.answers)
            result = (rescore_report(data) if args.unified_scoring else evaluate(data)) if args.command == 'report' else compare_review(data, read(args.labels))
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            save_json(args.out, result)
            if args.command == 'report':
                for key, r in result['metrics'].items():
                    print(f'{key:45} {r["successes"]}/{r["n"]} {r["rate"]}')
            if args.command == 'report' and args.unified_scoring:
                print('[reference: raw vs extracted EM/F1]', json.dumps(result['reference_summary']))
                print('[sensitivity] negation screen is heuristic, not endorsement')
            print(f'[out] {args.out}')
        elif args.command == 'audit-export':
            from .analysis import export_review
            if args.queries_per_stratum < 1: raise ValueError('positive strata sizes required')
            export_review(read(args.answers), args.out_dir, args.queries_per_stratum, args.seed, args.all)
        elif args.command == 'plot':
            from .figures import plot_report
            plot_report(read(args.report), args.out_prefix)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        ap.exit(2, f'[error] {exc}\n')


if __name__ == '__main__':
    main()
