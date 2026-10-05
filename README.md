# AlignedRAG-Eval ACL Supplement

This supplement contains the code used to build aligned RAG-poisoning evaluation records and the small audit artifacts reported in the paper.

The repository is intentionally lightweight.
It does not include benchmark corpora, model weights, external defense repositories, API keys, or full experiment dumps.
Please download those resources from their original maintainers and follow their licenses.

## What is included

- `aligned_rag/`: evaluation, scoring, defense-adapter, audit, and JSONL harness code.
- `examples/jsonl/`: a tiny four-condition example for checking the harness.
- `paper_artifacts/audit/`: the blinded review page and completed headline audit labels used for the paper's judge-validation analysis.
- `phase1_pilot.py`, `run_defenses.py`, and `run_generalization.py`: experiment runners for the larger PoisonedRAG, detector, and generalization runs.
- `requirements-phase1.txt` and `requirements-defense-suite.txt`: Python requirements for the main attack/retrieval path and defense path.

## Install

Use a fresh Python environment.
The main scripts were developed with Python 3.9 for the defense suite and a CUDA-enabled PyTorch environment for generation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install -r requirements-phase1.txt
python -m pip install -r requirements-defense-suite.txt
python -m spacy download en_core_web_sm
python - <<'PY'
import nltk
nltk.download('stopwords')
PY
```

Install PyTorch and vLLM separately for your CUDA driver and GPU stack.
For CPU-only checks of saved JSONL records, the editable install is enough.
For model generation and detector scoring, use a CUDA environment with `torch`, `transformers`, `accelerate`, `sentence-transformers`, and optionally `vllm`.

## External resources

Create a working directory for external data and baselines.
The paths below match the scripts, but you can choose different paths and pass them as command-line arguments.

```bash
mkdir -p data baselines
```

### PoisonedRAG

We thank the PoisonedRAG authors for releasing the attack code and artifacts.
The adapter expects the released repository layout.

```bash
git clone https://github.com/sleeepeer/PoisonedRAG.git baselines/PoisonedRAG
git -C baselines/PoisonedRAG checkout f660d72174f06b13fae5163ce656e7b235db858f
```

### RobustRAG

We thank the RobustRAG authors for releasing their defense code.
Our pipeline adapter uses the released KeywordAgg components under a common victim-model interface.

```bash
git clone https://github.com/inspire-group/RobustRAG.git baselines/RobustRAG
git -C baselines/RobustRAG checkout 9bc35b2fa5fa7d1088383b2789ec19a512d316b1
```

### TrustRAG

We thank the TrustRAG authors for releasing their defense code.
Our pipeline adapter uses the released filtering and conflict-resolution stages under the same saved-context interface.

```bash
git clone https://github.com/HuichiZhou/TrustRAG.git baselines/TrustRAG
git -C baselines/TrustRAG checkout 11dcea0262d14b0e38e22e1d7ddce4a151e82a61
```

### HotpotQA and BEIR corpora

We thank the HotpotQA and BEIR authors for the benchmark data.
The retrieval corpus is downloaded through BEIR.
The original HotpotQA fullwiki development answers can be downloaded from the HotpotQA distribution.

```bash
python -m aligned_rag.cli download-beir \
  --dataset hotpotqa \
  --out-dir data/downloads

curl -fL --retry 3 \
  'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json' \
  -o data/hotpot_dev_fullwiki_v1.json
```

Convert the BEIR corpus into the local SQLite format:

```bash
python -m aligned_rag.cli import-corpus \
  --input data/downloads/hotpotqa/corpus.jsonl \
  --format beir-jsonl \
  --out-dir data/hotpotqa_sqlite
```

### Natural Questions and DPR passages

We thank the Natural Questions, BEIR, and DPR authors for the benchmark and passage resources.
The NQ retrieval corpus and queries are downloaded through BEIR.
For DPR-style passage experiments, download the DPR Wikipedia passage file as well.
Use the NQ answer-alias file required by your reproduction setup as the `--benchmark` argument to `run_defenses.py`.

```bash
python -m aligned_rag.cli download-beir \
  --dataset nq \
  --out-dir data/nq_pilot/downloads

python -m aligned_rag.cli download-dpr \
  --out data/nq_pilot/psgs_w100.tsv.gz
```

Convert the NQ BEIR corpus if you are using the BEIR NQ corpus path:

```bash
python -m aligned_rag.cli import-corpus \
  --input data/nq_pilot/downloads/nq/corpus.jsonl \
  --format beir-jsonl \
  --out-dir data/nq_pilot/corpus_sqlite
```

## Minimal four-condition harness

This is the quickest way to check that the artifact emits the records named in the paper.
It does not require GPUs or external datasets.

```bash
aligned-rag-eval init-example --out examples/jsonl

aligned-rag-eval build-record \
  --clean-undefended examples/jsonl/clean_undefended.jsonl \
  --poisoned-undefended examples/jsonl/poisoned_undefended.jsonl \
  --clean-defended examples/jsonl/clean_defended.jsonl \
  --poisoned-defended examples/jsonl/poisoned_defended.jsonl \
  --out examples/outputs/record_example.json

aligned-rag-eval report \
  --records examples/outputs/record_example.json \
  --out examples/outputs

aligned-rag-eval audit-export \
  --records examples/outputs/record_example.json \
  --out examples/outputs/audit_packet_example.json
```

The report step writes:

- `summary.json`
- `p_j_n_h.csv`
- `transitions.csv`

The audit step writes a small review packet that can be labeled or checked by another annotator.

Input JSONL rows should contain at least:

```json
{"query_id": "...", "question": "...", "answer": "..."}
```

Recommended fields are:

```json
{"group_id": "...", "target": "...", "references": ["..."], "reference_source": "..."}
```

## Audit artifacts

`paper_artifacts/audit/review.html` is the static review interface shown to the annotator.
`paper_artifacts/audit/completed_headline_audit.json` contains the completed labels for the paper's targeted 100-output audit sample.
The sample deliberately oversamples contested transitions and judge-uncertain cases, so it validates label behavior rather than estimating population-level judge accuracy.

## Acknowledgments

This artifact builds on released resources from PoisonedRAG, RobustRAG, TrustRAG, HotpotQA, Natural Questions, BEIR, DPR, Hugging Face models, vLLM, Contriever, and BGE.
We thank the authors and maintainers of these resources for making reproducible evaluation possible.
Please cite the original papers and follow the licenses for each external dataset, model, and codebase when using this supplement.
