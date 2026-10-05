"""Lazy GPU backends. Importing the evaluation schema never loads a model."""
from pathlib import Path
import numpy as np
import os
from importlib.metadata import version, PackageNotFoundError

def environment_versions():
    result = {}
    for name in ("numpy", "torch", "transformers", "accelerate", "vllm"):
        try: result[name] = version(name)
        except PackageNotFoundError: result[name] = None
    return result


from exp11_ablation import fingerprint, save_json
from research_integrity import file_sha256
from .data import Corpus, read


class Encoder:
    def __init__(self, name, revision=None, device='cuda', score='dot', max_length=128):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        self.torch, self.device, self.score = torch, device, score
        self.name, self.max_length = name, max_length
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModel.from_pretrained(name, revision=revision).to(device).eval()
        self.is_bge = 'bge' in name.lower()
        self.config = {'name': name, 'requested_revision': revision,
                       'resolved_revision': getattr(self.model.config, '_commit_hash', None),
                       'score': score, 'max_length': max_length, 'query_max_length': 512, 'poison_max_length': 512,
                       'pooling': 'cls' if self.is_bge else 'masked_mean',
                       'implementation': file_sha256(__file__), 'environment': environment_versions()}

    def encode(self, texts, query=False, max_length=None):
        torch = self.torch
        if query and self.is_bge:
            texts = ['Represent this sentence for searching relevant passages: ' + t for t in texts]
        x = self.tok(texts, padding=True, truncation=True, max_length=max_length or (512 if query else self.max_length),
                     return_tensors='pt').to(self.device)
        with torch.inference_mode():
            hidden = self.model(**x).last_hidden_state
            if self.is_bge:
                emb = hidden[:, 0]
            else:
                mask = x['attention_mask'].unsqueeze(-1)
                emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            if self.score == 'cosine':
                emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.float().cpu().numpy()


def encode_corpus(corpus_path, index_path, encoder, batch_size=128, include_title=True):
    corpus = Corpus(corpus_path)
    dest = Path(index_path)
    dest.mkdir(parents=True, exist_ok=True)
    config = {'corpus': corpus.meta, 'encoder': encoder.config, 'include_title': include_title}
    signature = fingerprint(config)
    meta_path = dest / 'index.json'
    dim = encoder.model.config.hidden_size
    n = corpus.meta['n']
    start = 0
    if meta_path.exists():
        meta = read(meta_path)
        if meta['signature'] != signature:
            raise ValueError('index provenance differs; use a new index directory')
        start = meta['encoded_rows']
        if meta['complete']:
            print('[encode] complete matching index exists')
            return
    elif (dest / 'embeddings.npy').exists():
        raise ValueError('orphan embeddings without provenance; choose a new directory')
    import shutil
    if not start and shutil.disk_usage(dest).free < n * dim * 4:
        raise ValueError(f'index needs at least {n * dim * 4 / 1e9:.1f} GB free')
    print(f'[index] {n} passages x {dim} float32 dimensions = {n * dim * 4 / 1e9:.1f} GB', flush=True)
    embeddings = np.lib.format.open_memmap(dest / 'embeddings.npy', mode='r+' if start else 'w+',
                                          dtype='float32', shape=(n, dim))
    if not meta_path.exists():
        save_json(meta_path, {'signature': signature, 'config': config, 'shape': [n, dim],
                             'encoded_rows': 0, 'complete': False})
    for batch in corpus.batches(start, batch_size):
        texts = [((r['title'] + ' ' + r['text']).strip() if include_title else r['text']) for r in batch]
        values = encoder.encode(texts)
        embeddings[start:start + len(batch)] = values
        start += len(batch)
        embeddings.flush()
        save_json(meta_path, {'signature': signature, 'config': config, 'shape': [n, dim],
                             'encoded_rows': start, 'complete': start == n})
        print(f'[encode] {start}/{n}', flush=True)


def top_indices(scores, k):
    """Stable descending score, with index order breaking ties."""
    k = min(k, len(scores))
    if k == 0:
        return np.array([], dtype=int)
    if k < len(scores):
        cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
        greater = np.flatnonzero(scores > cutoff)
        equal = np.flatnonzero(scores == cutoff)[:k - len(greater)]
        candidates = np.concatenate([greater, equal])
    else:
        candidates = np.arange(len(scores))
    return candidates[np.lexsort((candidates, -scores[candidates]))]


class ExactRetriever:
    """Memory-mapped exact search; bounded GPU blocks, no full-corpus GPU allocation."""
    def __init__(self, corpus_path, index_path, encoder, block_size=50000):
        self.corpus = Corpus(corpus_path)
        self.meta = read(Path(index_path) / 'index.json')
        if not self.meta['complete']:
            raise ValueError('index incomplete')
        config = self.meta['config']
        if config['corpus'] != self.corpus.meta or config['encoder'] != encoder.config:
            raise ValueError('corpus/encoder differs from index provenance')
        self.emb = np.load(Path(index_path) / 'embeddings.npy', mmap_mode='r')
        if list(self.emb.shape) != self.meta['shape']:
            raise ValueError('embedding shape mismatch')
        self.encoder, self.block = encoder, block_size
        self.torch = encoder.torch

    def search(self, queries, k):
        q = self.encoder.encode(queries, query=True)
        torch = self.torch
        tq = torch.as_tensor(q, device=self.encoder.device)
        best = [[] for _ in queries]
        for start in range(0, len(self.emb), self.block):
            values = np.array(self.emb[start:start + self.block], copy=True)
            with torch.inference_mode():
                scores = (torch.as_tensor(values, device=self.encoder.device) @ tq.T).cpu().numpy()
            for j in range(len(queries)):
                ids = top_indices(scores[:, j], k)
                best[j].extend((float(scores[i, j]), start + int(i)) for i in ids)
                best[j] = sorted(best[j], key=lambda x: (-x[0], x[1]))[:k]
        return [[{**self.corpus.get(i), 'score': score, 'injected': False, 'owner': None}
                 for score, i in row] for row in best], q

    def score_poisons(self, records):
        texts = [r['text'] for r in records]
        if not texts:
            return np.empty((0, self.emb.shape[1]), dtype='float32')
        return np.concatenate([self.encoder.encode(texts[i:i + 128], max_length=512) for i in range(0, len(texts), 128)])


class Generator:
    def __init__(self, model, revision=None, backend='hf', tensor_parallel=1,
                 max_new_tokens=150, max_input_tokens=8192, device='cuda'):
        import torch
        from transformers import AutoTokenizer
        if not torch.cuda.is_available():
            raise ValueError('GPU generation required; run unit tests for a CPU-only smoke test')
        self.backend, self.limit, self.max_new = backend, max_input_tokens, max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model, revision=revision, padding_side='left')
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        if backend == 'vllm':
            from vllm import LLM
            gpu_memory_utilization = float(os.environ.get('VLLM_GPU_MEMORY_UTILIZATION', '0.8'))
            self.model = LLM(model=model, revision=revision, tensor_parallel_size=tensor_parallel,
                             max_model_len=max_input_tokens + max_new_tokens,
                             gpu_memory_utilization=gpu_memory_utilization)
            resolved = getattr(self.model.llm_engine.model_config.hf_config, '_commit_hash', None)
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision,
                         torch_dtype=torch.bfloat16, device_map='auto').eval()
            resolved = getattr(self.model.config, '_commit_hash', None)
        self.config = {'model': model, 'requested_revision': revision, 'resolved_revision': resolved,
                       'backend': backend, 'tensor_parallel': tensor_parallel,
                       'max_new_tokens': max_new_tokens, 'max_input_tokens': max_input_tokens,
                       'decoding': 'greedy', 'implementation': file_sha256(__file__), 'environment': environment_versions()}

    def generate(self, prompts):
        import torch
        rendered = [self.tok.apply_chat_template([{'role': 'user', 'content': p}],
                    tokenize=False, add_generation_prompt=True) for p in prompts]
        tokens = self.tok(rendered, add_special_tokens=False)['input_ids']
        if any(len(t) > self.limit for t in tokens):
            raise ValueError('prompt exceeds configured input limit; no silent truncation allowed')
        if self.backend == 'vllm':
            from vllm import SamplingParams
            outputs = self.model.generate(rendered, SamplingParams(temperature=0, max_tokens=self.max_new),
                                          use_tqdm=False)
            return [o.outputs[0].text.strip() for o in outputs]
        inputs = self.tok(rendered, padding=True, return_tensors='pt', add_special_tokens=False).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new,
                                          pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(t[inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip() for t in outputs]
