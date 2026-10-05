"""Execute released defense methods with explicit common-victim interface adaptations."""
import ast
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
from collections import Counter, defaultdict
import numpy as np
from exp11_ablation import fingerprint
from research_integrity import file_sha256

PINS={'RobustRAG':'9bc35b2fa5fa7d1088383b2789ec19a512d316b1',
      'TrustRAG':'11dcea0262d14b0e38e22e1d7ddce4a151e82a61'}

def source_info(root):
    import subprocess
    root=Path(root); info={}
    for name,pin in PINS.items():
        repo=root/name
        rev=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        if rev!=pin:raise ValueError(f'{name}: expected pinned revision {pin}, got {rev}')
        if subprocess.check_output(['git','-C',str(repo),'diff','HEAD','--','*.py'],text=True).strip():
            raise ValueError(f'{name}: tracked Python source is modified; use the pinned source')
        info[name]={'revision':rev,'files':{str(p.relative_to(repo)):file_sha256(p) for p in sorted(repo.rglob('*.py')) if '.git' not in p.parts}}
    return info

def selected_source(path, names, namespace):
    """Compile original AST nodes verbatim; avoid unrelated top-level model/private imports."""
    tree=ast.parse(Path(path).read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError(f'Missing upstream definitions: {names}')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

class TraceLLM:
    def __init__(self, generator):self.generator=generator;self.trace=[]
    def query(self, prompt, limit=150, first_line=False):
        previous=self.generator.max_new; self.generator.max_new=limit
        try:answer=self.generator.generate([prompt])[0]
        finally:self.generator.max_new=previous
        result=answer.strip().split('\n')[0].strip() if first_line else answer
        self.trace.append({'prompt':prompt,'raw_output':answer,'output':result,'max_new_tokens':limit,'first_line_only':first_line})
        return result
    def __call__(self, prompts, sampling_params=None):
        return [SimpleNamespace(text=self.query(p,limit=4096)) for p in prompts]

class RobustAdapter:
    def __init__(self, traced, templates):
        self.traced=traced;self.tmpl=templates;self.model_name='llama'
        self.aggregation_prompts=set()
    def wrap_prompt(self, item, as_multi_choice=False, hints=None, seperate=False):
        q=item['question']; docs=item['topk_content']
        if seperate:return [self.tmpl['qa'].format(context_str=d,query_str=q) for d in docs]
        if hints is not None:
            prompt=self.tmpl['qa-hint'].format(hints=hints,query_str=q)
            self.aggregation_prompts.add(prompt)
            return prompt
        return self.tmpl['qa'].format(context_str='\n'.join(docs),query_str=q)
    def query(self,prompt):
        return self.traced.query(prompt,limit=150,first_line=prompt not in self.aggregation_prompts)
    def batch_query(self,prompts):return [self.query(p) for p in prompts]

class Pipelines:
    def __init__(self, generator, baseline_root, method):
        self.method=method; self.trace=TraceLLM(generator);root=Path(baseline_root)
        if method in ('robustrag_keyword','robustrag_prompt_control'):
            import spacy
            from nltk.corpus import stopwords
            ns={'spacy':spacy,'stopword_set':set(stopwords.words('english')),
                'Counter':Counter,'defaultdict':defaultdict,'logger':logging.getLogger('RRAG'),
                'punctuation':'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~','INJECTION':True}
            selected_source(root/'RobustRAG/src/defense.py',{'RRAG','KeywordAgg'},ns)
            spec=importlib.util.spec_from_file_location('rr_templates_aligned',root/'RobustRAG/src/prompt_template.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.adapter=RobustAdapter(self.trace,module.LLAMA_TMPL)
            self.rr=ns['KeywordAgg'](self.adapter,relative_threshold=.3,absolute_threshold=3,abstention_threshold=1)
        elif method=='trustrag_filter_conflict':
            ns={}
            selected_source(root/'TrustRAG/defend_module.py',{'conflict_query'},ns)
            self.conflict=ns['conflict_query']
        else:raise ValueError(method)
    def run(self, query, docs):
        self.trace.trace=[]
        if self.method.startswith('robustrag'):
            item={'question':query,'topk_content':docs}
            if self.method=='robustrag_prompt_control':answer=self.adapter.query(self.adapter.wrap_prompt(item))
            else:answer,_=self.rr.query(item,corruption_size=0)
            details={}
        else:
            outputs,internal,consolidated=self.conflict([docs],[query],self.trace,None)
            answer=outputs[0];details={'internal_knowledge':internal[0],'consolidated_evidence':consolidated[0]}
        return {'output':answer,'trace':self.trace.trace,**details}

class TrustFilter:
    def __init__(self, baseline_root, device='cuda'):
        from transformers import AutoTokenizer, AutoModel
        import torch
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics.pairwise import cosine_similarity
        from rouge_score import rouge_scorer
        self.torch=torch;self.device=device
        model='princeton-nlp/sup-simcse-bert-base-uncased'
        self.tok=AutoTokenizer.from_pretrained(model)
        self.model=AutoModel.from_pretrained(model).to(device).eval()
        ns={'np':np,'KMeans':KMeans,'StandardScaler':StandardScaler,
            'cosine_similarity':cosine_similarity,'rouge_scorer':rouge_scorer}
        selected_source(Path(baseline_root)/'TrustRAG/defend_module.py',
            {'calculate_similarity','calculate_pairwise_rouge','calculate_average_score','group_n_gram_filtering','k_mean_filtering'},ns)
        self.filter=ns['k_mean_filtering']
        self.config={'embedding_model':model,'resolved_revision':getattr(self.model.config,'_commit_hash',None),
                     'pooling':'last_hidden_state_CLS','truncation':self.tok.model_max_length,
                     'method':'kmeans_ngram','source':file_sha256(Path(baseline_root)/'TrustRAG/defend_module.py')}
    def run(self, docs):
        if len(docs)<2:return {'kept_texts':docs,'unchanged':True,'degenerate_embedding_guard':False}
        embeddings=[]
        with self.torch.inference_mode():
            for text in docs:
                x=self.tok(text,return_tensors='pt',truncation=True,padding=True).to(self.device)
                embeddings.append(self.model(**x,output_hidden_states=True,return_dict=True).hidden_states[-1][0,0].float().cpu().numpy())
        embeddings=np.asarray(embeddings)
        from sklearn.preprocessing import StandardScaler
        scaled=StandardScaler().fit_transform(embeddings)
        if np.any(np.linalg.norm(scaled,axis=1)==0):
            raise ValueError('TrustRAG degenerate standardized embedding: upstream normalization is undefined; inspect this case rather than silently treating it as filtered/unchanged')
        _,kept=self.filter(embeddings,docs,[],True) # adv_text_set is unused in the released filtering function.
        return {'kept_texts':list(kept),'unchanged':list(kept)==docs,'degenerate_embedding_guard':False}
