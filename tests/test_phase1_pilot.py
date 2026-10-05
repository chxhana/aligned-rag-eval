import ast
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
from exp11_ablation import fingerprint, save_json
from research_integrity import auroc_ties, select_utility, benchmark_qa
from aligned_rag.data import import_corpus, Corpus, released_poisonedrag, validate_manifest
from aligned_rag.models import top_indices
from aligned_rag.protocol import make_conditions, retrieve_manifest, generate_answers, official_prompt
from aligned_rag.analysis import evaluate, export_review, compare_review
from build_split_knowledge_attacks import craft_one, asserts


class FakeRetriever:
    def __init__(self):
        self.corpus = types.SimpleNamespace(meta={'n': 1000000, 'test_fixture': True})
        self.meta = {'mock': True}
    def search(self, queries, k):
        docs = [{'id':'a', 'title':'', 'text':'SUPPORT', 'score':2., 'injected':False,'owner':None},
                {'id':'h', 'title':'', 'text':'BACKGROUND', 'score':1.,'injected':False,'owner':None}]
        return [[dict(x) for x in docs][:k] for _ in queries], np.ones((len(queries),1))
    def score_poisons(self, docs):
        return np.full((len(docs),1), 3.)


class FakeGenerator:
    config = {'mock':True}
    def generate(self, prompts):
        return ['Target' if 'POISON' in p and 'SUPPORT' in p else 'Unknown' for p in prompts]


def manifest():
    return {'schema_version':1, 'dataset':'fixture', 'attack':{'name':'mock'}, 'records':[
        {'id':'q1','query':'Question','target':'Target','references':['Gold'],
         'poisons':['POISON'], 'support_ids':['a'], 'support_semantics':'designated_split_A',
         'group_id':'g', 'reference_source':'test_fixture'}]}


class IntegrityTests(unittest.TestCase):
    def test_auc_ties_and_order(self):
        self.assertEqual(auroc_ties([1,1],[1,1]),.5)
        self.assertEqual(auroc_ties([2,1],[1,0]),.875)
        self.assertEqual(auroc_ties([1,2],[0,1]),.875)
        self.assertEqual(auroc_ties([0],[1]),0)
        with self.assertRaises(ValueError):auroc_ties([float('nan')],[1])
    def test_deterministic_topk(self):
        self.assertEqual(top_indices(np.array([1.,3.,3.,2.,3.]),2).tolist(),[1,2])
    def test_utility_disjoint(self):
        row=lambda q:{'query':q,'answer':'A'}
        self.assertEqual(select_utility([row('A'),row(' B '),row('C')],[row('a')],[row('b')],1),[row('C')])
        with self.assertRaises(ValueError):select_utility([row('A')],[row('a')],[],1)
    def test_em_f1_does_not_count_negation_as_exact(self):
        self.assertEqual(benchmark_qa('The cat.', 'cat')['exact_match'],1)
        self.assertEqual(benchmark_qa('not Canada', 'Canada')['exact_match'],0)


class CorpusTests(unittest.TestCase):
    def test_import_get_and_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'in.jsonl';p.write_text(json.dumps({'_id':'a','title':'T','text':'A'})+'\n')
            dest=Path(d)/'db';import_corpus(p,dest)
            c=Corpus(dest);self.assertEqual(c.get(0)['id'],'a');c.conn.close()
            import sqlite3
            con=sqlite3.connect(dest/'corpus.sqlite');con.execute("UPDATE docs SET text='changed'");con.commit();con.close()
            with self.assertRaises(ValueError):Corpus(dest)
    def test_released_adapter_matches_upstream_method(self):
        repo=Path('baselines/PoisonedRAG')
        data=released_poisonedrag(repo)
        self.assertEqual(len(data['records']),100)
        tree=ast.parse((repo/'src/attack.py').read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Attacker')
        method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='get_attack')
        scope={};exec(compile(ast.Module(body=[method],type_ignores=[]),'upstream_get_attack','exec'),scope)
        original=json.loads((repo/'results/adv_targeted_results/hotpotqa.json').read_text())
        obj=types.SimpleNamespace(attack_method='LM_targeted',adv_per_query=5,all_adv_texts=original)
        queries=[{'query':r['query'],'id':r['id']} for r in data['records']]
        actual=scope['get_attack'](obj,queries)
        self.assertEqual(actual,[r['poisons'] for r in data['records']])
        self.assertEqual(len({r['group_id'] for r in data['records']}),10)
        self.assertTrue(all(r['reference_source']=='released_model_generated' for r in data['records']))
        self.assertIn('[question]',official_prompt(repo))


class ProtocolTests(unittest.TestCase):
    def test_controls_own_and_foreign(self):
        r=manifest()['records'][0]
        docs=[{'id':'b','text':'P','injected':True,'owner':'q1'},
              {'id':'c','text':'F','injected':True,'owner':'q2'},
              {'id':'a','text':'SUPPORT','injected':False,'owner':None}]
        cond,roles=make_conditions(r,[],docs,docs[:2])
        self.assertEqual([d['id'] for d in cond['no_poison']],['a'])
        self.assertEqual([d['id'] for d in cond['no_own_poison']],['c','a'])
        self.assertEqual(cond['background_only'],[])
    def test_end_to_end_review_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            contexts=retrieve_manifest(manifest(),FakeRetriever(),Path(d)/'retr',top_k=2)
            data=generate_answers(contexts,FakeGenerator(),Path(d)/'gen')
            again=generate_answers(contexts,FakeGenerator(),Path(d)/'gen')
            self.assertEqual(data,again)
            report=evaluate(data)
            self.assertEqual(report['metrics']['strict_newly_induced']['successes'],1)
            packet=export_review(data,Path(d)/'audit',all_queries=True)
            for j in packet['judgments'].values():
                yes=j['answer']=='Target'
                j.update(target_endorsed='yes' if yes else 'no',reference_correct='no',
                         target_evidence='Target' if yes else '',reference_evidence='',
                         rationale='Fixture truth, not human experimental evidence.',reviewer='test fixture')
            corrected=compare_review(data,packet)
            self.assertEqual(corrected['adjudicated_subset']['metrics']['strict_newly_induced']['successes'],1)
            bad=copy.deepcopy(packet);next(iter(bad['judgments'].values()))['answer']='tampered'
            with self.assertRaises(ValueError):compare_review(data,bad)
            bad=copy.deepcopy(packet);next(iter(bad['judgments'].values()))['target_endorsed']='uncertain'
            with self.assertRaises(ValueError):compare_review(data,bad)
    def test_ordinary_has_no_forced_composition(self):
        with tempfile.TemporaryDirectory() as d:
            m=manifest();m['records'][0]['support_ids']=[]
            m['records'][0]['support_semantics']='none'
            contexts=retrieve_manifest(m,FakeRetriever(),Path(d)/'r',top_k=2)
            data=generate_answers(contexts,FakeGenerator(),Path(d)/'g')
            self.assertEqual(evaluate(data)['metrics']['strict_newly_induced']['n'],0)
    def test_preserve_failed_construction(self):
        with tempfile.TemporaryDirectory() as d:
            m=manifest();m['records'][0]['poisons']=[]
            contexts=retrieve_manifest(m,FakeRetriever(),Path(d)/'r',top_k=2)
            data=generate_answers(contexts,FakeGenerator(),Path(d)/'g')
            self.assertEqual(evaluate(data)['denominator']['evaluated_queries'],1)
            self.assertEqual(evaluate(data)['metrics']['injection_dependent_new']['successes'],0)


class BuilderTests(unittest.TestCase):
    class Gen:
        def _chat(self,system,user,max_new_tokens):
            return json.dumps({'doc_b':['One.','Two.','Three.']})
    class Victim:
        def __init__(self):self.seen=[]
        def answer(self,q,docs):
            self.seen.append(list(docs))
            return 'Target' if len(docs)==6 else 'not Target'
        def judge(self,q,out,target):return out=='Target'
    def test_final_anchor_verified_and_denial_not_shortcut(self):
        args=types.SimpleNamespace(max_trials=1,verification_stage='post-anchor',anchor='query',judge=True)
        victim=self.Victim();item={'query':'Anchor query','context':['A.','B.','C.'],'answer':'Gold'}
        result,trace=craft_one(self.Gen(),victim,item,'Target',args)
        self.assertIsNotNone(result)
        self.assertTrue(all(any('Anchor query' in t for t in docs) for docs in victim.seen))
        self.assertEqual(result['final_context_sha256'],result['verified_context_sha256'])
        self.assertEqual(result['unanchored_doc_b'],['One.','Two.','Three.'])
    def test_pre_anchor_arm_retained(self):
        args=types.SimpleNamespace(max_trials=1,verification_stage='pre-anchor',anchor='query',judge=True)
        victim=self.Victim();item={'query':'Anchor query','context':['A.','B.','C.'],'answer':'Gold'}
        result,_=craft_one(self.Gen(),victim,item,'Target',args)
        self.assertNotEqual(result['final_context_sha256'],result['verified_context_sha256'])
        self.assertTrue(all(all('Anchor query' not in t for t in docs) for docs in victim.seen))

if __name__=='__main__':unittest.main()
