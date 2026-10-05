import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from aligned_rag.defense_metrics import threshold,interception,pipeline_effect,mechanism,hotpot_qa
from aligned_rag.defense_suite import checkpoint,persist,checked,select_holdout,report,context_jobs,generation_jobs
from aligned_rag.defense_backends import selected_source,TraceLLM,RobustAdapter
from exp11_ablation import fingerprint
from aligned_rag.protocol import answer_id,legacy_labels

class DefenseTests(unittest.TestCase):
    def test_tied_threshold_does_not_inflate_fpr(self):
        vals=[1.]*99+[2.]
        t=threshold(vals,.01)
        self.assertEqual(sum(v>t for v in vals),1)
        self.assertEqual(threshold([1.]*100,.01),1)
        with self.assertRaises(ValueError):threshold([float('nan')],.01)
    def test_interception_denominators(self):
        x=interception([True,True,False],[True,False,False],[False,True,True])
        self.assertEqual(x['successful_attack_interception']['rate'],.5)
        self.assertEqual(x['newly_induced_attack_interception']['rate'],1.)
        self.assertEqual(x['target_incidence_after_hard_abstention']['count'],1)
        self.assertIsNone(interception([False],[False],[True])['successful_attack_interception']['rate'])
    def test_pipeline_new_failures_offset_prevention(self):
        x=pipeline_effect([True,False],[False,True],[False,False],[False,True],['g','g'])
        self.assertEqual(x['successful_attacks_prevented']['rate'],1.)
        self.assertEqual(x['absolute_target_reduction'],0.)
        self.assertEqual(x['new_target_vs_defended_clean']['count'],0)
        self.assertEqual(x['new_target_vs_original_clean']['count'],1)
    def test_mechanisms_any_vs_all(self):
        r={'support_ids':['a','b'],'conditions':{'poisoned':[{'id':'p','injected':True},{'id':'a','injected':False}]}}
        self.assertEqual(mechanism(r),'some_support_present')
        r['conditions']['poisoned'].append({'id':'b','injected':False})
        self.assertEqual(mechanism(r),'all_annotated_supports_present')
        r['conditions']['poisoned']=r['conditions']['poisoned'][:1]
        self.assertEqual(mechanism(r),'all_retrieved_passages_injected')
    def test_hotpot_yes_no_f1_exception(self):
        self.assertEqual(hotpot_qa('Yes, both do.',['yes'])['token_f1'],0.)
        self.assertEqual(hotpot_qa('The United States.',['United States'])['exact_match'],1.)
    def test_cache_content_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cache.json'; d=checkpoint(path,{'model':'x'}); d['records']['k']=1;d['complete']=True;persist(path,d)
            self.assertEqual(checked(path)['records']['k'],1)
            with self.assertRaises(ValueError):checkpoint(path,{'model':'y'})
            d['records']['k']=2;path.write_text(json.dumps(d))
            with self.assertRaises(ValueError):checked(path)
    def test_holdout_disjoint_by_id_and_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); rows=[{'_id':str(i),'question':f'Q{i}','answer':'yes'} for i in range(5)]
            (p/'gold.json').write_text(json.dumps(rows))
            (p/'queries.jsonl').write_text('\n'.join(json.dumps({'_id':r['_id'],'text':r['question']}) for r in rows))
            runs={'a':{'data':{'retrieval':{'records':[{'id':'0','query':'Q0'}]}}}}
            selected=select_holdout(p/'gold.json',p/'queries.jsonl',runs,2,2)
            self.assertEqual({r['id'] for r in selected},{'1','2','3','4'})
            self.assertEqual(sum(r['split']=='utility' for r in selected),2)
    def test_full_trust_calls_three_stages_and_traces(self):
        ns=selected_source('baselines/TrustRAG/defend_module.py',{'conflict_query'},{})
        class Gen:
            max_new=150
            def generate(self,prompts):return ['Mock response' for _ in prompts]
        gen=Gen();llm=TraceLLM(gen)
        final,internal,consolidated=ns['conflict_query']([['Evidence']],['Question'],llm,None)
        self.assertEqual(len(llm.trace),3)
        self.assertEqual(gen.max_new,150)
        self.assertEqual(llm.trace[0]['max_new_tokens'],4096)
        self.assertIn('Externally Retrieved Document0:Evidence',llm.trace[1]['prompt'])
        self.assertIn('Mock response',llm.trace[2]['prompt'])
    def test_full_robust_query_uses_isolation_and_aggregation(self):
        from collections import defaultdict,Counter
        import logging
        ns={'spacy':SimpleNamespace(load=lambda name:lambda text:[]),'defaultdict':defaultdict,'Counter':Counter,
            'stopword_set':set(),'punctuation':'','logger':logging.getLogger('test')}
        selected_source('baselines/RobustRAG/src/defense.py',{'RRAG','KeywordAgg'},ns)
        class Gen:
            max_new=150
            def generate(self,prompts):return ['Canada' for _ in prompts]
        traced=TraceLLM(Gen());adapter=RobustAdapter(traced,{'qa':'{context_str} Q:{query_str}','qa-hint':'{hints} Q:{query_str}'})
        rr=ns['KeywordAgg'](adapter)
        result,_=rr.query({'question':'Where?','topk_content':['one','two']},corruption_size=0)
        self.assertEqual(result,'Canada');self.assertEqual(len(traced.trace),3)
        self.assertIn('Canada',traced.trace[-1]['prompt'])
    def test_report_end_to_end_with_saved_fixtures(self):
        from aligned_rag.defense_suite import run_config,METHODS
        from aligned_rag.defense_metrics import DETECTORS
        def doc(ident,text,injected=False):return {'id':ident,'title':'','text':text,'score':1.,'injected':injected,'owner':'q' if injected else None}
        clean=[doc('a','support'),doc('h','background')];poison=[doc('p','Canada',True),doc('a','support')]
        row={'id':'q','query':'Where?','target':'Canada','references':['USA'],'group_id':'g','support_ids':['a'],
             'conditions':{'clean':clean,'poisoned':poison}}
        data={'signature':'gen','answers_sha256':'raw','retrieval':{'signature':'ret','records':[row]},'answers':{}}
        for lane,out in [('clean','USA'),('poisoned','Canada')]:data['answers'][answer_id(row,row['conditions'][lane])]={'output':out}
        runs={'p1':{'path':'fixture','data':data}};rc=run_config(runs)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def artifact(name,records,extra=None):
                d=checkpoint(root/name,{'runs':rc,'fidelity':'mock fixture'});d['records']=records;d['complete']=True
                if extra:d.update(extra)
                persist(root/name,d);return d
            holdout=artifact('holdout.json',{'c':{'id':'c','query':'cal','references':['USA'],'split':'calibration','docs':clean},
                'u':{'id':'u','query':'util','references':['USA'],'split':'utility','docs':clean}})
            jobs,cl=context_jobs(runs,holdout);gj,gl=generation_jobs(runs,holdout)
            artifact('scores.json',{k:{d:float(k==cl['p1:q:poisoned']) for d in DETECTORS} for k in jobs})
            artifact('trust_filtered.json',{k:{'unchanged':True} for k in jobs})
            for m in (*METHODS,'vanilla'):artifact(m+'.json',{k:{'output':'USA'} for k in gj})
            args=SimpleNamespace(runs=[],out=str(root),labels=None,export_labels=None,legacy_diagnostics=True,report_out=None)
            with patch('aligned_rag.defense_suite.load_runs',return_value=runs):report(args)
            result=json.loads((root/'defense_report.json').read_text())
            self.assertEqual(result['cells']['p1']['detectors']['rc_gap']['sweep'][0]['successful_attack_interception']['rate'],1.)
            self.assertEqual(result['cells']['p1']['pipelines']['trustrag_filter_conflict']['successful_attacks_prevented']['rate'],1.)
            self.assertEqual(result['clean_utility']['vanilla']['exact_match'],1.)
