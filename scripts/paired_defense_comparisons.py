#!/usr/bin/env python3
import argparse, csv, json, random
from pathlib import Path
from math import comb

METHOD_LABEL={
    'robustrag_keyword':'KeywordAgg',
    'robustrag_prompt_control':'Prompt',
    'trustrag_filter_conflict':'TrustRAG',
}
PAIRS=[('trustrag_filter_conflict','robustrag_keyword'),('trustrag_filter_conflict','robustrag_prompt_control'),('robustrag_keyword','robustrag_prompt_control')]

def binom_two_sided(k,n,p=0.5):
    if n==0: return 1.0
    probs=[comb(n,i)*(p**i)*((1-p)**(n-i)) for i in range(n+1)]
    pk=probs[k]
    return min(1.0, sum(x for x in probs if x <= pk + 1e-15))

def quantile(xs,q):
    xs=sorted(xs)
    if not xs: return None
    pos=(len(xs)-1)*q
    lo=int(pos); hi=min(lo+1,len(xs)-1)
    frac=pos-lo
    return xs[lo]*(1-frac)+xs[hi]*frac

def load_setting(records_path, judge_path):
    d=json.load(open(records_path))
    j=json.load(open(judge_path))['judgments']
    rows=[]
    for rec in d['records']:
        m=rec['method']
        if m not in METHOD_LABEL: continue
        out_id=rec['conditions']['poisoned_defended']
        lab=j.get(out_id,{}).get('labels',{}).get('target_endorsed')
        if lab not in ('yes','no'):
            val=None
        else:
            val=1 if lab=='yes' else 0
        rows.append({'dataset':d['setting_id'].split('_')[0], 'setting_id':d['setting_id'], 'cell':rec['cell'], 'qid':rec['id'], 'group':rec['group_id'], 'method':m, 'value':val})
    return d['setting_id'], rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--records', nargs='+', required=True)
    ap.add_argument('--judge', required=True)
    ap.add_argument('--out-dir', default='factorial/inference_v1')
    ap.add_argument('--draws', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    all_rows=[]
    for rp in args.records:
        _, rows=load_setting(rp,args.judge)
        all_rows.extend(rows)
    by={}
    groups={}
    for r in all_rows:
        ds=r['setting_id']; cell=r['cell']; q=r['qid']
        by.setdefault((ds,cell,q),{})[r['method']]=r
        groups.setdefault((ds,cell),{}).setdefault(r['group'],[]).append(q)
    rng=random.Random(args.seed)
    results=[]
    for (ds,cell), gmap in sorted(groups.items()):
        qids=sorted({q for qs in gmap.values() for q in qs})
        group_ids=sorted(gmap)
        gq={g:sorted(set(qs)) for g,qs in gmap.items()}
        for a,b in PAIRS:
            paired=[]
            for q in qids:
                recs=by.get((ds,cell,q),{})
                if a in recs and b in recs and recs[a]['value'] is not None and recs[b]['value'] is not None:
                    paired.append((q,recs[a]['group'],recs[a]['value'],recs[b]['value']))
            if not paired: continue
            n=len(paired)
            aval=sum(x[2] for x in paired)/n
            bval=sum(x[3] for x in paired)/n
            diff=(aval-bval)*100
            a_only=sum(1 for _,_,av,bv in paired if av==1 and bv==0)
            b_only=sum(1 for _,_,av,bv in paired if av==0 and bv==1)
            pval=binom_two_sided(min(a_only,b_only), a_only+b_only) if (a_only+b_only)>0 else 1.0
            boots=[]
            for _ in range(args.draws):
                sample_groups=[rng.choice(group_ids) for __ in group_ids]
                sample=[]
                for g in sample_groups:
                    qs=set(gq[g])
                    sample.extend(x for x in paired if x[0] in qs)
                if not sample: continue
                boots.append((sum(x[2] for x in sample)/len(sample)-sum(x[3] for x in sample)/len(sample))*100)
            lo=quantile(boots,0.025); hi=quantile(boots,0.975)
            results.append({
                'setting_id':ds,'cell':cell,'comparison':f'{METHOD_LABEL[a]} - {METHOD_LABEL[b]}',
                'a':METHOD_LABEL[a],'b':METHOD_LABEL[b], 'n_decisive_pairs':n,
                'a_rate':aval*100,'b_rate':bval*100,'diff_pp':diff,'ci95_low':lo,'ci95_high':hi,
                'a_only':a_only,'b_only':b_only,'sign_p':pval,
            })
    csv_path=out/'paired_defense_comparisons.csv'
    with open(csv_path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    tex=[]
    tex.append('\\begin{table}[t]')
    tex.append('\\centering\\small')
    tex.append('\\begin{tabular}{@{}llrrrr@{}}')
    tex.append('\\toprule')
    tex.append('Setting & Comparison & $n$ & Rates & Diff. & 95\\% CI \\\\')
    tex.append('\\midrule')
    for r in results:
        if r['a']!='TrustRAG': continue
        setting=r['setting_id'].replace('_contriever_llama3','').replace('hotpotqa','HotpotQA').replace('nq','NQ')+' '+r['cell'].replace('_','/')
        rates=f"{r['a_rate']:.0f}--{r['b_rate']:.0f}"
        tex.append(f"{setting} & {r['comparison']} & {r['n_decisive_pairs']} & {rates} & {r['diff_pp']:.1f} & [{r['ci95_low']:.1f}, {r['ci95_high']:.1f}] \\")
    tex.append('\\bottomrule')
    tex.append('\\end{tabular}')
    tex.append('\\caption{Paired defense-to-defense differences in judged target endorsement on the defended poisoned endpoint. Rates are percentages for the two defenses in the comparison; Diff. is the first defense minus the second, so negative values mean the first defense has lower target endorsement. Intervals are group bootstraps over shared injection groups.}')
    tex.append('\\label{tab:paired-defense-comparisons}')
    tex.append('\\end{table}')
    (out/'paired_defense_comparisons.tex').write_text('\n'.join(tex)+'\n')
    print(csv_path)
    print(out/'paired_defense_comparisons.tex')
    for r in results:
        if r['a']=='TrustRAG':
            print(r)
if __name__=='__main__': main()
