#!/usr/bin/env python3
"""Compute paired uncertainty summaries for aligned RAG records.

Outputs CSV and LaTeX rows for the main four-condition quantities.
Intervals are nonparametric bootstrap percentile intervals over shared injection
or query groups. McNemar p-values use the exact binomial test on discordant
paired outcomes (prevented vs introduced).
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random
from collections import defaultdict
from pathlib import Path

METHODS = ["robustrag_keyword", "robustrag_prompt_control", "trustrag_filter_conflict"]
METHOD_LABEL = {
    "robustrag_keyword": "KeywordAgg",
    "robustrag_prompt_control": "Prompt",
    "trustrag_filter_conflict": "TrustRAG",
}
DATASET_LABEL = {"hotpotqa": "HotpotQA", "nq": "NQ"}


def target_string_label(out: dict) -> int:
    target = str(out.get("target", "")).strip().lower()
    answer = str(out.get("answer", "")).lower()
    return int(bool(target) and target in answer)


def judge_label(output_id: str, judgments: dict) -> int | None:
    j = judgments.get(output_id)
    if not j:
        return None
    val = str(j.get("labels", {}).get("target_endorsed", "")).strip().lower()
    if val in {"yes", "y", "true"}:
        return 1
    if val in {"no", "n", "false"}:
        return 0
    return None


def exact_mcnemar_p(prevented: int, introduced: int) -> float:
    n = prevented + introduced
    if n == 0:
        return 1.0
    k = min(prevented, introduced)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * prob)


def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def ci(vals):
    return percentile(vals, 0.025), percentile(vals, 0.975)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    den = 1 + z*z/n
    center = (phat + z*z/(2*n)) / den
    half = z * math.sqrt((phat*(1-phat) + z*z/(4*n)) / n) / den
    return center - half, center + half


def build_items(doc, labeler):
    outs = doc["outputs"]
    items = []
    for r in doc["records"]:
        if r["method"] == "no_defense":
            continue
        c = r["conditions"]
        vals = {
            "clean_undef": labeler(c["clean_undefended"], outs),
            "poison_undef": labeler(c["poisoned_undefended"], outs),
            "clean_def": labeler(c["clean_defended"], outs),
            "poison_def": labeler(c["poisoned_defended"], outs),
        }
        if any(v is None for v in vals.values()):
            continue
        items.append({"cell": r["cell"], "method": r["method"], "group": str(r.get("group_id", r["id"])), **vals})
    return items


def summarize(rows):
    n = len(rows)
    c0 = sum(r["clean_undef"] for r in rows)
    p0 = sum(r["poison_undef"] for r in rows)
    c1 = sum(r["clean_def"] for r in rows)
    p1 = sum(r["poison_def"] for r in rows)
    prevented = sum(1 for r in rows if r["poison_undef"] and not r["poison_def"])
    introduced = sum(1 for r in rows if not r["poison_undef"] and r["poison_def"])
    clean_added = sum(1 for r in rows if not r["clean_undef"] and r["clean_def"])
    clean_removed = sum(1 for r in rows if r["clean_undef"] and not r["clean_def"])
    N = 100 * (p0 - p1) / n
    H = 100 * (c1 - c0) / n
    G = N + H
    delta = 100 * (p1 - p0) / n
    return dict(n=n, c0=c0, p0=p0, c1=c1, p1=p1, prevented=prevented, introduced=introduced,
                clean_added=clean_added, clean_removed=clean_removed, N=N, H=H, Gamma=G, delta=delta,
                mcnemar_p=exact_mcnemar_p(prevented, introduced))


def bootstrap(rows, metric, B=10000, seed=0):
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    groups = list(by_group)
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        sample = []
        for g in (rng.choice(groups) for _ in groups):
            sample.extend(by_group[g])
        vals.append(summarize(sample)[metric])
    return ci(vals)


def fmt_ci(x, lo, hi):
    return f"{x:.0f} [{lo:.0f}, {hi:.0f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot", default="factorial/hotpotqa_records_v1.json")
    ap.add_argument("--nq", default="factorial/nq_records_v1.json")
    ap.add_argument("--judgments", default="factorial/gpt4omini_judgments_v1.json")
    ap.add_argument("--out-dir", default="factorial/inference_v1")
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    judgments = json.load(open(args.judgments))["judgments"]
    datasets = [("hotpotqa", json.load(open(args.hotpot))), ("nq", json.load(open(args.nq)))]
    rows_out = []
    tex_lines = []
    for label_name, labeler in [
        ("string", lambda oid, outs: target_string_label(outs[oid])),
        ("judge", lambda oid, outs: judge_label(oid, judgments)),
    ]:
        for ds, doc in datasets:
            items = build_items(doc, labeler)
            for cell in ["p1_k5", "p5_k5"]:
                for method in METHODS:
                    rows = [r for r in items if r["cell"] == cell and r["method"] == method]
                    if not rows:
                        continue
                    summ = summarize(rows)
                    for metric in ["delta", "N", "H", "Gamma"]:
                        lo, hi = bootstrap(rows, metric, B=args.bootstrap, seed=int(hashlib.sha256('|'.join([label_name, ds, cell, method, metric]).encode()).hexdigest()[:8], 16))
                        summ[f"{metric}_lo"] = lo; summ[f"{metric}_hi"] = hi
                    rec = {"label": label_name, "dataset": ds, "cell": cell, "method": method, **summ}
                    rows_out.append(rec)
                    if label_name == "string" and cell == "p1_k5":
                        row = (
                            f"{DATASET_LABEL[ds]} & {METHOD_LABEL[method]} & "
                            f"{fmt_ci(summ['N'], summ['N_lo'], summ['N_hi'])} & "
                            f"{fmt_ci(summ['H'], summ['H_lo'], summ['H_hi'])} & "
                            f"{fmt_ci(summ['Gamma'], summ['Gamma_lo'], summ['Gamma_hi'])} & "
                            f"{summ['mcnemar_p']:.3g}"
                        )
                        tex_lines.append(row + " " + "\\\\")
    fields = ["label","dataset","cell","method","n","c0","p0","c1","p1","prevented","introduced","clean_added","clean_removed","delta","delta_lo","delta_hi","N","N_lo","N_hi","H","H_lo","H_hi","Gamma","Gamma_lo","Gamma_hi","mcnemar_p"]
    with open(out/"paired_inference.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows_out)
    with open(out/"factorial_main_rows.tex", "w") as f:
        f.write("\n".join(tex_lines)+"\n")
    endpoint_rows=[]
    for ds, doc in datasets:
        items = build_items(doc, lambda oid, outs: judge_label(oid, judgments))
        for cell in ["p1_k5", "p5_k5"]:
            for method in ["robustrag_keyword", "robustrag_prompt_control", "trustrag_filter_conflict"]:
                rows=[r for r in items if r['cell']==cell and r['method']==method]
                if not rows: continue
                k=sum(r['poison_def'] for r in rows); n=len(rows); lo,hi=wilson(k,n)
                endpoint_rows.append(dict(dataset=ds,cell=cell,method=method,k=k,n=n,lo=100*lo,hi=100*hi))
            rows=[r for r in items if r['cell']==cell and r['method']=='robustrag_keyword']
            k=sum(r['poison_undef'] for r in rows); n=len(rows); lo,hi=wilson(k,n)
            endpoint_rows.append(dict(dataset=ds,cell=cell,method='undefended',k=k,n=n,lo=100*lo,hi=100*hi))
    with open(out/"judge_endpoint_wilson.csv", "w", newline="") as f:
        w=csv.DictWriter(f, endpoint_rows[0].keys()); w.writeheader(); w.writerows(endpoint_rows)
    print(out/"paired_inference.csv")
    print(out/"factorial_main_rows.tex")
    print(out/"judge_endpoint_wilson.csv")

if __name__ == "__main__":
    main()
