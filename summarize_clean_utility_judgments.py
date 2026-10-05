#!/usr/bin/env python3
"""Summarize judged semantic clean utility from aligned_rag factorial records.
Reads factorial records plus an independent judge file. Reports held-out clean
utility for each method: reference correctness, abstention, semantic utility
retained from vanilla, correctness losses/gains, and whether the answer changed.
Uncertain labels are kept explicit rather than coerced to incorrect.
"""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

from aligned_rag import factorial as f
from aligned_rag.data import read
from exp11_ablation import fingerprint

FIELDS = ("target_endorsed", "reference_correct", "abstains")


def load_records(paths):
    sources = {}
    outputs = {}
    for path in paths:
        data = f.unseal(read(path))
        if data.get("kind") != "factorial_records":
            raise ValueError(f"Expected factorial_records: {path}")
        sid = data["setting_id"]
        if sid in sources:
            raise ValueError(f"duplicate setting id: {sid}")
        sources[sid] = data
        for key, item in data["outputs"].items():
            if key in outputs and outputs[key] != item:
                raise ValueError("output identity collision")
            outputs[key] = item
    return sources, outputs


def load_judge(path, sources, outputs):
    judge = f.unseal(read(path))
    expected = {k: v["sha256"] for k, v in sources.items()}
    if judge.get("kind") != "independent_judge":
        raise ValueError("Expected independent_judge")
    if judge.get("config", {}).get("sources") != expected:
        raise ValueError("Judge source hashes do not match records")
    if set(judge.get("judgments", {})) != set(outputs):
        raise ValueError("Judge judgments do not match record outputs")
    return judge


def label(judge, key, field):
    return judge["judgments"][key]["labels"][field]


def frac(num, den):
    return None if den == 0 else num / den


def summarize_method(rows, judge):
    out = defaultdict(int)
    out["n"] = len(rows)
    for r in rows:
        b = r["before"]
        a = r["after"]
        before_ref = label(judge, b, "reference_correct")
        after_ref = label(judge, a, "reference_correct")
        before_abs = label(judge, b, "abstains")
        after_abs = label(judge, a, "abstains")
        out[f"before_ref_{before_ref}"] += 1
        out[f"after_ref_{after_ref}"] += 1
        out[f"before_abs_{before_abs}"] += 1
        out[f"after_abs_{after_abs}"] += 1
        if before_ref in ("yes", "no") and after_ref in ("yes", "no"):
            out["ref_resolved_pairs"] += 1
            out["correct_retained"] += before_ref == "yes" and after_ref == "yes"
            out["correct_lost"] += before_ref == "yes" and after_ref == "no"
            out["correct_gained"] += before_ref == "no" and after_ref == "yes"
            out["incorrect_retained"] += before_ref == "no" and after_ref == "no"
        else:
            out["ref_unresolved_pairs"] += 1
        if b != a:
            out["answer_changed"] += 1
        if before_abs in ("yes", "no") and after_abs in ("yes", "no"):
            out["abs_resolved_pairs"] += 1
            out["abstention_introduced"] += before_abs == "no" and after_abs == "yes"
            out["abstention_removed"] += before_abs == "yes" and after_abs == "no"
            out["abstention_retained"] += before_abs == "yes" and after_abs == "yes"
        else:
            out["abs_unresolved_pairs"] += 1
    n = out["n"]
    resolved = out["ref_resolved_pairs"]
    abs_resolved = out["abs_resolved_pairs"]
    return {
        "n": n,
        "before_reference_yes": out["before_ref_yes"],
        "before_reference_no": out["before_ref_no"],
        "before_reference_uncertain": out["before_ref_uncertain"],
        "after_reference_yes": out["after_ref_yes"],
        "after_reference_no": out["after_ref_no"],
        "after_reference_uncertain": out["after_ref_uncertain"],
        "after_abstains_yes": out["after_abs_yes"],
        "after_abstains_no": out["after_abs_no"],
        "after_abstains_uncertain": out["after_abs_uncertain"],
        "answer_changed": out["answer_changed"],
        "answer_changed_rate": frac(out["answer_changed"], n),
        "resolved_reference_pairs": resolved,
        "correct_retained": out["correct_retained"],
        "correct_lost": out["correct_lost"],
        "correct_gained": out["correct_gained"],
        "incorrect_retained": out["incorrect_retained"],
        "reference_accuracy_after_resolved": frac(out["after_ref_yes"], out["after_ref_yes"] + out["after_ref_no"]),
        "reference_accuracy_before_resolved": frac(out["before_ref_yes"], out["before_ref_yes"] + out["before_ref_no"]),
        "semantic_utility_retained_among_before_correct_resolved": frac(out["correct_retained"], out["correct_retained"] + out["correct_lost"]),
        "correct_loss_rate_among_before_correct_resolved": frac(out["correct_lost"], out["correct_retained"] + out["correct_lost"]),
        "abstention_introduced_resolved": out["abstention_introduced"],
        "abstention_introduction_rate_resolved": frac(out["abstention_introduced"], abs_resolved),
        "unresolved_reference_pairs": out["ref_unresolved_pairs"],
        "unresolved_abstention_pairs": out["abs_unresolved_pairs"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sources, outputs = load_records(args.records)
    judge = load_judge(args.judge, sources, outputs)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    report = {
        "kind": "clean_utility_judge_summary",
        "judge_sha256": judge["sha256"],
        "sources": judge["config"]["sources"],
        "note": "Uncertain labels are explicit. Rates with _resolved exclude uncertain labels from the denominator.",
        "settings": {},
    }
    rows_for_csv = []
    for sid, data in sources.items():
        by_method = defaultdict(list)
        for row in data["utility"]:
            by_method[row["method"]].append(row)
        setting = {}
        for method, rows in sorted(by_method.items()):
            sm = summarize_method(rows, judge)
            setting[method] = sm
            rows_for_csv.append({"setting": sid, "method": method, **sm})
        report["settings"][sid] = setting

    (root / "clean_utility_judged.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if rows_for_csv:
        fields = list(rows_for_csv[0].keys())
        with (root / "clean_utility_judged.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows_for_csv)
    for row in rows_for_csv:
        print(row["setting"], row["method"],
              "after_ref", f'{row["after_reference_yes"]}/{row["after_reference_yes"]+row["after_reference_no"]}',
              "unc", row["after_reference_uncertain"],
              "changed", row["answer_changed"],
              "lost", row["correct_lost"],
              "retained", row["correct_retained"],
              "abstain_after", row["after_abstains_yes"])
    print("[out]", root / "clean_utility_judged.json")
    print("[out]", root / "clean_utility_judged.csv")


if __name__ == "__main__":
    main()
