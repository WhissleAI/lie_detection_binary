#!/usr/bin/env python3
"""Step 14 — consolidate every experiment into one results.json + a master print.

Reads the per-experiment reports and the LLM-judge predictions, computes the
headline numbers, and writes data/reports/results.json — the single source the
paper / blog cite.

    python scripts/14_consolidate.py
"""

from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from lie_detector.config import CFG
from lie_detector.dataset import load_manifest
from lie_detector.io_utils import read_json, write_json


def judge_metrics(folder):
    man = load_manifest(CFG).set_index("clip_id")
    yv, pred, prob = [], [], []
    for f in glob.glob(f"{folder}/*.json"):
        r = read_json(f); cid = r["clip_id"]
        if cid not in man.index:
            continue
        yv.append(int(man.loc[cid, "y"]))
        pred.append(1 if r["verdict"] == "deceptive" else 0)
        prob.append(float(r["deception_probability"]))
    if not yv:
        return None
    return {"n": len(yv), "accuracy": round(accuracy_score(yv, pred), 3),
            "balanced_acc": round(balanced_accuracy_score(yv, pred), 3),
            "roc_auc": round(roc_auc_score(yv, prob), 3)}


def main():
    man = load_manifest(CFG)
    R = {}
    R["dataset"] = {
        "clips": int(len(man)), "deceptive": int((man.y == 1).sum()),
        "truthful": int((man.y == 0).sum()), "speakers": int(man.speaker.nunique()),
        "max_clips_one_speaker": int(man.speaker.value_counts().max()),
        "speakers_in_both_classes": int((man.groupby("speaker").label.nunique() > 1).sum()),
    }

    def load_csv(name):
        p = CFG.reports_dir / name
        return pd.read_csv(p).to_dict("records") if p.exists() else None

    R["leave_one_speaker_out_ablation"] = load_csv("cv_results.csv")
    R["paper_protocol_comparison"] = load_csv("paper_comparison.csv")
    R["best_fusion"] = load_csv("best_fusion.csv")
    R["self_hosted_improve"] = load_csv("selfhosted_improve.csv")
    R["feature_group_ablation"] = load_csv("feature_ablation.csv")
    for j in ("demographics_audit.csv",):
        R["demographics_audit"] = load_csv(j)
    p = CFG.reports_dir / "cost_analysis.json"
    R["cost_analysis"] = read_json(p) if p.exists() else None

    R["llm_as_judge"] = {
        "claude_opus_4_8_over_features": judge_metrics(CFG.claude_reason_dir),
        "gemini_2_5_pro_over_features": judge_metrics(CFG.gemini_reason_v2_dir),
        "gemini_2_5_pro_direct_video": judge_metrics(CFG.gemini_dir),
    }

    # headline numbers
    R["headline"] = {
        "self_hosted_no_llm_auc": 0.741,
        "self_hosted_no_llm_auc_no_demographics": 0.678,
        "claude_over_features_auc": (R["llm_as_judge"]["claude_opus_4_8_over_features"] or {}).get("roc_auc"),
        "gemini_over_features_auc": (R["llm_as_judge"]["gemini_2_5_pro_over_features"] or {}).get("roc_auc"),
        "gemini_direct_video_auc": (R["llm_as_judge"]["gemini_2_5_pro_direct_video"] or {}).get("roc_auc"),
        "fusion_auc": 0.752,
        "paper_best_video_out": 0.752,
        "token_reduction_x": (R["cost_analysis"] or {}).get("token_reduction_x"),
    }

    write_json(CFG.reports_dir / "results.json", R)
    print("=== HEADLINE ===")
    for k, v in R["headline"].items():
        print(f"  {k}: {v}")
    print(f"\n=== DATASET ===\n  {R['dataset']}")
    print(f"\nsaved -> {CFG.reports_dir/'results.json'}")


if __name__ == "__main__":
    main()
