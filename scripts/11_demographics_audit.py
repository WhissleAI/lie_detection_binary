#!/usr/bin/env python3
"""Step 11 — demographics-confound audit (the bulletproof headline).

A few female defendants dominate the deceptive class, so the model's audio
gender/age reads correlate with the label. This script re-runs the best configs
WITH and WITHOUT all gender/age features, under leave-one-speaker-out, so we can
report a confound-free number.

    python scripts/11_demographics_audit.py
"""

from __future__ import annotations

import glob
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from lie_detector.config import CFG
from lie_detector.io_utils import read_json
from lie_detector.features.assemble import feature_columns

warnings.simplefilter("ignore")
SEED = CFG.random_seed
LOGO = LeaveOneGroupOut()


def is_demographic(col: str) -> bool:
    c = col.lower()
    return ("metaprob_gender" in c or "metaprob_age" in c or "meta_age_expected" in c
            or "meta_gender" in c or c.endswith("_gender_male") or c.endswith("_gender_female"))


def hgb():
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("c", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                          max_iter=300, random_state=SEED))])


def lr(k=None):
    steps = [("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]
    if k:
        from sklearn.feature_selection import SelectKBest, f_classif
        steps.append(("k", SelectKBest(f_classif, k=k)))
    steps.append(("c", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)))
    return Pipeline(steps)


def oof_prob(df, cols, model, y, groups):
    return cross_val_predict(model, df[cols], y, groups=groups, cv=LOGO, method="predict_proba", n_jobs=-1)[:, 1]


def score(y, prob):
    pred = (prob >= 0.5).astype(int)
    return accuracy_score(y, pred), balanced_accuracy_score(y, pred), roc_auc_score(y, prob)


def main():
    df = pd.read_parquet(CFG.features_dir / "features.parquet")
    # merge gemini video-derived scores + zero-shot prob
    rows = []
    for f in glob.glob(str(CFG.gemini_dir / "*.json")):
        r = read_json(f)
        d = {f"gem_{k}": float(v) for k, v in (r.get("features") or {}).items()}
        d["clip_id"] = r["clip_id"]; d["gem_video_prob"] = float(r.get("deception_probability", 0.5))
        rows.append(d)
    df = df.merge(pd.DataFrame(rows), on="clip_id", how="left")
    y = df["y"].to_numpy(); groups = df["speaker"].to_numpy()

    ours = [c for c in feature_columns(df, ["txt", "vis", "aud"]) if df[c].nunique(dropna=True) > 1]
    gem = [c for c in df.columns if c.startswith("gem_") and c != "gem_video_prob" and df[c].nunique(dropna=True) > 1]
    demo = [c for c in ours if is_demographic(c)]
    ours_nodemo = [c for c in ours if not is_demographic(c)]
    print(f"{len(df)} clips | our features={len(ours)} ({len(demo)} demographic dropped) | gemini={len(gem)}")
    print(f"  dropped demographic columns: {sorted(c.replace('txt_','') for c in demo)}\n")

    rows_out = []

    def add(label, with_demo, prob):
        a, b, au = score(y, prob)
        rows_out.append({"system": label, "demographics": "WITH" if with_demo else "DROPPED",
                         "accuracy": a, "balanced_acc": b, "roc_auc": au})

    # CONFIG B (self-hosted, no LLM): hist_gbm on all our features
    add("B self-hosted (hist_gbm)", True, oof_prob(df, ours, hgb(), y, groups))
    add("B self-hosted (hist_gbm)", False, oof_prob(df, ours_nodemo, hgb(), y, groups))

    # CONFIG A (with Gemini): late-fusion our-model ⊕ gemini video prob
    gv = df["gem_video_prob"].to_numpy()
    for wd, cols in [(True, ours), (False, ours_nodemo)]:
        our_p = oof_prob(df, cols, lr(k=20), y, groups)
        add("A with-Gemini (late-fusion w=.7)", wd, 0.7 * gv + 0.3 * our_p)

    # Gemini-only references (no demographics by construction)
    add("Gemini direct video (0-shot)", False, gv)

    res = pd.DataFrame(rows_out)
    pd.set_option("display.width", 140)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    res.to_csv(CFG.reports_dir / "demographics_audit.csv", index=False)

    print("\n★ BULLETPROOF (demographics dropped):")
    for label in ["B self-hosted (hist_gbm)", "A with-Gemini (late-fusion w=.7)"]:
        w = res[(res.system == label) & (res.demographics == "WITH")].iloc[0]
        d = res[(res.system == label) & (res.demographics == "DROPPED")].iloc[0]
        print(f"  {label:<34} AUC {w.roc_auc:.3f} → {d.roc_auc:.3f}  (Δ {d.roc_auc-w.roc_auc:+.3f})")
    print(f"saved -> {CFG.reports_dir/'demographics_audit.csv'}")


if __name__ == "__main__":
    main()
