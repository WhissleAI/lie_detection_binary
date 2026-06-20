#!/usr/bin/env python3
"""Step 10 — can the self-hosted (no-LLM) number beat hist_gbm's AUC 0.670?

Same honest leave-one-speaker-out protocol, OUR features only (no Gemini, no raw
media leaving). Tests the standard ways to squeeze more from fixed features:
single models, soft-voting ensemble, per-modality LATE FUSION, and STACKING; and
reports best-threshold accuracy (the 0.5 threshold under-sells AUC 0.670).

    python scripts/10_improve_selfhosted.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve

from lie_detector.config import CFG
from lie_detector.features.assemble import feature_columns

warnings.simplefilter("ignore")
SEED = CFG.random_seed
LOGO = LeaveOneGroupOut()


def lr(C=1.0):
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("c", LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED))])


def svm():
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("c", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED))])


def rf():
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("c", RandomForestClassifier(n_estimators=600, min_samples_leaf=2,
                                                  class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))])


def et():
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("c", ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2,
                                                class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))])


def hgb():
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("c", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                          max_iter=300, random_state=SEED))])


def oof_prob(df, cols, model, y, groups):
    return cross_val_predict(model, df[cols], y, groups=groups, cv=LOGO, method="predict_proba", n_jobs=-1)[:, 1]


def best_threshold_acc(y, prob):
    """Accuracy at the speaker-out-optimal threshold (reported separately, mildly optimistic)."""
    fpr, tpr, thr = roc_curve(y, prob)
    best = 0.5
    for t in thr:
        acc = accuracy_score(y, (prob >= t).astype(int))
        if acc > best_threshold_acc.cache.get("best", 0):
            best_threshold_acc.cache["best"] = acc
            best = t
    a = best_threshold_acc.cache.pop("best", 0)
    return a
best_threshold_acc.cache = {}


def report(name, y, prob, rows):
    pred = (prob >= 0.5).astype(int)
    rows.append({"method": name, "accuracy@0.5": accuracy_score(y, pred),
                 "balanced_acc": balanced_accuracy_score(y, pred),
                 "roc_auc": roc_auc_score(y, prob),
                 "acc@best_thr": best_threshold_acc(y, prob)})


def main():
    df = pd.read_parquet(CFG.features_dir / "features.parquet")
    y = df["y"].to_numpy(); groups = df["speaker"].to_numpy()
    txt = feature_columns(df, ["txt"]); vis = feature_columns(df, ["vis"]); aud = feature_columns(df, ["aud"])
    allc = txt + vis + aud
    print(f"{len(df)} clips, {df.speaker.nunique()} speakers | txt={len(txt)} vis={len(vis)} aud={len(aud)}\n")

    rows = []
    models = {"logreg": lr(), "svm_rbf": svm(), "random_forest": rf(), "extra_trees": et(), "hist_gbm": hgb()}

    # --- single models on all features ---
    probs = {}
    for n, m in models.items():
        p = oof_prob(df, allc, m, y, groups); probs[n] = p
        report(f"single: {n}", y, p, rows)

    # --- soft-voting ensemble (avg of the single-model OOF probs) ---
    report("ensemble: mean(logreg,svm,rf,et,hgb)", y, np.mean(list(probs.values()), axis=0), rows)
    report("ensemble: mean(logreg,svm,hgb)", y, np.mean([probs["logreg"], probs["svm_rbf"], probs["hist_gbm"]], axis=0), rows)

    # --- per-modality LATE FUSION (train each modality, average probs) ---
    mod_probs = {}
    for mname, cols in [("text", txt), ("visual", vis), ("audio", aud)]:
        mod_probs[mname] = {"logreg": oof_prob(df, cols, lr(), y, groups),
                            "hgb": oof_prob(df, cols, hgb(), y, groups)}
        report(f"modality-only: {mname} (logreg)", y, mod_probs[mname]["logreg"], rows)
    lf_lr = np.mean([mod_probs[m]["logreg"] for m in mod_probs], axis=0)
    lf_hgb = np.mean([mod_probs[m]["hgb"] for m in mod_probs], axis=0)
    report("LATE-FUSION: mean per-modality logreg", y, lf_lr, rows)
    report("LATE-FUSION: mean per-modality hgb", y, lf_hgb, rows)
    # weighted (text+visual stronger than audio)
    wf = 0.4 * mod_probs["text"]["logreg"] + 0.4 * mod_probs["visual"]["logreg"] + 0.2 * mod_probs["audio"]["logreg"]
    report("LATE-FUSION: weighted .4txt/.4vis/.2aud", y, wf, rows)

    # --- STACKING: per-modality OOF probs -> meta logreg (CV again) ---
    meta_X = pd.DataFrame({f"{m}_{k}": mod_probs[m][k] for m in mod_probs for k in ("logreg", "hgb")})
    meta = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED))])
    stack_p = cross_val_predict(meta, meta_X, y, groups=groups, cv=LOGO, method="predict_proba", n_jobs=-1)[:, 1]
    report("STACKING: per-modality probs -> logreg", y, stack_p, rows)

    res = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    pd.set_option("display.width", 150)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    res.to_csv(CFG.reports_dir / "selfhosted_improve.csv", index=False)
    best = res.iloc[0]
    print(f"\n★ best self-hosted (no-LLM): {best['method']}  AUC={best['roc_auc']:.3f}  "
          f"balanced_acc={best['balanced_acc']:.3f}  acc@best_thr={best['acc@best_thr']:.3f}")
    print("  (baseline was hist_gbm AUC 0.670, acc@0.5 0.562)")
    print(f"saved -> {CFG.reports_dir/'selfhosted_improve.csv'}")


if __name__ == "__main__":
    main()
