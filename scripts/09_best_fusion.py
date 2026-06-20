#!/usr/bin/env python3
"""Step 09 — find the BEST honest (leave-one-speaker-out) config, two ways:

  CONFIG B  "self-hosted, no LLM": only our locally-extracted features
            (Whissle STT text+metadata, MediaPipe visual, librosa prosody).
            No raw audio/video ever leaves to an external LLM.
  CONFIG A  "with Gemini": adds the LLM's video-derived signals
            (gemini_features + late-fusion with Gemini's zero-shot video prob).

Naive concatenation of all 166 features dilutes signal (gemini_features alone
beats gemini+our_all). So we test feature selection, L1 sparsity, and late
fusion. Headline metric = ROC-AUC under LeaveOneSpeakerOut (honest, no leakage).

    python scripts/09_best_fusion.py
"""

from __future__ import annotations

import glob
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from lie_detector.config import CFG
from lie_detector.dataset import load_manifest
from lie_detector.io_utils import read_json
from lie_detector.features.assemble import feature_columns

warnings.simplefilter("ignore")
SEED = CFG.random_seed


def load_gemini_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    """Merge Gemini video-derived feature scores + zero-shot video prob."""
    man = load_manifest(CFG)
    rows = []
    for f in glob.glob(str(CFG.gemini_dir / "*.json")):
        r = read_json(f)
        feats = {f"gem_{k}": float(v) for k, v in (r.get("features") or {}).items()}
        feats["clip_id"] = r["clip_id"]
        feats["gem_video_prob"] = float(r.get("deception_probability", 0.5))
        rows.append(feats)
    gdf = pd.DataFrame(rows)
    df = df.merge(gdf, on="clip_id", how="left")
    gem_cols = [c for c in gdf.columns if c.startswith("gem_") and c != "gem_video_prob"]
    return df, gem_cols, "gem_video_prob"


def oof(df, cols, model, y, groups):
    """Leave-one-speaker-out out-of-fold predictions + probabilities."""
    X = df[cols]
    pred = cross_val_predict(model, X, y, groups=groups, cv=LeaveOneGroupOut(), n_jobs=-1)
    prob = cross_val_predict(model, X, y, groups=groups, cv=LeaveOneGroupOut(),
                             method="predict_proba", n_jobs=-1)[:, 1]
    return pred, prob


def score(y, pred, prob):
    return (accuracy_score(y, pred), balanced_accuracy_score(y, pred), roc_auc_score(y, prob))


def lr(C=1.0, penalty="l2", k=None):
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if k:
        steps.append(("select", SelectKBest(f_classif, k=k)))
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    steps.append(("clf", LogisticRegression(C=C, penalty=penalty, solver=solver,
                                            class_weight="balanced", max_iter=5000, random_state=SEED)))
    return Pipeline(steps)


def hgb():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                            max_iter=300, random_state=SEED))])


def main():
    df = pd.read_parquet(CFG.features_dir / "features.parquet")
    df, gem_cols, gem_prob_col = load_gemini_cols(df)
    y = df["y"].to_numpy()
    groups = df["speaker"].to_numpy()

    # Drop zero-variance columns (e.g. *_present flags, all-1 now) — they carry no
    # signal and pollute univariate feature selection with NaN F-scores.
    def keep_varying(cols):
        return [c for c in cols if df[c].nunique(dropna=True) > 1]

    ours = keep_varying(feature_columns(df, ["txt", "vis", "aud"]))
    gem_cols = keep_varying(gem_cols)
    print(f"{len(df)} clips, {df.speaker.nunique()} speakers | our features={len(ours)} (varying), gemini={len(gem_cols)}\n")

    results = []

    def run(config, name, cols, model):
        pred, prob = oof(df, cols, model, y, groups)
        a, b, au = score(y, pred, prob)
        results.append({"config": config, "method": name, "n": len(cols),
                        "accuracy": a, "balanced_acc": b, "roc_auc": au})

    # ---------- CONFIG B: self-hosted, NO LLM (our features only) ----------
    run("B no-LLM", "all-our-features  logreg-L2", ours, lr())
    run("B no-LLM", "L1-sparse         logreg-L1", ours, lr(C=0.5, penalty="l1"))
    for k in (10, 15, 20, 30):
        run("B no-LLM", f"SelectKBest(k={k})  logreg", ours, lr(k=k))
    run("B no-LLM", "hist_gbm", ours, hgb())

    # ---------- CONFIG A: WITH Gemini --------------------------------------
    run("A +Gemini", "gemini_features only  logreg", gem_cols, lr())
    run("A +Gemini", "gemini+our concat     logreg", gem_cols + ours, lr())
    for k in (10, 15, 20):
        run("A +Gemini", f"SelectKBest(k={k}) on all", gem_cols + ours, lr(k=k))

    # Late fusion: average our-features OOF prob with Gemini's zero-shot video prob.
    _, our_prob = oof(df, ours, lr(k=20), y, groups)
    gem_vid = df[gem_prob_col].to_numpy()
    for w in (0.3, 0.5, 0.7):  # weight on Gemini video prob
        fused = w * gem_vid + (1 - w) * our_prob
        a = accuracy_score(y, (fused >= 0.5).astype(int))
        b = balanced_accuracy_score(y, (fused >= 0.5).astype(int))
        au = roc_auc_score(y, fused)
        results.append({"config": "A +Gemini", "method": f"LATE-FUSION our⊕gemvid (w={w})",
                        "n": 21, "accuracy": a, "balanced_acc": b, "roc_auc": au})

    res = pd.DataFrame(results).sort_values(["config", "roc_auc"], ascending=[True, False])
    pd.set_option("display.width", 140)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    res.to_csv(CFG.reports_dir / "best_fusion.csv", index=False)

    print("\nBEST per config (by AUC):")
    for cfg_name, g in res.groupby("config"):
        r = g.iloc[0]
        print(f"  {cfg_name:<10} {r['method']:<32} acc={r['accuracy']:.3f} bal={r['balanced_acc']:.3f} AUC={r['roc_auc']:.3f}")
    print(f"\nsaved -> {CFG.reports_dir/'best_fusion.csv'}")


if __name__ == "__main__":
    main()
