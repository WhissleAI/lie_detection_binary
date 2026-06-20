#!/usr/bin/env python3
"""Step 06 — why does the original paper report ~75%? CV protocol vs. ours.

Pérez-Rosas et al. (2015) report up to 75.2% accuracy on this dataset. The two
big reasons their numbers are higher than our honest baseline:

  1. They use **leave-one-VIDEO-out** CV (not speaker-independent). With 32 Jodi
     Arias clips (all deceptive), testing one leaves 31 in training -> the model
     learns the *speaker*, not deception. We use **leave-one-SPEAKER-out**.
  2. Their best features are **manual human gesture annotations** (the shipped
     All_Gestures CSV), not automatically-extracted visual features.

This script quantifies both: it runs each feature set under both CV schemes so
the leakage gap is explicit. Run after 04_build_features.py.

    python scripts/06_paper_comparison.py
"""

from __future__ import annotations

import warnings

import glob

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline

from lie_detector.config import CFG
from lie_detector.features.assemble import feature_columns
from lie_detector.io_utils import read_json
from lie_detector.extraction.gemini import gemini_features as gem_feats

warnings.simplefilter("ignore")


def load_manual_gestures():
    """The paper's manual MUMIN gesture annotations (40 binary features)."""
    df = pd.read_csv(CFG.annotation_csv)
    df["clip_id"] = df["id"].str.replace(".mp4", "", regex=False)
    feat_cols = [c for c in df.columns if c not in ("id", "clip_id", "class")]
    df = df[["clip_id"] + feat_cols]
    return df, feat_cols


def load_gemini():
    """Gemini per-clip records -> (features df with gem_* cols, direct df)."""
    rows, direct = [], []
    for f in sorted(glob.glob(str(CFG.gemini_dir / "*.json"))):
        r = read_json(f)
        cid = r.get("clip_id")
        if not cid:
            continue
        feats = {f"gem_{k}": v for k, v in gem_feats(r).items()}
        feats["clip_id"] = cid
        rows.append(feats)
        direct.append({"clip_id": cid, "gem_verdict": r.get("verdict"),
                       "gem_prob": float(r.get("deception_probability", 0.5))})
    if not rows:
        return None, None, []
    fdf = pd.DataFrame(rows)
    gem_cols = [c for c in fdf.columns if c.startswith("gem_")]
    return fdf, pd.DataFrame(direct), gem_cols


def load_gemini_reason(folder, prefix):
    """Gemini-over-our-features verdicts from `folder` -> df (clip_id, <prefix>_verdict/prob)."""
    rows = []
    for f in sorted(glob.glob(str(folder / "*.json"))):
        r = read_json(f)
        if r.get("clip_id"):
            rows.append({"clip_id": r["clip_id"], f"{prefix}_verdict": r.get("verdict"),
                         f"{prefix}_prob": float(r.get("deception_probability", 0.5))})
    return pd.DataFrame(rows) if rows else None


def _zero_shot_row(df, verdict_col, prob_col, name, rows_out):
    """Print + record a zero-shot (no-training) accuracy/AUC row."""
    if verdict_col not in df:
        return
    mask = df[verdict_col].notna()
    if not mask.any():
        return
    yv = df.loc[mask, "y"].to_numpy()
    pred = (df.loc[mask, verdict_col] == "deceptive").astype(int).to_numpy()
    prob = df.loc[mask, prob_col].to_numpy()
    acc = accuracy_score(yv, pred)
    auc = roc_auc_score(yv, prob) if len(set(yv)) > 1 else float("nan")
    print(f"{name:<26}{'gemini-2.5-pro':<16}{acc:>19.3f}{acc:>21.3f}{0.0:>+8.3f}")
    print(f"   └─ zero-shot accuracy={acc:.3f}  AUC={auc:.3f}  (n={len(yv)}, no training → no leakage)")
    rows_out.append({"feature_set": name, "model": "gemini-2.5-pro", "n_features": 0,
                     "acc_leave_one_video_out": round(acc, 4),
                     "acc_leave_one_speaker_out": round(acc, 4), "leakage_gap": 0.0,
                     "zero_shot_auc": round(auc, 4)})


def pooled_accuracy(X, y, groups, scheme: str, model) -> float:
    if scheme == "leave-one-video-out":
        cv, grp = LeaveOneOut(), None
    elif scheme == "leave-one-speaker-out":
        cv, grp = LeaveOneGroupOut(), groups
    else:
        raise ValueError(scheme)
    pred = cross_val_predict(model, X, y, groups=grp, cv=cv, n_jobs=-1)
    return float((pred == y).mean())


def main() -> None:
    feats = pd.read_parquet(CFG.features_dir / "features.parquet")
    gestures, gest_cols = load_manual_gestures()
    df = feats.merge(gestures, on="clip_id", how="left")

    gem_fdf, gem_direct, gem_cols = load_gemini()
    if gem_fdf is not None:
        df = df.merge(gem_fdf, on="clip_id", how="left")
        df = df.merge(gem_direct, on="clip_id", how="left")
    gemr1 = load_gemini_reason(CFG.gemini_reason_dir, "gemr1")      # v1 forensic prompt
    if gemr1 is not None:
        df = df.merge(gemr1, on="clip_id", how="left")
    gemr2 = load_gemini_reason(CFG.gemini_reason_v2_dir, "gemr2")   # v2 neutral prompt
    if gemr2 is not None:
        df = df.merge(gemr2, on="clip_id", how="left")

    y = df["y"].to_numpy()
    groups = df["speaker"].to_numpy()

    def tree(clf):
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", clf)])

    models = {
        "decision_tree": tree(DecisionTreeClassifier(random_state=CFG.random_seed)),
        "random_forest": tree(RandomForestClassifier(n_estimators=400, random_state=CFG.random_seed, n_jobs=-1)),
    }
    feature_sets = {
        "our_text (auto)": feature_columns(df, ["txt"]),
        "our_visual (auto)": feature_columns(df, ["vis"]),
        "our_audio (auto)": feature_columns(df, ["aud"]),
        "our_all (auto)": feature_columns(df, ["txt", "vis", "aud"]),
        "manual_gestures (paper)": gest_cols,
    }
    if gem_cols:
        feature_sets["gemini_features (LLM)"] = gem_cols
        feature_sets["gemini+our_all"] = feature_columns(df, ["txt", "vis", "aud"]) + gem_cols

    print(f"{len(df)} clips, {df.speaker.nunique()} speakers. Pooled OOF accuracy.\n")
    header = f"{'feature_set':<26}{'model':<16}{'leave-1-VIDEO-out':>19}{'leave-1-SPEAKER-out':>21}{'gap':>8}"
    print(header)
    print("-" * len(header))
    rows = []
    for fs_name, cols in feature_sets.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        X = df[cols]
        for m_name, model in models.items():
            a_vid = pooled_accuracy(X, y, groups, "leave-one-video-out", model)
            a_spk = pooled_accuracy(X, y, groups, "leave-one-speaker-out", model)
            print(f"{fs_name:<26}{m_name:<16}{a_vid:>19.3f}{a_spk:>21.3f}{a_vid-a_spk:>+8.3f}")
            rows.append({"feature_set": fs_name, "model": m_name, "n_features": len(cols),
                         "acc_leave_one_video_out": round(a_vid, 4),
                         "acc_leave_one_speaker_out": round(a_spk, 4),
                         "leakage_gap": round(a_vid - a_spk, 4)})

    # Gemini zero-shot verdicts — no training, so protocol-invariant.
    _zero_shot_row(df, "gem_verdict", "gem_prob", "gemini_direct_video (0-shot)", rows)
    _zero_shot_row(df, "gemr1_verdict", "gemr1_prob", "gemini_features_v1forensic (0-shot)", rows)
    _zero_shot_row(df, "gemr2_verdict", "gemr2_prob", "gemini_features_v2neutral (0-shot)", rows)

    base = max(float(np.mean(y)), 1 - float(np.mean(y)))
    print("\nbaseline (majority):", f"{base:.3f}")
    out = CFG.reports_dir / "paper_comparison.csv"
    CFG.ensure_dirs()
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved -> {out}")
    print(
        "\nReading: the 'gap' column is speaker-identity leakage. The paper's "
        "leave-one-VIDEO-out\nnumbers are the left column; our honest baseline is "
        "the right column."
    )


if __name__ == "__main__":
    main()
