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

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline

from lie_detector.config import CFG
from lie_detector.features.assemble import feature_columns

warnings.simplefilter("ignore")


def load_manual_gestures() -> pd.DataFrame:
    """The paper's manual MUMIN gesture annotations (40 binary features)."""
    df = pd.read_csv(CFG.annotation_csv)
    df["clip_id"] = df["id"].str.replace(".mp4", "", regex=False)
    feat_cols = [c for c in df.columns if c not in ("id", "clip_id", "class")]
    df = df[["clip_id"] + feat_cols]
    return df, feat_cols


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
