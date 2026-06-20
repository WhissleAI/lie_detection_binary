#!/usr/bin/env python3
"""Step 13 — feature-group ablation (which signals matter).

Groups the ~250 features into interpretable families and runs, under
leave-one-speaker-out: (a) each group ALONE, and (b) leave-one-group-OUT (the
drop from the full set). Quantifies each modality's marginal contribution.

    python scripts/13_feature_ablation.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

from lie_detector.config import CFG
from lie_detector.features.assemble import feature_columns

warnings.simplefilter("ignore")
SEED = CFG.random_seed


def _assign(c: str) -> str:
    """Assign each feature column to exactly ONE group (priority order)."""
    if c.startswith("vis_"):
        return "visual (face/gaze/pose/gesture)"
    if c.startswith("aud_"):
        return "prosody (librosa)"
    if "_sa_" in c:
        return "speech_analysis (fluency/grammar/rhythm)"
    if "_fintent_" in c:
        return "deception-intent filter"
    if "metaprob_" in c or c.startswith("txt_meta_"):
        return "STT metadata probs (emotion/age/gender/intent)"
    if any(s in c for s in ("_sr_", "_pause_", "_word_conf", "_low_conf", "_filler_word",
                            "_asr_confidence", "_uncertain_", "_words_per_second",
                            "_n_segments", "_n_entities")):
        return "speech structure (rate/pauses/conf)"
    return "lexical (psycholinguistic)"


def groups_of(df: pd.DataFrame) -> dict[str, list[str]]:
    cols = [c for c in feature_columns(df, ["txt", "vis", "aud"]) if df[c].nunique(dropna=True) > 1]
    g: dict[str, list[str]] = {}
    for c in cols:
        g.setdefault(_assign(c), []).append(c)
    return g


def hgb():
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("c", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                          max_iter=300, random_state=SEED))])


def auc_acc(df, cols, y, groups):
    if not cols:
        return float("nan"), float("nan")
    prob = cross_val_predict(hgb(), df[cols], y, groups=groups, cv=LeaveOneGroupOut(),
                             method="predict_proba", n_jobs=-1)[:, 1]
    return roc_auc_score(y, prob), accuracy_score(y, (prob >= 0.5).astype(int))


def main():
    df = pd.read_parquet(CFG.features_dir / "features.parquet")
    y = df["y"].to_numpy(); groups = df["speaker"].to_numpy()
    G = groups_of(df)
    allc = [c for cols in G.values() for c in cols]
    full_auc, full_acc = auc_acc(df, allc, y, groups)
    print(f"{len(df)} clips | {len(allc)} features in {len(G)} groups | FULL: AUC {full_auc:.3f} acc {full_acc:.3f}\n")

    rows = []
    for name, cols in G.items():
        a_alone, _ = auc_acc(df, cols, y, groups)
        a_drop, _ = auc_acc(df, [c for c in allc if c not in cols], y, groups)
        rows.append({"group": name, "n_features": len(cols),
                     "alone_auc": round(a_alone, 3),
                     "drop_auc": round(a_drop, 3),
                     "marginal_delta": round(full_auc - a_drop, 3)})
    res = pd.DataFrame(rows).sort_values("alone_auc", ascending=False)
    pd.set_option("display.width", 140)
    print(res.to_string(index=False))
    res.to_csv(CFG.reports_dir / "feature_ablation.csv", index=False)
    print(f"\nFULL set AUC = {full_auc:.3f}. 'alone_auc' = that group only; "
          f"'drop_auc' = all-but-that-group; 'marginal_delta' = AUC lost by removing it.")
    print(f"saved -> {CFG.reports_dir/'feature_ablation.csv'}")


if __name__ == "__main__":
    main()
