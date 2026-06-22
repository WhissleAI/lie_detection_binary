"""Leave-one-SUBJECT-out fatigue classification (alert vs drowsy).

Reuses the deception feature builders. Reports each lane separately so we can see
whether the STT metadata generalises across people or is just session artefact
(the whole point of the honest LOSO test).

    python -m fatigue.train
"""

from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

from . import config as C
from lie_detector.features.text_features import text_features, collect_metadata_vocab
from lie_detector.features.visual_features import visual_features
from lie_detector.features.audio_features import audio_features

LANES = {
    "visual (face/gaze/blink/pose)": ("vis_",),
    "STT metadata (whissle-large)": ("txt_",),
    "prosody (librosa)": ("aud_",),
    "all combined": ("vis_", "txt_", "aud_"),
}


def build_matrix() -> pd.DataFrame:
    recs = []
    for p in sorted(glob.glob(str(C.AV / "*.json"))):
        av = json.load(open(p))
        cid = av.get("clip_id")
        ap = C.AUDIO / f"{cid}.json"
        audio = json.load(open(ap)) if ap.exists() else {}
        recs.append((av, audio))
    vocab = collect_metadata_vocab([av for av, _ in recs])
    rows = []
    for av, audio in recs:
        feat = {f"txt_{k}": v for k, v in text_features(av, vocab).items()}
        feat.update({f"vis_{k}": v for k, v in visual_features(av).items()})
        if audio:
            feat.update({f"aud_{k}": v for k, v in audio_features(audio).items()})
        feat["subject"] = av.get("subject")
        feat["label"] = av.get("label")
        rows.append(feat)
    return pd.DataFrame(rows)


def loso(df: pd.DataFrame, prefixes, classes=("alert", "drowsy")) -> dict:
    d = df[df.label.isin(classes)].copy()
    y = (d.label == classes[1]).astype(int).values
    groups = d.subject.values
    cols = [c for c in d.columns if any(c.startswith(p) for p in prefixes)]
    X = d[cols].apply(pd.to_numeric, errors="coerce")
    probs = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(set(y[tr])) < 2:
            probs[te] = y[tr].mean(); continue
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06,
                                             max_depth=2, random_state=0)
        clf.fit(X.iloc[tr], y[tr])
        probs[te] = clf.predict_proba(X.iloc[te])[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "n": len(y), "pos": int(y.sum()), "feats": len(cols),
        "acc": accuracy_score(y, preds),
        "bal_acc": balanced_accuracy_score(y, preds),
        "auc": roc_auc_score(y, probs) if len(set(y)) > 1 else float("nan"),
    }


def main() -> None:
    df = build_matrix()
    n_sub = df.subject.nunique()
    print(f"clips={len(df)}  subjects={n_sub}  labels={dict(df.label.value_counts())}\n")
    print(f"Leave-one-subject-out · alert vs drowsy · {n_sub} subjects")
    print(f"{'lane':<32}{'feats':>6}{'acc':>8}{'bal_acc':>9}{'AUC':>7}")
    print("-" * 62)
    for name, pref in LANES.items():
        r = loso(df, pref)
        print(f"{name:<32}{r['feats']:>6}{r['acc']:>8.3f}{r['bal_acc']:>9.3f}{r['auc']:>7.3f}")
    print(f"\n(n={r['n']} clips, {r['pos']} drowsy · chance AUC = 0.5)")


if __name__ == "__main__":
    main()
