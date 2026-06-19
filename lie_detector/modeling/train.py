"""Speaker-independent training, evaluation, and ablations.

Core methodology
----------------
We evaluate every model with **Leave-One-Speaker-Out** cross-validation
(``LeaveOneGroupOut`` grouped by speaker). All clips from a given person live in
the same fold, so the model can never see a speaker at train *and* test time —
the only honest way to estimate deception-detection generalisation on this
dataset (a random split leaks speaker identity and massively inflates accuracy).

Out-of-fold predictions are pooled across folds, then scored once, so AUC/F1 are
computed on held-out predictions for all 121 clips.

We sweep several classifiers across modality subsets (text / visual / audio and
their unions) so the contribution of each lane — and of fusion — is explicit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..config import CFG
from ..features.assemble import feature_columns
from .metrics import compute_metrics

MODALITY_SETS = {
    "text": ("txt",),
    "visual": ("vis",),
    "audio": ("aud",),
    "text+audio": ("txt", "aud"),
    "text+visual": ("txt", "vis"),
    "visual+audio": ("vis", "aud"),
    "all": ("txt", "vis", "aud"),
}


def build_models(seed: int = CFG.random_seed) -> dict[str, Pipeline]:
    """Each pipeline imputes (median) then optionally scales, then classifies."""
    def linear(clf):
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()), ("clf", clf)])

    def tree(clf):
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", clf)])

    return {
        "logreg": linear(LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=seed)),
        "svm_rbf": linear(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        "random_forest": tree(RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1)),
        "hist_gbm": tree(HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300, random_state=seed)),
    }


def evaluate_cv(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, model: Pipeline) -> dict:
    """Leave-One-Speaker-Out out-of-fold metrics."""
    logo = LeaveOneGroupOut()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = cross_val_predict(model, X, y, groups=groups, cv=logo, n_jobs=-1)
        try:
            y_prob = cross_val_predict(model, X, y, groups=groups, cv=logo,
                                       method="predict_proba", n_jobs=-1)[:, 1]
        except Exception:
            y_prob = None
    return compute_metrics(y, y_pred, y_prob)


def majority_baseline(y: np.ndarray) -> dict:
    maj = int(round(y.mean()))
    return compute_metrics(y, np.full_like(y, maj), np.full(len(y), y.mean()))


def run_ablations(df: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    """Sweep models × modality subsets under LOSO CV; return a tidy results table."""
    y = df["y"].to_numpy()
    groups = df["speaker"].to_numpy()
    models = build_models(cfg.random_seed)

    rows = [{"model": "majority_baseline", "modality": "—", "n_features": 0,
             **majority_baseline(y)}]

    for mod_name, prefixes in MODALITY_SETS.items():
        cols = feature_columns(df, prefixes)
        if not cols:
            continue
        X = df[cols]
        # skip a modality that is entirely empty/constant (e.g. visual in bootstrap)
        if X.nunique(dropna=True).sum() == 0:
            continue
        for model_name, model in models.items():
            m = evaluate_cv(X, y, groups, model)
            rows.append({"model": model_name, "modality": mod_name,
                         "n_features": len(cols), **m})

    res = pd.DataFrame(rows)
    return res.sort_values(["roc_auc", "accuracy"], ascending=False, na_position="last").reset_index(drop=True)


def permutation_importance_report(df: pd.DataFrame, model: Pipeline, prefixes, cfg=CFG, top_k: int = 25) -> pd.DataFrame:
    """Permutation importance (grouped-CV-fit refit on all data) for one model."""
    from sklearn.inspection import permutation_importance

    cols = feature_columns(df, prefixes)
    X, y = df[cols], df["y"].to_numpy()
    model.fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = permutation_importance(model, X, y, n_repeats=20,
                                   random_state=cfg.random_seed, scoring="roc_auc", n_jobs=-1)
    imp = pd.DataFrame({"feature": cols,
                        "importance_mean": r.importances_mean,
                        "importance_std": r.importances_std})
    return imp.sort_values("importance_mean", ascending=False).head(top_k).reset_index(drop=True)
