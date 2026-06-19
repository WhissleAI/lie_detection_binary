#!/usr/bin/env python3
"""Step 05 — train & evaluate deception classifiers (speaker-independent).

    python scripts/05_train.py

Runs Leave-One-Speaker-Out CV across models × modality subsets, prints a results
table, saves reports, and persists the best pipeline (refit on all clips).

Outputs:
    data/reports/cv_results.csv          model × modality → metrics (LOSO OOF)
    data/reports/feature_importance.csv  permutation importance for the best combo
    data/reports/summary.json            headline numbers
    data/models/best_model.joblib        refit best pipeline + metadata
"""

from __future__ import annotations

import json

import joblib
import pandas as pd

from lie_detector.config import CFG
from lie_detector.features.assemble import feature_columns
from lie_detector.modeling.train import (
    build_models, run_ablations, permutation_importance_report, MODALITY_SETS,
)


def main() -> None:
    CFG.ensure_dirs()
    feats_path = CFG.features_dir / "features.parquet"
    if not feats_path.exists():
        raise SystemExit("Run scripts/04_build_features.py first.")
    df = pd.read_parquet(feats_path)

    n_spk = df["speaker"].nunique()
    print(f"→ {len(df)} clips, {n_spk} speakers, "
          f"{int((df.y==1).sum())} deceptive / {int((df.y==0).sum())} truthful")
    print(f"→ Leave-One-Speaker-Out CV ({n_spk} folds)\n")

    results = run_ablations(df, CFG)
    pd.set_option("display.width", 140)
    cols = ["model", "modality", "n_features", "accuracy", "balanced_accuracy", "roc_auc", "f1"]
    print(results[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    results.to_csv(CFG.reports_dir / "cv_results.csv", index=False)

    # Best non-baseline row by AUC (fallback to balanced accuracy if AUC is NaN).
    ranked = results[results.model != "majority_baseline"].copy()
    ranked["rank_key"] = ranked["roc_auc"].fillna(ranked["balanced_accuracy"])
    best = ranked.sort_values("rank_key", ascending=False).iloc[0]
    print(f"\n★ best: {best['model']} on '{best['modality']}'  "
          f"acc={best['accuracy']:.3f}  AUC={best['roc_auc']:.3f}  f1={best['f1']:.3f}")

    # Feature importance for the best model+modality, refit on all data.
    prefixes = MODALITY_SETS[best["modality"]]
    model = build_models(CFG.random_seed)[best["model"]]
    try:
        imp = permutation_importance_report(df, model, prefixes, CFG)
        imp.to_csv(CFG.reports_dir / "feature_importance.csv", index=False)
        print("\nTop features (permutation importance, AUC drop):")
        print(imp.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    except Exception as e:
        print(f"(feature importance skipped: {type(e).__name__}: {e})")

    # Persist the refit best pipeline + metadata.
    model.fit(df[feature_columns(df, prefixes)], df["y"].to_numpy())
    joblib.dump({"pipeline": model, "modality": best["modality"], "prefixes": list(prefixes),
                 "features": feature_columns(df, prefixes), "metrics": best[cols].to_dict()},
                CFG.models_dir / "best_model.joblib")

    summary = {
        "n_clips": int(len(df)), "n_speakers": int(n_spk),
        "cv": "leave-one-speaker-out",
        "best_model": str(best["model"]), "best_modality": str(best["modality"]),
        "best_metrics": {k: (None if pd.isna(best[k]) else float(best[k]))
                         for k in ["accuracy", "balanced_accuracy", "roc_auc", "f1", "precision", "recall"]},
    }
    (CFG.reports_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n✅ reports -> {CFG.reports_dir}   model -> {CFG.models_dir/'best_model.joblib'}")


if __name__ == "__main__":
    main()
