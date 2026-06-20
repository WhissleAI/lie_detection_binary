#!/usr/bin/env python3
"""Step 15 — which speech INTENTS predict deception?

For every intent feature (Whissle's 33-way intent distribution `metaprob_intent_*`
and the focused deception-intent filter `fintent_*`), measure how it separates
deceptive from truthful clips: point-biserial correlation with the label, the
mean probability in each class, and significance. Render a publication-quality
diverging-bar figure.

    python scripts/15_intent_analysis.py

Outputs:
    data/reports/intent_analysis.csv
    paper/figures/intent_importance.png  (+ .pdf)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lie_detector.config import CFG

TOP_N = 16


def main() -> None:
    df = pd.read_parquet(CFG.features_dir / "features.parquet")
    y = df["y"].to_numpy()  # 1 = deceptive, 0 = truthful

    intent_cols = [c for c in df.columns
                   if (("metaprob_intent_" in c) or ("_fintent_" in c))
                   and not c.endswith("_entropy")
                   and pd.api.types.is_numeric_dtype(df[c])
                   and df[c].nunique(dropna=True) > 1]

    rows = []
    for c in intent_cols:
        x = df[c].fillna(0.0).to_numpy()
        r, p = pointbiserialr(y, x)
        source = "deception-filter" if "_fintent_" in c else "STT-intent"
        label = c.split("intent_")[-1].split("fintent_")[-1].replace("_", " ")
        rows.append({
            "intent": label, "source": source, "feature": c,
            "mean_deceptive": round(float(x[y == 1].mean()), 4),
            "mean_truthful": round(float(x[y == 0].mean()), 4),
            "diff_dec_minus_truth": round(float(x[y == 1].mean() - x[y == 0].mean()), 4),
            "pointbiserial_r": round(float(r), 4),
            "p_value": round(float(p), 4),
        })
    res = pd.DataFrame(rows).dropna(subset=["pointbiserial_r"])
    res["abs_r"] = res["pointbiserial_r"].abs()
    res = res.sort_values("abs_r", ascending=False).reset_index(drop=True)
    CFG.ensure_dirs()
    res.drop(columns="abs_r").to_csv(CFG.reports_dir / "intent_analysis.csv", index=False)

    print("Top intents by |point-biserial r| with deception (r>0 = deceptive-leaning):")
    print(res.head(TOP_N)[["intent", "source", "mean_deceptive", "mean_truthful",
                           "pointbiserial_r", "p_value"]].to_string(index=False))

    # ---- scientific diverging-bar figure ----------------------------------
    top = res.head(TOP_N).iloc[::-1]  # smallest |r| at top so largest is at the top visually
    colors = ["#C0392B" if r > 0 else "#1F8A70" for r in top["pointbiserial_r"]]
    labels = [f"{i}  ({s[0]})" for i, s in zip(top["intent"], top["source"])]

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    bars = ax.barh(range(len(top)), top["pointbiserial_r"], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("point-biserial correlation with deception  (r)", fontsize=10)
    ax.set_title("Which speech intents predict deception?", fontsize=13, fontweight="bold")
    # significance stars
    for i, (r, p) in enumerate(zip(top["pointbiserial_r"], top["p_value"])):
        star = "*" if p < 0.05 else ""
        if star:
            ax.text(r + (0.005 if r >= 0 else -0.005), i, star, va="center",
                    ha="left" if r >= 0 else "right", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    # legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#C0392B", label="deceptive-leaning"),
                       Patch(color="#1F8A70", label="truthful-leaning"),
                       Patch(color="none", label="* p < 0.05")],
              loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.5, 0.015, "(S) = Whissle 33-way STT intent distribution   ·   "
             "(d) = focused deception-intent filter   ·   n = 121 clips, leave-one-speaker-out set",
             ha="center", fontsize=7.5, color="#555")

    figdir = CFG.project_root / "paper" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "intent_importance.png", dpi=200, bbox_inches="tight")
    fig.savefig(figdir / "intent_importance.pdf", bbox_inches="tight")
    print(f"\nsaved -> {figdir/'intent_importance.png'}  +  {CFG.reports_dir/'intent_analysis.csv'}")


if __name__ == "__main__":
    main()
