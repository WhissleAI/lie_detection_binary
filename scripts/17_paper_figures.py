#!/usr/bin/env python3
"""Step 17 — generate the paper's data figures (Whissle-styled).

Produces publication figures embedded in the paper/blog:
  fig_dataset.png   — class split + clips-per-speaker (the leakage outlier)
  fig_results.png   — ROC-AUC of the six systems, coloured by 'video sent?'
  fig_leakage.png   — video-out vs speaker-out per feature set (the leakage gap)
  fig_pipeline.png  — the on-device pipeline diagram

    python scripts/17_paper_figures.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

from lie_detector.config import CFG
from lie_detector.dataset import load_manifest

GREEN = "#124E3F"
GREEN_L = "#2E8B6F"
RED = "#C0392B"
GREY = "#9aa3a0"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#444", "savefig.dpi": 200})
FIG = CFG.project_root / "paper" / "figures"


def dataset_fig():
    man = load_manifest(CFG)
    counts = man.speaker.value_counts().sort_values(ascending=False)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 2.2]})
    a.bar(["deceptive", "truthful"], [int((man.y == 1).sum()), int((man.y == 0).sum())],
          color=[RED, GREEN], edgecolor="black", linewidth=0.5, width=0.6)
    a.set_ylabel("clips"); a.set_title("Class balance", fontweight="bold")
    for i, v in enumerate([int((man.y == 1).sum()), int((man.y == 0).sum())]):
        a.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
    colors = [RED if c == counts.max() else GREEN_L for c in counts.values]
    b.bar(range(len(counts)), counts.values, color=colors, edgecolor="black", linewidth=0.3)
    b.set_title("Clips per speaker (33 speakers)", fontweight="bold")
    b.set_xlabel("speaker (sorted)"); b.set_ylabel("# clips")
    b.annotate(f"one defendant:\n{counts.max()} clips", xy=(0, counts.max()),
               xytext=(6, counts.max() - 4), fontsize=10, color=RED, fontweight="bold",
               arrowprops=dict(arrowstyle="->", color=RED))
    fig.suptitle("Real-life Trial Deception: 121 clips, few speakers", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig_dataset.png", bbox_inches="tight"); plt.close(fig)


def results_fig():
    rows = [
        ("Majority baseline", 0.500, "base"),
        ("Self-hosted → gradient-boosting", 0.741, "no"),
        ("Claude Opus 4.8 over features", 0.755, "no"),
        ("Gemini 2.5 Pro over features", 0.704, "no"),
        ("Gemini 2.5 Pro over raw video", 0.749, "yes"),
        ("Self-hosted + LLM (fusion)", 0.752, "yes"),
    ]
    rows = rows[::-1]
    cmap = {"no": GREEN, "yes": RED, "base": GREY}
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bars = ax.barh(range(len(rows)), [r[1] for r in rows],
                   color=[cmap[r[2]] for r in rows], edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.axvline(0.5, ls="--", color="#777", lw=1)
    ax.set_xlim(0.45, 0.80); ax.set_xlabel("ROC-AUC (leave-one-speaker-out)")
    ax.set_title("Honest deception-detection performance", fontsize=13, fontweight="bold")
    for i, r in enumerate(rows):
        ax.text(r[1] + 0.004, i, f"{r[1]:.3f}", va="center", fontsize=9.5, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREEN, label="no video sent"),
                       Patch(color=RED, label="video sent to LLM"),
                       Patch(color=GREY, label="baseline")],
              loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout(); fig.savefig(FIG / "fig_results.png", bbox_inches="tight"); plt.close(fig)


def leakage_fig():
    df = pd.read_csv(CFG.reports_dir / "paper_comparison.csv")
    df = df[df.model == "random_forest"].copy()
    df = df[df.feature_set.isin(["our_text (auto)", "our_visual (auto)", "our_all (auto)",
                                 "manual_gestures (paper)", "gemini_features (LLM)"])]
    labels = [f.split(" (")[0].replace("our_", "") for f in df.feature_set]
    import numpy as np
    x = np.arange(len(df)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.bar(x - w / 2, df.acc_leave_one_video_out, w, label="leave-one-video-out (leaky)",
           color=RED, edgecolor="black", linewidth=0.3)
    ax.bar(x + w / 2, df.acc_leave_one_speaker_out, w, label="leave-one-speaker-out (honest)",
           color=GREEN, edgecolor="black", linewidth=0.3)
    ax.axhline(0.5, ls="--", color="#777", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("accuracy"); ax.set_ylim(0.4, 0.85)
    ax.set_title("The speaker-leakage gap", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, frameon=True)
    fig.tight_layout(); fig.savefig(FIG / "fig_leakage.png", bbox_inches="tight"); plt.close(fig)


def pipeline_fig():
    fig, ax = plt.subplots(figsize=(9.5, 3.4)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.55)

    def box(x, y, w, h, text, fc, tc="white", fs=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=fc, ec="black", lw=0.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
                fontweight="bold", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                     color="#333", lw=1.2))

    box(0.1, 1.5, 1.5, 1.0, "clip\n(.mp4)", GREEN)
    lanes = [("Whissle STT\ntext + metadata", 3.0), ("Audio-visual\nface/gaze/pose", 1.7),
             ("Prosody\nF0 / pauses", 0.4)]
    for txt, y in lanes:
        box(2.2, y, 1.9, 0.95, txt, GREEN_L); arrow(1.6, 2.0, 2.2, y + 0.5)
    box(4.7, 1.5, 1.6, 1.0, "feature\ndigest\n(~250 nums)", GREEN)
    for _, y in lanes:
        arrow(4.1, y + 0.5, 4.7, 2.0)
    box(7.0, 2.3, 2.6, 0.9, "trained classifier\n(no LLM)", "#1F8A70")
    box(7.0, 0.9, 2.6, 0.9, "LLM-as-judge\n(no video sent)", "#1F8A70")
    arrow(6.3, 2.1, 7.0, 2.75); arrow(6.3, 1.9, 7.0, 1.35)
    ax.text(5.0, 4.25, "raw video never leaves the device", ha="center", fontsize=10,
            color=RED, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig_pipeline.png", bbox_inches="tight"); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    dataset_fig(); results_fig(); leakage_fig(); pipeline_fig()
    print(f"✓ wrote dataset / results / leakage / pipeline figures -> {FIG}")


if __name__ == "__main__":
    main()
