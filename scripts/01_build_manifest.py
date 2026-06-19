#!/usr/bin/env python3
"""Step 01 — build the dataset manifest (clip, label, speaker group).

    python scripts/01_build_manifest.py

Writes data/manifest.csv. No tokens or network needed.
"""

from __future__ import annotations

from lie_detector.config import CFG
from lie_detector.dataset import build_manifest


def main() -> None:
    CFG.ensure_dirs()
    df = build_manifest(CFG)
    df.to_csv(CFG.manifest_csv, index=False)

    n = len(df)
    n_dec = int((df.y == 1).sum())
    n_tru = int((df.y == 0).sum())
    n_spk = df.speaker.nunique()
    multi = (df.speaker.value_counts() > 1).sum()
    print(f"✅ manifest written: {CFG.manifest_csv}")
    print(f"   clips:            {n}  (deceptive={n_dec}, truthful={n_tru})")
    print(f"   unique speakers:  {n_spk}  ({multi} appear in >1 clip)")
    # Speakers that appear under BOTH labels — the hardest leakage case.
    both = (
        df.groupby("speaker")["label"].nunique()
        .pipe(lambda s: s[s > 1]).index.tolist()
    )
    if both:
        print(f"   speakers in BOTH classes: {both}")
    print("\n   top speakers by clip count:")
    print(df.speaker.value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
