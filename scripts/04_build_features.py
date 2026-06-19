#!/usr/bin/env python3
"""Step 04 — assemble the multimodal feature matrix (one row per clip).

    python scripts/04_build_features.py

Reads data/av/*.json (text + visual) + data/audio/*.json (prosody) + manifest.
Writes data/features/features.parquet (+ a .csv copy for inspection).
"""

from __future__ import annotations

from lie_detector.config import CFG
from lie_detector.features.assemble import build_feature_matrix, feature_columns


def main() -> None:
    CFG.ensure_dirs()
    df = build_feature_matrix(CFG)

    out_parquet = CFG.features_dir / "features.parquet"
    out_csv = CFG.features_dir / "features.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    def cov(p):
        return int(df.get(f"{p}_present", 0).sum()) if f"{p}_present" in df else 0

    print(f"✅ features written: {out_parquet}")
    print(f"   rows (clips):      {len(df)}")
    print(f"   modality coverage: txt={cov('txt')}  vis={cov('vis')}  aud={cov('aud')}  (of {len(df)})")
    for m in ("txt", "vis", "aud"):
        print(f"   {m}_* feature count: {len(feature_columns(df, [m]))}")
    print(f"   total numeric features: {len(feature_columns(df))}")


if __name__ == "__main__":
    main()
