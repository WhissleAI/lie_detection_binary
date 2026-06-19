#!/usr/bin/env python3
"""Step 04 — acoustic / prosodic feature extraction (librosa).

    python scripts/04_extract_audio.py
    python scripts/04_extract_audio.py --limit 2

No tokens needed. Writes data/audio/<clip_id>.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lie_detector.config import CFG
from lie_detector.dataset import load_manifest
from lie_detector.extraction.audio_prosody import extract_audio_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    CFG.ensure_dirs()
    df = load_manifest(CFG)
    if args.limit:
        df = df.head(args.limit)

    ok = skip = fail = 0
    for _, row in df.iterrows():
        clip_id = row["clip_id"]
        out = CFG.audio_dir / f"{clip_id}.json"
        if out.exists() and not args.overwrite:
            skip += 1
            continue
        try:
            f = extract_audio_features(Path(row["video_path"]), clip_id, CFG)
            ok += 1
            print(f"  [{ok+fail:>3}] {clip_id}  ✓  f0={f['f0_mean']:.0f}Hz pauses={f['n_pauses']:.0f}")
        except Exception as e:
            fail += 1
            print(f"  [{ok+fail:>3}] {clip_id}  ✗  {type(e).__name__}: {str(e)[:160]}")

    print(f"\n✅ audio done. new={ok} skipped={skip} failed={fail} -> {CFG.audio_dir}")


if __name__ == "__main__":
    main()
