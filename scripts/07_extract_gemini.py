#!/usr/bin/env python3
"""Step 07 — Gemini Pro multimodal baseline (video -> verdict + feature scores).

For each clip, Gemini watches the actual video+audio and returns a direct
deceptive/truthful verdict (+ probability) and a vector of 0-1 deception feature
scores. Needs GEMINI_API_KEY in .env and the google-genai SDK.

    python scripts/07_extract_gemini.py --limit 3   # smoke test
    python scripts/07_extract_gemini.py             # all clips (resumable)

Writes data/gemini/<clip_id>.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from lie_detector.config import CFG
from lie_detector.dataset import load_manifest
from lie_detector.extraction.gemini import analyze_clip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    CFG.ensure_dirs()
    if not CFG.gemini_api_key:
        sys.exit("GEMINI_API_KEY not set in .env.")
    df = load_manifest(CFG)
    if args.limit:
        df = df.head(args.limit)
    print(f"→ model={CFG.gemini_model}  clips={len(df)}")

    ok = skip = fail = 0
    for _, row in df.iterrows():
        clip_id = row["clip_id"]
        out = CFG.gemini_dir / f"{clip_id}.json"
        if out.exists() and not args.overwrite:
            skip += 1
            continue
        # retry transient API/rate-limit errors
        for attempt in range(1, 4):
            try:
                rec = analyze_clip(Path(row["video_path"]), clip_id, CFG)
                ok += 1
                p = rec["deception_probability"]
                print(f"  [{ok+fail:>3}] {clip_id}  ✓  {rec['verdict']:>9} P={p:.2f}  (truth={row['label']})")
                break
            except Exception as e:
                if attempt < 3:
                    time.sleep(5 * attempt)
                    continue
                fail += 1
                print(f"  [{ok+fail:>3}] {clip_id}  ✗  {type(e).__name__}: {str(e)[:160]}")

    print(f"\n✅ gemini done. new={ok} skipped={skip} failed={fail} -> {CFG.gemini_dir}")


if __name__ == "__main__":
    main()
