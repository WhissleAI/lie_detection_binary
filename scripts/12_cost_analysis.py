#!/usr/bin/env python3
"""Step 12 — token-cost analysis: feature digest vs raw video.

Our "no video to the cloud LLM" design sends a compact text feature digest
instead of the heavy video. This script measures, for a sample of clips, the
LLM input tokens for (a) our digest vs (b) the raw video, using the providers'
own count_tokens APIs, and estimates the cost/latency saving.

    python scripts/12_cost_analysis.py --n 8
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from lie_detector.config import CFG
from lie_detector.dataset import load_manifest
from lie_detector.io_utils import read_json, write_json
from lie_detector.extraction.gemini_reason import build_multimodal_summary

# Public list prices ($ / 1M input tokens), stated assumptions for the estimate.
PRICE_PER_M = {"gemini-2.5-pro": 1.25, "claude-opus-4-8": 15.0}


def gemini_count(client, model, contents) -> int:
    return int(client.models.count_tokens(model=model, contents=contents).total_tokens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="Number of clips to sample.")
    args = ap.parse_args()

    from google import genai
    client = genai.Client(api_key=CFG.gemini_api_key)

    df = load_manifest(CFG)
    # spread the sample across the manifest
    step = max(1, len(df) // args.n)
    sample = df.iloc[::step].head(args.n)

    digest_toks, video_toks, durations = [], [], []
    for _, row in sample.iterrows():
        cid = row["clip_id"]
        av = read_json(CFG.av_dir / f"{cid}.json")
        au_p = CFG.audio_dir / f"{cid}.json"
        au = read_json(au_p) if au_p.exists() else {}
        digest = build_multimodal_summary(av, au)
        dt = gemini_count(client, CFG.gemini_model, [digest])

        f = client.files.upload(file=row["video_path"])
        import time
        while getattr(f.state, "name", str(f.state)) == "PROCESSING":
            time.sleep(1.5); f = client.files.get(name=f.name)
        vt = gemini_count(client, CFG.gemini_model, [f])
        try:
            client.files.delete(name=f.name)
        except Exception:
            pass

        dur = float((av.get("video_params") or {}).get("duration", "0").rstrip("s") or 0)
        digest_toks.append(dt); video_toks.append(vt); durations.append(dur)
        print(f"  {cid}: digest={dt:>5} tok | video={vt:>6} tok | {dur:.0f}s | ratio×{vt/dt:.1f}")

    md, mv = statistics.mean(digest_toks), statistics.mean(video_toks)
    price = PRICE_PER_M.get(CFG.gemini_model, 1.25)
    summary = {
        "n": len(digest_toks),
        "mean_digest_tokens": round(md, 1),
        "mean_video_tokens": round(mv, 1),
        "token_reduction_x": round(mv / md, 1),
        "mean_clip_sec": round(statistics.mean(durations), 1),
        "video_tokens_per_sec": round(mv / max(1e-9, statistics.mean(durations)), 1),
        "input_cost_per_1k_clips_digest_usd": round(md * 1000 / 1e6 * price, 3),
        "input_cost_per_1k_clips_video_usd": round(mv * 1000 / 1e6 * price, 3),
        "price_assumption_per_M_tokens_usd": price,
        "model": CFG.gemini_model,
    }
    print("\n=== TOKEN-COST SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n→ Our digest uses ~{summary['token_reduction_x']}x fewer input tokens than the raw video.")
    CFG.ensure_dirs()
    write_json(CFG.reports_dir / "cost_analysis.json", summary)
    print(f"saved -> {CFG.reports_dir/'cost_analysis.json'}")


if __name__ == "__main__":
    main()
