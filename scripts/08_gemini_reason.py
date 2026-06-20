#!/usr/bin/env python3
"""Step 08 — Gemini reasoning over our EXTRACTED features (not raw video).

Builds a time-synced multimodal summary (transcript + STT metadata probs +
prosody + visual timeline aligned to speech) from data/av + data/audio, sends it
to Gemini, and stores both the input summary and the verdict.

    python scripts/08_gemini_reason.py --limit 3
    python scripts/08_gemini_reason.py --neutral --workers 12

Needs the av + audio extraction done first. Writes data/gemini_reason[_v2]/<clip>.json
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from lie_detector.config import CFG
from lie_detector.dataset import load_manifest
from lie_detector.io_utils import read_json
from lie_detector.extraction.gemini_reason import reason_over_features

_print_lock = threading.Lock()


def _process(clip_id: str, label: str, neutral: bool, provider: str, samples: int):
    """Run one clip with retries (incl. 429 backoff). Returns (status, msg)."""
    av_p = CFG.av_dir / f"{clip_id}.json"
    if not av_p.exists():
        return "fail", f"{clip_id}  ✗  missing av record (run 02 first)"
    av = read_json(av_p)
    au_p = CFG.audio_dir / f"{clip_id}.json"
    audio = read_json(au_p) if au_p.exists() else {}
    for attempt in range(1, 5):
        try:
            rec = reason_over_features(av, audio, clip_id, CFG, neutral=neutral,
                                       provider=provider, n_samples=samples)
            return "ok", (f"{clip_id}  ✓  {rec['verdict']:>9} "
                          f"P={rec['deception_probability']:.2f}  (truth={label})")
        except Exception as e:
            msg = str(e)
            transient = "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "timeout" in msg.lower()
            if attempt < 4 and transient:
                time.sleep(5 * attempt)
                continue
            return "fail", f"{clip_id}  ✗  {type(e).__name__}: {msg[:160]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--neutral", action="store_true",
                    help="Use the debiased neutral prompt; writes to gemini_reason_v2/ (or claude_reason/).")
    ap.add_argument("--provider", choices=["gemini", "claude"], default="gemini",
                    help="LLM-as-judge provider.")
    ap.add_argument("--samples", type=int, default=1,
                    help="Self-consistency: sample N times at temperature and average.")
    ap.add_argument("--workers", type=int, default=12, help="Concurrent requests.")
    args = ap.parse_args()
    if args.provider == "claude":
        out_dir = CFG.claude_reason_dir
    else:
        out_dir = CFG.gemini_reason_v2_dir if args.neutral else CFG.gemini_reason_dir

    CFG.ensure_dirs()
    if args.provider == "claude" and not CFG.anthropic_api_key:
        sys.exit("ANTHROPIC_API_KEY not set in .env (needed for --provider claude).")
    if args.provider == "gemini" and not CFG.gemini_api_key:
        sys.exit("GEMINI_API_KEY not set in .env.")
    df = load_manifest(CFG)
    if args.limit:
        df = df.head(args.limit)

    todo = [(r["clip_id"], r["label"]) for _, r in df.iterrows()
            if args.overwrite or not (out_dir / f"{r['clip_id']}.json").exists()]
    skip = len(df) - len(todo)
    model = CFG.anthropic_model if args.provider == "claude" else CFG.gemini_model
    print(f"→ provider={args.provider} model={model} prompt=neutral_v3 samples={args.samples} "
          f"todo={len(todo)} skip={skip} workers={args.workers}")

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process, cid, lbl, args.neutral, args.provider, args.samples): cid
                for cid, lbl in todo}
        for fut in as_completed(futs):
            status, msg = fut.result()
            if status == "ok":
                ok += 1
            else:
                fail += 1
            with _print_lock:
                print(f"  [{ok+fail:>3}/{len(todo)}] {msg}")

    print(f"\n✅ LLM-judge done ({args.provider}). new={ok} skipped={skip} failed={fail} -> {out_dir}")


if __name__ == "__main__":
    main()
