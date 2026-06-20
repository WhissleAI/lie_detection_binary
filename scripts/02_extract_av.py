#!/usr/bin/env python3
"""Step 02 — Whissle gateway extraction: STT (transcript+metadata) + audio-visual.

Calls the gateway's POST /video/analyze for each clip, which returns the
segmented transcript with emotion/intent/age/gender metadata AND the per-frame
visual timeline (face emotion/pose/gaze/blink + gestures). One call, both lanes.

Real mode (needs WHISSLE_API_TOKEN in .env; gateway at WHISSLE_GATEWAY_URL,
default http://localhost:9000):
    python scripts/02_extract_av.py --limit 2     # smoke test
    python scripts/02_extract_av.py               # all clips

Bootstrap mode (no token/gateway — builds a text-only record from the dataset's
bundled transcripts so the text+audio pipeline runs today; visual stays empty):
    python scripts/02_extract_av.py --bootstrap

Writes data/av/<clip_id>.json
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from lie_detector.config import CFG
from lie_detector.dataset import load_manifest
from lie_detector.io_utils import write_json
from lie_detector.extraction.gateway import extract_clip, refresh_asr, refresh_visual, health


def _bootstrap_record(clip_id: str, label: str) -> dict:
    """Text-only record from the dataset's bundled .txt transcript (no visual)."""
    sub = "Deceptive" if label == "deceptive" else "Truthful"
    txt = CFG.transcription_dir / sub / f"{clip_id}.txt"
    transcript = txt.read_text(encoding="utf-8", errors="ignore").strip() if txt.exists() else ""
    transcript = re.sub(r"\s+", " ", transcript)
    return {
        "clip_id": clip_id,
        "text": transcript,
        "metadata": {}, "metadata_probs": {}, "entities": [], "words": [],
        "pauses": [], "speech_rate": {}, "confidence": None, "uncertain_words": [],
        "visual_timeline": [],
        "semantic_samples": [],
        "backend": "dataset_bootstrap",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", action="store_true",
                    help="Use bundled dataset transcripts (text-only) instead of the gateway.")
    ap.add_argument("--asr-only", action="store_true",
                    help="Re-run only the ASR/text lane into existing records (keeps visual_timeline). "
                         "Use after switching ASR model.")
    ap.add_argument("--visual-only", action="store_true",
                    help="Re-run only the visual lane into existing records (keeps the text lane). "
                         "Use after improving the gateway's face detection.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    CFG.ensure_dirs()
    df = load_manifest(CFG)
    if args.limit:
        df = df.head(args.limit)

    if not args.bootstrap:
        if not CFG.whissle_api_token:
            sys.exit("No WHISSLE_API_TOKEN set. Add it to .env, or run with --bootstrap.")
        h = health(CFG)
        print(f"→ gateway {CFG.gateway_url}  /video/health -> {h['status_code']}")
        if h["status_code"] not in (200, 204):
            print(f"  ⚠  {h['body']}")

    ok = skip = fail = 0
    for _, row in df.iterrows():
        clip_id, label = row["clip_id"], row["label"]
        out = CFG.av_dir / f"{clip_id}.json"
        # --asr-only / --visual-only refresh existing records in place; otherwise skip unless --overwrite.
        if out.exists() and not args.overwrite and not args.asr_only and not args.visual_only:
            skip += 1
            continue
        try:
            if args.bootstrap:
                write_json(out, _bootstrap_record(clip_id, label))
                ok += 1
                print(f"  [{ok+fail:>3}] {clip_id}  ✓ (bootstrap)")
            elif args.asr_only:
                rec = refresh_asr(Path(row["video_path"]), clip_id, CFG)
                ok += 1
                meta = rec.get("metadata") or {}
                print(f"  [{ok+fail:>3}] {clip_id}  ✓  asr={rec.get('asr_model')} "
                      f"words={len(rec.get('words', []))} intent={meta.get('intent')}")
            elif args.visual_only:
                rec = refresh_visual(Path(row["video_path"]), clip_id, CFG)
                ok += 1
                fr = rec.get("visual_timeline", [])
                wf = sum(1 for f in fr if f.get("faces"))
                print(f"  [{ok+fail:>3}] {clip_id}  ✓  frames={len(fr)} faces={wf} "
                      f"rate={wf/len(fr) if fr else 0:.2f}")
            else:
                rec = extract_clip(Path(row["video_path"]), clip_id, CFG)
                ok += 1
                nframe = len(rec.get("visual_timeline", []))
                wf = sum(1 for f in rec["visual_timeline"] if f.get("faces"))
                emo = (rec.get("metadata") or {}).get("emotion", "?")
                print(f"  [{ok+fail:>3}] {clip_id}  ✓  words={len(rec.get('words', []))} "
                      f"frames={nframe} (faces={wf}) emo={emo}")
        except Exception as e:
            fail += 1
            print(f"  [{ok+fail:>3}] {clip_id}  ✗  {type(e).__name__}: {str(e)[:200]}")

    mode = "asr-only(whissle-large)" if args.asr_only else ("bootstrap" if args.bootstrap else "gateway")
    print(f"\n✅ AV extraction done ({mode}). new={ok} skipped={skip} failed={fail} -> {CFG.av_dir}")


if __name__ == "__main__":
    main()
