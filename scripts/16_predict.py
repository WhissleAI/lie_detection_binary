#!/usr/bin/env python3
"""Step 16 — inference: predict deceptive/truthful for one clip, every method.

Runs the saved best trained model on a clip's features and (optionally) the LLM
methods, side by side. Works on a dataset clip_id (uses cached records) or a new
video file (extracts features first).

    python scripts/16_predict.py trial_lie_001              # trained model only
    python scripts/16_predict.py trial_lie_001 --llm        # + Claude/Gemini judges + Gemini video
    python scripts/16_predict.py /path/to/clip.mp4 --llm    # new clip: extract then predict

Trained model: data/models/best_model.joblib (run scripts/05_train.py first).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import joblib
import pandas as pd

from lie_detector.config import CFG
from lie_detector.io_utils import read_json
from lie_detector.features.text_features import text_features, collect_metadata_vocab
from lie_detector.features.visual_features import visual_features
from lie_detector.features.audio_features import audio_features


def _load_or_extract(target: str) -> tuple[str, dict, dict]:
    """Return (clip_id, av_record, audio_record), extracting if a new video path."""
    p = Path(target)
    if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"} and p.exists():
        clip_id = p.stem
        from lie_detector.extraction.gateway import extract_clip
        from lie_detector.extraction.audio_prosody import extract_audio_features
        print(f"→ extracting features for new clip {clip_id} (gateway + prosody)...")
        av = extract_clip(p, clip_id, CFG)
        audio = extract_audio_features(p, clip_id, CFG)
        return clip_id, av, audio
    clip_id = p.stem if p.suffix else target
    av_p = CFG.av_dir / f"{clip_id}.json"
    if not av_p.exists():
        raise SystemExit(f"No av record for '{clip_id}'. Pass a video path, or run extraction.")
    audio_p = CFG.audio_dir / f"{clip_id}.json"
    return clip_id, read_json(av_p), (read_json(audio_p) if audio_p.exists() else {})


def _featurize(av: dict, audio: dict) -> pd.DataFrame:
    """Build a 1-row feature frame, with metadata-prob vocab matching the corpus."""
    corpus = [read_json(f) for f in glob.glob(str(CFG.av_dir / "*.json"))]
    vocab = collect_metadata_vocab(corpus)
    row = {}
    row.update({f"txt_{k}": v for k, v in text_features(av, vocab).items()})
    row.update({f"vis_{k}": v for k, v in visual_features(av).items()})
    if audio:
        row.update({f"aud_{k}": v for k, v in audio_features(audio).items()})
    return pd.DataFrame([row])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", help="dataset clip_id OR path to a video file")
    ap.add_argument("--llm", action="store_true", help="also run the LLM methods (needs keys).")
    args = ap.parse_args()

    clip_id, av, audio = _load_or_extract(args.clip)
    truth = ("deceptive" if "lie" in clip_id else "truthful" if "truth" in clip_id else "?")
    print(f"\n=== {clip_id}  (ground truth: {truth}) ===")
    print(f'transcript: "{(av.get("text") or "")[:160]}..."\n')

    results = []

    # --- trained model -----------------------------------------------------
    model_path = CFG.models_dir / "best_model.joblib"
    if model_path.exists():
        m = joblib.load(model_path)
        X = _featurize(av, audio).reindex(columns=m["features"])  # NaN -> pipeline imputes
        prob = float(m["pipeline"].predict_proba(X)[0, 1])
        results.append((f"trained {m['metrics']['model']} ({m['modality']})", prob, "no"))
    else:
        print("(no trained model — run scripts/05_train.py)")

    # --- LLM methods -------------------------------------------------------
    if args.llm:
        from lie_detector.extraction.gemini_reason import reason_over_features
        if CFG.anthropic_api_key:
            try:
                r = reason_over_features(av, audio, clip_id, CFG, provider="claude")
                results.append(("Claude Opus 4.8 (judge over features)", r["deception_probability"], "no"))
            except Exception as e:
                print(f"  claude failed: {e}")
        if CFG.gemini_api_key:
            try:
                r = reason_over_features(av, audio, clip_id, CFG, provider="gemini")
                results.append(("Gemini 2.5 Pro (judge over features)", r["deception_probability"], "no"))
            except Exception as e:
                print(f"  gemini-judge failed: {e}")
            vid = CFG.clips_dir / ("Deceptive" if truth == "deceptive" else "Truthful") / f"{clip_id}.mp4"
            if vid.exists():
                try:
                    from lie_detector.extraction.gemini import analyze_clip
                    r = analyze_clip(vid, clip_id, CFG)
                    results.append(("Gemini 2.5 Pro (watches raw video)", r["deception_probability"], "YES"))
                except Exception as e:
                    print(f"  gemini-video failed: {e}")

    # --- report ------------------------------------------------------------
    print(f"{'method':<42}{'P(deceptive)':>13}{'verdict':>12}{'video?':>8}")
    print("-" * 75)
    for name, prob, vid_sent in results:
        verdict = "DECEPTIVE" if prob >= 0.5 else "truthful"
        print(f"{name:<42}{prob:>13.2f}{verdict:>12}{vid_sent:>8}")


if __name__ == "__main__":
    main()
