"""Extract a windowed clip per UTA-RLDD video and run the gateway on it.

Reuses the deception pipeline's ``extract_clip`` (whissle-large STT + metadata +
``/video/analyze`` visual timeline) and ``extract_audio_features`` (prosody).
Resumable: skips clips whose av record already exists.

    python -m fatigue.extract [--limit N] [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import subprocess

from . import config as C
from .manifest import build_manifest
from lie_detector.extraction.gateway import extract_clip
from lie_detector.extraction.audio_prosody import extract_audio_features


def _window(video: str, clip_id: str):
    out = C.CLIPS / f"{clip_id}.mp4"
    if out.exists() and out.stat().st_size > 1000:
        return out
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(C.WINDOW_START),
         "-t", str(C.WINDOW_SEC), "-i", video,
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-ac", "1", "-ar", "16000", str(out)],
        check=True,
    )
    return out


def main(limit=None, overwrite=False) -> None:
    C.ensure_dirs()
    rows = build_manifest()
    if limit:
        rows = rows[:limit]
    done = fail = 0
    for i, r in enumerate(rows, 1):
        cid = r["clip_id"]
        av_path = C.AV / f"{cid}.json"
        if av_path.exists() and not overwrite:
            print(f"[{i}/{len(rows)}] {cid} cached"); done += 1; continue
        try:
            clip = _window(r["video"], cid)
            av = extract_clip(clip, cid, C.GATEWAY)
            av.update(subject=r["subject"], label=r["label"], state=r["state"])
            av_path.write_text(json.dumps(av))
            audio = extract_audio_features(clip, cid, C.GATEWAY)
            (C.AUDIO / f"{cid}.json").write_text(json.dumps(audio))
            frames = len(av.get("visual_timeline", []))
            print(f"[{i}/{len(rows)}] {cid} ✓  {r['label']:<13} visual_frames={frames}")
            done += 1
        except Exception as e:
            print(f"[{i}/{len(rows)}] {cid} FAIL: {str(e)[:140]}"); fail += 1
    print(f"\ndone={done}  fail={fail}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    main(a.limit, a.overwrite)
