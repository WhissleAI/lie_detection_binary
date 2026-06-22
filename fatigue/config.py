"""Fatigue-detection track — config.

Reuses the deception pipeline's gateway/feature code (``lie_detector``) but points
data at the external SSD and uses the UTA-RLDD drowsiness labels. Same Whissle
gateway, same ``whissle-large`` ASR + ``/video/analyze`` visual timeline.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

# Reuse the gateway/model config (URL, token, whissle-large, top_k, speech_analysis)
# but redirect all derived data dirs (wav/av/audio) to the external SSD.
from lie_detector.config import CFG as _BASE

# UTA-RLDD lives on the external SSD; keep all derived data there too.
DATA_ROOT = Path("/Volumes/Extreme Pro/fatigue_datasets/uta_rldd")
WORK = Path("/Volumes/Extreme Pro/fatigue_datasets/work")
VIDEOS = WORK / "videos"     # unzipped UTA-RLDD videos
CLIPS = WORK / "clips"       # windowed clips fed to the gateway
REPORTS = WORK / "reports"

GATEWAY = replace(_BASE, data_dir=WORK)   # wav/av/audio dirs now under WORK on SSD
AV = GATEWAY.av_dir          # /asr/transcribe + /video/analyze records
AUDIO = GATEWAY.audio_dir    # librosa prosody records

# UTA-RLDD state code -> label.  Files are <subject>/{0,5,10}.{mp4,mov,MOV}
STATE_LABEL = {"0": "alert", "5": "low_vigilant", "10": "drowsy"}
# Binary contrast for the headline detector (drop the ambiguous middle class).
BINARY = {"alert": 0, "drowsy": 1}

# Each UTA-RLDD video is ~10 min; sample one window to keep gateway cost bounded.
WINDOW_START = 150           # skip the first 2.5 min (camera/settling)
WINDOW_SEC = 120             # 2-minute window — enough blinks/yawns for drowsiness


def ensure_dirs() -> None:
    for d in (WORK, VIDEOS, CLIPS, AV, AUDIO, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
