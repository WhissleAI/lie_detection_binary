"""Media helpers built on ffmpeg/ffprobe: extract audio, sample frames, probe.

We shell out to ffmpeg (already a project prerequisite) for robust audio
decoding, and use OpenCV for frame sampling in the visual lane.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise RuntimeError(
            f"'{tool}' not found on PATH. Install ffmpeg (brew install ffmpeg / apt install ffmpeg)."
        )
    return path


def probe_duration(video_path: Path) -> float:
    """Return media duration in seconds (0.0 if unknown)."""
    ffprobe = _require("ffprobe")
    out = subprocess.run(
        [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", str(video_path)],
        capture_output=True, text=True, check=False,
    )
    try:
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def has_audio_stream(video_path: Path) -> bool:
    ffprobe = _require("ffprobe")
    out = subprocess.run(
        [ffprobe, "-v", "quiet", "-select_streams", "a", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True, check=False,
    )
    return "audio" in out.stdout


def extract_wav(video_path: Path, out_wav: Path, sample_rate: int = 16000, mono: bool = True) -> Path:
    """Decode the audio track to a 16 kHz mono PCM WAV (what Whissle/librosa want)."""
    ffmpeg = _require("ffmpeg")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", str(sample_rate),
        "-ac", "1" if mono else "2", str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out_wav.exists():
        raise RuntimeError(f"ffmpeg failed for {video_path.name}:\n{proc.stderr[-800:]}")
    return out_wav
