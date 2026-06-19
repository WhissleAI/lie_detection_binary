"""Step 04 — Acoustic / prosodic features from the clip audio (librosa).

Prosody is a classic deception signal: liars often show altered pitch, more/
longer pauses, vocal tension, and reduced fluency. We extract a compact, robust
set from the 16 kHz mono WAV produced in the STT step (or freshly decoded):

    pitch (F0):     mean, std, range, voiced fraction, jitter proxy
    loudness (RMS): mean, std, shimmer proxy
    pauses:         silence ratio, #pauses, mean pause length
    voice quality:  ZCR, spectral centroid/bandwidth/rolloff (mean+std)

Output per clip (data/audio/<clip_id>.json): a flat dict of float features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import CFG
from ..media import extract_wav
from ..io_utils import write_json

SR = 16000
FRAME = 2048
HOP = 512


def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def _pause_stats(rms: np.ndarray, hop: int, sr: int) -> dict:
    """Silence/pause statistics from a frame-level RMS envelope."""
    if rms.size == 0:
        return {"silence_ratio": 0.0, "n_pauses": 0.0, "mean_pause_s": 0.0, "speech_ratio": 0.0}
    # Adaptive threshold: 15% of the median voiced energy.
    thr = 0.15 * np.median(rms[rms > 0]) if np.any(rms > 0) else 0.0
    silent = rms < thr
    frame_s = hop / sr
    silence_ratio = float(silent.mean())

    # Count contiguous silent runs >= ~150 ms as "pauses".
    min_len = max(1, int(0.15 / frame_s))
    n_pauses, run = 0, 0
    pause_lengths = []
    for s in silent:
        if s:
            run += 1
        else:
            if run >= min_len:
                n_pauses += 1
                pause_lengths.append(run * frame_s)
            run = 0
    if run >= min_len:
        n_pauses += 1
        pause_lengths.append(run * frame_s)

    return {
        "silence_ratio": silence_ratio,
        "n_pauses": float(n_pauses),
        "mean_pause_s": float(np.mean(pause_lengths)) if pause_lengths else 0.0,
        "speech_ratio": 1.0 - silence_ratio,
    }


def extract_audio_features(video_path: Path, clip_id: str, cfg=CFG) -> dict:
    import librosa

    wav_path = cfg.wav_dir / f"{clip_id}.wav"
    if not wav_path.exists():
        extract_wav(video_path, wav_path, sample_rate=SR, mono=True)

    y, sr = librosa.load(str(wav_path), sr=SR, mono=True)
    duration = len(y) / sr if sr else 0.0
    feats: dict[str, float] = {"clip_id": clip_id, "audio_duration_s": _safe(duration)}  # type: ignore

    if y.size < FRAME:
        # Degenerate / silent clip — emit zeros so downstream stays aligned.
        zero_keys = [
            "f0_mean", "f0_std", "f0_range", "f0_voiced_frac", "f0_jitter",
            "rms_mean", "rms_std", "rms_shimmer",
            "silence_ratio", "n_pauses", "mean_pause_s", "speech_ratio",
            "zcr_mean", "zcr_std", "centroid_mean", "centroid_std",
            "bandwidth_mean", "rolloff_mean",
        ]
        feats.update({k: 0.0 for k in zero_keys})
        write_json(cfg.audio_dir / f"{clip_id}.json", feats)
        return feats

    # --- Pitch (F0) via probabilistic YIN ---------------------------------
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=65, fmax=400, sr=sr, frame_length=FRAME, hop_length=HOP
    )
    voiced = f0[~np.isnan(f0)]
    feats["f0_mean"] = _safe(np.mean(voiced)) if voiced.size else 0.0
    feats["f0_std"] = _safe(np.std(voiced)) if voiced.size else 0.0
    feats["f0_range"] = _safe(np.ptp(voiced)) if voiced.size else 0.0
    feats["f0_voiced_frac"] = _safe(np.mean(voiced_flag)) if voiced_flag.size else 0.0
    # Jitter proxy: mean abs. relative F0 change between consecutive voiced frames.
    if voiced.size > 1:
        d = np.abs(np.diff(voiced)) / (voiced[:-1] + 1e-6)
        feats["f0_jitter"] = _safe(np.mean(d))
    else:
        feats["f0_jitter"] = 0.0

    # --- Loudness (RMS) + pauses ------------------------------------------
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    feats["rms_mean"] = _safe(np.mean(rms))
    feats["rms_std"] = _safe(np.std(rms))
    # Shimmer proxy: mean abs. relative amplitude change between frames.
    if rms.size > 1:
        dr = np.abs(np.diff(rms)) / (rms[:-1] + 1e-6)
        feats["rms_shimmer"] = _safe(np.mean(dr))
    else:
        feats["rms_shimmer"] = 0.0
    feats.update(_pause_stats(rms, HOP, sr))

    # --- Voice quality / spectral -----------------------------------------
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME, hop_length=HOP)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME, hop_length=HOP)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=FRAME, hop_length=HOP)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=FRAME, hop_length=HOP)[0]
    feats["zcr_mean"] = _safe(np.mean(zcr))
    feats["zcr_std"] = _safe(np.std(zcr))
    feats["centroid_mean"] = _safe(np.mean(cent))
    feats["centroid_std"] = _safe(np.std(cent))
    feats["bandwidth_mean"] = _safe(np.mean(bw))
    feats["rolloff_mean"] = _safe(np.mean(rolloff))

    write_json(cfg.audio_dir / f"{clip_id}.json", feats)
    return feats
