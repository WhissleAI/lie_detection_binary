"""Step 02 — Whissle gateway extraction (STT + audio-visual).

For each clip we make **two** gateway calls and merge them into one record:

  1. ``POST /asr/transcribe`` (on the clip's 16 kHz wav) — Whissle ASR with the
     full metadata payload: transcript, per-word timing + confidence + filler
     flags, **pauses**, **speech_rate**, overall **confidence**, **uncertain
     words**, **entities**, and clip-level **metadata** (emotion / age / gender /
     behavior / role / intent) plus soft **metadata_probs** distributions.

  2. ``POST /video/analyze`` (on the mp4) — the audio-visual hybrid-intelligence
     lane; we keep its ``visual_timeline`` (per-frame face emotion / head pose /
     gaze / blink / mouth + hand gestures) and optional ``semantic_samples``.

Why two calls?  ``/video/analyze`` runs ASR internally but its fuser only
forwards a ``segments`` field, which this ASR model doesn't emit — so the rich
metadata above is lost if we rely on it alone. ``/asr/transcribe`` exposes
everything, so we use it for the text lane and use ``/video/analyze`` purely for
vision. The merged record is cached at ``data/av/<clip_id>.json``.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx

from ..config import CFG
from ..media import extract_wav
from ..io_utils import write_json

# The gateway proxies to internal services (ASR :8001, video :8002) that can
# briefly drop connections under load; retry those transient failures.
_MAX_RETRIES = 4
_RETRY_BACKOFF_S = 3.0
_RETRY_STATUS = {500, 502, 503, 504}


class GatewayError(RuntimeError):
    pass


def _auth(cfg) -> dict:
    if not cfg.whissle_api_token:
        raise GatewayError(
            "WHISSLE_API_TOKEN is not set. Add it to .env "
            "(the gateway requires 'Authorization: Bearer wh_...')."
        )
    return {"Authorization": f"Bearer {cfg.whissle_api_token}"}


def _post_with_retry(url: str, *, headers: dict, data: dict, file_path: Path,
                     file_field: tuple, timeout: float, label: str) -> dict:
    """POST a multipart file with retries on transient transport / 5xx errors."""
    last_err = ""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with file_path.open("rb") as f:
                files = {"file": (file_field[0], f, file_field[1])}
                r = httpx.post(url, headers=headers, data=data, files=files, timeout=timeout)
            if r.status_code in _RETRY_STATUS:
                last_err = f"{r.status_code}: {r.text[:200]}"
            else:
                r.raise_for_status()
                return r.json()
        except httpx.HTTPStatusError as e:
            # 4xx (auth, bad request) — not worth retrying.
            raise GatewayError(f"{label} {e.response.status_code}: {e.response.text[:300]}") from e
        except httpx.HTTPError as e:
            last_err = str(e)
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_BACKOFF_S * attempt)  # linear backoff
    raise GatewayError(f"{label} failed after {_MAX_RETRIES} attempts: {last_err}")


def transcribe_audio(wav_path: Path, cfg=CFG, timeout: float = 180.0) -> dict:
    """POST /asr/transcribe — rich STT (transcript + metadata + pauses + words)."""
    return _post_with_retry(
        f"{cfg.gateway_url}/asr/transcribe",
        headers=_auth(cfg),
        data={
            "language": "en",
            "model": cfg.asr_model_id,   # e.g. whissle-large
            "metadata_prob": "true",     # emotion/intent/age/gender + prob distributions
            "word_timestamps": "true",   # per-word timing, pauses, speech rate
            "use_lm": "true",
        },
        file_path=wav_path,
        file_field=(wav_path.name, "audio/wav"),
        timeout=timeout,
        label="/asr/transcribe",
    )


def analyze_video(video_path: Path, cfg=CFG, timeout: float = 600.0) -> dict:
    """POST /video/analyze — we keep only its visual_timeline + semantic_samples."""
    return _post_with_retry(
        f"{cfg.gateway_url}/video/analyze",
        headers=_auth(cfg),
        data={
            "language": "en",
            "frame_fps": str(cfg.visual_sample_fps),
            "semantic": str(cfg.visual_semantic_lane).lower(),
            "punctuation": "true",
            "itn": "true",
            "metadata_tags": cfg.metadata_tags,
            "diarization": str(cfg.diarization).lower(),
        },
        file_path=video_path,
        file_field=(video_path.name, "video/mp4"),
        timeout=timeout,
        label="/video/analyze",
    )


# Fields we lift verbatim from the /asr/transcribe response into our record.
_ASR_KEEP = (
    "transcript", "transcript_with_entities", "metadata", "metadata_probs",
    "entities", "words", "pauses", "speech_rate", "confidence",
    "uncertain_words", "model", "inference_time",
)


def extract_clip(video_path: Path, clip_id: str, cfg=CFG) -> dict:
    """Run both gateway lanes for one clip; merge, cache, and return the record."""
    wav_path = cfg.wav_dir / f"{clip_id}.wav"
    if not wav_path.exists():
        extract_wav(video_path, wav_path, sample_rate=16000, mono=True)

    asr = transcribe_audio(wav_path, cfg)
    video = analyze_video(video_path, cfg)

    record = {"clip_id": clip_id, "text": asr.get("transcript", "")}
    record.update({k: asr[k] for k in _ASR_KEEP if k in asr})
    record["asr_model"] = asr.get("model")
    record["visual_timeline"] = video.get("visual_timeline", [])
    record["semantic_samples"] = video.get("semantic_samples", [])
    record["video_models"] = video.get("models", {})
    record["video_params"] = video.get("video_params", {})

    write_json(cfg.av_dir / f"{clip_id}.json", record)
    return record


def refresh_asr(video_path: Path, clip_id: str, cfg=CFG) -> dict:
    """Re-run ONLY the ASR/text lane (e.g. after switching ASR model) and merge
    into the existing av record, preserving the visual_timeline. Falls back to a
    full extract_clip if no prior record exists."""
    from ..io_utils import read_json

    out = cfg.av_dir / f"{clip_id}.json"
    if not out.exists():
        return extract_clip(video_path, clip_id, cfg)

    wav_path = cfg.wav_dir / f"{clip_id}.wav"
    if not wav_path.exists():
        extract_wav(video_path, wav_path, sample_rate=16000, mono=True)

    record = read_json(out)
    asr = transcribe_audio(wav_path, cfg)
    record["text"] = asr.get("transcript", "")
    record.update({k: asr[k] for k in _ASR_KEEP if k in asr})
    record["asr_model"] = asr.get("model")
    write_json(out, record)
    return record


def health(cfg=CFG, timeout: float = 10.0) -> dict:
    """Quick reachability/auth check against the gateway video service.

    Never raises — returns status_code 0 if the gateway is unreachable (e.g. the
    docker is still cold-starting), so callers can warn instead of crashing.
    """
    headers = {"Authorization": f"Bearer {cfg.whissle_api_token}"} if cfg.whissle_api_token else {}
    try:
        r = httpx.get(f"{cfg.gateway_url}/video/health", headers=headers, timeout=timeout)
        return {"status_code": r.status_code, "body": r.text[:300]}
    except httpx.HTTPError as e:
        return {"status_code": 0, "body": f"gateway unreachable at {cfg.gateway_url}: {e}"}
