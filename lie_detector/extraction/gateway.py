"""Step 02 — Whissle gateway extraction (STT + audio-visual in one call).

We POST each clip to the gateway's ``/video/analyze`` endpoint, which:
  1. extracts the audio and runs Whissle ASR with metadata tags
     (emotion / intent / age / gender) + diarization + word timestamps, and
  2. samples frames and runs the audio-visual hybrid-intelligence lane
     (MediaPipe face: emotion, head pose, gaze, blink, mouth; hand gestures),
then **fuses** them: each transcript segment is annotated with the closest
visual frame, plus a full ``visual_timeline``.

So a single request yields everything the downstream feature builders need:
transcript + per-segment metadata (text lane) and the per-frame visual signals
(audio-visual lane). No local CV models required — the gateway owns that.

Response shape (the bits we use)::

    {
      "text": "<full transcript>",
      "segments": [ {speaker, text, start, end,
                     metadata:{emotion,intent,age,gender},
                     entities:[...], words:[{word,start,end,confidence}],
                     visual_emotion, visual_gaze, visual_attention,
                     visual_speaking, head_pose, gestures:[...]} , ... ],
      "visual_timeline": [ {timestamp, frame_idx,
                            faces:[{emotion,emotion_scores,head_pose,gaze,
                                    blink,attention,mouth_open,speaking,box}],
                            hands:[{gesture,handedness,confidence}]} , ... ],
      "semantic_samples": [...],            # only if semantic lane on
      "language","inference_time","model","diarization",
      "processing_time","models","video_params"
    }
"""

from __future__ import annotations

from pathlib import Path

import httpx

from ..config import CFG
from ..io_utils import write_json


class GatewayError(RuntimeError):
    pass


def analyze_video(video_path: Path, clip_id: str, cfg=CFG, timeout: float = 600.0) -> dict:
    """Call POST {gateway}/video/analyze for one clip; write + return the result."""
    if not cfg.whissle_api_token:
        raise GatewayError(
            "WHISSLE_API_TOKEN is not set. Add it to .env "
            "(the gateway requires 'Authorization: Bearer wh_...')."
        )

    url = f"{cfg.gateway_url}/video/analyze"
    headers = {"Authorization": f"Bearer {cfg.whissle_api_token}"}
    data = {
        "language": "en",
        "frame_fps": str(cfg.visual_sample_fps),
        "semantic": str(cfg.visual_semantic_lane).lower(),
        "punctuation": "true",
        "itn": "true",
        "metadata_tags": cfg.metadata_tags,
        "diarization": str(cfg.diarization).lower(),
    }
    try:
        with video_path.open("rb") as f:
            files = {"file": (video_path.name, f, "video/mp4")}
            resp = httpx.post(url, headers=headers, data=data, files=files, timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise GatewayError(f"{e.response.status_code}: {e.response.text[:300]}") from e
    except httpx.HTTPError as e:
        raise GatewayError(f"request failed: {e}") from e

    result = resp.json()
    result["clip_id"] = clip_id
    write_json(cfg.av_dir / f"{clip_id}.json", result)
    return result


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
