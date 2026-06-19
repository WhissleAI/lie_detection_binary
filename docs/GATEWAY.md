# The Whissle Gateway dependency (external — not in this repo)

This project does **not** run any speech or vision models itself. Two of the
three feature lanes — the **transcript + STT metadata** (text lane) and the
**audio-visual signals** (visual lane) — are produced by the **Whissle gateway**,
a separate service that runs as a Docker container. This repo only *calls* it
(`lie_detector/extraction/gateway.py`), parses the JSON, and builds features.

> Only the **audio prosody** lane (step 03, librosa) runs locally in this repo.
> It needs `ffmpeg` on PATH and nothing else.

```
 this repo (lie_detection_binary)                Whissle gateway (Docker, :9000)
 ┌───────────────────────────────┐               ┌─────────────────────────────┐
 │ 02_extract_av.py              │  POST          │ /video/analyze              │
 │   extraction/gateway.py  ─────┼──────────────► │   ├─ Whissle ASR (+metadata)│
 │                               │  multipart     │   └─ MediaPipe vision lane  │
 │   parses JSON → features      │ ◄──────────────┤   → fused JSON              │
 └───────────────────────────────┘  fused result  └─────────────────────────────┘
```

---

## Running the gateway

The gateway image is `whissleasr/whissle-gateway` (the `lie_detection_binary`
project used the `:liedetect` tag). It exposes everything on port **9000**.

```bash
docker run -d --name whissle-gateway -p 9000:9000 \
  whissleasr/whissle-gateway:liedetect
```

- **Cold start** takes ~1–2 min (ASR + vision models load on boot). Until ready,
  requests return connection-refused or `HTTP 000`. Check readiness:
  ```bash
  curl -s http://localhost:9000/health        # -> {"status":"healthy",...}
  ```
- Point this repo at it via `.env`:
  ```
  WHISSLE_GATEWAY_URL=http://localhost:9000     # or https://api.whissle.ai for cloud
  ```
- The gateway is also the place where the optional **Claude semantic visual
  lane** runs — it needs `ANTHROPIC_API_KEY` set *in the gateway container*. This
  repo only asks for it via `semantic=true` (`VISUAL_SEMANTIC_LANE=1`).

> The gateway's source lives in the `decoder_onnx` / unified-docker projects of
> the monorepo, **not here**. Treat it as a black-box service with the contract
> below. (For reference, inside the container the relevant code is
> `/app/asr/video/server.py` and `/app/asr/src/server.py`.)

---

## Authentication

Every `/asr/*` and `/video/*` call requires a bearer token:

```
Authorization: Bearer wh_xxxxxxxxxxxxxxxx
```

Create one at <https://lulu.whissle.ai/access> and put it in `.env` as
`WHISSLE_API_TOKEN`. Without it the gateway returns `401`.

---

## Endpoint we use: `POST /video/analyze`

Single call that runs ASR **and** the vision lane on a video and fuses them.

### Request (multipart/form-data)

| field | type | default | meaning |
|---|---|---|---|
| `file` | binary | — | the video clip (`.mp4`) |
| `language` | str | `en` | ASR language |
| `frame_fps` | float | `2.0` | frames/sec sampled for the vision lane (we send `VISUAL_SAMPLE_FPS`, default 5) |
| `semantic` | bool | `true` | run the Claude semantic lane (needs key on the gateway) |
| `punctuation` | bool | `true` | punctuate the transcript |
| `itn` | bool | `true` | inverse text normalisation (numbers, dates) |
| `metadata_tags` | str | `emotion,intent,age,gender` | per-segment metadata to emit |
| `diarization` | bool | `true` | speaker labels per segment |

### Response (the fields this repo consumes)

```jsonc
{
  "text": "full transcript ...",
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "text": "i did not do it",
      "start": 1.2, "end": 3.4,
      "metadata": { "emotion": "neutral", "intent": "deny",
                    "age": "30-40", "gender": "female" },
      "entities": [ { "type": "PERSON", "text": "..." } ],
      "words": [ { "word": "i", "start": 1.2, "end": 1.3, "confidence": 0.97 } ],
      // visual annotation of the closest-in-time frame (added by the fuser):
      "visual_emotion": "neutral", "visual_gaze": "down",
      "visual_attention": "distracted", "visual_speaking": true,
      "head_pose": { "yaw": 3.4, "pitch": -29.3, "roll": 13.8 },
      "gestures": ["Open_Palm"]
    }
  ],
  "visual_timeline": [
    {
      "timestamp": 0.4, "frame_idx": 12,
      "faces": [
        { "emotion": "neutral",
          "emotion_scores": { "happy": 0.0, "sad": 0.01, "angry": 0.03, ... },
          "head_pose": { "yaw": 3.4, "pitch": -29.3, "roll": 13.8 },
          "gaze": "down", "blink": false, "attention": "distracted",
          "mouth_open": 0.004, "speaking": false,
          "box": [0.31, 0.12, 0.18, 0.19] }     // normalized [x, y, w, h]
      ],
      "hands": [ { "gesture": "Open_Palm", "handedness": "Right", "confidence": 87.1 } ]
    }
  ],
  "semantic_samples": [ { "timestamp": 1.5, "scene": "...", "activity": "...",
                          "visual_entities": [...], "summary": "..." } ],
  "language": "en", "inference_time": 2.1, "model": "whissle-...",
  "diarization": true, "processing_time": "4.20s",
  "models": { "asr": "...", "vision_fast": "mediapipe-face-landmarker",
              "vision_semantic": "claude-..." },
  "video_params": { "duration": "17.0s", "frame_fps": 5,
                    "frames_analyzed": 85, "semantic_samples": 6 }
}
```

### How the response maps to features

| Response part | Built by | Produces |
|---|---|---|
| `text`, `segments[].text` | `features/text_features.py` | lexical/psycholinguistic `txt_*` |
| `segments[].metadata`, `entities`, `words` | `features/text_features.py` | STT-metadata `txt_*` (audio-emotion mix, intents, confidence, speech rate) |
| `visual_timeline[].faces / .hands` | `features/visual_features.py` | aggregated `vis_*` (gaze aversion, head motion, blink, emotion mix, gestures) |

The raw fused JSON for each clip is cached at `data/av/<clip_id>.json`, so you can
re-run feature building / modelling without re-hitting the gateway.

---

## Alternative endpoint: `POST /asr/transcribe` (STT only)

If you ever want the transcript + metadata **without** the vision lane (faster),
the gateway also exposes `/asr/transcribe` (multipart `file`, plus
`metadata_prob`, `word_timestamps`, `language`, `hotwords`, `use_lm`,
`speaker_embedding`, …). Returns `{transcript, metadata, word_timestamps, ...}`.
This repo does not use it by default, but it's a drop-in if the vision lane is
unavailable — wire it in `gateway.py` next to `analyze_video`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `connection refused` / `HTTP 000` | gateway still cold-starting, or not running. `docker ps`, then wait for `/health` = 200. |
| `401 API token required` | missing/invalid `WHISSLE_API_TOKEN` in `.env`. |
| `vis_*` features all zero / `face_detect_rate` 0 | no face detected (small/oblique face) **or** you're in `--bootstrap` mode (no visual). Check `data/av/<clip>.json` → `visual_timeline`. |
| timeouts on long clips | bump the `timeout` arg in `analyze_video` (default 600 s). |
| want richer/cheaper vision sampling | tune `VISUAL_SAMPLE_FPS` in `.env` (more fps = more frames = slower, finer signals). |
