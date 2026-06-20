# The Whissle Gateway dependency (external — not in this repo)

This project does **not** run any speech or vision models itself. Two of the
three feature lanes — the **transcript + STT metadata** (text lane) and the
**audio-visual signals** (visual lane) — are produced by the **Whissle gateway**,
a separate service that runs as a Docker container. This repo only *calls* it
(`lie_detector/extraction/gateway.py`), parses the JSON, and builds features.

> Only the **audio prosody** lane (step 03, librosa) runs locally in this repo.
> It needs `ffmpeg` on PATH and nothing else.

Step 02 makes **two** calls per clip and merges them:

```
 this repo (lie_detection_binary)                Whissle gateway (Docker, :9000)
 ┌───────────────────────────────┐  POST wav      ┌─────────────────────────────┐
 │ 02_extract_av.py              │ ─────────────► │ /asr/transcribe             │
 │   extraction/gateway.py       │ ◄───────────── │   Whissle ASR + metadata    │
 │                               │  rich JSON     │   (transcript, pauses,      │
 │                               │                │    word conf, metadata_probs)│
 │   merge → data/av/<clip>.json │  POST mp4      ├─────────────────────────────┤
 │   parses → text + visual feats│ ─────────────► │ /video/analyze              │
 │                               │ ◄───────────── │   MediaPipe vision lane     │
 └───────────────────────────────┘  visual_timeline└─────────────────────────────┘
```

Why two calls? `/video/analyze` runs ASR internally but its fuser only forwards a
`segments` field that this ASR model doesn't emit — so the rich metadata
(`metadata_probs`, `pauses`, `speech_rate`, per-word confidence) is lost. We get
it from `/asr/transcribe`, and use `/video/analyze` purely for `visual_timeline`.

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

**Local docker token.** The self-hosted gateway keeps its own SQLite token store
(`/data/auth/tokens.db`). You can mint a valid `wh_` token without the cloud:

```bash
docker exec -w /app/agent whissle-gateway-liedetect \
  python -c "from app.local_auth import create_token; \
             print(create_token('liedetect','lie_detection_binary')['token'])"
# -> wh_...  (paste into .env as WHISSLE_API_TOKEN)
```

Validate it: `curl -H "Authorization: Bearer wh_..." localhost:9000/video/health`
should return `{"status":"ok",...}`.

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

## The text lane: `POST /asr/transcribe`

This is the **primary text/metadata source** (called for every clip alongside
`/video/analyze`). Multipart `file` (wav) + form fields `metadata_prob=true`,
`word_timestamps=true`, `use_lm=true`, `language`. The response is rich:

```jsonc
{
  "transcript": "no sir i did not ...",
  "confidence": 0.8948,                       // overall ASR confidence
  "words": [ {"word":"no","start":0.46,"end":0.58,"confidence":0.999,"filler":false}, ... ],
  "pauses": [ {"start":0.578,"end":0.738,"duration":0.16,"type":"short"}, ... ],
  "speech_rate": { "words_per_minute":145, "articulation_rate_wpm":270.6,
                   "word_count":41, "duration_sec":16.96, "spoken_sec":9.09,
                   "pause_count":29, "total_pause_sec":7.87, "filler_rate":0.0 },
  "uncertain_words": [],
  "entities": [],
  "metadata": { "emotion":"EMOTION_HAPPY","age":"AGE_30_45","gender":"GENDER_MALE",
                "behavior":"BEHAVIOR_FILLER","role":"ROLE_INTERVIEWEE", ... },
  "metadata_probs": {                          // FULL soft distributions, used as features
     "emotion":[ {"token":"EMOTION_HAPPY","probability":0.563},
                 {"token":"EMOTION_SAD","probability":0.364}, ... ],
     "age":[...], "gender":[...], "behavior":[...], "eval":[...], "role":[...] },
  "model": "en-in-tech-misc"
}
```

`features/text_features.py` turns every `metadata_probs` token into a
`metaprob_<cat>_<token>` feature (not just the top-1 label), plus per-category
entropy, the speech-rate / pause / word-confidence stats, and the lexical rates.

---

## Improving the visual lane (face detection)

Courtroom faces are often small/oblique, so MediaPipe's **default detection
confidence (0.5)** rejects most frames (~10% face-detect rate on the hardest
clips). Lowering it dramatically improves coverage. In the gateway container,
`/app/asr/video/vision/local.py` builds `FaceLandmarkerOptions` — add:

```python
min_face_detection_confidence=0.2,
min_face_presence_confidence=0.2,
```

then restart the video service (it has `autorestart=true`, so killing its
process works if `supervisorctl` has no socket):

```bash
docker exec whissle-gateway-liedetect sh -c \
  "kill $(pgrep -f 'uvicorn video')"   # supervisord restarts it with the patch
```

This lifted the worst clip 0.10 → 0.16 and most clips to 0.9–1.0. Re-extract the
visual lane only (keeps the text lane): `python scripts/02_extract_av.py
--visual-only`. **Note:** this edit lives in the running container and is lost if
the container is recreated — bake it into the image for a permanent fix.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `connection refused` / `HTTP 000` | gateway still cold-starting, or not running. `docker ps`, then wait for `/health` = 200. |
| `401 API token required` | missing/invalid `WHISSLE_API_TOKEN` in `.env`. |
| `vis_*` features all zero / `face_detect_rate` 0 | no face detected (small/oblique face) **or** you're in `--bootstrap` mode (no visual). Check `data/av/<clip>.json` → `visual_timeline`. |
| timeouts on long clips | bump the `timeout` arg in `analyze_video` (default 600 s). |
| want richer/cheaper vision sampling | tune `VISUAL_SAMPLE_FPS` in `.env` (more fps = more frames = slower, finer signals). |
