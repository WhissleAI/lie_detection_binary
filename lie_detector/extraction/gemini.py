"""Gemini Pro multimodal baseline — process each clip's video directly.

Two things in one call per clip (the model watches the actual video + audio):

  1. a **direct zero-shot verdict** — deceptive vs. truthful + P(deceptive),
  2. a fixed vector of **deception-relevant feature scores** (0–1), which we then
     feed into the *same* ML algorithms / CV protocols as the other modalities.

This gives two more rows for the results table: "gemini_direct" (no training) and
"gemini_features" (Gemini's reads → DecisionTree/RandomForest under both CV
schemes).

Needs GEMINI_API_KEY (or GOOGLE_API_KEY) in .env and the `google-genai` SDK.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..config import CFG
from ..io_utils import write_json

# The 0–1 feature dimensions Gemini scores for each clip (higher = more present).
FEATURE_KEYS = [
    "verbal_hedging", "verbal_inconsistency", "story_specificity",
    "emotional_congruence", "nervousness", "gaze_aversion",
    "facial_microexpression_leakage", "vocal_stress", "fidgeting",
    "defensiveness", "speaker_confidence", "overall_credibility",
]

_PROMPT = (
    "You are a forensic deception-analysis expert. Watch this short real courtroom "
    "clip of a defendant or witness (video + audio) and assess deception.\n"
    "Return ONLY JSON with:\n"
    "- verdict: 'deceptive' or 'truthful'\n"
    "- deception_probability: 0..1 (probability the speaker is being deceptive)\n"
    "- features: an object scoring each 0..1 (higher = more present):\n"
    "    verbal_hedging, verbal_inconsistency, story_specificity, "
    "emotional_congruence, nervousness, gaze_aversion, "
    "facial_microexpression_leakage, vocal_stress, fidgeting, defensiveness, "
    "speaker_confidence, overall_credibility (1 = highly credible/truthful)\n"
    "- rationale: one short sentence.\n"
    "Judge only from behavioural and verbal cues in the clip. If uncertain, give "
    "your best estimate near 0.5; do not refuse."
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["deceptive", "truthful"]},
        "deception_probability": {"type": "number"},
        "features": {
            "type": "object",
            "properties": {k: {"type": "number"} for k in FEATURE_KEYS},
            "required": FEATURE_KEYS,
        },
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "deception_probability", "features"],
}


def _client(cfg):
    from google import genai

    if not cfg.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not set (add it to .env).")
    return genai.Client(api_key=cfg.gemini_api_key)


def _upload_active(client, video_path: Path, poll_s: float = 2.0, timeout_s: float = 180.0):
    """Upload a video and wait until the Files API marks it ACTIVE."""
    f = client.files.upload(file=str(video_path))
    waited = 0.0
    while getattr(f.state, "name", str(f.state)) == "PROCESSING" and waited < timeout_s:
        time.sleep(poll_s)
        waited += poll_s
        f = client.files.get(name=f.name)
    state = getattr(f.state, "name", str(f.state))
    if state != "ACTIVE":
        raise RuntimeError(f"file not ACTIVE (state={state})")
    return f


def analyze_clip(video_path: Path, clip_id: str, cfg=CFG) -> dict:
    """Run Gemini on one clip; write + return {verdict, prob, features, ...}."""
    from google.genai import types

    client = _client(cfg)
    f = _upload_active(client, video_path)
    try:
        resp = client.models.generate_content(
            model=cfg.gemini_model,
            contents=[f, _PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
                temperature=0.0,
            ),
        )
        data = resp.parsed if getattr(resp, "parsed", None) else _loads(resp.text)
    finally:
        try:
            client.files.delete(name=f.name)
        except Exception:
            pass

    record = {
        "clip_id": clip_id,
        "model": cfg.gemini_model,
        "verdict": data.get("verdict"),
        "deception_probability": float(data.get("deception_probability", 0.5)),
        "features": {k: float(data.get("features", {}).get(k, 0.5)) for k in FEATURE_KEYS},
        "rationale": data.get("rationale", ""),
    }
    write_json(cfg.gemini_dir / f"{clip_id}.json", record)
    return record


def _loads(text: str) -> dict:
    import json
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1].lstrip("json").strip("` \n")
    s, e = t.find("{"), t.rfind("}")
    return json.loads(t[s:e + 1] if s != -1 else t)


def gemini_features(record: dict) -> dict[str, float]:
    """Feature builder: Gemini's 0–1 dimension scores (gem_* columns)."""
    feats = {k: float(v) for k, v in (record.get("features") or {}).items()}
    # also expose the model's direct probability as a feature
    feats["direct_prob"] = float(record.get("deception_probability", 0.5))
    return feats
