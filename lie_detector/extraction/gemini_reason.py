"""Gemini reasoning over our EXTRACTED features (not raw video).

Instead of giving Gemini the video, we give it a structured, time-synced summary
of everything our pipeline extracted — transcript + Whissle STT metadata
probabilities + acoustic prosody + a visual timeline aligned to the speech — and
ask it to judge deception. This isolates "can an LLM reason over our feature
representation?" from "can an LLM watch video?".

We store BOTH the exact prompt/summary we send and Gemini's response, so every
output is auditable.
"""

from __future__ import annotations

from pathlib import Path

from ..config import CFG
from ..io_utils import write_json
from ..features.visual_features import visual_features
from ..features.audio_features import audio_features
from .gemini import _client, _loads  # reuse the client + json fallback

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["deceptive", "truthful"]},
        "deception_probability": {"type": "number"},
        "key_evidence": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "deception_probability"],
}

_INSTRUCTION = (
    "You are a forensic deception analyst. Below is a structured, time-synced "
    "summary of multimodal evidence automatically extracted from a short "
    "real courtroom clip of a defendant/witness: the transcript, speech metadata "
    "(emotion/age/gender/behavior probabilities + speech rate + pauses), acoustic "
    "prosody, and a visual timeline (gaze/head-pose/facial-emotion/gestures) "
    "aligned to the speech. Reason over ALL modalities and judge deception.\n"
    "Return ONLY JSON: verdict ('deceptive'|'truthful'), deception_probability "
    "(0..1), key_evidence (short bullet strings), rationale (one sentence). "
    "If uncertain, estimate near 0.5; do not refuse.\n\n"
)

# v2 — debiased: neutral persona, explicit 50/50 base rate, symmetric evidence,
# caution that behavioural cues are weak, and calibration toward 0.5.
_INSTRUCTION_NEUTRAL = (
    "You are a careful, impartial analyst deciding whether a short courtroom "
    "statement is TRUTHFUL or DECEPTIVE, based on automatically-extracted "
    "multimodal evidence (transcript, speech metadata, acoustic prosody, and a "
    "visual timeline aligned to speech).\n\n"
    "Calibrate carefully:\n"
    "- BASE RATE: this set is balanced — about 50% of speakers are truthful and "
    "50% deceptive. Do NOT assume deception by default.\n"
    "- Behavioural cues (nervousness, gaze aversion, pauses, emotion, fidgeting) "
    "are WEAK and AMBIGUOUS: truthful people under high-stakes courtroom stress "
    "show them too. They are not proof of lying.\n"
    "- Weigh evidence for BOTH interpretations symmetrically before deciding.\n"
    "- Assign a high deception_probability ONLY when evidence clearly favours "
    "deception; when ambiguous, stay near 0.5.\n\n"
    "Return ONLY JSON with: truthful_evidence (array of short strings), "
    "deceptive_evidence (array of short strings), verdict "
    "('truthful'|'deceptive'), deception_probability (0..1, calibrated), "
    "rationale (one sentence). Do not refuse.\n\n"
    "EVIDENCE:\n"
)

_SCHEMA_NEUTRAL = {
    "type": "object",
    "properties": {
        "truthful_evidence": {"type": "array", "items": {"type": "string"}},
        "deceptive_evidence": {"type": "array", "items": {"type": "string"}},
        "verdict": {"type": "string", "enum": ["deceptive", "truthful"]},
        "deception_probability": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "deception_probability"],
}


def _fmt_dist(probs: dict, cat: str, top: int = 4) -> str:
    dist = (probs or {}).get(cat)
    if not isinstance(dist, list):
        return "n/a"
    items = sorted(dist, key=lambda d: -float(d.get("probability", 0)))[:top]
    return ", ".join(f"{d['token'].split('_',1)[-1].lower()} {float(d['probability']):.2f}" for d in items)


def _synced_timeline(av: dict, window_s: float = 2.0, max_windows: int = 20) -> str:
    """Per-window lines aligning spoken words with the visual state + pauses."""
    words = av.get("words") or []
    pauses = av.get("pauses") or []
    frames = av.get("visual_timeline") or []
    dur = 0.0
    for src in (words, frames):
        for it in src:
            t = it.get("end") or it.get("timestamp") or 0
            dur = max(dur, float(t or 0))
    if dur <= 0:
        return "(no timing available)"

    lines = []
    n = min(max_windows, int(dur / window_s) + 1)
    for i in range(n):
        a, b = i * window_s, (i + 1) * window_s
        said = " ".join(w["word"] for w in words
                        if w.get("start") is not None and a <= float(w["start"]) < b)
        win_frames = [f for f in frames if a <= float(f.get("timestamp", -1)) < b]
        faces = [f["faces"][0] for f in win_frames if f.get("faces")]
        if faces:
            from collections import Counter
            gaze = Counter(f.get("gaze") for f in faces).most_common(1)[0][0]
            emo = Counter(f.get("emotion") for f in faces).most_common(1)[0][0]
            pitch = sum(f.get("head_pose", {}).get("pitch", 0) for f in faces) / len(faces)
            vis = f"gaze {gaze}, face {emo}, head_pitch {pitch:+.0f}°"
        else:
            vis = "no face detected"
        pause_here = sum(float(p.get("duration", 0)) for p in pauses
                         if a <= float(p.get("start", -1)) < b)
        pause_str = f", pause {pause_here:.1f}s" if pause_here > 0.2 else ""
        lines.append(f"  [{a:>4.0f}-{b:<4.0f}s] said: \"{said or '—'}\" | {vis}{pause_str}")
    return "\n".join(lines)


def build_multimodal_summary(av: dict, audio: dict) -> str:
    """The structured, synced text representation we hand to Gemini."""
    sr = av.get("speech_rate") or {}
    probs = av.get("metadata_probs") or {}
    vis = visual_features(av)
    aud = audio_features(audio) if audio else {}

    parts = []
    parts.append(f"TRANSCRIPT:\n\"{av.get('text','').strip()}\"\n")

    parts.append("SPEECH METADATA (Whissle STT, probabilities):")
    parts.append(f"  audio emotion: {_fmt_dist(probs,'emotion')}")
    parts.append(f"  behavior: {_fmt_dist(probs,'behavior',5)}")
    parts.append(f"  age: {_fmt_dist(probs,'age',3)} | gender: {_fmt_dist(probs,'gender',2)}")
    parts.append(f"  ASR confidence: {av.get('confidence','?')} | "
                 f"speech: {sr.get('words_per_minute','?')} wpm, "
                 f"{sr.get('pause_count','?')} pauses ({sr.get('total_pause_sec','?')}s), "
                 f"filler_rate {sr.get('filler_rate','?')}\n")

    # Focused deception-intent filter (richest new signal)
    fi = av.get("filtered_intents") or []
    if fi:
        top = sorted([x for x in fi if isinstance(x, dict)],
                     key=lambda x: -float(x.get("probability", 0)))[:6]
        parts.append("DECEPTION-INTENT FILTER (Whissle, probability):")
        parts.append("  " + ", ".join(f"{x['label'].lower()} {float(x.get('probability',0)):.2f}" for x in top) + "\n")

    # Gateway speech-analysis: fluency / grammar / rhythm
    sa = av.get("speech_analysis") or {}
    if sa:
        rh = sa.get("rhythm") or {}
        parts.append("SPEECH ANALYSIS (Whissle):")
        parts.append(f"  fluency {sa.get('fluency_score','?')}, grammar {sa.get('grammar_score','?')}, "
                     f"vocab_range {sa.get('vocabulary_range','?')} | "
                     f"speaking_ratio {rh.get('speaking_ratio','?')}, pause_rate {rh.get('pause_rate','?')}, "
                     f"inter-word-interval {rh.get('iwi_mean_sec','?')}±{rh.get('iwi_std_sec','?')}s\n")

    parts.append("ACOUSTIC PROSODY:")
    parts.append(f"  pitch(F0) mean {aud.get('aud_f0_mean', aud.get('f0_mean','?'))}Hz, "
                 f"jitter {aud.get('f0_jitter','?')}, shimmer {aud.get('rms_shimmer','?')}")
    parts.append(f"  silence_ratio {aud.get('silence_ratio','?')}, "
                 f"pauses {aud.get('n_pauses','?')}, voiced_frac {aud.get('f0_voiced_frac','?')}\n")

    parts.append("VISUAL SUMMARY (whole clip):")
    parts.append(f"  face_detect_rate {vis['face_detect_rate']:.2f} | "
                 f"gaze_aversion {vis['gaze_aversion']:.2f} (center {vis['gaze_frac_center']:.2f}, "
                 f"down {vis['gaze_frac_down']:.2f}) | blink_rate {vis['blink_rate']:.2f}")
    parts.append(f"  head motion yaw {vis['pose_yaw_motion']:.1f}°/frame, "
                 f"attention_engaged {vis['attention_engaged_frac']:.2f} | "
                 f"top facial emotion frac: neutral {vis['emo_frac_neutral']:.2f}, "
                 f"happy {vis['emo_frac_happy']:.2f}, sad {vis['emo_frac_sad']:.2f}\n")

    parts.append("TIME-SYNCED TIMELINE (speech aligned with visual state):")
    parts.append(_synced_timeline(av))
    return "\n".join(parts)


def reason_over_features(av: dict, audio: dict, clip_id: str, cfg=CFG, neutral: bool = False) -> dict:
    """Build the synced summary, ask Gemini to judge, store input + output.

    neutral=True uses the debiased v2 prompt (base-rate-anchored, symmetric
    evidence) and writes to gemini_reason_v2/; otherwise the v1 forensic prompt.
    """
    from google.genai import types

    summary = build_multimodal_summary(av, audio)
    instruction = _INSTRUCTION_NEUTRAL if neutral else _INSTRUCTION
    schema = _SCHEMA_NEUTRAL if neutral else _VERDICT_SCHEMA
    out_dir = cfg.gemini_reason_v2_dir if neutral else cfg.gemini_reason_dir

    client = _client(cfg)
    resp = client.models.generate_content(
        model=cfg.gemini_model,
        contents=[instruction + summary],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
        ),
    )
    data = resp.parsed if getattr(resp, "parsed", None) else _loads(resp.text)

    record = {
        "clip_id": clip_id,
        "model": cfg.gemini_model,
        "prompt_variant": "neutral_v2" if neutral else "forensic_v1",
        "verdict": data.get("verdict"),
        "deception_probability": float(data.get("deception_probability", 0.5)),
        "truthful_evidence": data.get("truthful_evidence", []),
        "deceptive_evidence": data.get("deceptive_evidence", data.get("key_evidence", [])),
        "rationale": data.get("rationale", ""),
        "input_summary": summary,   # store exactly what we sent
    }
    write_json(out_dir / f"{clip_id}.json", record)
    return record
