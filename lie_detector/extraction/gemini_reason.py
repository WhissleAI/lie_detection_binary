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
    "statement is TRUTHFUL or DECEPTIVE, from automatically-extracted multimodal "
    "evidence (transcript, speech metadata, acoustic prosody, a visual timeline).\n\n"
    "How to weigh the evidence (important):\n"
    "- BASE RATE: this set is balanced — ~50% truthful, ~50% deceptive. Do NOT "
    "default to deceptive. If you notice you are calling most clips deceptive, "
    "recalibrate: a confident, fluent, or even nervous speaker is just as likely "
    "truthful.\n"
    "- VERBAL CONTENT IS THE STRONGEST SIGNAL: what they actually say — direct "
    "denials, internal contradictions, vagueness vs. concrete verifiable detail, "
    "admissions, the deception-intent probabilities — should drive your judgement.\n"
    "- BEHAVIOURAL CUES ARE WEAK: nervousness, gaze aversion, pauses, fidgeting, "
    "emotion are unreliable — truthful people under courtroom stress show them too. "
    "Use them only as minor tie-breakers, never as proof.\n"
    "- Reason step by step: first estimate from the transcript/intent content, then "
    "nudge only slightly for prosody/visual. Stay near 0.5 when content is neutral.\n\n"
    "Return ONLY JSON with: truthful_evidence (array of short strings), "
    "deceptive_evidence (array of short strings), verdict "
    "('truthful'|'deceptive'), deception_probability (0..1, calibrated — a true "
    "probability, not a hunch), rationale (one sentence). Do not refuse.\n\n"
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


def reason_over_features(av: dict, audio: dict, clip_id: str, cfg=CFG, neutral: bool = True,
                         provider: str = "gemini", n_samples: int = 1) -> dict:
    """Build the synced summary, ask an LLM to judge, store input + output.

    neutral=True uses the debiased prompt (base-rate-anchored, symmetric evidence).
    provider in {"gemini","claude"} selects the judge LLM; n_samples>1 enables
    self-consistency (sample at temperature, average probability / majority vote).
    """
    summary = build_multimodal_summary(av, audio)
    instruction = _INSTRUCTION_NEUTRAL if neutral else _INSTRUCTION
    prompt = instruction + summary
    temp = 0.0 if n_samples <= 1 else 0.6

    samples = [_judge(prompt, cfg, provider, temp) for _ in range(max(1, n_samples))]
    samples = [s for s in samples if s]
    probs = [float(s.get("deception_probability", 0.5)) for s in samples] or [0.5]
    prob = sum(probs) / len(probs)
    verdict = "deceptive" if prob >= 0.5 else "truthful"
    last = samples[-1] if samples else {}

    out_dir = cfg.claude_reason_dir if provider == "claude" else cfg.gemini_reason_v2_dir
    record = {
        "clip_id": clip_id,
        "provider": provider,
        "model": cfg.anthropic_model if provider == "claude" else cfg.gemini_model,
        "prompt_variant": "neutral_v3",
        "n_samples": len(samples),
        "verdict": verdict,
        "deception_probability": prob,
        "truthful_evidence": last.get("truthful_evidence", []),
        "deceptive_evidence": last.get("deceptive_evidence", last.get("key_evidence", [])),
        "rationale": last.get("rationale", ""),
        "input_summary": summary,   # store exactly what we sent
    }
    write_json(out_dir / f"{clip_id}.json", record)
    return record


def _judge(prompt: str, cfg, provider: str, temperature: float) -> dict:
    """Single LLM-as-judge call. Returns parsed JSON dict (or {} on failure)."""
    if provider == "claude":
        return _judge_claude(prompt, cfg, temperature)
    return _judge_gemini(prompt, cfg, temperature)


def _judge_gemini(prompt: str, cfg, temperature: float) -> dict:
    from google.genai import types
    client = _client(cfg)
    resp = client.models.generate_content(
        model=cfg.gemini_model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_SCHEMA_NEUTRAL,
            temperature=temperature,
        ),
    )
    return resp.parsed if getattr(resp, "parsed", None) else _loads(resp.text)


def _judge_claude(prompt: str, cfg, temperature: float) -> dict:
    import anthropic
    if not cfg.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set (add it to .env).")
    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    # Forced tool-use guarantees schema-valid JSON (no fragile free-text parsing).
    tool = {"name": "submit_verdict",
            "description": "Submit the truthful/deceptive judgement.",
            "input_schema": _SCHEMA_NEUTRAL}
    kwargs = dict(
        model=cfg.anthropic_model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
        tools=[tool],
        tool_choice={"type": "tool", "name": "submit_verdict"},
    )
    if temperature and temperature > 0:
        try:
            msg = client.messages.create(temperature=temperature, **kwargs)
        except anthropic.BadRequestError as e:
            if "temperature" in str(e):
                msg = client.messages.create(**kwargs)
            else:
                raise
    else:
        msg = client.messages.create(**kwargs)
    for b in msg.content:
        if getattr(b, "type", "") == "tool_use":
            return dict(b.input)
    # fallback: parse any text
    text = "".join(getattr(b, "text", "") for b in msg.content)
    return _loads(text)
