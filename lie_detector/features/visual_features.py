"""Aggregate the gateway's per-frame ``visual_timeline`` into a per-clip vector.

The audio-visual lane (run on the gateway) emits one record per sampled frame:
faces (emotion, head pose, gaze, blink, attention, mouth openness, speaking) and
hand gestures. Deception-relevant *behavioural* summaries come from how these
vary over the clip: gaze aversion, emotional leakage/variability, head-movement
(fidgeting), blink rate, expressivity. We reduce the frame stream to
interpretable statistics over frames where the speaker's face was detected, and
expose ``face_detect_rate`` so the model knows the coverage.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter

_EMOTIONS = ["happy", "sad", "angry", "surprised", "disgust", "fear", "neutral"]
_GAZES = ["center", "left", "right", "up", "down"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.mean(xs)) if xs else 0.0


def _std(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def _primary_face(faces: list[dict]) -> dict | None:
    """Largest face in the frame = the on-camera speaker."""
    if not faces:
        return None

    def area(f):
        b = f.get("box") or [0, 0, 0, 0]
        return (b[2] * b[3]) if len(b) >= 4 else 0.0

    return max(faces, key=area)


def _zero_vector() -> dict[str, float]:
    feats: dict[str, float] = {}
    for e in _EMOTIONS:
        feats[f"emo_frac_{e}"] = 0.0
        feats[f"emo_score_{e}"] = 0.0
    feats["emo_entropy"] = 0.0
    for g in _GAZES:
        feats[f"gaze_frac_{g}"] = 0.0
    feats["gaze_aversion"] = 0.0
    for k in ("yaw", "pitch", "roll"):
        for stat in ("mean", "std", "absmean", "motion"):
            feats[f"pose_{k}_{stat}"] = 0.0
    feats.update({
        "blink_rate": 0.0, "attention_engaged_frac": 0.0,
        "mouth_open_mean": 0.0, "mouth_open_std": 0.0, "speaking_frac": 0.0,
        "hand_present_frac": 0.0, "gesture_diversity": 0.0, "top_gesture_frac": 0.0,
        "n_faces_mean": 0.0,
    })
    return feats


def visual_features(av_record: dict) -> dict[str, float]:
    frames = av_record.get("visual_timeline", []) or []
    n_total = len(frames)

    feats: dict[str, float] = {"n_frames": float(n_total)}

    primaries = []
    for fr in frames:
        pf = _primary_face(fr.get("faces") or [])
        if pf is not None:
            primaries.append(pf)
    n_face = len(primaries)
    feats["face_detect_rate"] = (n_face / n_total) if n_total else 0.0

    if n_face == 0:
        feats.update(_zero_vector())
        return feats

    # --- emotion distribution + intensity ---
    emo_labels = [fa.get("emotion") for fa in primaries]
    emo_counter = Counter(e for e in emo_labels if e)
    for e in _EMOTIONS:
        feats[f"emo_frac_{e}"] = emo_counter.get(e, 0) / n_face
        feats[f"emo_score_{e}"] = _mean([fa.get("emotion_scores", {}).get(e) for fa in primaries])
    probs = [c / n_face for c in emo_counter.values()]
    feats["emo_entropy"] = float(-sum(p * math.log(p + 1e-12) for p in probs if p > 0))

    # --- gaze ---
    gaze_counter = Counter(fa.get("gaze") for fa in primaries if fa.get("gaze"))
    for g in _GAZES:
        feats[f"gaze_frac_{g}"] = gaze_counter.get(g, 0) / n_face
    feats["gaze_aversion"] = 1.0 - feats["gaze_frac_center"]

    # --- head pose: position, spread, frame-to-frame motion (fidgeting) ---
    for k in ("yaw", "pitch", "roll"):
        seq = [fa.get("head_pose", {}).get(k) for fa in primaries]
        seq = [s for s in seq if s is not None]
        feats[f"pose_{k}_mean"] = _mean(seq)
        feats[f"pose_{k}_std"] = _std(seq)
        feats[f"pose_{k}_absmean"] = _mean([abs(s) for s in seq])
        diffs = [abs(seq[i] - seq[i - 1]) for i in range(1, len(seq))]
        feats[f"pose_{k}_motion"] = _mean(diffs)

    # --- blink / attention / mouth / speaking ---
    feats["blink_rate"] = _mean([1.0 if fa.get("blink") else 0.0 for fa in primaries])
    feats["attention_engaged_frac"] = _mean(
        [1.0 if fa.get("attention") == "engaged" else 0.0 for fa in primaries])
    feats["mouth_open_mean"] = _mean([fa.get("mouth_open") for fa in primaries])
    feats["mouth_open_std"] = _std([fa.get("mouth_open") for fa in primaries])
    feats["speaking_frac"] = _mean([1.0 if fa.get("speaking") else 0.0 for fa in primaries])

    # --- hands / gestures (over all frames) ---
    feats["hand_present_frac"] = _mean([1.0 if fr.get("hands") else 0.0 for fr in frames])
    gestures = [h.get("gesture") for fr in frames for h in (fr.get("hands") or [])
                if h.get("gesture") and h.get("gesture") != "None"]
    gcount = Counter(gestures)
    feats["gesture_diversity"] = float(len(gcount))
    feats["top_gesture_frac"] = (max(gcount.values()) / len(gestures)) if gestures else 0.0

    feats["n_faces_mean"] = _mean([float(len(fr.get("faces") or [])) for fr in frames])
    return feats
