"""Drowsiness-specific visual features from the gateway visual_timeline.

The deception visual lane (gaze aversion, fidget, emotional variability) is not
tuned for fatigue. Here we compute the canonical drowsiness signals the timeline
exposes per frame: blink/eye-closure (PERCLOS), blink rate, yawning (mouth_open),
head droop/nod (head_pose.pitch), attention, gaze, and face-presence.
"""

from __future__ import annotations

import numpy as np


def _transitions(seq: list[int]) -> int:
    return sum(1 for a, b in zip(seq, seq[1:]) if a == 0 and b == 1)


def fatigue_visual_features(av_record: dict) -> dict[str, float]:
    tl = av_record.get("visual_timeline", []) or []
    n = len(tl)
    if n == 0:
        return {}
    dur = max(1e-3, float(tl[-1].get("timestamp", 0)) - float(tl[0].get("timestamp", 0)))
    blink, mouth, pitch, yaw, roll = [], [], [], [], []
    distracted = gaze_off = speaking = faceless = 0
    for fr in tl:
        faces = fr.get("faces") or []
        if not faces:
            faceless += 1
            continue
        f = faces[0]
        blink.append(1 if f.get("blink") else 0)
        mouth.append(float(f.get("mouth_open") or 0.0))
        hp = f.get("head_pose") or {}
        pitch.append(float(hp.get("pitch") or 0.0))
        yaw.append(float(hp.get("yaw") or 0.0))
        roll.append(float(hp.get("roll") or 0.0))
        distracted += int(f.get("attention") == "distracted")
        gaze_off += int((f.get("gaze") or "center") != "center")
        speaking += int(bool(f.get("speaking")))
    nf = max(1, n - faceless)
    mouth = np.array(mouth or [0.0])
    pitch = np.array(pitch or [0.0]); yaw = np.array(yaw or [0.0]); roll = np.array(roll or [0.0])
    return {
        # eye closure — the PERCLOS family (gold-standard drowsiness signal)
        "perclos": sum(blink) / nf,                          # fraction of frames eyes closed
        "blink_rate_pm": _transitions(blink) / (dur / 60.0),  # blinks per minute
        # yawning
        "mouth_open_mean": float(mouth.mean()),
        "mouth_open_max": float(mouth.max()),
        "yawn_frac": float((mouth > 0.5).mean()),            # sustained wide mouth
        # head droop / nodding
        "head_pitch_mean": float(pitch.mean()),
        "head_pitch_std": float(pitch.std()),
        "head_yaw_std": float(yaw.std()),
        "head_roll_std": float(roll.std()),
        # gaze / attention / presence
        "distracted_frac": distracted / nf,
        "gaze_off_frac": gaze_off / nf,
        "speaking_frac": speaking / nf,
        "faceless_frac": faceless / n,                        # head down/away -> no face
    }
