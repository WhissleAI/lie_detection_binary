"""Lightweight smoke tests — no network/token needed.

Run:  python -m pytest tests/ -q
"""

from __future__ import annotations

import numpy as np

from lie_detector.config import CFG
from lie_detector.dataset import build_manifest, parse_speakers_from_readme
from lie_detector.features.text_features import text_features
from lie_detector.features.visual_features import visual_features
from lie_detector.features.audio_features import audio_features
from lie_detector.modeling.metrics import compute_metrics


def test_manifest_shape_and_groups():
    df = build_manifest(CFG)
    assert len(df) == 121
    assert set(df["label"]) == {"deceptive", "truthful"}
    assert int((df.y == 1).sum()) == 61 and int((df.y == 0).sum()) == 60
    # speaker grouping must collapse 121 clips into far fewer groups
    assert df["speaker"].nunique() < 60


def test_readme_speaker_parsing():
    m = parse_speakers_from_readme(CFG.readme_txt)
    assert m.get("trial_lie_001", {}).get("speaker") == "amanda hayes"
    # Jodi Arias should own many clips (the big leakage case)
    jodi = [c for c, v in m.items() if v["speaker"] == "jodi arias"]
    assert len(jodi) >= 15


def test_text_features_keys():
    rec = {"text": "I did not. I absolutely did not, sir.",
           "segments": [{"text": "I did not.", "metadata": {"emotion": "neutral"},
                         "words": [{"word": "I", "start": 0.0, "end": 0.1, "confidence": 0.9}]}]}
    f = text_features(rec)
    for k in ("word_count", "rate_i_singular", "type_token_ratio",
              "audio_emo_frac_neutral", "words_per_second"):
        assert k in f
    assert f["word_count"] > 0


def test_visual_features_empty_and_populated():
    # empty timeline -> zero vector with coverage 0
    f0 = visual_features({"visual_timeline": []})
    assert f0["face_detect_rate"] == 0.0 and f0["gaze_aversion"] == 0.0
    # one frame with a centered, neutral face
    frame = {"timestamp": 0.0, "faces": [{
        "emotion": "neutral", "emotion_scores": {"happy": 0.0},
        "head_pose": {"yaw": 1.0, "pitch": 2.0, "roll": 0.0},
        "gaze": "center", "blink": False, "attention": "engaged",
        "mouth_open": 0.1, "speaking": True, "box": [0.1, 0.1, 0.3, 0.4]}], "hands": []}
    f1 = visual_features({"visual_timeline": [frame]})
    assert f1["face_detect_rate"] == 1.0
    assert f1["gaze_frac_center"] == 1.0 and f1["gaze_aversion"] == 0.0
    assert f1["emo_frac_neutral"] == 1.0


def test_audio_features_derived():
    f = audio_features({"clip_id": "x", "audio_duration_s": 10.0, "n_pauses": 5.0,
                        "f0_mean": 150.0, "f0_std": 30.0})
    assert abs(f["pause_density"] - 0.5) < 1e-9
    assert abs(f["f0_cv"] - 0.2) < 1e-9


def test_metrics_perfect_and_chance():
    y = np.array([0, 0, 1, 1])
    perfect = compute_metrics(y, y, y.astype(float))
    assert perfect["accuracy"] == 1.0 and perfect["roc_auc"] == 1.0
    chance = compute_metrics(y, np.array([0, 0, 0, 0]), np.full(4, 0.5))
    assert chance["accuracy"] == 0.5
