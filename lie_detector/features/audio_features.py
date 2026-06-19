"""Audio features: the prosody extractor already emits a flat float dict, so this
is mostly a pass-through plus a couple of derived ratios."""

from __future__ import annotations


def audio_features(audio_record: dict) -> dict[str, float]:
    feats = {k: float(v) for k, v in audio_record.items()
             if k not in {"clip_id"} and isinstance(v, (int, float))}
    # Derived: pause density (pauses per second of speech), F0 coefficient of variation.
    dur = feats.get("audio_duration_s", 0.0)
    feats["pause_density"] = (feats.get("n_pauses", 0.0) / dur) if dur else 0.0
    f0m = feats.get("f0_mean", 0.0)
    feats["f0_cv"] = (feats.get("f0_std", 0.0) / f0m) if f0m else 0.0
    return feats
