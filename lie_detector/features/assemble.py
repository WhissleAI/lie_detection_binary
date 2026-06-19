"""Assemble the multimodal feature matrix: one row per clip.

Joins the manifest (label + speaker group) with the three modality feature sets,
prefixing columns by modality so ablations can select them:

    txt_*  linguistic (Whissle STT transcript + metadata)
    vis_*  audio-visual hybrid intelligence (MediaPipe)
    aud_*  acoustic / prosody (librosa)

Missing modalities for a clip are filled with NaN (imputed at train time), and a
`*_present` flag records availability.
"""

from __future__ import annotations

import pandas as pd

from ..config import CFG
from ..dataset import load_manifest
from ..io_utils import read_json
from .text_features import text_features
from .visual_features import visual_features
from .audio_features import audio_features

# text + visual both come from the single fused gateway record (data/av/<clip>.json);
# audio prosody is a complementary local librosa step (data/audio/<clip>.json).
_BUILDERS = {
    "txt": ("av_dir", text_features),
    "vis": ("av_dir", visual_features),
    "aud": ("audio_dir", audio_features),
}


def _load_modality(clip_id: str, dir_attr: str, builder, prefix: str, cfg) -> dict:
    path = getattr(cfg, dir_attr) / f"{clip_id}.json"
    if not path.exists():
        return {f"{prefix}_present": 0.0}
    try:
        rec = read_json(path)
        feats = builder(rec)
        out = {f"{prefix}_{k}": v for k, v in feats.items()}
        out[f"{prefix}_present"] = 1.0
        return out
    except Exception as e:  # bad json shouldn't sink the whole matrix
        return {f"{prefix}_present": 0.0, f"{prefix}_error": str(e)[:80]}


def build_feature_matrix(cfg=CFG) -> pd.DataFrame:
    manifest = load_manifest(cfg)
    rows = []
    for _, m in manifest.iterrows():
        clip_id = m["clip_id"]
        row = {
            "clip_id": clip_id,
            "label": m["label"],
            "y": int(m["y"]),
            "speaker": m["speaker"],
            "role": m.get("role", ""),
        }
        for prefix, (dir_attr, builder) in _BUILDERS.items():
            row.update(_load_modality(clip_id, dir_attr, builder, prefix, cfg))
        rows.append(row)

    df = pd.DataFrame(rows)
    # Drop stray error columns from the numeric matrix (keep only meta + numerics).
    err_cols = [c for c in df.columns if c.endswith("_error")]
    return df.drop(columns=err_cols)


def feature_columns(df: pd.DataFrame, modalities=("txt", "vis", "aud")) -> list[str]:
    """Numeric feature columns for the chosen modalities (excludes meta columns)."""
    meta = {"clip_id", "label", "y", "speaker", "role"}
    wanted = tuple(f"{m}_" for m in modalities)
    cols = []
    for c in df.columns:
        if c in meta or not c.startswith(wanted):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols
