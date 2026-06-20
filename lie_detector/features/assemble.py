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
from .text_features import text_features, collect_metadata_vocab
from .visual_features import visual_features
from .audio_features import audio_features


def _prefixed(prefix: str, feats: dict) -> dict:
    out = {f"{prefix}_{k}": v for k, v in feats.items()}
    out[f"{prefix}_present"] = 1.0
    return out


def build_feature_matrix(cfg=CFG) -> pd.DataFrame:
    """One row per clip.

    Text + visual both come from the gateway record (``data/av/<clip>.json``);
    audio prosody is the complementary local librosa step. The metadata
    probability vocabulary is collected once across all clips so the per-token
    ``txt_metaprob_*`` columns are consistent across the matrix.
    """
    manifest = load_manifest(cfg)

    # Pre-load all gateway records once (used for both txt + vis, and for vocab).
    av: dict[str, dict] = {}
    for clip_id in manifest["clip_id"]:
        p = cfg.av_dir / f"{clip_id}.json"
        if p.exists():
            try:
                av[clip_id] = read_json(p)
            except Exception:
                pass
    meta_vocab = collect_metadata_vocab(av.values())

    rows = []
    for _, m in manifest.iterrows():
        clip_id = m["clip_id"]
        row = {"clip_id": clip_id, "label": m["label"], "y": int(m["y"]),
               "speaker": m["speaker"], "role": m.get("role", "")}

        rec = av.get(clip_id)
        if rec is not None:
            row.update(_prefixed("txt", text_features(rec, meta_vocab)))
            row.update(_prefixed("vis", visual_features(rec)))
        else:
            row["txt_present"] = 0.0
            row["vis_present"] = 0.0

        ap = cfg.audio_dir / f"{clip_id}.json"
        if ap.exists():
            try:
                row.update(_prefixed("aud", audio_features(read_json(ap))))
            except Exception:
                row["aud_present"] = 0.0
        else:
            row["aud_present"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


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
