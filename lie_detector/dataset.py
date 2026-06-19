"""Build the dataset manifest: clip path, binary label, and speaker group.

Why speaker grouping matters
----------------------------
The Real-life Trial dataset has 121 clips but only ~56 unique speakers, and
several clips come from the same person (e.g. 18 Jodi Arias clips). A naive
random train/test split lets the model memorise *who* is speaking instead of
*whether they are lying* — wildly optimistic, leaky accuracy. We therefore parse
each clip's speaker from the dataset README and expose it as a CV grouping key so
training can do leave-one-speaker-out / grouped CV (see modeling/cv.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from .config import CFG, Config


# Matches a README table row:  | trial_lie_001.mp4 | Defendant / Amanda Hayes | Guilty | <link> |
_ROW_RE = re.compile(
    r"\|\s*(trial_(?:lie|truth)_\d+\.mp4)\s*\|\s*([^|]*?)\s*\|", re.IGNORECASE
)


@dataclass
class ClipRecord:
    clip_id: str          # e.g. "trial_lie_001"
    video_path: str
    label: str            # "deceptive" | "truthful"
    y: int                # 1 = deceptive, 0 = truthful
    speaker: str          # grouping key for CV
    role: str             # "Defendant" / "Witness" / ...


def parse_speakers_from_readme(readme_path: Path) -> dict[str, dict[str, str]]:
    """Map clip_id -> {'speaker':..., 'role':...} parsed from the README tables.

    The 'Role / trial name' column looks like 'Defendant / Amanda Hayes'. We take
    the role before the slash and the speaker/case name after it. Falls back
    gracefully when a row can't be parsed.
    """
    out: dict[str, dict[str, str]] = {}
    if not readme_path.exists():
        return out
    text = readme_path.read_text(encoding="utf-8", errors="ignore")
    for m in _ROW_RE.finditer(text):
        clip = m.group(1).lower().replace(".mp4", "")
        role_name = m.group(2).strip()
        if "/" in role_name:
            role, speaker = role_name.split("/", 1)
        else:
            role, speaker = "", role_name
        role = role.strip()
        speaker = re.sub(r"\s+", " ", speaker).strip().lower()
        if speaker:
            out[clip] = {"speaker": speaker, "role": role}
    return out


def build_manifest(cfg: Config = CFG) -> pd.DataFrame:
    """Scan Clips/{Deceptive,Truthful} and build a manifest DataFrame."""
    speaker_map = parse_speakers_from_readme(cfg.readme_txt)

    records: list[ClipRecord] = []
    label_dirs = {
        "deceptive": cfg.clips_dir / "Deceptive",
        "truthful": cfg.clips_dir / "Truthful",
    }
    for label, d in label_dirs.items():
        if not d.exists():
            raise FileNotFoundError(f"Expected clips under {d}")
        for mp4 in sorted(d.glob("*.mp4")):
            clip_id = mp4.stem
            meta = speaker_map.get(clip_id, {})
            # Fall back to the clip id as its own speaker group if README missing.
            speaker = meta.get("speaker") or clip_id
            records.append(
                ClipRecord(
                    clip_id=clip_id,
                    video_path=str(mp4),
                    label=label,
                    y=1 if label == "deceptive" else 0,
                    speaker=speaker,
                    role=meta.get("role", ""),
                )
            )

    df = pd.DataFrame([asdict(r) for r in records]).sort_values("clip_id").reset_index(drop=True)
    return df


def load_manifest(cfg: Config = CFG) -> pd.DataFrame:
    """Load a previously-saved manifest.csv."""
    if not cfg.manifest_csv.exists():
        raise FileNotFoundError(
            f"{cfg.manifest_csv} not found. Run scripts/01_build_manifest.py first."
        )
    return pd.read_csv(cfg.manifest_csv)
