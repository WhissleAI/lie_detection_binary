"""Build the UTA-RLDD manifest from unzipped videos.

Layout:  <VIDEOS>/Fold*/<subject>/<state>.<ext>   where state in {0,5,10}.
clip_id = "s<subject>_<state>";  group = subject (globally unique across folds).
"""

from __future__ import annotations

import csv
from pathlib import Path

from . import config as C

_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def build_manifest() -> list[dict]:
    rows = []
    for vid in sorted(C.VIDEOS.rglob("*")):
        if vid.suffix.lower() not in _EXTS:
            continue
        state = vid.stem  # "0" / "5" / "10"
        if state not in C.STATE_LABEL:
            continue
        subject = vid.parent.name
        label = C.STATE_LABEL[state]
        rows.append({
            "clip_id": f"s{subject}_{state}",
            "subject": subject,
            "state": state,
            "label": label,
            "binary": C.BINARY.get(label, ""),   # "" for low_vigilant
            "video": str(vid),
        })
    return rows


def write_manifest(rows: list[dict]) -> Path:
    C.ensure_dirs()
    out = C.WORK / "manifest.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "subject", "state", "label", "binary", "video"])
        w.writeheader()
        w.writerows(rows)
    return out


if __name__ == "__main__":
    rows = build_manifest()
    path = write_manifest(rows)
    subs = sorted({r["subject"] for r in rows})
    from collections import Counter
    print(f"✓ {len(rows)} clips, {len(subs)} subjects -> {path}")
    print("  subjects:", ", ".join(subs))
    print("  label counts:", dict(Counter(r["label"] for r in rows)))
