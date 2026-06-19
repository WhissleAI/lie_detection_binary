#!/usr/bin/env python3
"""Run the full pipeline end-to-end.

    python scripts/run_all.py                 # real: gateway STT + audio-visual
    python scripts/run_all.py --bootstrap     # offline: bundled transcripts (text+audio only)
    python scripts/run_all.py --limit 5       # quick smoke run on 5 clips

Steps: 01 manifest → 02 gateway AV (or bootstrap) → 03 audio prosody →
       04 build features → 05 train/evaluate.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PY = sys.executable


def run(script: str, *args: str) -> None:
    cmd = [PY, str(SCRIPTS / script), *args]
    print(f"\n{'='*70}\n$ {' '.join(cmd)}\n{'='*70}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"step {script} failed (exit {r.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    lim = ["--limit", str(args.limit)] if args.limit else []

    run("01_build_manifest.py")
    run("02_extract_av.py", *(["--bootstrap"] if args.bootstrap else []), *lim)
    run("03_extract_audio.py", *lim)
    run("04_build_features.py")
    run("05_train.py")


if __name__ == "__main__":
    main()
