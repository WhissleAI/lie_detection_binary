"""Small IO helpers: JSON read/write and a simple file-based cache guard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)  # atomic-ish on the same filesystem


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def is_fresh(out_path: Path, *inputs: Path) -> bool:
    """True if ``out_path`` exists and is newer than every input (cache hit)."""
    out_path = Path(out_path)
    if not out_path.exists():
        return False
    out_mtime = out_path.stat().st_mtime
    for inp in inputs:
        inp = Path(inp)
        if inp.exists() and inp.stat().st_mtime > out_mtime:
            return False
    return True
