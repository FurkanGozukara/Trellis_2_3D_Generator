from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    run_dir: Path


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def allocate_run_dir(outputs_dir: os.PathLike | str, digits: int = 4) -> RunInfo:
    """
    Allocate a new numbered run directory: outputs/0001, outputs/0002, ...
    Never overwrites existing runs.
    """
    out = ensure_dir(outputs_dir)

    # Start from max existing numeric folder name, then attempt to create next.
    max_existing = 0
    for child in out.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name.isdigit():
            try:
                max_existing = max(max_existing, int(name))
            except Exception:
                pass

    n = max_existing + 1
    while True:
        run_id = f"{n:0{digits}d}"
        run_dir = out / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return RunInfo(run_id=run_id, run_dir=run_dir)
        except FileExistsError:
            n += 1


def next_indexed_path(
    directory: os.PathLike | str,
    *,
    prefix: str,
    ext: str,
    digits: int = 4,
    start: int = 1,
) -> Tuple[int, Path]:
    """
    Create a non-overwriting numbered filename inside `directory`, e.g.
      prefix_0001.ext, prefix_0002.ext, ...
    Returns (index, path). Does NOT create the file.
    """
    if not ext.startswith("."):
        ext = "." + ext
    d = ensure_dir(directory)

    max_existing = start - 1
    for child in d.iterdir():
        if not child.is_file():
            continue
        name = child.name
        if not name.startswith(prefix + "_") or not name.endswith(ext):
            continue
        mid = name[len(prefix) + 1 : -len(ext)]
        if mid.isdigit():
            try:
                max_existing = max(max_existing, int(mid))
            except Exception:
                pass

    idx = max_existing + 1
    while True:
        candidate = d / f"{prefix}_{idx:0{digits}d}{ext}"
        if not candidate.exists():
            return idx, candidate
        idx += 1


def safe_relpath(path: os.PathLike | str, start: os.PathLike | str) -> str:
    """
    Best-effort relative path for UI/logging (never raises).
    """
    try:
        return str(Path(path).resolve().relative_to(Path(start).resolve()))
    except Exception:
        return str(path)






