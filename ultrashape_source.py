from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable


ULTRASHAPE_SOURCE_REPO = "https://github.com/PKU-YuanGroup/UltraShape-1.0"
ULTRASHAPE_SOURCE_ZIP_URL = f"{ULTRASHAPE_SOURCE_REPO}/archive/refs/heads/main.zip"
ULTRASHAPE_REQUIRED_CONFIG = "infer_dit_refine.yaml"


def is_ultrashape_source_root(root: str | Path) -> bool:
    root_path = Path(root)
    return (
        (root_path / "ultrashape").is_dir()
        and (root_path / "configs").is_dir()
        and (root_path / "configs" / ULTRASHAPE_REQUIRED_CONFIG).is_file()
    )


def candidate_ultrashape_roots(app_dir: str | Path) -> list[Path]:
    app_root = Path(app_dir).resolve()
    candidates: list[Path] = []

    def _add(path_like: str | Path | None) -> None:
        if not path_like:
            return
        path = Path(path_like).expanduser()
        if not path.is_absolute():
            path = app_root / path
        path = path.resolve()
        if path not in candidates:
            candidates.append(path)

    for env_name in ("ULTRASHAPE_ROOT", "ULTRASHAPE_SOURCE_DIR"):
        _add(os.environ.get(env_name))

    for path in (
        app_root / "UltraShape-1.0",
        app_root / "ComfyUI-UltraShape1" / "UltraShape-1.0",
        app_root / "UltraShape_v2",
        app_root / "models" / "UltraShape-1.0",
    ):
        _add(path)

    spec = importlib.util.find_spec("ultrashape")
    if spec is not None:
        if spec.origin:
            _add(Path(spec.origin).resolve().parent.parent)
        for search_loc in spec.submodule_search_locations or ():
            _add(Path(search_loc).resolve().parent)

    return candidates


def format_ultrashape_source_not_found(app_dir: str | Path) -> str:
    candidates = candidate_ultrashape_roots(app_dir)
    return (
        "UltraShape source not found. The UltraShape model weights alone are not enough; "
        "the source checkout is also required for configs and Python modules. "
        "Run `python ensure_ultrashape_source.py` from the Trellis_2_3D_Generator folder, "
        "set ULTRASHAPE_ROOT to the UltraShape source checkout, or place it in one of: "
        + ", ".join(str(p) for p in candidates)
    )


def _log(log_fn: Callable[[str], None] | None, msg: str) -> None:
    if log_fn is not None:
        log_fn(msg)


def _is_empty_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        next(path.iterdir())
        return False
    except StopIteration:
        return True


def _select_install_target(app_dir: Path) -> Path | None:
    for target in (
        app_dir / "UltraShape-1.0",
        app_dir / "models" / "UltraShape-1.0",
    ):
        if is_ultrashape_source_root(target):
            return target
        if not target.exists() or _is_empty_dir(target):
            return target
    return None


def _find_extracted_source_root(extract_dir: Path) -> Path:
    for root in [extract_dir, *extract_dir.iterdir()]:
        if root.is_dir() and is_ultrashape_source_root(root):
            return root
    raise RuntimeError("Downloaded UltraShape archive did not contain the expected ultrashape/ and configs/ folders.")


def _download_source_archive(target: Path, *, log_fn: Callable[[str], None] | None = None) -> None:
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    _log(log_fn, f"Downloading UltraShape source from {ULTRASHAPE_SOURCE_REPO}")

    with tempfile.TemporaryDirectory(prefix="ultrashape_source_") as tmp:
        tmp_dir = Path(tmp)
        archive_path = tmp_dir / "UltraShape-1.0.zip"
        extract_dir = tmp_dir / "extract"
        urllib.request.urlretrieve(ULTRASHAPE_SOURCE_ZIP_URL, archive_path)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)
        source_root = _find_extracted_source_root(extract_dir)

        if target.exists() and not _is_empty_dir(target):
            raise FileExistsError(f"Refusing to overwrite non-empty UltraShape source target: {target}")

        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source_root, target)

    if not is_ultrashape_source_root(target):
        raise RuntimeError(f"UltraShape source install did not produce a valid checkout at {target}")
    _log(log_fn, f"UltraShape source ready: {target}")


def ensure_ultrashape_source(
    app_dir: str | Path,
    *,
    allow_download: bool = True,
    log_fn: Callable[[str], None] | None = None,
) -> Path:
    app_root = Path(app_dir).resolve()

    for root in candidate_ultrashape_roots(app_root):
        if is_ultrashape_source_root(root):
            _log(log_fn, f"Using UltraShape source: {root}")
            return root

    if not allow_download:
        raise FileNotFoundError(format_ultrashape_source_not_found(app_root))

    target = _select_install_target(app_root)
    if target is None:
        raise FileNotFoundError(
            format_ultrashape_source_not_found(app_root)
            + ". Automatic install could not choose a safe target because the default source folders already exist "
            "but are incomplete. Remove or fix the incomplete folder, then rerun `python ensure_ultrashape_source.py`."
        )

    _download_source_archive(target, log_fn=log_fn)
    return target.resolve()
