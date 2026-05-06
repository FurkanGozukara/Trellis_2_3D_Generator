from __future__ import annotations

import argparse
from pathlib import Path

from ultrashape_source import ensure_ultrashape_source


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensure the UltraShape source checkout is available.")
    parser.add_argument(
        "--app-dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Trellis_2_3D_Generator directory (default: this script's directory).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only check for an existing source checkout; do not download it.",
    )
    args = parser.parse_args()

    try:
        root = ensure_ultrashape_source(
            Path(args.app_dir),
            allow_download=not args.no_download,
            log_fn=lambda msg: print(f"[ultrashape-source] {msg}", flush=True),
        )
    except Exception as e:
        print(f"[ultrashape-source] ERROR: {type(e).__name__}: {e}", flush=True)
        return 1

    print(f"[ultrashape-source] OK: {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
