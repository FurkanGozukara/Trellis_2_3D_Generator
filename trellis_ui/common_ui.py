"""
Common UI utilities and helpers for Trellis 2 3D Generator.
Shared across all UI modules.
"""
import gradio as gr
import os
from pathlib import Path
from typing import Optional, List


def open_folder(path: str) -> None:
    """Open folder in system file explorer."""
    import sys
    import subprocess
    
    path = os.path.abspath(path)
    if os.name == "nt":
        os.startfile(path)
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", path])
        return
    # Linux / WSL / others
    for cmd in (["xdg-open", path], ["gio", "open", path]):
        try:
            subprocess.Popen(cmd)
            return
        except FileNotFoundError:
            continue
    raise FileNotFoundError("No folder opener found (tried: xdg-open, gio).")


def append_status(current: str, msg: str) -> str:
    """Append message to status string."""
    current = current or ""
    msg = msg or ""
    if not current:
        return msg
    if not msg:
        return current
    return current + "\n" + msg


def trim_status(
    status: str,
    *,
    max_lines: int = 200,
    max_chars: int = 20000,
) -> str:
    """Trim status to prevent overwhelming Gradio UI."""
    status = status or ""
    if not status:
        return status

    # Char-bound first
    if max_chars and len(status) > max_chars * 2:
        status = status[-max_chars * 2 :]

    if max_lines:
        lines = status.splitlines()
        if len(lines) > max_lines:
            lines = ["… (truncated) …"] + lines[-max_lines:]
            status = "\n".join(lines)

    if max_chars and len(status) > max_chars:
        status = "… (truncated) …\n" + status[-max_chars:]

    return status
