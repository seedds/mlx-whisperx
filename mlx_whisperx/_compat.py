import importlib
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _prepend_path(path: Path) -> None:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def ensure_local_mlx_whisper() -> None:
    try:
        importlib.import_module("mlx_whisper")
        return
    except ImportError:
        _prepend_path(PROJECT_ROOT / "mlx-examples" / "whisper")


def ensure_local_whisperx() -> None:
    try:
        importlib.import_module("whisperx")
        return
    except ImportError:
        _prepend_path(PROJECT_ROOT / "whisperX")


def import_mlx_whisper():
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    return importlib.import_module("mlx_whisperx.backend.mlx_whisper")


def import_whisperx_module(name: str):
    ensure_local_whisperx()
    return importlib.import_module(f"whisperx.{name}")
