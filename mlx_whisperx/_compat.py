"""Compatibility helpers that isolate imports of the vendored MLX Whisper backend."""

import importlib
import os


def import_mlx_whisper():
    """Import the vendored backend after applying process-level defaults.

    Hugging Face progress bars are disabled here because the higher-level CLI already
    owns user-facing progress output. Keeping this in a helper also avoids importing
    the backend at module import time, which makes lightweight operations such as
    `--help` faster and less dependent on MLX being ready.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    return importlib.import_module("mlx_whisperx.backend.mlx_whisper")
