import importlib
import os


def import_mlx_whisper():
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    return importlib.import_module("mlx_whisperx.backend.mlx_whisper")
