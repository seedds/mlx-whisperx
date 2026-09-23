"""Compatibility helpers that isolate imports of the vendored MLX Whisper backend."""

import contextlib
import importlib
import os


@contextlib.contextmanager
def hf_offline(enabled: bool):
    """Force Hugging Face Hub offline for libraries without a cache-only argument.

    Pyannote loads models through the Hub but exposes no `local_files_only` flag, so
    offline mode is the only way to honor `--model_cache_only` there. The Hub reads
    `HF_HUB_OFFLINE` into a module constant once at import, and the ASR backend has
    usually imported it already, so the constant is patched alongside the variable.
    """
    if not enabled:
        yield
        return

    from huggingface_hub import constants

    previous_env = os.environ.get("HF_HUB_OFFLINE")
    previous_constant = constants.HF_HUB_OFFLINE
    os.environ["HF_HUB_OFFLINE"] = "1"
    constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        constants.HF_HUB_OFFLINE = previous_constant
        if previous_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_env


def patch_torchaudio_for_pyannote(torchaudio_module) -> None:
    """Add missing torchaudio backend APIs expected by some pyannote stacks.

    `speechbrain`, pulled in by recent `pyannote.audio` releases, still probes
    `torchaudio.list_audio_backends()` during import in some environments. Newer
    torchaudio builds can omit that helper entirely, which breaks pyannote import
    before any VAD/diarization code runs. mlx-whisperx does not rely on that API,
    so provide a minimal compatibility shim when it is absent.
    """
    if not hasattr(torchaudio_module, "list_audio_backends"):
        torchaudio_module.list_audio_backends = lambda: ["soundfile"]


def prepare_pyannote_audio_compat() -> None:
    """Patch optional torchaudio APIs so pyannote imports succeed when possible."""
    try:
        torchaudio_module = importlib.import_module("torchaudio")
    except Exception:
        return
    patch_torchaudio_for_pyannote(torchaudio_module)


def import_mlx_whisper():
    """Import the vendored backend after applying process-level defaults.

    Hugging Face progress bars are disabled here because the higher-level CLI already
    owns user-facing progress output. Keeping this in a helper also avoids importing
    the backend at module import time, which makes lightweight operations such as
    `--help` faster and less dependent on MLX being ready.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    return importlib.import_module("mlx_whisperx.backend.mlx_whisper")
