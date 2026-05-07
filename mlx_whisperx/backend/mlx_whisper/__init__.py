# Copyright © 2023-2024 Apple Inc.

"""Vendored MLX Whisper backend used by the outer mlx-whisperx pipeline.

This subtree is intentionally close to upstream `mlx-whisper`. The surrounding
project imports it through `_compat.import_mlx_whisper` and treats `transcribe` as the
stable ASR boundary.
"""

import importlib

from ._version import __version__


def transcribe(*args, **kwargs):
    """Lazy wrapper preserving the backend's top-level transcribe entry point."""
    from .transcribe import transcribe as _transcribe

    return _transcribe(*args, **kwargs)


def __getattr__(name: str):
    """Lazily expose backend modules without importing tokenizer dependencies early."""
    if name in {"audio", "decoding", "languages", "load_models"}:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["__version__", "audio", "decoding", "languages", "load_models", "transcribe"]
