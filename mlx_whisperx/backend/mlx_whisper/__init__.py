# Copyright © 2023-2024 Apple Inc.

"""Vendored MLX Whisper backend used by the outer mlx-whisperx pipeline.

This subtree is intentionally close to upstream `mlx-whisper`. The surrounding
project imports it through `_compat.import_mlx_whisper` and treats `transcribe` as the
stable ASR boundary.
"""

from . import audio, decoding, load_models
from ._version import __version__
from .transcribe import transcribe
