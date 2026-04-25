# Copyright © 2023 Apple Inc.

"""Model loading for MLX-converted Whisper checkpoints."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

from . import whisper


def load_model(
    path_or_hf_repo: str,
    dtype: mx.Dtype = mx.float32,
) -> whisper.Whisper:
    """Load a local or Hugging Face MLX Whisper checkpoint.

    Checkpoints are expected to contain `config.json` plus either `weights.safetensors`
    or `weights.npz`. Quantized checkpoints are detected from the config and the model
    layers are quantized before weights are installed.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        # Treat non-existent local paths as Hugging Face repo IDs and let the Hub cache
        # provide the snapshot directory.
        model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

    with open(str(model_path / "config.json"), "r") as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        # MLX quantization settings live beside model dimensions in converted configs.
        quantization = config.pop("quantization", None)

    model_args = whisper.ModelDimensions(**config)

    wf = model_path / "weights.safetensors"
    if not wf.exists():
        wf = model_path / "weights.npz"
    weights = mx.load(str(wf))

    model = whisper.Whisper(model_args, dtype)

    if quantization is not None:
        # Only quantize layers with corresponding scale tensors in the checkpoint.
        class_predicate = (
            lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
            and f"{p}.scales" in weights
        )
        nn.quantize(model, **quantization, class_predicate=class_predicate)

    weights = tree_unflatten(list(weights.items()))
    model.update(weights)
    # Force parameter materialization before returning so first decode does not pay the
    # lazy-evaluation setup cost unpredictably.
    mx.eval(model.parameters())
    return model
