"""Shared top-level language normalization and model-language guard helpers."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Callable, Optional

from .backend.mlx_whisper.languages import LANGUAGES, TO_LANGUAGE_CODE


ENGLISH_ONLY_VOCAB_THRESHOLD = 51865
LANGUAGE_OPTION_HELP = (
    "Language code or name/alias. Examples: en, English, Portuguese. "
    "Defaults to auto-detect."
)


def _warn(message: str) -> None:
    warnings.warn(message, stacklevel=3)


def normalize_language(language: Optional[str]) -> Optional[str]:
    """Normalize user-provided language values to canonical Whisper codes."""
    if language is None:
        return None

    normalized = language.lower()
    if normalized in LANGUAGES:
        return normalized
    if normalized in TO_LANGUAGE_CODE:
        return TO_LANGUAGE_CODE[normalized]
    raise ValueError(
        f"Unsupported language: {normalized}. Use a Whisper language code or name like 'en' or 'English'."
    )


def parse_language(value: str) -> str:
    """Argparse-compatible language parser with early normalization."""
    try:
        normalized = normalize_language(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if normalized is None:
        raise argparse.ArgumentTypeError("Language must not be empty")
    return normalized


def looks_english_only_model(model: str) -> bool:
    """Return whether the model name follows Whisper's `.en` convention."""
    stripped = model.rstrip("/\\")
    return stripped.endswith(".en")


def _read_local_model_n_vocab(model_path: Path) -> int:
    config_path = model_path / "config.json"
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    return int(config["n_vocab"])


def is_english_only_model(model: str) -> bool:
    """Best-effort detection of English-only Whisper checkpoints."""
    model_path = Path(model)
    if model_path.exists():
        try:
            return _read_local_model_n_vocab(model_path) < ENGLISH_ONLY_VOCAB_THRESHOLD
        except Exception:
            return looks_english_only_model(model)
    return looks_english_only_model(model)


def normalize_language_settings(
    model: str,
    language: Optional[str],
    task: Optional[str],
    warn: Callable[[str], None] = _warn,
) -> tuple[Optional[str], Optional[str]]:
    """Normalize language input and enforce English-only model constraints."""
    normalized_language = normalize_language(language)

    if not is_english_only_model(model):
        return normalized_language, task

    if task == "translate":
        raise ValueError(
            "English-only Whisper models do not support task='translate'. "
            "Use task='transcribe' or choose a multilingual model."
        )

    if normalized_language not in {None, "en"}:
        warn(
            f"{model} is an English-only model but received {normalized_language!r}; using English instead."
        )

    return "en", task
