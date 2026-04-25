"""Logging helpers for the top-level `mlx_whisperx` logger tree."""

import logging


LOGGER_NAME = "mlx_whisperx"


def setup_logging(level: str = "warning", log_file: str | None = None) -> None:
    """Configure root logging for CLI runs.

    `force=True` deliberately replaces previous handlers so repeated CLI invocations
    from test harnesses or notebooks do not duplicate log lines.
    """
    numeric_level = getattr(logging, level.upper(), logging.WARNING)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a named logger, defaulting to the package logger."""
    return logging.getLogger(name or LOGGER_NAME)
