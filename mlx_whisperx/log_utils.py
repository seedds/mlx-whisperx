import logging


LOGGER_NAME = "mlx_whisperx"


def setup_logging(level: str = "warning", log_file: str | None = None) -> None:
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
    return logging.getLogger(name or LOGGER_NAME)
