import logging
from pathlib import Path


def configure_logging(name: str = __name__) -> logging.Logger:
    """
    Configure application logging to console and file.
    """

    log_directory = Path(__file__).resolve().parents[1] / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(
            log_directory / "application.log", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
