"""
Central logging setup. Console + rotating file (logs/app.log). Everything
else in the project should do `logger = logging.getLogger("ycf.<module>")`
and use the bracketed stage prefixes from spec section 27, e.g.:

    logger.info("[RESEARCH] starting topic=%s", topic)
    logger.error("[FAL] image generation failed: %s", err)

Recognised prefixes: [RESEARCH] [CLAUDE] [FAL] [TTS] [FFMPEG] [YOUTUBE]
[TELEGRAM] [PIPELINE] [STORAGE] [CONFIG]
"""
from __future__ import annotations

import logging
import logging.handlers
import sys

from core.config import settings

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger("ycf")
    if _CONFIGURED:
        return root

    root.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        settings.LOG_DIR / "app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Defensive: the Telegram Bot API embeds the bot token directly in the
    # request URL (https://api.telegram.org/bot<TOKEN>/method). httpx logs
    # full request URLs at INFO level under its own "httpx"/"httpcore"
    # loggers (silent under our default config, but not if something later
    # enables root-level DEBUG logging) - pin them to WARNING here so a
    # secret never ends up in logs/app.log by way of a logging config
    # change elsewhere, regardless of what level `ycf` itself is set to.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"ycf.{name}")
