import logging
import sys

import structlog

from app.core.settings import get_settings


def configure_logging() -> None:
    """JSON logs in prod (machine-readable), coloured console in dev. Idempotent."""
    settings = get_settings()
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.env == "prod"
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level.upper())
        ),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(stream=sys.stdout, level=settings.log_level.upper(), format="%(message)s")
