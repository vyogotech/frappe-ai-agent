"""Structured logging configuration via structlog."""

from __future__ import annotations

import logging
import sys
from datetime import datetime

import structlog


def _short_local_timestamp(_logger, _method, event_dict):
    now = datetime.now()
    event_dict["timestamp"] = (
        now.strftime("%H:%M:%S") + f".{now.microsecond // 1000:03d}"
    )
    return event_dict


def _strip_ai_agent_logger_prefix(_logger, _method, event_dict):
    name = event_dict.get("logger")
    if isinstance(name, str) and name.startswith("ai_agent."):
        event_dict["logger"] = name[len("ai_agent.") :]
    return event_dict


def _drop_request_id(_logger, _method, event_dict):
    event_dict.pop("request_id", None)
    return event_dict


def setup_logging(level: str = "info", log_format: str = "json") -> None:
    """Configure structlog with JSON or console output."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    base_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_format == "console":
        # Strippers run after base_processors so add_logger_name has
        # populated `logger` and merge_contextvars has surfaced any
        # contextvar-bound request_id.
        shared_processors: list[structlog.types.Processor] = [
            *base_processors,
            _short_local_timestamp,
            _strip_ai_agent_logger_prefix,
            _drop_request_id,
        ]
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        shared_processors = [
            *base_processors,
            structlog.processors.TimeStamper(fmt="iso"),
        ]
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Quiet noisy libraries
    for name in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
