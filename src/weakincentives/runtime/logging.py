# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Structured logging helpers for :mod:`weakincentives`."""

from __future__ import annotations

import json
import logging
import logging.config
import os
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from typing import Any, cast, override

__all__ = [
    "StructuredLogger",
    "configure_logging",
    "get_logger",
]

_LOG_LEVEL_ENV = "WEAKINCENTIVES_LOG_LEVEL"
_LOG_FORMAT_ENV = "WEAKINCENTIVES_LOG_FORMAT"
_LEVEL_NAMES = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


class StructuredLogger(logging.LoggerAdapter[logging.Logger]):
    """Logger adapter enforcing a minimal structured event schema."""

    def __init__(
        self,
        logger: logging.Logger,
        *,
        context: Mapping[str, object] | None = None,
    ) -> None:
        base_context = dict(context) if context is not None else {}
        super().__init__(logger, base_context)

    def bind(self, **context: object) -> StructuredLogger:
        """Return a new adapter with ``context`` merged into the baseline payload."""

        base_extra = cast(Mapping[str, object], self.extra)
        merged: dict[str, object] = {**dict(base_extra), **context}
        return type(self)(self.logger, context=merged)

    @override
    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        extra_obj = kwargs.setdefault("extra", {})
        if extra_obj is None:
            extra_obj = {}
            kwargs["extra"] = extra_obj
        if not isinstance(
            extra_obj, MutableMapping
        ):  # pragma: no cover - defensive guard
            raise TypeError(
                "Structured logs require a mutable mapping for extra context."
            )

        extra_mapping = cast(MutableMapping[str, object], extra_obj)

        context_payload: dict[str, object] = dict(
            cast(Mapping[str, object], self.extra)
        )

        inline_context = kwargs.pop("context", None)
        if inline_context is not None:
            if isinstance(inline_context, Mapping):
                context_payload.update(cast(Mapping[str, object], inline_context))
            else:  # pragma: no cover - defensive guard
                raise TypeError("context must be a mapping when provided.")

        for key in tuple(extra_mapping.keys()):
            if key == "event":
                continue
            context_payload[key] = extra_mapping.pop(key)

        event_obj = kwargs.pop("event", None)
        if event_obj is None:
            event_obj = extra_mapping.pop("event", None)
        if not isinstance(event_obj, str):
            raise TypeError("Structured logs require an 'event' field.")

        extra_mapping.clear()
        extra_mapping.update(
            {
                "event": event_obj,
                "context": context_payload,
            }
        )
        return msg, kwargs


def get_logger(
    name: str,
    *,
    logger_override: logging.Logger
    | logging.LoggerAdapter[logging.Logger]
    | None = None,
    context: Mapping[str, object] | None = None,
) -> StructuredLogger:
    """Return a :class:`StructuredLogger` scoped to ``name``.

    When ``logger_override`` is provided, the returned adapter reuses the supplied
    logger and merges its contextual ``extra`` payload when available.
    """

    base_context: dict[str, object] = dict(context or {})
    base_logger: logging.Logger

    if isinstance(logger_override, StructuredLogger):
        base_logger = logger_override.logger
        base_context = {
            **dict(cast(Mapping[str, object], logger_override.extra)),
            **base_context,
        }
    elif isinstance(logger_override, logging.Logger):
        base_logger = logger_override
    elif isinstance(logger_override, logging.LoggerAdapter):
        base_logger = _unwrap_logger(logger_override)
        adapter_extra = getattr(logger_override, "extra", None)
        if isinstance(adapter_extra, Mapping):
            base_context = {
                **dict(cast(Mapping[str, object], adapter_extra)),
                **base_context,
            }
    else:
        base_logger = logging.getLogger(name)

    return StructuredLogger(base_logger, context=base_context)


def configure_logging(
    *,
    level: int | str | None = None,
    json_mode: bool | None = None,
    env: Mapping[str, str] | None = None,
    force: bool = False,
) -> None:
    """Configure the root logger with sensible defaults.

    ``level`` and ``json_mode`` can be supplied directly or via the
    ``WEAKINCENTIVES_LOG_LEVEL`` and ``WEAKINCENTIVES_LOG_FORMAT`` environment
    variables respectively (``json`` enables structured output, ``text`` keeps the
    plain formatter).

    The function avoids installing duplicate handlers when the host application has
    already configured logging unless ``force=True`` is supplied.
    """

    env = env or os.environ

    resolved_level = _coerce_level(level or env.get(_LOG_LEVEL_ENV) or logging.INFO)

    if json_mode is None:
        format_value = env.get(_LOG_FORMAT_ENV)
        if format_value is not None:
            json_mode = format_value.lower() == "json"
        else:
            json_mode = False

    root_logger = logging.getLogger()

    if root_logger.handlers and not force:
        root_logger.setLevel(resolved_level)
        return

    formatter_key = "json" if json_mode else "text"
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "text": {
                    "format": "%(asctime)s %(levelname)s %(name)s %(event)s %(message)s %(context)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "json": {
                    "()": "weakincentives.runtime.logging._JsonFormatter",
                },
            },
            "handlers": {
                "stderr": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                    "formatter": formatter_key,
                }
            },
            "root": {
                "handlers": ["stderr"],
                "level": resolved_level,
            },
        }
    )


class _JsonFormatter(logging.Formatter):
    """Formatter that renders structured records as compact JSON."""

    @override
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event
        context = getattr(record, "context", None)
        if context:
            payload["context"] = context
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=_json_default, separators=(",", ":"))


def _json_default(value: Any) -> Any:  # noqa: ANN401
    """Fallback serializer returning ``repr`` for unsupported values."""

    return repr(value)


_JSON_FORMATTER_CLASS = _JsonFormatter


def _unwrap_logger(adapter: logging.LoggerAdapter[Any]) -> logging.Logger:
    """Return the underlying :class:`logging.Logger` from an adapter."""

    logger_obj = adapter.logger
    if not isinstance(logger_obj, logging.Logger):  # pragma: no cover - defensive guard
        raise TypeError("LoggerAdapter.logger must be a logging.Logger instance.")
    return logger_obj


def _coerce_level(level: int | str | None) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        try:
            return _LEVEL_NAMES[level.upper()]
        except KeyError:  # pragma: no cover - defensive guard
            raise TypeError(f"Unknown log level: {level!r}") from None
    return logging.INFO
