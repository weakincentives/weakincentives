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

"""Structured logging helpers for :mod:`weakincentives`.

This module provides structured logging infrastructure that enforces a consistent
event-based schema across all log messages. Key features:

- **Structured events**: All logs require an ``event`` field for categorization
- **Contextual binding**: Logger instances can carry bound context across calls
- **Dual formatters**: JSON output for production, human-readable text for development
- **Environment configuration**: Control level and format via environment variables

Example:
    >>> from weakincentives.runtime.logging import get_logger, configure_logging
    >>> configure_logging(level="DEBUG")
    >>> log = get_logger("myapp").bind(user_id="u123")
    >>> log.info("User logged in", event="auth.login", extra={"ip": "10.0.0.1"})
"""

from __future__ import annotations

import json
import logging
import logging.config
import os
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, cast, override

from ..types import JSONValue

if TYPE_CHECKING:
    from .run_context import RunContext

#: Type alias for structured log context payloads.
#:
#: A StructuredLogPayload is an immutable mapping of string keys to JSON-serializable
#: values. It represents the contextual data attached to log records, including
#: bound context from :meth:`StructuredLogger.bind` and inline context passed
#: to log calls.
#:
#: Supported value types: str, int, float, bool, None, list, dict (nested).
type StructuredLogPayload = Mapping[str, JSONValue]


class SupportsLogMessage(Protocol):
    """Protocol for objects that can be used as log message arguments.

    Any object implementing ``__str__`` satisfies this protocol. The logging
    infrastructure calls ``str()`` on message arguments to produce the final
    log output.

    This protocol enables type-safe logging with custom message objects that
    defer string formatting until the log level is actually enabled.
    """

    @override
    def __str__(self) -> str: ...


__all__ = [
    "JSONValue",
    "StructuredLogPayload",
    "StructuredLogger",
    "SupportsLogMessage",
    "bind_run_context",
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
    """Logger adapter enforcing a minimal structured event schema.

    StructuredLogger wraps a standard :class:`logging.Logger` and enforces that
    all log calls include an ``event`` field for categorization. Context can be
    bound at construction time or via :meth:`bind` to create child loggers with
    additional fields.

    Log calls support three ways to pass structured data:

    1. **event** (required): Pass as keyword argument or in ``extra``
    2. **context**: Merged from bound context + inline ``context=`` kwarg + ``extra`` fields
    3. **extra**: Additional fields merged into context (except ``event``)

    Example:
        >>> log = StructuredLogger(logging.getLogger("myapp"))
        >>> log.info("Request received", event="http.request", context={"path": "/api"})
        >>> child = log.bind(request_id="req-123")
        >>> child.info("Processing", event="http.process")  # includes request_id

    Args:
        logger: The underlying :class:`logging.Logger` to wrap.
        context: Optional initial context payload bound to all log calls.

    Raises:
        TypeError: If a log call is made without an ``event`` field.
    """

    def __init__(
        self,
        logger: logging.Logger,
        *,
        context: StructuredLogPayload | None = None,
    ) -> None:
        """Initialize the structured logger with optional bound context.

        Args:
            logger: The underlying Logger instance to wrap.
            context: Optional mapping of key-value pairs to include in every
                log record emitted by this adapter.
        """
        base_context: dict[str, JSONValue] = (
            dict(context) if context is not None else {}
        )
        super().__init__(logger, base_context)
        self._context: dict[str, JSONValue] = base_context

    def bind(self, **context: JSONValue) -> StructuredLogger:
        """Create a child logger with additional bound context.

        Returns a new StructuredLogger that inherits all context from this logger
        plus the additional key-value pairs provided. The original logger is
        unchanged. Context keys in ``context`` override keys from the parent.

        This is the primary mechanism for adding correlation IDs, request metadata,
        or other contextual information that should appear in all subsequent logs.

        Args:
            **context: Key-value pairs to add to the bound context. Values must
                be JSON-serializable (str, int, float, bool, None, list, dict).

        Returns:
            A new StructuredLogger with merged context.

        Example:
            >>> log = get_logger("myapp")
            >>> request_log = log.bind(request_id="req-123", user_id="u456")
            >>> request_log.info("Starting", event="request.start")
            >>> # All logs from request_log include request_id and user_id
        """
        merged: dict[str, JSONValue] = {**dict(self._context), **context}
        return type(self)(self.logger, context=merged)

    @staticmethod
    def _get_extra_mapping(
        kwargs: MutableMapping[str, object],
    ) -> MutableMapping[str, JSONValue]:
        """Extract or create the extra mapping from kwargs."""
        extra_value = kwargs.setdefault("extra", {})
        if extra_value is None:
            extra_mapping: MutableMapping[str, JSONValue] = {}
            kwargs["extra"] = extra_mapping
            return extra_mapping
        if isinstance(extra_value, MutableMapping):
            return cast(MutableMapping[str, JSONValue], extra_value)
        raise TypeError(  # pragma: no cover - defensive guard
            "Structured logs require a mutable mapping for extra context."
        )

    @staticmethod
    def _extract_event(
        kwargs: MutableMapping[str, object],
        extra_mapping: MutableMapping[str, JSONValue],
    ) -> str:
        """Extract the event field from kwargs or extra_mapping."""
        event_obj = kwargs.pop("event", None)
        if event_obj is None:
            event_obj = extra_mapping.pop("event", None)
        if not isinstance(event_obj, str):
            raise TypeError("Structured logs require an 'event' field.")
        return event_obj

    @override
    def process(
        self, msg: SupportsLogMessage, kwargs: MutableMapping[str, object]
    ) -> tuple[SupportsLogMessage, MutableMapping[str, object]]:
        """Process log call arguments into the structured schema.

        This method is called by the logging framework before each log emission.
        It merges bound context with inline context, extracts the required
        ``event`` field, and restructures ``extra`` to contain exactly
        ``{"event": ..., "context": {...}}``.

        Args:
            msg: The log message (supports lazy string formatting).
            kwargs: Mutable mapping of keyword arguments from the log call.

        Returns:
            Tuple of (message, modified kwargs) for the underlying logger.

        Raises:
            TypeError: If no ``event`` field is provided in kwargs or extra.
        """
        extra_mapping = self._get_extra_mapping(kwargs)
        context_payload: dict[str, JSONValue] = dict(self._context)

        inline_context = kwargs.pop("context", None)
        if inline_context is not None:
            if not isinstance(inline_context, Mapping):  # pragma: no cover - defensive
                raise TypeError("context must be a mapping when provided.")
            context_payload.update(cast(StructuredLogPayload, inline_context))

        for key in tuple(extra_mapping.keys()):
            if key != "event":
                context_payload[key] = extra_mapping.pop(key)

        event_obj = self._extract_event(kwargs, extra_mapping)

        extra_mapping.clear()
        extra_mapping.update({"event": event_obj, "context": context_payload})
        return msg, kwargs


def get_logger(
    name: str,
    *,
    logger_override: logging.Logger
    | logging.LoggerAdapter[logging.Logger]
    | None = None,
    context: StructuredLogPayload | None = None,
) -> StructuredLogger:
    """Create or wrap a logger as a StructuredLogger.

    This is the primary entry point for obtaining a structured logger. It either
    creates a new logger with the given name or wraps an existing logger/adapter,
    preserving any bound context from the original.

    Args:
        name: Logger name, typically ``__name__`` for module-scoped loggers.
            Ignored if ``logger_override`` is provided.
        logger_override: Optional existing logger or adapter to wrap. If a
            StructuredLogger or LoggerAdapter with ``extra``, its context is
            preserved and merged with ``context``.
        context: Optional initial context to bind to the returned logger.
            Merged with any context from ``logger_override`` (override wins).

    Returns:
        A StructuredLogger wrapping the resolved underlying Logger.

    Example:
        >>> # Module-scoped logger
        >>> log = get_logger(__name__)
        >>>
        >>> # Wrap existing logger with added context
        >>> stdlib_logger = logging.getLogger("legacy")
        >>> log = get_logger("wrapped", logger_override=stdlib_logger, context={"v": 2})
        >>>
        >>> # Chain from existing StructuredLogger
        >>> parent = get_logger("parent").bind(trace_id="abc")
        >>> child = get_logger("child", logger_override=parent)  # inherits trace_id
    """
    base_context: dict[str, JSONValue] = dict(context or {})
    base_logger: logging.Logger

    if isinstance(logger_override, StructuredLogger):
        base_logger = logger_override.logger
        base_context = {
            **dict(cast(StructuredLogPayload, logger_override.extra)),
            **base_context,
        }
    elif isinstance(logger_override, logging.Logger):
        base_logger = logger_override
    elif isinstance(logger_override, logging.LoggerAdapter):
        base_logger = _unwrap_logger(cast(_SupportsNestedLogger, logger_override))
        adapter_extra = getattr(logger_override, "extra", None)
        if isinstance(adapter_extra, Mapping):
            base_context = {
                **dict(cast(StructuredLogPayload, adapter_extra)),
                **base_context,
            }
    else:
        base_logger = logging.getLogger(name)

    return StructuredLogger(base_logger, context=base_context)


def bind_run_context(
    logger: StructuredLogger,
    run_context: RunContext | None,
) -> StructuredLogger:
    """Bind RunContext fields to a logger for request-scoped tracing.

    When run_context is provided, returns a new logger with run_id, request_id,
    attempt, worker_id, and trace context bound. When run_context is None,
    returns the original logger unchanged.

    This enables consistent correlation of logs across the entire request
    lifecycle: mailbox receipt -> prompt render -> provider call -> tool calls
    -> completion -> reply/DLQ.

    Args:
        logger: The structured logger to bind context to.
        run_context: Optional execution context with correlation identifiers.

    Returns:
        Logger with run context fields bound, or original logger if no context.

    Example:
        >>> from weakincentives.runtime import RunContext, get_logger
        >>> from weakincentives.runtime.logging import bind_run_context
        >>> base_log = get_logger(__name__)
        >>> run_ctx = RunContext(worker_id="worker-1")
        >>> log = bind_run_context(base_log, run_ctx)
        >>> log.info("Processing request", event="request.start")
    """
    if run_context is None:
        return logger
    return logger.bind(**run_context.to_log_context())


def configure_logging(
    *,
    level: int | str | None = None,
    json_mode: bool | None = None,
    env: Mapping[str, str] | None = None,
    force: bool = False,
) -> None:
    """Configure the root logger with sensible defaults for structured logging.

    Sets up the root logger with either a human-readable text formatter (default)
    or a machine-parseable JSON formatter. Configuration can be provided via
    arguments or environment variables.

    Environment Variables:
        WEAKINCENTIVES_LOG_LEVEL: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        WEAKINCENTIVES_LOG_FORMAT: Output format ("json" or "text")

    Args:
        level: Log level as int (e.g., ``logging.DEBUG``) or string (e.g., "DEBUG").
            Falls back to ``WEAKINCENTIVES_LOG_LEVEL`` env var, then INFO.
        json_mode: If True, use JSON formatter; if False, use text formatter.
            Falls back to ``WEAKINCENTIVES_LOG_FORMAT`` env var (``json`` = True).
        env: Environment mapping for variable lookup. Defaults to ``os.environ``.
        force: If True, reconfigure even if handlers already exist. If False
            (default), only set the level when handlers are already configured
            to avoid disrupting the host application's logging setup.

    Example:
        >>> # Basic setup for development
        >>> configure_logging(level="DEBUG")
        >>>
        >>> # Production setup with JSON output
        >>> configure_logging(level="INFO", json_mode=True)
        >>>
        >>> # Let environment control configuration
        >>> import os
        >>> os.environ["WEAKINCENTIVES_LOG_LEVEL"] = "WARNING"
        >>> os.environ["WEAKINCENTIVES_LOG_FORMAT"] = "json"
        >>> configure_logging()

    Note:
        Call this once at application startup. Library code should not call
        this function; let the application configure logging.
    """
    env = env or os.environ

    if level is not None:
        resolved_level = _coerce_level(level)
    else:
        resolved_level = _coerce_level(env.get(_LOG_LEVEL_ENV))

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
                    "()": "weakincentives.runtime.logging._TextFormatter",
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


class _SupportsNestedLogger(Protocol):
    logger: logging.Logger | logging.LoggerAdapter[logging.Logger]


class _TextFormatter(logging.Formatter):
    """Formatter that renders structured records as human-readable text.

    Handles log records from third-party libraries that lack the ``event``
    and ``context`` fields expected by the structured logging schema.
    """

    _FMT = "%(asctime)s %(levelname)s %(name)s"
    _DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt=self._DATEFMT)

    @override
    def format(self, record: logging.LogRecord) -> str:
        # Prepare record for formatting (sets record.message and record.asctime)
        record.message = record.getMessage()
        record.asctime = self.formatTime(record, self.datefmt)

        # Format base fields only (timestamp, level, logger name) - no exception
        base = self.formatMessage(record)

        # Extract structured fields if present
        event = getattr(record, "event", None)
        context = getattr(record, "context", None)

        # Build output parts
        parts = [base]
        if event is not None:
            parts.append(str(event))
        parts.append(record.message)
        if context:
            parts.append(str(context))

        result = " ".join(parts)

        # Append exception info if present (cache to avoid repeated formatting)
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            result = f"{result}\n{record.exc_text}"
        if record.stack_info:
            result = f"{result}\n{self.formatStack(record.stack_info)}"

        return result


class _JsonFormatter(logging.Formatter):
    """Formatter that renders structured records as compact JSON."""

    @override
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, JSONValue] = {
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


def _json_default(value: object) -> JSONValue:
    """Fallback serializer returning ``repr`` for unsupported values."""

    return repr(value)


_TEXT_FORMATTER_CLASS = _TextFormatter
_JSON_FORMATTER_CLASS = _JsonFormatter


def _unwrap_logger(adapter: _SupportsNestedLogger) -> logging.Logger:
    """Return the underlying :class:`logging.Logger` from an adapter."""

    logger_value = cast(object, adapter.logger)
    if isinstance(logger_value, logging.LoggerAdapter):
        return _unwrap_logger(cast(_SupportsNestedLogger, logger_value))
    if isinstance(logger_value, logging.Logger):
        return logger_value
    raise TypeError("LoggerAdapter.logger must be a logging.Logger instance.")


def _coerce_level(level: int | str | None) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        try:
            return _LEVEL_NAMES[level.upper()]
        except KeyError:  # pragma: no cover - defensive guard
            raise TypeError(f"Unknown log level: {level!r}") from None
    return logging.INFO
