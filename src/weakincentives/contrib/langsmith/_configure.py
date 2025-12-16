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

"""Auto-instrumentation for WINK with LangSmith."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...runtime.events import InProcessEventBus
from ...runtime.logging import StructuredLogger, get_logger
from ._config import LangSmithConfig
from ._telemetry import LangSmithTelemetryHandler

if TYPE_CHECKING:
    from collections.abc import Callable

logger: StructuredLogger = get_logger(
    __name__, context={"component": "langsmith_configure"}
)

# Global state for auto-instrumentation
_configured_handler: LangSmithTelemetryHandler | None = None
_original_init: Callable[..., None] | None = None


def configure_wink(  # noqa: PLR0913
    api_key: str | None = None,
    project: str | None = None,
    *,
    tracing_enabled: bool = True,
    trace_sample_rate: float = 1.0,
    hub_enabled: bool = True,
    async_upload: bool = True,
    flush_on_exit: bool = True,
    trace_native_tools: bool = True,
) -> LangSmithTelemetryHandler:
    """Enable automatic LangSmith tracing for all WINK evaluations.

    This function patches ``InProcessEventBus`` to automatically attach
    a ``LangSmithTelemetryHandler`` to every new bus instance.

    Call this once at application startup::

        from weakincentives.contrib.langsmith import configure_wink

        configure_wink()

        # All WINK evaluations are now traced
        response = adapter.evaluate(prompt, session=session)

    Args:
        api_key: LangSmith API key. Falls back to ``LANGCHAIN_API_KEY`` env var.
        project: LangSmith project name. Falls back to ``LANGCHAIN_PROJECT`` env var.
        tracing_enabled: Enable tracing. Default ``True``.
        trace_sample_rate: Fraction of traces to sample (0.0-1.0). Default ``1.0``.
        hub_enabled: Enable Prompt Hub integration. Default ``True``.
        async_upload: Upload traces asynchronously. Default ``True``.
        flush_on_exit: Flush pending traces on interpreter exit. Default ``True``.
        trace_native_tools: Trace Claude SDK native tools. Default ``True``.
            Set to ``False`` when using ``configure_claude_agent_sdk()`` to
            avoid duplicate traces.

    Returns:
        The ``LangSmithTelemetryHandler`` instance managing traces.

    Example with LangSmith's Claude SDK integration::

        from langsmith.integrations.claude_agent_sdk import configure_claude_agent_sdk
        from weakincentives.contrib.langsmith import configure_wink

        configure_claude_agent_sdk()  # Traces Claude SDK internals
        configure_wink(
            project="my-agent",
            trace_native_tools=False,  # Let Claude SDK handle tool tracing
        )
    """
    global _configured_handler, _original_init

    # Create config
    config = LangSmithConfig(
        api_key=api_key,
        project=project,
        tracing_enabled=tracing_enabled,
        trace_sample_rate=trace_sample_rate,
        hub_enabled=hub_enabled,
        async_upload=async_upload,
        flush_on_exit=flush_on_exit,
        trace_native_tools=trace_native_tools,
    )

    # Create handler
    handler = LangSmithTelemetryHandler(config)
    _configured_handler = handler

    # Patch InProcessEventBus if not already patched
    if _original_init is None:
        _original_init = InProcessEventBus.__init__

        def patched_init(self: InProcessEventBus) -> None:
            if _original_init is None:  # pragma: no cover - defensive check
                return
            _original_init(self)
            if _configured_handler is not None:
                _configured_handler.attach(self)

        InProcessEventBus.__init__ = patched_init  # type: ignore[method-assign]

        logger.info(
            "LangSmith auto-instrumentation enabled",
            event="langsmith_configured",
            context={
                "project": config.resolved_project(),
                "tracing_enabled": config.is_tracing_enabled(),
                "async_upload": config.async_upload,
            },
        )

    return handler


def unconfigure_wink() -> None:
    """Disable LangSmith auto-instrumentation.

    Restores the original ``InProcessEventBus.__init__`` and shuts down
    the telemetry handler.
    """
    global _configured_handler, _original_init

    # Restore original init
    if _original_init is not None:
        InProcessEventBus.__init__ = _original_init  # type: ignore[method-assign]
        _original_init = None

    # Shutdown handler
    if _configured_handler is not None:
        _configured_handler.shutdown()
        _configured_handler = None

    logger.info(
        "LangSmith auto-instrumentation disabled",
        event="langsmith_unconfigured",
    )


def get_configured_handler() -> LangSmithTelemetryHandler | None:
    """Get the currently configured telemetry handler.

    Returns:
        The active ``LangSmithTelemetryHandler`` or ``None`` if not configured.
    """
    return _configured_handler


__all__ = ["configure_wink", "get_configured_handler", "unconfigure_wink"]
