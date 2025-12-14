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

"""In-process event primitives for adapter telemetry."""

from __future__ import annotations

from dataclasses import field
from datetime import datetime
from threading import RLock
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from ...adapters._names import AdapterName
from ...dataclasses import FrozenDataclass
from ..annotations import SliceMeta, register_annotations
from ..logging import StructuredLogger, get_logger
from ._types import (
    ControlBus,
    EventBus,
    EventHandler,
    HandlerFailure,
    PublishResult,
    TelemetryBus,
    TokenUsage,
    ToolInvoked,
)

if TYPE_CHECKING:
    from ...prompt.overrides import PromptDescriptor
else:  # pragma: no cover - runtime alias avoids import cycles during module init
    PromptDescriptor = Any


def _describe_handler(handler: EventHandler) -> str:
    module_name = getattr(handler, "__module__", None)
    qualname = getattr(handler, "__qualname__", None)
    if isinstance(qualname, str):
        prefix = f"{module_name}." if isinstance(module_name, str) else ""
        return f"{prefix}{qualname}"
    return repr(handler)  # pragma: no cover - defensive fallback


logger: StructuredLogger = get_logger(__name__, context={"component": "event_bus"})


class InProcessEventBus:
    """Process-local event bus that delivers events synchronously."""

    def __init__(self) -> None:
        super().__init__()
        self._handlers: dict[type[object], list[EventHandler]] = {}
        self._lock = RLock()

    def subscribe(self, event_type: type[object], handler: EventHandler) -> None:
        with self._lock:
            handlers = self._handlers.setdefault(event_type, [])
            handlers.append(handler)

    def unsubscribe(self, event_type: type[object], handler: EventHandler) -> bool:
        with self._lock:
            handlers = self._handlers.get(event_type)
            if handlers is None:
                return False
            try:
                handlers.remove(handler)
            except ValueError:
                return False
            else:
                return True

    def publish(self, event: object) -> PublishResult:
        with self._lock:
            handlers = tuple(self._handlers.get(type(event), ()))
        invoked: list[EventHandler] = []
        failures: list[HandlerFailure] = []
        for handler in handlers:
            invoked.append(handler)
            try:
                handler(event)
            except Exception as error:
                logger.exception(
                    "Error delivering event.",
                    event="event_delivery_failed",
                    context={
                        "handler": _describe_handler(handler),
                        "event_type": type(event).__name__,
                    },
                )
                failures.append(HandlerFailure(handler=handler, error=error))

        return PublishResult(
            event=event,
            handlers_invoked=tuple(invoked),
            errors=tuple(failures),
        )


@FrozenDataclass()
class PromptExecuted:
    """Event emitted after an adapter finishes evaluating a prompt."""

    __slice_meta__ = SliceMeta(
        label="Prompt Executed",
        description="Event emitted after an adapter finishes evaluating a prompt.",
        icon="message-square",
        sort_key="created_at",
        sort_order="desc",
    )

    prompt_name: str = field(
        metadata={
            "display": "primary",
            "description": "Name of the prompt that was executed.",
        }
    )
    adapter: AdapterName = field(
        metadata={
            "display": "secondary",
            "description": "Adapter used for execution.",
        }
    )
    result: Any = field(
        metadata={
            "display": "primary",
            "format": "json",
            "description": "Result returned from the adapter.",
        }
    )
    session_id: UUID | None = field(
        metadata={
            "display": "secondary",
            "description": "Session identifier for the execution.",
        }
    )
    created_at: datetime = field(
        metadata={
            "display": "secondary",
            "description": "Timestamp of execution completion.",
        }
    )
    usage: TokenUsage | None = field(
        default=None,
        metadata={
            "display": "secondary",
            "format": "json",
            "description": "Token usage statistics for the request.",
        },
    )
    event_id: UUID = field(
        default_factory=uuid4,
        metadata={
            "display": "hidden",
            "description": "Unique identifier for this event.",
        },
    )


@FrozenDataclass()
class PromptRendered:
    """Event emitted immediately before dispatching a rendered prompt."""

    __slice_meta__ = SliceMeta(
        label="Prompt Rendered",
        description="Pre-dispatch prompt rendering event with full text.",
        icon="file-text",
        sort_key="created_at",
        sort_order="desc",
    )

    # Required fields (no defaults)
    prompt_ns: str = field(
        metadata={
            "display": "secondary",
            "description": "Namespace of the rendered prompt.",
        }
    )
    prompt_key: str = field(
        metadata={
            "display": "secondary",
            "description": "Key of the rendered prompt.",
        }
    )
    adapter: AdapterName = field(
        metadata={
            "display": "primary",
            "description": "Adapter processing this prompt.",
        }
    )
    rendered_prompt: str = field(
        metadata={
            "display": "primary",
            "format": "markdown",
            "description": "Full rendered prompt text.",
        }
    )
    created_at: datetime = field(
        metadata={
            "display": "secondary",
            "description": "Timestamp of rendering.",
        }
    )
    # Optional fields (with defaults)
    prompt_name: str | None = field(
        default=None,
        metadata={
            "display": "primary",
            "description": "Human-readable name of the prompt.",
        },
    )
    session_id: UUID | None = field(
        default=None,
        metadata={
            "display": "secondary",
            "description": "Session identifier.",
        },
    )
    render_inputs: tuple[Any, ...] = field(
        default=(),
        metadata={
            "display": "hidden",
            "format": "json",
            "description": "Inputs used to render the prompt.",
        },
    )
    descriptor: PromptDescriptor | None = field(
        default=None,
        metadata={
            "display": "hidden",
            "description": "Prompt metadata descriptor.",
        },
    )
    event_id: UUID = field(
        default_factory=uuid4,
        metadata={
            "display": "hidden",
            "description": "Unique event identifier.",
        },
    )


__all__ = [
    "ControlBus",
    "EventBus",
    "HandlerFailure",
    "InProcessEventBus",
    "PromptExecuted",
    "PromptRendered",
    "PublishResult",
    "TelemetryBus",
    "TokenUsage",
    "ToolInvoked",
]

# Register annotations at module import time
register_annotations(PromptExecuted)
register_annotations(PromptRendered)
register_annotations(ToolInvoked)
