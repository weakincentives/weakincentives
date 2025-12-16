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

"""Logging helpers shared by runnable demos."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping, Sequence
from typing import Any, cast

from weakincentives.runtime import EventBus
from weakincentives.runtime.events import (
    PromptExecuted,
    PromptRendered,
    TokenUsage,
    ToolInvoked,
)
from weakincentives.serde import dump

__all__ = [
    "attach_logging_subscribers",
    "configure_logging",
    "format_for_log",
]

_LOG_STRING_LIMIT = 256


def configure_logging() -> None:
    """Initialize root logging for interactive demos."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def attach_logging_subscribers(bus: EventBus) -> None:
    """Subscribe to prompt lifecycle events for structured logging."""

    bus.subscribe(PromptRendered, _print_rendered_prompt)
    bus.subscribe(ToolInvoked, _log_tool_invocation)
    bus.subscribe(PromptExecuted, _log_prompt_executed)


def _print_rendered_prompt(event: object) -> None:
    prompt_event = event
    if not isinstance(prompt_event, PromptRendered):  # pragma: no cover - defensive
        return

    prompt_label = prompt_event.prompt_name or (
        f"{prompt_event.prompt_ns}:{prompt_event.prompt_key}"
    )
    print(f"\n[prompt] Rendered prompt ({prompt_label})")
    print(prompt_event.rendered_prompt)
    print()


def _log_tool_invocation(event: object) -> None:
    tool_event = event
    if not isinstance(tool_event, ToolInvoked):  # pragma: no cover - defensive
        return

    # Handle both dataclass params (weakincentives tools) and dict params (SDK native)
    params = tool_event.params
    if isinstance(params, dict):
        params_repr = format_for_log(params)
    elif hasattr(params, "__dataclass_fields__"):
        params_repr = format_for_log(dump(params, exclude_none=True))
    else:
        params_repr = format_for_log({"params": params})

    # Handle both ToolResult dataclass and dict result (SDK native tools)
    result = tool_event.result
    if isinstance(result, dict):
        result_message = _truncate_for_log(str(result.get("content", result)))
        payload = result.get("value")
    elif hasattr(result, "message"):
        result_message = _truncate_for_log(result.message or "")
        payload = result.value if hasattr(result, "value") else None
    else:
        result_message = _truncate_for_log(str(result) if result else "")
        payload = None

    payload_repr: str | None = None
    if payload is not None:
        try:
            if hasattr(payload, "__dataclass_fields__"):
                payload_repr = format_for_log(dump(payload, exclude_none=True))
            else:
                payload_repr = format_for_log({"value": payload})
        except TypeError:
            payload_repr = format_for_log({"value": payload})

    lines = [
        f"{tool_event.name} ({tool_event.prompt_name})",
        f"  params: {params_repr}",
        f"  result: {result_message}",
    ]
    lines.append(f"  {_format_usage_for_log(tool_event.usage)}")
    if payload_repr is not None:
        lines.append(f"  payload: {payload_repr}")

    print("\n[tool] " + "\n".join(lines))


def _log_prompt_executed(event: object) -> None:
    prompt_event = event
    if not isinstance(prompt_event, PromptExecuted):  # pragma: no cover - defensive
        return

    print("\n[prompt] Execution complete")
    print(f"  {_format_usage_for_log(prompt_event.usage)}\n")


def _format_usage_for_log(usage: TokenUsage | None) -> str:
    if usage is None:
        return "token usage: (not reported)"

    parts: list[str] = []
    if usage.input_tokens is not None:
        parts.append(f"input={usage.input_tokens}")
    if usage.output_tokens is not None:
        parts.append(f"output={usage.output_tokens}")
    if usage.cached_tokens is not None:
        parts.append(f"cached={usage.cached_tokens}")

    total = usage.total_tokens
    if total is not None:
        parts.append(f"total={total}")

    if not parts:
        return "token usage: (not reported)"
    return f"token usage: {', '.join(parts)}"


def format_for_log(payload: object, *, limit: int = _LOG_STRING_LIMIT) -> str:
    """Serialize payloads for concise console logging."""

    serializable = _coerce_for_log(payload)
    try:
        rendered = json.dumps(serializable, ensure_ascii=False)
    except TypeError:
        rendered = repr(serializable)
    return _truncate_for_log(rendered, limit=limit)


def _truncate_for_log(text: str, *, limit: int = _LOG_STRING_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"


def _coerce_for_log(payload: object) -> object:
    if payload is None or isinstance(payload, (str, int, float, bool)):
        return payload
    if isinstance(payload, Mapping):
        mapping_payload: Mapping[Any, Any] = cast(Mapping[Any, Any], payload)
        return {
            str(key): _coerce_for_log(value) for key, value in mapping_payload.items()
        }
    if isinstance(payload, Sequence) and not isinstance(
        payload, (str, bytes, bytearray)
    ):
        sequence_payload: Sequence[Any] = cast(Sequence[Any], payload)
        return [_coerce_for_log(item) for item in sequence_payload]
    if hasattr(payload, "__dataclass_fields__"):
        return dump(payload, exclude_none=True)
    if isinstance(payload, set):
        set_payload: set[Any] = cast(set[Any], payload)
        return sorted((_coerce_for_log(item) for item in set_payload), key=str)
    return str(payload)
