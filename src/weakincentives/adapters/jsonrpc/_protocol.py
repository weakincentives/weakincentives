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

"""Generic JSON-RPC protocol orchestration.

Provides the turn-based execution loop shared by all JSON-RPC adapter
subclasses.  Provider-specific protocol details (initialization, turn
start, notification processing, server request handling) are delegated
to hook methods on the adapter instance.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, cast

from ...budget import BudgetTracker
from ...clock import SYSTEM_CLOCK, AsyncSleeper
from ...deadlines import Deadline
from ...runtime.events.types import TokenUsage
from ...runtime.run_context import RunContext
from ...runtime.session.protocols import SessionProtocol
from .._shared._bridge import BridgedTool
from .._shared._guardrails import accumulate_usage
from .._shared._visibility_signal import VisibilityExpansionSignal
from ..core import PromptEvaluationError
from ._types import JsonRpcMessage
from .client import JsonRpcClient, JsonRpcClientError

if TYPE_CHECKING:
    from ...prompt.protocols import PromptProtocol

# Use TYPE_CHECKING + string annotation won't work here because ty
# detects the cycle even under TYPE_CHECKING.  Instead we use a
# module-level type alias that resolves lazily.
type _Adapter = Any  # _Adapter at runtime


def deadline_remaining_s(deadline: Deadline | None, prompt_name: str) -> float | None:
    """Return remaining seconds from deadline, or None if no deadline.

    Raises PromptEvaluationError if the deadline has already expired.
    """
    if deadline is None:
        return None
    remaining = deadline.remaining().total_seconds()
    if remaining <= 0:
        raise PromptEvaluationError(
            message="Deadline expired during setup",
            prompt_name=prompt_name,
            phase="request",
        )
    return remaining


async def execute_protocol(  # noqa: PLR0913
    *,
    adapter: _Adapter,
    client: JsonRpcClient,
    session: SessionProtocol,
    adapter_name: str,
    prompt_name: str,
    prompt_text: str,
    tool_lookup: dict[str, BridgedTool],
    deadline: Deadline | None,
    budget_tracker: BudgetTracker | None,
    run_context: RunContext | None,
    visibility_signal: VisibilityExpansionSignal,
    async_sleeper: AsyncSleeper = SYSTEM_CLOCK,
    prompt: PromptProtocol[Any] | None = None,
) -> tuple[str | None, TokenUsage | None]:
    """Execute the generic JSON-RPC protocol lifecycle.

    Delegates provider-specific steps to hook methods on *adapter*.
    Returns (accumulated_text, usage).
    """
    await client.start()
    bridge = adapter._create_transcript_bridge(session, prompt_name)

    try:
        session_state = await adapter._initialize_session(
            client,
            deadline=deadline,
            prompt_name=prompt_name,
        )

        # Turn + Stream with task completion continuation loop
        max_continuation_rounds = 10
        continuation_round = 0
        current_prompt_text = prompt_text
        accumulated_text: str | None = None
        usage: TokenUsage | None = None

        while True:
            adapter._on_user_message_for_transcript(bridge, current_prompt_text)

            timeout = deadline_remaining_s(deadline, prompt_name)
            try:
                turn_state = await adapter._start_turn(
                    client,
                    session_state,
                    current_prompt_text,
                    deadline=deadline,
                    prompt_name=prompt_name,
                    timeout=timeout,
                )
            except JsonRpcClientError as error:
                raise PromptEvaluationError(
                    message=str(error),
                    prompt_name=prompt_name,
                    phase="request",
                ) from error

            turn_text, turn_usage = await stream_turn(
                adapter=adapter,
                client=client,
                session=session,
                adapter_name=adapter_name,
                prompt_name=prompt_name,
                session_state=session_state,
                turn_state=turn_state,
                tool_lookup=tool_lookup,
                deadline=deadline,
                run_context=run_context,
                bridge=bridge,
                visibility_signal=visibility_signal,
                async_sleeper=async_sleeper,
                prompt=prompt,
            )
            accumulated_text = turn_text
            if turn_usage is not None:
                usage = accumulate_usage(usage, turn_usage)

            # Skip continuation when visibility expansion is pending.
            if visibility_signal.is_set():
                break

            should_continue, feedback = adapter._check_task_completion(
                prompt=prompt,
                session=session,
                accumulated_text=accumulated_text,
                deadline=deadline,
                budget_tracker=budget_tracker,
            )
            if should_continue and continuation_round < max_continuation_rounds:
                current_prompt_text = feedback
                continuation_round += 1
                continue
            break
    finally:
        adapter._stop_transcript_bridge(bridge)

    # Visibility signal
    stored_exc = visibility_signal.get_and_clear()
    if stored_exc is not None:
        raise stored_exc

    return accumulated_text, usage


async def stream_turn(  # noqa: PLR0913
    *,
    adapter: _Adapter,
    client: JsonRpcClient,
    session: SessionProtocol,
    adapter_name: str,
    prompt_name: str,
    session_state: object,
    turn_state: object,
    tool_lookup: dict[str, BridgedTool],
    deadline: Deadline | None,
    run_context: RunContext | None,
    bridge: object | None = None,
    visibility_signal: VisibilityExpansionSignal | None = None,
    async_sleeper: AsyncSleeper = SYSTEM_CLOCK,
    prompt: PromptProtocol[Any] | None = None,
) -> tuple[str | None, TokenUsage | None]:
    """Stream turn notifications until turn/completed.

    Returns (accumulated_text, token_usage).
    """
    accumulated_text = ""
    usage: TokenUsage | None = None

    watchdog_task = create_deadline_watchdog(
        adapter, client, session_state, turn_state, deadline, async_sleeper
    )

    try:
        accumulated_text, usage = await consume_messages(
            adapter=adapter,
            client=client,
            session=session,
            adapter_name=adapter_name,
            prompt_name=prompt_name,
            tool_lookup=tool_lookup,
            run_context=run_context,
            accumulated_text=accumulated_text,
            usage=usage,
            bridge=bridge,
            visibility_signal=visibility_signal,
            prompt=prompt,
            deadline=deadline,
        )
    finally:
        if watchdog_task is not None:
            _ = watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog_task

    return accumulated_text or None, usage


def create_deadline_watchdog(  # noqa: PLR0913, PLR0917
    adapter: _Adapter,
    client: JsonRpcClient,
    session_state: object,
    turn_state: object,
    deadline: Deadline | None,
    async_sleeper: AsyncSleeper = SYSTEM_CLOCK,
) -> asyncio.Task[None] | None:
    """Create a watchdog task if a deadline is set."""
    if deadline is None:
        return None
    remaining = deadline.remaining().total_seconds()
    if remaining <= 0:
        return None
    return asyncio.create_task(
        _deadline_watchdog(
            adapter, client, session_state, turn_state, remaining, async_sleeper
        )
    )


async def _deadline_watchdog(  # noqa: PLR0913, PLR0917
    adapter: _Adapter,
    client: JsonRpcClient,
    session_state: object,
    turn_state: object,
    remaining_seconds: float,
    async_sleeper: AsyncSleeper = SYSTEM_CLOCK,
) -> None:
    """Sleep until deadline, then send interrupt via adapter hook."""
    await async_sleeper.async_sleep(remaining_seconds)
    with contextlib.suppress(JsonRpcClientError):
        await adapter._send_interrupt(client, session_state, turn_state)


def raise_for_terminal_notification(
    kind: str,
    value: str,
    prompt_name: str,
    message: JsonRpcMessage,
    adapter: _Adapter,
) -> None:
    """Raise PromptEvaluationError for terminal notification kinds."""
    if kind == "error":
        phase = adapter._map_error_phase(message)
        raise PromptEvaluationError(
            message=value,
            prompt_name=prompt_name,
            phase=phase,
        )
    if kind == "interrupted":
        raise PromptEvaluationError(
            message="Turn was interrupted (deadline or user)",
            prompt_name=prompt_name,
            phase="request",
        )


async def consume_messages(  # noqa: PLR0913
    *,
    adapter: _Adapter,
    client: JsonRpcClient,
    session: SessionProtocol,
    adapter_name: str,
    prompt_name: str,
    tool_lookup: dict[str, BridgedTool],
    run_context: RunContext | None,
    accumulated_text: str,
    usage: TokenUsage | None,
    bridge: object | None = None,
    visibility_signal: VisibilityExpansionSignal | None = None,
    prompt: PromptProtocol[Any] | None = None,
    deadline: Deadline | None = None,
) -> tuple[str, TokenUsage | None]:
    """Consume messages from the client until turn is completed."""
    turn_completed = False
    async for raw_message in client.read_messages():
        message = cast(JsonRpcMessage, raw_message)
        if "id" in message and "method" in message:
            await adapter._handle_server_request(
                client,
                message,
                tool_lookup,
                bridge=bridge,
                prompt=prompt,
                session=session,
                deadline=deadline,
            )
            # Break early if a tool call triggered visibility expansion.
            if visibility_signal is not None and visibility_signal.is_set():
                turn_completed = True
                break
            continue

        # Forward notification to transcript bridge.
        adapter._on_notification_for_transcript(bridge, message)

        result = adapter._process_notification(
            message, session, adapter_name, prompt_name, run_context
        )
        if result is None:
            continue

        accumulated_text, usage, done = apply_notification(
            result, message, accumulated_text, usage, prompt_name, adapter
        )
        if done:
            turn_completed = True
            break
    if not turn_completed:
        raise PromptEvaluationError(
            message="Stream ended before turn completion",
            prompt_name=prompt_name,
            phase="response",
        )
    return accumulated_text, usage


def apply_notification(  # noqa: PLR0913, PLR0917
    result: tuple[str, str],
    message: JsonRpcMessage,
    accumulated_text: str,
    usage: TokenUsage | None,
    prompt_name: str,
    adapter: _Adapter,
) -> tuple[str, TokenUsage | None, bool]:
    """Apply a single notification result. Returns (text, usage, done)."""
    kind, value = result
    if kind == "text":
        return value, usage, False
    if kind == "delta":
        return accumulated_text + value, usage, False
    if kind == "usage":
        return (
            accumulated_text,
            adapter._extract_token_usage(message.get("params", {})),
            False,
        )
    if kind == "done":
        return accumulated_text, usage, True
    raise_for_terminal_notification(kind, value, prompt_name, message, adapter)
    return accumulated_text, usage, False  # pragma: no cover
