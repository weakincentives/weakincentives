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

"""Prompt loop helpers extracted from ACPAdapter."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ...clock import AsyncSleeper, MonotonicClock
from ...deadlines import Deadline
from ...runtime.events.types import TokenUsage
from ..core import PromptEvaluationError
from ._events import dispatch_tool_invoked, extract_token_usage
from ._guardrails import accumulate_usage, check_task_completion

if TYPE_CHECKING:
    from ...budget import BudgetTracker
    from ...prompt import Prompt
    from ...runtime.run_context import RunContext
    from ...runtime.session.protocols import SessionProtocol
    from .._shared._visibility_signal import VisibilityExpansionSignal
    from .client import ACPClient

__all__ = [
    "drain_quiet_period",
    "extract_text",
    "run_prompt_loop",
    "send_prompt",
]

_MAX_DRAIN_S: float = 30.0


def _extract_chunk_text(chunk: Any) -> str:
    """Extract text content from an ACP update chunk.

    The ``content`` attribute may be a plain string, a ``TextContentBlock``
    with a ``.text`` attribute, or a list of content blocks.
    """
    raw = getattr(chunk, "content", "")
    if isinstance(raw, str):
        return raw
    # TextContentBlock or similar pydantic model
    text = getattr(raw, "text", None)
    if isinstance(text, str):
        return text
    # List of content blocks
    if isinstance(raw, list):
        return "".join(getattr(b, "text", str(b)) for b in raw if b)
    return str(raw) if raw else ""


def extract_text(client: ACPClient, *, emit_thought_chunks: bool) -> str | None:
    """Extract accumulated text from client message chunks."""
    if not client.message_chunks:
        return None

    parts: list[str] = []

    if emit_thought_chunks and client.thought_chunks:
        for chunk in client.thought_chunks:
            text = _extract_chunk_text(chunk)
            if text:
                parts.append(text)

    for chunk in client.message_chunks:
        text = _extract_chunk_text(chunk)
        if text:
            parts.append(text)

    return "".join(parts) if parts else None


async def drain_quiet_period(
    client: ACPClient,
    deadline: Deadline | None,
    *,
    quiet_period_ms: float,
    clock: MonotonicClock,
    async_sleeper: AsyncSleeper,
    max_drain_s: float = _MAX_DRAIN_S,
) -> None:
    """Wait until no new updates arrive for quiet_period_ms.

    If no updates have been received (``client.last_update_time is None``),
    the drain exits immediately.  A hard cap of ``max_drain_s`` prevents
    unbounded waiting when no deadline is set.
    """
    if client.last_update_time is None:
        return

    quiet_s = quiet_period_ms / 1000.0
    now = clock.monotonic()
    hard_cap = now + max_drain_s
    deadline_time = (
        clock.monotonic() + deadline.remaining().total_seconds() if deadline else None
    )
    if deadline_time is not None:
        effective_deadline = min(deadline_time, hard_cap)
    else:
        effective_deadline = hard_cap

    while True:
        now = clock.monotonic()
        if now >= effective_deadline:
            break

        snapshot = client.last_update_time
        if snapshot is None:
            break

        elapsed = now - snapshot
        if elapsed >= quiet_s:
            break

        wait_s = quiet_s - elapsed
        wait_s = min(wait_s, effective_deadline - now)
        await async_sleeper.async_sleep(wait_s)


async def send_prompt(
    conn: Any,
    acp_session_id: str,
    text: str,
    prompt_name: str,
    deadline: Deadline | None,
    text_content_block_cls: Any,
) -> Any:
    """Send a single prompt and return the response."""
    prompt_coro = conn.prompt(
        [text_content_block_cls(type="text", text=text)],
        session_id=acp_session_id,
    )
    timeout_s = deadline.remaining().total_seconds() if deadline else None
    try:
        return await asyncio.wait_for(prompt_coro, timeout=timeout_s)
    except TimeoutError:
        raise PromptEvaluationError(
            message="ACP prompt timed out (deadline expired)",
            prompt_name=prompt_name,
            phase="request",
        ) from None


async def run_prompt_loop(
    *,
    conn: Any,
    acp_session_id: str,
    client: ACPClient,
    session: SessionProtocol,
    adapter_name: str,
    prompt_name: str,
    prompt_text: str,
    deadline: Deadline | None,
    budget_tracker: BudgetTracker | None,
    prompt: Prompt[Any] | None,
    run_context: RunContext | None,
    visibility_signal: VisibilityExpansionSignal,
    structured_capture: Any,
    emit_thought_chunks: bool,
    quiet_period_ms: float,
    clock: MonotonicClock,
    async_sleeper: AsyncSleeper,
    detect_empty_response: Callable[[ACPClient, Any], None],
) -> tuple[str | None, TokenUsage | None]:
    """Run the prompt turn + task completion continuation loop."""
    from acp.schema import TextContentBlock

    max_continuation_rounds = 10
    continuation_round = 0
    current_prompt_text = prompt_text
    accumulated_text: str | None = None
    usage: TokenUsage | None = None

    while True:
        prompt_resp = await send_prompt(
            conn,
            acp_session_id,
            current_prompt_text,
            prompt_name,
            deadline,
            TextContentBlock,
        )
        await drain_quiet_period(
            client,
            deadline,
            quiet_period_ms=quiet_period_ms,
            clock=clock,
            async_sleeper=async_sleeper,
        )

        # Skip empty response check on first round only.
        need_empty_check = continuation_round == 0 and not (
            structured_capture is not None and structured_capture.called
        )
        if need_empty_check:
            detect_empty_response(client, prompt_resp)

        stored_exc = visibility_signal.get_and_clear()
        if stored_exc is not None:
            raise stored_exc

        for tc_id, tc_data in client.tool_call_tracker.items():
            dispatch_tool_invoked(
                session=session,
                adapter_name=adapter_name,
                prompt_name=prompt_name,
                run_context=run_context,
                tool_call_id=tc_id,
                title=tc_data.get("title", ""),
                status=tc_data.get("status", "completed"),
                rendered_output=tc_data.get("output", ""),
            )

        accumulated_text = extract_text(client, emit_thought_chunks=emit_thought_chunks)
        turn_usage = extract_token_usage(prompt_resp.usage if prompt_resp else None)
        if turn_usage is not None:
            usage = accumulate_usage(usage, turn_usage)

        should_continue, feedback = check_task_completion(
            prompt=prompt,
            session=session,
            accumulated_text=accumulated_text,
            deadline=deadline,
            budget_tracker=budget_tracker,
        )
        if should_continue and continuation_round < max_continuation_rounds:
            current_prompt_text = feedback  # type: ignore[assignment]
            continuation_round += 1
            client.reset_tracking()
            continue
        break

    return accumulated_text, usage
