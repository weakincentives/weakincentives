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

"""ACP protocol execution helpers.

Free functions extracted from ACPAdapter for protocol-level operations:
handshake, prompt sending, drain, text extraction, and structured output.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import cast

from ...clock import AsyncSleeper, MonotonicClock
from ...deadlines import Deadline
from ...prompt import RenderedPrompt
from ...prompt.structured_output import OutputParseError, parse_structured_output
from ...runtime.logging import StructuredLogger, get_logger
from ..core import PromptEvaluationError
from .client import ACPClient
from .config import ACPAdapterConfig, ACPClientConfig

__all__ = [
    "build_env",
    "configure_session",
    "drain_quiet_period",
    "extract_chunk_text",
    "extract_client_text",
    "handshake",
    "resolve_structured_output",
    "send_prompt",
]

logger: StructuredLogger = get_logger(__name__, context={"component": "acp_execution"})

_MAX_DRAIN_S: float = 30.0


def extract_chunk_text(chunk: object) -> str:
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


def build_env(client_config: ACPClientConfig) -> dict[str, str] | None:
    """Build merged environment variables.

    When ``config.env`` is set, the full ``os.environ`` is forwarded with
    config entries taking precedence.  This mirrors stdlib
    ``subprocess.Popen`` behaviour where ``env=None`` inherits the parent
    environment.  Returning ``None`` (no config env) lets the subprocess
    inherit the parent env via the stdlib default.
    """
    if not client_config.env:
        return None
    return {**os.environ, **client_config.env}


async def handshake(
    conn: object,
    effective_cwd: str,
    mcp_servers: list[object],
    *,
    client_config: ACPClientConfig,
) -> tuple[str, list[object]]:
    """Initialize and create session. Returns (session_id, available_models)."""
    from acp import PROTOCOL_VERSION
    from acp.schema import (
        ClientCapabilities,
        FileSystemCapability,
        Implementation,
    )

    _ = await conn.initialize(  # type: ignore[attr-defined]
        protocol_version=PROTOCOL_VERSION,
        client_capabilities=ClientCapabilities(
            fs=FileSystemCapability(
                read_text_file=client_config.allow_file_reads,
                write_text_file=client_config.allow_file_writes,
            ),
            terminal=False,
        ),
        client_info=Implementation(
            name="wink",
            title="WINK",
            version="0.1.0",
        ),
    )

    new_session_resp = await conn.new_session(  # type: ignore[attr-defined]
        cwd=effective_cwd,
        mcp_servers=mcp_servers,
    )
    acp_session_id: str = new_session_resp.session_id

    available_models = (
        new_session_resp.models.available_models if new_session_resp.models else []
    )

    return acp_session_id, available_models


async def configure_session(
    conn: object,
    session_id: str,
    *,
    adapter_config: ACPAdapterConfig,
) -> Exception | None:
    """Configure model and mode on the session (best-effort).

    Returns:
        The mode-setting exception if one occurred, or None.
    """
    if adapter_config.model_id:
        try:
            await conn.set_session_model(  # type: ignore[attr-defined]
                session_id=session_id,
                model_id=adapter_config.model_id,
            )
        except Exception as err:
            logger.warning(
                "acp.set_model.failed",
                event="set_model.failed",
                context={"error": str(err)},
            )

    mode_error: Exception | None = None
    if adapter_config.mode_id:
        try:
            await conn.set_session_mode(  # type: ignore[attr-defined]
                session_id=session_id,
                mode_id=adapter_config.mode_id,
            )
        except Exception as err:
            mode_error = err

    return mode_error


async def send_prompt(
    *,
    conn: object,
    acp_session_id: str,
    text: str,
    prompt_name: str,
    deadline: Deadline | None,
) -> object:
    """Send a single prompt and return the response."""
    from acp.schema import TextContentBlock

    prompt_coro = conn.prompt(  # type: ignore[attr-defined]
        [TextContentBlock(type="text", text=text)],
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


async def drain_quiet_period(
    client: ACPClient,
    deadline: Deadline | None,
    *,
    quiet_period_ms: float,
    clock: MonotonicClock,
    async_sleeper: AsyncSleeper,
) -> None:
    """Wait until no new updates arrive for quiet_period_ms.

    If no updates have been received (``client.last_update_time is None``),
    the drain exits immediately.  A hard cap of ``_MAX_DRAIN_S`` prevents
    unbounded waiting when no deadline is set.
    """
    if client.last_update_time is None:
        return

    quiet_s = quiet_period_ms / 1000.0
    now = clock.monotonic()
    hard_cap = now + _MAX_DRAIN_S
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


def extract_client_text(client: ACPClient, *, emit_thought_chunks: bool) -> str | None:
    """Extract accumulated text from client message chunks."""
    if not client.message_chunks:
        return None

    parts: list[str] = []

    if emit_thought_chunks and client.thought_chunks:
        for chunk in client.thought_chunks:
            text = extract_chunk_text(chunk)
            if text:
                parts.append(text)

    for chunk in client.message_chunks:
        text = extract_chunk_text(chunk)
        if text:
            parts.append(text)

    return "".join(parts) if parts else None


def resolve_structured_output[OutputT](
    accumulated_text: str | None,
    rendered: RenderedPrompt[OutputT],
    prompt_name: str,
    structured_capture: object,
) -> OutputT | None:
    """Resolve structured output from capture or text."""
    if structured_capture is not None and structured_capture.called:
        try:
            return cast(
                OutputT,
                parse_structured_output(json.dumps(structured_capture.data), rendered),
            )
        except (OutputParseError, TypeError, ValueError) as error:
            raise PromptEvaluationError(
                message=f"Failed to parse structured output: {error}",
                prompt_name=prompt_name,
                phase="response",
            ) from error

    if accumulated_text:
        try:
            return cast(OutputT, parse_structured_output(accumulated_text, rendered))
        except (OutputParseError, TypeError, ValueError) as error:
            raise PromptEvaluationError(
                message=f"Failed to parse structured output: {error}",
                prompt_name=prompt_name,
                phase="response",
                provider_payload={"raw_text": accumulated_text[:2000]},
            ) from error

    raise PromptEvaluationError(
        message="Structured output required but model did not produce output",
        prompt_name=prompt_name,
        phase="response",
    )
