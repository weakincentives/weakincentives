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

"""Client wrapper for ClaudeSDKClient with lifecycle management.

This module provides a wrapper around the Claude Agent SDK's ClaudeSDKClient
that integrates with weakincentives' session management, deadline/budget
enforcement, and event publishing.

The client-based approach enables:
- Real cancellation via interrupt()
- Streaming progress through receive_response()
- Clean lifecycle management with connect/disconnect
- Proper termination at ResultMessage
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from types import TracebackType
from typing import TYPE_CHECKING, Any

from ...budget import BudgetTracker
from ...deadlines import Deadline
from ...runtime.logging import StructuredLogger, get_logger

if TYPE_CHECKING:
    from ._hooks import HookContext

__all__ = [
    "ClientConfig",
    "ClientSession",
    "QueryResult",
]

logger: StructuredLogger = get_logger(__name__, context={"component": "sdk_client"})


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True, frozen=True)
class ClientConfig:
    """Configuration for ClaudeSDKClient wrapper.

    Attributes:
        model: Claude model identifier.
        cwd: Working directory for SDK operations.
        permission_mode: Tool permission handling mode.
        max_turns: Maximum conversation turns.
        output_format: Structured output format (JSON schema).
        allowed_tools: Tools Claude can use.
        disallowed_tools: Tools to block.
        suppress_stderr: Hide CLI stderr output.
        env: Environment variables for the SDK process.
        setting_sources: SDK setting sources.
        mcp_servers: MCP server configurations.
        hooks: Hook configurations.
    """

    model: str
    cwd: str | None = None
    permission_mode: str = "bypassPermissions"
    max_turns: int | None = None
    output_format: dict[str, Any] | None = None
    allowed_tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] = ()
    suppress_stderr: bool = True
    env: dict[str, str] | None = None
    setting_sources: list[str] | None = None
    mcp_servers: dict[str, Any] | None = None
    hooks: dict[str, list[Any]] | None = None


@dataclass(slots=True)
class QueryResult:
    """Result of a client query.

    Attributes:
        messages: All messages received during the query.
        result_text: Extracted result text from ResultMessage.
        structured_output: Parsed structured output if available.
        input_tokens: Total input tokens consumed.
        output_tokens: Total output tokens consumed.
        stop_reason: Reason for stopping (end_turn, interrupted, etc.).
        interrupted: Whether the query was interrupted.
    """

    messages: list[Any]
    result_text: str | None = None
    structured_output: dict[str, Any] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    interrupted: bool = False


def _process_message(
    message: object,
    result: QueryResult,
    result_message_type: type[object],
) -> None:
    """Process a single message and update the result.

    Extracts relevant data from the message based on its type.
    """
    # Extract usage from any message type
    usage_attr = getattr(message, "usage", None)
    if usage_attr is not None and isinstance(usage_attr, dict):
        result.input_tokens += int(usage_attr.get("input_tokens") or 0)
        result.output_tokens += int(usage_attr.get("output_tokens") or 0)

    # Handle ResultMessage specifically
    if isinstance(message, result_message_type):
        result_attr = getattr(message, "result", None)
        if result_attr:
            result.result_text = str(result_attr)

        structured_attr = getattr(message, "structured_output", None)
        if structured_attr is not None:
            result.structured_output = structured_attr

        stop_attr = getattr(message, "stop_reason", None)
        if stop_attr is not None:
            result.stop_reason = str(stop_attr)
        else:
            result.stop_reason = "end_turn"


def _build_options_kwargs(config: ClientConfig) -> dict[str, Any]:  # noqa: C901
    """Build options kwargs dict from ClientConfig."""
    options_kwargs: dict[str, Any] = {"model": config.model}

    if config.cwd:
        options_kwargs["cwd"] = config.cwd

    if config.permission_mode:
        options_kwargs["permission_mode"] = config.permission_mode

    if config.max_turns:
        options_kwargs["max_turns"] = config.max_turns

    if config.output_format:
        options_kwargs["output_format"] = config.output_format

    if config.allowed_tools is not None:
        options_kwargs["allowed_tools"] = list(config.allowed_tools)

    if config.disallowed_tools:
        options_kwargs["disallowed_tools"] = list(config.disallowed_tools)

    if config.suppress_stderr:
        options_kwargs["stderr"] = lambda _: None

    if config.env:
        options_kwargs["env"] = config.env

    if config.setting_sources is not None:
        options_kwargs["setting_sources"] = config.setting_sources

    if config.mcp_servers:
        options_kwargs["mcp_servers"] = config.mcp_servers

    if config.hooks:
        options_kwargs["hooks"] = config.hooks

    return options_kwargs


class ClientSession:
    """Manages a ClaudeSDKClient session with lifecycle control.

    This class wraps the Claude Agent SDK's ClaudeSDKClient to provide:
    - Clean connect/disconnect lifecycle
    - Query execution with streaming response handling
    - Real interrupt() support for cancellation
    - Integration with deadline/budget enforcement

    Example:
        >>> async with ClientSession(config, hook_context) as session:
        ...     result = await session.query("Do something")
        ...     if result.interrupted:
        ...         print("Query was interrupted")
    """

    def __init__(
        self,
        config: ClientConfig,
        hook_context: HookContext,
        *,
        deadline: Deadline | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> None:
        """Initialize the client session.

        Args:
            config: Client configuration.
            hook_context: Hook context for state management.
            deadline: Optional deadline for execution timeout.
            budget_tracker: Optional budget tracker for token limits.
        """
        self._config = config
        self._hook_context = hook_context
        self._deadline = deadline
        self._budget_tracker = budget_tracker
        self._client: Any | None = None
        self._interrupt_requested = False
        self._connected = False

    async def __aenter__(self) -> ClientSession:
        """Enter async context and connect."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context and disconnect."""
        await self.disconnect()

    async def connect(self) -> None:
        """Connect the client.

        Imports the SDK, creates the client with configured options, and
        establishes the connection. The client is now ready for queries.

        Raises:
            ImportError: If claude-agent-sdk is not installed.
        """
        if self._connected:
            return

        sdk = _import_sdk()
        ClaudeSDKClient = sdk.ClaudeSDKClient
        ClaudeAgentOptions = sdk.types.ClaudeAgentOptions

        options_kwargs = _build_options_kwargs(self._config)
        options = ClaudeAgentOptions(**options_kwargs)

        logger.debug(
            "sdk_client.connecting",
            event="client.connecting",
            context={"model": self._config.model},
        )

        self._client = ClaudeSDKClient(options=options)
        await self._client.connect()
        self._connected = True

        logger.debug(
            "sdk_client.connected",
            event="client.connected",
            context={"model": self._config.model},
        )

    async def disconnect(self) -> None:
        """Disconnect the client.

        Gracefully closes the connection. Safe to call multiple times.
        """
        if not self._connected or self._client is None:
            return  # pragma: no cover

        logger.debug(
            "sdk_client.disconnecting",
            event="client.disconnecting",
            context={},
        )

        with suppress(Exception):
            await self._client.disconnect()

        self._connected = False
        self._client = None

        logger.debug(
            "sdk_client.disconnected",
            event="client.disconnected",
            context={},
        )

    async def query(
        self,
        prompt: str,
        *,
        session_id: str = "default",
    ) -> QueryResult:
        """Execute a query and collect all responses.

        Sends the prompt to Claude and iterates through receive_response()
        until a ResultMessage is received or the query is interrupted.

        Args:
            prompt: The prompt text to send.
            session_id: Session identifier for conversation continuity.

        Returns:
            QueryResult with all messages and extracted data.

        Raises:
            RuntimeError: If client is not connected.
        """
        if not self._connected or self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        self._interrupt_requested = False

        logger.info(
            "sdk_client.query.start",
            event="client.query.start",
            context={"session_id": session_id, "prompt_length": len(prompt)},
        )

        start_time = _utcnow()

        # Send the query
        await self._client.query(prompt, session_id=session_id)

        # Get ResultMessage type for processing
        sdk = _import_sdk()
        ResultMessage = sdk.types.ResultMessage

        # Collect messages, checking for interrupt conditions
        messages: list[Any] = []
        result = QueryResult(messages=messages)

        try:
            async for message in self._client.receive_response():
                # Check for interrupt conditions before processing
                if self._should_interrupt():  # pragma: no cover
                    await self._perform_interrupt()
                    result.interrupted = True
                    result.stop_reason = "interrupted"
                    break

                messages.append(message)
                _process_message(message, result, ResultMessage)

        except asyncio.CancelledError:  # pragma: no cover
            result.interrupted = True
            result.stop_reason = "cancelled"
            raise

        duration_ms = int((_utcnow() - start_time).total_seconds() * 1000)

        logger.info(
            "sdk_client.query.complete",
            event="client.query.complete",
            context={
                "session_id": session_id,
                "message_count": len(messages),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "interrupted": result.interrupted,
                "duration_ms": duration_ms,
            },
        )

        return result

    async def interrupt(self) -> None:
        """Request interruption of the current query.

        Signals the SDK to stop the current operation. This method is
        safe to call from another task or coroutine.
        """
        self._interrupt_requested = True

        if self._client is not None and self._connected:
            logger.info(
                "sdk_client.interrupt",
                event="client.interrupt",
                context={},
            )
            with suppress(Exception):
                await self._client.interrupt()

    def _should_interrupt(self) -> bool:
        """Check if the query should be interrupted.

        Returns True if:
        - interrupt() was called
        - Deadline has expired
        - Budget is exhausted
        """
        if self._interrupt_requested:
            return True

        if self._deadline and self._deadline.remaining().total_seconds() <= 0:
            logger.warning(
                "sdk_client.deadline_exceeded",
                event="client.deadline_exceeded",
                context={},
            )
            return True

        if self._budget_tracker is not None:
            budget = self._budget_tracker.budget
            consumed = self._budget_tracker.consumed
            consumed_total = (consumed.input_tokens or 0) + (
                consumed.output_tokens or 0
            )
            if (
                budget.max_total_tokens is not None
                and consumed_total >= budget.max_total_tokens
            ):
                logger.warning(
                    "sdk_client.budget_exhausted",
                    event="client.budget_exhausted",
                    context={
                        "consumed": consumed_total,
                        "max": budget.max_total_tokens,
                    },
                )
                return True

        return False

    async def _perform_interrupt(self) -> None:  # pragma: no cover
        """Perform the interrupt call on the client."""
        if self._client is not None and self._connected:
            with suppress(Exception):
                await self._client.interrupt()


def _import_sdk() -> Any:  # noqa: ANN401  # pragma: no cover
    """Import the Claude Agent SDK, raising a helpful error if not installed."""
    try:
        import claude_agent_sdk
    except ImportError as error:
        raise ImportError(
            "claude-agent-sdk is not installed. Install it with: "
            "pip install 'weakincentives[claude-agent-sdk]'"
        ) from error
    else:
        return claude_agent_sdk
