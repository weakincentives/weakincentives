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

"""File monitoring for Claude Agent SDK transcripts.

This module provides real-time file tailing and collection of Claude Agent SDK
transcripts from the main session and all sub-agent sessions.  Entry
transformation logic (parsing JSONL lines, splitting mixed content blocks)
lives in :mod:`._transcript_parser`.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from ...clock import SYSTEM_CLOCK, AsyncSleeper
from ...dataclasses import FrozenDataclass
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.transcript import TranscriptEmitter
from ._transcript_parser import emit_entry

__all__ = [
    "TranscriptCollector",
    "TranscriptCollectorConfig",
]

logger: StructuredLogger = get_logger(
    __name__, context={"component": "transcript_collector"}
)

_SHUTDOWN_DRAIN_ATTEMPTS = 5
"""Number of drain polls after SDK query completes."""

_SHUTDOWN_DRAIN_DELAY = 0.2
"""Seconds between drain polls."""


def _extract_text_from_content(content: object) -> str:
    """Extract text from an SDK message content field.

    Content may be a string, a list of dicts (JSONL transcript format),
    or a list of typed objects with ``.text`` attributes (SDK API format).
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
            continue
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts)


def _extract_assistant_text(messages: list[Any]) -> str:
    """Extract text from the last assistant message in SDK response.

    Handles both SDK API objects (``AssistantMessage`` with ``.content``)
    and JSONL transcript dicts (``{"role": "assistant", "content": ...}``).
    """
    for message in reversed(messages):
        # SDK API format: AssistantMessage with .content attribute
        if type(message).__name__ == "AssistantMessage":
            return _extract_text_from_content(getattr(message, "content", None))
        # JSONL transcript format: dict with role/content
        inner = getattr(message, "message", None)
        if isinstance(inner, dict) and inner.get("role") == "assistant":
            return _extract_text_from_content(inner.get("content"))
    return ""


@FrozenDataclass()
class TranscriptCollectorConfig:
    """Configuration for TranscriptCollector."""

    poll_interval: float = 0.25
    """Seconds between file polls."""

    subagent_discovery_interval: float = 1.0
    """Seconds between subagent scans."""

    max_read_bytes: int = 65536
    """Maximum bytes per read cycle."""

    emit_raw: bool = True
    """Include raw JSONL in ``raw`` field."""


@dataclass(slots=True)
class _TailerState:
    """State for a single file tailer."""

    path: Path
    """Absolute path to transcript file."""

    source: str
    """Source identifier: 'main' or 'subagent:{id}'."""

    position: int = 0
    """Current read position in bytes."""

    inode: int = 0
    """Inode for rotation detection."""

    partial_line: str = ""
    """Incomplete line buffer."""

    entry_count: int = 0
    """Entries emitted from this file."""

    tool_names: dict[str, str] = field(default_factory=dict)
    """Map of tool_use_id → tool_name for correlating results."""

    observed_types: set[str] = field(default_factory=set)
    """Canonical entry types observed from file parsing."""


@dataclass(slots=True)
class TranscriptCollector:
    """Collects transcripts from Claude Agent SDK execution.

    Monitors the main transcript and sub-agent transcripts, emitting
    each entry as a DEBUG-level structured log message via
    :class:`~weakincentives.runtime.transcript.TranscriptEmitter`.

    Example:
        collector = TranscriptCollector(
            prompt_name="code-review",
            config=TranscriptCollectorConfig(),
        )

        async with collector.run():
            async for message in sdk.query(prompt=prompt, options=options):
                # SDK query executes here
                # Transcripts are collected in background
                process_message(message)
    """

    prompt_name: str
    """Name of the prompt being evaluated (for log context)."""

    config: TranscriptCollectorConfig = field(default_factory=TranscriptCollectorConfig)
    """Collector configuration."""

    session_id: str | None = None
    """Optional session UUID."""

    async_sleeper: AsyncSleeper = field(default=SYSTEM_CLOCK)
    """Async sleeper for delay operations (injectable for testing)."""

    _tailers: dict[str, _TailerState] = field(default_factory=dict, init=False)
    """Active tailers by source."""

    _pending_tailers: dict[str, Path] = field(default_factory=dict, init=False)
    """Sources awaiting file creation (path not yet on disk)."""

    _running: bool = field(default=False, init=False)
    """Whether the collector is running."""

    _main_transcript_path: Path | None = field(default=None, init=False)
    """Path to main transcript file."""

    _session_dir: Path | None = field(default=None, init=False)
    """Session directory for subagent discovery."""

    _poll_task: asyncio.Task[None] | None = field(default=None, init=False)
    """Background polling task."""

    _discovery_task: asyncio.Task[None] | None = field(default=None, init=False)
    """Subagent discovery task."""

    _emitter: TranscriptEmitter | None = field(default=None, init=False)
    """Shared emitter instance (created at run() time)."""

    _fallback_user_text: str | None = field(default=None, init=False)
    """Stored user message text for fallback emission."""

    _fallback_messages: list[Any] | None = field(default=None, init=False)
    """Stored assistant messages for fallback emission."""

    @property
    def main_entry_count(self) -> int:
        """Number of entries from main transcript."""
        main_tailer = self._tailers.get("main")
        return main_tailer.entry_count if main_tailer else 0

    @property
    def subagent_count(self) -> int:
        """Number of subagent transcripts discovered."""
        return sum(1 for key in self._tailers if key.startswith("subagent:"))

    @property
    def total_entries(self) -> int:
        """Total entries collected from all transcripts."""
        return sum(tailer.entry_count for tailer in self._tailers.values())

    @property
    def transcript_paths(self) -> list[Path]:
        """List of transcript paths being monitored."""
        return [tailer.path for tailer in self._tailers.values()]

    def _get_emitter(self) -> TranscriptEmitter:
        """Return the emitter, creating lazily if needed."""
        if self._emitter is None:
            self._emitter = TranscriptEmitter(
                prompt_name=self.prompt_name,
                adapter="claude_agent_sdk",
                session_id=self.session_id,
                emit_raw=self.config.emit_raw,
            )
        return self._emitter

    def set_user_message_fallback(self, text: str) -> None:
        """Store user message text for fallback emission at shutdown.

        Called by the adapter before the SDK query starts.  The message
        is emitted during shutdown only if the transcript file did not
        already contain a ``user_message`` entry.
        """
        self._fallback_user_text = text

    def set_assistant_message_fallback(self, messages: list[Any]) -> None:
        """Store assistant messages for fallback emission at shutdown.

        Called by the adapter after the SDK query completes.  The message
        is emitted during shutdown only if the transcript file did not
        already contain an ``assistant_message`` entry.
        """
        self._fallback_messages = messages

    def _emit_fallbacks(self) -> None:
        """Emit fallback user/assistant messages if not observed from file.

        Checks the main tailer's ``observed_types``.  If ``user_message``
        was not observed and ``_fallback_user_text`` is set, emits a
        ``user_message``.  Same for ``assistant_message``.
        """
        main_tailer = self._tailers.get("main")
        observed = main_tailer.observed_types if main_tailer else set()
        emitter = self._get_emitter()

        if "user_message" not in observed and self._fallback_user_text is not None:
            emitter.emit("user_message", detail={"text": self._fallback_user_text})

        if "assistant_message" not in observed and self._fallback_messages is not None:
            text = _extract_assistant_text(self._fallback_messages)
            if text:
                emitter.emit("assistant_message", detail={"text": text})

    async def hook_callback(
        self,
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: Any,  # noqa: ANN401 - SDK hook context type varies by hook
    ) -> dict[str, Any]:
        """SDK hook callback that captures transcript_path.

        Register this callback for supported hook events to discover
        the transcript path as early as possible.

        Args:
            input_data: Hook input containing transcript_path.
            tool_use_id: Tool use ID (for tool hooks).
            context: Hook execution context (not used).

        Returns:
            Empty dict (no modifications to hook behavior).
        """
        _ = tool_use_id
        _ = context
        transcript_path = input_data.get("transcript_path")
        if transcript_path:
            await self._remember_transcript_path(transcript_path)
        return {}

    def hooks_config(self) -> dict[str, list[Any]]:
        """Returns hook configuration for SDK integration.

        Registers the collector's hook callback for all supported events:
        - UserPromptSubmit (earliest discovery)
        - PreToolUse, PostToolUse
        - SubagentStop
        - Stop
        - PreCompact

        Note: The SDK does not support SubagentStart or Notification hooks.

        Returns:
            Hook configuration dict for ClaudeAgentOptions.hooks.
        """
        # Import SDK types for hook configuration
        from claude_agent_sdk.types import (
            HookContext as SdkHookContext,
            HookInput,
            HookMatcher,
            SyncHookJSONOutput,
        )

        # Create a wrapper function that calls the public hook_callback method
        async def hook_fn(
            input_data: HookInput,
            tool_use_id: str | None,
            context: SdkHookContext,
        ) -> SyncHookJSONOutput:
            """SDK hook wrapper that delegates to hook_callback."""
            # HookInput is a TypedDict union - cast to dict[str, Any] for hook_callback
            await self.hook_callback(
                cast(dict[str, Any], input_data), tool_use_id, context
            )
            # Return empty dict (no hook modifications)
            return cast(SyncHookJSONOutput, {})

        matcher = HookMatcher(matcher=None, hooks=[hook_fn])

        return {
            "UserPromptSubmit": [matcher],
            "PreToolUse": [matcher],
            "PostToolUse": [matcher],
            "SubagentStop": [matcher],
            "Stop": [matcher],
            "PreCompact": [matcher],
        }

    @asynccontextmanager
    async def run(self) -> AsyncIterator[None]:
        """Run the transcript collector as an async context manager.

        Yields control to the caller while collecting transcripts
        in the background. Stops collection when the context exits.

        Yields:
            None - control is yielded to allow SDK execution.
        """
        self._running = True
        emitter = self._get_emitter()
        emitter.start()

        try:
            # Start background polling task
            self._poll_task = asyncio.create_task(self._poll_loop())

            # Start subagent discovery task
            self._discovery_task = asyncio.create_task(self._discovery_loop())

            # Yield control to caller
            yield
        finally:
            # Stop running
            self._running = False

            # Cancel tasks
            if self._poll_task is not None:
                self._poll_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._poll_task

            if self._discovery_task is not None:
                self._discovery_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._discovery_task

            # Drain: the SDK may still be flushing transcript content to
            # disk after the query completes, especially for fast evaluations.
            # Poll a few extra times with a short delay to capture trailing
            # entries and to resolve any pending tailers.  Skip the loop
            # entirely when there are no pending tailers (common case).
            for _ in range(_SHUTDOWN_DRAIN_ATTEMPTS):
                if not self._pending_tailers:
                    break
                await self.async_sleeper.async_sleep(_SHUTDOWN_DRAIN_DELAY)
                await self._poll_once()

            # Final poll to pick up any trailing content from active tailers.
            await self._poll_once()

            self._emit_fallbacks()
            emitter.stop()

    async def _remember_transcript_path(self, transcript_path: str) -> None:
        """Remember the main transcript path and derive session directory.

        Args:
            transcript_path: Path to main transcript from SDK hook.
        """
        if self._main_transcript_path is not None:
            return  # Already discovered

        path = Path(transcript_path)
        self._main_transcript_path = path

        # Derive session directory from transcript path
        # Main transcript: /path/to/{sessionId}.jsonl
        # Session dir: /path/to/{sessionId}/
        session_id = path.stem  # Remove .jsonl extension
        self._session_dir = path.parent / session_id

        logger.debug(
            "transcript path discovered",
            event="transcript.path_discovered",
            context={
                "prompt_name": self.prompt_name,
                "transcript_path": str(path),
                "session_dir": str(self._session_dir),
            },
        )

        # Start tailing the main transcript
        await self._start_tailer(path, "main")

    async def _start_tailer(self, path: Path, source: str) -> None:
        """Start tailing a transcript file.

        Args:
            path: Path to transcript file.
            source: Source identifier ('main' or 'subagent:{id}').
        """
        if source in self._tailers:
            return  # Already tailing

        try:
            stat = path.stat()
            tailer = _TailerState(
                path=path,
                source=source,
                position=0,
                inode=stat.st_ino,
            )
            self._tailers[source] = tailer
            self._pending_tailers.pop(source, None)

            if source.startswith("subagent:"):
                logger.debug(
                    "subagent transcript discovered",
                    event="transcript.subagent_discovered",
                    context={
                        "prompt_name": self.prompt_name,
                        "source": source,
                        "path": str(path),
                    },
                )
        except OSError as e:
            already_pending = source in self._pending_tailers
            self._pending_tailers[source] = path
            if not already_pending:
                logger.warning(
                    "transcript file not accessible",
                    event="transcript.error",
                    context={
                        "prompt_name": self.prompt_name,
                        "source": source,
                        "path": str(path),
                        "error": str(e),
                    },
                )

    async def _poll_loop(self) -> None:
        """Background polling loop for transcript content."""
        while self._running:
            await self._poll_once()
            await self.async_sleeper.async_sleep(self.config.poll_interval)

    async def _discovery_loop(self) -> None:
        """Background loop for subagent discovery."""
        while self._running:
            await self._discover_subagents()
            await self.async_sleeper.async_sleep(
                self.config.subagent_discovery_interval
            )

    async def _poll_once(self) -> None:
        """Poll all active tailers for new content."""
        # Retry any pending tailers whose files may now exist on disk.
        for source, path in list(self._pending_tailers.items()):
            await self._start_tailer(path, source)
        for tailer in list(self._tailers.values()):
            await self._read_transcript_content(tailer)

    async def _discover_subagents(self) -> None:
        """Scan session directory for new sub-agent transcripts.

        Session directory is derived from main transcript path:
        - Main transcript: /path/to/{sessionId}.jsonl
        - Session dir: /path/to/{sessionId}/
        - Sub-agents: /path/to/{sessionId}/subagents/agent-{id}.jsonl

        New sub-agent transcripts start tailing immediately.
        """
        if self._session_dir is None:
            return  # No session directory yet

        subagents_dir = self._session_dir / "subagents"
        if not subagents_dir.exists():
            return

        try:
            for path in subagents_dir.glob("agent-*.jsonl"):
                # Extract agent ID from filename
                agent_id = path.stem[6:]  # Remove "agent-" prefix
                source = f"subagent:{agent_id}"
                await self._start_tailer(path, source)
        except OSError:
            # Directory may not exist or be inaccessible
            pass

    async def _read_transcript_content(self, tailer: _TailerState) -> None:
        """Read new content from a transcript file.

        Args:
            tailer: Tailer state for the file.
        """
        try:
            # Check if file still exists
            if not tailer.path.exists():
                return

            stat = tailer.path.stat()

            # Detect file rotation (inode changed) or truncation
            if stat.st_ino != tailer.inode:
                # File was rotated (new inode)
                tailer.position = 0
                tailer.inode = stat.st_ino
                tailer.partial_line = ""
            elif stat.st_size < tailer.position:
                # File was truncated (e.g., compaction)
                tailer.position = 0
                tailer.partial_line = ""

            # No new content
            if stat.st_size <= tailer.position:
                return

            # Read new content
            bytes_to_read = min(
                stat.st_size - tailer.position, self.config.max_read_bytes
            )

            # Use run_in_executor for file I/O to avoid blocking
            content = await asyncio.get_running_loop().run_in_executor(
                None,
                self._read_bytes,
                tailer.path,
                tailer.position,
                bytes_to_read,
            )

            if content:
                tailer.position += len(content)
                await self._emit_entries(tailer, content)

        except OSError as e:
            logger.warning(
                "transcript read error",
                event="transcript.error",
                context={
                    "prompt_name": self.prompt_name,
                    "source": tailer.source,
                    "path": str(tailer.path),
                    "error": str(e),
                },
            )

    @staticmethod
    def _read_bytes(path: Path, offset: int, count: int) -> bytes:
        """Read bytes from file at offset (runs in executor).

        Args:
            path: File path.
            offset: Byte offset to start reading.
            count: Number of bytes to read.

        Returns:
            Bytes read from file.
        """
        with path.open("rb") as f:
            f.seek(offset)
            return f.read(count)

    async def _emit_entries(self, tailer: _TailerState, content: bytes) -> None:
        """Parse and emit transcript entries from content.

        Args:
            tailer: Tailer state with partial line buffer.
            content: Raw bytes read from file.
        """
        # Decode content
        text = content.decode("utf-8", errors="replace")

        # Prepend any partial line from previous read
        if tailer.partial_line:
            text = tailer.partial_line + text
            tailer.partial_line = ""

        # Split into lines
        lines = text.splitlines(keepends=True)

        # Check if last line is complete
        if lines and not lines[-1].endswith("\n"):
            # Buffer the partial line
            tailer.partial_line = lines.pop()

        # Process complete lines (JSONL entries)
        emitter = self._get_emitter()
        for line in lines:
            line = line.rstrip("\n\r")
            if not line:
                continue

            emit_entry(emitter, tailer, line)
