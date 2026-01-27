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

"""Transcript collection for Claude Agent SDK execution.

This module provides real-time collection and logging of Claude Agent SDK
transcripts from the main session and all sub-agent sessions. Transcript
entries are parsed and emitted as DEBUG-level structured log messages.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...runtime.logging import StructuredLogger, get_logger

__all__ = [
    "TranscriptCollector",
    "TranscriptCollectorConfig",
]

logger: StructuredLogger = get_logger(
    __name__, context={"component": "transcript_collector"}
)


@dataclass(slots=True, frozen=True)
class TranscriptCollectorConfig:
    """Configuration for TranscriptCollector."""

    poll_interval: float = 0.25
    """Seconds between file polls."""

    subagent_discovery_interval: float = 1.0
    """Seconds between subagent scans."""

    max_read_bytes: int = 65536
    """Maximum bytes per read cycle."""

    emit_raw_json: bool = True
    """Include raw JSON in log context."""

    parse_entries: bool = True
    """Parse and type transcript entries."""


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


@dataclass(slots=True)
class TranscriptCollector:
    """Collects transcripts from Claude Agent SDK execution.

    Monitors the main transcript and sub-agent transcripts, emitting
    each entry as a DEBUG-level structured log message.

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

    _tailers: dict[str, _TailerState] = field(default_factory=dict, init=False)
    """Active tailers by source."""

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

        Returns:
            Hook configuration dict for ClaudeAgentOptions.hooks.
        """
        # Import the SDK's HookMatcher type
        from claude_agent_sdk.types import HookMatcher

        # Create a wrapper function that captures self
        async def hook_fn(
            input_data: Any,  # noqa: ANN401
            tool_use_id: str | None,
            context: Any,  # noqa: ANN401
        ) -> dict[str, Any]:
            """Wrapper function for the hook callback."""
            return await self.hook_callback(input_data, tool_use_id, context)

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

        logger.debug(
            "transcript.collector.start",
            event="transcript.collector.start",
            context={
                "prompt_name": self.prompt_name,
                "poll_interval": self.config.poll_interval,
                "subagent_discovery_interval": self.config.subagent_discovery_interval,
            },
        )

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

            # Final poll to capture remaining content
            await self._poll_once()

            logger.debug(
                "transcript.collector.stop",
                event="transcript.collector.stop",
                context={
                    "prompt_name": self.prompt_name,
                    "main_entry_count": self.main_entry_count,
                    "subagent_count": self.subagent_count,
                    "total_entries": self.total_entries,
                },
            )

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
            "transcript.collector.path_discovered",
            event="transcript.collector.path_discovered",
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

            if source.startswith("subagent:"):
                logger.debug(
                    "transcript.collector.subagent_discovered",
                    event="transcript.collector.subagent_discovered",
                    context={
                        "prompt_name": self.prompt_name,
                        "source": source,
                        "path": str(path),
                    },
                )
        except OSError as e:
            logger.warning(
                "transcript.collector.error",
                event="transcript.collector.error",
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
            await asyncio.sleep(self.config.poll_interval)

    async def _discovery_loop(self) -> None:
        """Background loop for subagent discovery."""
        while self._running:
            await self._discover_subagents()
            await asyncio.sleep(self.config.subagent_discovery_interval)

    async def _poll_once(self) -> None:
        """Poll all active tailers for new content."""
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
                "transcript.collector.error",
                event="transcript.collector.error",
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
        for line in lines:
            line = line.rstrip("\n\r")
            if not line:
                continue

            tailer.entry_count += 1
            await self._emit_entry(tailer, line)

    async def _emit_entry(self, tailer: _TailerState, line: str) -> None:
        """Emit a single transcript entry as a log message.

        Args:
            tailer: Tailer state.
            line: Raw JSONL line.
        """
        context: dict[str, Any] = {
            "prompt_name": self.prompt_name,
            "transcript_source": tailer.source,
            "sequence_number": tailer.entry_count,
        }

        # Include raw JSON if configured
        if self.config.emit_raw_json:
            context["raw_json"] = line

        # Parse entry if configured
        if self.config.parse_entries:
            try:
                entry = json.loads(line)
                entry_type = entry.get("type", "unknown")
                context["entry_type"] = entry_type
                context["parsed"] = entry
            except json.JSONDecodeError:
                context["entry_type"] = "invalid"
                context["parse_error"] = "Invalid JSON"
        else:
            context["entry_type"] = "unparsed"

        logger.debug(
            "transcript.collector.entry",
            event="transcript.collector.entry",
            context=context,
        )
