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

"""OpenCode ACP client implementation.

This module provides the ACP client implementation that interfaces with
OpenCode via the Agent Client Protocol. The client handles:

- Spawning the OpenCode process with ACP server
- Protocol handshake (initialize, session/new or session/load)
- Session updates via session_update notifications
- Permission request handling
- File read/write operations (when workspace is configured)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...runtime.logging import StructuredLogger, get_logger
from .config import OpenCodeACPClientConfig

if TYPE_CHECKING:
    from .workspace import OpenCodeWorkspaceSection

__all__ = [
    "OpenCodeACPClient",
    "SessionAccumulator",
]

logger: StructuredLogger = get_logger(__name__, context={"component": "acp_client"})


def _import_acp() -> Any:  # pragma: no cover
    """Import the ACP SDK, raising a helpful error if not installed."""
    try:
        import acp

        return acp
    except ImportError as error:
        raise ImportError(
            "agent-client-protocol is not installed. Install it with: "
            "pip install 'weakincentives[acp]'"
        ) from error


@dataclass(slots=True)
class ToolCallUpdate:
    """Represents a tool call update from the ACP session.

    Attributes:
        tool_use_id: The unique ID of the tool call.
        tool_name: The name of the tool.
        params: The tool parameters.
        status: Current status (pending, running, completed, failed).
        result: The result text, if completed.
        mcp_server_name: The MCP server name, if available.
    """

    tool_use_id: str
    tool_name: str
    params: dict[str, Any] | None = None
    status: str = "pending"
    result: str | None = None
    mcp_server_name: str | None = None


@dataclass(slots=True)
class SessionAccumulator:
    """Accumulates session updates from ACP notifications.

    This class tracks the state of an ACP session, collecting agent messages,
    tool calls, and thought chunks as they stream in.

    Attributes:
        agent_messages: List of agent message text chunks.
        tool_calls: Mapping of tool_use_id to ToolCallUpdate.
        thoughts: List of thought chunks (if emit_thought_chunks is enabled).
        final_text: Concatenated agent message text.
    """

    agent_messages: list[str] = field(default_factory=list)
    tool_calls: dict[str, ToolCallUpdate] = field(default_factory=dict)
    thoughts: list[str] = field(default_factory=list)
    _emit_thoughts: bool = False

    def handle_agent_message_chunk(self, text: str) -> None:
        """Handle an agent message chunk.

        Args:
            text: The message text chunk.
        """
        self.agent_messages.append(text)

    def handle_thought_chunk(self, text: str) -> None:
        """Handle a thought chunk.

        Args:
            text: The thought text chunk.
        """
        if self._emit_thoughts:
            self.thoughts.append(text)

    def handle_tool_call(
        self,
        tool_use_id: str,
        tool_name: str,
        params: dict[str, Any] | None = None,
        mcp_server_name: str | None = None,
    ) -> None:
        """Handle a tool call notification.

        Args:
            tool_use_id: The unique ID of the tool call.
            tool_name: The name of the tool.
            params: The tool parameters.
            mcp_server_name: The MCP server name if available.
        """
        if tool_use_id not in self.tool_calls:
            self.tool_calls[tool_use_id] = ToolCallUpdate(
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                params=params,
                status="running",
                mcp_server_name=mcp_server_name,
            )
        else:
            # Update existing tool call
            call = self.tool_calls[tool_use_id]
            call.status = "running"
            if params:
                call.params = params
            if mcp_server_name:
                call.mcp_server_name = mcp_server_name

    def handle_tool_call_update(
        self,
        tool_use_id: str,
        status: str,
        result: str | None = None,
    ) -> None:
        """Handle a tool call status update.

        Args:
            tool_use_id: The unique ID of the tool call.
            status: The new status (completed, failed).
            result: The result text if available.
        """
        if tool_use_id in self.tool_calls:
            call = self.tool_calls[tool_use_id]
            call.status = status
            call.result = result
        else:
            # Tool call update without prior notification - create entry
            self.tool_calls[tool_use_id] = ToolCallUpdate(
                tool_use_id=tool_use_id,
                tool_name="unknown",
                status=status,
                result=result,
            )

    @property
    def final_text(self) -> str:
        """Return concatenated agent message text."""
        return "".join(self.agent_messages)

    @property
    def final_text_with_thoughts(self) -> str:
        """Return concatenated text including thoughts."""
        parts = []
        if self.thoughts:
            parts.append("".join(self.thoughts))
        if self.agent_messages:
            parts.append("".join(self.agent_messages))
        return "\n\n".join(parts) if parts else ""

    def completed_tool_calls(self) -> list[ToolCallUpdate]:
        """Return tool calls with terminal status (completed or failed)."""
        return [
            call
            for call in self.tool_calls.values()
            if call.status in {"completed", "failed"}
        ]


class OpenCodeACPClient:
    """ACP client for OpenCode integration.

    This client handles the ACP protocol communication with OpenCode,
    including spawning the process, protocol handshake, and session management.

    Example:
        >>> client = OpenCodeACPClient(config)
        >>> await client.connect()
        >>> session_id = await client.new_session(cwd="/path/to/workspace")
        >>> await client.prompt(session_id, "List files in the repo")
        >>> # Stream updates via client.accumulator
        >>> await client.disconnect()
    """

    def __init__(
        self,
        config: OpenCodeACPClientConfig,
        *,
        workspace: OpenCodeWorkspaceSection | None = None,
    ) -> None:
        """Initialize the ACP client.

        Args:
            config: Client configuration.
            workspace: Optional workspace section for file operations.
        """
        self._config = config
        self._workspace = workspace
        self._process: Any = None
        self._connection: Any = None
        self._session_id: str | None = None
        self._accumulator = SessionAccumulator()

        logger.debug(
            "opencode_acp.client.init",
            event="client.init",
            context={
                "opencode_bin": config.opencode_bin,
                "opencode_args": config.opencode_args,
                "cwd": config.cwd,
                "permission_mode": config.permission_mode,
                "allow_file_reads": config.allow_file_reads,
                "allow_file_writes": config.allow_file_writes,
                "has_workspace": workspace is not None,
            },
        )

    @property
    def accumulator(self) -> SessionAccumulator:
        """Return the session accumulator."""
        return self._accumulator

    @property
    def session_id(self) -> str | None:
        """Return the current session ID."""
        return self._session_id

    async def connect(self) -> None:
        """Spawn OpenCode and establish ACP connection.

        Raises:
            RuntimeError: If connection fails.
            TimeoutError: If startup timeout is exceeded.
        """
        acp = _import_acp()

        # Resolve cwd
        cwd = self._config.cwd
        if cwd is None:
            cwd = str(Path.cwd().resolve())

        # Build command
        cmd = [self._config.opencode_bin, *self._config.opencode_args]

        # Build environment
        env = dict(self._config.env) if self._config.env else None

        logger.debug(
            "opencode_acp.client.spawning",
            event="client.spawning",
            context={"cmd": cmd, "cwd": cwd},
        )

        try:
            # Spawn the agent process using ACP SDK
            self._process, self._connection = await acp.spawn_agent_process(
                cmd,
                cwd=cwd,
                env=env,
                client=self,  # Self implements acp.interfaces.Client
            )

            logger.debug(
                "opencode_acp.client.spawned",
                event="client.spawned",
                context={"pid": self._process.pid if self._process else None},
            )

        except Exception as error:
            logger.warning(
                "opencode_acp.client.spawn_failed",
                event="client.spawn_failed",
                context={"error": str(error)},
            )
            raise RuntimeError(f"Failed to spawn OpenCode: {error}") from error

    async def initialize(self) -> None:
        """Send initialize request to establish protocol version.

        Raises:
            RuntimeError: If initialization fails.
        """
        if self._connection is None:
            raise RuntimeError("Not connected")

        acp = _import_acp()

        # Determine advertised capabilities based on config
        # File capabilities are only advertised if workspace is configured
        has_workspace = self._workspace is not None
        allow_file_reads = self._config.allow_file_reads and has_workspace
        allow_file_writes = self._config.allow_file_writes and has_workspace

        capabilities = {
            "fs": {
                "readTextFile": allow_file_reads,
                "writeTextFile": allow_file_writes,
            },
            "terminal": self._config.allow_terminal,
        }

        logger.debug(
            "opencode_acp.client.initializing",
            event="client.initializing",
            context={
                "protocol_version": acp.PROTOCOL_VERSION,
                "capabilities": capabilities,
            },
        )

        try:
            await self._connection.initialize(
                protocol_version=acp.PROTOCOL_VERSION,
                client_capabilities=capabilities,
            )

            logger.debug(
                "opencode_acp.client.initialized",
                event="client.initialized",
            )

        except Exception as error:
            logger.warning(
                "opencode_acp.client.initialize_failed",
                event="client.initialize_failed",
                context={"error": str(error)},
            )
            raise RuntimeError(f"Failed to initialize: {error}") from error

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> str:
        """Create a new OpenCode session.

        Args:
            cwd: Working directory for the session.
            mcp_servers: MCP server configurations.

        Returns:
            The session ID.

        Raises:
            RuntimeError: If session creation fails.
        """
        if self._connection is None:
            raise RuntimeError("Not connected")

        logger.debug(
            "opencode_acp.client.new_session",
            event="client.new_session",
            context={
                "cwd": cwd,
                "mcp_server_count": len(mcp_servers) if mcp_servers else 0,
            },
        )

        try:
            result = await self._connection.new_session(
                cwd=cwd,
                mcp_servers=mcp_servers or [],
            )
            self._session_id = result.session_id

            logger.debug(
                "opencode_acp.client.session_created",
                event="client.session_created",
                context={"session_id": self._session_id},
            )

            return self._session_id

        except Exception as error:
            logger.warning(
                "opencode_acp.client.new_session_failed",
                event="client.new_session_failed",
                context={"error": str(error)},
            )
            raise RuntimeError(f"Failed to create session: {error}") from error

    async def load_session(
        self,
        session_id: str,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Attempt to load an existing OpenCode session.

        Args:
            session_id: The session ID to load.
            cwd: Working directory for the session.
            mcp_servers: MCP server configurations.

        Returns:
            True if session was loaded successfully, False if failed.
        """
        if self._connection is None:
            raise RuntimeError("Not connected")

        logger.debug(
            "opencode_acp.client.load_session",
            event="client.load_session",
            context={"session_id": session_id, "cwd": cwd},
        )

        try:
            await self._connection.load_session(
                session_id=session_id,
                cwd=cwd,
                mcp_servers=mcp_servers or [],
            )
            self._session_id = session_id

            logger.debug(
                "opencode_acp.client.session_loaded",
                event="client.session_loaded",
                context={"session_id": session_id},
            )

            return True

        except Exception as error:
            logger.debug(
                "opencode_acp.client.load_session_failed",
                event="client.load_session_failed",
                context={"session_id": session_id, "error": str(error)},
            )
            return False

    async def set_mode(self, mode_id: str) -> bool:
        """Attempt to set the session mode.

        Args:
            mode_id: The mode ID to set.

        Returns:
            True if successful, False if method not supported.
        """
        if self._connection is None or self._session_id is None:
            return False

        try:
            await self._connection.set_mode(
                session_id=self._session_id,
                mode_id=mode_id,
            )
            logger.debug(
                "opencode_acp.client.mode_set",
                event="client.mode_set",
                context={"mode_id": mode_id},
            )
            return True
        except Exception:
            # Method may not be implemented - ignore errors
            logger.debug(
                "opencode_acp.client.set_mode_unsupported",
                event="client.set_mode_unsupported",
                context={"mode_id": mode_id},
            )
            return False

    async def set_model(self, model_id: str) -> bool:
        """Attempt to set the session model.

        Args:
            model_id: The model ID to set.

        Returns:
            True if successful, False if method not supported.
        """
        if self._connection is None or self._session_id is None:
            return False

        try:
            await self._connection.set_model(
                session_id=self._session_id,
                model_id=model_id,
            )
            logger.debug(
                "opencode_acp.client.model_set",
                event="client.model_set",
                context={"model_id": model_id},
            )
            return True
        except Exception:
            # Method may not be implemented - ignore errors
            logger.debug(
                "opencode_acp.client.set_model_unsupported",
                event="client.set_model_unsupported",
                context={"model_id": model_id},
            )
            return False

    async def prompt(self, text: str) -> None:
        """Send a prompt to the session.

        Args:
            text: The prompt text.

        Raises:
            RuntimeError: If no session is active.
        """
        if self._connection is None or self._session_id is None:
            raise RuntimeError("No active session")

        logger.debug(
            "opencode_acp.client.prompt",
            event="client.prompt",
            context={
                "session_id": self._session_id,
                "text_length": len(text),
            },
        )

        # Clear accumulator for new prompt
        self._accumulator = SessionAccumulator()

        try:
            await self._connection.prompt(
                session_id=self._session_id,
                prompt=[{"type": "text", "text": text}],
            )
        except Exception as error:
            logger.warning(
                "opencode_acp.client.prompt_failed",
                event="client.prompt_failed",
                context={"error": str(error)},
            )
            raise

    async def cancel(self) -> None:
        """Cancel the current session operation.

        This is a notification (no response expected).
        """
        if self._connection is None or self._session_id is None:
            return

        logger.debug(
            "opencode_acp.client.cancel",
            event="client.cancel",
            context={"session_id": self._session_id},
        )

        try:
            await self._connection.cancel(session_id=self._session_id)
        except Exception as error:
            logger.debug(
                "opencode_acp.client.cancel_error",
                event="client.cancel_error",
                context={"error": str(error)},
            )

    async def disconnect(self) -> None:
        """Disconnect from OpenCode and terminate the process."""
        logger.debug(
            "opencode_acp.client.disconnecting",
            event="client.disconnecting",
        )

        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception:  # nosec B110 - cleanup should proceed regardless
                pass
            self._connection = None

        if self._process is not None:
            try:
                self._process.terminate()
                await self._process.wait()
            except Exception:  # nosec B110 - cleanup should proceed regardless
                pass
            self._process = None

        self._session_id = None

        logger.debug(
            "opencode_acp.client.disconnected",
            event="client.disconnected",
        )

    # --- ACP Client Interface Methods ---

    async def session_update(
        self,
        session_id: str,
        update: dict[str, Any],
    ) -> None:
        """Handle session update notification from OpenCode.

        This method is called by the ACP connection when OpenCode sends
        a session/update notification.

        Args:
            session_id: The session ID.
            update: The update payload.
        """
        update_type = update.get("type")

        if update_type == "agent_message_chunk":
            text = update.get("text", "")
            self._accumulator.handle_agent_message_chunk(text)

        elif update_type == "thought":
            text = update.get("text", "")
            self._accumulator.handle_thought_chunk(text)

        elif update_type == "tool_call":
            self._accumulator.handle_tool_call(
                tool_use_id=update.get("tool_use_id", ""),
                tool_name=update.get("tool_name", "unknown"),
                params=update.get("params"),
                mcp_server_name=update.get("mcp_server_name"),
            )

        elif update_type == "tool_call_update":
            self._accumulator.handle_tool_call_update(
                tool_use_id=update.get("tool_use_id", ""),
                status=update.get("status", "unknown"),
                result=update.get("result"),
            )

        logger.debug(
            "opencode_acp.client.session_update",
            event="client.session_update",
            context={
                "session_id": session_id,
                "update_type": update_type,
            },
        )

    async def request_permission(
        self,
        session_id: str,
        permission: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle permission request from OpenCode.

        Responds based on the configured permission_mode.

        Args:
            session_id: The session ID.
            permission: The permission request payload.

        Returns:
            Permission response with approved/denied status.
        """
        mode = self._config.permission_mode

        if mode == "auto":
            approved = True
            reason = None
        elif mode == "deny":
            approved = False
            reason = "Permission denied by policy."
        else:  # mode == "prompt"
            # Non-interactive - deny with explanation
            approved = False
            reason = (
                "Interactive permission prompting is not supported. "
                "Permission denied by policy."
            )

        logger.debug(
            "opencode_acp.client.request_permission",
            event="client.request_permission",
            context={
                "session_id": session_id,
                "permission_type": permission.get("type"),
                "mode": mode,
                "approved": approved,
            },
        )

        return {
            "approved": approved,
            "reason": reason,
        }

    async def read_text_file(
        self,
        session_id: str,
        path: str,
    ) -> dict[str, Any]:
        """Handle file read request from OpenCode.

        Only processes if allow_file_reads is True and workspace is configured.

        Args:
            session_id: The session ID.
            path: The file path to read.

        Returns:
            File content or error response.
        """
        if not self._config.allow_file_reads or self._workspace is None:
            return {
                "error": "File reading is not enabled.",
            }

        try:
            # Validate path is within workspace
            workspace_root = self._workspace.temp_dir
            file_path = Path(path)

            # Make relative to workspace
            if file_path.is_absolute():
                try:
                    file_path = file_path.relative_to(workspace_root)
                except ValueError:
                    return {
                        "error": f"Path is outside workspace: {path}",
                    }

            full_path = workspace_root / file_path
            resolved = full_path.resolve()

            # Security check - ensure path is within workspace
            try:
                resolved.relative_to(workspace_root.resolve())
            except ValueError:
                return {
                    "error": f"Path is outside workspace: {path}",
                }

            if not resolved.exists():
                return {
                    "error": f"File not found: {path}",
                }

            if not resolved.is_file():
                return {
                    "error": f"Path is not a file: {path}",
                }

            content = resolved.read_text(encoding="utf-8")

            logger.debug(
                "opencode_acp.client.read_text_file",
                event="client.read_text_file",
                context={
                    "path": path,
                    "size": len(content),
                },
            )

            return {
                "content": content,
            }

        except Exception as error:
            logger.warning(
                "opencode_acp.client.read_text_file_error",
                event="client.read_text_file_error",
                context={"path": path, "error": str(error)},
            )
            return {
                "error": str(error),
            }

    async def write_text_file(
        self,
        session_id: str,
        path: str,
        content: str,
    ) -> dict[str, Any]:
        """Handle file write request from OpenCode.

        Only processes if allow_file_writes is True and workspace is configured.

        Args:
            session_id: The session ID.
            path: The file path to write.
            content: The content to write.

        Returns:
            Success or error response.
        """
        if not self._config.allow_file_writes or self._workspace is None:
            return {
                "error": "File writing is not enabled.",
            }

        try:
            # Validate path is within workspace
            workspace_root = self._workspace.temp_dir
            file_path = Path(path)

            # Make relative to workspace
            if file_path.is_absolute():
                try:
                    file_path = file_path.relative_to(workspace_root)
                except ValueError:
                    return {
                        "error": f"Path is outside workspace: {path}",
                    }

            full_path = workspace_root / file_path
            resolved = full_path.resolve()

            # Security check - ensure path is within workspace
            try:
                resolved.relative_to(workspace_root.resolve())
            except ValueError:
                return {
                    "error": f"Path is outside workspace: {path}",
                }

            # Create parent directories
            resolved.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            resolved.write_text(content, encoding="utf-8")

            logger.debug(
                "opencode_acp.client.write_text_file",
                event="client.write_text_file",
                context={
                    "path": path,
                    "size": len(content),
                },
            )

            return {
                "success": True,
            }

        except Exception as error:
            logger.warning(
                "opencode_acp.client.write_text_file_error",
                event="client.write_text_file_error",
                context={"path": path, "error": str(error)},
            )
            return {
                "error": str(error),
            }

    async def create_terminal(
        self,
        session_id: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle terminal creation request from OpenCode.

        Currently not implemented.

        Args:
            session_id: The session ID.
            options: Terminal options.

        Returns:
            Error response indicating not supported.
        """
        if not self._config.allow_terminal:
            return {
                "error": "Terminal creation is not supported.",
            }

        # Terminal creation is not implemented
        return {
            "error": "Terminal creation is not implemented.",
        }
