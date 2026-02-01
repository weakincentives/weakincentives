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

"""Type stubs for claude-agent-sdk package.

These stubs provide type information for the Claude Agent SDK, enabling
pyright strict mode type checking for the claude_agent_sdk adapter.
"""

from collections.abc import AsyncIterable, AsyncIterator, Callable, Coroutine
from typing import Any, TypeVar

from .types import ClaudeAgentOptions, HookMatcher, ResultMessage

__all__ = [
    "ClaudeAgentOptions",
    "ClaudeSDKClient",
    "HookMatcher",
    "McpSdkServerConfig",
    "ResultMessage",
    "SdkMcpTool",
    "create_sdk_mcp_server",
    "tool",
]

_T = TypeVar("_T")

class ClaudeSDKClient:
    """Claude Agent SDK client for programmatic agentic interactions.

    This client provides async methods for connecting to Claude, sending
    prompts, and receiving messages. It supports hooks for intercepting
    tool use, prompt submission, and stop events.
    """

    def __init__(self, *, options: ClaudeAgentOptions) -> None:
        """Initialize the SDK client with configuration options.

        Args:
            options: Configuration options for the client.
        """
        ...

    async def connect(self, *, prompt: AsyncIterable[dict[str, Any]]) -> None:
        """Connect to the SDK with an initial prompt stream.

        Args:
            prompt: Async iterable yielding prompt messages.
        """
        ...

    def receive_messages(self) -> AsyncIterator[ResultMessage]:
        """Receive messages from the SDK.

        Yields:
            ResultMessage instances containing model responses.
        """
        ...

    async def disconnect(self) -> None:
        """Disconnect from the SDK and clean up resources."""
        ...

    async def query(
        self,
        *,
        prompt: str,
        session_id: str,
    ) -> None:
        """Send a follow-up query to continue the conversation.

        Args:
            prompt: The prompt text to send.
            session_id: Session identifier for the conversation.
        """
        ...

class McpSdkServerConfig:
    """Configuration for an MCP server in the Claude Agent SDK.

    This is an opaque configuration object returned by create_sdk_mcp_server
    and passed to ClaudeAgentOptions.mcp_servers.
    """

    ...

class SdkMcpTool[T]:
    """Wrapper for an MCP tool registered with the SDK.

    Generic over the return type T of the tool handler.
    """

    name: str
    description: str
    input_schema: dict[str, Any]

def create_sdk_mcp_server(
    *,
    name: str,
    version: str,
    tools: list[SdkMcpTool[Any]],
) -> McpSdkServerConfig:
    """Create an MCP server configuration with the given tools.

    Args:
        name: Name of the MCP server.
        version: Version string for the server.
        tools: List of SdkMcpTool instances to register.

    Returns:
        McpSdkServerConfig ready for use with ClaudeAgentOptions.
    """
    ...

def tool(
    name: str,
    description: str,
    input_schema: dict[str, Any],
) -> Callable[
    [Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]],
    SdkMcpTool[Any],
]:
    """Decorator for creating an MCP tool from an async handler.

    Args:
        name: Tool name for MCP registration.
        description: Tool description for the schema.
        input_schema: JSON schema for tool parameters.

    Returns:
        Decorator that wraps an async handler into an SdkMcpTool.
    """
    ...
