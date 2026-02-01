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

"""Type stubs for claude_agent_sdk.types module."""

from collections.abc import Awaitable, Callable
from typing import Any

__all__ = [
    "ClaudeAgentOptions",
    "HookMatcher",
    "ResultMessage",
]

class ClaudeAgentOptions:
    """Configuration options for the Claude Agent SDK client.

    This dataclass-like object configures how the SDK client behaves,
    including model selection, permissions, hooks, and tool configuration.
    """

    model: str
    cwd: str | None
    permission_mode: str | None
    max_turns: int | None
    max_budget_usd: float | None
    betas: list[str] | None
    output_format: dict[str, Any] | None
    allowed_tools: list[str] | None
    disallowed_tools: list[str] | None
    env: dict[str, str] | None
    setting_sources: list[str] | None
    stderr: Callable[[str], None] | None
    hooks: dict[str, list[HookMatcher]] | None
    mcp_servers: dict[str, Any] | None
    max_thinking_tokens: int | None

    def __init__(
        self,
        *,
        model: str = ...,
        cwd: str | None = ...,
        permission_mode: str | None = ...,
        max_turns: int | None = ...,
        max_budget_usd: float | None = ...,
        betas: list[str] | None = ...,
        output_format: dict[str, Any] | None = ...,
        allowed_tools: list[str] | None = ...,
        disallowed_tools: list[str] | None = ...,
        env: dict[str, str] | None = ...,
        setting_sources: list[str] | None = ...,
        stderr: Callable[[str], None] | None = ...,
        hooks: dict[str, list[HookMatcher]] | None = ...,
        mcp_servers: dict[str, Any] | None = ...,
        max_thinking_tokens: int | None = ...,
    ) -> None:
        """Initialize ClaudeAgentOptions with configuration values.

        All parameters are keyword-only with sensible defaults.
        """
        ...

class HookMatcher:
    """Matcher for SDK hook callbacks.

    Associates a matcher pattern with a list of async hook callbacks.
    The matcher can be None to match all tools, or a string pattern.
    """

    matcher: str | None
    hooks: list[Callable[[Any, str | None, Any], Awaitable[dict[str, Any]]]]

    def __init__(
        self,
        *,
        matcher: str | None = ...,
        hooks: list[Callable[[Any, str | None, Any], Awaitable[dict[str, Any]]]] = ...,
    ) -> None:
        """Initialize HookMatcher.

        Args:
            matcher: Pattern to match tools. None matches all.
            hooks: List of async callback functions.
        """
        ...

class ResultMessage:
    """Result message from the Claude Agent SDK.

    Contains the model's response, usage statistics, and any structured
    output if configured.
    """

    result: str | None
    usage: dict[str, Any] | None
    structured_output: dict[str, Any] | None
    thinking: str | None
    message: dict[str, Any] | None
