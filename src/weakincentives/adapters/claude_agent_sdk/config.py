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

"""Configuration dataclasses for the Claude Agent SDK adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ...dataclasses import FrozenDataclass
from ..config import LLMConfig

if TYPE_CHECKING:
    from ._task_completion import TaskCompletionChecker
    from .isolation import IsolationConfig

__all__ = [
    "ClaudeAgentSDKClientConfig",
    "ClaudeAgentSDKModelConfig",
    "PermissionMode",
]

PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]
"""Permission handling mode for Claude Agent SDK tool execution.

Values:
    default: Use interactive prompts for tool permissions.
    acceptEdits: Auto-accept file edits, prompt for other operations.
    plan: Generate a plan before execution, auto-accept within plan.
    bypassPermissions: Auto-accept all tool calls without prompts.
        Recommended for programmatic/automated usage.

Example:
    >>> config = ClaudeAgentSDKClientConfig(permission_mode="bypassPermissions")
"""


@FrozenDataclass()
class ClaudeAgentSDKClientConfig:
    """Client-level configuration for Claude Agent SDK.

    Controls runtime behavior of the Claude Agent SDK including permissions,
    working directory, budget limits, and hermetic isolation. This configuration
    is passed to the adapter at construction time.

    Example (basic usage):
        >>> from weakincentives.adapters.claude_agent_sdk import (
        ...     ClaudeAgentSDKClientConfig,
        ... )
        >>> config = ClaudeAgentSDKClientConfig(
        ...     max_turns=10,
        ...     max_budget_usd=1.0,
        ... )

    Example (with task completion checking):
        >>> from weakincentives.adapters.claude_agent_sdk import (
        ...     ClaudeAgentSDKClientConfig,
        ...     PlanBasedChecker,
        ... )
        >>> from weakincentives.contrib.tools.planning import Plan
        >>> config = ClaudeAgentSDKClientConfig(
        ...     task_completion_checker=PlanBasedChecker(plan_type=Plan),
        ... )

    Example (with hermetic isolation):
        >>> from weakincentives.adapters.claude_agent_sdk import (
        ...     ClaudeAgentSDKClientConfig,
        ...     IsolationConfig,
        ... )
        >>> config = ClaudeAgentSDKClientConfig(
        ...     isolation=IsolationConfig.inherit_host_auth(),
        ... )

    Attributes:
        permission_mode: Tool permission handling mode. Defaults to
            ``"bypassPermissions"`` for programmatic access. See
            :data:`PermissionMode` for available modes.
        cwd: Working directory for SDK operations. When None and no workspace
            section is present, an empty temporary folder is created. This
            prevents agents from inheriting the host's current working directory.
        max_turns: Maximum number of conversation turns. None means unlimited.
            Use this to prevent runaway loops.
        max_budget_usd: Maximum budget in USD for the session. None means
            unlimited budget. When set, the SDK will stop execution if the
            budget is exceeded.
        suppress_stderr: If True, suppress stderr output from the Claude Code
            CLI process. Useful for hiding bun-related errors or other CLI
            noise in programmatic usage. Defaults to True.
        stop_on_structured_output: If True, stop execution immediately after
            the StructuredOutput tool is called. This ensures the turn ends
            cleanly after structured output is produced. Defaults to True.
        task_completion_checker: Checker for verifying task completion before
            allowing the agent to stop. Both the stop hook and StructuredOutput
            handling use this checker to determine if tasks are complete.
            Supports various implementations:

            - :class:`PlanBasedChecker`: Checks session Plan state
            - :class:`CompositeChecker`: Combines multiple checkers

            When None, no task completion checking is performed and the agent
            can stop freely.
        isolation: Hermetic isolation configuration. When provided, creates
            an ephemeral home directory and prevents access to the host's
            ``~/.claude`` configuration. See :class:`IsolationConfig` for
            details on authentication modes and factory methods.
        betas: Beta features to enable. Passed to the SDK as a list of
            beta feature identifiers (e.g., ``("interleaved_thinking",)``).
            None means no beta features.
    """

    permission_mode: PermissionMode = "bypassPermissions"
    cwd: str | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    suppress_stderr: bool = True
    stop_on_structured_output: bool = True
    task_completion_checker: TaskCompletionChecker | None = None
    isolation: IsolationConfig | None = None
    betas: tuple[str, ...] | None = None


@FrozenDataclass()
class ClaudeAgentSDKModelConfig(LLMConfig):
    """Model-level configuration for Claude Agent SDK.

    Extends :class:`~weakincentives.adapters.config.LLMConfig` with parameters
    specific to Claude models via the Claude Agent SDK. Use this to configure
    model selection, sampling parameters, and extended thinking.

    Example (basic usage with temperature):
        >>> from weakincentives.adapters.claude_agent_sdk import (
        ...     ClaudeAgentSDKModelConfig,
        ... )
        >>> config = ClaudeAgentSDKModelConfig(
        ...     model="claude-sonnet-4-5-20250929",
        ...     temperature=0.7,
        ... )

    Example (with extended thinking):
        >>> config = ClaudeAgentSDKModelConfig(
        ...     model="claude-sonnet-4-5-20250929",
        ...     max_thinking_tokens=10000,
        ... )

    Example (using Opus for complex tasks):
        >>> config = ClaudeAgentSDKModelConfig(
        ...     model="claude-opus-4-5-20251101",
        ...     max_tokens=8192,
        ... )

    Inherited Attributes (from LLMConfig):
        temperature: Sampling temperature (0.0-2.0). Higher values increase
            randomness. None uses the provider default.
        max_tokens: Maximum tokens to generate. None uses the provider default.
        top_p: Nucleus sampling probability mass (0.0-1.0). None uses the
            provider default.

    Attributes:
        model: Claude model identifier. Defaults to ``"claude-sonnet-4-5-20250929"``.
            Supported models include:

            - ``claude-opus-4-5-20251101``: Most capable model for complex tasks
            - ``claude-sonnet-4-5-20250929``: Balanced performance (default)
            - ``claude-sonnet-4-20250514``: Previous Sonnet version

        max_thinking_tokens: Maximum tokens for extended thinking mode. When
            set, enables extended thinking and allocates up to this many
            tokens for the model's internal reasoning. Requires a minimum
            of approximately 1024 tokens. None disables extended thinking.

    Raises:
        ValueError: If any unsupported parameters are provided. The Claude
            Agent SDK does not support ``seed``, ``stop``, ``presence_penalty``,
            or ``frequency_penalty``. Remove these from your configuration.
    """

    model: str = "claude-sonnet-4-5-20250929"
    max_thinking_tokens: int | None = None

    def __post_init__(self) -> None:
        unsupported: dict[str, object | None] = {
            "seed": self.seed,
            "stop": self.stop,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }

        set_unsupported = [
            key for key, value in unsupported.items() if value is not None
        ]
        if set_unsupported:
            raise ValueError(
                "Unsupported Claude Agent SDK parameters: "
                + ", ".join(sorted(set_unsupported))
                + ". Remove them from ClaudeAgentSDKModelConfig."
            )
