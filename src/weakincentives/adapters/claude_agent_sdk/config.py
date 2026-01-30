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
from ._transcript_collector import TranscriptCollectorConfig

if TYPE_CHECKING:
    from ._task_completion import TaskCompletionChecker
    from .isolation import IsolationConfig

__all__ = [
    "ClaudeAgentSDKClientConfig",
    "ClaudeAgentSDKModelConfig",
    "PermissionMode",
]

PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]
"""Permission handling mode for Claude Agent SDK tool execution."""


@FrozenDataclass()
class ClaudeAgentSDKClientConfig:
    """Client-level configuration for Claude Agent SDK.

    Attributes:
        permission_mode: Tool permission handling mode. Defaults to
            ``"bypassPermissions"`` for programmatic access.
        cwd: Working directory for SDK operations. When None and no workspace
            section is present, an empty temporary folder is created. This
            prevents agents from inheriting the host's current working directory.
        max_turns: Maximum number of conversation turns. None means unlimited.
        max_budget_usd: Maximum budget in USD for the session. None means
            unlimited budget. When set, the SDK will stop execution if the
            budget is exceeded.
        suppress_stderr: If True, suppress stderr output from the Claude Code
            CLI process. Useful for hiding bun-related errors or other CLI
            noise in programmatic usage.
        stop_on_structured_output: If True, stop execution immediately after
            the StructuredOutput tool is called. This ensures the turn ends
            cleanly after structured output is produced.
        task_completion_checker: Checker for verifying task completion before
            allowing the agent to stop. Both the stop hook and StructuredOutput
            handling use this checker to determine if tasks are complete.
            Supports various implementations:
            - ``PlanBasedChecker``: Checks session Plan state
            - ``CompositeChecker``: Combines multiple checkers
            When None, no task completion checking is performed and the agent
            can stop freely.
        isolation: Hermetic isolation configuration. When provided, creates
            an ephemeral home directory and prevents access to the host's
            ~/.claude configuration. See :class:`IsolationConfig` for details.
        betas: Beta features to enable. Passed to the SDK as a list of
            beta feature identifiers. None means no beta features.
        transcript_collection: Configuration for transcript collection. By
            default, collects and logs transcript entries from the main session
            and all sub-agent sessions. Set to None to disable transcript
            collection. See :class:`TranscriptCollectorConfig` for details.
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
    transcript_collection: TranscriptCollectorConfig | None = (
        TranscriptCollectorConfig()
    )


@FrozenDataclass()
class ClaudeAgentSDKModelConfig(LLMConfig):
    """Model-level configuration for Claude Agent SDK.

    Extends LLMConfig with parameters specific to Claude models via the
    Claude Agent SDK.

    Attributes:
        model: Claude model identifier. Defaults to the latest Sonnet model.
        max_thinking_tokens: Maximum tokens for extended thinking mode. When
            set, enables extended thinking and allocates up to this many
            tokens for the model's internal reasoning. Requires a minimum
            of approximately 1024 tokens. None disables extended thinking.

    Notes:
        The Claude Agent SDK does not support ``seed``, ``stop``,
        ``presence_penalty``, or ``frequency_penalty``. If any of these fields
        are provided, ``ClaudeAgentSDKModelConfig`` raises ``ValueError``.
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
