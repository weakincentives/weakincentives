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

"""Hook registry for SDK query execution.

This module provides a centralized registry for creating and managing
SDK hooks, extracting hook composition logic from _run_sdk_query() for
better testability and separation of concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...dataclasses import FrozenDataclass
from ._hooks import (
    AsyncHookCallback,
    create_notification_hook,
    create_post_tool_use_hook,
    create_pre_compact_hook,
    create_pre_tool_use_hook,
    create_stop_hook,
    create_subagent_start_hook,
    create_subagent_stop_hook,
    create_task_completion_stop_hook,
    create_user_prompt_submit_hook,
)

if TYPE_CHECKING:
    from ._hooks import HookContext
    from ._task_completion import TaskCompletionChecker

__all__ = [
    "HookRegistry",
    "HookSet",
]


@FrozenDataclass()
class HookSet:
    """Complete set of hooks for an SDK query.

    Contains all hook callbacks needed for SDK execution. Each hook
    is an AsyncHookCallback matching the SDK's expected signature.

    Attributes:
        pre_tool_use: Hook called before tool execution.
        post_tool_use: Hook called after tool execution.
        stop: Hook called when execution stops.
        user_prompt_submit: Hook called on user prompt submission.
        subagent_start: Hook called when a subagent starts.
        subagent_stop: Hook called when a subagent stops.
        pre_compact: Hook called before context compaction.
        notification: Hook called on notifications.
    """

    pre_tool_use: AsyncHookCallback
    post_tool_use: AsyncHookCallback
    stop: AsyncHookCallback
    user_prompt_submit: AsyncHookCallback
    subagent_start: AsyncHookCallback
    subagent_stop: AsyncHookCallback
    pre_compact: AsyncHookCallback
    notification: AsyncHookCallback


class HookRegistry:
    """Factory for creating coordinated hook sets.

    Encapsulates all hook creation logic, making it easier to test
    individual hooks and customize hook combinations.

    Example:
        >>> registry = HookRegistry(hook_context)
        >>> hook_set = registry.create_hook_set(
        ...     stop_on_structured_output=True,
        ...     task_completion_checker=checker,
        ... )
        >>> hooks_dict = registry.to_sdk_hooks(hook_set)
    """

    def __init__(self, context: HookContext) -> None:
        """Initialize the hook registry.

        Args:
            context: Hook context shared by all hooks.
        """
        self._context = context

    def create_hook_set(
        self,
        *,
        stop_on_structured_output: bool = True,
        task_completion_checker: TaskCompletionChecker | None = None,
    ) -> HookSet:
        """Create a complete, coordinated hook set.

        Creates all hooks needed for SDK execution with the configured
        options. The stop hook is chosen based on whether a task
        completion checker is provided.

        Args:
            stop_on_structured_output: If True, stop after StructuredOutput
                when no checker is configured.
            task_completion_checker: Optional checker for verifying task
                completion at stop points and StructuredOutput.

        Returns:
            HookSet containing all configured hooks.
        """
        pre_hook = create_pre_tool_use_hook(self._context)
        post_hook = create_post_tool_use_hook(
            self._context,
            stop_on_structured_output=stop_on_structured_output,
            task_completion_checker=task_completion_checker,
        )

        # Use task completion stop hook if checker is configured
        if task_completion_checker is not None:  # pragma: no cover
            stop_hook = create_task_completion_stop_hook(
                self._context,
                checker=task_completion_checker,
            )
        else:
            stop_hook = create_stop_hook(self._context)

        return HookSet(
            pre_tool_use=pre_hook,
            post_tool_use=post_hook,
            stop=stop_hook,
            user_prompt_submit=create_user_prompt_submit_hook(self._context),
            subagent_start=create_subagent_start_hook(self._context),
            subagent_stop=create_subagent_stop_hook(self._context),
            pre_compact=create_pre_compact_hook(self._context),
            notification=create_notification_hook(self._context),
        )

    @staticmethod
    def to_sdk_hooks(hook_set: HookSet) -> dict[str, list[Any]]:
        """Convert HookSet to SDK-compatible hooks dict.

        Wraps each hook in the HookMatcher format expected by the SDK.
        Imported lazily to avoid SDK dependency at module load time.

        Args:
            hook_set: The hook set to convert.

        Returns:
            Dictionary mapping hook type names to HookMatcher lists.
        """
        from claude_agent_sdk.types import HookMatcher

        return {
            "PreToolUse": [HookMatcher(matcher=None, hooks=[hook_set.pre_tool_use])],
            "PostToolUse": [HookMatcher(matcher=None, hooks=[hook_set.post_tool_use])],
            "Stop": [HookMatcher(matcher=None, hooks=[hook_set.stop])],
            "UserPromptSubmit": [
                HookMatcher(matcher=None, hooks=[hook_set.user_prompt_submit])
            ],
            "SubagentStart": [
                HookMatcher(matcher=None, hooks=[hook_set.subagent_start])
            ],
            "SubagentStop": [HookMatcher(matcher=None, hooks=[hook_set.subagent_stop])],
            "PreCompact": [HookMatcher(matcher=None, hooks=[hook_set.pre_compact])],
            "Notification": [HookMatcher(matcher=None, hooks=[hook_set.notification])],
        }

    @staticmethod
    def get_hook_type_names() -> list[str]:
        """Get the list of hook type names for logging.

        Returns:
            List of hook type name strings.
        """
        return [
            "PreToolUse",
            "PostToolUse",
            "Stop",
            "UserPromptSubmit",
            "SubagentStart",
            "SubagentStop",
            "PreCompact",
            "Notification",
        ]
