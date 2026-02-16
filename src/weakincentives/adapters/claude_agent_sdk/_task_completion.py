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

"""Re-exports and resolution logic for task completion checking.

The canonical location for task completion types is the ``prompt`` package.
This module re-exports those types and provides ``resolve_checker`` for
resolving the effective checker from prompt and client config.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from ...prompt.task_completion import (
    CompositeChecker,
    FileOutputChecker,
    TaskCompletionChecker,
    TaskCompletionContext,
    TaskCompletionResult,
)
from ...runtime.logging import get_logger

if TYPE_CHECKING:
    from ...prompt.protocols import PromptProtocol
    from .config import ClaudeAgentSDKClientConfig

logger = get_logger(__name__, context={"component": "claude_agent_sdk"})


def resolve_checker(
    *,
    prompt: PromptProtocol[object] | None,
    client_config: ClaudeAgentSDKClientConfig,
) -> TaskCompletionChecker | None:
    """Resolve the effective task completion checker.

    Resolution order:
    1. Prompt-scoped checker (``prompt.task_completion_checker``) takes priority.
    2. Falls back to adapter-scoped checker (``client_config.task_completion_checker``)
       with a deprecation warning, since adapter-scoped configuration is deprecated
       in favor of prompt-scoped declaration.

    Args:
        prompt: The bound prompt, if available.
        client_config: The adapter client configuration.

    Returns:
        The resolved checker, or None if neither source provides one.
    """
    prompt_checker: TaskCompletionChecker | None = None
    if prompt is not None:
        prompt_checker = prompt.task_completion_checker

    if prompt_checker is not None:
        if client_config.task_completion_checker is not None:
            logger.info(
                "claude_agent_sdk.resolve_checker.prompt_overrides_config",
                event="resolve_checker.prompt_overrides_config",
            )
        return prompt_checker

    config_checker = client_config.task_completion_checker
    if config_checker is not None:
        warnings.warn(
            "Configuring task_completion_checker on ClaudeAgentSDKClientConfig is "
            "deprecated. Declare the checker on PromptTemplate instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return config_checker

    return None


__all__ = [
    "CompositeChecker",
    "FileOutputChecker",
    "TaskCompletionChecker",
    "TaskCompletionContext",
    "TaskCompletionResult",
    "resolve_checker",
]
