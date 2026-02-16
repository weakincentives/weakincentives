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
resolving the effective checker from the prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...prompt.task_completion import (
    CompositeChecker,
    FileOutputChecker,
    TaskCompletionChecker,
    TaskCompletionContext,
    TaskCompletionResult,
)

if TYPE_CHECKING:
    from ...prompt.protocols import PromptProtocol


def resolve_checker(
    *,
    prompt: PromptProtocol[object] | None,
) -> TaskCompletionChecker | None:
    """Resolve the effective task completion checker from the prompt.

    Args:
        prompt: The bound prompt, if available.

    Returns:
        The prompt's checker, or None if not configured.
    """
    if prompt is not None:
        return prompt.task_completion_checker
    return None


__all__ = [
    "CompositeChecker",
    "FileOutputChecker",
    "TaskCompletionChecker",
    "TaskCompletionContext",
    "TaskCompletionResult",
    "resolve_checker",
]
