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

"""Re-exports from ``weakincentives.prompt.task_completion``.

The canonical location for task completion types is the ``prompt`` package.
This module exists only to keep internal adapter imports working.
"""

from __future__ import annotations

from ...prompt.task_completion import (
    CompositeChecker,
    FileOutputChecker,
    TaskCompletionChecker,
    TaskCompletionContext,
    TaskCompletionResult,
)

__all__ = [
    "CompositeChecker",
    "FileOutputChecker",
    "TaskCompletionChecker",
    "TaskCompletionContext",
    "TaskCompletionResult",
]
