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

"""Slice policies for session snapshot and restore behavior."""

from __future__ import annotations

from enum import Enum
from typing import Final


class SlicePolicy(Enum):
    """Determines restore behavior for session slices.

    Slice policies control which slices are included in snapshots and
    restored during rollback operations. This enables fine-grained control
    over state management, distinguishing between working state that should
    be rolled back on failure and append-only logs that should be preserved.

    Attributes:
        STATE: Working state that is restored on tool failure (default).
            Examples: Plan, VisibilityOverrides, WorkspaceDigest.
        LOG: Append-only historical records preserved during restore.
            Examples: ToolInvoked, PromptRendered, PromptExecuted.
    """

    STATE = "state"
    LOG = "log"


DEFAULT_SNAPSHOT_POLICIES: Final[frozenset[SlicePolicy]] = frozenset(
    {SlicePolicy.STATE}
)


__all__ = ["DEFAULT_SNAPSHOT_POLICIES", "SlicePolicy"]
