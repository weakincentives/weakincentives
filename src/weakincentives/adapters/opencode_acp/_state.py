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

"""Session state storage for OpenCode ACP adapter.

This module provides a typed dataclass slice for storing OpenCode session
state. WINK sessions use typed dataclass slices for state storage, enabling
session reuse across multiple prompt evaluations.
"""

from __future__ import annotations

from ...dataclasses import FrozenDataclass

__all__ = [
    "OpenCodeACPSessionState",
]


@FrozenDataclass()
class OpenCodeACPSessionState:
    """Stores OpenCode session ID and workspace fingerprint for reuse.

    When ``reuse_session=True`` is configured, the adapter stores the
    session ID and workspace fingerprint after ``session/new``. On subsequent
    evaluations, if the stored cwd and fingerprint match, the adapter calls
    ``session/load`` instead of ``session/new``.

    Attributes:
        session_id: The OpenCode session ID returned by ``session/new``.
        cwd: The absolute working directory passed to ``session/new``.
        workspace_fingerprint: A hash of mount config and budgets for
            deterministic reuse validation. None if no workspace section.
    """

    session_id: str
    cwd: str
    workspace_fingerprint: str | None
