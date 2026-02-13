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

"""ACP session state slice for session reuse."""

from __future__ import annotations

from ...dataclasses import FrozenDataclass

__all__ = ["ACPSessionState"]


@FrozenDataclass()
class ACPSessionState:
    """Stores ACP session ID and workspace fingerprint for reuse.

    Attributes:
        session_id: The ACP session identifier.
        cwd: Working directory associated with the session.
        workspace_fingerprint: Optional fingerprint for detecting workspace
            changes that invalidate the session.
    """

    session_id: str
    cwd: str
    workspace_fingerprint: str | None
