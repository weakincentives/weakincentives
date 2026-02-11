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

"""Workspace re-exports for Codex App Server adapter.

The canonical implementation now lives in :mod:`weakincentives.prompt.workspace`.
This module re-exports the public API so that existing ``from
weakincentives.adapters.codex_app_server.workspace import …`` statements
continue to work.
"""

from __future__ import annotations

from ...prompt.workspace import (
    HostMount,
    HostMountPreview,
    WorkspaceBudgetExceededError,
    WorkspaceSection,
    WorkspaceSecurityError,
    compute_workspace_fingerprint,
)

#: Backward-compatible alias.
CodexWorkspaceSection = WorkspaceSection

__all__ = [
    "CodexWorkspaceSection",
    "HostMount",
    "HostMountPreview",
    "WorkspaceBudgetExceededError",
    "WorkspaceSection",
    "WorkspaceSecurityError",
    "compute_workspace_fingerprint",
]
