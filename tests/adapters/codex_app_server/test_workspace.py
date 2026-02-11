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

"""Tests verifying the Codex App Server workspace re-exports.

The canonical WorkspaceSection tests live in ``tests/prompt/test_workspace.py``.
This file verifies that the adapter's backward-compatible re-exports resolve
correctly.
"""

from __future__ import annotations

from weakincentives.adapters.codex_app_server.workspace import (
    CodexWorkspaceSection,
    HostMount,
    HostMountPreview,
    WorkspaceBudgetExceededError,
    WorkspaceSecurityError,
    compute_workspace_fingerprint,
)
from weakincentives.prompt.workspace import (
    HostMount as CoreHostMount,
    HostMountPreview as CoreHostMountPreview,
    WorkspaceBudgetExceededError as CoreBudgetError,
    WorkspaceSection,
    WorkspaceSecurityError as CoreSecurityError,
    compute_workspace_fingerprint as core_compute_fingerprint,
)


class TestReExports:
    """Adapter workspace module re-exports point to core implementations."""

    def test_workspace_section_alias(self) -> None:
        assert CodexWorkspaceSection is WorkspaceSection

    def test_host_mount_reexport(self) -> None:
        assert HostMount is CoreHostMount

    def test_host_mount_preview_reexport(self) -> None:
        assert HostMountPreview is CoreHostMountPreview

    def test_budget_error_reexport(self) -> None:
        assert WorkspaceBudgetExceededError is CoreBudgetError

    def test_security_error_reexport(self) -> None:
        assert WorkspaceSecurityError is CoreSecurityError

    def test_compute_fingerprint_reexport(self) -> None:
        assert compute_workspace_fingerprint is core_compute_fingerprint
