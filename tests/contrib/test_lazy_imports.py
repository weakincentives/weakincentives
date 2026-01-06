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

"""Tests for contrib package lazy imports."""

from __future__ import annotations

import pytest


class TestContribLazyImports:
    """Test lazy import behavior in contrib package."""

    def test_dir_includes_submodules(self) -> None:
        """Verify __dir__ returns expected submodules."""
        from weakincentives import contrib

        module_dir = dir(contrib)
        assert "mailbox" in module_dir
        assert "optimizers" in module_dir
        assert "tools" in module_dir

    def test_getattr_loads_mailbox(self) -> None:
        """Verify __getattr__ lazily loads mailbox submodule."""
        from weakincentives import contrib

        mailbox = contrib.mailbox
        assert mailbox is not None
        assert hasattr(mailbox, "RedisMailbox")

    def test_getattr_loads_tools(self) -> None:
        """Verify __getattr__ lazily loads tools submodule."""
        from weakincentives import contrib

        tools = contrib.tools
        assert tools is not None
        assert hasattr(tools, "PlanningToolsSection")

    def test_getattr_loads_optimizers(self) -> None:
        """Verify __getattr__ lazily loads optimizers submodule."""
        from weakincentives import contrib

        optimizers = contrib.optimizers
        assert optimizers is not None
        assert hasattr(optimizers, "WorkspaceDigestOptimizer")

    def test_getattr_raises_attribute_error_for_unknown(self) -> None:
        """Verify __getattr__ raises AttributeError for unknown attributes."""
        from weakincentives import contrib

        with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
            _ = contrib.nonexistent  # type: ignore[attr-defined]

    def test_direct_submodule_import_works(self) -> None:
        """Verify direct imports from submodules work correctly."""
        from weakincentives.contrib.tools import PlanningToolsSection

        assert PlanningToolsSection is not None

    def test_tools_imported_before_optimizers(self) -> None:
        """Verify optimizers can import from tools without circular dependency.

        This tests the core issue that lazy imports solve: workspace_digest
        in optimizers imports from tools.asteval, so if we eagerly import
        optimizers before tools, we get a ModuleNotFoundError.
        """
        from weakincentives.contrib.optimizers import WorkspaceDigestOptimizer

        assert WorkspaceDigestOptimizer is not None
