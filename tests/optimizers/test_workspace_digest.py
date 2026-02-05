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

"""Tests for WorkspaceDigestOptimizer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    HostMount,
)
from weakincentives.contrib.optimizers import (
    WorkspaceDigestOptimizer,
    WorkspaceDigestResult,
)
from weakincentives.contrib.tools.digests import latest_workspace_digest
from weakincentives.runtime import InProcessDispatcher, Session


@pytest.fixture
def session() -> Session:
    """Create a session for testing."""
    dispatcher = InProcessDispatcher()
    return Session(dispatcher=dispatcher)


@pytest.fixture
def mock_adapter() -> MagicMock:
    """Create a mock adapter."""
    return MagicMock(spec=ClaudeAgentSDKAdapter)


class TestWorkspaceDigestOptimizer:
    """Tests for WorkspaceDigestOptimizer."""

    def test_result_defaults(self) -> None:
        """WorkspaceDigestResult has sensible defaults."""
        result = WorkspaceDigestResult(section_key="test")
        assert result.section_key == "test"
        assert result.summary == ""
        assert result.digest == ""
        assert result.success is True
        assert result.error == ""

    def test_result_with_error(self) -> None:
        """WorkspaceDigestResult can represent failure."""
        result = WorkspaceDigestResult(
            section_key="test",
            success=False,
            error="Something went wrong",
        )
        assert not result.success
        assert result.error == "Something went wrong"

    def test_optimizer_uses_provided_adapter(
        self, session: Session, mock_adapter: MagicMock
    ) -> None:
        """Optimizer uses provided adapter instead of creating new one."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.output = MagicMock(
            summary="Test summary",
            digest="Test digest content",
        )
        mock_adapter.evaluate.return_value = mock_response

        optimizer = WorkspaceDigestOptimizer(
            mounts=[],
            adapter=mock_adapter,  # type: ignore[arg-type]
        )

        result = optimizer.optimize(session)

        assert result.success
        assert result.summary == "Test summary"
        assert result.digest == "Test digest content"
        mock_adapter.evaluate.assert_called_once()

    def test_optimizer_stores_digest_in_session(
        self, session: Session, mock_adapter: MagicMock
    ) -> None:
        """Optimizer stores generated digest in session."""
        mock_response = MagicMock()
        mock_response.output = MagicMock(
            summary="Project summary",
            digest="Full project digest",
        )
        mock_adapter.evaluate.return_value = mock_response

        optimizer = WorkspaceDigestOptimizer(
            mounts=[],
            adapter=mock_adapter,  # type: ignore[arg-type]
        )

        _ = optimizer.optimize(session, section_key="my-digest")

        # Verify digest is stored in session
        digest = latest_workspace_digest(session, "my-digest")
        assert digest is not None
        assert digest.summary == "Project summary"
        assert digest.body == "Full project digest"

    def test_optimizer_returns_error_on_no_output(
        self, session: Session, mock_adapter: MagicMock
    ) -> None:
        """Optimizer returns error result when no output is returned."""
        mock_response = MagicMock()
        mock_response.output = None
        mock_adapter.evaluate.return_value = mock_response

        optimizer = WorkspaceDigestOptimizer(
            mounts=[],
            adapter=mock_adapter,  # type: ignore[arg-type]
        )

        result = optimizer.optimize(session)

        assert not result.success
        assert "No structured output" in result.error

    def test_optimizer_handles_evaluation_error(
        self, session: Session, mock_adapter: MagicMock
    ) -> None:
        """Optimizer returns error result when evaluation fails."""
        mock_adapter.evaluate.side_effect = RuntimeError("SDK error")

        optimizer = WorkspaceDigestOptimizer(
            mounts=[],
            adapter=mock_adapter,  # type: ignore[arg-type]
        )

        result = optimizer.optimize(session)

        assert not result.success
        assert "SDK error" in result.error

    def test_optimizer_creates_adapter_when_not_provided(
        self, session: Session
    ) -> None:
        """Optimizer creates adapter when none is provided."""
        optimizer = WorkspaceDigestOptimizer(mounts=[])

        # Create a mock adapter
        mock_adapter = MagicMock()
        mock_adapter.evaluate.return_value = MagicMock(
            output=MagicMock(summary="s", digest="d")
        )

        # Mock the adapter creation to avoid actual SDK calls
        with patch.object(
            WorkspaceDigestOptimizer,
            "_create_adapter",
            return_value=mock_adapter,
        ) as mock_create:
            _ = optimizer.optimize(session)
            mock_create.assert_called_once()

    def test_optimizer_builds_prompt_with_workspace_section(
        self, session: Session
    ) -> None:
        """Optimizer builds prompt containing workspace section."""
        optimizer = WorkspaceDigestOptimizer(mounts=[])

        prompt = optimizer._build_optimization_prompt(session)

        # Check the prompt has expected sections
        rendered = prompt.render()
        assert "Optimization Goal" in rendered.text
        assert "Expectations" in rendered.text
        assert "Output Format" in rendered.text

    def test_create_adapter_default_config(self) -> None:
        """_create_adapter creates adapter with default config."""
        optimizer = WorkspaceDigestOptimizer(mounts=[], max_turns=15)

        # Patch the ClaudeAgentSDKAdapter to avoid SDK dependency
        with patch(
            "weakincentives.contrib.optimizers.workspace_digest.ClaudeAgentSDKAdapter"
        ) as mock_cls:
            mock_cls.__getitem__ = MagicMock(return_value=mock_cls)
            _ = optimizer._create_adapter()

            # Verify it was called with expected config
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["client_config"].max_turns == 15
            assert call_kwargs["client_config"].permission_mode == "bypassPermissions"


class TestWorkspaceDigestOptimizerIntegration:
    """Integration-level tests for WorkspaceDigestOptimizer."""

    def test_full_optimization_flow_with_mounts(self, tmp_path: Path) -> None:
        """Test full optimization flow with file mounts."""
        # Create a test file
        test_file = tmp_path / "README.md"
        test_file.write_text("# Test Project\n\nA test project for optimization.")

        mount = HostMount(host_path=str(tmp_path))

        optimizer = WorkspaceDigestOptimizer(mounts=[mount])

        # Verify prompt is built correctly with mounts
        session = Session(dispatcher=InProcessDispatcher())
        prompt = optimizer._build_optimization_prompt(session)

        # The workspace section should have the mount
        rendered = prompt.render()
        assert "Workspace" in rendered.text
