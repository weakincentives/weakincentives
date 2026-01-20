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

"""Tests for hook registry module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from weakincentives.adapters.claude_agent_sdk import HookRegistry, HookSet
from weakincentives.adapters.claude_agent_sdk._hooks import HookContext


@pytest.fixture
def mock_hook_context() -> MagicMock:
    """Create a mock HookContext for testing."""
    mock = MagicMock(spec=HookContext)
    mock.prompt_name = "test-prompt"
    mock.adapter_name = "test-adapter"
    mock.session = MagicMock()
    mock.resources = MagicMock()
    mock.deadline = None
    mock.budget_tracker = None
    mock.heartbeat = None
    mock.run_context = None
    mock.stats = MagicMock()
    return mock


class TestHookSet:
    """Tests for HookSet dataclass."""

    def test_creation(self) -> None:
        """Test that HookSet can be created with all hooks."""
        hooks = {
            "pre_tool_use": MagicMock(),
            "post_tool_use": MagicMock(),
            "stop": MagicMock(),
            "user_prompt_submit": MagicMock(),
            "subagent_start": MagicMock(),
            "subagent_stop": MagicMock(),
            "pre_compact": MagicMock(),
            "notification": MagicMock(),
        }

        hook_set = HookSet(
            pre_tool_use=hooks["pre_tool_use"],
            post_tool_use=hooks["post_tool_use"],
            stop=hooks["stop"],
            user_prompt_submit=hooks["user_prompt_submit"],
            subagent_start=hooks["subagent_start"],
            subagent_stop=hooks["subagent_stop"],
            pre_compact=hooks["pre_compact"],
            notification=hooks["notification"],
        )

        assert hook_set.pre_tool_use is hooks["pre_tool_use"]
        assert hook_set.post_tool_use is hooks["post_tool_use"]
        assert hook_set.stop is hooks["stop"]
        assert hook_set.user_prompt_submit is hooks["user_prompt_submit"]
        assert hook_set.subagent_start is hooks["subagent_start"]
        assert hook_set.subagent_stop is hooks["subagent_stop"]
        assert hook_set.pre_compact is hooks["pre_compact"]
        assert hook_set.notification is hooks["notification"]


class TestHookRegistry:
    """Tests for HookRegistry class."""

    def test_create_hook_set_without_checker(
        self, mock_hook_context: MagicMock
    ) -> None:
        """Test creating hook set without task completion checker."""
        registry = HookRegistry(mock_hook_context)
        hook_set = registry.create_hook_set(
            stop_on_structured_output=True,
            task_completion_checker=None,
        )

        assert hook_set.pre_tool_use is not None
        assert hook_set.post_tool_use is not None
        assert hook_set.stop is not None
        assert hook_set.user_prompt_submit is not None
        assert hook_set.subagent_start is not None
        assert hook_set.subagent_stop is not None
        assert hook_set.pre_compact is not None
        assert hook_set.notification is not None

    def test_create_hook_set_with_checker(self, mock_hook_context: MagicMock) -> None:
        """Test creating hook set with task completion checker."""
        mock_checker = MagicMock()
        registry = HookRegistry(mock_hook_context)
        hook_set = registry.create_hook_set(
            stop_on_structured_output=True,
            task_completion_checker=mock_checker,
        )

        # All hooks should be created
        assert hook_set.pre_tool_use is not None
        assert hook_set.post_tool_use is not None
        assert hook_set.stop is not None
        assert hook_set.user_prompt_submit is not None

    def test_to_sdk_hooks(self, mock_hook_context: MagicMock) -> None:
        """Test converting HookSet to SDK hooks dict."""
        registry = HookRegistry(mock_hook_context)
        hook_set = registry.create_hook_set()

        # Import the SDK types for testing
        from claude_agent_sdk.types import HookMatcher

        hooks_dict = HookRegistry.to_sdk_hooks(hook_set)

        # Verify all hook types are present
        assert "PreToolUse" in hooks_dict
        assert "PostToolUse" in hooks_dict
        assert "Stop" in hooks_dict
        assert "UserPromptSubmit" in hooks_dict
        assert "SubagentStart" in hooks_dict
        assert "SubagentStop" in hooks_dict
        assert "PreCompact" in hooks_dict
        assert "Notification" in hooks_dict

        # Each hook type should have exactly one HookMatcher
        for matchers in hooks_dict.values():
            assert len(matchers) == 1
            assert isinstance(matchers[0], HookMatcher)

    def test_get_hook_type_names(self) -> None:
        """Test getting hook type names."""
        names = HookRegistry.get_hook_type_names()

        assert names == [
            "PreToolUse",
            "PostToolUse",
            "Stop",
            "UserPromptSubmit",
            "SubagentStart",
            "SubagentStop",
            "PreCompact",
            "Notification",
        ]
