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

"""Tests for SDK configuration options."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tests.adapters.claude_agent_sdk.conftest import (
    MockResultMessage,
    MockSDKQuery,
    SimpleOutput,
    StrictClaudeAgentOptionsNoReasoning,
    create_sdk_mock,
    sdk_patches,
    setup_mock_query,
)
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentSDKModelConfig,
)
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime.session import Session

from .conftest import MockClaudeSDKClient, MockHookMatcher


class TestSDKConfigOptions:
    def test_passes_cwd_option(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(cwd="/home/user/project"),
        )

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        assert MockSDKQuery.captured_options[0].cwd == "/home/user/project"

    def test_passes_max_turns_option(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(max_turns=10),
        )

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        assert MockSDKQuery.captured_options[0].max_turns == 10

    def test_passes_allowed_tools_option(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(allowed_tools=("Read", "Write"))

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        assert MockSDKQuery.captured_options[0].allowed_tools == [
            "Read",
            "Write",
        ]

    def test_passes_disallowed_tools_option(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(disallowed_tools=("Bash",))

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        assert MockSDKQuery.captured_options[0].disallowed_tools == ["Bash"]

    def test_model_config_does_not_pass_unsupported_params(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Verify model config params that aren't supported by SDK are ignored."""
        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(
            model_config=ClaudeAgentSDKModelConfig(
                model="claude-sonnet-4-5-20250929",
                temperature=0.7,
                max_tokens=2000,
            ),
        )

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        assert not hasattr(MockSDKQuery.captured_options[0], "max_thinking_tokens") or (
            MockSDKQuery.captured_options[0].max_thinking_tokens is None
        )

    def test_creates_mcp_server_for_prompt_tools(self, session: Session) -> None:
        """Test that prompts with tools create an MCP server."""
        from tests.adapters.claude_agent_sdk.test_bridge import search_tool

        template_with_tools = PromptTemplate[SimpleOutput](
            ns="test",
            key="with_tools",
            sections=[
                MarkdownSection(
                    title="Task",
                    key="task",
                    template="Use the tool",
                    tools=(search_tool,),
                ),
            ],
        )
        prompt_with_tools = Prompt(template_with_tools)

        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter()

        mock_mcp_server = MagicMock(return_value={"type": "sdk"})

        with (
            sdk_patches(),
            patch(
                "weakincentives.adapters.claude_agent_sdk._sdk_options.create_mcp_server",
                mock_mcp_server,
            ),
        ):
            adapter.evaluate(prompt_with_tools, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        assert MockSDKQuery.captured_options[0].mcp_servers is not None
        assert "wink" in MockSDKQuery.captured_options[0].mcp_servers
        mock_mcp_server.assert_called_once()

    def test_passes_hooks_to_options(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Test that hooks are passed to ClaudeAgentOptions."""
        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        options = MockSDKQuery.captured_options[0]
        assert hasattr(options, "hooks")
        assert options.hooks is not None
        assert "PreToolUse" in options.hooks
        assert "PostToolUse" in options.hooks
        assert "Stop" in options.hooks
        assert "UserPromptSubmit" in options.hooks

    def test_passes_max_budget_usd_option(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(max_budget_usd=5.0),
        )

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        assert MockSDKQuery.captured_options[0].max_budget_usd == 5.0

    def test_passes_betas_option(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                betas=("extended-thinking", "computer-use")
            ),
        )

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        assert MockSDKQuery.captured_options[0].betas == [
            "extended-thinking",
            "computer-use",
        ]

    def test_transcript_collection_disabled_with_none(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Test that setting transcript_collection=None disables collection."""
        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(transcript_collection=None),
        )

        with sdk_patches():
            response = adapter.evaluate(simple_prompt, session=session)

        assert response.text == "Done"
        assert len(MockSDKQuery.captured_options) == 1

    def test_passes_reasoning_option_default_high(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        assert MockSDKQuery.captured_options[0].reasoning == "high"

    def test_passes_reasoning_option_max(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(
            model_config=ClaudeAgentSDKModelConfig(reasoning="max"),
        )

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        assert MockSDKQuery.captured_options[0].reasoning == "max"

    def test_passes_reasoning_option_disabled(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(
            model_config=ClaudeAgentSDKModelConfig(reasoning=None),
        )

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        assert (
            not hasattr(MockSDKQuery.captured_options[0], "reasoning")
            or MockSDKQuery.captured_options[0].reasoning is None
        )

    def test_filters_reasoning_when_sdk_options_do_not_support_it(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(
            model_config=ClaudeAgentSDKModelConfig(reasoning="high"),
        )

        with (
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                return_value=create_sdk_mock(),
            ),
            patch(
                "claude_agent_sdk.ClaudeSDKClient",
                MockClaudeSDKClient,
            ),
            patch(
                "claude_agent_sdk.types.ClaudeAgentOptions",
                StrictClaudeAgentOptionsNoReasoning,
            ),
            patch(
                "claude_agent_sdk.types.HookMatcher",
                MockHookMatcher,
            ),
            patch(
                "claude_agent_sdk.types.ResultMessage",
                MockResultMessage,
            ),
        ):
            response = adapter.evaluate(simple_prompt, session=session)

        assert response.text == "Done"
        assert len(MockSDKQuery.captured_options) == 1
        assert not hasattr(MockSDKQuery.captured_options[0], "reasoning")


class TestSupportedOptionNames:
    """Tests for supported_option_names and filter_unsupported_options."""

    def test_non_dataclass_with_fixed_params(self) -> None:
        from weakincentives.adapters.claude_agent_sdk._sdk_options import (
            supported_option_names,
        )

        class FixedOptions:
            def __init__(self, *, cwd: str, max_turns: int) -> None:
                pass

        result = supported_option_names(FixedOptions)
        assert result == {"cwd", "max_turns"}

    def test_signature_raises_returns_none(self) -> None:
        from weakincentives.adapters.claude_agent_sdk._sdk_options import (
            supported_option_names,
        )

        result = supported_option_names(int)
        assert result is None

    def test_filter_with_no_unsupported_keys(self) -> None:
        from weakincentives.adapters.claude_agent_sdk._sdk_options import (
            filter_unsupported_options,
        )

        class FixedOptions:
            def __init__(self, *, cwd: str) -> None:
                pass

        kwargs = {"cwd": "/tmp"}
        result = filter_unsupported_options(kwargs, options_type=FixedOptions)
        assert result == {"cwd": "/tmp"}
