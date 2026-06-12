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

"""Tests for IsolationConfig integration with the adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.adapters.claude_agent_sdk.conftest import (
    MockResultMessage,
    MockSDKQuery,
    SimpleOutput,
    sdk_patches,
    setup_mock_query,
)
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
)
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime.events import PromptExecuted
from weakincentives.runtime.session import Session


class TestIsolationConfig:
    """Tests for IsolationConfig integration with the adapter."""

    def test_evaluate_with_isolation_creates_ephemeral_home(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Test that isolation config creates an ephemeral home and cleans it up."""
        from weakincentives.adapters.claude_agent_sdk import (
            IsolationConfig,
            NetworkPolicy,
        )

        setup_mock_query(
            [MockResultMessage(result="Hello!", usage={"input_tokens": 10})]
        )

        isolation = IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
            sandbox_enabled=True,
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                permission_mode="bypassPermissions",
                isolation=isolation,
            ),
        )

        with sdk_patches():
            response = adapter.evaluate(simple_prompt, session=session)

        assert response.text == "Hello!"

        assert len(MockSDKQuery.captured_options) == 1
        options = MockSDKQuery.captured_options[0]

        assert hasattr(options, "env")
        env: dict[str, str] = options.env  # type: ignore[assignment]
        assert isinstance(env, dict)
        assert "HOME" in env
        home_value = env["HOME"]
        assert isinstance(home_value, str)
        assert "claude-agent-" in home_value

        assert hasattr(options, "setting_sources")
        assert options.setting_sources == ["user"]

    def test_evaluate_with_isolation_cleans_up_on_error(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Test that ephemeral home is cleaned up even when SDK raises an error."""
        from weakincentives.adapters.claude_agent_sdk import (
            IsolationConfig,
            NetworkPolicy,
        )

        MockSDKQuery.reset()
        MockSDKQuery.set_error(RuntimeError("SDK error"))

        isolation = IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                permission_mode="bypassPermissions",
                isolation=isolation,
            ),
        )

        with sdk_patches():
            with pytest.raises(Exception):  # noqa: B017
                adapter.evaluate(simple_prompt, session=session)

    def test_evaluate_with_isolation_mounts_section_skills(
        self, session: Session, tmp_path: Path
    ) -> None:
        """Skills attached to sections are mounted via ephemeral home."""
        from weakincentives.adapters.claude_agent_sdk import (
            IsolationConfig,
            NetworkPolicy,
        )
        from weakincentives.skills import SkillMount

        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill\n---\n\n# Test Skill\n"
        )

        skill = SkillMount(source=skill_dir)
        template = PromptTemplate[SimpleOutput].create(
            ns="test",
            key="with-skills",
            sections=[
                MarkdownSection(
                    title="Section with Skills",
                    template="Do something",
                    key="section-with-skills",
                    skills=[skill],
                ),
            ],
        )
        prompt_with_skills: Prompt[SimpleOutput] = Prompt(template)

        setup_mock_query(
            [MockResultMessage(result="Done!", usage={"input_tokens": 10})]
        )

        isolation = IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                permission_mode="bypassPermissions",
                isolation=isolation,
            ),
        )

        with sdk_patches():
            response = adapter.evaluate(prompt_with_skills, session=session)

        assert response.text == "Done!"

        assert len(MockSDKQuery.captured_options) == 1
        options = MockSDKQuery.captured_options[0]
        assert hasattr(options, "env")
        env: dict[str, str] = options.env  # type: ignore[assignment]
        assert "HOME" in env

    def test_no_permission_mode_when_none(
        self, session: Session, simple_prompt: Prompt[None]
    ) -> None:
        """When permission_mode is None, it's not added to options."""
        config = ClaudeAgentSDKClientConfig(permission_mode=None)
        adapter = ClaudeAgentSDKAdapter(client_config=config)

        MockSDKQuery.reset()
        MockSDKQuery.set_results([MockResultMessage(result="Done")])

        with sdk_patches():
            _ = adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        options = MockSDKQuery.captured_options[0]
        assert (
            not hasattr(options, "permission_mode") or options.permission_mode is None
        )

    def test_suppress_stderr_option(
        self, session: Session, simple_prompt: Prompt[None]
    ) -> None:
        """When suppress_stderr is True, stderr callback is added to options."""
        config = ClaudeAgentSDKClientConfig(suppress_stderr=True)
        adapter = ClaudeAgentSDKAdapter(client_config=config)

        MockSDKQuery.reset()
        MockSDKQuery.set_results([MockResultMessage(result="Done")])

        with sdk_patches():
            _ = adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        options = MockSDKQuery.captured_options[0]
        assert hasattr(options, "stderr")
        assert callable(options.stderr)

    def test_suppress_stderr_false(
        self, session: Session, simple_prompt: Prompt[None]
    ) -> None:
        """When suppress_stderr is False, stderr is still captured for debug logging."""
        config = ClaudeAgentSDKClientConfig(suppress_stderr=False)
        adapter = ClaudeAgentSDKAdapter(client_config=config)

        MockSDKQuery.reset()
        MockSDKQuery.set_results([MockResultMessage(result="Done")])

        with sdk_patches():
            _ = adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        options = MockSDKQuery.captured_options[0]
        assert hasattr(options, "stderr") and options.stderr is not None

    def test_stderr_handler_buffers_output(
        self, session: Session, simple_prompt: Prompt[None]
    ) -> None:
        """Stderr handler buffers output for debug logging and error payloads."""
        adapter = ClaudeAgentSDKAdapter()

        MockSDKQuery.reset()
        MockSDKQuery.set_results([MockResultMessage(result="Done")])

        with sdk_patches():
            _ = adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        options = MockSDKQuery.captured_options[0]
        stderr_handler = options.stderr

        stderr_handler("Test stderr line 1\n")
        stderr_handler("Test stderr line 2\n")

        assert len(adapter._stderr_buffer) == 2
        assert adapter._stderr_buffer[0] == "Test stderr line 1\n"
        assert adapter._stderr_buffer[1] == "Test stderr line 2\n"

    def test_message_without_result(
        self, session: Session, untyped_prompt: Prompt[None]
    ) -> None:
        """Messages without result attribute or with falsy result are handled."""
        MockSDKQuery.reset()
        message_without_result = MockResultMessage()
        message_without_result.result = None
        MockSDKQuery.set_results([message_without_result])

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            response = adapter.evaluate(untyped_prompt, session=session)

        assert response.text is None

    def test_non_dict_usage_ignored(
        self, session: Session, simple_prompt: Prompt[None]
    ) -> None:
        """Non-dict usage values are gracefully ignored."""
        MockSDKQuery.reset()
        message = MockResultMessage(result="Done")
        message.usage = "not a dict"  # type: ignore[assignment]
        MockSDKQuery.set_results([message])

        events: list[PromptExecuted] = []
        session.dispatcher.subscribe(PromptExecuted, lambda e: events.append(e))

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            _ = adapter.evaluate(simple_prompt, session=session)

        assert len(events) == 1
        usage = events[0].usage
        assert usage is not None
        assert usage.input_tokens is None
        assert usage.output_tokens is None

    def test_opens_empty_sandbox_when_no_sandbox_config(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """When the template declares no sandbox, an empty sandbox is the cwd."""
        MockSDKQuery.reset()
        MockSDKQuery.set_results([MockResultMessage(result="Done")])

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            _ = adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        options = MockSDKQuery.captured_options[0]
        assert hasattr(options, "cwd")
        assert options.cwd is not None
        assert "wink-sandbox-" in options.cwd

    def test_temp_folder_cleaned_up_after_evaluate(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Temp folder is cleaned up after evaluate completes."""
        MockSDKQuery.reset()
        MockSDKQuery.set_results([MockResultMessage(result="Done")])

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            _ = adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        temp_cwd = MockSDKQuery.captured_options[0].cwd

        assert not Path(temp_cwd).exists()

    def test_temp_folder_cleaned_up_on_error(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Temp folder is cleaned up even when SDK raises an error."""
        MockSDKQuery.reset()
        MockSDKQuery.set_error(RuntimeError("SDK error"))

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            with pytest.raises(Exception):  # noqa: B017
                adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        temp_cwd = MockSDKQuery.captured_options[0].cwd

        assert not Path(temp_cwd).exists()

    def test_cwd_is_sandbox_root_for_sandbox_template(
        self, session: Session, tmp_path: Path
    ) -> None:
        """Templates with sandbox config run with cwd at the opened sandbox root."""
        from weakincentives.sandbox import HostMount, WorkspaceConfig

        MockSDKQuery.reset()
        MockSDKQuery.set_results([MockResultMessage(result="Done")])

        (tmp_path / "README.md").write_text("hello")
        template = PromptTemplate[SimpleOutput].create(
            ns="test",
            key="with-sandbox",
            sections=[
                MarkdownSection(
                    title="Test",
                    template="Hello",
                    key="test",
                ),
            ],
            workspace=WorkspaceConfig(mounts=(HostMount(host_path=str(tmp_path)),)),
        )
        prompt_with_sandbox: Prompt[SimpleOutput] = Prompt(template)

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            _ = adapter.evaluate(prompt_with_sandbox, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        options = MockSDKQuery.captured_options[0]

        cwd = getattr(options, "cwd", None)
        assert cwd is not None
        assert "wink-sandbox-" in cwd

    def test_workspace_preview_renders_opened_sandbox(
        self, session: Session, tmp_path: Path
    ) -> None:
        """The workspace preview lists the opened sandbox contents."""
        from weakincentives.sandbox import HostMount, WorkspaceConfig

        MockSDKQuery.reset()
        MockSDKQuery.set_results([MockResultMessage(result="Done")])

        (tmp_path / "README.md").write_text("hello")
        template = PromptTemplate[SimpleOutput].create(
            ns="test",
            key="with-sandbox-preview",
            workspace=WorkspaceConfig(
                mounts=(HostMount(host_path=str(tmp_path), mount_path="src"),)
            ),
        )
        prompt_with_sandbox: Prompt[SimpleOutput] = Prompt(template)

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            _ = adapter.evaluate(prompt_with_sandbox, session=session)

        assert len(MockSDKQuery.captured_prompts) == 1
        assert "- src/" in MockSDKQuery.captured_prompts[0]
