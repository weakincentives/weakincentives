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

"""Tests for LangSmith testing utilities."""

from __future__ import annotations

from uuid import uuid4

import pytest

from weakincentives.contrib.langsmith.testing import (
    MockHubPrompt,
    MockLangSmithClient,
    MockLangSmithHub,
    MockRun,
)


class TestMockRun:
    """Tests for MockRun dataclass."""

    def test_creation(self) -> None:
        """MockRun can be created."""
        run_id = uuid4()
        run = MockRun(
            run_id=run_id,
            name="test_run",
            run_type="chain",
        )

        assert run.run_id == run_id
        assert run.name == "test_run"
        assert run.run_type == "chain"
        assert run.inputs is None
        assert run.outputs is None


class TestMockLangSmithClient:
    """Tests for MockLangSmithClient."""

    def test_create_run_records_call(self) -> None:
        """create_run records the call."""
        client = MockLangSmithClient()
        run_id = uuid4()

        client.create_run(
            name="test",
            run_type="chain",
            run_id=run_id,
            project_name="project",
            inputs={"key": "value"},
        )

        assert client.runs_created == 1
        assert len(client.create_calls) == 1
        assert client.create_calls[0]["name"] == "test"

    def test_create_run_stores_run(self) -> None:
        """create_run stores the run for later access."""
        client = MockLangSmithClient()
        run_id = uuid4()

        client.create_run(
            name="test",
            run_type="chain",
            run_id=run_id,
        )

        assert run_id in client.runs
        assert client.runs[run_id].name == "test"

    def test_update_run_records_call(self) -> None:
        """update_run records the call."""
        client = MockLangSmithClient()
        run_id = uuid4()

        # Create run first
        client.create_run(
            name="test",
            run_type="chain",
            run_id=run_id,
        )

        # Update it
        client.update_run(
            run_id=run_id,
            outputs={"result": "done"},
        )

        assert client.runs_updated == 1
        assert client.runs[run_id].outputs == {"result": "done"}

    def test_last_run(self) -> None:
        """last_run returns most recent run."""
        client = MockLangSmithClient()

        run_id1 = uuid4()
        run_id2 = uuid4()

        client.create_run(name="first", run_type="chain", run_id=run_id1)
        client.create_run(name="second", run_type="tool", run_id=run_id2)

        assert client.last_run is not None
        assert client.last_run.name == "second"

    def test_get_runs_by_type(self) -> None:
        """get_runs_by_type filters by run_type."""
        client = MockLangSmithClient()

        client.create_run(name="chain1", run_type="chain", run_id=uuid4())
        client.create_run(name="tool1", run_type="tool", run_id=uuid4())
        client.create_run(name="chain2", run_type="chain", run_id=uuid4())

        chain_runs = client.get_runs_by_type("chain")
        tool_runs = client.get_runs_by_type("tool")

        assert len(chain_runs) == 2
        assert len(tool_runs) == 1

    def test_get_child_runs(self) -> None:
        """get_child_runs filters by parent_run_id."""
        client = MockLangSmithClient()

        parent_id = uuid4()
        child1_id = uuid4()
        child2_id = uuid4()

        client.create_run(name="parent", run_type="chain", run_id=parent_id)
        client.create_run(
            name="child1",
            run_type="tool",
            run_id=child1_id,
            parent_run_id=parent_id,
        )
        client.create_run(
            name="child2",
            run_type="tool",
            run_id=child2_id,
            parent_run_id=parent_id,
        )

        children = client.get_child_runs(parent_id)

        assert len(children) == 2

    def test_reset(self) -> None:
        """reset clears all state."""
        client = MockLangSmithClient()

        client.create_run(name="test", run_type="chain", run_id=uuid4())
        client.reset()

        assert client.runs_created == 0
        assert len(client.runs) == 0


class TestMockLangSmithHub:
    """Tests for MockLangSmithHub."""

    def test_add_and_pull_prompt(self) -> None:
        """add_prompt and pull_prompt work together."""
        hub = MockLangSmithHub()

        hub.add_prompt("test-prompt", "Template content")

        prompt = hub.pull_prompt("test-prompt")

        assert prompt is not None
        assert prompt.template == "Template content"

    def test_pull_missing_raises(self) -> None:
        """pull_prompt raises for missing prompt."""
        hub = MockLangSmithHub()

        with pytest.raises(ValueError, match="not found"):
            hub.pull_prompt("nonexistent")

    def test_pull_with_version(self) -> None:
        """pull_prompt handles versioned identifiers."""
        hub = MockLangSmithHub()

        hub.add_prompt("test-prompt", "Template")

        # Should work with version suffix
        prompt = hub.pull_prompt("test-prompt:v1")

        assert prompt is not None

    def test_push_prompt_stores_and_returns_hash(self) -> None:
        """push_prompt stores prompt and returns hash."""
        hub = MockLangSmithHub()

        commit_hash = hub.push_prompt(
            "new-prompt",
            object={"template": "New template"},
        )

        assert len(commit_hash) == 64  # SHA-256 hex
        assert "new-prompt" in hub.prompts
        assert hub.prompts["new-prompt"].template == "New template"

    def test_push_records_call(self) -> None:
        """push_prompt records the call."""
        hub = MockLangSmithHub()

        hub.push_prompt(
            "prompt",
            object={"template": "Test"},
            description="Test commit",
        )

        assert len(hub.push_calls) == 1
        assert hub.push_calls[0]["identifier"] == "prompt"
        assert hub.push_calls[0]["description"] == "Test commit"

    def test_pull_records_call(self) -> None:
        """pull_prompt records the call."""
        hub = MockLangSmithHub()
        hub.add_prompt("test", "Template")

        hub.pull_prompt("test")

        assert len(hub.pull_calls) == 1
        assert hub.pull_calls[0] == "test"

    def test_reset(self) -> None:
        """reset clears all state."""
        hub = MockLangSmithHub()

        hub.add_prompt("test", "Template")
        hub.pull_prompt("test")
        hub.push_prompt("new", object={"template": "New"})

        hub.reset()

        assert len(hub.prompts) == 0
        assert len(hub.pull_calls) == 0
        assert len(hub.push_calls) == 0


class TestMockHubPrompt:
    """Tests for MockHubPrompt dataclass."""

    def test_creation(self) -> None:
        """MockHubPrompt can be created."""
        prompt = MockHubPrompt(
            template="Test template",
            metadata={"key": "value"},
        )

        assert prompt.template == "Test template"
        assert prompt.metadata == {"key": "value"}

    def test_default_metadata(self) -> None:
        """MockHubPrompt has default empty metadata."""
        prompt = MockHubPrompt(template="Test")

        assert prompt.metadata == {}
