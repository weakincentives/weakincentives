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

"""Testing utilities for LangSmith integration.

This module provides mock implementations for testing LangSmith-integrated
code without making actual API calls.

Example::

    from weakincentives.contrib.langsmith import LangSmithConfig, LangSmithTelemetryHandler
    from weakincentives.contrib.langsmith.testing import MockLangSmithClient

    def test_telemetry():
        mock_client = MockLangSmithClient()
        config = LangSmithConfig(async_upload=False)
        handler = LangSmithTelemetryHandler(config, client=mock_client)

        # ... run evaluation ...

        assert mock_client.runs_created == 3
        assert mock_client.last_run.name == "my_prompt"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass
class MockRun:
    """Record of a created or updated run."""

    run_id: UUID
    name: str
    run_type: str
    project_name: str | None = None
    inputs: dict[str, Any] | None = None
    outputs: dict[str, Any] | None = None
    parent_run_id: UUID | None = None
    extra: dict[str, Any] | None = None
    tags: list[str] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: str | None = None
    events: list[dict[str, Any]] | None = None


@dataclass
class MockLangSmithClient:
    """Mock LangSmith client for testing.

    Captures all create_run and update_run calls for assertions.

    Example::

        mock = MockLangSmithClient()
        handler = LangSmithTelemetryHandler(config, client=mock)

        # After evaluation
        assert mock.runs_created == 2
        assert mock.runs["run-id"].name == "my_prompt"
        assert mock.last_run.run_type == "chain"
    """

    runs: dict[UUID, MockRun] = field(default_factory=dict)
    create_calls: list[dict[str, Any]] = field(default_factory=list)
    update_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def runs_created(self) -> int:
        """Number of runs created."""
        return len(self.create_calls)

    @property
    def runs_updated(self) -> int:
        """Number of runs updated."""
        return len(self.update_calls)

    @property
    def last_run(self) -> MockRun | None:
        """Most recently created run."""
        if not self.create_calls:
            return None
        last_call = self.create_calls[-1]
        run_id = last_call.get("run_id")
        if run_id is None:
            return None
        return self.runs.get(run_id)

    def create_run(  # noqa: PLR0913, PLR0917
        self,
        name: str,
        run_type: str,
        inputs: dict[str, Any] | None = None,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        project_name: str | None = None,
        extra: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        start_time: datetime | None = None,
    ) -> None:
        """Record a create_run call."""
        call = {
            "name": name,
            "run_type": run_type,
            "inputs": inputs,
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "project_name": project_name,
            "extra": extra,
            "tags": tags,
            "start_time": start_time,
        }
        self.create_calls.append(call)

        if run_id is not None:
            self.runs[run_id] = MockRun(
                run_id=run_id,
                name=name,
                run_type=run_type,
                project_name=project_name,
                inputs=inputs,
                parent_run_id=parent_run_id,
                extra=extra,
                tags=tags,
                start_time=start_time,
            )

    def update_run(  # noqa: PLR0913, PLR0917
        self,
        run_id: UUID,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
        end_time: datetime | None = None,
        extra: dict[str, Any] | None = None,
        events: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record an update_run call."""
        call = {
            "run_id": run_id,
            "outputs": outputs,
            "error": error,
            "end_time": end_time,
            "extra": extra,
            "events": events,
        }
        self.update_calls.append(call)

        if run_id in self.runs:
            run = self.runs[run_id]
            if outputs is not None:
                run.outputs = outputs
            if error is not None:
                run.error = error
            if end_time is not None:
                run.end_time = end_time
            if extra is not None:
                if run.extra is None:
                    run.extra = extra
                else:
                    run.extra.update(extra)
            if events is not None:
                run.events = events

    def reset(self) -> None:
        """Clear all recorded calls and runs."""
        self.runs.clear()
        self.create_calls.clear()
        self.update_calls.clear()

    def get_runs_by_type(self, run_type: str) -> list[MockRun]:
        """Get all runs of a specific type."""
        return [run for run in self.runs.values() if run.run_type == run_type]

    def get_child_runs(self, parent_run_id: UUID) -> list[MockRun]:
        """Get all runs with a specific parent."""
        return [run for run in self.runs.values() if run.parent_run_id == parent_run_id]


@dataclass
class MockHubPrompt:
    """Mock Hub prompt for testing."""

    template: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockLangSmithHub:
    """Mock LangSmith Hub for testing prompt management.

    Example::

        mock_hub = MockLangSmithHub()
        mock_hub.add_prompt("my-ns-my-prompt", "Template content")

        store = LangSmithPromptOverridesStore(config, client=mock_hub)
        override = store.resolve(descriptor, tag="latest")
    """

    prompts: dict[str, MockHubPrompt] = field(default_factory=dict)
    push_calls: list[dict[str, Any]] = field(default_factory=list)
    pull_calls: list[str] = field(default_factory=list)

    def add_prompt(
        self,
        identifier: str,
        template: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a mock prompt to the hub."""
        self.prompts[identifier] = MockHubPrompt(
            template=template,
            metadata=metadata or {},
        )

    def pull_prompt(
        self,
        prompt_identifier: str,
        *,
        include_model: bool = False,
    ) -> MockHubPrompt | None:
        """Pull a prompt from the mock hub."""
        self.pull_calls.append(prompt_identifier)

        # Handle versioned identifiers
        base_name = prompt_identifier.split(":")[0]

        # Try exact match first, then base name
        prompt = self.prompts.get(prompt_identifier)
        if prompt is None:
            prompt = self.prompts.get(base_name)

        if prompt is None:
            msg = f"Prompt not found: {prompt_identifier}"
            raise ValueError(msg)

        return prompt

    def push_prompt(  # noqa: PLR0913
        self,
        prompt_identifier: str,
        *,
        object: Any,  # noqa: A002, ANN401 - matches LangSmith SDK API
        parent_commit_hash: str | None = None,
        is_public: bool = False,
        description: str | None = None,
        readme: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Push a prompt to the mock hub."""
        call = {
            "identifier": prompt_identifier,
            "object": object,
            "parent_commit_hash": parent_commit_hash,
            "is_public": is_public,
            "description": description,
            "readme": readme,
            "tags": tags,
        }
        self.push_calls.append(call)

        # Extract template from object
        template = ""
        if isinstance(object, dict):
            template = object.get("template", "")
            metadata = object.get("metadata", {})
        else:
            template = str(object)
            metadata = {}

        self.prompts[prompt_identifier] = MockHubPrompt(
            template=template,
            metadata=metadata,
        )

        # Return mock commit hash
        import hashlib

        return hashlib.sha256(template.encode()).hexdigest()

    def reset(self) -> None:
        """Clear all prompts and calls."""
        self.prompts.clear()
        self.push_calls.clear()
        self.pull_calls.clear()


__all__ = [
    "MockHubPrompt",
    "MockLangSmithClient",
    "MockLangSmithHub",
    "MockRun",
]
