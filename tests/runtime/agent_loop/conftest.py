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

"""Shared test infrastructure for AgentLoop tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    SectionPath,
    SectionVisibility,
    VisibilityExpansionRequired,
)
from weakincentives.runtime.agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopRequest,
    AgentLoopResult,
)
from weakincentives.runtime.mailbox import (
    FakeMailbox,
    InMemoryMailbox,
)
from weakincentives.runtime.run_context import RunContext
from weakincentives.runtime.session import Session
from weakincentives.runtime.session.protocols import SessionProtocol


@dataclass(slots=True, frozen=True)
class SampleRequest:
    """Sample request type for testing."""

    message: str


@dataclass(slots=True, frozen=True)
class SampleOutput:
    """Sample output type for testing."""

    result: str


@dataclass(slots=True, frozen=True)
class SampleParams:
    """Sample params type for testing."""

    content: str


@dataclass(slots=True, frozen=True)
class CustomResource:
    """Custom resource for testing resource injection."""

    name: str


class MockAdapter(ProviderAdapter[SampleOutput]):
    """Mock adapter for testing AgentLoop behavior."""

    def __init__(
        self,
        *,
        response: PromptResponse[SampleOutput] | None = None,
        error: Exception | None = None,
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]]
        | None = None,
    ) -> None:
        self._response = response or PromptResponse(
            prompt_name="test",
            text="test output",
            output=SampleOutput(result="success"),
        )
        self._error = error
        self._visibility_requests = list(visibility_requests or [])
        self._call_count = 0
        self._last_budget_tracker: BudgetTracker | None = None
        self._budget_trackers: list[BudgetTracker | None] = []
        self._last_deadline: Deadline | None = None
        self._last_session: SessionProtocol | None = None
        # Track prompts to verify resources through prompt.resources
        self._last_prompt: Prompt[SampleOutput] | None = None
        self._prompts: list[Prompt[SampleOutput]] = []
        # Track resources captured during evaluate (while context is active)
        self._last_custom_resource: CustomResource | None = None
        self._custom_resources: list[CustomResource | None] = []
        # Track run_context passed during evaluate
        self._last_run_context: RunContext | None = None
        self._run_contexts: list[RunContext | None] = []

    def evaluate(
        self,
        prompt: Prompt[SampleOutput],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: object = None,
        run_context: RunContext | None = None,
    ) -> PromptResponse[SampleOutput]:
        del budget, heartbeat
        self._call_count += 1
        self._last_run_context = run_context
        self._run_contexts.append(run_context)
        self._last_budget_tracker = budget_tracker
        self._budget_trackers.append(budget_tracker)
        self._last_deadline = deadline
        self._last_session = session
        # Capture prompt to verify resources via prompt.resources
        self._last_prompt = prompt
        self._prompts.append(prompt)

        # Enter resource context (like real adapters do)
        with prompt.resources:
            # Capture resource during evaluate while context is active
            try:
                self._last_custom_resource = prompt.resources.get(CustomResource)
            except Exception:
                # UnboundResourceError: resource type not registered
                self._last_custom_resource = None
            self._custom_resources.append(self._last_custom_resource)

            # If there are visibility requests remaining, raise the exception
            if self._visibility_requests:
                overrides = self._visibility_requests.pop(0)
                raise VisibilityExpansionRequired(
                    "Expansion required",
                    requested_overrides=overrides,
                    reason="test",
                    section_keys=tuple(k[0] for k in overrides),
                )

            if self._error is not None:
                raise self._error

            return self._response


class SampleLoop(AgentLoop[SampleRequest, SampleOutput]):
    """Test implementation of AgentLoop."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[SampleOutput],
        requests: InMemoryMailbox[
            AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
        ]
        | FakeMailbox[AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]],
        config: AgentLoopConfig | None = None,
        worker_id: str = "",
    ) -> None:
        super().__init__(
            adapter=adapter, requests=requests, config=config, worker_id=worker_id
        )
        self._template = PromptTemplate[SampleOutput](
            ns="test",
            key="test-prompt",
            sections=[
                MarkdownSection[SampleParams](
                    title="Test",
                    template="$content",
                    key="test",
                ),
            ],
        )
        self.session_created: Session | None = None
        self.finalize_called = False

    def prepare(
        self,
        request: SampleRequest,
        *,
        experiment: object = None,
    ) -> tuple[Prompt[SampleOutput], Session]:
        _ = experiment
        prompt = Prompt(self._template).bind(SampleParams(content=request.message))
        session = Session(tags={"loop": "test"})
        self.session_created = session
        return prompt, session

    def finalize(
        self,
        prompt: Prompt[SampleOutput],
        session: Session,
        output: SampleOutput | None,
    ) -> SampleOutput | None:
        del prompt
        self.finalize_called = True
        _ = session
        return output
