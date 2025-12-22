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

"""Main loop orchestration for agent workflow execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from ..budget import Budget, BudgetTracker
from ..dataclasses import FrozenDataclass
from ..deadlines import Deadline
from ..prompt.errors import VisibilityExpansionRequired
from ..prompt.tool import ResourceRegistry
from ..prompt.visibility_overrides import SetVisibilityOverride
from .mailbox import Mailbox, Message
from .session import Session

if TYPE_CHECKING:
    from ..adapters.core import PromptResponse, ProviderAdapter
    from ..prompt import Prompt


@FrozenDataclass()
class MainLoopConfig:
    """Configuration for MainLoop execution defaults.

    Request-level ``budget``, ``deadline``, and ``resources`` override config defaults.
    """

    deadline: Deadline | None = None
    budget: Budget | None = None
    resources: ResourceRegistry | None = None
    visibility_timeout: int = 300
    wait_time_seconds: int = 20


@FrozenDataclass()
class MainLoopRequest[UserRequestT]:
    """Request for MainLoop execution with optional constraints.

    The ``budget``, ``deadline``, and ``resources`` fields override config defaults
    when set. A fresh ``BudgetTracker`` is created per execution.
    """

    request: UserRequestT
    budget: Budget | None = None
    deadline: Deadline | None = None
    resources: ResourceRegistry | None = None
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@FrozenDataclass()
class MainLoopResult[OutputT]:
    """Result of MainLoop execution (success or failure)."""

    request_id: UUID
    response: PromptResponse[OutputT] | None
    error: Exception | None
    session_id: UUID
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def success(self) -> bool:
        """Return True if execution succeeded."""
        return self.error is None


class MainLoop[UserRequestT, OutputT](ABC):
    """Abstract orchestrator for agent workflow execution.

    MainLoop standardizes agent workflow orchestration: receive request, build
    prompt, evaluate, handle visibility expansion, return result. Implementations
    define only the domain-specific factories via ``create_prompt`` and
    ``create_session``.

    Execution modes:

    1. **Mailbox-driven** (via ``run()``): Process requests from a mailbox queue.
       Supports request-reply pattern with ``send_expecting_reply()``.

    2. **Direct** (via ``execute()``): Synchronous single-request execution.

    Execution flow:
        1. Receive ``MainLoopRequest`` via mailbox or direct ``execute()`` call
        2. Create session via ``create_session()``
        3. Create prompt via ``create_prompt(request)``
        4. Evaluate with adapter
        5. On ``VisibilityExpansionRequired``: accumulate overrides, retry step 4
        6. Return ``MainLoopResult`` (or reply via mailbox)

    Usage::

        # Mailbox-driven usage
        mailbox: InMemoryMailbox[MainLoopRequest[ReviewRequest], MainLoopResult[ReviewResult]] = InMemoryMailbox()
        loop = CodeReviewLoop(adapter=adapter, requests=mailbox)

        # Client sends request expecting reply
        reply = mailbox.send_expecting_reply(MainLoopRequest(request=ReviewRequest(...)))

        # Loop processes in background
        loop.run(max_iterations=1)

        # Client awaits result
        result = reply.wait(timeout=60)

        # Or direct usage
        result, session = loop.execute(ReviewRequest(...))
    """

    _adapter: ProviderAdapter[OutputT]
    _requests: Mailbox[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]]
    _config: MainLoopConfig

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[OutputT],
        requests: Mailbox[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]],
        config: MainLoopConfig | None = None,
    ) -> None:
        """Initialize the MainLoop with an adapter and mailbox.

        Args:
            adapter: Provider adapter for prompt evaluation.
            requests: Mailbox for receiving requests and sending replies.
            config: Optional configuration for default deadline/budget.
        """
        super().__init__()
        self._adapter = adapter
        self._requests = requests
        self._config = config if config is not None else MainLoopConfig()

    @abstractmethod
    def create_prompt(self, request: UserRequestT) -> Prompt[OutputT]:
        """Create a prompt for the given request.

        Subclasses must implement this method to construct the prompt
        appropriate for their domain.

        Args:
            request: The user request to create a prompt for.

        Returns:
            A bound prompt ready for evaluation.
        """
        ...

    @abstractmethod
    def create_session(self) -> Session:
        """Create a session for execution.

        Subclasses must implement this method to construct the session
        with appropriate tags, reducers, and initial state.

        Returns:
            A session configured for the loop's domain.
        """
        ...

    def run(self, *, max_iterations: int | None = None) -> None:
        """Process requests from the mailbox.

        Continuously receives messages from the mailbox and processes them.
        For messages expecting a reply, sends the result via the reply channel.
        For fire-and-forget messages, acknowledges after processing.

        Args:
            max_iterations: Maximum number of messages to process. None means
                run until mailbox is empty (with wait).
        """
        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            messages = self._requests.receive(
                visibility_timeout=self._config.visibility_timeout,
                wait_time_seconds=self._config.wait_time_seconds,
            )
            if not messages:
                break

            for msg in messages:
                self._process_message(msg)
                iterations += 1
                if max_iterations is not None and iterations >= max_iterations:
                    break

    def _process_message(
        self, msg: Message[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]]
    ) -> None:
        """Process a single message from the mailbox."""
        request_event = msg.body
        result = self._execute_request(request_event)

        if msg.expects_reply():
            msg.reply(result)
        else:
            _ = msg.acknowledge()

    def _execute_request(
        self, request_event: MainLoopRequest[UserRequestT]
    ) -> MainLoopResult[OutputT]:
        """Execute a request and return the result."""
        session: Session | None = None
        try:
            response, session = self.execute(
                request_event.request,
                budget=request_event.budget,
                deadline=request_event.deadline,
                resources=request_event.resources,
            )
            return MainLoopResult(
                request_id=request_event.request_id,
                response=response,
                error=None,
                session_id=session.session_id,
            )
        except Exception as exc:
            return MainLoopResult(
                request_id=request_event.request_id,
                response=None,
                error=exc,
                session_id=session.session_id if session else uuid4(),
            )

    def execute(
        self,
        request: UserRequestT,
        *,
        budget: Budget | None = None,
        deadline: Deadline | None = None,
        resources: ResourceRegistry | None = None,
    ) -> tuple[PromptResponse[OutputT], Session]:
        """Execute the main loop for a request.

        Creates a session and prompt, then evaluates with the adapter.
        Visibility expansion exceptions are handled automatically by
        accumulating overrides and retrying. A shared ``BudgetTracker`` is
        used across retries to enforce budget limits cumulatively.

        Args:
            request: The user request to process.
            budget: Optional budget override (takes precedence over config).
            deadline: Optional deadline override (takes precedence over config).
            resources: Optional resources to inject (merged with workspace resources,
                user-provided resources take precedence).

        Returns:
            A tuple of (response, session) from evaluation.

        Raises:
            Any exception from prompt creation, session creation, or evaluation
            (except VisibilityExpansionRequired which is handled internally).
        """
        session = self.create_session()
        prompt = self.create_prompt(request)

        effective_budget = budget if budget is not None else self._config.budget
        effective_deadline = deadline if deadline is not None else self._config.deadline
        effective_resources = (
            resources if resources is not None else self._config.resources
        )

        budget_tracker = (
            BudgetTracker(budget=effective_budget)
            if effective_budget is not None
            else None
        )

        while True:
            try:
                response = self._adapter.evaluate(
                    prompt,
                    session=session,
                    deadline=effective_deadline,
                    budget_tracker=budget_tracker,
                    resources=effective_resources,
                )
            except VisibilityExpansionRequired as e:
                # Update session state with requested visibility overrides
                for path, visibility in e.requested_overrides.items():
                    _ = session.dispatch(
                        SetVisibilityOverride(path=path, visibility=visibility)
                    )
            else:
                return response, session


__all__ = [
    "MainLoop",
    "MainLoopConfig",
    "MainLoopRequest",
    "MainLoopResult",
    "ResourceRegistry",
]
