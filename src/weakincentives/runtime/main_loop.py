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
from ..prompt._visibility import SectionVisibility
from ..prompt.errors import SectionPath, VisibilityExpansionRequired
from .events._types import EventBus
from .session import Session

if TYPE_CHECKING:
    from ..adapters.core import PromptResponse, ProviderAdapter
    from ..prompt import Prompt


@FrozenDataclass()
class MainLoopConfig:
    """Configuration for MainLoop execution defaults.

    Request-level ``budget`` and ``deadline`` override config defaults.
    """

    deadline: Deadline | None = None
    budget: Budget | None = None
    parse_output: bool = True


@FrozenDataclass()
class MainLoopRequest[UserRequestT]:
    """Event requesting MainLoop execution with optional constraints.

    The ``budget`` and ``deadline`` fields override config defaults when set.
    A fresh ``BudgetTracker`` is created per execution.

    Note: ``InProcessEventBus`` dispatches by ``type(event)``, not generic alias.
    ``MainLoopRequest[T]`` is for static type checking; at runtime all events are
    ``MainLoopRequest``. For multiple loop types on one bus, filter by request type
    in the handler or use separate buses.
    """

    request: UserRequestT
    budget: Budget | None = None
    deadline: Deadline | None = None
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@FrozenDataclass()
class MainLoopCompleted[OutputT]:
    """Event published when MainLoop execution succeeds."""

    request_id: UUID
    response: PromptResponse[OutputT]
    session_id: UUID
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@FrozenDataclass()
class MainLoopFailed:
    """Event published when MainLoop execution fails."""

    request_id: UUID
    error: Exception
    session_id: UUID | None
    failed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class MainLoop[UserRequestT, OutputT](ABC):
    """Abstract orchestrator for agent workflow execution.

    MainLoop standardizes agent workflow orchestration: receive request, build
    prompt, evaluate, handle visibility expansion, publish result. Implementations
    define only the domain-specific factories via ``create_prompt`` and
    ``create_session``.

    Execution flow:
        1. Receive ``MainLoopRequest`` via bus or direct ``execute()`` call
        2. Create session via ``create_session()``
        3. Create prompt via ``create_prompt(request)``
        4. Evaluate with adapter
        5. On ``VisibilityExpansionRequired``: accumulate overrides, retry step 4
        6. Publish ``MainLoopCompleted`` or ``MainLoopFailed``

    Usage::

        class CodeReviewLoop(MainLoop[ReviewRequest, ReviewResult]):
            def __init__(
                self, *, adapter: ProviderAdapter[ReviewResult], bus: EventBus
            ) -> None:
                super().__init__(adapter=adapter, bus=bus)
                self._template = PromptTemplate[ReviewResult](
                    ns="reviews",
                    key="code-review",
                    sections=[...],
                )

            def create_prompt(self, request: ReviewRequest) -> Prompt[ReviewResult]:
                return Prompt(self._template).bind(ReviewParams.from_request(request))

            def create_session(self) -> Session:
                return Session(bus=self._bus, tags={"loop": "code-review"})

        # Bus-driven usage
        loop = CodeReviewLoop(adapter=adapter, bus=bus)
        bus.subscribe(MainLoopRequest, loop.handle_request)
        bus.publish(MainLoopRequest(request=ReviewRequest(...)))

        # Direct usage
        response = loop.execute(ReviewRequest(...))
    """

    _adapter: ProviderAdapter[OutputT]
    _bus: EventBus
    _config: MainLoopConfig

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[OutputT],
        bus: EventBus,
        config: MainLoopConfig | None = None,
    ) -> None:
        """Initialize the MainLoop with an adapter, bus, and optional config.

        Args:
            adapter: Provider adapter for prompt evaluation.
            bus: Event bus for request/response event routing.
            config: Optional configuration for default deadline/budget.
        """
        super().__init__()
        self._adapter = adapter
        self._bus = bus
        self._config = config if config is not None else MainLoopConfig()
        bus.subscribe(MainLoopRequest, self.handle_request)

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

    def execute(
        self,
        request: UserRequestT,
        *,
        budget: Budget | None = None,
        deadline: Deadline | None = None,
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

        Returns:
            A tuple of (response, session) from evaluation.

        Raises:
            Any exception from prompt creation, session creation, or evaluation
            (except VisibilityExpansionRequired which is handled internally).
        """
        session = self.create_session()
        prompt = self.create_prompt(request)
        visibility_overrides: dict[SectionPath, SectionVisibility] = {}

        effective_budget = budget if budget is not None else self._config.budget
        effective_deadline = deadline if deadline is not None else self._config.deadline

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
                    visibility_overrides=visibility_overrides,
                    deadline=effective_deadline,
                    budget_tracker=budget_tracker,
                )
            except VisibilityExpansionRequired as e:
                visibility_overrides.update(e.requested_overrides)
            else:
                return response, session

    def handle_request(self, event: object) -> None:
        """Handle a MainLoopRequest event from the bus.

        This method is designed to be subscribed to the event bus::

            bus.subscribe(MainLoopRequest, loop.handle_request)

        On success, publishes ``MainLoopCompleted``. On failure, publishes
        ``MainLoopFailed`` and re-raises the exception.

        Args:
            event: A ``MainLoopRequest`` instance (type is ``object`` for
                compatibility with ``EventHandler`` signature).
        """
        request_event: MainLoopRequest[UserRequestT] = event  # type: ignore[assignment]

        try:
            response, session = self.execute(
                request_event.request,
                budget=request_event.budget,
                deadline=request_event.deadline,
            )

            completed = MainLoopCompleted[OutputT](
                request_id=request_event.request_id,
                response=response,
                session_id=session.session_id,
            )
            _ = self._bus.publish(completed)

        except Exception as exc:
            failed = MainLoopFailed(
                request_id=request_event.request_id,
                error=exc,
                session_id=self.create_session().session_id,
            )
            _ = self._bus.publish(failed)
            raise


__all__ = [
    "MainLoop",
    "MainLoopCompleted",
    "MainLoopConfig",
    "MainLoopFailed",
    "MainLoopRequest",
]
