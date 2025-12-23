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
from .events._types import ControlDispatcher
from .session import Session
from .session.visibility_overrides import SetVisibilityOverride

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


@FrozenDataclass()
class MainLoopRequest[UserRequestT]:
    """Event requesting MainLoop execution with optional constraints.

    The ``budget``, ``deadline``, and ``resources`` fields override config defaults
    when set. A fresh ``BudgetTracker`` is created per execution.

    Note: ``InProcessDispatcher`` dispatches by ``type(event)``, not generic alias.
    ``MainLoopRequest[T]`` is for static type checking; at runtime all events are
    ``MainLoopRequest``. For multiple loop types on one bus, filter by request type
    in the handler or use separate buses.
    """

    request: UserRequestT
    budget: Budget | None = None
    deadline: Deadline | None = None
    resources: ResourceRegistry | None = None
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@FrozenDataclass()
class MainLoopCompleted[OutputT]:
    """Event dispatched when MainLoop execution succeeds."""

    request_id: UUID
    response: PromptResponse[OutputT]
    session_id: UUID
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@FrozenDataclass()
class MainLoopFailed:
    """Event dispatched when MainLoop execution fails."""

    request_id: UUID
    error: Exception
    session_id: UUID | None
    failed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class MainLoop[UserRequestT, OutputT](ABC):
    """Abstract orchestrator for agent workflow execution.

    MainLoop standardizes agent workflow orchestration: receive request, build
    prompt, evaluate, handle visibility expansion, dispatch result. Implementations
    define only the domain-specific factory via ``initialize``.

    Execution flow:
        1. Receive ``MainLoopRequest`` via bus or direct ``execute()`` call
        2. Initialize prompt and session via ``initialize(request)``
        3. Evaluate with adapter
        4. On ``VisibilityExpansionRequired``: accumulate overrides, retry step 3
        5. Call ``finalize(prompt, session)`` for post-processing
        6. Publish ``MainLoopCompleted`` or ``MainLoopFailed``

    Usage::

        class CodeReviewLoop(MainLoop[ReviewRequest, ReviewResult]):
            def __init__(
                self, *, adapter: ProviderAdapter[ReviewResult], bus: ControlDispatcher
            ) -> None:
                super().__init__(adapter=adapter, bus=bus)
                self._template = PromptTemplate[ReviewResult](
                    ns="reviews",
                    key="code-review",
                    sections=[...],
                )

            def initialize(
                self, request: ReviewRequest
            ) -> tuple[Prompt[ReviewResult], Session]:
                prompt = Prompt(self._template).bind(ReviewParams.from_request(request))
                session = Session(bus=self._bus, tags={"loop": "code-review"})
                return prompt, session

            def finalize(
                self, prompt: Prompt[ReviewResult], session: Session
            ) -> None:
                # Optional: cleanup or logging
                pass

        # Dispatcher-driven usage (subscription is automatic in __init__)
        loop = CodeReviewLoop(adapter=adapter, bus=bus)
        bus.dispatch(MainLoopRequest(request=ReviewRequest(...)))

        # Direct usage
        response, session = loop.execute(ReviewRequest(...))
    """

    _adapter: ProviderAdapter[OutputT]
    _bus: ControlDispatcher
    _config: MainLoopConfig

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[OutputT],
        bus: ControlDispatcher,
        config: MainLoopConfig | None = None,
    ) -> None:
        """Initialize the MainLoop with an adapter, dispatcher, and optional config.

        Args:
            adapter: Provider adapter for prompt evaluation.
            bus: Control dispatcher for request/response event routing.
            config: Optional configuration for default deadline/budget.
        """
        super().__init__()
        self._adapter = adapter
        self._bus = bus
        self._config = config if config is not None else MainLoopConfig()
        bus.subscribe(MainLoopRequest, self.handle_request)

    @abstractmethod
    def initialize(self, request: UserRequestT) -> tuple[Prompt[OutputT], Session]:
        """Initialize prompt and session for the given request.

        Subclasses must implement this method to construct the prompt
        and session appropriate for their domain. This consolidates
        the setup phase into a single method.

        Args:
            request: The user request to process.

        Returns:
            A tuple of (prompt, session) ready for evaluation.
        """
        ...

    def finalize(self, prompt: Prompt[OutputT], session: Session) -> None:
        """Finalize after execution completes.

        Called after successful evaluation with the prompt and session used.
        Subclasses can override this method to perform cleanup, logging,
        or post-processing tasks.

        Args:
            prompt: The prompt that was evaluated.
            session: The session used for evaluation.
        """
        _ = (self, prompt, session)  # Default implementation does nothing

    def execute(
        self,
        request: UserRequestT,
        *,
        budget: Budget | None = None,
        deadline: Deadline | None = None,
        resources: ResourceRegistry | None = None,
    ) -> tuple[PromptResponse[OutputT], Session]:
        """Execute the main loop for a request.

        Initializes prompt and session via ``initialize()``, then evaluates
        with the adapter. Visibility expansion exceptions are handled
        automatically by accumulating overrides and retrying. A shared
        ``BudgetTracker`` is used across retries to enforce budget limits
        cumulatively. After successful evaluation, ``finalize()`` is called.

        Args:
            request: The user request to process.
            budget: Optional budget override (takes precedence over config).
            deadline: Optional deadline override (takes precedence over config).
            resources: Optional resources to inject (merged with workspace resources,
                user-provided resources take precedence).

        Returns:
            A tuple of (response, session) from evaluation.

        Raises:
            Any exception from initialization or evaluation
            (except VisibilityExpansionRequired which is handled internally).
        """
        prompt, session = self.initialize(request)

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
                self.finalize(prompt, session)
                return response, session

    def handle_request(self, event: object) -> None:
        """Handle a MainLoopRequest event from the bus.

        This method is designed to be subscribed to the event bus::

            bus.subscribe(MainLoopRequest, loop.handle_request)

        On success, dispatches ``MainLoopCompleted``. On failure, dispatches
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
                resources=request_event.resources,
            )

            completed = MainLoopCompleted[OutputT](
                request_id=request_event.request_id,
                response=response,
                session_id=session.session_id,
            )
            _ = self._bus.dispatch(completed)

        except Exception as exc:
            failed = MainLoopFailed(
                request_id=request_event.request_id,
                error=exc,
                session_id=None,
            )
            _ = self._bus.dispatch(failed)
            raise


__all__ = [
    "MainLoop",
    "MainLoopCompleted",
    "MainLoopConfig",
    "MainLoopFailed",
    "MainLoopRequest",
    "ResourceRegistry",
]
