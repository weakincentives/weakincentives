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

"""Main loop orchestration for agent workflow execution.

MainLoop provides a mailbox-based pattern for durable request processing with
at-least-once delivery semantics. Requests are received from a mailbox queue
and results are sent back via the reply pattern.

Example::

    class CodeReviewLoop(MainLoop[ReviewRequest, ReviewResult]):
        def __init__(
            self,
            *,
            adapter: ProviderAdapter[ReviewResult],
            requests: Mailbox[MainLoopRequest[ReviewRequest], MainLoopResult[ReviewResult]],
        ) -> None:
            super().__init__(adapter=adapter, requests=requests)
            self._template = PromptTemplate[ReviewResult](...)

        def prepare(
            self, request: ReviewRequest
        ) -> tuple[Prompt[ReviewResult], Session]:
            prompt = Prompt(self._template).bind(ReviewParams.from_request(request))
            session = Session(tags={"loop": "code-review"})
            return prompt, session

    # Setup with reply pattern
    responses: Mailbox[MainLoopResult[ReviewResult], None] = InMemoryMailbox(
        name="responses"
    )
    requests: Mailbox[MainLoopRequest[ReviewRequest], MainLoopResult[ReviewResult]] = (
        InMemoryMailbox(
            name="requests",
            reply_resolver=lambda name: responses if name == "responses" else None,
        )
    )

    # Submit request with reply_to
    requests.send(MainLoopRequest(request=my_request), reply_to="responses")

    # Run the worker loop
    loop = CodeReviewLoop(adapter=adapter, requests=requests)
    loop.run(max_iterations=100)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from ..budget import Budget, BudgetTracker
from ..dataclasses import FrozenDataclass
from ..deadlines import Deadline
from ..prompt.errors import VisibilityExpansionRequired
from ..resources import ResourceRegistry
from .mailbox import (
    Mailbox,
    Message,
    ReceiptHandleExpiredError,
    ReplyMailboxUnavailableError,
)
from .session import Session
from .session.visibility_overrides import SetVisibilityOverride

if TYPE_CHECKING:
    from ..adapters.core import PromptResponse, ProviderAdapter
    from ..prompt import Prompt


@dataclass(frozen=True, slots=True)
class MainLoopResult[OutputT]:
    """Response from MainLoop execution.

    Consolidates success and failure into a single type. Check ``success``
    property to determine outcome.
    """

    request_id: UUID
    """Correlates with MainLoopRequest.request_id."""

    output: OutputT | None = None
    """Present on success. The parsed output from the prompt response."""

    error: str | None = None
    """Error message on failure."""

    session_id: UUID | None = None
    """Session that processed the request (if available)."""

    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """Timestamp when processing completed."""

    @property
    def success(self) -> bool:
        """Return True if this result represents successful completion."""
        return self.error is None


@FrozenDataclass()
class MainLoopConfig:
    """Configuration for MainLoop execution defaults.

    Request-level ``budget``, ``deadline``, and ``resources`` override these defaults.
    """

    deadline: Deadline | None = None
    budget: Budget | None = None
    resources: ResourceRegistry | None = None


@FrozenDataclass()
class MainLoopRequest[UserRequestT]:
    """Request for MainLoop execution with optional constraints.

    The ``budget``, ``deadline``, and ``resources`` fields override config defaults.
    """

    request: UserRequestT
    budget: Budget | None = None
    deadline: Deadline | None = None
    resources: ResourceRegistry | None = None
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class MainLoop[UserRequestT, OutputT](ABC):
    """Abstract orchestrator for mailbox-based agent workflow execution.

    MainLoop processes requests from a mailbox queue and sends responses
    via the reply pattern. This pattern supports durable, distributed processing
    with at-least-once delivery semantics.

    Features:
        - Polls requests mailbox for incoming work
        - Sends responses via ``msg.reply_mailbox()``
        - Acknowledges messages after successful processing
        - Visibility timeout prevents duplicate processing
        - Automatic retry with backoff on response send failure

    Execution flow:
        1. Receive message from requests mailbox
        2. Initialize prompt and session via ``prepare(request)``
        3. Evaluate with adapter
        4. On ``VisibilityExpansionRequired``: accumulate overrides, retry step 3
        5. Call ``finalize(prompt, session)`` for post-processing
        6. Send ``MainLoopResult`` via ``msg.reply_mailbox()``
        7. Acknowledge the request message

    Error handling:
        - On success: send result via reply_mailbox, acknowledge request
        - On failure: send error result via reply_mailbox, acknowledge request
        - On no reply_to: just acknowledge (no response expected)
        - On response send failure: nack with backoff (will retry)
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
        """Initialize the MainLoop.

        Args:
            adapter: Provider adapter for prompt evaluation.
            requests: Mailbox to receive MainLoopRequest messages from.
                Must be configured with a reply_resolver if callers set reply_to.
            config: Optional configuration for default deadline/budget.
        """
        super().__init__()
        self._adapter = adapter
        self._requests = requests
        self._config = config if config is not None else MainLoopConfig()

    @abstractmethod
    def prepare(self, request: UserRequestT) -> tuple[Prompt[OutputT], Session]:
        """Prepare prompt and session for the given request.

        Subclasses must implement this method to construct the prompt
        and session appropriate for their domain.

        Args:
            request: The user request to process.

        Returns:
            A tuple of (prompt, session) ready for evaluation.
        """
        ...

    def finalize(self, prompt: Prompt[OutputT], session: Session) -> None:
        """Finalize after execution completes.

        Called after successful evaluation. Override to perform cleanup,
        logging, or post-processing tasks.

        Args:
            prompt: The prompt that was evaluated.
            session: The session used for evaluation.
        """
        _ = (self, prompt, session)

    def execute(
        self,
        request: UserRequestT,
        *,
        budget: Budget | None = None,
        deadline: Deadline | None = None,
        resources: ResourceRegistry | None = None,
    ) -> tuple[PromptResponse[OutputT], Session]:
        """Execute directly without mailbox routing.

        Convenience method for synchronous execution. For durable processing
        with at-least-once semantics, use ``run()`` with mailboxes instead.

        Args:
            request: The user request to process.
            budget: Optional budget override (takes precedence over config).
            deadline: Optional deadline override (takes precedence over config).
            resources: Optional resource registry override.

        Returns:
            Tuple of (PromptResponse, Session) from the evaluation.
        """
        request_event = MainLoopRequest(
            request=request,
            budget=budget,
            deadline=deadline,
            resources=resources,
        )
        return self._execute(request_event)

    def _execute(
        self,
        request_event: MainLoopRequest[UserRequestT],
    ) -> tuple[PromptResponse[OutputT], Session]:
        """Execute the main loop for a request event.

        Handles core execution logic including visibility expansion retries.
        """
        prompt, session = self.prepare(request_event.request)

        effective_budget = (
            request_event.budget
            if request_event.budget is not None
            else self._config.budget
        )
        effective_deadline = (
            request_event.deadline
            if request_event.deadline is not None
            else self._config.deadline
        )
        effective_resources = (
            request_event.resources
            if request_event.resources is not None
            else self._config.resources
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
                for path, visibility in e.requested_overrides.items():
                    _ = session.dispatch(
                        SetVisibilityOverride(path=path, visibility=visibility)
                    )
            else:
                self.finalize(prompt, session)
                return response, session

    def _handle_message(
        self, msg: Message[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]]
    ) -> None:
        """Process a single message from the requests mailbox."""
        request_event = msg.body

        try:
            response, session = self._execute(request_event)

            result = MainLoopResult[OutputT](
                request_id=request_event.request_id,
                output=response.output,
                session_id=session.session_id,
            )

            self._send_and_ack(msg, result)

        except Exception as exc:
            result = MainLoopResult[OutputT](
                request_id=request_event.request_id,
                error=str(exc),
            )

            self._send_and_ack(msg, result)

    @staticmethod
    def _send_and_ack(
        msg: Message[MainLoopRequest[Any], MainLoopResult[Any]],
        result: MainLoopResult[Any],
    ) -> None:
        """Send result via reply_mailbox and acknowledge, handling errors gracefully."""
        try:
            _ = msg.reply_mailbox().send(result)
            msg.acknowledge()
        except ReplyMailboxUnavailableError:
            # No reply_to specified - just acknowledge without sending response.
            # This is valid for fire-and-forget requests.
            msg.acknowledge()
        except ReceiptHandleExpiredError:
            # Handle expired during processing - message already requeued by reaper.
            # This is expected for long-running requests. The duplicate response
            # will be sent when the message is reprocessed.
            pass
        except Exception:
            # Response send failed - nack so message is retried
            try:
                backoff = min(60 * msg.delivery_count, 900)
                msg.nack(visibility_timeout=backoff)
            except ReceiptHandleExpiredError:
                # Handle expired - message already requeued, nothing to do
                pass

    def run(
        self,
        *,
        max_iterations: int | None = None,
        visibility_timeout: int = 300,
        wait_time_seconds: int = 20,
    ) -> None:
        """Run the worker loop, processing messages from the requests mailbox.

        Polls the requests mailbox for messages and processes each one.
        Messages are acknowledged after successful processing or after
        sending an error response.

        The loop exits when:
        - max_iterations is reached
        - The requests mailbox is closed

        Args:
            max_iterations: Maximum polling iterations. None for unlimited.
            visibility_timeout: Seconds messages remain invisible during processing.
                Should exceed maximum expected execution time.
            wait_time_seconds: Long poll duration for receiving messages.
        """
        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            # Exit if mailbox closed
            if self._requests.closed:
                break

            messages = self._requests.receive(
                visibility_timeout=visibility_timeout,
                wait_time_seconds=wait_time_seconds,
            )

            for msg in messages:
                self._handle_message(msg)

            iterations += 1


__all__ = [
    "MainLoop",
    "MainLoopConfig",
    "MainLoopRequest",
    "MainLoopResult",
    "ResourceRegistry",
]
