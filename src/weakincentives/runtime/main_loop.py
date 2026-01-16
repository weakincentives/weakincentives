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
and results are sent via Message.reply().

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

    # Run the worker loop
    loop = CodeReviewLoop(adapter=adapter, requests=requests)
    loop.run(max_iterations=100)
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Self
from uuid import UUID, uuid4

from ..budget import Budget, BudgetTracker
from ..dataclasses import FrozenDataclass
from ..deadlines import Deadline
from ..prompt.errors import VisibilityExpansionRequired
from .dlq import DeadLetter, DLQPolicy
from .lease_extender import LeaseExtender, LeaseExtenderConfig
from .lifecycle import wait_until
from .mailbox import Mailbox, Message, ReceiptHandleExpiredError, ReplyNotAvailableError
from .run_context import RunContext
from .session import Session
from .session.visibility_overrides import SetVisibilityOverride
from .watchdog import Heartbeat

if TYPE_CHECKING:
    from ..adapters.core import PromptResponse, ProviderAdapter
    from ..evals._experiment import Experiment
    from ..prompt import Prompt

_logger = logging.getLogger(__name__)


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

    run_context: RunContext | None = None
    """Execution context with correlation identifiers and metadata."""

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

    The ``lease_extender`` field controls automatic message visibility extension
    during processing. When enabled, heartbeats from tool execution extend the
    message lease, preventing timeout during long-running requests.
    """

    deadline: Deadline | None = None
    budget: Budget | None = None
    resources: Mapping[type[object], object] | None = None
    lease_extender: LeaseExtenderConfig | None = None


@FrozenDataclass()
class MainLoopRequest[UserRequestT]:
    """Request for MainLoop execution with optional constraints.

    The ``budget``, ``deadline``, and ``resources`` fields override config defaults.
    The ``experiment`` field specifies a configuration variant for A/B testing.
    """

    request: UserRequestT
    budget: Budget | None = None
    deadline: Deadline | None = None
    resources: Mapping[type[object], object] | None = None
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    run_context: RunContext | None = None
    """Optional execution context. If not provided, MainLoop creates one."""
    experiment: Experiment | None = None
    """Optional experiment for A/B testing. When provided, prepare() receives it."""


class MainLoop[UserRequestT, OutputT](ABC):
    """Abstract orchestrator for mailbox-based agent workflow execution.

    MainLoop processes requests from a mailbox queue and sends responses
    via Message.reply(). This pattern supports durable, distributed processing
    with at-least-once delivery semantics.

    Features:
        - Polls requests mailbox for incoming work
        - Uses Message.reply() for response routing (derives from reply_to)
        - Acknowledges messages after successful processing
        - Visibility timeout prevents duplicate processing
        - Automatic retry with backoff on response send failure
        - Graceful shutdown with in-flight message completion

    Execution flow:
        1. Receive message from requests mailbox
        2. Initialize prompt and session via ``prepare(request)``
        3. Evaluate with adapter
        4. On ``VisibilityExpansionRequired``: accumulate overrides, retry step 3
        5. Call ``finalize(prompt, session)`` for post-processing
        6. Send ``MainLoopResult`` via msg.reply()
        7. Acknowledge the request message

    Error handling:
        - On success: reply with result, acknowledge request
        - On failure: reply with error result, acknowledge request
        - On reply send failure: nack with backoff (will retry)

    Lifecycle:
        - Use ``run()`` to start processing messages
        - Call ``shutdown()`` to request graceful stop
        - In-flight messages complete before exit
        - Supports context manager protocol for automatic cleanup
    """

    _adapter: ProviderAdapter[OutputT]
    _requests: Mailbox[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]]
    _config: MainLoopConfig
    _dlq: DLQPolicy[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]] | None
    _shutdown_event: threading.Event
    _running: bool
    _lock: threading.Lock
    _heartbeat: Heartbeat
    _lease_extender: LeaseExtender
    _worker_id: str

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[OutputT],
        requests: Mailbox[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]],
        config: MainLoopConfig | None = None,
        worker_id: str | None = None,
        dlq: DLQPolicy[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]]
        | None = None,
    ) -> None:
        """Initialize the MainLoop.

        Args:
            adapter: Provider adapter for prompt evaluation.
            requests: Mailbox to receive MainLoopRequest messages from.
                Response routing derives from each message's reply_to field.
            config: Optional configuration for default deadline/budget.
            worker_id: Identifier for this worker instance. If None or empty,
                auto-generates as "{hostname}-{pid}". Recommended formats:
                - Production: "{hostname}-{pid}" or "{k8s_pod_name}"
                - Testing: "test-worker" or descriptive test name
            dlq: Optional dead letter queue policy. When configured, messages
                that fail repeatedly are sent to the DLQ mailbox instead of
                retrying indefinitely.
        """
        super().__init__()
        self._adapter = adapter
        self._requests = requests
        self._config = config if config is not None else MainLoopConfig()
        self._dlq = dlq
        self._shutdown_event = threading.Event()
        self._running = False
        self._lock = threading.Lock()
        self._heartbeat = Heartbeat()
        # Initialize lease extender with config or defaults
        lease_config = (
            self._config.lease_extender
            if self._config.lease_extender is not None
            else LeaseExtenderConfig()
        )
        self._lease_extender = LeaseExtender(config=lease_config)
        # Auto-generate worker_id if not provided
        if worker_id:
            self._worker_id = worker_id
        else:
            hostname = socket.gethostname()
            self._worker_id = f"{hostname}-{os.getpid()}"

    @property
    def heartbeat(self) -> Heartbeat:
        """Heartbeat tracker for watchdog monitoring.

        The loop beats after receiving messages and after processing each
        message, enabling the watchdog to detect stuck workers.
        """
        return self._heartbeat

    @property
    def worker_id(self) -> str:
        """Identifier for this worker instance."""
        return self._worker_id

    @abstractmethod
    def prepare(
        self,
        request: UserRequestT,
        *,
        experiment: Experiment | None = None,
    ) -> tuple[Prompt[OutputT], Session]:
        """Prepare prompt and session for the given request.

        Subclasses must implement this method to construct the prompt
        and session appropriate for their domain.

        Args:
            request: The user request to process.
            experiment: Optional experiment configuration. When provided,
                implementations should:
                1. Use experiment.overrides_tag for prompt construction
                2. Pass experiment to session for tracking
                3. Check experiment.flags for behavior changes

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

    def execute(  # noqa: PLR0913
        self,
        request: UserRequestT,
        *,
        budget: Budget | None = None,
        deadline: Deadline | None = None,
        resources: Mapping[type[object], object] | None = None,
        heartbeat: Heartbeat | None = None,
        experiment: Experiment | None = None,
    ) -> tuple[PromptResponse[OutputT], Session]:
        """Execute directly without mailbox routing.

        Convenience method for synchronous execution. For durable processing
        with at-least-once semantics, use ``run()`` with mailboxes instead.

        Args:
            request: The user request to process.
            budget: Optional budget override (takes precedence over config).
            deadline: Optional deadline override (takes precedence over config).
            resources: Optional resources override (dict mapping types to instances).
            heartbeat: Optional heartbeat for lease extension. If provided, this
                heartbeat is used for adapter evaluation instead of the loop's
                internal heartbeat. Use this when EvalLoop needs to extend its
                own message lease based on MainLoop work.
            experiment: Optional experiment for A/B testing. Passed to prepare().

        Returns:
            Tuple of (PromptResponse, Session) from the evaluation.
        """
        request_event = MainLoopRequest(
            request=request,
            budget=budget,
            deadline=deadline,
            resources=resources,
            experiment=experiment,
        )
        return self._execute(request_event, heartbeat=heartbeat)

    def _execute(
        self,
        request_event: MainLoopRequest[UserRequestT],
        *,
        heartbeat: Heartbeat | None = None,
        run_context: RunContext | None = None,
    ) -> tuple[PromptResponse[OutputT], Session]:
        """Execute the main loop for a request event.

        Handles core execution logic including visibility expansion retries.

        Args:
            request_event: The request to process.
            heartbeat: Optional heartbeat override. If provided, uses this
                instead of self._heartbeat for adapter evaluation.
            run_context: Optional execution context for distributed tracing.
        """
        prompt, session = self.prepare(
            request_event.request,
            experiment=request_event.experiment,
        )

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

        # Bind resources to prompt if provided
        if effective_resources is not None:
            prompt = prompt.bind(resources=effective_resources)

        budget_tracker = (
            BudgetTracker(budget=effective_budget)
            if effective_budget is not None
            else None
        )

        # Use provided heartbeat or fall back to loop's internal heartbeat
        effective_heartbeat = heartbeat if heartbeat is not None else self._heartbeat

        while True:
            try:
                response = self._adapter.evaluate(
                    prompt,
                    session=session,
                    deadline=effective_deadline,
                    budget_tracker=budget_tracker,
                    heartbeat=effective_heartbeat,
                    run_context=run_context,
                )
            except VisibilityExpansionRequired as e:
                for path, visibility in e.requested_overrides.items():
                    _ = session.dispatch(
                        SetVisibilityOverride(path=path, visibility=visibility)
                    )
            else:
                self.finalize(prompt, session)
                return response, session

    def _build_run_context(
        self,
        request_event: MainLoopRequest[UserRequestT],
        delivery_count: int,
        session_id: UUID | None = None,
    ) -> RunContext:
        """Build RunContext for this execution.

        The run_id is always fresh for each execution attempt.

        Request ID always comes from request_event.request_id to ensure
        correlation with MainLoopResult.request_id. The run_context parameter
        is only used for distributed trace context (trace_id, span_id).

        Args:
            request_event: The incoming request with optional run_context.
            delivery_count: Message delivery count (attempt number).
            session_id: Session ID to embed (typically set via replace() later).

        Returns:
            Fresh RunContext with new run_id and preserved trace context.
        """
        trace_id: str | None = None
        span_id: str | None = None
        if request_event.run_context is not None:
            trace_id = request_event.run_context.trace_id
            span_id = request_event.run_context.span_id

        return RunContext(
            run_id=uuid4(),
            request_id=request_event.request_id,
            session_id=session_id,
            attempt=delivery_count,
            worker_id=self._worker_id,
            trace_id=trace_id,
            span_id=span_id,
        )

    def _handle_message(
        self, msg: Message[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]]
    ) -> None:
        """Process a single message from the requests mailbox."""
        request_event = msg.body

        # Build RunContext ONCE before execution. Session_id will be added
        # via replace() after prepare() creates the session.
        run_context = self._build_run_context(
            request_event, msg.delivery_count, session_id=None
        )

        # Attach lease extender to heartbeat for this message
        with self._lease_extender.attach(msg, self._heartbeat):
            try:
                response, session = self._execute(
                    request_event, run_context=run_context
                )

                # Add session_id while preserving the same run_id
                run_context = replace(run_context, session_id=session.session_id)

                result = MainLoopResult[OutputT](
                    request_id=request_event.request_id,
                    output=response.output,
                    session_id=session.session_id,
                    run_context=run_context,
                )

            except Exception as exc:
                self._handle_failure(msg, exc, run_context=run_context)
                return

        self._reply_and_ack(msg, result)

    def _handle_failure(
        self,
        msg: Message[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]],
        error: Exception,
        *,
        run_context: RunContext,
    ) -> None:
        """Handle message processing failure.

        When DLQ is configured:
        - Checks DLQ policy and either dead-letters or retries with backoff
        - Only sends error replies on terminal outcomes (DLQ)

        When DLQ is not configured:
        - Sends error reply and acknowledges (original behavior)
        """
        if self._dlq is None:
            # No DLQ configured - use original behavior: error reply + acknowledge
            result = MainLoopResult[OutputT](
                request_id=msg.body.request_id,
                error=str(error),
                run_context=run_context,
            )
            self._reply_and_ack(msg, result)
            return

        # DLQ is configured - check if we should dead-letter
        if self._dlq.should_dead_letter(msg, error):
            self._dead_letter(msg, error, run_context=run_context)
            return

        # Retry with backoff - do NOT send error reply here.
        # The message will be redelivered and may succeed on retry.
        # Only send error replies on terminal outcomes (DLQ or final failure).
        backoff = min(60 * msg.delivery_count, 900)
        with contextlib.suppress(ReceiptHandleExpiredError):
            msg.nack(visibility_timeout=backoff)

    def _dead_letter(
        self,
        msg: Message[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]],
        error: Exception,
        *,
        run_context: RunContext,
    ) -> None:
        """Send message to dead letter queue."""
        if self._dlq is None:  # pragma: no cover - defensive check
            return

        dead_letter = DeadLetter(
            message_id=msg.id,
            body=msg.body,
            source_mailbox=self._requests.name,
            delivery_count=msg.delivery_count,
            last_error=str(error),
            last_error_type=f"{type(error).__module__}.{type(error).__qualname__}",
            dead_lettered_at=datetime.now(UTC),
            first_received_at=msg.enqueued_at,
            request_id=msg.body.request_id,
            reply_to=msg.reply_to.name if msg.reply_to else None,
            trace_id=run_context.trace_id,
        )

        # Send error reply - this is a terminal outcome. Failures here
        # should not prevent the dead letter from being sent.
        try:
            if msg.reply_to:
                _ = msg.reply(
                    MainLoopResult(
                        request_id=msg.body.request_id,
                        error=f"Dead-lettered after {msg.delivery_count} attempts: {error}",
                        run_context=run_context,
                    )
                )
        except Exception as reply_error:  # nosec B110 - intentional: reply failure should not block dead-lettering
            _logger.debug(
                "Failed to send error reply during dead-lettering",
                extra={"message_id": msg.id, "error": str(reply_error)},
            )

        _ = self._dlq.mailbox.send(dead_letter)
        _logger.warning(
            "Message dead-lettered",
            extra={
                "message_id": msg.id,
                "request_id": str(msg.body.request_id),
                "delivery_count": msg.delivery_count,
                "error_type": dead_letter.last_error_type,
            },
        )

        # Acknowledge to remove from source queue
        with contextlib.suppress(ReceiptHandleExpiredError):
            msg.acknowledge()

    def _reply_and_ack(
        self,
        msg: Message[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]],
        result: MainLoopResult[OutputT],
    ) -> None:
        """Reply with result and acknowledge message, handling expired handles gracefully."""
        _ = self  # Uses self implicitly via Message callbacks
        try:
            _ = msg.reply(result)
            msg.acknowledge()
        except ReplyNotAvailableError:
            # No reply_to specified - log and acknowledge without reply
            _logger.warning(
                "No reply_to for message %s, acknowledging without reply", msg.id
            )
            with contextlib.suppress(ReceiptHandleExpiredError):
                msg.acknowledge()
        except ReceiptHandleExpiredError:
            # Handle expired during processing - message already requeued by reaper.
            # This is expected for long-running requests. The duplicate response
            # will be sent when the message is reprocessed.
            pass
        except Exception:
            # Reply send failed - nack so message is retried
            with contextlib.suppress(ReceiptHandleExpiredError):
                backoff = min(60 * msg.delivery_count, 900)
                msg.nack(visibility_timeout=backoff)

    def run(
        self,
        *,
        max_turns: int | None = None,
        visibility_timeout: int = 300,
        wait_time_seconds: int = 20,
    ) -> None:
        """Run the worker loop, processing messages from the requests mailbox.

        Polls the requests mailbox for messages and processes each one.
        Messages are acknowledged after successful processing or after
        sending an error response.

        The loop exits when:
        - max_turns is reached
        - shutdown() is called
        - The requests mailbox is closed

        In-flight messages complete before exit. Unprocessed messages from
        the current batch are nacked for redelivery.

        Args:
            max_turns: Maximum number of turns to execute (None = unlimited).
                A turn is one iteration through the loop's main processing cycle.
            visibility_timeout: Seconds messages remain invisible during processing.
                Should exceed maximum expected execution time.
            wait_time_seconds: Long poll duration for receiving messages.
        """
        with self._lock:
            self._running = True
            self._shutdown_event.clear()

        turns = 0
        try:
            while max_turns is None or turns < max_turns:
                # Check shutdown before blocking on receive
                if self._shutdown_event.is_set():
                    break

                # Exit if mailbox closed
                if self._requests.closed:
                    break

                messages = self._requests.receive(
                    visibility_timeout=visibility_timeout,
                    wait_time_seconds=wait_time_seconds,
                )

                # Beat after receive (proves we're not stuck waiting)
                self._heartbeat.beat()

                for msg in messages:
                    # Check shutdown between messages
                    if self._shutdown_event.is_set():
                        # Nack unprocessed message for redelivery
                        with contextlib.suppress(ReceiptHandleExpiredError):
                            msg.nack(visibility_timeout=0)
                        break

                    self._handle_message(msg)

                    # Beat after each message (proves processing completes)
                    self._heartbeat.beat()

                turns += 1
        finally:
            with self._lock:
                self._running = False

    def shutdown(self, *, timeout: float = 30.0) -> bool:
        """Request graceful shutdown and wait for completion.

        Sets the shutdown flag. If the loop is running, waits up to timeout
        seconds for it to stop.

        Args:
            timeout: Maximum seconds to wait for the loop to stop.

        Returns:
            True if loop stopped cleanly, False if timeout expired.
        """
        self._shutdown_event.set()
        return wait_until(lambda: not self.running, timeout=timeout)

    @property
    def running(self) -> bool:
        """True if the loop is currently processing messages."""
        with self._lock:
            return self._running

    def __enter__(self) -> Self:
        """Context manager entry. Returns self for use in with statement."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit. Triggers shutdown and waits for completion."""
        _ = (exc_type, exc_val, exc_tb)
        _ = self.shutdown()


__all__ = [
    "MainLoop",
    "MainLoopConfig",
    "MainLoopRequest",
    "MainLoopResult",
]
