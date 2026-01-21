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
import os
import socket
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Self
from uuid import UUID, uuid4

from ..budget import Budget, BudgetTracker
from ..deadlines import Deadline
from ..prompt.errors import VisibilityExpansionRequired
from .dlq import DLQPolicy
from .lease_extender import LeaseExtender, LeaseExtenderConfig
from .lifecycle import wait_until
from .logging import StructuredLogger, bind_run_context, get_logger
from .mailbox import Mailbox, Message, ReceiptHandleExpiredError
from .main_loop_types import MainLoopConfig, MainLoopRequest, MainLoopResult
from .message_handlers import handle_failure, reply_and_ack
from .run_context import RunContext
from .session import Session
from .session.visibility_overrides import SetVisibilityOverride
from .watchdog import Heartbeat

if TYPE_CHECKING:
    from ..adapters.core import PromptResponse, ProviderAdapter
    from ..debug.bundle import BundleConfig
    from ..evals._experiment import Experiment
    from ..prompt import Prompt
    from .session.visibility_overrides import VisibilityOverrides

_logger: StructuredLogger = get_logger(
    __name__, context={"component": "runtime.main_loop"}
)


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
        response, session, _ = self._execute(request_event, heartbeat=heartbeat)
        return response, session

    def _execute(
        self,
        request_event: MainLoopRequest[UserRequestT],
        *,
        heartbeat: Heartbeat | None = None,
        run_context: RunContext | None = None,
    ) -> tuple[PromptResponse[OutputT], Session, Prompt[OutputT]]:
        """Execute the main loop for a request event.

        Handles core execution logic including visibility expansion retries.

        Args:
            request_event: The request to process.
            heartbeat: Optional heartbeat override. If provided, uses this
                instead of self._heartbeat for adapter evaluation.
            run_context: Optional execution context for distributed tracing.

        Returns:
            Tuple of (response, session, prompt) for access to execution artifacts.
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
                return response, session, prompt

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

        # Create run-scoped logger for tracing this request
        log = bind_run_context(_logger, run_context)
        log.debug(
            "Processing request from mailbox.",
            event="main_loop.message_received",
            context={
                "message_id": msg.id,
                "delivery_count": msg.delivery_count,
            },
        )

        # Check if debug bundling is enabled (per-request overrides config-level)
        bundle_config = (
            request_event.debug_bundle
            if request_event.debug_bundle is not None
            else self._config.debug_bundle
        )

        # Attach lease extender to heartbeat for this message
        with self._lease_extender.attach(msg, self._heartbeat):
            if bundle_config is not None and bundle_config.enabled:
                self._handle_message_with_bundle(
                    msg, request_event, run_context, log, bundle_config
                )
            else:
                self._handle_message_without_bundle(msg, request_event, run_context)

        # Result is created and replied in the helper methods

    def _handle_message_with_bundle(
        self,
        msg: Message[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]],
        request_event: MainLoopRequest[UserRequestT],
        run_context: RunContext,
        log: StructuredLogger,
        bundle_config: BundleConfig,
    ) -> None:
        """Process message with debug bundling enabled."""
        from ..debug.bundle import BundleWriter
        from ..filesystem import Filesystem
        from .session.visibility_overrides import VisibilityOverrides

        if bundle_config.target is None:  # pragma: no cover
            # Defensive guard: _handle_message only calls this when enabled=True
            self._handle_message_without_bundle(msg, request_event, run_context)
            return

        # Determine trigger based on where config came from
        trigger = "request" if request_event.debug_bundle is not None else "config"

        writer: BundleWriter | None = None
        started_at = datetime.now(UTC)
        budget_tracker: BudgetTracker | None = None
        try:
            with BundleWriter(
                bundle_config.target,
                bundle_id=run_context.run_id,
                config=bundle_config,
                trigger=trigger,
            ) as writer:
                # Write request input
                writer.write_request_input(request_event)

                # Prepare prompt and session first so we can capture session/before
                prompt, session = self.prepare(
                    request_event.request,
                    experiment=request_event.experiment,
                )

                # Write session before execution
                writer.write_session_before(session)

                # Update run_context with session_id for early write
                run_context = replace(run_context, session_id=session.session_id)
                writer.write_run_context(run_context)

                # Set prompt info for manifest
                adapter_name = self._get_adapter_name()
                writer.set_prompt_info(
                    ns=prompt.ns,
                    key=prompt.key,
                    adapter=adapter_name,
                )

                # Resolve effective settings and execute
                response, budget_tracker = self._execute_with_bundled_settings(
                    request_event=request_event,
                    prompt=prompt,
                    session=session,
                    run_context=run_context,
                    writer=writer,
                )

                ended_at = datetime.now(UTC)

                # Write session after execution
                writer.write_session_after(session)
                writer.write_request_output(response)
                writer.write_config(self._config)

                # Write final run_context (session_id already set above)
                writer.write_run_context(run_context)

                # Write metrics: timing, token usage, budget status
                metrics = self._collect_metrics(
                    started_at=started_at,
                    ended_at=ended_at,
                    session=session,
                    budget_tracker=budget_tracker,
                )
                writer.write_metrics(metrics)

                # Write prompt overrides from session
                visibility_overrides = session[VisibilityOverrides].latest()
                if visibility_overrides is not None and visibility_overrides.overrides:
                    prompt_overrides = self._format_visibility_overrides(
                        visibility_overrides, session
                    )
                    writer.write_prompt_overrides(prompt_overrides)

                # Write filesystem snapshot if available from prompt resources
                fs = prompt.resources.get_optional(Filesystem)
                if fs is not None:
                    writer.write_filesystem(fs)

                # Write environment capture
                writer.write_environment()

            # Bundle path is set after context manager exits (in __exit__ -> _finalize)
            bundle_path = writer.path

            result = MainLoopResult[OutputT](
                request_id=request_event.request_id,
                output=response.output,
                session_id=session.session_id,
                run_context=run_context,
                bundle_path=bundle_path,
            )
            reply_and_ack(msg, result)

        except Exception as exc:
            # Check if bundle was created despite the error
            bundle_path = writer.path if writer is not None else None

            if bundle_path is not None:
                # Bundle was created successfully, error was during execution
                log.info(
                    "Execution failed but debug bundle was created",
                    event="main_loop.execution_failed_with_bundle",
                    context={"error": str(exc), "bundle_path": str(bundle_path)},
                )
            else:
                # True bundle creation failure
                log.warning(
                    "Debug bundle creation failed, falling back to unbundled execution",
                    event="main_loop.bundle_failed",
                    context={"error": str(exc)},
                )

            handle_failure(
                msg,
                exc,
                run_context=run_context,
                dlq=self._dlq,
                requests_mailbox=self._requests,
                result_class=MainLoopResult,
                bundle_path=bundle_path,
            )

    def _execute_with_bundled_settings(
        self,
        *,
        request_event: MainLoopRequest[UserRequestT],
        prompt: Prompt[OutputT],
        session: Session,
        run_context: RunContext,
        writer: object,  # BundleWriter, but avoid import for typing
    ) -> tuple[PromptResponse[OutputT], BudgetTracker | None]:
        """Execute prompt with settings resolved and log capture enabled.

        This helper reduces nesting in _handle_message_with_bundle by
        encapsulating the execution loop with visibility override retry.
        """
        # Resolve effective settings
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

        # Use capture_logs from writer
        with writer.capture_logs():  # type: ignore[union-attr]
            while True:
                try:
                    response = self._adapter.evaluate(
                        prompt,
                        session=session,
                        deadline=effective_deadline,
                        budget_tracker=budget_tracker,
                        heartbeat=self._heartbeat,
                        run_context=run_context,
                    )
                except VisibilityExpansionRequired as e:
                    for path, visibility in e.requested_overrides.items():
                        _ = session.dispatch(
                            SetVisibilityOverride(path=path, visibility=visibility)
                        )
                else:
                    self.finalize(prompt, session)
                    break

        return response, budget_tracker

    def _get_adapter_name(self) -> str:
        """Get the canonical adapter name for the current adapter."""
        from ..adapters import (
            CLAUDE_AGENT_SDK_ADAPTER_NAME,
            LITELLM_ADAPTER_NAME,
            OPENAI_ADAPTER_NAME,
        )
        from ..adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
        from ..adapters.litellm import LiteLLMAdapter
        from ..adapters.openai import OpenAIAdapter

        if isinstance(self._adapter, ClaudeAgentSDKAdapter):
            return CLAUDE_AGENT_SDK_ADAPTER_NAME  # pragma: no cover
        if isinstance(self._adapter, OpenAIAdapter):
            return OPENAI_ADAPTER_NAME  # pragma: no cover
        if isinstance(self._adapter, LiteLLMAdapter):
            return LITELLM_ADAPTER_NAME  # pragma: no cover
        return type(self._adapter).__name__

    @staticmethod
    def _collect_metrics(
        *,
        started_at: datetime,
        ended_at: datetime,
        session: Session,
        budget_tracker: BudgetTracker | None,
    ) -> dict[str, object]:
        """Collect metrics for the bundle: timing, token usage, budget status."""
        from weakincentives.runtime.events import PromptExecuted

        # Calculate timing
        duration_ms = (ended_at - started_at).total_seconds() * 1000

        # Collect token usage from PromptExecuted events in session
        total_input_tokens = 0
        total_output_tokens = 0
        total_cached_tokens = 0
        prompt_count = 0

        # Try to get PromptExecuted events from session telemetry
        try:
            telemetry_slice = session[PromptExecuted]
            for event in telemetry_slice.all():  # pragma: no cover
                prompt_count += 1  # pragma: no cover
                if event.usage is not None:  # pragma: no cover
                    total_input_tokens += (
                        event.usage.input_tokens or 0
                    )  # pragma: no cover
                    total_output_tokens += (
                        event.usage.output_tokens or 0
                    )  # pragma: no cover
                    total_cached_tokens += (
                        event.usage.cached_tokens or 0
                    )  # pragma: no cover
        except (KeyError, AttributeError):  # pragma: no cover
            pass  # pragma: no cover

        metrics: dict[str, object] = {
            "timing": {
                "started_at": started_at.isoformat(),
                "ended_at": ended_at.isoformat(),
                "duration_ms": duration_ms,
            },
            "token_usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "cached_tokens": total_cached_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "prompt_count": prompt_count,
            },
        }

        # Add budget status if budget tracking was enabled
        if budget_tracker is not None:
            consumed = budget_tracker.consumed
            budget = budget_tracker.budget
            metrics["budget"] = {
                "consumed": {
                    "input_tokens": consumed.input_tokens,
                    "output_tokens": consumed.output_tokens,
                    "cached_tokens": consumed.cached_tokens,
                    "total_tokens": consumed.total_tokens,
                },
                "limits": {
                    "max_input_tokens": budget.max_input_tokens,
                    "max_output_tokens": budget.max_output_tokens,
                    "max_total_tokens": budget.max_total_tokens,
                    "deadline": budget.deadline.expires_at.isoformat()
                    if budget.deadline is not None
                    else None,
                },
            }

        return metrics

    @staticmethod
    def _format_visibility_overrides(
        overrides: VisibilityOverrides,
        session: Session,
    ) -> dict[str, object]:
        """Format visibility overrides for bundle export."""
        from weakincentives.runtime.session.visibility_overrides import (
            VisibilityOverrides as VOType,
        )

        formatted: dict[str, object] = {"overrides": {}}
        overrides_dict: dict[str, str] = {}

        for path, visibility in overrides.overrides.items():
            # Convert tuple path to string format (e.g., "section.subsection")
            path_str = ".".join(path)
            overrides_dict[path_str] = visibility.value

        formatted["overrides"] = overrides_dict

        # Try to get provenance info from session history
        try:
            vo_slice = session[VOType]
            history = [{"overrides": dict(item.overrides)} for item in vo_slice.all()]
            if history:  # pragma: no cover
                formatted["history_count"] = len(history)  # pragma: no cover
        except (KeyError, AttributeError):  # pragma: no cover
            pass  # pragma: no cover

        return formatted

    def _handle_message_without_bundle(
        self,
        msg: Message[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]],
        request_event: MainLoopRequest[UserRequestT],
        run_context: RunContext,
    ) -> None:
        """Process message without debug bundling."""
        try:
            response, session, _ = self._execute(request_event, run_context=run_context)

            # Add session_id while preserving the same run_id
            run_context = replace(run_context, session_id=session.session_id)

            result = MainLoopResult[OutputT](
                request_id=request_event.request_id,
                output=response.output,
                session_id=session.session_id,
                run_context=run_context,
            )

        except Exception as exc:
            handle_failure(
                msg,
                exc,
                run_context=run_context,
                dlq=self._dlq,
                requests_mailbox=self._requests,
                result_class=MainLoopResult,
            )
            return

        reply_and_ack(msg, result)

    def _reply_and_ack(
        self,
        msg: Message[MainLoopRequest[UserRequestT], MainLoopResult[OutputT]],
        result: MainLoopResult[OutputT],
    ) -> None:
        """Reply with result and acknowledge message.

        Wrapper method for backwards compatibility with tests.
        Delegates to the standalone reply_and_ack function.
        """
        _ = self  # Instance method for API compatibility
        reply_and_ack(msg, result)

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
        - shutdown() is called
        - The requests mailbox is closed

        In-flight messages complete before exit. Unprocessed messages from
        the current batch are nacked for redelivery.

        Args:
            max_iterations: Maximum polling iterations. None for unlimited.
            visibility_timeout: Seconds messages remain invisible during processing.
                Should exceed maximum expected execution time.
            wait_time_seconds: Long poll duration for receiving messages.
        """
        with self._lock:
            self._running = True
            self._shutdown_event.clear()

        iterations = 0
        try:
            while max_iterations is None or iterations < max_iterations:
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

                iterations += 1
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
