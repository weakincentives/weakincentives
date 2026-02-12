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

"""Agent loop orchestration for agent workflow execution.

AgentLoop provides a mailbox-based pattern for durable request processing with
at-least-once delivery semantics. Requests are received from a mailbox queue
and results are sent via Message.reply().

Example::

    class CodeReviewLoop(AgentLoop[ReviewRequest, ReviewResult]):
        def __init__(
            self,
            *,
            adapter: ProviderAdapter[ReviewResult],
            requests: Mailbox[AgentLoopRequest[ReviewRequest], AgentLoopResult[ReviewResult]],
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

import os
import socket
import time
from abc import abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, override
from uuid import UUID, uuid4

from ..budget import Budget, BudgetTracker
from ..deadlines import Deadline
from ..prompt.errors import VisibilityExpansionRequired
from .agent_loop_types import (
    AgentLoopConfig,
    AgentLoopRequest,
    AgentLoopResult,
    LoopFinalResponse,
    LoopRawResponse,
    LoopRequestState,
)
from .dlq import DLQPolicy
from .logging import StructuredLogger, bind_run_context, get_logger
from .mailbox import Mailbox, Message
from .mailbox_worker import MailboxWorker
from .message_handlers import handle_failure, reply_and_ack
from .run_context import RunContext
from .session import Session
from .session.visibility_overrides import SetVisibilityOverride
from .watchdog import Heartbeat

if TYPE_CHECKING:
    from ..adapters.core import PromptResponse, ProviderAdapter
    from ..debug import BundleWriter
    from ..debug.bundle import BundleConfig
    from ..experiment import Experiment
    from ..prompt import Prompt
    from .agent_loop_types import BundleContext
    from .session.visibility_overrides import VisibilityOverrides

_logger: StructuredLogger = get_logger(
    __name__, context={"component": "runtime.agent_loop"}
)

_MAX_VISIBILITY_RETRIES: int = 10
"""Maximum number of visibility expansion retries before giving up."""


class AgentLoop[UserRequestT, OutputT](
    MailboxWorker[AgentLoopRequest[UserRequestT], AgentLoopResult[OutputT]]
):
    """Abstract orchestrator for mailbox-based agent workflow execution.

    AgentLoop processes requests from a mailbox queue and sends responses
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
        6. Send ``AgentLoopResult`` via msg.reply()
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
    _config: AgentLoopConfig
    _dlq: DLQPolicy[AgentLoopRequest[UserRequestT], AgentLoopResult[OutputT]] | None
    _worker_id: str

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[OutputT],
        requests: Mailbox[AgentLoopRequest[UserRequestT], AgentLoopResult[OutputT]],
        config: AgentLoopConfig | None = None,
        worker_id: str | None = None,
        dlq: DLQPolicy[AgentLoopRequest[UserRequestT], AgentLoopResult[OutputT]]
        | None = None,
    ) -> None:
        """Initialize the AgentLoop.

        Args:
            adapter: Provider adapter for prompt evaluation.
            requests: Mailbox to receive AgentLoopRequest messages from.
                Response routing derives from each message's reply_to field.
            config: Optional configuration for default budget/resources.
            worker_id: Identifier for this worker instance. If None or empty,
                auto-generates as "{hostname}-{pid}". Recommended formats:
                - Production: "{hostname}-{pid}" or "{k8s_pod_name}"
                - Testing: "test-worker" or descriptive test name
            dlq: Optional dead letter queue policy. When configured, messages
                that fail repeatedly are sent to the DLQ mailbox instead of
                retrying indefinitely.
        """
        effective_config = config if config is not None else AgentLoopConfig()
        super().__init__(
            requests=requests,
            lease_extender_config=effective_config.lease_extender,
        )
        self._adapter = adapter
        self._config = effective_config
        self._dlq = dlq
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

    def finalize(
        self,
        prompt: Prompt[OutputT],
        session: Session,
        output: OutputT | None,
    ) -> OutputT | None:
        """Finalize after execution completes.

        Called after successful evaluation. Override to perform cleanup,
        logging, or post-processing tasks. The returned output replaces
        the original output in the response.

        Args:
            prompt: The prompt that was evaluated.
            session: The session used for evaluation.
            output: The parsed output from the model response, or None
                for text-only prompts without structured output.

        Returns:
            The (possibly transformed) output to use in the final result.
        """
        _ = (self, prompt, session)
        return output

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
            deadline: Optional deadline for this execution.
            resources: Optional resources override (dict mapping types to instances).
            heartbeat: Optional heartbeat for lease extension. If provided, this
                heartbeat is used for adapter evaluation instead of the loop's
                internal heartbeat. Use this when EvalLoop needs to extend its
                own message lease based on AgentLoop work.
            experiment: Optional experiment for A/B testing. Passed to prepare().

        Returns:
            Tuple of (PromptResponse, Session) from the evaluation.
        """
        request_event = AgentLoopRequest(
            request=request,
            budget=budget,
            deadline=deadline,
            resources=resources,
            experiment=experiment,
        )
        response, session, _ = self._execute(request_event, heartbeat=heartbeat)
        return response, session

    @contextmanager
    def execute_with_bundle(  # noqa: PLR0913
        self,
        request: UserRequestT,
        *,
        bundle_target: Path,
        bundle_config: BundleConfig | None = None,
        budget: Budget | None = None,
        deadline: Deadline | None = None,
        resources: Mapping[type[object], object] | None = None,
        heartbeat: Heartbeat | None = None,
        experiment: Experiment | None = None,
    ) -> Iterator[BundleContext[OutputT]]:
        """Execute with debug bundling, allowing metadata injection.

        Creates a debug bundle with the same artifacts as mailbox-driven execution:
        request input/output, session state, logs, config, metrics, and environment.

        The yielded context provides access to execution results and allows adding
        eval-specific metadata before the bundle is finalized.

        Args:
            request: The user request to process.
            bundle_target: Directory for bundle creation. The bundle zip file will
                be created in this directory with a generated filename.
            bundle_config: Optional configuration for bundle creation. When provided,
                settings like storage_handler (for uploading to external storage),
                retention policy, max_file_size, and compression are used. If not
                provided, default BundleConfig settings are used.
            budget: Optional budget override (takes precedence over config).
            deadline: Optional deadline for this execution.
            resources: Optional resources override.
            heartbeat: Optional heartbeat for lease extension.
            experiment: Optional experiment for A/B testing.

        Yields:
            BundleContext with response, session, latency_ms, and write_metadata().

        Example::

            bundle_dir = Path("./bundles")
            with loop.execute_with_bundle(request, bundle_target=bundle_dir) as ctx:
                score = compute_score(ctx.response.output)
                ctx.write_metadata("eval", {
                    "sample_id": "sample-1",
                    "score": {"value": score.value, "passed": score.passed},
                })
            # Bundle is now finalized
            print(f"Bundle at: {ctx.bundle_path}")

            # With storage handler for external upload:
            config = BundleConfig(
                target=bundle_dir,
                storage_handler=S3StorageHandler(bucket="my-bucket"),
            )
            with loop.execute_with_bundle(request, bundle_target=bundle_dir,
                                          bundle_config=config) as ctx:
                ...  # Bundle will be uploaded to S3 after finalization
        """
        from ..debug import BundleWriter
        from .agent_loop_types import BundleContext

        bundle_target.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now(UTC)
        start_mono = time.monotonic()

        with BundleWriter(
            bundle_target, bundle_id=uuid4(), config=bundle_config, trigger="direct"
        ) as writer:
            # Create request event for tracking
            request_event = AgentLoopRequest(
                request=request,
                budget=budget,
                deadline=deadline,
                resources=resources,
                experiment=experiment,
            )

            # Write request input
            writer.write_request_input(request_event)

            # Prepare and execute
            response, session, prompt, budget_tracker = self._execute_for_bundle(
                request=request,
                budget=budget,
                deadline=deadline,
                resources=resources,
                heartbeat=heartbeat,
                experiment=experiment,
                writer=writer,
                request_event=request_event,
            )

            ended_at = datetime.now(UTC)

            # Create context for caller to access results and add metadata
            ctx: BundleContext[OutputT] = BundleContext(
                writer=writer,
                response=response,
                session=session,
                latency_ms=int((time.monotonic() - start_mono) * 1000),
            )

            # Yield to allow caller to compute score and add eval metadata
            yield ctx

            # Write remaining artifacts after caller has added metadata
            self._write_bundle_artifacts(
                writer=writer,
                response=response,
                session=session,
                prompt=prompt,
                started_at=started_at,
                ended_at=ended_at,
                budget_tracker=budget_tracker,
            )

            prompt.cleanup()

        # BundleWriter context has exited, bundle is finalized
        # ctx.bundle_path now returns writer.path

    def _execute_for_bundle(  # noqa: PLR0913
        self,
        *,
        request: UserRequestT,
        budget: Budget | None,
        deadline: Deadline | None,
        resources: Mapping[type[object], object] | None,
        heartbeat: Heartbeat | None,
        experiment: Experiment | None,
        writer: BundleWriter,
        request_event: AgentLoopRequest[UserRequestT],
    ) -> tuple[PromptResponse[OutputT], Session, Prompt[OutputT], BudgetTracker | None]:
        """Execute within bundle context with log capture."""
        prompt, session = self.prepare(request, experiment=experiment)

        # Publish request state to session after prepare()
        _ = session.dispatch(LoopRequestState(request=request_event))

        writer.write_session_before(session)
        writer.set_prompt_info(
            ns=prompt.ns, key=prompt.key, adapter=self._get_adapter_name()
        )

        prompt, budget_tracker, eff_deadline = self._resolve_settings(
            prompt, budget=budget, deadline=deadline, resources=resources
        )
        eff_heartbeat = heartbeat if heartbeat is not None else self._heartbeat

        with writer.capture_logs():
            response = self._evaluate_with_retries(
                prompt=prompt,
                session=session,
                deadline=eff_deadline,
                budget_tracker=budget_tracker,
                heartbeat=eff_heartbeat,
            )

        return response, session, prompt, budget_tracker

    def _write_bundle_artifacts(  # noqa: PLR0913
        self,
        *,
        writer: BundleWriter,
        response: PromptResponse[OutputT],
        session: Session,
        prompt: Prompt[OutputT],
        started_at: datetime,
        ended_at: datetime,
        budget_tracker: BudgetTracker | None,
        run_context: RunContext | None = None,
    ) -> None:
        """Write bundle artifacts after execution.

        Shared by execute_with_bundle and _handle_message_with_bundle.
        """
        from ..filesystem import Filesystem
        from .session.visibility_overrides import VisibilityOverrides

        writer.write_session_after(session)
        writer.write_request_output(response)
        writer.write_config(self._config)

        if run_context is not None:
            writer.write_run_context(run_context)

        writer.write_metrics(
            self._collect_metrics(
                started_at=started_at,
                ended_at=ended_at,
                session=session,
                budget_tracker=budget_tracker,
            )
        )

        visibility_overrides = session[VisibilityOverrides].latest()
        if visibility_overrides is not None and visibility_overrides.overrides:
            writer.write_prompt_overrides(
                self._format_visibility_overrides(visibility_overrides, session)
            )

        fs = prompt.resources.get_optional(Filesystem)
        if fs is not None:
            writer.write_filesystem(fs)

        writer.write_environment()

    def _execute(
        self,
        request_event: AgentLoopRequest[UserRequestT],
        *,
        heartbeat: Heartbeat | None = None,
        run_context: RunContext | None = None,
    ) -> tuple[PromptResponse[OutputT], Session, Prompt[OutputT]]:
        """Execute the agent loop for a request event.

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

        # Publish request state to session after prepare()
        _ = session.dispatch(LoopRequestState(request=request_event))

        # Resolve and apply effective settings
        prompt, budget_tracker, deadline = self._resolve_settings(
            prompt,
            budget=request_event.budget,
            deadline=request_event.deadline,
            resources=request_event.resources,
        )

        # Use provided heartbeat or fall back to loop's internal heartbeat
        effective_heartbeat = heartbeat if heartbeat is not None else self._heartbeat

        try:
            response = self._evaluate_with_retries(
                prompt=prompt,
                session=session,
                deadline=deadline,
                budget_tracker=budget_tracker,
                heartbeat=effective_heartbeat,
                run_context=run_context,
            )
        finally:
            prompt.cleanup()
        return response, session, prompt

    def _resolve_settings(
        self,
        prompt: Prompt[OutputT],
        *,
        budget: Budget | None,
        deadline: Deadline | None,
        resources: Mapping[type[object], object] | None,
    ) -> tuple[Prompt[OutputT], BudgetTracker | None, Deadline | None]:
        """Resolve effective settings and bind resources to prompt.

        Returns:
            Tuple of (prompt_with_resources, budget_tracker, effective_deadline).
        """
        eff_budget = budget if budget is not None else self._config.budget
        eff_resources = resources if resources is not None else self._config.resources

        if eff_resources is not None:
            prompt = prompt.bind(resources=eff_resources)

        budget_tracker = BudgetTracker(budget=eff_budget) if eff_budget else None
        return prompt, budget_tracker, deadline

    def _evaluate_with_retries(  # noqa: PLR0913
        self,
        *,
        prompt: Prompt[OutputT],
        session: Session,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        heartbeat: Heartbeat,
        run_context: RunContext | None = None,
    ) -> PromptResponse[OutputT]:
        """Run evaluation with visibility expansion retry loop.

        Handles the core evaluate → catch VisibilityExpansionRequired →
        dispatch overrides → retry cycle. Calls finalize() on success.

        Returns:
            The prompt response from successful evaluation.
        """
        retries = 0
        while True:
            try:
                response = self._adapter.evaluate(
                    prompt,
                    session=session,
                    deadline=deadline,
                    budget_tracker=budget_tracker,
                    heartbeat=heartbeat,
                    run_context=run_context,
                )
            except VisibilityExpansionRequired as e:
                retries += 1
                if retries > _MAX_VISIBILITY_RETRIES:
                    from ..adapters.core import PromptEvaluationError

                    raise PromptEvaluationError(
                        f"Visibility expansion retries exceeded ({_MAX_VISIBILITY_RETRIES})",
                        prompt_name=prompt.key,
                        phase="request",
                    ) from e
                for path, visibility in e.requested_overrides.items():
                    _ = session.dispatch(
                        SetVisibilityOverride(path=path, visibility=visibility)
                    )
            else:
                # Publish raw response before finalize()
                _ = session.dispatch(
                    LoopRawResponse(
                        prompt_name=response.prompt_name,
                        text=response.text,
                        output=response.output,
                    )
                )

                output = self.finalize(prompt, session, response.output)
                final_response = replace(response, output=output)

                # Publish final response after finalize()
                _ = session.dispatch(
                    LoopFinalResponse(
                        prompt_name=final_response.prompt_name,
                        text=final_response.text,
                        output=final_response.output,
                    )
                )

                return final_response

    def _build_run_context(
        self,
        request_event: AgentLoopRequest[UserRequestT],
        delivery_count: int,
        session_id: UUID | None = None,
    ) -> RunContext:
        """Build RunContext for this execution.

        The run_id is always fresh for each execution attempt.

        Request ID always comes from request_event.request_id to ensure
        correlation with AgentLoopResult.request_id. The run_context parameter
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

    @override
    def _process_message(
        self, msg: Message[AgentLoopRequest[UserRequestT], AgentLoopResult[OutputT]]
    ) -> None:
        """Process a single message from the requests mailbox.

        Implements the abstract method from MailboxWorker. Called with lease
        extension already attached by the base class.
        """
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
            event="agent_loop.message_received",
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

        # Lease extension is handled by MailboxWorker.run()
        if bundle_config is not None and bundle_config.enabled:
            self._handle_message_with_bundle(
                msg, request_event, run_context, log, bundle_config
            )
        else:
            self._handle_message_without_bundle(msg, request_event, run_context)

    def _handle_message_with_bundle(
        self,
        msg: Message[AgentLoopRequest[UserRequestT], AgentLoopResult[OutputT]],
        request_event: AgentLoopRequest[UserRequestT],
        run_context: RunContext,
        log: StructuredLogger,
        bundle_config: BundleConfig,
    ) -> None:
        """Process message with debug bundling enabled."""
        from ..debug import BundleWriter

        if bundle_config.target is None:  # pragma: no cover
            # Defensive guard: _handle_message only calls this when enabled=True
            self._handle_message_without_bundle(msg, request_event, run_context)
            return

        # Determine trigger based on where config came from
        trigger = "request" if request_event.debug_bundle is not None else "config"

        writer: BundleWriter | None = None
        prompt: Prompt[OutputT] | None = None
        started_at = datetime.now(UTC)
        budget_tracker: BudgetTracker | None = None
        prompt_cleaned_up = False
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

                # Publish request state to session after prepare()
                _ = session.dispatch(LoopRequestState(request=request_event))

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

                self._write_bundle_artifacts(
                    writer=writer,
                    response=response,
                    session=session,
                    prompt=prompt,
                    started_at=started_at,
                    ended_at=ended_at,
                    budget_tracker=budget_tracker,
                    run_context=run_context,
                )

                prompt_cleaned_up = True
                prompt.cleanup()

            # Bundle path is set after context manager exits (in __exit__ -> _finalize)
            bundle_path = writer.path

            result = AgentLoopResult[OutputT](
                request_id=request_event.request_id,
                output=response.output,
                session_id=session.session_id,
                run_context=run_context,
                bundle_path=bundle_path,
            )
            reply_and_ack(msg, result)

        except Exception as exc:
            # Clean up prompt resources if prompt was created
            if prompt is not None and not prompt_cleaned_up:
                prompt.cleanup()

            # Check if bundle was created despite the error
            bundle_path = writer.path if writer is not None else None

            if bundle_path is not None:
                # Bundle was created successfully, error was during execution
                log.info(
                    "Execution failed but debug bundle was created",
                    event="agent_loop.execution_failed_with_bundle",
                    context={"error": str(exc), "bundle_path": str(bundle_path)},
                )
            else:
                # True bundle creation failure
                log.warning(
                    "Debug bundle creation failed, falling back to unbundled execution",
                    event="agent_loop.bundle_failed",
                    context={"error": str(exc)},
                )

            handle_failure(
                msg,
                exc,
                run_context=run_context,
                dlq=self._dlq,
                requests_mailbox=self._requests,
                result_class=AgentLoopResult,
                bundle_path=bundle_path,
            )

    def _execute_with_bundled_settings(
        self,
        *,
        request_event: AgentLoopRequest[UserRequestT],
        prompt: Prompt[OutputT],
        session: Session,
        run_context: RunContext,
        writer: object,  # BundleWriter, but avoid import for typing
    ) -> tuple[PromptResponse[OutputT], BudgetTracker | None]:
        """Execute prompt with settings resolved and log capture enabled.

        This helper reduces nesting in _handle_message_with_bundle by
        encapsulating the execution loop with visibility override retry.
        """
        prompt, budget_tracker, eff_deadline = self._resolve_settings(
            prompt,
            budget=request_event.budget,
            deadline=request_event.deadline,
            resources=request_event.resources,
        )

        with writer.capture_logs():  # type: ignore[union-attr]
            response = self._evaluate_with_retries(
                prompt=prompt,
                session=session,
                deadline=eff_deadline,
                budget_tracker=budget_tracker,
                heartbeat=self._heartbeat,
                run_context=run_context,
            )

        return response, budget_tracker

    def _get_adapter_name(self) -> str:
        """Get the canonical adapter name for the current adapter."""
        from ..adapters import (
            CLAUDE_AGENT_SDK_ADAPTER_NAME,
            CODEX_APP_SERVER_ADAPTER_NAME,
        )
        from ..adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
        from ..adapters.codex_app_server import CodexAppServerAdapter

        if isinstance(self._adapter, ClaudeAgentSDKAdapter):
            return CLAUDE_AGENT_SDK_ADAPTER_NAME  # pragma: no cover
        if isinstance(self._adapter, CodexAppServerAdapter):
            return CODEX_APP_SERVER_ADAPTER_NAME
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
        msg: Message[AgentLoopRequest[UserRequestT], AgentLoopResult[OutputT]],
        request_event: AgentLoopRequest[UserRequestT],
        run_context: RunContext,
    ) -> None:
        """Process message without debug bundling."""
        try:
            response, session, _ = self._execute(request_event, run_context=run_context)

            # Add session_id while preserving the same run_id
            run_context = replace(run_context, session_id=session.session_id)

            result = AgentLoopResult[OutputT](
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
                result_class=AgentLoopResult,
            )
            return

        reply_and_ack(msg, result)

    def _reply_and_ack(
        self,
        msg: Message[AgentLoopRequest[UserRequestT], AgentLoopResult[OutputT]],
        result: AgentLoopResult[OutputT],
    ) -> None:
        """Reply with result and acknowledge message.

        Wrapper method for backwards compatibility with tests.
        Delegates to the standalone reply_and_ack function.
        """
        _ = self  # Instance method for API compatibility
        reply_and_ack(msg, result)


__all__ = [
    "AgentLoop",
    "AgentLoopConfig",
    "AgentLoopRequest",
    "AgentLoopResult",
]
