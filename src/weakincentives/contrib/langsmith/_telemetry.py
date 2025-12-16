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

"""LangSmith telemetry handler for WINK event-based tracing."""

from __future__ import annotations

import atexit
import random
from datetime import UTC, datetime
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID, uuid4

from ...runtime.events import PromptExecuted, PromptRendered, ToolInvoked
from ...runtime.logging import StructuredLogger, get_logger
from ._config import LangSmithConfig
from ._context import (
    TraceContext,
    clear_context,
    clear_current_run_tree,
    get_context,
    set_context,
    set_current_run_tree,
)
from ._events import (
    LangSmithTraceCompleted,
    LangSmithTraceStarted,
)

if TYPE_CHECKING:
    from ...runtime.events import EventBus


logger: StructuredLogger = get_logger(
    __name__, context={"component": "langsmith_telemetry"}
)


class LangSmithClientProtocol(Protocol):
    """Protocol for LangSmith client operations used by telemetry handler.

    This protocol matches the langsmith SDK's Client API.
    """

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
    ) -> None: ...

    def update_run(  # noqa: PLR0913, PLR0917
        self,
        run_id: UUID,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
        end_time: datetime | None = None,
        extra: dict[str, Any] | None = None,
        events: list[dict[str, Any]] | None = None,
    ) -> None: ...


class RunUploadItem:
    """Base class for items in the upload queue."""


class CreateRunItem(RunUploadItem):
    """Queue item for creating a new run."""

    __slots__ = (
        "end_time",
        "extra",
        "inputs",
        "name",
        "parent_run_id",
        "project_name",
        "run_id",
        "run_type",
        "start_time",
        "tags",
    )

    def __init__(  # noqa: PLR0913
        self,
        *,
        name: str,
        run_type: str,
        run_id: UUID,
        project_name: str,
        inputs: dict[str, Any] | None = None,
        parent_run_id: UUID | None = None,
        extra: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> None:
        self.name = name
        self.run_type = run_type
        self.run_id = run_id
        self.project_name = project_name
        self.inputs = inputs
        self.parent_run_id = parent_run_id
        self.extra = extra
        self.tags = tags
        self.start_time = start_time
        self.end_time = end_time


class UpdateRunItem(RunUploadItem):
    """Queue item for updating an existing run."""

    __slots__ = ("end_time", "error", "events", "extra", "outputs", "run_id")

    def __init__(  # noqa: PLR0913
        self,
        *,
        run_id: UUID,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
        end_time: datetime | None = None,
        extra: dict[str, Any] | None = None,
        events: list[dict[str, Any]] | None = None,
    ) -> None:
        self.run_id = run_id
        self.outputs = outputs
        self.error = error
        self.end_time = end_time
        self.extra = extra
        self.events = events


class LangSmithTelemetryHandler:
    """Subscribes to WINK events and publishes to LangSmith.

    This handler captures:
    - ``PromptRendered`` → Creates chain run (root of trace)
    - ``ToolInvoked`` → Creates tool run (child of chain)
    - ``PromptExecuted`` → Updates chain run with outputs

    All LangSmith API calls are made asynchronously via a background thread
    to avoid blocking the evaluation loop.

    Example::

        config = LangSmithConfig(project="my-agent")
        telemetry = LangSmithTelemetryHandler(config)

        bus = InProcessEventBus()
        telemetry.attach(bus)

        try:
            response = adapter.evaluate(prompt, session=session)
        finally:
            telemetry.flush()
            telemetry.detach(bus)
    """

    def __init__(
        self,
        config: LangSmithConfig,
        *,
        client: LangSmithClientProtocol | None = None,
    ) -> None:
        """Initialize the telemetry handler.

        Args:
            config: LangSmith configuration settings.
            client: Optional LangSmith client for testing. If not provided,
                creates a client using the config settings.
        """
        self._config = config
        self._client = client
        self._attached_buses: list[EventBus] = []
        self._traced_call_ids: set[str] = set()

        # Async upload queue
        self._queue: Queue[RunUploadItem | None] = Queue(maxsize=config.max_queue_size)
        self._stop_event = Event()
        self._upload_thread: Thread | None = None

        if config.async_upload and config.is_tracing_enabled():
            self._start_upload_thread()

        if config.flush_on_exit:
            _ = atexit.register(self._atexit_flush)

    def _get_client(self) -> LangSmithClientProtocol:
        """Get or create the LangSmith client."""
        if self._client is not None:
            return self._client

        try:
            from langsmith import Client  # type: ignore[import-not-found]

            self._client = Client(
                api_key=self._config.resolved_api_key(),
                api_url=self._config.resolved_api_url(),
            )
        except ImportError as error:
            msg = "langsmith package is required for LangSmith telemetry"
            raise ImportError(msg) from error
        else:
            return self._client

    def _start_upload_thread(self) -> None:
        """Start the background upload thread."""
        self._upload_thread = Thread(
            target=self._upload_loop,
            name="langsmith-upload",
            daemon=True,
        )
        self._upload_thread.start()

    def _upload_loop(self) -> None:
        """Background loop that processes the upload queue."""
        batch: list[RunUploadItem] = []
        batch_size = self._config.upload_batch_size
        interval = self._config.upload_interval_seconds

        while not self._stop_event.is_set():
            try:
                # Wait for item with timeout
                item = self._queue.get(timeout=interval)
                if item is None:  # Shutdown signal
                    break
                batch.append(item)

                # Drain queue up to batch size
                while len(batch) < batch_size:
                    try:
                        item = self._queue.get_nowait()
                        if item is None:
                            break
                        batch.append(item)
                    except Empty:
                        break

                # Upload batch
                if batch:
                    self._upload_batch(batch)
                    batch = []

            except Empty:
                # Timeout - upload any pending items
                if batch:
                    self._upload_batch(batch)
                    batch = []

        # Final flush on shutdown
        if batch:
            self._upload_batch(batch)

    def _upload_batch(self, batch: list[RunUploadItem]) -> None:
        """Upload a batch of run items to LangSmith."""
        try:
            client = self._get_client()
            for item in batch:
                LangSmithTelemetryHandler._upload_item(client, item)
        except Exception as error:
            logger.warning(
                "Failed to upload batch to LangSmith",
                event="langsmith_upload_failed",
                context={
                    "batch_size": len(batch),
                    "error": str(error),
                },
            )

    @staticmethod
    def _upload_item(client: LangSmithClientProtocol, item: RunUploadItem) -> None:
        """Upload a single item to LangSmith."""
        if isinstance(item, CreateRunItem):
            client.create_run(
                name=item.name,
                run_type=item.run_type,
                run_id=item.run_id,
                inputs=item.inputs,
                parent_run_id=item.parent_run_id,
                project_name=item.project_name,
                extra=item.extra,
                tags=item.tags,
                start_time=item.start_time,
            )
            # If end_time is set, also update to close the run
            if item.end_time is not None:
                client.update_run(
                    run_id=item.run_id,
                    end_time=item.end_time,
                )
        elif isinstance(item, UpdateRunItem):
            client.update_run(
                run_id=item.run_id,
                outputs=item.outputs,
                error=item.error,
                end_time=item.end_time,
                extra=item.extra,
                events=item.events,
            )

    def _enqueue(self, item: RunUploadItem) -> None:
        """Add item to upload queue."""
        if not self._config.async_upload:
            # Sync mode - upload immediately
            try:
                client = self._get_client()
                LangSmithTelemetryHandler._upload_item(client, item)
            except Exception as error:
                logger.warning(
                    "Failed to upload to LangSmith",
                    event="langsmith_upload_failed",
                    context={"error": str(error)},
                )
            return

        try:
            self._queue.put_nowait(item)
        except Full:
            logger.warning(
                "LangSmith upload queue full, dropping item",
                event="langsmith_queue_full",
                context={"queue_size": self._config.max_queue_size},
            )

    def attach(self, bus: EventBus) -> None:
        """Subscribe to all telemetry events on the bus.

        Args:
            bus: The event bus to subscribe to.
        """
        if not self._config.is_tracing_enabled():
            return

        bus.subscribe(PromptRendered, self._on_prompt_rendered)
        bus.subscribe(ToolInvoked, self._on_tool_invoked)
        bus.subscribe(PromptExecuted, self._on_prompt_executed)
        self._attached_buses.append(bus)

    def detach(self, bus: EventBus) -> None:
        """Unsubscribe from all telemetry events on the bus.

        Args:
            bus: The event bus to unsubscribe from.
        """
        bus.unsubscribe(PromptRendered, self._on_prompt_rendered)
        bus.unsubscribe(ToolInvoked, self._on_tool_invoked)
        bus.unsubscribe(PromptExecuted, self._on_prompt_executed)
        if bus in self._attached_buses:
            self._attached_buses.remove(bus)

    def flush(self, *, timeout: float | None = None) -> None:
        """Block until pending uploads complete.

        Args:
            timeout: Maximum time to wait in seconds. If ``None``, waits indefinitely.
        """
        if not self._config.async_upload or self._upload_thread is None:
            return

        # Signal upload thread to flush
        self._queue.put(None)

        # Wait for thread to process queue
        self._upload_thread.join(timeout=timeout)

        # Restart thread if still running
        if self._upload_thread.is_alive():
            self._upload_thread = None
            self._start_upload_thread()
        elif not self._stop_event.is_set():
            self._start_upload_thread()

    def shutdown(self) -> None:
        """Stop the upload thread and flush pending items."""
        self._stop_event.set()
        if self._upload_thread is not None:
            self._queue.put(None)  # Signal shutdown
            self._upload_thread.join(timeout=5.0)
            self._upload_thread = None

    def _atexit_flush(self) -> None:
        """Flush handler called at interpreter exit."""
        self.shutdown()

    def _should_sample(self) -> bool:
        """Check if this trace should be sampled."""
        rate = self._config.trace_sample_rate
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        return random.random() < rate

    def _is_traced_by_native_integration(self, call_id: str | None) -> bool:
        """Check if a tool call was already traced (for deduplication)."""
        if call_id is None:
            return False
        # Check if we've already traced this call_id
        return call_id in self._traced_call_ids

    def _on_prompt_rendered(self, event: object) -> None:
        """Handle PromptRendered event - create chain run."""
        if not isinstance(event, PromptRendered):
            return

        if not self._should_sample():
            return

        now = datetime.now(UTC)
        trace_id = uuid4()
        run_id = uuid4()
        project = self._config.resolved_project()

        # Create trace context
        context = TraceContext(
            trace_id=trace_id,
            root_run_id=run_id,
            current_run_id=run_id,
            session_id=event.session_id,
            run_count=1,
        )

        if event.session_id is not None:
            set_context(event.session_id, context)

        # Build metadata
        metadata: dict[str, Any] = {
            "prompt_ns": event.prompt_ns,
            "prompt_key": event.prompt_key,
        }
        if event.session_id is not None:
            metadata["wink_session_id"] = str(event.session_id)
        if event.descriptor is not None:
            metadata["descriptor_sections"] = len(event.descriptor.sections)
            metadata["descriptor_tools"] = len(event.descriptor.tools)

        # Create chain run
        self._enqueue(
            CreateRunItem(
                name=event.prompt_name or f"{event.prompt_ns}/{event.prompt_key}",
                run_type="chain",
                run_id=run_id,
                project_name=project,
                inputs={
                    "rendered_prompt": event.rendered_prompt,
                    "render_inputs": list(event.render_inputs),
                },
                extra={"metadata": metadata},
                tags=[event.prompt_ns, event.adapter],
                start_time=event.created_at,
            )
        )

        # Publish trace started event
        for bus in self._attached_buses:
            bus.publish(
                LangSmithTraceStarted(
                    trace_id=trace_id,
                    session_id=event.session_id,
                    project=project,
                    created_at=now,
                )
            )

        # Set run context for LangSmith SDK integration
        self._set_langsmith_parent_context(run_id, trace_id)

    def _on_tool_invoked(self, event: object) -> None:
        """Handle ToolInvoked event - create tool run."""
        if not isinstance(event, ToolInvoked):
            return

        # Check for deduplication
        if self._is_traced_by_native_integration(event.call_id):
            return

        # Track this call_id
        if event.call_id is not None:
            self._traced_call_ids.add(event.call_id)

        # Get trace context
        context = get_context(event.session_id)
        if context is None:
            # No active trace - skip
            return

        run_id = uuid4()
        context.run_count += 1

        # Track tokens
        if event.usage is not None:
            total = event.usage.total_tokens
            if total is not None:
                context.total_tokens += total

        # Build inputs/outputs
        inputs: dict[str, Any] = {"params": event.params}
        outputs: dict[str, Any] = {"result": event.result}
        if event.rendered_output:
            outputs["rendered_output"] = event.rendered_output

        # Create tool run (child of chain)
        self._enqueue(
            CreateRunItem(
                name=event.name,
                run_type="tool",
                run_id=run_id,
                parent_run_id=context.root_run_id,
                project_name=self._config.resolved_project(),
                inputs=inputs,
                extra={
                    "metadata": {
                        "prompt_name": event.prompt_name,
                        "call_id": event.call_id,
                    }
                },
                start_time=event.created_at,
                end_time=event.created_at,  # Tool runs are instantaneous
            )
        )

    def _on_prompt_executed(self, event: object) -> None:
        """Handle PromptExecuted event - update chain run."""
        if not isinstance(event, PromptExecuted):
            return

        # Get and clear trace context
        context = clear_context(event.session_id)
        if context is None:
            return

        # Track final tokens
        total_tokens = context.total_tokens
        if event.usage is not None:
            total = event.usage.total_tokens
            if total is not None:
                total_tokens += total

        # Update chain run with outputs
        outputs: dict[str, Any] = {"result": event.result}
        extra: dict[str, Any] = {}
        if event.usage is not None:
            extra["usage"] = {
                "input_tokens": event.usage.input_tokens,
                "output_tokens": event.usage.output_tokens,
                "cached_tokens": event.usage.cached_tokens,
                "total_tokens": event.usage.total_tokens,
            }

        self._enqueue(
            UpdateRunItem(
                run_id=context.root_run_id,
                outputs=outputs,
                end_time=event.created_at,
                extra=extra if extra else None,
            )
        )

        # Publish trace completed event
        now = datetime.now(UTC)
        for bus in self._attached_buses:
            bus.publish(
                LangSmithTraceCompleted(
                    trace_id=context.trace_id,
                    run_count=context.run_count,
                    total_tokens=total_tokens,
                    trace_url=self._build_trace_url(context.trace_id),
                    created_at=now,
                )
            )

        # Clear run context
        clear_current_run_tree()

    def _build_trace_url(self, trace_id: UUID) -> str | None:
        """Build LangSmith UI URL for trace."""
        try:
            api_url = self._config.resolved_api_url()
            project = self._config.resolved_project()
            # Convert API URL to UI URL
            if "api.smith.langchain.com" in api_url:
                base = "https://smith.langchain.com"
            else:
                # Self-hosted - assume same domain
                base = api_url.replace("/api", "")
        except Exception:
            return None
        else:
            return f"{base}/public/{project}/r/{trace_id}"

    def _set_langsmith_parent_context(self, run_id: UUID, trace_id: UUID) -> None:
        """Set LangSmith parent context for SDK integration."""
        try:
            from langsmith.run_trees import RunTree  # type: ignore[import-not-found]

            # Create a RunTree for manual control
            run_tree = RunTree(
                name="wink_trace",
                run_type="chain",
                id=run_id,
                trace_id=trace_id,
                project_name=self._config.resolved_project(),
            )
            set_current_run_tree(run_tree)

        except ImportError:
            # LangSmith SDK not available - skip context propagation
            pass


__all__ = ["LangSmithTelemetryHandler"]
