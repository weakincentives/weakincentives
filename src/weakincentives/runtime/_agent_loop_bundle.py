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

"""Bundle-related helpers extracted from AgentLoop.

Standalone functions for collecting metrics, formatting visibility overrides,
writing bundle artifacts, resolving adapter names, and orchestrating bundled
message handling.  These were originally static/instance methods on
:class:`AgentLoop` but have no dependency on the loop instance beyond a small
number of parameters that are passed explicitly.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from ..budget import Budget, BudgetTracker
from ..clock import SYSTEM_CLOCK
from ..deadlines import Deadline
from .agent_loop_types import (
    AgentLoopConfig,
    AgentLoopRequest,
    AgentLoopResult,
    LoopRequestState,
)
from .logging import StructuredLogger
from .message_handlers import handle_failure
from .run_context import RunContext
from .session import Session
from .watchdog import Heartbeat

if TYPE_CHECKING:
    from ..adapters.core import PromptResponse, ProviderAdapter
    from ..debug import BundleWriter
    from ..debug.bundle import BundleConfig
    from ..experiment import Experiment
    from ..prompt import Prompt
    from .dlq import DLQPolicy
    from .mailbox import Mailbox, Message
    from .session.visibility_overrides import VisibilityOverrides


class _LoopLike(Protocol):
    """Structural protocol for the subset of AgentLoop used by bundle helpers.

    Defined here to avoid a circular import with ``agent_loop.py``.
    """

    _adapter: ProviderAdapter[Any]
    _config: AgentLoopConfig
    _dlq: DLQPolicy[AgentLoopRequest[Any], AgentLoopResult[Any]] | None
    _requests: Mailbox[AgentLoopRequest[Any], AgentLoopResult[Any]]
    _heartbeat: Heartbeat

    def prepare(
        self,
        request: object,
        *,
        experiment: Experiment | None = ...,
    ) -> tuple[Prompt[Any], Session]: ...

    def _handle_message_without_bundle(
        self,
        msg: Message[AgentLoopRequest[Any], AgentLoopResult[Any]],
        request_event: AgentLoopRequest[Any],
        run_context: RunContext,
    ) -> None: ...

    def _resolve_settings(
        self,
        prompt: Prompt[Any],
        *,
        budget: Budget | None,
        deadline: Deadline | None,
        resources: Mapping[type[object], object] | None,
    ) -> tuple[Prompt[Any], BudgetTracker | None, Deadline | None]: ...

    def _evaluate_with_retries(  # noqa: PLR0913
        self,
        *,
        prompt: Prompt[Any],
        session: Session,
        deadline: Deadline | None,
        budget_tracker: BudgetTracker | None,
        heartbeat: Heartbeat,
        run_context: RunContext | None = ...,
    ) -> PromptResponse[Any]: ...

    def _reply_and_ack(
        self,
        msg: Message[AgentLoopRequest[Any], AgentLoopResult[Any]],
        result: AgentLoopResult[Any],
    ) -> None: ...


def collect_bundle_metrics(
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
                total_input_tokens += event.usage.input_tokens or 0  # pragma: no cover
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


def format_visibility_overrides(
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


def write_bundle_artifacts(  # noqa: PLR0913
    *,
    writer: BundleWriter,
    response: PromptResponse[Any],
    session: Session,
    prompt: Prompt[Any],
    started_at: datetime,
    ended_at: datetime,
    budget_tracker: BudgetTracker | None,
    config: AgentLoopConfig,
) -> None:
    """Write bundle artifacts after execution.

    Shared by ``execute_with_bundle`` and ``handle_message_with_bundle``.

    Note: ``run_context`` is written by the caller before execution
    (not here) to avoid duplicate entries in the manifest files list.
    """
    from ..filesystem import Filesystem
    from .session.visibility_overrides import VisibilityOverrides

    writer.write_session_after(session)
    writer.write_request_output(response)
    writer.write_config(config)

    writer.write_metrics(
        collect_bundle_metrics(
            started_at=started_at,
            ended_at=ended_at,
            session=session,
            budget_tracker=budget_tracker,
        )
    )

    visibility_overrides = session[VisibilityOverrides].latest()
    if visibility_overrides is not None and visibility_overrides.overrides:
        writer.write_prompt_overrides(
            format_visibility_overrides(visibility_overrides, session)
        )

    fs = prompt.resources.get_optional(Filesystem)
    if fs is not None:
        writer.write_filesystem(fs)

    writer.write_environment()


def get_adapter_name(adapter: ProviderAdapter[Any]) -> str:
    """Get the canonical adapter name for the given adapter instance."""
    return adapter.adapter_name


# ---------------------------------------------------------------------------
# Bundle-related orchestration helpers
# ---------------------------------------------------------------------------
# The functions below were extracted from AgentLoop instance methods.  They
# accept the loop instance as their first argument (typed as ``Any`` to avoid
# a circular import with ``agent_loop.py``).
# ---------------------------------------------------------------------------


def _handle_bundle_error(  # noqa: PLR0913, PLR0917
    loop: _LoopLike,
    msg: Message[AgentLoopRequest[Any], AgentLoopResult[Any]],
    exc: Exception,
    prompt: Prompt[Any] | None,
    writer: Any,  # noqa: ANN401 - BundleWriter | None
    prompt_cleaned_up: bool,
    run_context: RunContext,
    log: StructuredLogger,
) -> None:
    """Handle exceptions during bundled message processing."""
    if prompt is not None and not prompt_cleaned_up:
        prompt.cleanup()

    bundle_path = writer.path if writer is not None else None

    if bundle_path is not None:
        log.info(
            "Execution failed but debug bundle was created",
            event="agent_loop.execution_failed_with_bundle",
            context={"error": str(exc), "bundle_path": str(bundle_path)},
        )
    else:
        log.warning(
            "Debug bundle creation failed, falling back to unbundled execution",
            event="agent_loop.bundle_failed",
            context={"error": str(exc)},
        )

    handle_failure(
        msg,
        exc,
        run_context=run_context,
        dlq=loop._dlq,
        requests_mailbox=loop._requests,
        result_class=AgentLoopResult,
        bundle_path=bundle_path,
    )


def _complete_bundled_request(  # noqa: PLR0913, PLR0917
    loop: _LoopLike,
    msg: Message[AgentLoopRequest[Any], AgentLoopResult[Any]],
    request_event: AgentLoopRequest[Any],
    session: Session,
    response: PromptResponse[Any],
    run_context: RunContext,
    writer: Any,  # noqa: ANN401 - BundleWriter
) -> None:
    """Build result from bundled execution and reply to the message."""
    result: AgentLoopResult[Any] = AgentLoopResult(
        request_id=request_event.request_id,
        output=response.output,
        session_id=session.session_id,
        run_context=run_context,
        bundle_path=writer.path,
    )
    loop._reply_and_ack(msg, result)


def handle_message_with_bundle(  # noqa: PLR0913, PLR0917
    loop: _LoopLike,
    msg: Message[AgentLoopRequest[Any], AgentLoopResult[Any]],
    request_event: AgentLoopRequest[Any],
    run_context: RunContext,
    log: StructuredLogger,
    bundle_config: BundleConfig,
) -> None:
    """Process a mailbox message with debug bundling enabled.

    This was previously ``AgentLoop._handle_message_with_bundle``.
    """
    from ..debug import BundleWriter

    if bundle_config.target is None:  # pragma: no cover
        # Defensive guard: _handle_message only calls this when enabled=True
        loop._handle_message_without_bundle(msg, request_event, run_context)
        return

    # Determine trigger based on where config came from
    trigger = "request" if request_event.debug_bundle is not None else "config"

    writer: BundleWriter | None = None
    prompt: Prompt[Any] | None = None
    started_at = SYSTEM_CLOCK.utcnow()
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
            prompt, session = loop.prepare(
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
            writer.set_prompt_info(
                ns=prompt.ns,
                key=prompt.key,
                adapter=get_adapter_name(loop._adapter),
            )

            # Resolve effective settings and execute
            response, budget_tracker = _execute_with_bundled_settings(
                loop,
                request_event=request_event,
                prompt=prompt,
                session=session,
                run_context=run_context,
                writer=writer,
            )

            ended_at = SYSTEM_CLOCK.utcnow()

            write_bundle_artifacts(
                writer=writer,
                response=response,
                session=session,
                prompt=prompt,
                started_at=started_at,
                ended_at=ended_at,
                budget_tracker=budget_tracker,
                config=loop._config,
            )

            prompt_cleaned_up = True
            prompt.cleanup()

        _complete_bundled_request(
            loop, msg, request_event, session, response, run_context, writer
        )

    except Exception as exc:
        _handle_bundle_error(
            loop, msg, exc, prompt, writer, prompt_cleaned_up, run_context, log
        )


def _execute_with_bundled_settings(  # noqa: PLR0913
    loop: _LoopLike,
    *,
    request_event: AgentLoopRequest[Any],
    prompt: Prompt[Any],
    session: Session,
    run_context: RunContext,
    writer: object,  # BundleWriter, but avoid import for typing
) -> tuple[PromptResponse[Any], BudgetTracker | None]:
    """Execute prompt with settings resolved and log capture enabled."""
    prompt, budget_tracker, eff_deadline = loop._resolve_settings(
        prompt,
        budget=request_event.budget,
        deadline=request_event.deadline,
        resources=request_event.resources,
    )

    with writer.capture_logs():  # type: ignore[union-attr]
        response = loop._evaluate_with_retries(
            prompt=prompt,
            session=session,
            deadline=eff_deadline,
            budget_tracker=budget_tracker,
            heartbeat=loop._heartbeat,
            run_context=run_context,
        )

    return response, budget_tracker
