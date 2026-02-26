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

"""Runtime primitives for agent workflow execution and state management.

This package provides the core runtime infrastructure for weakincentives,
including session state management, event dispatching, message queuing,
and lifecycle coordination for agent workflows.

Architecture Overview
---------------------

The runtime consists of four major components:

**Session (State Management)**
    Redux-style state container with typed slices, reducers, and event-driven
    dispatch. Sessions store dataclass payloads from prompt executions and
    tool invocations.

**Events (Telemetry)**
    In-process event dispatching for adapter telemetry including prompt
    rendering, tool invocation, and execution completion tracking.

**Mailbox (Message Queuing)**
    Point-to-point message delivery with SQS-compatible semantics for
    durable request processing with at-least-once delivery.

**Lifecycle (Orchestration)**
    AgentLoop orchestration for mailbox-based agent workflows with graceful
    shutdown coordination and watchdog monitoring.

Session State Pattern
---------------------

Sessions follow a Redux-inspired pattern where state is stored in typed
slices and mutations flow through pure reducers::

    from weakincentives.runtime import Session
    from dataclasses import dataclass, replace

    @dataclass(frozen=True)
    class AgentPlan:
        steps: tuple[str, ...]

    # Create session and register reducer
    session = Session()
    session[AgentPlan].register(AddStep, lambda view, event, context: ...)

    # Query state
    plan = session[AgentPlan].latest()
    all_plans = session[AgentPlan].all()
    active = session[AgentPlan].where(lambda p: p.active)

    # Dispatch events to reducers
    session.dispatch(AddStep(step="implement feature"))

AgentLoop Pattern
-----------------

AgentLoop provides durable request processing with message queues::

    from weakincentives.runtime import AgentLoop, AgentLoopRequest, Session
    from weakincentives.runtime.mailbox import InMemoryMailbox

    class MyLoop(AgentLoop[MyRequest, MyOutput]):
        def prepare(self, request: MyRequest) -> tuple[Prompt[MyOutput], Session]:
            # Build prompt and session for the request
            return prompt, Session()

    requests = InMemoryMailbox[AgentLoopRequest[MyRequest], AgentLoopResult[MyOutput]]()
    loop = MyLoop(adapter=adapter, requests=requests)
    loop.run(max_iterations=100)

Transaction Support
-------------------

Tool execution supports transactional semantics with automatic rollback::

    from weakincentives.runtime import tool_transaction, restore_snapshot

    with tool_transaction(session, resources.context) as snapshot:
        result = execute_tool(...)
        if not result.success:
            restore_snapshot(session, resources.context, snapshot)

Exports
-------

**Session & State:**
    - :class:`Session` - Main state container
    - :class:`SessionProtocol` - Protocol for session implementations
    - :class:`Snapshot` - Immutable session state snapshot
    - :class:`DataEvent` - Union of prompt/tool telemetry events
    - :class:`ReducerContext` - Context passed to reducers
    - :class:`SlicePolicy` - STATE, LOG, or TRANSIENT retention policy

**Reducers:**
    - :func:`append_all` - Ledger semantics (always append)
    - :func:`replace_latest` - Keep only most recent value
    - :func:`replace_latest_by` - Keep latest per derived key
    - :func:`upsert_by` - Update or insert by derived key

**Events:**
    - :class:`Dispatcher` - Event dispatch protocol
    - :class:`InProcessDispatcher` - Synchronous in-process dispatcher
    - :class:`PromptExecuted` - Emitted after prompt evaluation
    - :class:`PromptRendered` - Emitted before prompt dispatch
    - :class:`ToolInvoked` - Emitted after tool execution
    - :class:`TokenUsage` - Token accounting from providers

**Mailbox:**
    - :class:`Mailbox` - Message queue protocol
    - :class:`Message` - Received message with lifecycle methods
    - :class:`InMemoryMailbox` - Thread-safe in-memory implementation
    - :class:`FakeMailbox` - Testing implementation with failure injection
    - :class:`CollectingMailbox` - Stores sent messages for inspection
    - :class:`NullMailbox` - Drops all messages silently

**AgentLoop:**
    - :class:`AgentLoop` - Abstract orchestrator for request processing
    - :class:`AgentLoopConfig` - Configuration for budgets, resources
    - :class:`AgentLoopRequest` - Wrapper for incoming requests
    - :class:`AgentLoopResult` - Response containing output and metadata

**Lifecycle:**
    - :class:`LoopGroup` - Coordinates multiple loops with shutdown
    - :class:`ShutdownCoordinator` - Signal handler for graceful termination
    - :class:`Runnable` - Protocol for loops supporting shutdown
    - :class:`Watchdog` - Detects and terminates stuck workers

**Transactions:**
    - :func:`create_snapshot` - Capture session and resource state
    - :func:`restore_snapshot` - Restore from composite snapshot
    - :func:`tool_transaction` - Context manager for atomic tool execution
    - :class:`CompositeSnapshot` - Combined session and resource snapshot
    - :class:`PendingToolTracker` - Manages in-flight tool transactions

**Logging:**
    - :func:`configure_logging` - Set up structured logging
    - :func:`get_logger` - Get a StructuredLogger instance
    - :class:`StructuredLogger` - JSON-capable logging wrapper

**Dead Letter Queue:**
    - :class:`DeadLetter` - Unprocessable message record
    - :class:`DLQPolicy` - Policy for routing failed messages
    - :class:`DLQConsumer` - Consumer for dead letter messages

Subpackages
-----------

- :mod:`weakincentives.runtime.events` - Event types and dispatching
- :mod:`weakincentives.runtime.mailbox` - Message queue abstractions
- :mod:`weakincentives.runtime.session` - Session state management
- :mod:`weakincentives.runtime.session.slices` - Slice storage backends
"""

from __future__ import annotations

from . import (
    agent_loop,
    agent_loop_types,
    dlq,
    events,
    lease_extender,
    lifecycle,
    mailbox,
    message_handlers,
    session,
    transcript,
    watchdog,
)
from .agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopRequest,
    AgentLoopResult,
)
from .dlq import (
    DeadLetter,
    DLQConsumer,
    DLQPolicy,
)
from .events import (
    ControlDispatcher,
    Dispatcher,
    DispatchResult,
    HandlerFailure,
    InProcessDispatcher,
    PolicyChecked,
    PromptExecuted,
    PromptRendered,
    TelemetryDispatcher,
    TokenUsage,
    ToolInvoked,
)
from .lease_extender import LeaseExtender, LeaseExtenderConfig
from .lifecycle import (
    LoopGroup,
    Runnable,
    ShutdownCoordinator,
    wait_until,
)
from .logging import StructuredLogger, configure_logging, get_logger
from .mailbox import (
    CollectingMailbox,
    FakeMailbox,
    InMemoryMailbox,
    Mailbox,
    MailboxConnectionError,
    MailboxError,
    MailboxFullError,
    Message,
    NullMailbox,
    ReceiptHandleExpiredError,
    SerializationError,
)
from .mailbox_worker import MailboxWorker
from .run_context import RunContext
from .session import (
    DEFAULT_SNAPSHOT_POLICIES,
    DataEvent,
    ReducerContext,
    ReducerContextProtocol,
    ReducerEvent,
    Session,
    SessionProtocol,
    SlicePolicy,
    Snapshot,
    SnapshotProtocol,
    SnapshotRestoreError,
    SnapshotSerializationError,
    TypedReducer,
    append_all,
    build_reducer_context,
    iter_sessions_bottom_up,
    replace_latest,
    replace_latest_by,
    upsert_by,
)
from .snapshotable import Snapshotable
from .transactions import (
    CompositeSnapshot,
    PendingToolExecution,
    PendingToolTracker,
    SnapshotMetadata,
    create_snapshot,
    restore_snapshot,
    tool_transaction,
)
from .transcript import (
    TranscriptEmitter,
    TranscriptEntry,
    TranscriptSummary,
    reconstruct_transcript,
)
from .watchdog import (
    HealthServer,
    Heartbeat,
    Watchdog,
)

__all__ = [
    "DEFAULT_SNAPSHOT_POLICIES",
    "AgentLoop",
    "AgentLoopConfig",
    "AgentLoopRequest",
    "AgentLoopResult",
    "CollectingMailbox",
    "CompositeSnapshot",
    "ControlDispatcher",
    "DLQConsumer",
    "DLQPolicy",
    "DataEvent",
    "DeadLetter",
    "DispatchResult",
    "Dispatcher",
    "FakeMailbox",
    "HandlerFailure",
    "HealthServer",
    "Heartbeat",
    "InMemoryMailbox",
    "InProcessDispatcher",
    "LeaseExtender",
    "LeaseExtenderConfig",
    "LoopGroup",
    "Mailbox",
    "MailboxConnectionError",
    "MailboxError",
    "MailboxFullError",
    "MailboxWorker",
    "Message",
    "NullMailbox",
    "PendingToolExecution",
    "PendingToolTracker",
    "PolicyChecked",
    "PromptExecuted",
    "PromptRendered",
    "ReceiptHandleExpiredError",
    "ReducerContext",
    "ReducerContextProtocol",
    "ReducerEvent",
    "RunContext",
    "Runnable",
    "SerializationError",
    "Session",
    "SessionProtocol",
    "ShutdownCoordinator",
    "SlicePolicy",
    "Snapshot",
    "SnapshotMetadata",
    "SnapshotProtocol",
    "SnapshotRestoreError",
    "SnapshotSerializationError",
    "Snapshotable",
    "StructuredLogger",
    "TelemetryDispatcher",
    "TokenUsage",
    "ToolInvoked",
    "TranscriptEmitter",
    "TranscriptEntry",
    "TranscriptSummary",
    "TypedReducer",
    "Watchdog",
    "agent_loop",
    "agent_loop_types",
    "append_all",
    "build_reducer_context",
    "configure_logging",
    "create_snapshot",
    "dlq",
    "events",
    "get_logger",
    "iter_sessions_bottom_up",
    "lease_extender",
    "lifecycle",
    "mailbox",
    "message_handlers",
    "reconstruct_transcript",
    "replace_latest",
    "replace_latest_by",
    "restore_snapshot",
    "session",
    "tool_transaction",
    "transcript",
    "upsert_by",
    "wait_until",
    "watchdog",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})  # pragma: no cover - convenience shim
