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

"""Runtime primitives for :mod:`weakincentives`."""

from __future__ import annotations

from . import (
    dlq,
    events,
    lease_extender,
    lifecycle,
    mailbox,
    main_loop,
    main_loop_types,
    message_handlers,
    session,
    watchdog,
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
from .main_loop import (
    MainLoop,
    MainLoopConfig,
    MainLoopRequest,
    MainLoopResult,
)
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
    TypedReducerProtocol,
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
from .watchdog import (
    HealthServer,
    Heartbeat,
    Watchdog,
)

__all__ = [
    "DEFAULT_SNAPSHOT_POLICIES",
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
    "MainLoop",
    "MainLoopConfig",
    "MainLoopRequest",
    "MainLoopResult",
    "Message",
    "NullMailbox",
    "PendingToolExecution",
    "PendingToolTracker",
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
    "TypedReducerProtocol",
    "Watchdog",
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
    "main_loop",
    "main_loop_types",
    "message_handlers",
    "replace_latest",
    "replace_latest_by",
    "restore_snapshot",
    "session",
    "tool_transaction",
    "upsert_by",
    "wait_until",
    "watchdog",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})  # pragma: no cover - convenience shim
