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

from . import events, mailbox, main_loop, session
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
from .execution_state import (
    CompositeSnapshot,
    ExecutionState,
    SnapshotMetadata,
)
from .logging import StructuredLogger, configure_logging, get_logger
from .mailbox import (
    ImmediateReply,
    InMemoryMailbox,
    InMemoryMessage,
    InMemoryReply,
    InMemoryReplyChannel,
    InMemoryReplyStore,
    Mailbox,
    MailboxError,
    MailboxFullError,
    Message,
    MessageData,
    NeverResolvingReply,
    NoReplyChannelError,
    NullMailbox,
    ReceiptHandleExpiredError,
    RecordingMailbox,
    Reply,
    ReplyAlreadySentError,
    ReplyCancelledError,
    ReplyChannel,
    ReplyEntry,
    ReplyError,
    ReplyExpectedError,
    ReplyState,
    ReplyStore,
    ReplyTimeoutError,
    SerializationError,
)
from .main_loop import (
    MainLoop,
    MainLoopConfig,
    MainLoopRequest,
    MainLoopResult,
)
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

__all__ = [
    "DEFAULT_SNAPSHOT_POLICIES",
    "CompositeSnapshot",
    "ControlDispatcher",
    "DataEvent",
    "DispatchResult",
    "Dispatcher",
    "ExecutionState",
    "HandlerFailure",
    "ImmediateReply",
    "InMemoryMailbox",
    "InMemoryMessage",
    "InMemoryReply",
    "InMemoryReplyChannel",
    "InMemoryReplyStore",
    "InProcessDispatcher",
    "Mailbox",
    "MailboxError",
    "MailboxFullError",
    "MainLoop",
    "MainLoopConfig",
    "MainLoopRequest",
    "MainLoopResult",
    "Message",
    "MessageData",
    "NeverResolvingReply",
    "NoReplyChannelError",
    "NullMailbox",
    "PromptExecuted",
    "PromptRendered",
    "ReceiptHandleExpiredError",
    "RecordingMailbox",
    "ReducerContext",
    "ReducerContextProtocol",
    "ReducerEvent",
    "Reply",
    "ReplyAlreadySentError",
    "ReplyCancelledError",
    "ReplyChannel",
    "ReplyEntry",
    "ReplyError",
    "ReplyExpectedError",
    "ReplyState",
    "ReplyStore",
    "ReplyTimeoutError",
    "SerializationError",
    "Session",
    "SessionProtocol",
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
    "TypedReducer",
    "append_all",
    "build_reducer_context",
    "configure_logging",
    "events",
    "get_logger",
    "iter_sessions_bottom_up",
    "mailbox",
    "main_loop",
    "replace_latest",
    "replace_latest_by",
    "session",
    "upsert_by",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})  # pragma: no cover - convenience shim
