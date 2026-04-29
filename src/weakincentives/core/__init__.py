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

"""WINK spine: the load-bearing core of weakincentives.

This package is the only required install. Everything else (provider
adapters, resource registries, evaluation, debug bundles, …) lives in
optional extras that depend on the spine and never the other way around.

See ``specs/SPINE.md`` for the full design.
"""

from __future__ import annotations

from .errors import (
    BudgetExceededError,
    ContractError,
    DeadlineExceededError,
    PolicyDeniedError,
    PromptError,
    SessionError,
    SnapshotError,
    ToolError,
    TransactionError,
    WinkError,
)
from .prompt import (
    MarkdownSection,
    Prompt,
    RenderedPrompt,
    Section,
)
from .protocols import (
    Budget,
    Deadline,
    EventListener,
    PromptResponse,
    ProviderAdapter,
    ResourceProvider,
    ToolCall,
    Usage,
)
from .session import (
    Append,
    PromptExecuted,
    PromptRendered,
    Replace,
    Session,
    SliceAccessor,
    SliceOp,
    ToolInvoked,
    TypedReducer,
    reducer,
)
from .snapshot import (
    Snapshot,
    Snapshotable,
    TypeRegistry,
    capture,
    restore,
)
from .tool import (
    Tool,
    ToolContext,
    ToolHandler,
    ToolPolicy,
    ToolResult,
)
from .transactions import (
    PendingToolTracker,
    execute_tool,
    tool_transaction,
)

__all__ = [
    "Append",
    "Budget",
    "BudgetExceededError",
    "ContractError",
    "Deadline",
    "DeadlineExceededError",
    "EventListener",
    "MarkdownSection",
    "PendingToolTracker",
    "PolicyDeniedError",
    "Prompt",
    "PromptError",
    "PromptExecuted",
    "PromptRendered",
    "PromptResponse",
    "ProviderAdapter",
    "RenderedPrompt",
    "Replace",
    "ResourceProvider",
    "Section",
    "Session",
    "SessionError",
    "SliceAccessor",
    "SliceOp",
    "Snapshot",
    "SnapshotError",
    "Snapshotable",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolError",
    "ToolHandler",
    "ToolInvoked",
    "ToolPolicy",
    "ToolResult",
    "TransactionError",
    "TypeRegistry",
    "TypedReducer",
    "Usage",
    "WinkError",
    "capture",
    "execute_tool",
    "reducer",
    "restore",
    "tool_transaction",
]
