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

"""weakincentives — the WINK toolkit for building reliable agents.

The base install ships only the **spine** (:mod:`weakincentives.core`):
hierarchical typed prompts that bundle tools, event-sourced sessions with
pure reducers, snapshot/restore with JSON persistence, transactional tool
execution, and a small set of extension protocols.

Provider adapters, resource registries, debug bundles, evaluation, and
other higher-level features live in optional extras (`pip install
weakincentives[claude]`, etc.) that depend on this spine.

Symbols re-exported from :mod:`weakincentives.core` are the public,
semver-frozen surface of the package.
"""

from __future__ import annotations

from .core import (
    Append,
    Budget,
    BudgetExceededError,
    ContractError,
    Deadline,
    DeadlineExceededError,
    EventListener,
    MarkdownSection,
    PendingToolTracker,
    PolicyDeniedError,
    Prompt,
    PromptError,
    PromptExecuted,
    PromptRendered,
    PromptResponse,
    ProviderAdapter,
    RenderedPrompt,
    Replace,
    ResourceProvider,
    Section,
    SectionVisibility,
    Session,
    SessionError,
    SliceAccessor,
    SliceOp,
    Snapshot,
    Snapshotable,
    SnapshotError,
    Tool,
    ToolCall,
    ToolContext,
    ToolError,
    ToolHandler,
    ToolInvoked,
    ToolPolicy,
    ToolResult,
    TransactionError,
    TypedReducer,
    TypeRegistry,
    Usage,
    WinkError,
    capture,
    execute_tool,
    reducer,
    restore,
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
    "SectionVisibility",
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
