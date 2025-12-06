# Conversation History Manager Specification

## Overview

**Scope:** This document defines the `HistoryManager` abstraction—a mechanism
for curating the conversation history that gets sent to providers during prompt
evaluation. It addresses the unbounded growth of message lists in multi-turn
conversations with tool calls.

**The Problem**

During prompt evaluation, `ConversationRunner` accumulates messages:

```
system → assistant → tool_call → tool_result → assistant → tool_call → ...
```

This list is sent in its entirety on every provider call. Problems arise:

1. **Context limits**: Providers have token budgets (e.g., 128K, 200K). Long
   conversations exceed these limits.
2. **Cost**: Larger contexts cost more tokens and money.
3. **Relevance decay**: Old messages become less relevant to the current task.
4. **Noise**: Verbose tool results can dominate the context window.

**Design Goals**

- Provide a minimal, composable interface for history curation.
- Support common strategies without mandating complex configuration.
- Integrate cleanly with `ConversationRunner` without invasive changes.
- Preserve semantic coherence (e.g., keep tool call/result pairs together).
- Remain pure and deterministic, aligned with the library's philosophy.
- Enable observability through the existing `EventBus` mechanism.

**Non-Goals**

- Automatic provider-specific token limit detection (caller provides limits).
- Persistent conversation storage (sessions handle event persistence).
- Semantic search or embedding-based retrieval (future extension).

## First Principles

Before diving into the interface, consider what history curation fundamentally
requires:

1. **System message preservation**: The system message contains the rendered
   prompt—instructions, tool definitions, context. It must almost always be
   preserved in full.

2. **Recency bias**: Recent messages are generally more relevant than old ones.
   A sliding window over recent history is often sufficient.

3. **Semantic coherence**: Tool calls and their results form atomic units. An
   assistant message requesting a tool call is meaningless without the
   corresponding tool result. Curation must respect these pairings.

4. **Graceful degradation**: When constraints are tight, prefer losing old
   context over truncating the system message or breaking coherence.

## Core Interface

### HistoryManager Protocol

```python
from typing import Protocol, Sequence, Any

from weakincentives.runtime.session.protocols import SessionProtocol


class HistoryManager(Protocol):
    """Protocol for conversation history curation."""

    def curate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        session: SessionProtocol,
    ) -> list[dict[str, Any]]:
        """
        Return a curated view of the conversation history.

        Args:
            messages: The full accumulated message list. The first message
                is typically the system message.
            session: The active session, available for state queries if
                the strategy needs context beyond the raw messages.

        Returns:
            A new list containing the curated messages. May be a subset,
            a transformation, or identical to the input.

        Contract:
            - Must return at least the system message if present.
            - Must preserve tool_call/tool_result pairing integrity.
            - Must not mutate the input sequence.
            - Should be deterministic given the same inputs.
        """
        ...
```

The protocol is intentionally minimal. It takes messages and returns messages.
The `session` parameter allows strategies that need additional context (e.g.,
checking which tools were most recently used) without coupling the interface
to specific session internals.

### CurationContext (Optional Enhancement)

For strategies that need richer context, an optional context object bundles
commonly needed information:

```python
from dataclasses import dataclass
from weakincentives.runtime.session.protocols import SessionProtocol


@dataclass(slots=True, frozen=True)
class CurationContext:
    """Immutable context for history curation decisions."""

    session: SessionProtocol
    """The active session for state queries."""

    preserve_system: bool = True
    """Whether to always preserve the system message."""

    preserve_last_n: int | None = None
    """If set, always preserve at least the last N messages."""
```

Strategies that only need the message list can ignore the context. Strategies
that need other constraints can use it.

## Conversation Entry Model

The history manager operates on typed conversation entries rather than raw
dictionaries. This provides type safety, serialization support, and a
provider-agnostic representation of messages.

### MessageRole Enum

```python
from enum import StrEnum


class MessageRole(StrEnum):
    """Role of a message in the conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
```

### ToolCall Model

```python
@dataclass(slots=True, frozen=True)
class ToolCall:
    """A tool invocation requested by the assistant."""

    id: str
    """Provider-assigned identifier for correlating with tool results."""

    name: str
    """Name of the tool being invoked."""

    arguments: str
    """JSON-encoded arguments for the tool."""
```

### ConversationEntry Model

```python
from uuid import UUID

@dataclass(slots=True, frozen=True)
class ConversationEntry:
    """A single entry in the conversation history."""

    role: MessageRole
    """The role of the message sender."""

    content: str | None
    """Text content of the message. May be None for tool-call-only messages."""

    evaluation_id: UUID
    """Identifier correlating all entries from a single evaluate() call."""

    sequence: int
    """Zero-based position within the evaluation's conversation."""

    created_at: datetime
    """Timestamp when the entry was created."""

    tool_calls: tuple[ToolCall, ...] | None = None
    """Tool calls requested by the assistant. Only present for assistant role."""

    tool_call_id: str | None = None
    """Identifier of the tool call this message responds to. Only for tool role."""

    tool_name: str | None = None
    """Name of the tool that produced this result. Only for tool role."""
```

### Design Rationale

- **Immutable**: Frozen dataclass ensures entries cannot be modified after
  creation.
- **evaluation_id**: Links all entries from a single `evaluate()` call,
  enabling reconstruction of complete conversations from the session.
- **sequence**: Preserves ordering within an evaluation, critical for replay.
- **Provider-agnostic**: No provider-specific fields. Adapters convert to/from
  provider formats.

### Conversion Utilities

```python
def entry_to_provider_message(entry: ConversationEntry) -> dict[str, Any]:
    """Convert a ConversationEntry to provider message format."""
    msg: dict[str, Any] = {
        "role": entry.role.value,
        "content": entry.content,
    }
    if entry.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in entry.tool_calls
        ]
    if entry.tool_call_id:
        msg["tool_call_id"] = entry.tool_call_id
    return msg


def provider_message_to_entry(
    msg: dict[str, Any],
    *,
    evaluation_id: UUID,
    sequence: int,
    created_at: datetime,
) -> ConversationEntry:
    """Convert a provider message to ConversationEntry."""
    role = MessageRole(msg["role"])
    tool_calls = None
    if raw_calls := msg.get("tool_calls"):
        tool_calls = tuple(
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"],
            )
            for tc in raw_calls
        )
    return ConversationEntry(
        role=role,
        content=msg.get("content"),
        evaluation_id=evaluation_id,
        sequence=sequence,
        created_at=created_at,
        tool_calls=tool_calls,
        tool_call_id=msg.get("tool_call_id"),
        tool_name=msg.get("name"),
    )
```

## Session Tracking

**Invariant**: The complete conversation history is always preserved in the
session, regardless of what the history manager sends to the provider.

The history manager affects only the *provider's view*—what gets sent in API
calls. The session maintains the *authoritative record* of all conversation
entries for debugging, auditing, and replay.

### ConversationRecorded Event

```python
@FrozenDataclass()
class ConversationRecorded:
    """Emitted when a conversation entry is added to history."""

    entry: ConversationEntry
    """The recorded conversation entry."""

    session_id: UUID | None
    """Session that recorded this entry."""

    created_at: datetime
    """When the event was emitted."""
```

### Recording Flow

The `ConversationRunner` records entries to the session at two points:

1. **After provider response**: Record the assistant's message (with any
   tool calls).
2. **After tool execution**: Record each tool result message.

```python
@dataclass(slots=True)
class ConversationRunner[OutputT]:
    # ... existing fields ...

    evaluation_id: UUID = field(default_factory=uuid4)
    """Unique identifier for this evaluation."""

    _sequence: int = field(init=False, default=0)
    """Next sequence number for entries."""

    def _record_entry(self, msg: dict[str, Any]) -> ConversationEntry:
        """Record a message to the session and return the entry."""
        entry = provider_message_to_entry(
            msg,
            evaluation_id=self.evaluation_id,
            sequence=self._sequence,
            created_at=datetime.now(UTC),
        )
        self._sequence += 1

        self.bus.publish(ConversationRecorded(
            entry=entry,
            session_id=getattr(self.session, "session_id", None),
            created_at=entry.created_at,
        ))

        return entry

    def _handle_tool_calls(self, message: object, tool_calls: ...) -> None:
        # Record assistant message with tool calls
        assistant_msg = self._serialize_assistant_message(message)
        self._record_entry(assistant_msg)
        self._messages.append(assistant_msg)

        # Execute tools and record results
        for tool_result in self._execute_tools(tool_calls):
            self._record_entry(tool_result)
            self._messages.append(tool_result)
```

### Session Slice for Conversation History

Sessions automatically collect `ConversationRecorded` events into a
`ConversationEntry` slice:

```python
# Default reducer appends entries
session.register_reducer(
    ConversationRecorded,
    lambda slice, event, *, context: (*slice, event.entry),
    slice_type=ConversationEntry,
)
```

### Querying Conversation History

```python
from weakincentives.runtime.session import select_all, select_where

# Get all entries for an evaluation
def get_evaluation_history(
    session: SessionProtocol,
    evaluation_id: UUID,
) -> tuple[ConversationEntry, ...]:
    """Return all conversation entries for a specific evaluation."""
    return select_where(
        session,
        ConversationEntry,
        lambda e: e.evaluation_id == evaluation_id,
    )

# Get complete conversation history
all_entries = select_all(session, ConversationEntry)

# Get entries by role
assistant_entries = select_where(
    session,
    ConversationEntry,
    lambda e: e.role == MessageRole.ASSISTANT,
)
```

### Relationship to HistoryManager

```
┌─────────────────────────────────────────────────────────────────┐
│                     ConversationRunner                          │
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  _messages  │───▶│HistoryManager│───▶│ Provider API     │   │
│  │ (complete)  │    │  (curates)   │    │ (receives subset)│   │
│  └─────────────┘    └──────────────┘    └──────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐    ┌──────────────┐                           │
│  │ _record_    │───▶│   EventBus   │                           │
│  │  entry()    │    │  (publishes) │                           │
│  └─────────────┘    └──────────────┘                           │
│                            │                                    │
└────────────────────────────┼────────────────────────────────────┘
                             ▼
                    ┌──────────────┐
                    │   Session    │
                    │  (records    │
                    │   complete   │
                    │   history)   │
                    └──────────────┘
```

The history manager only affects the path to the provider. The session always
receives the complete, unfiltered conversation history.

## Built-in Strategies

The library provides several composable strategies. Each is a concrete
implementation of `HistoryManager`.

### 1. PassthroughManager

The default: returns messages unchanged.

```python
class PassthroughManager:
    """No-op manager that returns messages unchanged."""

    def curate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        session: SessionProtocol,
    ) -> list[dict[str, Any]]:
        return list(messages)
```

### 2. SlidingWindowManager

Keeps the system message plus the last N messages.

```python
@dataclass(slots=True, frozen=True)
class SlidingWindowManager:
    """Keep system message plus last N messages."""

    window_size: int
    """Number of recent messages to retain (excluding system)."""

    def curate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        session: SessionProtocol,
    ) -> list[dict[str, Any]]:
        if len(messages) <= 1:
            return list(messages)

        system = messages[0] if messages[0].get("role") == "system" else None
        rest = messages[1:] if system else messages

        if len(rest) <= self.window_size:
            return list(messages)

        kept = list(rest[-self.window_size:])
        # Ensure we don't start with an orphaned tool result
        kept = _align_to_coherent_boundary(kept)

        if system:
            return [system] + kept
        return kept
```

### 3. ToolResultTruncator

Truncates verbose tool results while preserving structure.

```python
@dataclass(slots=True, frozen=True)
class ToolResultTruncator:
    """Truncate tool result content to a maximum length."""

    max_length: int = 2000
    """Maximum characters per tool result."""

    suffix: str = "\n... [truncated]"
    """Suffix to append when truncating."""

    def curate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        session: SessionProtocol,
    ) -> list[dict[str, Any]]:
        result = []
        for msg in messages:
            if msg.get("role") == "tool":
                msg = self._truncate_tool_message(msg)
            result.append(msg)
        return result

    def _truncate_tool_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > self.max_length:
            truncated = content[:self.max_length - len(self.suffix)] + self.suffix
            return {**msg, "content": truncated}
        return msg
```

### 4. ComposedManager

Chains multiple managers together.

```python
@dataclass(slots=True, frozen=True)
class ComposedManager:
    """Apply multiple managers in sequence."""

    managers: tuple[HistoryManager, ...]
    """Managers to apply, in order."""

    def curate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        session: SessionProtocol,
    ) -> list[dict[str, Any]]:
        result = list(messages)
        for manager in self.managers:
            result = manager.curate(result, session=session)
        return result
```

## Semantic Coherence

Tool calls and results must stay together. The helper function
`_align_to_coherent_boundary` adjusts a message slice to maintain pairing:

```python
def _align_to_coherent_boundary(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Adjust message list to start at a coherent boundary.

    Removes leading tool results that lack their corresponding tool calls,
    ensuring the conversation makes semantic sense.
    """
    if not messages:
        return messages

    # Skip orphaned tool results at the start
    start = 0
    while start < len(messages) and messages[start].get("role") == "tool":
        start += 1

    return messages[start:]
```

A more sophisticated implementation might also ensure that assistant messages
with `tool_calls` have all their results present, but the simple version
handles the common case.

## Integration with ConversationRunner

The `HistoryManager` integrates into `ConversationRunner` as an optional
dependency:

```python
@dataclass(slots=True)
class ConversationRunner[OutputT]:
    # ... existing fields ...

    history_manager: HistoryManager | None = None
    """Optional manager for curating conversation history."""

    def _issue_provider_request(self) -> object:
        # ... existing throttle/retry logic ...

        messages = self._messages
        if self.history_manager is not None:
            messages = self.history_manager.curate(
                messages,
                session=self.session,
            )

        return self.call_provider(
            messages,
            self._tool_specs,
            self._next_tool_choice if self._tool_specs else None,
            self.response_format,
        )
```

The curation happens just before each provider call, ensuring the manager
sees the complete accumulated history and can make informed decisions.

### Adapter Integration

Adapters expose the history manager through their `evaluate` method:

```python
class ProviderAdapter(ABC):
    @abstractmethod
    def evaluate(
        self,
        prompt: Prompt[OutputT],
        *,
        bus: EventBus,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        visibility_overrides: Mapping[SectionPath, SectionVisibility] | None = None,
        history_manager: HistoryManager | None = None,  # NEW
    ) -> PromptResponse[OutputT]:
        ...
```

## Observability

When curation occurs, the manager can emit events for debugging and monitoring:

```python
@FrozenDataclass()
class HistoryCurated:
    """Emitted when conversation history is curated."""

    prompt_name: str
    original_count: int
    curated_count: int
    strategy: str
    session_id: UUID | None
    created_at: datetime
```

The event is optional—simple strategies may skip emission.

### Event Emission Pattern

```python
@dataclass(slots=True, frozen=True)
class ObservableSlidingWindowManager:
    """SlidingWindowManager with event emission."""

    inner: SlidingWindowManager
    bus: EventBus
    prompt_name: str

    def curate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        session: SessionProtocol,
    ) -> list[dict[str, Any]]:
        result = self.inner.curate(messages, session=session)

        self.bus.publish(HistoryCurated(
            prompt_name=self.prompt_name,
            original_count=len(messages),
            curated_count=len(result),
            strategy="sliding_window",
            session_id=getattr(session, "session_id", None),
            created_at=datetime.now(UTC),
        ))

        return result
```

This pattern wraps any manager with observability without modifying the core
implementation.

## Usage Examples

### Basic: Sliding Window

```python
from weakincentives.history import SlidingWindowManager

manager = SlidingWindowManager(window_size=20)

response = adapter.evaluate(
    prompt,
    bus=bus,
    session=session,
    history_manager=manager,
)
```

### Sliding Window with Truncation

```python
from weakincentives.history import (
    ComposedManager,
    SlidingWindowManager,
    ToolResultTruncator,
)

manager = ComposedManager(managers=(
    ToolResultTruncator(max_length=1000),  # First: truncate verbose results
    SlidingWindowManager(window_size=50),  # Then: keep recent messages
))

response = adapter.evaluate(
    prompt,
    bus=bus,
    session=session,
    history_manager=manager,
)
```

### Custom Strategy

```python
class KeepToolCallsForActiveTools:
    """Only keep tool results for tools still in the prompt."""

    def curate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        session: SessionProtocol,
    ) -> list[dict[str, Any]]:
        # Get active tool names from session
        active_tools = self._get_active_tool_names(session)

        result = []
        for msg in messages:
            if msg.get("role") == "tool":
                tool_name = self._extract_tool_name(msg)
                if tool_name not in active_tools:
                    continue  # Skip results for inactive tools
            result.append(msg)

        return _align_to_coherent_boundary(result)
```

## Design Decisions and Rationale

### Why a Protocol, Not a Base Class?

Protocols allow duck typing. A simple function wrapped in a class works:

```python
class MyManager:
    def curate(self, messages, *, session):
        return messages[-10:]  # Keep last 10
```

No inheritance required. This follows the library's preference for composition.

### Why Curate Before Provider Call, Not After Tool Execution?

Curating before each provider call:

1. Sees the complete current state
2. Allows strategies to consider the full context
3. Keeps the logic centralized in one place
4. Avoids modifying `_messages` (the curated list is ephemeral)

The original `_messages` list remains intact for debugging and event emission.

### Why Composition Over Configuration?

Instead of one manager with many options:

```python
# NOT this
manager = HistoryManager(
    window_size=20,
    truncate_tools=True,
    ...
)
```

We use composition:

```python
# This
manager = ComposedManager(managers=(
    ToolResultTruncator(max_length=1000),
    SlidingWindowManager(window_size=20),
))
```

Composition is more flexible, testable, and follows the library's patterns.

## Module Structure

```
src/weakincentives/history/
├── __init__.py           # Public exports
├── _protocol.py          # HistoryManager protocol
├── _context.py           # CurationContext dataclass
├── _entry.py             # ConversationEntry, ToolCall, MessageRole
├── _conversion.py        # entry_to_provider_message, provider_message_to_entry
├── _coherence.py         # _align_to_coherent_boundary helper
├── _managers.py          # Built-in manager implementations
└── _events.py            # HistoryCurated, ConversationRecorded events
```

## Testing Requirements

- Unit tests for each built-in manager with various message configurations.
- Tests for semantic coherence: orphaned tool results are removed.
- Tests for composition: multiple managers chain correctly.
- Tests for edge cases: empty messages, no system message, window of zero.
- Integration tests verifying ConversationRunner uses the manager correctly.
- Tests for ConversationEntry serialization round-trip.
- Tests verifying session records complete history even when manager curates.
- Tests for evaluation_id correlation: all entries from one evaluate() share ID.
- Tests for sequence ordering within an evaluation.

## Future Extensions

The following are explicitly out of scope for the initial implementation but
could be added later:

1. **Token budget manager**: Keep messages that fit within a token budget,
   prioritizing recent messages. Would require a `TokenEstimator` protocol
   for counting tokens.

2. **Summarization**: Replace old messages with an LLM-generated summary.
   Would require provider access, similar to `PromptOptimizer`.

3. **Importance scoring**: Use embeddings or heuristics to keep the most
   relevant messages regardless of recency.

4. **Streaming curation**: Curate incrementally as messages arrive rather
   than on each provider call.

5. **Provider-aware managers**: Automatically query provider for limits and
   adjust strategies accordingly.

## Limitations and Caveats

- **No cross-session continuity**: The manager operates on a single
  conversation's messages. Long-running agents that span sessions need
  external orchestration.

- **Summarization requires provider calls**: Strategies that summarize old
  context would need adapter access and would incur additional API costs.

- **Alpha stability**: The interface may evolve. No backward compatibility
  shims will be added.

## Related Specifications

| Spec | Relationship |
|------|--------------|
| `SESSIONS.md` | Sessions track events; managers curate provider input |
| `ADAPTERS.md` | Adapters integrate managers into evaluation |
| `PROMPT_OPTIMIZERS.md` | Similar pattern for prompt transformation |
| `TOOLS.md` | Tool results are subject to curation |
| `THREAD_SAFETY.md` | Managers must be thread-safe if shared |
