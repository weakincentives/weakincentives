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
├── _coherence.py         # _align_to_coherent_boundary helper
├── _managers.py          # Built-in manager implementations
└── _events.py            # HistoryCurated event
```

## Testing Requirements

- Unit tests for each built-in manager with various message configurations.
- Tests for semantic coherence: orphaned tool results are removed.
- Tests for composition: multiple managers chain correctly.
- Tests for edge cases: empty messages, no system message, window of zero.
- Integration tests verifying ConversationRunner uses the manager correctly.

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
