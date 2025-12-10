# Conversation State Specification

## Purpose

Enable session snapshots to capture full conversation history, allowing prompt
evaluations to resume after process restarts. This specification covers the
`ConversationTurn` dataclass, session integration, snapshot serialization, and
the resume flow for adapters.

## Problem Statement

Currently, conversation messages live exclusively in `InnerLoop._messages`, a
transient list that exists only for the duration of a single `evaluate()` call.
When a process terminates mid-evaluation—especially during a tool call—all
conversation context is lost:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Current Architecture                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Adapter.evaluate()                                                        │
│       │                                                                     │
│       ▼                                                                     │
│   InnerLoop._messages: list[dict]  ◄── Transient, lost on restart          │
│       │                                                                     │
│       ▼                                                                     │
│   Session (slices only)            ◄── Persisted via snapshot              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

Session snapshots capture accumulated dataclass state but not the raw message
history required to resume provider conversations.

## Guiding Principles

- **Single source of truth**: Conversation state lives in the session, not the
  adapter loop.
- **Snapshot-complete**: A snapshot contains everything needed to resume,
  including messages.
- **Provider-agnostic messages**: Stored format is provider-neutral; adapters
  translate on resume.
- **Incremental capture**: Each turn appends to session state; no bulk rewrites.
- **Backward compatible**: Existing code paths continue to work; resume is
  opt-in.

## Core Data Model

### ConversationMessage

Provider-neutral representation of a single message:

```python
@FrozenDataclass()
class ConversationMessage:
    """A single message in a conversation."""

    role: Literal["system", "assistant", "user", "tool"]
    content: str
    tool_calls: tuple[ToolCallRecord, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None  # Tool name for role="tool"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    message_id: UUID = field(default_factory=uuid4)
```

### ToolCallRecord

Captures tool invocation metadata for assistant messages:

```python
@FrozenDataclass()
class ToolCallRecord:
    """Record of a tool call made by the assistant."""

    call_id: str
    name: str
    arguments: str  # JSON-encoded arguments
```

### ConversationTurn

Groups a provider response with any resulting tool messages:

```python
@FrozenDataclass()
class ConversationTurn:
    """A single turn in the conversation: assistant response + tool results."""

    turn_number: int
    assistant_message: ConversationMessage
    tool_messages: tuple[ConversationMessage, ...] = ()
    provider_payload: Mapping[str, JSONValue] | None = None
    evaluation_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

### ConversationState

Top-level container stored as a session slice:

```python
@FrozenDataclass()
class ConversationState:
    """Complete conversation history for an evaluation."""

    evaluation_id: str
    prompt_ns: str
    prompt_key: str
    system_message: ConversationMessage
    turns: tuple[ConversationTurn, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    state_id: UUID = field(default_factory=uuid4)
```

## Session Integration

### Slice Registration

`ConversationState` is stored as a session slice using `replace_latest`:

```python
from weakincentives.runtime.session import replace_latest

session.mutate(ConversationState).register(ConversationState, replace_latest)
```

This ensures only the current conversation is retained per evaluation.

### Turn Recording

The `InnerLoop` records each turn after tool execution completes:

```python
def _record_turn(self, assistant_msg: ProviderMessage, tool_results: list[dict]) -> None:
    turn = ConversationTurn(
        turn_number=len(self._conversation_state.turns) + 1,
        assistant_message=self._to_conversation_message(assistant_msg),
        tool_messages=tuple(
            self._to_conversation_message(msg) for msg in tool_results
        ),
        provider_payload=self._provider_payload,
        evaluation_id=self._evaluation_id,
    )

    updated_state = replace(
        self._conversation_state,
        turns=(*self._conversation_state.turns, turn),
    )
    self.config.session.mutate(ConversationState).seed(updated_state)
```

### State Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Proposed Architecture                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Adapter.evaluate()                                                        │
│       │                                                                     │
│       ▼                                                                     │
│   InnerLoop                                                                 │
│       │                                                                     │
│       ├──► _messages (working copy for provider calls)                      │
│       │                                                                     │
│       └──► Session.mutate(ConversationState).seed(...)                      │
│                   │                                                         │
│                   ▼                                                         │
│            ConversationState slice  ◄── Included in snapshot               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Snapshot Serialization

`ConversationState` serializes naturally via the existing snapshot mechanism:

```python
snapshot = session.snapshot()
json_str = snapshot.to_json()

# ConversationState appears in slices:
# {
#   "slices": [
#     {
#       "slice_type": "weakincentives.runtime.conversation:ConversationState",
#       "item_type": "weakincentives.runtime.conversation:ConversationState",
#       "items": [{ ... serialized state ... }]
#     }
#   ]
# }
```

### Serialization Considerations

- `provider_payload` contains arbitrary provider data; stored as `JSONValue`
- `tool_calls` arguments remain JSON-encoded strings to preserve fidelity
- All timestamps are timezone-aware ISO 8601

## Resume Flow

### Adapter Interface

Adapters gain an optional `resume` parameter:

```python
class ProviderAdapter(ABC):
    @abstractmethod
    def evaluate(
        self,
        prompt: Prompt[OutputT],
        *,
        # ... existing parameters ...
        resume_from: ConversationState | None = None,  # NEW
    ) -> PromptResponse[OutputT]: ...
```

### Resume Behavior

When `resume_from` is provided:

1. **Skip rendering** if prompt matches `resume_from.prompt_ns` and
   `resume_from.prompt_key`
2. **Reconstruct messages** from `ConversationState.turns`
3. **Continue from last turn** by issuing the next provider request

```python
def evaluate(
    self,
    prompt: Prompt[OutputT],
    *,
    resume_from: ConversationState | None = None,
    **kwargs,
) -> PromptResponse[OutputT]:
    if resume_from is not None:
        return self._resume_evaluation(prompt, resume_from, **kwargs)
    return self._fresh_evaluation(prompt, **kwargs)

def _resume_evaluation(
    self,
    prompt: Prompt[OutputT],
    state: ConversationState,
    **kwargs,
) -> PromptResponse[OutputT]:
    # Validate prompt identity
    if (prompt.ns, prompt.key) != (state.prompt_ns, state.prompt_key):
        raise ValueError("Cannot resume with mismatched prompt")

    # Reconstruct message list
    messages = self._state_to_messages(state)

    # Create InnerLoop with reconstructed messages
    inputs = InnerLoopInputs[OutputT](
        # ... other fields ...
        initial_messages=messages,
    )
    # ... continue execution ...
```

### Message Reconstruction

Convert `ConversationState` back to provider message format:

```python
def _state_to_messages(self, state: ConversationState) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": state.system_message.content}
    ]

    for turn in state.turns:
        # Assistant message
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": turn.assistant_message.content,
        }
        if turn.assistant_message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in turn.assistant_message.tool_calls
            ]
        messages.append(assistant_msg)

        # Tool result messages
        for tool_msg in turn.tool_messages:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_msg.tool_call_id,
                "content": tool_msg.content,
            })

    return messages
```

## MainLoop Integration

### Checkpoint Strategy

`MainLoop` can implement automatic checkpointing:

```python
class MainLoop(ABC, Generic[UserRequestT, OutputT]):
    def __init__(
        self,
        *,
        checkpoint_handler: CheckpointHandler | None = None,
        **kwargs,
    ) -> None:
        self._checkpoint_handler = checkpoint_handler
        # ...

    def execute(self, request: UserRequestT) -> PromptResponse[OutputT]:
        session = self.create_session()

        # Check for existing checkpoint
        if self._checkpoint_handler:
            checkpoint = self._checkpoint_handler.load(request)
            if checkpoint:
                session.mutate().rollback(checkpoint.snapshot)
                state = session.query(ConversationState).latest()
                return self._adapter.evaluate(
                    self.create_prompt(request),
                    resume_from=state,
                    session=session,
                    # ...
                )

        # Fresh execution with checkpointing
        # ...
```

### CheckpointHandler Protocol

```python
class CheckpointHandler(Protocol):
    def load(self, request: UserRequestT) -> Checkpoint | None:
        """Load checkpoint for request, if exists."""
        ...

    def save(self, request: UserRequestT, snapshot: Snapshot) -> None:
        """Persist checkpoint after each turn."""
        ...

    def clear(self, request: UserRequestT) -> None:
        """Remove checkpoint after successful completion."""
        ...
```

## Tool Idempotency

Resume safety depends on tool idempotency. Tools must handle re-execution:

### Idempotency Patterns

1. **Natural idempotency**: Read-only tools (search, fetch) are safe by default

2. **Idempotency keys**: Mutating tools use `call_id` for deduplication:

   ```python
   def write_file_handler(
       params: WriteFileParams,
       *,
       context: ToolContext,
   ) -> ToolResult[WriteFileResult]:
       call_id = context.call_id
       if was_already_executed(call_id):
           return cached_result(call_id)
       # ... execute and cache ...
   ```

3. **Compensation**: Some tools may need undo logic if partial execution occurred

### ToolContext Enhancement

```python
@FrozenDataclass()
class ToolContext:
    # ... existing fields ...
    call_id: str  # Already present, used for idempotency
    is_resume: bool = False  # NEW: indicates resumed execution
```

## Error Handling

### Resume Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| `ConversationStateMismatch` | Prompt changed since checkpoint | Clear checkpoint, restart |
| `SnapshotRestoreError` | Corrupted or incompatible snapshot | Clear checkpoint, restart |
| `ToolIdempotencyError` | Tool cannot safely re-execute | Manual intervention |

### Partial Turn Recovery

If a turn partially completed (some tools ran, others didn't):

1. Snapshot captures state before the incomplete turn
2. On resume, the entire turn re-executes
3. Completed tools return cached results via idempotency
4. Incomplete tools execute normally

## Usage Example

### Basic Resume

```python
from weakincentives.runtime.session import Session, Snapshot
from weakincentives.runtime.conversation import ConversationState

# Save checkpoint
def save_checkpoint(session: Session, path: str) -> None:
    snapshot = session.snapshot()
    Path(path).write_text(snapshot.to_json())

# Resume from checkpoint
def resume_evaluation(
    adapter: ProviderAdapter[OutputT],
    prompt: Prompt[OutputT],
    checkpoint_path: str,
) -> PromptResponse[OutputT]:
    snapshot = Snapshot.from_json(Path(checkpoint_path).read_text())

    session = Session(bus=InProcessEventBus())
    session.mutate().rollback(snapshot)

    state = session.query(ConversationState).latest()
    if state is None:
        raise ValueError("No conversation state in checkpoint")

    return adapter.evaluate(
        prompt,
        session=session,
        resume_from=state,
    )
```

### With MainLoop

```python
class FileCheckpointHandler:
    def __init__(self, checkpoint_dir: Path) -> None:
        self._dir = checkpoint_dir

    def load(self, request: ReviewRequest) -> Checkpoint | None:
        path = self._dir / f"{request.id}.json"
        if path.exists():
            snapshot = Snapshot.from_json(path.read_text())
            return Checkpoint(snapshot=snapshot)
        return None

    def save(self, request: ReviewRequest, snapshot: Snapshot) -> None:
        path = self._dir / f"{request.id}.json"
        path.write_text(snapshot.to_json())

    def clear(self, request: ReviewRequest) -> None:
        path = self._dir / f"{request.id}.json"
        path.unlink(missing_ok=True)


loop = CodeReviewLoop(
    adapter=adapter,
    bus=bus,
    checkpoint_handler=FileCheckpointHandler(Path("/tmp/checkpoints")),
)
```

## Migration

### Existing Code

Existing code continues to work unchanged. `ConversationState` recording is
additive; adapters that don't use `resume_from` behave identically.

### Opt-In Recording

To enable conversation recording without checkpointing:

```python
from weakincentives.runtime.conversation import enable_conversation_recording

session = Session(bus=bus)
enable_conversation_recording(session)  # Registers reducer

response = adapter.evaluate(prompt, session=session)

# Conversation is now in session
state = session.query(ConversationState).latest()
```

## Limitations

- **Provider compatibility**: Resume assumes provider accepts reconstructed
  message history; some providers may have session affinity
- **Token drift**: Reconstructed prompts may tokenize differently than originals
- **No mid-tool resume**: Recovery granularity is per-turn, not per-tool within
  a turn
- **Manual checkpoint persistence**: Library provides serialization; storage is
  caller's responsibility
- **Tool result size**: Large tool outputs increase snapshot size

## File Locations

| Component | Location |
|-----------|----------|
| Data model | `src/weakincentives/runtime/conversation.py` |
| Session integration | `src/weakincentives/runtime/session/` |
| Adapter changes | `src/weakincentives/adapters/shared.py` |
| MainLoop integration | `src/weakincentives/runtime/main_loop.py` |

## Related Specifications

- `specs/SESSIONS.md` - Session state, snapshots, reducers
- `specs/ADAPTERS.md` - Provider adapter protocol
- `specs/MAIN_LOOP.md` - MainLoop orchestration
- `specs/TOOLS.md` - Tool handler patterns
