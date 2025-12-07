# Main Loop Specification

## Purpose

`MainLoop` standardizes agent workflow orchestration: receive request, build
prompt, evaluate, handle visibility expansion, publish result. Implementations
define only the domain-specific factories.

## Guiding Principles

- **Event-driven**: Requests arrive via bus; results return the same way
- **Factory-based**: Subclasses own prompt and session construction
- **Visibility-transparent**: Expansion exceptions retry automatically
- **Type-safe**: Generic parameters ensure request-prompt alignment

## Core Components

### MainLoop

```python
class MainLoop(ABC, Generic[UserRequestT, OutputT]):
    def __init__(
        self,
        *,
        adapter: ProviderAdapter[OutputT],
        bus: EventBus,
        config: MainLoopConfig | None = None,
    ) -> None: ...

    @abstractmethod
    def create_prompt(self, request: UserRequestT) -> Prompt[OutputT]: ...

    @abstractmethod
    def create_session(self) -> Session: ...

    def execute(self, request: UserRequestT) -> PromptResponse[OutputT]: ...
```

### Events

```python
@FrozenDataclass()
class MainLoopRequest(Generic[UserRequestT]):
    request: UserRequestT
    budget: Budget | None = None       # Overrides config default
    deadline: Deadline | None = None   # Overrides config default
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@FrozenDataclass()
class MainLoopCompleted(Generic[OutputT]):
    request_id: UUID
    response: PromptResponse[OutputT]
    session_id: UUID
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@FrozenDataclass()
class MainLoopFailed:
    request_id: UUID
    error: Exception
    session_id: UUID | None
    failed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

### Configuration

```python
@FrozenDataclass()
class MainLoopConfig:
    deadline: Deadline | None = None
    budget: Budget | None = None
    parse_output: bool = True
```

Request-level `budget` and `deadline` override config defaults. A fresh
`BudgetTracker` is created per execution.

## Execution

```
Request ──▶ Session ──▶ Prompt ──▶ Evaluate ──┬──▶ Result
                                              │
                                              ▼
                                     VisibilityExpansion
                                              │
                                              └──▶ (retry)
```

1. Receive `MainLoopRequest` via bus or direct `execute()` call
1. Create session via `create_session()`
1. Create prompt via `create_prompt(request)`
1. Evaluate with adapter
1. On `VisibilityExpansionRequired`: accumulate overrides, retry step 4
1. Publish `MainLoopCompleted` or `MainLoopFailed`

### Visibility Handling

```python
def execute(self, request: UserRequestT) -> PromptResponse[OutputT]:
    session = self.create_session()
    prompt = self.create_prompt(request)
    visibility_overrides: dict[SectionPath, SectionVisibility] = {}

    while True:
        try:
            return self._adapter.evaluate(
                prompt,
                bus=self._bus,
                session=session,
                visibility_overrides=visibility_overrides,
                budget=self._effective_budget,
                deadline=self._effective_deadline,
            )
        except VisibilityExpansionRequired as e:
            visibility_overrides.update(e.requested_overrides)
```

Overrides accumulate; session persists across retries; prompt is not recreated.

## Usage

### Bus-Driven

```python
loop = MyMainLoop(adapter=adapter, bus=bus)

# Subscribe to concrete type (generic params are compile-time only)
bus.subscribe(MainLoopRequest, loop.handle_request)

# With request-specific constraints
bus.publish(MainLoopRequest(
    request=MyRequest(...),
    budget=Budget(max_total_tokens=10000),
))
```

**Note:** `InProcessEventBus` dispatches by `type(event)`, not generic alias.
`MainLoopRequest[T]` is for static type checking; at runtime all events are
`MainLoopRequest`. For multiple loop types on one bus, filter by request type
in the handler or use separate buses.

### Direct

```python
response = loop.execute(MyRequest(...))
```

## Implementation

```python
class CodeReviewLoop(MainLoop[ReviewRequest, ReviewResult]):
    def __init__(self, *, adapter: ProviderAdapter[ReviewResult], bus: EventBus) -> None:
        super().__init__(adapter=adapter, bus=bus)
        self._template = PromptTemplate[ReviewResult](
            ns="reviews",
            key="code-review",
            sections=[...],
        )

    def create_prompt(self, request: ReviewRequest) -> Prompt[ReviewResult]:
        return Prompt(self._template).bind(ReviewParams.from_request(request))

    def create_session(self) -> Session:
        return Session(bus=self._bus, tags={"loop": "code-review"})
```

### With Reducers

```python
def create_session(self) -> Session:
    session = Session(bus=self._bus)
    session.mutate(Plan).register(SetupPlan, plan_reducer)
    return session
```

### With Progressive Disclosure

```python
def create_prompt(self, request: Request) -> Prompt[Output]:
    return Prompt(PromptTemplate[Output](
        ns="agent",
        key="task",
        sections=[
            MarkdownSection[Params](
                title="Reference",
                template="...",
                summary="Reference available.",
                visibility=SectionVisibility.SUMMARY,
                key="reference",
            ),
        ],
    )).bind(Params.from_request(request))
```

## Error Handling

| Exception | Behavior |
|-----------|----------|
| `VisibilityExpansionRequired` | Retry with updated overrides |
| All others | Publish `MainLoopFailed`, re-raise |

## Code Reviewer Integration

The code reviewer agent uses `MainLoop` with these specifics:

**Session reuse:** A single session is created at loop construction and reused
across all `execute()` calls. State accumulates across turns.

**Auto-optimization:** The explicit `optimize` command is removed. Before each
evaluation, the loop checks for `WorkspaceDigest` in session state. If absent,
optimization runs automatically.

```python
def execute(self, request: UserRequestT) -> PromptResponse[OutputT]:
    if not self._session.query(WorkspaceDigest).exists():
        self._run_optimization()
    # ... proceed with evaluation
```

**Default deadline:** All requests receive a 5-minute deadline unless overridden
at the request level.

```python
config = MainLoopConfig(
    deadline=Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5)),
)
```

## Limitations

- Synchronous execution
- One adapter per loop instance
- No mid-execution cancellation
- Events local to process
