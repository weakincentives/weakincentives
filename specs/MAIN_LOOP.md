# Main Loop Specification

## Purpose

The `MainLoop` abstraction provides a standardized orchestration pattern for
executing background agent workflows. It encapsulates the request-response
lifecycle, visibility override handling, session management, and event-driven
execution so implementations can focus on domain-specific prompt and session
construction.

## Guiding Principles

- **Event-driven orchestration**: Execution is triggered by publishing request
  events to the bus; results flow back through the same channel.
- **Factory-based customization**: Subclasses define prompt and session
  construction through abstract factory methods.
- **Visibility override transparency**: The loop automatically handles
  `VisibilityExpansionRequired` exceptions, retrying with expanded sections.
- **Type-safe request routing**: The generic type parameter ensures request
  payloads match prompt expectations.
- **Single responsibility**: The main loop owns the execution lifecycle; prompt
  content and session state live in their respective abstractions.

## Core Components

### MainLoop

`MainLoop[UserRequestT]` is an abstract base class parameterized by the user
request type:

```python
class MainLoop(ABC, Generic[UserRequestT, OutputT]):
    def __init__(
        self,
        *,
        adapter: ProviderAdapter[OutputT],
        bus: EventBus,
    ) -> None: ...

    @property
    def event_bus(self) -> EventBus: ...

    @abstractmethod
    def create_prompt(self, request: UserRequestT) -> Prompt[OutputT]:
        """Factory method to construct a prompt from a user request.

        Implementations should build the appropriate PromptTemplate and bind
        parameters derived from the request object.
        """
        ...

    @abstractmethod
    def create_session(self) -> Session:
        """Factory method to provision a session for the execution.

        Implementations may configure custom reducers, tags, or parent
        relationships as needed.
        """
        ...

    def execute(self, request: UserRequestT) -> PromptResponse[OutputT]:
        """Execute the main loop for a single request.

        Handles visibility expansion retries internally.
        """
        ...
```

### Request and Response Events

The main loop uses typed events for request/response coordination:

```python
@FrozenDataclass()
class MainLoopRequest(Generic[UserRequestT]):
    """Event published to trigger main loop execution."""

    request: UserRequestT
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@FrozenDataclass()
class MainLoopCompleted(Generic[OutputT]):
    """Event published when main loop execution completes successfully."""

    request_id: UUID
    response: PromptResponse[OutputT]
    session_id: UUID
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@FrozenDataclass()
class MainLoopFailed:
    """Event published when main loop execution fails."""

    request_id: UUID
    error: Exception
    session_id: UUID | None
    failed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

## Execution Lifecycle

### Phase Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Receive        │────▶│  Create         │────▶│  Create         │
│  Request        │     │  Session        │     │  Prompt         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Publish        │◀────│  Parse          │◀────│  Evaluate       │
│  Result         │     │  Response       │     │  with Adapter   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │  Handle         │
                                                │  Visibility     │◀──┐
                                                │  Expansion      │───┘
                                                └─────────────────┘
```

### Execution Steps

1. **Receive request** - Subscribe to `MainLoopRequest[UserRequestT]` events
   or call `execute()` directly.

2. **Create session** - Invoke `create_session()` to provision execution state.

3. **Create prompt** - Invoke `create_prompt(request)` to build the bound prompt.

4. **Evaluate** - Call `adapter.evaluate()` with prompt, session, and bus.

5. **Handle visibility expansion** - If `VisibilityExpansionRequired` is raised,
   update visibility overrides and retry from step 4.

6. **Publish result** - Emit `MainLoopCompleted` or `MainLoopFailed` to the bus.

### Visibility Override Handling

The main loop maintains a visibility override map across retries:

```python
def execute(self, request: UserRequestT) -> PromptResponse[OutputT]:
    session = self.create_session()
    prompt = self.create_prompt(request)
    visibility_overrides: dict[SectionPath, SectionVisibility] = {}

    while True:
        try:
            response = self._adapter.evaluate(
                prompt,
                bus=self._bus,
                session=session,
                visibility_overrides=visibility_overrides,
            )
            self._publish_completed(request, response, session)
            return response
        except VisibilityExpansionRequired as e:
            visibility_overrides.update(e.requested_overrides)
            # Continue loop to retry with expanded visibility
```

**Key behaviors:**

- Overrides accumulate across retries within a single execution
- Each retry uses the same session instance
- The prompt is not re-created; only visibility changes
- No limit on expansion retries (bounded by section count)

## Event-Driven Execution

### Bus Integration

The main loop subscribes to request events and publishes results:

```python
loop = MyMainLoop(adapter=adapter, bus=bus)

# Subscribe to handle incoming requests
bus.subscribe(MainLoopRequest[MyRequest], loop.handle_request)

# Trigger execution by publishing
result = bus.publish(MainLoopRequest(request=MyRequest(...)))

# Or call directly
response = loop.execute(MyRequest(...))
```

### Handler Pattern

```python
def handle_request(self, event: MainLoopRequest[UserRequestT]) -> None:
    """Event handler for bus-driven execution."""
    try:
        response = self.execute(event.request)
        self._bus.publish(MainLoopCompleted(
            request_id=event.request_id,
            response=response,
            session_id=self._current_session.session_id,
        ))
    except Exception as e:
        self._bus.publish(MainLoopFailed(
            request_id=event.request_id,
            error=e,
            session_id=getattr(self, '_current_session', None),
        ))
```

## Configuration

### MainLoopConfig

Optional configuration for execution behavior:

```python
@FrozenDataclass()
class MainLoopConfig:
    """Configuration options for main loop execution."""

    deadline: Deadline | None = None
    budget: Budget | None = None
    budget_tracker: BudgetTracker | None = None
    parse_output: bool = True
```

Configuration flows through to adapter evaluation:

```python
loop = MyMainLoop(
    adapter=adapter,
    bus=bus,
    config=MainLoopConfig(
        deadline=Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5)),
        budget=Budget(max_total_tokens=50000),
    ),
)
```

## Implementation Pattern

### Minimal Implementation

```python
@dataclass(slots=True, frozen=True)
class ReviewRequest:
    file_path: str
    focus_areas: tuple[str, ...]

@dataclass(slots=True, frozen=True)
class ReviewResult:
    summary: str
    issues: list[str]

class CodeReviewLoop(MainLoop[ReviewRequest, ReviewResult]):
    def __init__(
        self,
        *,
        adapter: ProviderAdapter[ReviewResult],
        bus: EventBus,
    ) -> None:
        super().__init__(adapter=adapter, bus=bus)
        self._template = self._build_template()

    def create_prompt(self, request: ReviewRequest) -> Prompt[ReviewResult]:
        return Prompt(self._template).bind(
            ReviewParams(
                file_path=request.file_path,
                focus_areas=request.focus_areas,
            )
        )

    def create_session(self) -> Session:
        return Session(bus=self._bus, tags={"loop": "code-review"})

    def _build_template(self) -> PromptTemplate[ReviewResult]:
        return PromptTemplate[ReviewResult](
            ns="reviews",
            key="code-review",
            sections=[...],
        )
```

### With Custom Reducers

```python
class StatefulLoop(MainLoop[Request, Output]):
    def create_session(self) -> Session:
        session = Session(bus=self._bus)
        session.mutate(Metrics).register(MetricEvent, metrics_reducer)
        session.mutate(Plan).register(SetupPlan, plan_reducer)
        return session
```

### With Progressive Disclosure

```python
class DisclosureLoop(MainLoop[Request, Output]):
    def create_prompt(self, request: Request) -> Prompt[Output]:
        template = PromptTemplate[Output](
            ns="agent",
            key="task",
            sections=[
                MarkdownSection[Params](
                    title="Reference",
                    template="Detailed reference content...",
                    summary="Reference available on request.",
                    visibility=SectionVisibility.SUMMARY,
                    key="reference",
                ),
                # ... other sections
            ],
        )
        return Prompt(template).bind(Params.from_request(request))
```

## Error Handling

### Exception Propagation

| Exception | Behavior |
|-----------|----------|
| `VisibilityExpansionRequired` | Caught and retried with updated overrides |
| `PromptEvaluationError` | Wrapped in `MainLoopFailed` event, re-raised |
| `DeadlineExceededError` | Wrapped in `MainLoopFailed` event, re-raised |
| `BudgetExceededError` | Wrapped in `MainLoopFailed` event, re-raised |
| Other exceptions | Wrapped in `MainLoopFailed` event, re-raised |

### Error Events

Failed executions publish `MainLoopFailed` before re-raising:

```python
try:
    response = loop.execute(request)
except PromptEvaluationError as e:
    # MainLoopFailed already published to bus
    handle_failure(e)
```

## Testing

### Unit Tests

- Mock adapter to verify prompt/session factory invocation
- Simulate `VisibilityExpansionRequired` to test retry loop
- Verify event publication for success and failure paths
- Test configuration propagation to adapter

### Integration Tests

```python
def test_visibility_expansion_flow():
    bus = InProcessEventBus()
    adapter = MockAdapter(
        responses=[
            VisibilityExpansionRequired(...),  # First call
            PromptResponse(...)                 # After expansion
        ]
    )
    loop = TestLoop(adapter=adapter, bus=bus)

    response = loop.execute(TestRequest())

    assert response.output is not None
    # Verify two adapter calls occurred
```

### Fixtures

- `tests/helpers/loops.py` provides `MockMainLoop` for prompt tests
- `tests/fixtures/requests/` contains sample request payloads

## Limitations

- **Synchronous execution**: Main loop blocks until completion or failure
- **Single adapter**: Each loop instance binds to one provider adapter
- **No mid-execution cancellation**: Visibility retries run to completion
- **Session per execution**: No session reuse across `execute()` calls
- **Event bus locality**: Events do not cross process boundaries
