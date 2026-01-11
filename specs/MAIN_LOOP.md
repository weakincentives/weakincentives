# Main Loop Specification

## Purpose

`MainLoop` standardizes agent workflow orchestration: receive request, build
prompt, evaluate within prompt's resource context, handle visibility expansion,
publish result. Implementations define only the domain-specific factories.

## Guiding Principles

- **Event-driven**: Requests arrive via bus; results return the same way
- **Factory-based**: Subclasses own prompt and session construction
- **Prompt-owned resources**: Resource lifecycle managed by prompt context
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
        bus: ControlDispatcher,
        config: MainLoopConfig | None = None,
    ) -> None: ...

    @abstractmethod
    def prepare(self, request: UserRequestT) -> tuple[Prompt[OutputT], Session]: ...

    def finalize(self, prompt: Prompt[OutputT], session: Session) -> None: ...

    def execute(self, request: UserRequestT) -> tuple[PromptResponse[OutputT], Session]: ...
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
```

Request-level `budget` and `deadline` override config defaults. A fresh
`BudgetTracker` is created per execution.

## Execution

```mermaid
flowchart LR
    Request --> Prepare --> EnterContext["Enter prompt context"]
    EnterContext --> Evaluate
    Evaluate --> Visibility["VisibilityExpansion"]
    Visibility -->|retry| Evaluate
    Evaluate --> Finalize --> ExitContext["Exit prompt context"]
    ExitContext --> Result
```

1. Receive `MainLoopRequest` via bus or direct `execute()` call
1. Initialize prompt and session via `prepare(request)`
1. Enter prompt context (`with prompt.resources:`) to initialize resources
1. Evaluate with adapter
1. On `VisibilityExpansionRequired`: write overrides into session state, retry
   step 4
1. Call `finalize(prompt, session)` for post-processing
1. Exit prompt context to cleanup resources
1. Publish `MainLoopCompleted` or `MainLoopFailed`

### Resource Lifecycle

Resources are owned by the prompt and managed via context manager. The
`execute()` method ensures proper lifecycle:

```python
def execute(self, request: UserRequestT) -> tuple[PromptResponse[OutputT], Session]:
    prompt, session = self.prepare(request)
    budget_tracker = BudgetTracker(budget=self._effective_budget) if self._effective_budget else None

    with prompt.resources:  # Resources initialized here
        while True:
            try:
                response = self._adapter.evaluate(
                    prompt,
                    session=session,
                    deadline=self._effective_deadline,
                    budget_tracker=budget_tracker,
                )
                self.finalize(prompt, session)
                return response, session
            except VisibilityExpansionRequired as e:
                for path, visibility in e.requested_overrides.items():
                    session[VisibilityOverrides].apply(
                        SetVisibilityOverride(path=path, visibility=visibility)
                    )
    # Resources cleaned up on context exit
```

**Key points:**

- Resources are initialized when entering the prompt context
- The same resources persist across visibility expansion retries
- Resources are cleaned up when exiting the context (success or failure)
- Adapters access resources via `prompt.resources`

### Visibility Handling

Overrides are stored in the session; session persists across retries; prompt
is not recreated. The `finalize` hook is called only on successful evaluation.

## Usage

### Bus-Driven

```python
loop = MyMainLoop(adapter=adapter, bus=bus)

# MainLoop subscribes to MainLoopRequest in __init__

# With request-specific constraints
bus.dispatch(MainLoopRequest(
    request=MyRequest(...),
    budget=Budget(max_total_tokens=10000),
))
```

**Note:** `InProcessDispatcher` dispatches by `type(event)`, not generic alias.
`MainLoopRequest[T]` is for static type checking; at runtime all events are
`MainLoopRequest`. For multiple loop types on one bus, filter by request type
in the handler or use separate buses.

### Direct

```python
response, session = loop.execute(MyRequest(...))
```

## Implementation

```python
class CodeReviewLoop(MainLoop[ReviewRequest, ReviewResult]):
    def __init__(self, *, adapter: ProviderAdapter[ReviewResult], bus: ControlDispatcher) -> None:
        super().__init__(adapter=adapter, bus=bus)
        self._template = PromptTemplate[ReviewResult](
            ns="reviews",
            key="code-review",
            sections=[...],
        )

    def prepare(self, request: ReviewRequest) -> tuple[Prompt[ReviewResult], Session]:
        prompt = Prompt(self._template).bind(ReviewParams.from_request(request))
        session = Session(bus=self._bus, tags={"loop": "code-review"})
        return prompt, session

    def finalize(self, prompt: Prompt[ReviewResult], session: Session) -> None:
        # Optional: cleanup, logging, or post-processing
        pass
```

### With Resources

Resources are declared on the template or bound at runtime:

```python
def prepare(self, request: ReviewRequest) -> tuple[Prompt[ReviewResult], Session]:
    # Template already has default resources
    # Add request-specific resources at bind time (pass mapping, not ResourceRegistry)
    prompt = Prompt(self._template).bind(
        ReviewParams.from_request(request),
        resources={GitClient: GitClient(repo=request.repo_path)},
    )
    session = Session(bus=self._bus)
    return prompt, session
```

### With Reducers

```python
def prepare(self, request: ReviewRequest) -> tuple[Prompt[ReviewResult], Session]:
    prompt = Prompt(self._template).bind(ReviewParams.from_request(request))
    session = Session(bus=self._bus)
    session[Plan].register(SetupPlan, plan_reducer)
    return prompt, session
```

### With Progressive Disclosure

```python
def prepare(self, request: Request) -> tuple[Prompt[Output], Session]:
    prompt = Prompt(PromptTemplate[Output](
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
    session = Session(bus=self._bus)
    return prompt, session
```

## Error Handling

| Exception | Behavior |
| ----------------------------- | ---------------------------------- |
| `VisibilityExpansionRequired` | Retry with updated overrides |
| All others | Publish `MainLoopFailed`, re-raise |

Resource cleanup happens in both success and failure cases due to context
manager semantics.

## Code Reviewer Integration

The code reviewer agent uses `MainLoop` with these specifics:

**Session reuse:** A single session is created at loop construction and reused
across all `prepare()` calls. State accumulates across turns.

**Auto-optimization:** The explicit `optimize` command is removed. Before each
evaluation, the loop checks for `WorkspaceDigest` in session state. If absent,
optimization runs automatically.

```python
def execute(self, request: UserRequestT) -> PromptResponse[OutputT]:
    if self._session[WorkspaceDigest].latest() is None:
        self._run_optimization()
    # ... proceed with evaluation
```

**Default deadline:** All requests receive a 5-minute deadline unless
overridden at the request level.

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

## Related Specifications

- `specs/DLQ.md` - Dead letter queue configuration for failed messages
- `specs/MAILBOX.md` - Mailbox protocol and message lifecycle
- `specs/LIFECYCLE.md` - LoopGroup coordination and graceful shutdown
- `specs/EVALS.md` - EvalLoop wrapping MainLoop for evaluations
