# Main Loop Specification

## Purpose

`MainLoop` standardizes agent workflow: receive request, build prompt, evaluate
within resource context, handle visibility expansion, publish result.
Core at `runtime/main_loop.py`.

## Principles

- **Event-driven**: Requests via dispatcher; results return same way
- **Factory-based**: Subclasses own prompt and session construction
- **Prompt-owned resources**: Lifecycle managed by prompt context
- **Visibility-transparent**: Expansion exceptions retry automatically
- **Type-safe**: Generic parameters ensure request-prompt alignment

## Core Components

### MainLoop

At `runtime/main_loop.py`:

| Method | Description |
| --- | --- |
| `prepare(request)` | Create `(Prompt, Session)` for request |
| `finalize(prompt, session)` | Post-processing hook |
| `execute(request)` | Full execution returning `(PromptResponse, Session)` |

### Events

| Event | Fields |
| --- | --- |
| `MainLoopRequest[T]` | request, budget, deadline, request_id, created_at |
| `MainLoopCompleted[T]` | request_id, response, session_id, completed_at |
| `MainLoopFailed` | request_id, error, session_id, failed_at |

### Configuration

`MainLoopConfig`: `deadline`, `budget`. Request-level overrides config defaults.
Fresh `BudgetTracker` per execution.

## Execution Flow

1. Receive `MainLoopRequest` or direct `execute()` call
2. `prepare(request)` → `(Prompt, Session)`
3. Enter `with prompt.resources:` context
4. Evaluate with adapter
5. On `VisibilityExpansionRequired`: apply overrides to session, retry step 4
6. `finalize(prompt, session)`
7. Exit context (cleanup)
8. Publish `MainLoopCompleted` or `MainLoopFailed`

### Resource Lifecycle

Resources initialized on context entry, persist across visibility retries,
cleaned up on context exit. Adapters access via `prompt.resources`.

## Implementation Pattern

```python
class CodeReviewLoop(MainLoop[ReviewRequest, ReviewResult]):
    def prepare(self, request: ReviewRequest) -> tuple[Prompt[ReviewResult], Session]:
        prompt = Prompt(self._template).bind(ReviewParams.from_request(request))
        session = Session(dispatcher=self._dispatcher)
        return prompt, session
```

### With Resources

```python
def prepare(self, request: ReviewRequest) -> tuple[Prompt[ReviewResult], Session]:
    prompt = Prompt(self._template).bind(
        ReviewParams.from_request(request),
        resources={GitClient: GitClient(repo=request.repo_path)},
    )
    return prompt, session
```

### With Reducers

```python
session[Plan].register(SetupPlan, plan_reducer)
```

### With Progressive Disclosure

Use `visibility=SectionVisibility.SUMMARY` on sections with `summary` text.

## Error Handling

| Exception | Behavior |
| --- | --- |
| `VisibilityExpansionRequired` | Retry with updated overrides |
| All others | Publish `MainLoopFailed`, re-raise |

## Usage

### Bus-Driven

```python
loop = MyMainLoop(adapter=adapter, dispatcher=dispatcher)
dispatcher.dispatch(MainLoopRequest(request=MyRequest(...)))
```

**Note:** `InProcessDispatcher` dispatches by `type(event)`. Filter by request
type in handler or use separate dispatchers for multiple loop types.

### Direct

```python
response, session = loop.execute(MyRequest(...))
```

## Limitations

- Synchronous execution
- One adapter per loop
- No mid-execution cancellation
- Events local to process

## Related Specifications

- `specs/DLQ.md` - Dead letter queue configuration
- `specs/MAILBOX.md` - Mailbox protocol
- `specs/LIFECYCLE.md` - LoopGroup coordination
- `specs/EVALS.md` - EvalLoop wrapping MainLoop
