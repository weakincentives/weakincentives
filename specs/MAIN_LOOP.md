# Main Loop Specification

## Purpose

`MainLoop` standardizes agent workflow: receive request, build prompt, evaluate
within resource context, handle visibility expansion, publish result.
Core at `src/weakincentives/runtime/main_loop.py`.

## Principles

- **Mailbox-driven**: Requests via mailbox; results return via `msg.reply()`
- **Factory-based**: Subclasses own prompt and session construction
- **Prompt-owned resources**: Lifecycle managed by prompt context
- **Visibility-transparent**: Expansion exceptions retry automatically
- **Type-safe**: Generic parameters ensure request-prompt alignment

## Core Components

### MainLoop

At `src/weakincentives/runtime/main_loop.py`:

| Method | Description |
| --- | --- |
| `prepare(request)` | Create `(Prompt, Session)` for request |
| `finalize(prompt, session)` | Post-processing hook |
| `execute(request)` | Full execution returning `(PromptResponse, Session)` |

### Request and Result Types

`MainLoopRequest[T]`: `request`, `budget`, `deadline`, `resources`, `request_id`,
`created_at`, `run_context`, `experiment`. Request-level fields override config.

`MainLoopResult[T]`: `request_id`, `output`, `error`, `session_id`, `run_context`,
`completed_at`. Check `success` property for outcome.

### Configuration

`MainLoopConfig`: `deadline`, `budget`, `resources`, `lease_extender`.
Request-level overrides config defaults. Fresh `BudgetTracker` per execution.

## Execution Flow

1. Receive `MainLoopRequest` or direct `execute()` call
1. `prepare(request)` → `(Prompt, Session)`
1. Enter `with prompt.resources:` context
1. Evaluate with adapter
1. On `VisibilityExpansionRequired`: apply overrides to session, retry step 4
1. `finalize(prompt, session)`
1. Exit context (cleanup)
1. Return `MainLoopResult` (with `output` on success, `error` on failure)

### Resource Lifecycle

Resources initialized on context entry, persist across visibility retries,
cleaned up on context exit. Adapters access via `prompt.resources`.

## Implementation Pattern

```python
class CodeReviewLoop(MainLoop[ReviewRequest, ReviewResult]):
    def prepare(self, request: ReviewRequest) -> tuple[Prompt[ReviewResult], Session]:
        prompt = Prompt(self._template).bind(ReviewParams.from_request(request))
        session = Session()
        return prompt, session
```

### With Resources

```python
def prepare(self, request: ReviewRequest) -> tuple[Prompt[ReviewResult], Session]:
    prompt = Prompt(self._template).bind(
        ReviewParams.from_request(request),
        resources={GitClient: GitClient(repo=request.repo_path)},
    )
    session = Session()
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
| All others | Return `MainLoopResult` with `error` set |

## Usage

### Mailbox-Driven

```python
requests = InMemoryMailbox[MainLoopRequest[MyRequest], MainLoopResult](name="requests")
loop = MyMainLoop(adapter=adapter, requests=requests)

# Client sends request
requests.send(MainLoopRequest(request=MyRequest(...)), reply_to=responses)

# Start processing (blocking)
loop.run()
```

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
- `specs/EVALS.md` - EvalLoop (communicates via mailbox)
