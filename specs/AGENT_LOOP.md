# Agent Loop Specification

## Purpose

`AgentLoop` standardizes agent workflow: receive request, build prompt, evaluate
within resource context, handle visibility expansion, publish result.
Core at `src/weakincentives/runtime/agent_loop.py`.

## Principles

- **Mailbox-driven**: Requests via mailbox; results return via `msg.reply()`
- **Factory-based**: Subclasses own prompt and session construction
- **Prompt-owned resources**: Lifecycle managed by prompt context
- **Visibility-transparent**: Expansion exceptions retry automatically
- **Type-safe**: Generic parameters ensure request-prompt alignment

## Core Components

### AgentLoop

At `src/weakincentives/runtime/agent_loop.py`:

| Method | Description |
| --- | --- |
| `prepare(request)` | Create `(Prompt, Session)` for request |
| `finalize(prompt, session, output)` | Post-process output; returns transformed `OutputT` |
| `execute(request)` | Full execution returning `(PromptResponse, Session)` |
| `execute_with_bundle(request, bundle_target=)` | Execute with debug bundling |

### Request and Result Types

At `src/weakincentives/runtime/agent_loop_types.py`:

`AgentLoopRequest[T]`: `request`, `budget`, `deadline`, `resources`, `request_id`,
`created_at`, `run_context`, `experiment`, `debug_bundle`. Request-level fields
override config.

`AgentLoopResult[T]`: `request_id`, `output`, `error`, `session_id`, `run_context`,
`completed_at`, `bundle_path`. Check `success` property for outcome.

### Configuration

`AgentLoopConfig` at `src/weakincentives/runtime/agent_loop_types.py`:

| Field | Description |
| --- | --- |
| `budget` | Optional default budget |
| `resources` | Optional default resources |
| `lease_extender` | Lease extension configuration |
| `debug_bundle` | `BundleConfig` for debug bundling |

Request-level overrides config defaults. Fresh `BudgetTracker` per execution.

### BundleConfig

At `src/weakincentives/debug/bundle.py`:

| Field | Description |
| --- | --- |
| `target` | Output directory for bundles (None disables bundling) |
| `max_file_size` | Skip files larger than this (default 10MB) |
| `max_total_size` | Maximum filesystem capture size (default 50MB) |
| `compression` | Zip compression method |
| `retention` | Policy for cleaning up old bundles |
| `storage_handler` | Handler for external storage upload |

`enabled` property returns `True` when `target` is set.

## Execution Flow

1. Receive `AgentLoopRequest` or direct `execute()` call
1. `prepare(request)` -> `(Prompt, Session)`
1. Resolve effective settings (budget, deadline, resources)
1. Evaluate with adapter
1. On `VisibilityExpansionRequired`: apply overrides to session, retry step 4
1. `finalize(prompt, session, output)` -> `OutputT` (post-processing/transformation)
1. `prompt.cleanup()` - Release section resources
1. Return `AgentLoopResult` (with `output` on success, `error` on failure)

### Visibility Expansion Retry Limit

The retry loop is capped at `_MAX_VISIBILITY_RETRIES = 10`
(at `src/weakincentives/runtime/agent_loop.py`). When exceeded, raises
`PromptEvaluationError` with `phase="request"` to prevent infinite
expansion loops.

### Prompt Cleanup

`prompt.cleanup()` is called after evaluation completes. In bundled execution,
cleanup is deferred until after bundle artifacts are captured. A
`prompt_cleaned_up` guard flag prevents double-cleanup in error paths.

### Resource Lifecycle

Resources initialized on context entry, persist across visibility retries,
cleaned up on context exit. Adapters access via `prompt.resources`.

## Implementation Pattern

```python
class CodeReviewLoop(AgentLoop[ReviewRequest, ReviewResult]):
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
| `VisibilityExpansionRequired` | Retry with updated overrides (up to 10 times) |
| All others | Return `AgentLoopResult` with `error` set |

## Usage

### Mailbox-Driven

```python
requests = InMemoryMailbox[AgentLoopRequest[MyRequest], AgentLoopResult](name="requests")
loop = MyAgentLoop(adapter=adapter, requests=requests)

# Client sends request
requests.send(AgentLoopRequest(request=MyRequest(...)), reply_to=responses)

# Start processing (blocking)
loop.run()
```

### Direct

```python
response, session = loop.execute(MyRequest(...))
```

### With Debug Bundle

```python
with loop.execute_with_bundle(request, bundle_target=Path("./bundles")) as ctx:
    score = compute_score(ctx.response.output)
    ctx.write_metadata("eval", {"score": score})
# Bundle is finalized on context exit
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
- `specs/EVALS.md` - EvalLoop wrapping AgentLoop
- `specs/DEBUG_BUNDLE.md` - Bundle creation and artifacts
