# Deadlines Specification

## Goal

Allow orchestration hosts to enforce a wall-clock deadline for an entire
`evaluate()` run, ensuring tool handlers and delegated subagents stop before the
window closes. The deadline is provided by the caller and applies uniformly to
all nested work spawned during that evaluation.

## API Changes

- Extend `ProviderAdapter.evaluate` with a new keyword-only argument:

  ```python
  from datetime import datetime

  def evaluate(
      self,
      prompt: Prompt[OutputT],
      *params: SupportsDataclass,
      parse_output: bool = True,
      *,
      bus: EventBus,
      session: SessionProtocol,
      deadline: datetime | None = None,
  ) -> PromptResponse[OutputT]:
      ...
  ```

  - `deadline` MUST be timezone-aware (`tzinfo` set) and represent a timestamp in
    the future relative to `datetime.now(datetime.UTC)` when the call starts.
  - Callers may omit `deadline`; adapters treat `None` as "no deadline" to remain
    backwards compatible with existing integrations.
  - Adapters validate the argument eagerly and raise `PromptEvaluationError`
    with `phase="preflight"` if the timestamp is missing `tzinfo`, lies in the
    past, or falls within the current second.

- Mirror the new keyword in every public adapter (`openai`, `litellm`, etc.) and
  thread it through helper layers such as prompt runners, orchestration
  utilities, and the `ToolContext` dataclass so handlers can inspect the value.

## Deadline Propagation

1. **Rendered Prompt Metadata** – After validation the orchestrator stores the
   deadline on the rendered prompt instance (`RenderedPrompt.deadline: datetime |
   None`). All child prompts created during the run inherit this value unless a
   stricter deadline is provided explicitly.
1. **Session State** – Publish a `PromptDeadlineAssigned` event containing the
   normalized deadline. Reducers can surface the remaining time to tools or
   logging sinks without accessing adapter internals.
1. **Tool Context** – Extend `ToolContext` with a `deadline: datetime | None`
   attribute. The orchestrator populates it before every handler invocation so
   tools can self-limit work or spawn nested prompts with the same deadline.

## Enforcement Rules

- **Before Provider Calls** – Adapters compare `datetime.now(datetime.UTC)` to
  the deadline immediately before sending a request. If the deadline has passed
  they skip the provider call and raise `PromptEvaluationError` with
  `phase="deadline"`.
- **Tool Execution** – The runtime checks the deadline before invoking each tool
  handler. When time has expired it returns `ToolResult(success=False, value=None,
  message="deadline exceeded")` without calling the handler.
- **Subagents** – Delegation helpers reuse the deadline supplied by the parent
  context. Child prompts run with the earlier of the parent deadline and any
  explicit child override. Subagents exceeding the deadline return a failing
  `ToolResult` whose error message notes the timeout.
- **Polling & Retries** – Any retry loops (for streaming responses or provider
  backoffs) must re-check the deadline before each iteration to avoid overshooting
  the wall-clock budget.

## Observability

- Include the deadline timestamp in the `provider_payload` envelope when raising
  deadline errors so hosts can log the cutoff time.
- Emit structured log lines or events summarizing time remaining when a prompt
  finishes, making deadline usage visible even on successful runs.

## Testing Strategy

- Unit tests cover validation failures (`tzinfo` missing, past deadlines, etc.).
- Adapter integration tests simulate deadlines a few seconds in the future and
  assert that tool handlers receive the value via `ToolContext`.
- Subagent tests create parent/child prompts with conflicting deadlines and
  verify that the tighter deadline prevails.
- Regression tests confirm that omitting `deadline` preserves current behavior.

