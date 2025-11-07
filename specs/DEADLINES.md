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
  - Adapters validate the argument eagerly with
    `weakincentives.adapters.shared.normalize_deadline`. The helper converts the
    timestamp to UTC, rejects values without `tzinfo`, and raises
    `PromptEvaluationError` with `phase="preflight"` if the timestamp is in the
    past or lands in the current whole second.

- Mirror the new keyword in every public adapter (`openai`, `litellm`, etc.) and
  thread it through helper layers such as prompt runners, orchestration
  utilities, and the `ToolContext` dataclass so handlers can inspect the value.

- Introduce a `DeadlineExceededError(RuntimeError)` in
  `weakincentives.tools.errors` and re-export it from
  `weakincentives.tools`. Tools raise it when they cannot finish before the
  cutoff.

## Deadline Propagation

1. **Rendered Prompt Metadata** – After validation the orchestrator stores the
   deadline on the rendered prompt instance (`RenderedPrompt.deadline: datetime | None`).
   Child prompts inherit this value unless a stricter deadline is provided
   explicitly.
1. **Tool Context** – `ToolContext` already exposes the active `RenderedPrompt`,
   so handlers read the deadline from `context.rendered_prompt.deadline`.
   Orchestrators MUST ensure the `RenderedPrompt` attached to each invocation
   carries the appropriate timestamp before dispatching the tool.
1. **Subagent Overrides** – Delegation helpers treat naive override values as
   invalid. When a child provides a timezone-aware override the helper runs the
   delegation with the earlier of the parent deadline and the override.

## Enforcement Rules

- **Before Provider Calls** – Adapters invoke
  `weakincentives.adapters.shared.ensure_deadline_active` immediately before
  contacting the provider. If the deadline has passed they skip the provider call
  and raise `PromptEvaluationError` with `phase="deadline"`. The helper includes
  `{"deadline": ISO8601, "time_remaining_seconds": float}` in
  `provider_payload` for observability.
- **Tool Execution** – Before invoking a handler the runtime calls
  `ensure_deadline_active` with a message scoped to the tool name. Handlers that
  cannot complete in time MUST raise `DeadlineExceededError`. The runtime
  translates the exception into `PromptEvaluationError` with `phase="deadline"`
  and the same payload structure described above.
- **Deadline Budget During Tool Runs** – `DeadlineExceededError` is the sole
  signal that the deadline has elapsed inside a handler. Callers SHOULD include a
  helpful message when raising it; otherwise the runtime supplies a default that
  names the tool.
- **Subagents** – Delegation helpers reuse the deadline supplied by the parent
  context. Child prompts run with the earlier of the parent deadline and any
  explicit child override, and the helper allows `DeadlineExceededError` to
  propagate to the parent evaluation so the adapter can surface a
  `PromptEvaluationError` with `phase="deadline"` immediately.
- **Polling & Retries** – Any retry loops (for streaming responses or provider
  backoffs) must re-check the deadline via `ensure_deadline_active` before each
  iteration to avoid overshooting the wall-clock budget.

## Observability

- Include the deadline timestamp and remaining budget in the
  `provider_payload` envelope when raising deadline errors so hosts can log the
  cutoff time.
- Emit structured log lines or events summarizing time remaining when a prompt
  finishes, making deadline usage visible even on successful runs.

## Testing Strategy

- Unit tests cover validation failures (`tzinfo` missing, past deadlines, same
  second as `now`, etc.).
- Adapter integration tests simulate deadlines a few seconds in the future and
  assert that tool handlers receive the value via `ToolContext`.
- Subagent tests create parent/child prompts with conflicting deadlines and
  verify that the tighter deadline prevails while naive overrides are ignored.
- Regression tests confirm that omitting `deadline` preserves current behavior.
