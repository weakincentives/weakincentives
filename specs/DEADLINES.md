# Deadlines Specification

## Goal

Allow orchestration hosts to enforce a wall-clock deadline for an entire
`evaluate()` run, ensuring tool handlers and delegated subagents stop before the
window closes. The deadline is provided by the caller and applies uniformly to
all nested work spawned during that evaluation.

## API Changes

- Extend `ProviderAdapter.evaluate` with a dedicated keyword-only argument for
  deadline enforcement that accepts a `Deadline` value instead of a raw
  `datetime` so validation happens at construction time.

  ```python
  from weakincentives.deadlines import Deadline

  def evaluate(
      self,
      prompt: Prompt[OutputT],
      *params: SupportsDataclass,
      parse_output: bool = True,
      *,
      bus: EventBus,
      session: SessionProtocol,
      deadline: Deadline | None = None,
  ) -> PromptResponse[OutputT]:
      ...
  ```

  - Define `Deadline` as an immutable value object in `weakincentives.deadlines`
    with a timezone-aware `expires_at: datetime` attribute. Its constructor MUST
    reject naive datetimes, timestamps in the past, or values that fall within
    the current second relative to `datetime.now(datetime.UTC)`.
    Expose helper methods such as
    `remaining(*, now: datetime | None = None) -> timedelta` so adapters and
    tools can reuse consistent remaining-time calculations.
  - Callers may omit `deadline`; adapters treat `None` as "no deadline" to remain
    backwards compatible with existing integrations.
  - Adapters validate the argument eagerly and raise `PromptEvaluationError`
    with `phase="preflight"` when the `Deadline` instance fails validation or
    does not extend past the current time (this protects adapters that receive
    stale objects from deserialization or untrusted callers).

- Mirror the `deadline` keyword in every public adapter (`openai`, `litellm`,
  etc.) and thread it through helper layers such as prompt runners,
  orchestration utilities, and the `ToolContext` dataclass so handlers can
  inspect the value.

- Introduce a `DeadlineExceededError(RuntimeError)` in `weakincentives.tools.errors`.
  Tools raise it when they cannot finish before the cutoff.

## Deadline Propagation

1. **Rendered Prompt Metadata** – After validation the orchestrator stores the
   deadline on the rendered prompt instance
   (`RenderedPrompt.deadline: Deadline | None`). All child prompts created during
   the run inherit this value unless a stricter deadline is provided explicitly.
1. **Tool Context** – `ToolContext` already exposes the active
   `RenderedPrompt`, so handlers read the deadline from
   `context.rendered_prompt.deadline`. Orchestrators MUST ensure the
   `RenderedPrompt` attached to each invocation carries the appropriate
   `Deadline` before dispatching the tool.

## Enforcement Rules

- **Before Provider Calls** – Adapters compare `datetime.now(datetime.UTC)` to
  `deadline.expires_at` immediately before sending a request. If the deadline has
  passed they skip the provider call and raise `PromptEvaluationError` with
  `phase="deadline"`.
- **Tool Execution** – Add a `DeadlineExceededError` exception to
  `weakincentives.tools.errors`. Tool handlers MUST raise this error if the
  remaining time is insufficient to complete work safely. The runtime converts
  it into a `PromptEvaluationError` with `phase="deadline"`, aborting the entire
  evaluation loop immediately. When the deadline expires before invocation, the
  runtime raises the same `PromptEvaluationError` without calling the handler.
- **Tool Execution Contract** – All components treat `DeadlineExceededError`
  as the sole signal that the deadline has been exceeded. Handlers that cannot
  complete in time MUST raise it directly so the runtime can translate it into
  a `PromptEvaluationError` with `phase="deadline"` and halt the evaluation
  loop immediately.
- **Subagents** – Delegation helpers reuse the deadline supplied by the parent
  context. Child prompts run with the earlier of the parent deadline and any
  explicit child override. When a delegated prompt overruns the deadline the
  helper lets `DeadlineExceededError` propagate so the parent runtime converts
  it into `PromptEvaluationError` with `phase="deadline"` and terminates the
  evaluation immediately.
- **Polling & Retries** – Any retry loops (for streaming responses or provider
  backoffs) must re-check the deadline before each iteration to avoid overshooting
  the wall-clock budget.

## Observability

- Include the `Deadline.expires_at` timestamp in the `provider_payload` envelope when raising
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
