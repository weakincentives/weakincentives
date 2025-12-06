# Deadlines Specification

## Goal

Allow orchestration hosts to enforce a wall-clock deadline for an entire
`evaluate()` run, ensuring tool handlers and delegated subagents stop before the
window closes. The deadline is provided by the caller and applies uniformly to
all nested work spawned during that evaluation.

## Rationale

- Protect hosts from runaway evaluations and tool handlers that would otherwise
  overrun their resource budgets.
- Standardize how providers, tools, and subagents receive deadline data so
  integrations can make consistent decisions about when to abort.
- Preserve compatibility with existing prompts by keeping the deadline optional
  while still enforcing validation when a value is present.

## Scope

- Applies to every provider adapter (`openai`, `litellm`, and shared orchestration
  helpers) plus all tool handler entry points executed through
  `ToolContext.deadline`.
- Covers deadline validation, propagation into rendered prompts, enforcement
  before provider requests, and translation of tool-level deadline errors into
  prompt failures.
- Excludes any out-of-band cancellation of in-flight provider calls; enforcement
  happens at discrete checkpoints in the orchestrator and inside tool handlers.

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

## Implementation today

- **Value object (`weakincentives.deadlines.Deadline`)** – Validates
  timezone-aware expirations at construction, rejects deadlines fewer than one
  second in the future, and exposes `remaining()` for shared comparisons.
  `remaining()` enforces a timezone-aware `now` argument and uses the same UTC
  clock source as `__post_init__` for consistency.
- **Adapter entrypoints (`openai`, `litellm`)** – `evaluate()` guards immediately
  after collecting render inputs: when `deadline.remaining()` is non-positive,
  the adapters raise `PromptEvaluationError` with `phase="request"` before the
  provider is called. When a deadline is provided it replaces
  `RenderedPrompt.deadline` so downstream helpers see the same value.
- **Inner loop (`adapters.shared.run_inner_loop`)** – Picks the
  effective deadline from the explicit argument or the rendered prompt, then
  carries it through each loop iteration.
- **Pre-flight checks (`InnerLoop._ensure_deadline_remaining`)** –
  Called before every provider request and during response finalization to
  translate an expired deadline into `PromptEvaluationError` tagged with
  `phase="request"` or `phase="response"` respectively. Provider payloads
  include `deadline_expires_at` when available.
- **Tool execution (`ToolExecutor` and `tool_execution`)** – Checks
  `deadline.remaining()` before entering each tool call and again inside the
  handler context. When time has elapsed, `_raise_tool_deadline_error()` raises a
  `PromptEvaluationError` with `phase="tool"` and a provider payload containing
  the ISO timestamp. Handlers that raise `DeadlineExceededError` are translated
  into the same prompt error shape.
- **Propagation to tool context** – `ToolContext.deadline` passes the active
  deadline to handlers, allowing them to compute remaining time or raise
  `DeadlineExceededError` proactively.
- **Provider payloads** – `deadline_provider_payload()` serializes
  `deadline.expires_at` to ISO-8601 for structured logging and telemetry across
  provider, tool, and finalization paths.

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

## Caveats & limitations

- Deadline checks happen at well-defined checkpoints (before provider calls,
  before tool execution, and while finalizing responses) but do not interrupt
  in-flight SDK requests; long-running provider calls may still consume time
  past the cutoff.
- Tool handlers must cooperate by consulting `context.deadline` or raising
  `DeadlineExceededError`; the runtime cannot preempt handlers that ignore the
  deadline.
- The minimum one-second buffer may be too coarse for extremely short-running
  tasks and could reject aggressive deadlines that are valid for specific hosts.
- Deadlines rely on synchronized UTC clocks between callers and the runtime; skew
  can cause premature expirations or unintended slack.

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
