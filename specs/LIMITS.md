# Limits Specification

## Goal

Allow orchestration hosts to enforce hard execution limits for an entire `evaluate()`
run. Hosts may bound wall-clock time via deadlines and constrain model/token usage
as a proxy for monetary budget. The limits are provided by the caller and apply
uniformly to all nested work spawned during that evaluation.

## API Changes

- Extend `ProviderAdapter.evaluate` with two new keyword-only arguments:

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
      token_budget: TokenBudget | None = None,
  ) -> PromptResponse[OutputT]:
      ...
  ```

  - `deadline` MUST be timezone-aware (`tzinfo` set) and represent a timestamp in
    the future relative to `datetime.now(datetime.UTC)` when the call starts.

  - `token_budget` encapsulates the maximum allowed token usage for the entire
    evaluation, with distinct limits for **input** (request) and **output**
    (response) tokens across all providers. It is optional to preserve backwards
    compatibility; adapters treat `None` as "no token limit".

  - Define `TokenBudget` in `weakincentives.limits` as a frozen dataclass with
    the following fields:

    ```python
    @dataclass(frozen=True)
    class TokenBudget:
        total: int | None = None
        input: int | None = None
        output: int | None = None
        provider_shares: Mapping[str, "TokenBudget"] | None = None
    ```

    Values MUST be positive integers when supplied. `TokenBudget.total`
    constrains combined input + output usage, `TokenBudget.input` constrains all
    request-side tokens, and `TokenBudget.output` constrains all response-side
    tokens. When `provider_shares` is provided, it specifies per-provider limits
    keyed by adapter slug (e.g., `"openai"`) using the same structure—nested
    budgets omit any dimension they leave unbounded. Any omitted field (for
    example `input=None`) indicates that dimension is unbounded while the other
    populated fields remain active.

  - Callers may omit either argument; adapters validate the arguments eagerly and
    raise `PromptEvaluationError` with `phase="preflight"` when the deadline is
    invalid (missing `tzinfo`, lies in the past, or falls within the current
    second) or when the token budget contains non-positive values or combines
    conflicting limits (for example `total` smaller than either `input` or
    `output`).

- Mirror the new keywords in every public adapter (`openai`, `litellm`, etc.) and
  thread them through helper layers such as prompt runners, orchestration
  utilities, and the `ToolContext` dataclass so handlers can inspect the active
  limits.

- Introduce `DeadlineExceededError(RuntimeError)` and
  `TokenBudgetExceededError(RuntimeError)` in `weakincentives.tools.errors`. Tools
  raise them when they cannot finish before the cutoff or when the remaining
  token allowance is insufficient.

## Limit Propagation

1. **Rendered Prompt Metadata** – After validation the orchestrator stores the
   deadline and token budget on the rendered prompt instance
   (`RenderedPrompt.deadline: datetime | None` and
   `RenderedPrompt.token_budget: TokenBudget | None`). All child prompts created
   during the run inherit these values unless stricter limits are provided
   explicitly.
1. **Tool Context** – `ToolContext` already exposes the active `RenderedPrompt`,
   so handlers read limits from `context.rendered_prompt`. Orchestrators MUST
   ensure the `RenderedPrompt` attached to each invocation carries the
   appropriate limit metadata before dispatching the tool.
1. **Delegated Subagents** – Delegation helpers reuse the limits supplied by the
   parent context. Child prompts run with the tightest combination of inherited
   limits and any explicit overrides supplied by the caller.

## Enforcement Rules

- **Before Provider Calls** – Adapters compare `datetime.now(datetime.UTC)` to
  the deadline immediately before sending a request. If the deadline has passed
  they skip the provider call and raise `PromptEvaluationError` with
  `phase="deadline"`. They also compute the projected input and output token
  usage of the pending call (including streaming retries) and raise
  `PromptEvaluationError` with `phase="token_budget"` when any relevant
  allowance (aggregate, input-only, or output-only) would be exceeded.
- **Token Accounting** – Maintain a `TokenLedger` component that records tokens
  consumed per provider and in total, split into input and output buckets.
  Adapters update the ledger after every provider response (using actual usage
  from the provider when available) and consult it before initiating new work.
  The ledger MUST expose helper methods for `reserve` (optimistic deductions
  before calls) and `consume` (actual usage reconciliation), rolling back
  reservations if the call fails. Reservations and consumption MUST annotate
  whether the deduction came from input tokens, output tokens, or both so the
  orchestrator can enforce the corresponding limits precisely. After each
  mutation the ledger publishes a `TokenLedgerUpdated` event (defined in the
  [Prompt Event Emission spec](EVENTS.md)) so downstream observers, including
  the active `Session`, can observe the current totals.
- **Tool Execution** – Tool handlers check deadlines and token availability via
  the context before performing work. When they cannot complete in time or would
  exceed any relevant token allowance they MUST raise the corresponding limit
  error. The runtime converts `DeadlineExceededError` into
  `PromptEvaluationError` with `phase="deadline"` and
  `TokenBudgetExceededError` into `PromptEvaluationError` with
  `phase="token_budget"`, aborting the evaluation loop immediately. When a
  limit expires before invocation, the runtime raises the same
  `PromptEvaluationError` without calling the handler.
- **Polling & Retries** – Any retry loops (for streaming responses or provider
  backoffs) must re-check both limits before each iteration to avoid overshooting
  the wall-clock or token budget.

## Observability

- Include the deadline timestamp and remaining token allowances (aggregate,
  input, and output) in the `provider_payload` envelope when raising limit
  errors so hosts can log the cutoff state.
- Emit structured log lines or events summarizing time remaining and tokens
  consumed, split by input/output, when a prompt finishes. This keeps limit
  usage visible even on successful runs.
- Surface token consumption deltas in metrics collectors (e.g.,
  `limits.tokens` with labels for provider and direction) to help capacity
  planning. The emitted `TokenLedgerUpdated` events provide a canonical snapshot
  for these reporters; hosts SHOULD derive observability payloads from the
  event instead of querying adapters directly.

## Testing Strategy

- Unit tests cover validation failures (deadline `tzinfo` missing, past deadlines,
  non-positive token budgets, invalid provider shares).
- Adapter integration tests simulate deadlines a few seconds in the future and
  assert that tool handlers receive the value via `ToolContext`. Additional tests
  exhaust token budgets through repeated provider calls and assert that adapters
  raise `PromptEvaluationError` with `phase="token_budget"`.
- Subagent tests create parent/child prompts with conflicting limits and verify
  that the tightest bound prevails.
- Regression tests confirm that omitting both `deadline` and `token_budget`
  preserves current behavior.
