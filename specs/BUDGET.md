# Budget Specification

## Goal

Define a first-class `Budget` abstraction that merges wall-clock deadlines with
token-based ceilings so orchestrators enforce hard limits during prompt
evaluation. The budget must propagate across nested work, including delegated
subagent runs, to guarantee callers never exceed either dimension of their
allocation.

## Rationale

- Combine time and token constraints into a single contract that providers,
  tools, and planners can consult uniformly.
- Prevent runaway prompt evaluations by rejecting work that would overshoot the
  remaining time or token allotment.
- Make budget enforcement visible to hosts so they can size allocations and
  respond to overages consistently.

## Scope

- Applies to all prompt evaluations initiated through provider adapters and the
  orchestration helpers they call.
- Covers budget validation, propagation through `RenderedPrompt`, enforcement
  checkpoints before provider calls and tool execution, and translation of
  overages into standardized prompt failures.
- Includes subagent invocations: delegated prompts must consume the same budget
  and halt when time or tokens run out.

## Abstraction

- `Budget` lives in `weakincentives.budget` as an immutable dataclass holding:
  - `deadline: Deadline | None` – optional wall-clock cutoff.
  - `token_usage: TokenUsage` – tracks consumed and remaining tokens.
- Construction validates both components and rejects impossible states
  (negative remaining tokens, expired deadlines, or missing token ceilings).
- Provide helpers:
  - `remaining_time(now: datetime | None = None) -> timedelta | None` mirroring
    `Deadline.remaining()` semantics.
  - `remaining_tokens() -> int` exposing the maximum tokens still available
    across prompt, provider response, and tool output phases.
  - `consume_tokens(count: int) -> Budget` returning a new instance with updated
    usage; raising `BudgetExceededError` when consumption would cross the cap.

## API Changes

- Extend provider adapter `evaluate()` signatures with a keyword-only
  `budget: Budget | None = None` argument that replaces separate deadline and
  token parameters. Adapters treat `None` as "no budget" to preserve backwards
  compatibility.
- `RenderedPrompt` gains an optional `budget: Budget | None` attribute. When
  present it supersedes any legacy deadline or token fields.
- Introduce `BudgetExceededError(PromptEvaluationError)` tagged with
  `phase="budget"` to signal either time or token overages.
- `ToolContext` and `ToolExecution` mirror the new budget field so handlers can
  proactively bail out when limits are at risk.

## Enforcement Rules

- **Pre-flight validation** – Adapters validate the `Budget` immediately;
  expired deadlines or depleted tokens raise `PromptEvaluationError` with
  `phase="preflight"` before any provider call.
- **Provider requests** – Before each provider invocation, orchestrators check
  `budget.remaining_time()` and `budget.remaining_tokens()` against the
  projected costs. When either is exhausted, raise `BudgetExceededError` without
  issuing the request.
- **Token accounting** – Provider adapters increment `token_usage` using the
  reported prompt/response token counts. Tool handlers report estimated token
  costs (or measured counts when available) through `ToolContext`, allowing the
  orchestrator to update the budget via `consume_tokens`.
- **Subagents** – Delegation helpers forward the active `Budget` into child
  prompts. Subagents MUST share the same object so consumption and expiration in
  the child reduce the remaining budget for the parent. When a child exceeds the
  budget the error propagates to the parent and halts the entire evaluation.
- **Finalization** – Response assembly verifies that the final token tally fits
  within the budget; otherwise it raises `BudgetExceededError` tagged with
  `phase="response"`.

## Observability

- Emit `budget_remaining_time` and `budget_remaining_tokens` metrics or log
  fields alongside provider calls and tool invocations.
- Include a serialized budget payload in `PromptEvaluationError` structures so
  hosts can debug where overages occurred.

## Testing Strategy

- Unit tests validate `Budget` construction, token consumption, and deadline
  rejection paths.
- Adapter tests simulate evaluations with tight budgets to ensure requests are
  blocked before exceeding token or time limits.
- Subagent integration tests assert that child invocations reduce the shared
  budget and that overages propagate as `BudgetExceededError`.
