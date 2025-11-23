# LiteLLM Adapter Specification

## Goals

- Provide a drop-in LiteLLM adapter that matches the OpenAI adapter's API
  surface, lifecycle, and error handling so callers can switch providers
  without code changes.
- Keep parity in automated coverage across dependency guards, tool invocation
  flows, structured outputs, and error conditions.
- Maintain documentation and packaging that present LiteLLM as a first-class
  integration path for end users.

## Constraints

- Treat LiteLLM as an optional dependency; missing packages must fail fast with
  actionable errors instead of implicit fallbacks.
- Preserve adapter ergonomics and telemetry shapes to avoid downstream
  migrations when swapping providers.
- Reuse shared adapter utilities and prompt abstractions rather than adding
  provider-specific branches.

## Guiding Principles

- **Symmetry with OpenAI**: Align module structure, adapter methods, and
  conversation flow helpers with the OpenAI adapter to keep feature behavior
  identical between providers.
- **Contract preservation**: Honor existing structured output parsing, tool
  execution semantics, and deadline handling so higher-level prompts continue
  to behave consistently.
- **Explicit configuration**: Surface explicit hooks for dependency injection
  (custom completions or factories) while preventing ambiguous combinations
  that could mask misconfiguration.

## Current Implementation: API Surface

- `LiteLLMAdapter.evaluate` mirrors the OpenAI adapter's signature, accepting a
  `Prompt`, render parameters, an `EventBus`, a `SessionProtocol`, optional
  `Deadline`, and optional prompt override context. It delegates to the shared
  `run_conversation` helper, ensuring identical message assembly, tool routing,
  and event publication behavior.
- The adapter exposes structured output helpers (`extract_parsed_content`,
  `message_text_content`, `parse_schema_constrained_payload`) via
  `__all__`, matching the OpenAI-facing surface for downstream imports.
- Initial messages are seeded with the rendered system prompt, and tool choice
  defaults to `"auto"` while respecting prompt-level structured output settings
  such as `inject_output_instructions`.

## Current Implementation: Configuration and Dependency Handling

- Construction requires a `model` name and accepts `tool_choice`, an explicit
  `completion` callable, a `completion_factory`, and `completion_kwargs` for the
  factory path. The initializer rejects mixing `completion` with factory inputs
  to avoid ambiguous execution paths.
- `create_litellm_completion` defers import until call time and wraps
  `litellm.completion`, merging any `completion_kwargs` with per-request kwargs
  supplied during evaluation. `_load_litellm_module` raises a `RuntimeError`
  with installation guidance if the optional dependency is missing.
- Requests are assembled in `_call_provider` with model, messages, tool
  definitions, tool choice directives, and `response_format` payloads when
  structured output parsing is enabled, mirroring OpenAI request shapes.

## Current Implementation: Fallbacks and Error Paths

- Deadline handling rejects already-expired deadlines before rendering to avoid
  wasted provider calls.
- Provider failures during completion are wrapped in `PromptEvaluationError`
  with consistent phase metadata, keeping error propagation aligned with the
  OpenAI adapter.
- Choice selection is delegated to `first_choice`, ensuring deterministic
  selection semantics even if the provider returns multiple candidates.

## Caveats and Limitations

- LiteLLM coverage depends on the optional `litellm` package; environments
  lacking the extra will fail at import or factory creation time.
- Integration tests remain opt-in and may require external credentials; the
  default CI path relies on mocked completions rather than live LiteLLM calls.
- Structured output parsing is limited to schemas supported by LiteLLM's
  `response_format` contract; provider-side deviations could still surface as
  parsing errors despite symmetric adapter handling.
