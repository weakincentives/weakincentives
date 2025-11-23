# Adapter architecture

## Scope
This document covers the adapter layer under `src/weakincentives/adapters/`. It
applies to the provider-neutral contracts in `core.py`, shared plumbing in
`shared.py`, and the provider-specific bindings in `openai.py` and `litellm.py`.
Use it as the single reference for how adapter entrypoints are wired, what they
promise, and where to configure them.

## Design goals
- **Provider interchangeability**: expose a common `ProviderAdapter` interface so
  runtime code, prompts, and tools can swap providers without changing call
  sites.
- **Strict contracts**: surface errors with the `PromptEvaluationError` phases
  defined in `core.py` so failures are attributed to request, tool, or response
  handling.
- **Observability**: publish `PromptRendered`, `PromptExecuted`, `ToolInvoked`,
  and handler failures through the `EventBus` so hosts can trace every call.
- **Structured output fidelity**: honor prompt-level schema hints and prefer
  native provider response formats when available to reduce instruction
  injection and parsing errors.
- **Deadline-aware execution**: propagate `Deadline` state into provider payloads
  and tool invocations so cooperative cancellation behaves consistently.

## Constraints and rationale
- **Optional dependencies**: adapters are opt-in; OpenAI requires the `openai`
  extra and LiteLLM requires the `litellm` extra. Import guards raise actionable
  errors to keep the base install lean while still offering integrations.
- **SDK surface stability**: the adapter boundary consumes narrow provider
  payloads (`ProviderMessage`, `ProviderToolCall`, `ProviderCompletionResponse`)
  defined in `_provider_protocols.py` to shield the rest of the runtime from SDK
  churn.
- **Tooling parity**: tool rendering, invocation, and result application live in
  `shared.py` so behavior is identical across providers. This keeps tool
  validation (`ToolValidationError`), deadline checks, and reducer error
  reporting consistent.
- **Response-phase clarity**: the phase constants in `core.py` deliberately split
  request, tool, and response handling to keep failure modes debuggable and
  contract-friendly.

## Guiding principles
- Treat adapters as thin translators: prompts render once, then each adapter only
  translates to provider wire formats and reconciles responses back into
  `PromptResponse` without embedding business logic.
- Prefer explicit configuration over ambient defaults; constructor kwargs should
  mirror provider options and remain serializable for logging/debugging.
- Keep provider payloads observable: include `provider_payload` when available so
  logs and retries can reason about what was sent and received.
- Avoid silent fallbacks: raise when structured output or tool payloads are
  malformed instead of skipping parsing.

## Adapter map
### OpenAI (`OpenAIAdapter`)
- **Code path**: `openai.py` (`OpenAIAdapter.evaluate`). Uses the provider-agnostic
  conversation loop in `shared.run_conversation` and parses tool messages via
  `_tool_messages.serialize_tool_message`.
- **Configuration**: constructor accepts `model`, `tool_choice`,
  `use_native_response_format`, and optional client hooks (`client`,
  `client_factory`, `client_kwargs`). Per-call knobs include `parse_output`,
  `deadline`, and prompt overrides (`overrides_store`, `overrides_tag`).
- **Live behaviors**: when `use_native_response_format` is true and the prompt
  declares structured output, the adapter builds a JSON Schema payload via
  `build_json_schema_response_format` and suppresses injected output
  instructions to avoid conflicting guidance. Deadline expiration before request
  dispatch triggers a request-phase `PromptEvaluationError`. Tool calls are
  serialized with `serialize_tool_message`, validated with
  `parse_tool_arguments`, and failures propagate reducer errors via
  `format_publish_failures`.
- **Caveats**: requires the `openai` extra; missing it raises a runtime error.
  Structured output parsing depends on provider support for the Responses API and
  may reject mismatched payloads. Tool choice handling follows the provider's
  semantics; incompatible directives surface as request-phase errors.

### LiteLLM (`LiteLLMAdapter`)
- **Code path**: `litellm.py` (`LiteLLMAdapter.evaluate`). Shares the same
  `shared.run_conversation` loop and tool serialization helpers as OpenAI.
- **Configuration**: constructor accepts `model`, `tool_choice`, and an optional
  completion callable (`completion`, `completion_factory`, `completion_kwargs`).
  Per-call options mirror the OpenAI adapter: `parse_output`, `deadline`, and
  prompt overrides.
- **Live behaviors**: wraps the provider completion callable, applying any
  pre-configured kwargs before dispatch. Structured output support uses the same
  JSON Schema builder when `parse_output` is enabled. Deadline checks guard both
  request submission and tool execution. Provider responses are normalized via
  `_provider_protocols` helpers before being parsed into `PromptResponse`.
- **Caveats**: requires the `litellm` extra. The adapter assumes LiteLLM emits a
  `choices` array compatible with `_provider_protocols.ProviderChoice`; mismatched
  shapes raise response-phase errors. Tool choice directives are passed through to
  LiteLLM; unsupported values surface as request-phase errors.

### Base contracts (`ProviderAdapter`)
- **Code path**: `core.py`. Defines the abstract `evaluate` contract and the
  `optimize` helper that orchestrates workspace digest prompts using
  `PlanningToolsSection`, `WorkspaceDigestSection`, and `PromptOverridesStore`.
- **Configuration**: subclasses provide constructor knobs; `optimize` accepts
  `store_scope`, `overrides_store`, `overrides_tag`, and an optional
  `optimization_session` for isolation.
- **Live behaviors**: `optimize` clones workspace sections into a fresh session,
  renders an optimization prompt, and persists the resulting digest either in the
  active `Session` (`OptimizationScope.SESSION`) or in prompt overrides (`GLOBAL`).
  Failures to locate a workspace digest section raise request-phase errors. Global
  scope writes also clear session-level digests to avoid stale caches.
- **Caveats**: global storage requires both `overrides_store` and `overrides_tag`.
  Digest extraction depends on a `WorkspaceDigestSection`; prompts lacking one
  cannot be optimized and will raise accordingly.

## Configuration surfaces
- **Constructor-level**: choose adapter type (OpenAI vs LiteLLM) based on
  available extras and provider preference; supply model and tool choice values
  aligned with provider capabilities; inject custom clients or completion
  callables when credential/bootstrap needs differ from defaults.
- **Per-evaluation**: toggle structured output parsing (`parse_output`), pass
  `Deadline` objects, and supply prompt overrides to reuse pre-rendered sections
  or cached digests. Event buses and sessions control observability and state.

## Known limitations
- **Provider feature drift**: provider SDK changes can surface as parsing errors
  despite the protocol guards; review `_provider_protocols.py` when upgrading
  dependencies.
- **Tool payload validation**: tool argument parsing rejects invalid JSON or
  schema mismatches; callers should ensure tool results align with declared
  dataclasses.
- **Native schema support**: structured output depends on provider-specific
  support for JSON Schema. When unavailable, adapters fall back to instruction
  parsing and may be more brittle.
