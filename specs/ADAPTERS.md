# Adapter Implementation Reference

## Overview

**Scope:** This document covers the adapter architecture in `src/weakincentives/adapters/`, the public
`ProviderAdapter` contract, and the concrete adapters that ship today. It is the design source of truth and the
runtime reference guide for how prompts become provider calls.

**Design goals**

- Present a uniform adapter surface so prompts behave identically across providers and test doubles.
- Preserve deterministic, synchronous execution to align with session reducers and the DbC enforcement model.
- Keep all provider interactions observable through the `EventBus` while isolating prompt rendering from transport
  details.
- Minimize adapter-specific branches by centralizing lifecycles in shared helpers.

**Constraints**

- Adapters must run synchronously and honor the caller-supplied `Deadline` at every blocking boundary (request,
  tools, response parsing).
- Structured output parsing must remain compatible with providers that lack native schema enforcement; adapters cannot
  rely solely on provider-side validation.
- Tool execution must reuse the active `Session` and its `EventBus` so reducers observe the same state mutations that
  the adapter returns.

**Rationale for the architecture**

- A single runner (`shared.run_conversation` / `ConversationRunner`) coordinates provider calls, tool dispatch, and
  structured output parsing so new adapters inherit identical behavior without reimplementing loops.
- Provider-specific layers (`openai.py`, `litellm.py`, and future modules) only handle payload translation and choice
  selection, keeping SDK dependencies out of core logic.
- Centralizing error handling (`PromptEvaluationError`) gives callers stable failure semantics while leaving providers
  free to change their native exceptions.

**Guiding principles**

- Prefer adapter configuration over branching logic—surface provider toggles as constructor parameters rather than
  inline conditionals.
- Treat prompt rendering as immutable input: adapters should not mutate `RenderedPrompt` objects outside of sanctioned
  instruction toggles for response formats.
- Keep observability first-class: every render, tool invocation, and final response should emit the corresponding bus
  events unless publication fails and rollbacks occur.
- Document live behaviors alongside code paths so tests and operators can trace how settings map to runtime effects.

Adapter responsibilities:

- render the prompt with the supplied dataclass params (respecting overrides and output instructions),
- translate prompt tools into the provider's JSON schema,
- call the provider until a final assistant message is produced,
- execute requested tools inside the active session,
- parse structured output into typed dataclasses, and
- raise `PromptEvaluationError` with precise context when any phase fails.

### Architecture map

- `core.py` – Defines `ProviderAdapter`, `PromptResponse`, and the shared error model. All adapters inherit from this
  layer to guarantee consistent typing and DbC coverage.
- `shared.py` – Implements `run_conversation` / `ConversationRunner`, tool serialization (`tool_to_spec`), and response
  parsing. This is the live execution path for every adapter.
- `_provider_protocols.py` – Structural typing for provider choices, responses, and completion callables so adapters can
  remain import-light when extras are missing.
- `_tool_messages.py` – Utilities for formatting tool transcripts that flow back into provider conversations.
- `openai.py` / `litellm.py` – Provider-specific shims that translate runner payloads into SDK calls and map responses
  back into the shared representation.

### Adapter type map

- **LiteLLMAdapter** (`src/weakincentives/adapters/litellm.py`)

  - **Configuration surfaces:** Optional dependency (`uv sync --extra litellm`); requires `model` and accepts
    `tool_choice`; callers provide either a `completion` callable or a `completion_factory` plus `completion_kwargs`
    to wrap `litellm.completion`. Structured output requests always enable
    `require_structured_output_text=True` so the runner expects text even when forwarding schemas.
  - **Known caveats/limitations:** Behavior depends on downstream providers LiteLLM proxies. Errors bubble up as
    LiteLLM exceptions wrapped in `PromptEvaluationError`. Because the API lacks native parsed payloads, structured
    outputs rely on textual content and prompt-level parsing.

- **OpenAIAdapter** (`src/weakincentives/adapters/openai.py`)

  - **Configuration surfaces:** Optional dependency (`uv sync --extra openai`); requires `model` and accepts
    `tool_choice`; callers can pass a concrete client or a `client_factory` with `client_kwargs`. The
    `use_native_response_format` toggle disables inline output instructions when structured parsing is requested and
    forwards JSON schemas under `response_format`.
  - **Known caveats/limitations:** Requires OpenAI's Responses API. Deadlines are enforced before each request and tool
    execution. When `use_native_response_format=False`, parsing falls back to prompt instructions and requires textual
    outputs.

Use this map when wiring new adapters: mirror the configuration surfaces, expose extras explicitly, and document any
parser or provider limitations up front.

## Core Interfaces

```python
class ProviderAdapter(ABC):
    @abstractmethod
    def evaluate(
        self,
        prompt: Prompt[OutputT],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: EventBus,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        overrides_store: PromptOverridesStore | None = None,
        overrides_tag: str = "latest",
    ) -> PromptResponse[OutputT]: ...
```

- `prompt` / `params`: forwarded directly to `Prompt.render`. Each adapter must keep the positional dataclass order
  intact so template placeholders continue to line up with `supports_dataclass`.
- `parse_output`: toggles whether structured output metadata on the rendered prompt is honored.
- `bus`: every adapter publishes `PromptRendered`, `ToolInvoked`, and `PromptExecuted` events through the supplied bus.
- `session`: provides storage for tool handlers, digest slices, and rollback when the bus rejects an event.
- `deadline`: optional guard enforced before issuing a provider request, before every tool handler, and while
  finalizing responses. When omitted the prompt's own `RenderedPrompt.deadline` flows through.
- `overrides_store` / `overrides_tag`: passed straight to `Prompt.render` so callers can pin prompts to a versioned
  override store.

```python
@dataclass(slots=True)
class PromptResponse(Generic[OutputT]):
    prompt_name: str
    text: str | None
    output: OutputT | None
    tool_results: tuple[ToolInvoked, ...]
    provider_payload: dict[str, Any] | None = None
```

`tool_results` stores the exact `ToolInvoked` events emitted through the bus so downstream consumers can either inspect
the response object or subscribe to the event stream. `provider_payload` is populated via `extract_payload` when the
provider SDK exposes a `model_dump()` or mapping representation.

`PromptEvaluationError` carries a human-readable message, the prompt name, a `phase` literal (`"request"`, `"response"`,
or `"tool"`), and the provider payload (when available). Adapters use this exception for every failure mode so tests and
callers can safely assert on phase-specific error handling.

### Optimization API

`ProviderAdapter.optimize(...)` automates workspace digest generation for prompts that include a
`WorkspaceDigestSection`. The helper renders an internal prompt composed of:

1. a Markdown section describing the optimization goal,
1. a Markdown section enumerating expectations,
1. a `PlanningToolsSection` configured for the goal-decompose-route-synthesise strategy, and
1. clones of the prompt's workspace section (`PodmanSandboxSection` or `VfsToolsSection`) plus its digest section.

The optimization run executes in an inner `Session` and `EventBus` so tool handlers used for standard prompts cannot
mutate the original session. Callers may override this isolation by supplying an explicit
`optimization_session`; in that case the method reuses the provided session and event bus verbatim. The result is stored
according to the selected scope:

- `OptimizationScope.SESSION`: call `set_workspace_digest` on the caller's session.
- `OptimizationScope.GLOBAL`: require both `overrides_store` and `overrides_tag`; persist the digest in the overrides
  store and remove session copies via `clear_workspace_digest`.

```python
@dataclass(slots=True)
class OptimizationResult:
    response: PromptResponse[Any]
    digest: str
    scope: OptimizationScope
    section_key: str
```

## Execution Lifecycle

All adapters rely on `shared.run_conversation` (implemented through `ConversationRunner`) to coordinate provider calls.
The lifecycle is identical for LiteLLM, OpenAI, and future adapters:

1. **Render** – Call `prompt.render(*params, overrides_store, tag)` once. When structured output parsing is enabled
   and the prompt supports inline instructions (`inject_output_instructions=True`), adapters may disable those
   instructions when the provider has first-class JSON schema support (OpenAI's `response_format` path). Explicit
   deadlines on `evaluate` override the rendered prompt's deadline via `dataclasses.replace`.
1. **Publish `PromptRendered`** – `ConversationRunner` emits `PromptRendered` with the namespace, key, optional name,
   adapter label, session id, serialized render inputs, and the full rendered markdown. Subscribers (for example,
   the `Session` reducers in `specs/SESSIONS.md`) now have complete context before the provider request runs.
1. **Prepare tool payloads** – Every tool from `rendered.tools` is converted to the provider-agnostic JSON schema
   produced by `tool_to_spec`. The adapter keeps a registry keyed by tool name for resolving tool calls.
1. **Call the provider** – `run_conversation` builds the chat payload (`messages`, `tools`, `tool_choice`,
   optional `response_format`) and invokes the provider-specific callable supplied by the concrete adapter. The first
   choice (`first_choice`) becomes authoritative. Failures are wrapped in `PromptEvaluationError` with
   `phase="request"`.
1. **Handle tool calls** – When the provider responds with `tool_calls`, `ConversationRunner` records an assistant turn
   containing the serialized tool calls before executing them via `ToolExecutor`. Each tool invocation produces
   `ToolInvoked` events, textual responses (`serialize_tool_message`), and optional dataclass values. When tool_choice
   was forced to a specific function, the runner automatically reverts to `"auto"` after the first accepted call so the
   next provider turn can elide the restriction.
1. **Loop until completion** – The provider call/response/tool loop repeats until a message arrives without tool calls.
1. **Parse output** – `ResponseParser` inspects the final assistant message. When `parse_output` is `True` and the prompt
   declares structured output, the parser prefers provider-native structured payloads (`message.parsed` or segmented
   content) and falls back to the prompt's `parse_structured_output` helper. Parsed dataclasses null out the `text`
   field. When parsing is disabled the final text is returned verbatim.
1. **Publish `PromptExecuted`** – The finished `PromptResponse` is published as a `PromptExecuted` event. Subscribers can
   inspect the dataclass output or the raw text depending on which field is populated.

Throughout the lifecycle adapters call `deadline.remaining()` before issuing provider requests, before each tool handler,
and before finalizing the response. When the deadline is exceeded `PromptEvaluationError` includes the ISO timestamp
returned by `deadline_provider_payload`.

## Tool Invocation Semantics

`ToolExecutor` in `shared.py` owns tool scheduling, argument parsing, and event publication:

- Tool call arguments are decoded via `parse_tool_arguments`, which enforces JSON objects with string keys. Providers
  that omit arguments simply receive `{}`.
- Arguments map to dataclasses through `serde.parse`. Validation errors produce `ToolResult(success=False)` responses and
  log warnings rather than raising immediately. When validation fails, the raw arguments and error message are stored in
  `_RejectedToolParams` and emitted as the tool parameters for observability.
- `ToolContext` exposes the `Prompt`, `RenderedPrompt`, current adapter, session, event bus, and deadline to tool
  handlers. The context matches `PromptProtocol`/`ProviderAdapterProtocol` so handlers do not need to import concrete
  classes.
- Tool handlers run synchronously. If a handler raises `DeadlineExceededError` or the `Deadline` has already expired,
  `PromptEvaluationError` is raised and the provider turn aborts. Other exceptions are logged and converted into failed
  `ToolResult` instances that still flow back to the provider.
- Before emitting an event, the adapter captures a session snapshot. After a handler completes it publishes a
  `ToolInvoked` event and appends the instance to `PromptResponse.tool_results`. When the publish fails the session rolls
  back to the saved snapshot and the tool message is replaced with the aggregated reducer errors
  (`format_publish_failures`).
- Tool messages forwarded to the provider contain the handler's message plus the rendered value (if the tool opted into
  sharing it). After structured output parsing succeeds, the final successful tool message is patched with the parsed
  payload so the provider can continue referencing it if needed.

## Structured Output and Response Formats

`build_json_schema_response_format` inspects the rendered prompt's `output_type`, container type (`"object"` or
`"array"`), and `allow_extra_keys` flag to construct a JSON schema payload suitable for providers that support schema
constrained outputs. `run_conversation` passes this payload to the concrete adapter, which decides whether the provider
supports it:

- OpenAI: when `use_native_response_format=True`, adapters disable inline structured-output instructions and pass the
  schema under `response_format`. OpenAI responses include a `.parsed` payload which `ResponseParser` prefers. The
  parser accepts empty message text because the schema covers validation.
- LiteLLM: forwards the schema when structured output parsing is requested (LiteLLM proxies it to the target model) but
  still requires textual output for parsing (`require_structured_output_text=True`) because the API does not return a
  structured `.parsed` payload.

When providers return schema-constrained payloads the parser calls `parse_schema_constrained_payload`. Otherwise it falls
back to `parse_structured_output`, raising `PromptEvaluationError` on any mismatch. If `parse_output=False`, the parser
skips structured parsing entirely and simply returns the assistant text.

## Deadlines, Sessions, and Overrides

- The `Deadline` passed to `evaluate` (or attached to the rendered prompt) flows into `ConversationRunner`, each
  tool's `ToolContext`, and every error message produced by deadline guards. The runner rejects requests immediately if
  `deadline.remaining()` is already non-positive.
- `SessionProtocol` provides snapshots, rollbacks, and typed slices. Tool handlers use it for stateful reducers,
  workspace digests (`set_workspace_digest` / `clear_workspace_digest`), and domain-specific caches. Because tool event
  publication can trigger reducer failures, adapters roll back the session snapshot whenever a publish fails.
- The same session's `event_bus` should be injected into `evaluate` so reducers listening for `PromptRendered`,
  `ToolInvoked`, and `PromptExecuted` events stay in sync with the state slices.
- `PromptOverridesStore` is optional yet supported by every adapter. Callers provide the store plus a tag (defaults to
  `"latest"`) to render prompts with the same overrides system used by the CLI.

## Optimization Workflow

`ProviderAdapter.optimize(...)` streamlines workspace digest refreshes:

1. Locate the prompt's `WorkspaceDigestSection` and workspace section (`PodmanSandboxSection` or `VfsToolsSection`).
   Failure to find either raises `PromptEvaluationError` with `phase="request"`.
1. Clone both sections with a fresh inner `Session`/`EventBus` so destructive tools (filesystem exploration, podman
   sandboxes) cannot mutate the caller's session.
1. Compose the optimization prompt (namespace `f"{prompt.ns}.optimization"` and key `f"{prompt.key}-workspace-digest"`).
1. Evaluate the prompt through `evaluate` with `parse_output=True`. Any overrides/tag passed to `optimize` are forwarded.
1. Extract digest content from `PromptResponse.output` (`str` or dataclass with a `digest` field) or fall back to
   `PromptResponse.text`. Empty responses raise `PromptEvaluationError` with `phase="response"`.
1. Store the digest based on the requested `OptimizationScope` (session or overrides store).

The returned `OptimizationResult` includes the original `PromptResponse` so callers can inspect the assistant reasoning
that produced the digest.

## Built-in Adapters

### `LiteLLMAdapter`

- Optional dependency enabled via `uv sync --extra litellm`.
- Accepts either a concrete `completion` callable or a factory + kwargs that wrap `litellm.completion`.
- Forwards `model`, rendered system instructions, tools, tool choice, and JSON schema response format to LiteLLM.
- Because LiteLLM proxies downstream providers, `require_structured_output_text=True` ensures we still see a readable
  assistant message even if the provider does not populate a structured payload.
- Tool choice defaults to `"auto"` but callers can provide any supported `ToolChoice` literal/mapping.
- Provider failures (network, authentication, model errors) are wrapped in `PromptEvaluationError` with
  `phase="request"`.

### `OpenAIAdapter`

- Optional dependency enabled via `uv sync --extra openai`.
- Accepts either a concrete OpenAI client (`openai.OpenAI`) or a factory + kwargs. The helper raises a descriptive
  `RuntimeError` if the package is missing.
- Calls `client.chat.completions.create(model=..., messages=..., tools=..., tool_choice=..., response_format=...)`.
- When `use_native_response_format=True` the adapter disables prompt-level output instructions any time structured
  output parsing is requested. This prevents duplicate requirements because OpenAI now enforces the JSON schema.
- `response_format` is only passed when structured output is requested **and** native response formats are enabled.
  Otherwise OpenAI renders plain text output and the parser falls back to prompt-based instructions.
- OpenAI responses include native `parsed` content when response formats are enabled, allowing
  `require_structured_output_text=False`.

## Implementing New Adapters

1. **Decide on an `AdapterName`** – Extend `_names.py` if you need a new identifier and export it from
   `src/weakincentives/adapters/__init__.py`.
1. **Instantiate the provider client** – Follow the LiteLLM/OpenAI examples by accepting either a concrete client or a
   factory so tests can inject fakes.
1. **Render the prompt** – Always call `prompt.render` exactly once, respecting the `override_store`, `tag`, and optional
   instruction toggles when structured output parsing needs provider help.
1. **Delegate to `run_conversation`** – Supply provider-specific `call_provider` and `select_choice` functions plus any
   adapter-specific defaults (`tool_choice`, `response_format`, `require_structured_output_text`). The helper handles
   logging, deadlines, events, tool execution, and structured output.
1. **Wrap SDK failures** – Catch provider-specific exceptions and re-raise `PromptEvaluationError` with
   `phase="request"` so callers can respond consistently.
1. **Expose extras** – If the adapter requires optional dependencies, raise a helpful `RuntimeError` mirroring the
   LiteLLM/OpenAI error strings when the import fails.

Following these conventions ensures every adapter integrates seamlessly with the prompt orchestration runtime, event bus,
session reducers, and optimization workflows described across the rest of the specs.
