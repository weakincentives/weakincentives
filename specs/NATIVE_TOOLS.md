# Native Tool Specification

## Introduction

Large language model providers now offer built-in capabilities—web search, file search,
code execution, patching, and shell access—without relying on client-side handlers. A
`NativeTool` represents these provider-managed affordances in the same way the runtime
models traditional, client-implemented tools. This specification explains how to extend
`Tool` with a marker interface that captures provider-owned execution while preserving the
existing prompt, schema, and telemetry contracts.

Provider docs for reference:

- OpenAI web search: https://platform.openai.com/docs/guides/tools-web-search?api-mode=responses
- OpenAI code interpreter: https://platform.openai.com/docs/guides/tools-code-interpreter
- OpenAI apply_patch: https://platform.openai.com/docs/guides/tools-apply-patch
- OpenAI shell: https://platform.openai.com/docs/guides/tools-shell

## Goals

- Expose provider-native capabilities through the same section-first tooling pipeline used
  by client-side tools.
- Preserve type-safe inputs/outputs so validation and structured telemetry remain
  consistent regardless of where execution happens.
- Emit `ToolInvoked` events for native calls so sessions retain a uniform audit trail.
- Allow adapters to advertise provider-native affordances without requiring local
  implementations or sandbox access.

Non-goals:

- Implementing provider-specific networking or credential flow; adapters remain
  responsible for transport and authentication.
- Reproducing provider UIs; native tools surface the provider's own result payloads.

## Terminology

- **Client tool** – A `Tool` with a runtime `handler` executed in-process.
- **Native tool** – A provider-managed capability declared via `NativeTool` and executed on
  the provider. No local handler runs.
- **Provider** – The LLM platform offering built-in tools (e.g., OpenAI Responses API).

## Interface Overview

`NativeTool[ParamsT, ResultT]` is a marker interface layered on top of the existing
`Tool[ParamsT, ResultT]` contract:

- **Marker only**: It does not introduce new fields beyond `Tool`; it signals to adapters
  that execution is delegated to the provider.
- **Schema-carrying**: `name`, `description`, `params_type`, `result_type`, and
  `examples` remain mandatory for validation and documentation.
- **Handler-free**: `handler` must be `None`. Runtime dispatchers never attempt to execute
  local code for a native tool.
- **Override posture**: `accepts_overrides` defaults to `False` because provider-native
  affordances generally cannot be redirected to client code without changing semantics.
  Sections may opt in explicitly when safe.

## Declaration Rules

- Native tools live alongside client tools in prompt sections; registration and validation
  reuse the same pipelines documented in `TOOLS.md`.
- Tool names and descriptions follow the existing validation rules; providers still expect
  OpenAI-style function naming constraints.
- Params and results must be dataclasses satisfying `SupportsDataclass` and
  `SupportsToolResult` respectively. This preserves schema inference for adapters and
  keeps rendered examples coherent.
- Examples SHOULD document realistic invocations to help providers steer request routing,
  just as with client tools.

## Adapter Responsibilities

Adapters map `NativeTool` declarations to provider-specific payloads:

- **Capability selection**: Adapters MUST include native tools only when the provider
  advertises support (e.g., OpenAI Responses API "tools" array with `"type": "web_search"`,
  `"code_interpreter"`, `"file_search"`, `"shell"`, or `"apply_patch"`).
- **Schema projection**: When providers expose fixed shapes (e.g., web search parameters),
  adapters SHOULD validate that `ParamsT` matches the provider contract and SHOULD reject
  mismatches during prompt validation.
- **Result hydration**: Adapter responses MUST be converted into `ToolResult` instances so
  downstream reducers observe a consistent container. Native payloads SHOULD be preserved
  in `ToolResult.value` where possible and rendered via `ToolRenderableResult.render()`.
- **Error mapping**: Provider errors map to `ToolResult(success=False, value=None)` with a
  descriptive `message`. Adapters SHOULD include provider error codes in the message for
  auditability.

## OpenAI Implementation and Integration Points

OpenAI wiring currently depends on the standard tool normalization path and the new
`NativeTool` marker:

- The `NativeTool` class in `weakincentives.prompt.tool` enforces the marker semantics by
  defaulting `accepts_overrides` to `False` and rejecting any supplied `handler`,
  preserving the provider-managed execution contract while reusing the `Tool`
  initialization pipeline.
- The OpenAI adapter builds the `tools` payload in
  `OpenAIAdapter._build_provider_invoker`, normalizing each spec via
  `_responses_tool_spec`. That helper currently **rejects any tool whose `type` is not
  `"function"`**, so native tool entries must extend that function to accept OpenAI's
  native tool types (`"web_search"`, `"code_interpreter"`, `"file_search"`,
  `"shell"`, and `"apply_patch"`). The adapter will include the resulting specs in the
  `tools` array passed to `client.responses.create` and honor tool choice directives via
  `_responses_tool_choice`.
- When OpenAI returns a tool call, the existing tool-call parsing helpers
  (`parse_tool_arguments`, `_tool_call_from_output`, and `_tool_calls_from_content`) still
  apply because native calls arrive in the same shape as function calls; adapters only
  need to skip local handler execution when a `NativeTool` marker is present.

## OpenAI Usage Examples

Native tools are declared exactly like client tools, but without a handler. The examples
below use simplified payloads; real payloads should mirror the provider's response schema
so result rendering stays faithful.

```python
from dataclasses import dataclass

from weakincentives.prompt import NativeTool, ToolExample


@dataclass
class WebSearchParams:
    query: str


@dataclass
class WebSearchResult:
    # Capture the provider payload verbatim; rendering is handled by ToolResult.
    payload: dict[str, object]


web_search_tool = NativeTool[WebSearchParams, WebSearchResult](
    name="web_search",
    description="Use the provider's built-in search to gather fresh results.",
    examples=(
        ToolExample(
            description="Search for current weather in Paris.",
            input=WebSearchParams(query="current weather in Paris"),
            output=WebSearchResult(payload={"results": []}),
        ),
    ),
)

# Register alongside client tools inside a prompt section. When the OpenAI adapter is
# updated to accept native tool types in `_responses_tool_spec`, this declaration will be
# serialized into the `tools` array without requiring a local handler.
```

## Invocation Lifecycle

Native tools flow through the same runtime stages as client tools, with a different
execution step:

1. **Selection** – Provider selects a native tool name from the advertised tool list.
1. **Dispatch** – The runtime forwards the request to the adapter; no local handler is
   called.
1. **Provider execution** – The adapter relays the call to the provider's native tool API
   and awaits completion.
1. **Result normalization** – Adapter converts the provider payload into `ToolResult` and
   invokes `render()` for the textual tool message.
1. **Telemetry** – A `ToolInvoked` event is emitted with the same shape as client tools:
   tool name, params, rendered result, success flag, and payload. A flag indicating the
   native origin SHOULD be included for debugging but MUST NOT change downstream event
   consumption semantics.
1. **Session aggregation** – Reducers and transcripts treat the invocation identically to a
   client-side tool call, ensuring the session history remains unified.

## Event Contract

- Native tool invocations MUST emit `ToolInvoked` events even though no local handler
  executes. This keeps session timelines and rollbacks coherent.
- Events SHOULD carry provider metadata (provider identifier, capability type) in the
  event payload to aid observability without altering the contract for consumers that only
  expect the generic fields.
- Event ordering and retries mirror the client-tool semantics described in `EVENTS.md`;
  adapters SHOULD surface provider retry information in messages rather than creating new
  event types.

## Security and Sandbox Considerations

- Because native tools execute remotely, the runtime MUST NOT assume local sandboxing or
  filesystem access. Any local staging (e.g., file uploads for code interpreter) must flow
  through existing VFS tools or adapter-managed uploads.
- Native tools SHOULD be disabled automatically when provider capabilities are unavailable
  or prohibited for the session's trust level. Sections MAY gate registration behind
  feature flags to avoid surprising exposure.

## Compatibility and Extensibility

- Client code that inspects `Tool` instances SHOULD treat `NativeTool` polymorphically;
  equality, hashing, and repr semantics remain unchanged.
- Future provider-native capabilities can be added by introducing new enum-like constants
  or helper constructors without changing the marker interface.
- Adapters SHOULD centralize capability-to-provider mappings (e.g., within adapter-specific
  registries) to avoid scattering literal strings across sections.

## Testing Expectations

- Unit tests MUST validate that native tool declarations fail when a handler is provided or
  when params/results violate the base `Tool` contract.
- Adapter tests SHOULD cover successful invocations, provider error mapping, and
  `ToolInvoked` event emission parity with client tools.
- Integration tests SHOULD assert that session transcripts include native tool entries with
  rendered outputs and structured payloads intact.
