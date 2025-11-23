# OpenAI Responses API Migration

## Overview

This document specifies how `weakincentives` migrates the OpenAI adapter from
`ChatCompletion` calls to the OpenAI **Responses API**. It enumerates contract
changes, SDK configuration, runtime expectations, and integration-test updates
needed to keep side-effect-free agent runs stable during and after the
transition.

### Goals and Scope

- Replace Chat Completions usage in the OpenAI adapter with the Responses API
  while preserving the adapter's public interfaces and DbC guarantees.
- Maintain feature parity: prompt sections, native structured outputs, streaming
  tokens, tool execution, and error translation must keep working with no
  behavior regressions.
- Keep compatibility toggles so callers can opt out (or fall back) until the
  migration is fully validated in production.
- Extend integration tests to cover responses-based flows, including streaming
  and tool call handling.

### Non-Goals

- Modifying prompt templating, section rendering, or tool registration APIs.
- Introducing new provider-specific prompt metadata beyond what Responses
  already requires.
- Rewriting general structured output parsing; only the provider binding
  changes.

## Request Construction

1. **Entry Point**: Update the OpenAI adapter to build `client.responses.create`
   payloads. The adapter must continue to accept the existing prompt-to-request
   inputs and translate them into `input` + `modalities` rather than
   `messages`.
1. **System and User Content**: Render prompt sections into a single markdown
   string inserted into the `input` array. Preserve the ordering of system and
   user chapters; ensure assistant-prefill messages are encoded as
   `{"role": "assistant", "content": ...}` entries inside `input` when
   necessary for guardrails.
1. **Tools and Functions**: Map each registered tool definition into
   `tools=[{"type": "function", ...}]` and pass `tool_choice` exactly as
   configured by the prompt metadata. Validate that the DbC preconditions reject
   mismatched tool names or missing schemas before issuing the request.
1. **Structured Outputs**: When a prompt declares a structured output type,
   attach `response_format={"type": "json_schema", "json_schema": ...}`.
   Preserve the fallback path to instruction-based JSON when the target model
   does not advertise native structured outputs.
1. **Streaming**: Use `stream=true` and pass a `ResponsesStream` iterator into
   the runtime's streaming loop. The adapter must emit the same incremental
   events (`on_delta`, `on_tool_call`, `on_final`) expected by existing
   consumers, translating response segments into the runtime's event model.
1. **Metadata and Logging**: Include request identifiers, model name, and token
   limits in the adapter's debug logging so existing observability hooks remain
   usable. Preserve `client_timeout` semantics when constructing the Responses
   client call.

## Response Handling

1. **Content Assembly**: For non-streaming calls, extract the final text from
   `response.output[0].content[0].text`. For streaming, concatenate
   `output_text.delta` chunks while respecting tool calls and finish reasons.
1. **Tool Calls**: Convert any `response.output[0].content` entries of type
   `"input_text"` + `tool_calls` into the adapter's tool invocation format.
   Ensure tool call arguments are parsed as JSON and validated against the
   registered schema before dispatching.
1. **Finish Reasons and Safety**: Map Responses finish reasons (`stop`,
   `length`, `tool_calls`, `content_filter`) into the adapter's normalized enum.
   Raise assertion errors when content filtering blocks required output so DbC
   invariants continue to enforce safe fallbacks.
1. **Error Translation**: Wrap SDK exceptions into the adapter's existing error
   types. Preserve retry classification (timeouts vs. rate limits vs. bad
   requests) to keep higher-level retry policies intact.

## Configuration and Compatibility

- **Model Selection**: Default models should shift to the Responses-compatible
  `gpt-4.1` and `gpt-4o` families. Keep an escape hatch to force the legacy
  Chat Completions path for models that are not yet supported.
- **Feature Flags**: Add a provider-level flag (e.g., `use_responses=True`) so
  integration tests and early adopters can toggle the new path. The flag must be
  plumbed through `OpenAIAdapterConfig` and respected by the runtime.
- **Backwards Compatibility**: When `use_responses` is disabled, the adapter
  must retain current behavior. When enabled with a non-supported model, raise a
  clear configuration error rather than silently downgrading behavior.
- **Telemetry**: Record whether Responses is enabled in any emitted analytics or
  debug metadata so downstream dashboards can track adoption.

## Integration Test Plan

1. **Adapter Smoke Tests**: Add a new integration test module under
   `integration-tests/` that exercises the Responses path end-to-end with a real
   API key. Cover:
   - Basic text completion and stop reason handling.
   - Streaming deltas with incremental token accumulation.
   - Tool call discovery and execution round-trips.
   - Native structured output parsing for a simple dataclass response.
1. **Feature Flag Coverage**: Extend existing adapter tests to parameterize over
   `use_responses` on/off. The off-path should continue to use Chat Completions;
   the on-path must assert that the Responses-specific request envelope is used
   (e.g., via mocked client calls).
1. **Fallback Validation**: Add tests that force an unsupported model with
   `use_responses=True` and assert that a configuration error is raised with a
   descriptive message.
1. **Streaming Harness**: Update any test helpers that previously assumed
   `ChatCompletionChunk` types to accept Responses streaming objects. Keep
   fixture factories that synthesize stream segments for deterministic tests.
1. **Tool Contract Tests**: Ensure existing DbC enforcement for tool schemas is
   re-run against Responses-derived tool calls by adjusting helper functions that
   normalize tool payloads.
1. **CI Execution**: Gate new integration tests behind the existing
   `make integration-tests` target so that they only run when `OPENAI_API_KEY`
   is present. Unit tests covering the adapter should run under `make test` with
   mocked Responses clients to maintain 100% coverage without network access.

## Rollout Checklist

- Land adapter changes behind the `use_responses` flag with full unit coverage.
- Update documentation and changelog entries describing the new flag and default
  model expectations.
- Run `make integration-tests` against production credentials to validate the
  streaming, structured output, and tool call scenarios end-to-end.
- Flip the default configuration to `use_responses=True` once integration tests
  are stable and telemetry confirms parity.
