# Native OpenAI Structured Outputs

## Introduction

The OpenAI adapter already renders prompts into a `RenderedPrompt` object that
carries the final markdown, declared output dataclass, container type, and
extra-key policy. Structured outputs today are achieved by instructing the
model to emit fenced JSON and scraping the response via
`parse_structured_output`. This document functions as both a design note for the
native OpenAI structured output path and a reference for the current
implementation and fallbacks.

## Goals, Scope, and Design Principles

- **Scope**: Enable native structured outputs for OpenAI chat completions while
  retaining backward-compatible behavior for other providers and older OpenAI
  models.
- **Schema-Driven Requests**: Derive JSON Schema definitions from the prompt's
  structured output metadata and attach them to OpenAI requests via the
  provider's `response_format` contract.
- **Provider Parsing First**: When the SDK returns parsed content, convert it
  directly into the target dataclass and bypass regex-based scraping.
- **Resilient Fallbacks**: Keep the current instruction injection and text
  parsing so providers without schema support continue to work without API
  changes.
- **Transparent API**: Contain changes within the OpenAI adapter so public
  prompt and adapter interfaces remain unchanged.

### Non-Goals

- Modifying the prompt templating API or its dataclass definition helpers.
- Introducing streaming or asynchronous evaluation paths.
- Changing provider-agnostic structured output parsing utilities.

## Current Design and Implementation

### Provider Schema Derivation

1. Inspect `RenderedPrompt.output_type`, `output_container`, and
   `allow_extra_keys`.
1. Use the serde helpers to build a JSON Schema fragment for the target
   dataclass. Pass `extra="forbid"` when extra keys are disallowed so OpenAI can
   reject unexpected fields.
1. When `output_container == "array"`, wrap the schema in an array definition
   with `items` pointing to the dataclass schema.
1. Generate a deterministic schema `name` (for example, derived from
   `prompt.name`) so repeated calls stay stable.
   - The OpenAI `json_schema["name"]` field accepts only ASCII letters, digits,
     underscores, and hyphens (`^[A-Za-z0-9_-]+$`) and must be 1–64 characters
     long; enforce these limits during generation.
   - Sanitize prompt names (e.g., via slugification that lowercases, replaces
     invalid characters with hyphens, and trims consecutive separators) before
     truncating to the 64-character ceiling to remain within OpenAI's
     constraints.

### Request Payload Construction

- During `OpenAIAdapter.evaluate`, detect structured output metadata on the
  rendered prompt.
- Attach a `response_format` block to the request payload with:
  - `type="json_schema"`.
  - `json_schema={"name": generated_name, "schema": derived_schema}`.
- Leave existing payload construction untouched when the prompt does not
  declare structured output so non-structured calls remain identical.

### Response Handling

1. When OpenAI returns parsed content (for example, via `message.parsed` or a
   content part typed as `output_json`), deserialize that payload directly into
   the target dataclass using the serde parsing helpers.
1. Skip the fallback JSON extraction when the parsed payload succeeds so the
   OpenAI-provided structure remains authoritative.
1. If the provider response lacks parsed data, continue using
   `parse_structured_output` on the assistant's text message.
1. Propagate failures as `PromptEvaluationError` with the `"response"` phase to
   align with existing error handling and monitoring.

### Prompt Instructions

- Preserve the automatic "Response Format" section appended during prompt
  rendering unless authors opt out via `inject_output_instructions=False`.
- This ensures non-OpenAI adapters and legacy models still receive explicit
  formatting guidance even when native schemas are available.

## Fallbacks and Compatibility

- Native structured outputs are opportunistic. The adapter only adds
  `response_format` when the rendered prompt declares structured output
  metadata; otherwise requests and behavior are unchanged.
- If OpenAI declines to return parsed content (unsupported model, transient API
  behavior, or schema mismatch), the legacy regex-based scraping path remains in
  place and continues to enforce dataclass shapes via serde.
- Instruction injection stays enabled by default to provide deterministic JSON
  fences for providers that ignore schema hints or for callers targeting models
  without native support.
- Error surfaces are preserved: failures during either the native or fallback
  path surface as `PromptEvaluationError` in the response phase.

## Migration Notes and Caveats

- Schema name generation must respect the provider's character set and length
  rules to avoid request-time failures; prompts with long or highly punctuated
  names need sanitization before truncation.
- Allow-listing extra keys via `allow_extra_keys` changes the generated schema
  and downstream validation behavior; callers relying on permissive parsing
  should verify the chosen setting across both native and fallback paths.
- Because instruction injection remains enabled, prompt authors should avoid
  duplicating or conflicting formatting guidance; the adapter will continue to
  append the "Response Format" section unless explicitly disabled.
- Contract coverage remains identical to the legacy flow: serde enforces the
  dataclass contract whether data is parsed by OpenAI or scraped from text.

## Testing Expectations

- Extend OpenAI adapter tests to assert that `response_format` is populated
  whenever structured output metadata is present.
- Add fixtures covering provider responses with parsed content to validate the
  new happy path end-to-end.
- Keep tests that cover the legacy fallback path where only text is returned so
  regressions in regex scraping are caught early.
- Add a unit asserting sanitized schema names stay within the OpenAI character
  set and length limits.
