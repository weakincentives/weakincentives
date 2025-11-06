# Native OpenAI Structured Outputs

## Introduction

The OpenAI adapter already renders prompts into a `RenderedPrompt` object that carries the final markdown, declared output dataclass, container type, and extra-key policy. Structured outputs today are achieved by instructing the model to emit fenced JSON and scraping the response via `parse_structured_output`. This spec captures the changes needed for the adapter to take advantage of OpenAI's native structured outputs while preserving the existing fallback path.

## Goals

- **Schema-Driven Requests**: Derive JSON Schema definitions from the prompt's structured output metadata and attach them to OpenAI requests via the provider's `response_format` contract.
- **Provider Parsing First**: When the SDK returns parsed content, convert it directly into the target dataclass and bypass regex-based scraping.
- **Resilient Fallbacks**: Keep the current instruction injection and text parsing so older models or providers without schema support continue to work.
- **Transparent API**: Contain all changes within the OpenAI adapter so public Prompt and adapter interfaces remain unchanged.

## Non-Goals

- Modifying the prompt templating API or its dataclass definition helpers.
- Introducing streaming or asynchronous evaluation paths.
- Changing provider-agnostic structured output parsing utilities.

## Provider Schema Derivation

1. Inspect `RenderedPrompt.output_type`, `output_container`, and `allow_extra_keys`.
1. Use the existing serde helpers to build a JSON Schema fragment for the target dataclass. Pass `extra="forbid"` when extra keys are disallowed.
1. When `output_container == "array"`, wrap the schema in an array definition with `items` pointing to the dataclass schema.
1. Generate a deterministic schema `name` (e.g., based on `prompt.name`) so repeated calls stay stable.
   - The OpenAI `json_schema["name"]` field accepts only ASCII letters, digits, underscores, and hyphens (`^[A-Za-z0-9_-]+$`) and must be 1–64 characters long; enforce these limits during generation.
   - Sanitize prompt names (e.g., via slugification that lowercases, replaces invalid characters with hyphens, and trims consecutive separators) before truncating to the 64-character ceiling to remain within OpenAI's constraints.

## Request Payload Updates

- During `OpenAIAdapter.evaluate`, detect structured output metadata on the rendered prompt.
- Attach a `response_format` block to the request payload with:
  - `type="json_schema"`.
  - `json_schema={"name": generated_name, "schema": derived_schema}`.
- Leave existing payload construction untouched when the prompt does not declare structured output.

## Response Handling

1. When OpenAI returns parsed content (e.g., via `message.parsed` or a content part typed as `output_json`), deserialize that payload directly into the target dataclass using the existing serde parsing helpers.
1. Skip the fallback JSON extraction when the parsed payload succeeds.
1. If the provider response lacks parsed data, continue using `parse_structured_output` on the assistant's text message.
1. Propagate failures as `PromptEvaluationError` with the `"response"` phase to align with existing error handling.

## Prompt Instructions

- Preserve the automatic "Response Format" section appended during prompt rendering unless authors opt out via `inject_output_instructions=False`.
- This ensures non-OpenAI adapters and legacy models still receive explicit formatting guidance.

## Testing

- Extend OpenAI adapter tests to assert that `response_format` is populated whenever structured output metadata is present.
- Add fixtures covering provider responses with parsed content to validate the new happy path.
- Ensure tests continue to cover the legacy fallback path where only text is returned.
- Add a unit asserting sanitized schema names stay within the OpenAI character set and length limits.
