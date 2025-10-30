# LiteLLM Adapter Specification

## Overview
- Extend the adapters package with a LiteLLM-backed implementation that mirrors the OpenAI adapter's behavior and ergonomics.
- Preserve existing evaluation loops, tool execution semantics, structured output handling, and event publication expectations to maintain consistency across providers.
- Surface LiteLLM as an officially supported optional dependency and document how consumers can adopt it alongside the current OpenAI integration.

## Goals
1. Provide a drop-in LiteLLM adapter that matches the OpenAI adapter's API surface, lifecycle, and error handling so callers can switch providers without code changes.
2. Ensure parity in automated test coverage, including dependency guards, tool invocation flows, structured outputs, and error conditions.
3. Update packaging, documentation, and examples so LiteLLM becomes a first-class integration path for end users.

## Non-Goals
- Introducing provider-specific features beyond what the OpenAI adapter already exposes.
- Refactoring the broader adapter architecture or event model.
- Implementing live end-to-end tests that require external LiteLLM credentials (optional stretch goal only).

## Key Design Considerations
- **Module symmetry**: Replicate the structure of `src/weakincentives/adapters/openai.py` when implementing `litellm.py`, including factory helpers, adapter classes, and event publishing patterns. This keeps feature behavior and ergonomics identical between providers.
- **Dependency guarding**: Add a loader helper that raises a clear runtime error when `litellm` is missing, mirroring `_load_openai_module()` so optional installs behave predictably.
- **Tool execution**: Preserve the existing tool invocation loop, including how tool selections are surfaced, executed, and logged via `ToolInvoked` events.
- **Structured output**: Maintain the structured output parsing and validation logic so responses stay compatible with the higher-level prompt framework.
- **Event parity**: Continue emitting `PromptExecuted` and related telemetry events with the same payload shapes to avoid downstream ingestion changes.
- **Adapter exports**: Update `src/weakincentives/adapters/__init__.py` to export the new adapter and helper protocols so external consumers can import LiteLLM alongside OpenAI.
- **Packaging**: Introduce a `litellm` optional dependency in `pyproject.toml`, matching the structure of the existing `openai` extra to keep installation flows aligned.

## Implementation Plan
1. **Adapter module**
   - Create `src/weakincentives/adapters/litellm.py` with the Apache license header.
   - Define the LiteLLM adapter class and completion factory, reusing the OpenAI adapter's control flow while swapping in LiteLLM client constructs (e.g., `litellm.completion`).
   - Guard imports and expose helpers to simplify testing and dependency injection.
2. **Exports and registry**
   - Update adapter package `__init__` to include LiteLLM symbols and ensure any registries remain synchronized.
3. **Testing**
   - Add `tests/adapters/test_litellm_adapter.py` by adapting the OpenAI test suite.
   - Validate missing dependency errors, injected completion callables, plain responses, tool loops, structured outputs, and error propagation.
   - Share or refactor test utilities as needed to avoid duplication between OpenAI and LiteLLM suites.
4. **Optional integration tests** (nice-to-have)
   - If desired, add integration coverage that runs only when LiteLLM credentials are provided via environment variables, mirroring OpenAI's optional tests.

## Documentation & Packaging Updates
- Update the README optional extras section to describe installing with `litellm` support and include a short usage snippet.
- Either generalize the existing OpenAI example or add a new LiteLLM-specific example script to illustrate usage.
- Record the addition in `CHANGELOG.md` (or similar release notes) to highlight the new adapter.

## Rationale
- Maintaining parity with the OpenAI adapter ensures consumers can swap providers without learning new semantics, preserving the current developer experience.
- Mirroring the OpenAI test coverage provides confidence that LiteLLM integration adheres to established quality and behavior expectations.
- Documenting the new optional dependency and example workflows lowers adoption friction and signals official support for LiteLLM-based setups.

## Open Questions
- Should LiteLLM integration tests run in CI, or remain opt-in via environment markers?
- Are there provider-specific configuration nuances (rate limits, retries) that warrant future enhancements beyond the initial parity-focused implementation?

