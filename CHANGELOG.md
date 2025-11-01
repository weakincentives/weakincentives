# Changelog

Release highlights for weakincentives.

## Unreleased

### Prompt

- Renamed several prompt authoring primitives for clarity, including `MarkdownSection`,
  `SectionNode`, `ToolResult.value`, and `parse_structured_output`, and consolidated
  the `prompt/` module layout under a new top-level import surface.
- Added required prompt namespaces and explicit section keys so overrides map cleanly to
  rendered content and response formats.
- Extended prompt descriptors with tool metadata hashing and override support to track
  tool changes across versions.

### Events

- Implemented the event emission spec with typed prompt and tool lifecycle events and
  wired the adapters and examples to publish them.

### Session

- Introduced a session state container that collects emitted event payloads and exposes
  built-in reducers and selectors for downstream agents.

### Integrations

- Added a LiteLLM adapter behind the new `litellm` extra with full tool execution, structured output parsing, and telemetry parity with the existing OpenAI integration.
- Updated the OpenAI adapter to attach native JSON schema response formats, prefer parsed
  structured outputs, tighten `tool_choice` typing, and avoid echoing tool payloads in tool
  role messages.

### Examples

- Rebuilt the OpenAI and LiteLLM demos as CLI entry points backed by a shared repository-
  aware code review agent scaffold.

### Packaging

- Lowered the supported Python baseline to 3.12 and curated package exports to match the
  new module layout.

### Dependencies

- Raised the optional `litellm` extra to require the latest upstream release.

### Documentation

- Documented the LiteLLM optional dependency, updated installation instructions, and introduced a dedicated example script showcasing LiteLLM usage.
- Refreshed the README messaging to highlight the new prompt workflow, minimum Python
  version, and example entry points.

## v0.2.0 - 2025-10-29

### Highlights

- Launched the prompt composition system with typed `Prompt`, `Section`, and `TextSection` building blocks, structured rendering, and placeholder validation backed by comprehensive tests.
- Added tool orchestration primitives including the `Tool` dataclass, shared dataclass handling, duplicate detection, and prompt-level aggregation utilities.
- Delivered stdlib-only dataclass serde helpers (`parse`, `dump`, `clone`, `schema`) for lightweight validation and JSON serialization.

### Integrations

- Introduced an optional OpenAI adapter behind the `openai` extra that builds configured clients and provides friendly guidance when the dependency is missing.

### Developer Experience

- Tightened the quality gate with quiet wrappers for Ruff, Ty, pytest (100% coverage), Bandit, Deptry, and pip-audit, all wired through `make check`.
- Adopted Hatch VCS versioning, refreshed `pyproject.toml` metadata, and standardized automation scripts for releases.

### Documentation

- Replaced `WARP.md` with a comprehensive `AGENTS.md` handbook describing workflows, TDD guidance, and integration expectations.
- Added prompt and tool specifications under `specs/` and refreshed the README to highlight the new primitives and developer tooling.

## v0.1.0 - 2025-10-22

Initial repository bootstrap with the package scaffold, testing and linting toolchain, CI configuration, and contributor documentation.
