# Changelog

Release highlights for weakincentives.

## Unreleased

- Added an `asteval`-powered Python evaluation tool section that bridges the
  sandbox with the virtual filesystem, including timeout handling, templated
  writes, and captured stdout/stderr telemetry.
- Declared the `asteval` runtime dependency to support the new evaluation
  tooling.

## v0.3.0 - 2025-11-01

### Prompt & Rendering

- Renamed and reorganized the prompt authoring primitives (`MarkdownSection`,
  `SectionNode`, `Tool`, `ToolResult`, `parse_structured_output`, …) under the
  consolidated `weakincentives.prompt` surface.
- Prompts now require namespaces and explicit section keys so overrides line up with
  rendered content and structured response formats.
- Added tool-aware prompt version metadata and the `PromptVersionStore` override
  workflow to track section edits and tool changes across revisions.

### Session & State

- Introduced the `Session` container with typed reducers/selectors that capture prompt
  outputs and tool payloads directly from emitted events.
- Added helper reducers (`append`, `replace_latest`, `upsert_by`) and selectors
  (`select_latest`, `select_where`) to simplify downstream state management.

### Built-in Tools

- Shipped the planning tool suite (`PlanningToolsSection` plus typed plan dataclasses)
  for creating, updating, and tracking multi-step execution plans inside a session.
- Added the virtual filesystem tool suite (`VfsToolsSection`) with host-mount
  materialization, ASCII write limits, and reducers that maintain a versioned snapshot.

### Events & Telemetry

- Implemented the event bus with `ToolInvoked` and `PromptExecuted` payloads and wired
  adapters/examples to publish them for sessions or external observers.

### Adapters

- Added a LiteLLM adapter behind the `litellm` extra with tool execution parity and
  structured output parsing.
- Updated the OpenAI adapter to emit native JSON schema response formats, tighten
  `tool_choice` handling, avoid echoing tool payloads, and surface richer telemetry.

### Examples

- Rebuilt the OpenAI and LiteLLM demos as shared CLI entry points powered by the new
  code review agent scaffold, complete with planning and virtual filesystem sections.

### Tooling & Packaging

- Lowered the supported Python baseline to 3.12 (the repository now pins 3.14 for
  development) and curated package exports to match the reorganized modules.
- Added OpenAI integration tests and stabilized the tool execution loop used by the
  adapters.
- Raised the optional `litellm` extra to require the latest upstream release.

### Documentation

- Documented the planning and virtual filesystem tool suites, optional provider extras,
  and updated installation guidance.
- Refreshed the README and supporting docs to highlight the new prompt workflow,
  adapters, and development tooling expectations.

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
