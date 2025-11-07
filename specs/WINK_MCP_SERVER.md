# wink MCP Server Specification

## Overview

The `wink` CLI now serves exclusively as the launcher for a Model Context Protocol (MCP) server. The CLI should expose the prompt override surface area already implemented in `weakincentives`, letting Codex, Claude Desktop, and other MCP-compatible clients inspect and mutate overrides while delegating all storage and validation to the core library.

## Runtime & Dependencies

- **Python & package manager**: Target Python 3.12 and orchestrate installs with `uv`, matching the rest of the repository.
- **Project extras**: The server code depends on the OpenAI adapter to reach the `gpt-5` default. Document that operators must run `uv sync --extra openai` (or `pip install weakincentives[openai]`) before launching the server. Keep helper routines defensive if the extra is missing.
- **MCP protocol bindings**: Add a dedicated optional dependency group for the MCP server (for example, `model-context-protocol>=0.1.0` and any transport helpers). Ship a `uv` task, e.g. `uv sync --group mcp`, so agents can prepare a fully wired environment with a single command.
- **Local configuration**: The server reads overrides directories, environment flags, and optional auth tokens from a TOML or YAML config file. Provide a sample config and reference paths in documentation so headless clients can launch deterministically.
- **Logging & diagnostics**: Standardize on structured, single-line logs (JSON or key-value) and expose `--log-level` and `--log-format` flags to let hosting clients control verbosity.

## Execution Model

1. `wink mcp` (or an equivalent entry point) spins up the MCP server, advertises available tools, and blocks until the client disconnects.
2. Resolve configuration strictly from CLI flags and a static config file so Codex/Claude can start the server without environment discovery.
3. Helper routines that need an LLM session should default to the `gpt-5` model via the OpenAI adapter but accept overrides in the config.
4. Emit structured logs throughout startup, capability registration, and shutdown. Errors must map to MCP error semantics so clients can surface actionable messages.

## MCP Capabilities

Expose the smallest viable toolset for prompt override workflows. Each tool must operate on individual sections or tools to avoid clobbering complete override files.

### `wink.list_overrides`

- **Input**: Optional namespace filter.
- **Output**: Namespaces, prompt keys, tags, and handles for overridden sections/tools. Include metadata (storage path, hash, last modified timestamp) so clients can render pickers without fetching full payloads.
- **Errors**: Propagate storage and permission issues through MCP error payloads.

### `wink.get_section_override`

- **Input**: `ns`, `prompt`, `tag` (default `latest`), `section_path` (slash-delimited).
- **Output**: Rendered body, expected hash, descriptor metadata, and the backing file path. Return explicit `null` values when the section exists but is unmodified.
- **Usage**: Primary read primitive when a user edits a specific section.

### `wink.write_section_override`

- **Input**: Namespace, prompt key, tag (default `latest`), section path, replacement Markdown body, and optional `expected_hash` or `descriptor_version` guards.
- **Behavior**: Validate the section path against the prompt descriptor, ensure hash alignment, and persist only the targeted section via `PromptOverridesStore.upsert` while leaving other entries untouched.
- **Safety**: Require an explicit `confirm` flag before applying mutations. Provide precise validation errors for conversational guidance.

### `wink.delete_section_override`

- **Input**: Namespace, prompt key, tag (default `latest`), section path.
- **Behavior**: Remove the specified section from the override file without affecting other entries. Delete the file entirely when all overrides are gone so reads fall back to defaults.
- **Errors**: Missing entries are treated as no-ops but return structured warnings.

### `wink.get_tool_override`

- **Input**: Namespace, prompt key, tag (default `latest`), tool name.
- **Behavior**: Return the tool override (description, parameters, contract hash, backing path) or descriptor defaults if no override exists.

### `wink.write_tool_override`

- **Input**: Namespace, prompt key, tag (default `latest`), tool name, override payload (description and parameter metadata), optional `expected_contract_hash` guard.
- **Behavior**: Validate the tool against the descriptor, merge the payload into the existing override file, and persist via `PromptOverridesStore.upsert`.
- **Safety**: Enforce the same `confirm` flag semantics as section writes.

### `wink.delete_tool_override`

- **Input**: Namespace, prompt key, tag (default `latest`), tool name.
- **Behavior**: Remove the target tool override while preserving other entries. Delete the override file when it becomes empty to restore defaults.
- **Errors**: Surface structured warnings when the tool entry is absent.

## Client Integration

- Publish a sample MCP client configuration that invokes `wink mcp --config ~/.config/wink/config.toml`, sets `WINK_OVERRIDES_DIR`, and pins the working directory. Confirm that the server exits cleanly when the client disconnects so headless launchers do not hang.
- Keep tool responses terse. Claude and Codex enforce tight token budgets; return only the metadata a client needs to render UI controls or stage follow-up calls.
- Include namespace, prompt key, and tag in every response payload so conversational clients can compose subsequent requests without extra lookups.

## Operational Guidance

- Ship sensible defaults for override directories and config file locations, while still honoring environment variable overrides for multi-client setups.
- Emit detailed diagnostics when validation fails or an override cannot be found. MCP clients surface these messages verbatim.
- Encourage users to version control overrides by including the backing file path in all mutation responses.

## Extensibility

- Keep the default tool surface minimal. Additional helpers (diffs, descriptor discovery, streaming notifications) can arrive later without breaking existing clients.
- Evaluate adding streaming or notification channels once the MCP ecosystem stabilizes; the initial release should stick to synchronous tool calls.
