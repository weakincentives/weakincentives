# wink MCP server requirements

## Purpose

The `wink` CLI exists to start a Model Context Protocol (MCP) server that
exposes the prompt override workflows already implemented in
`weakincentives`. MCP-compatible clients (Codex, Claude Desktop, etc.) should
be able to list, inspect, and mutate overrides while the core library handles
storage, validation, and rendering.

## Packaging

- Ship the CLI and every runtime dependency required to host the MCP server
  under a single optional extra (for example,
  `[project.optional-dependencies.wink]` in `pyproject.toml`). Installing
  `weakincentives[wink]` must provide the `wink` entry point and the MCP
  transport/bindings so the server comes up without additional manual installs.

## Runtime

1. `wink mcp` (or an equivalent command) starts the server, advertises its
   tools, and blocks until the client disconnects.
1. Resolve configuration from CLI flags and a static config file (TOML or YAML)
   that captures override directories, environment flags, and any optional
   authentication. Document a sample configuration so headless launchers can
   boot deterministically.
1. Emit structured, single-line logs and map failures onto MCP error semantics
   so clients receive actionable diagnostics.

## Tools

Expose a minimal set of MCP tools, each acting on a specific section or tool
definition so clients can make precise changes.

- **`wink.list_overrides`** — Accepts an optional namespace filter. Returns
  namespaces, prompt keys, tags, and overridden handles along with lightweight
  metadata (paths, hashes, timestamps) suitable for UI pickers. Propagate
  storage or permission errors through MCP error payloads.
- **`wink.get_section_override`** — Takes `ns`, `prompt`, `tag` (default
  `latest`), and a slash-delimited `section_path`. Returns the rendered body,
  descriptor metadata, expected hash, and backing file path. When the section
  exists but is unmodified, return explicit `null` values so clients can seed a
  new override without extra calls.
- **`wink.write_section_override`** — Accepts namespace, prompt key, tag,
  section path, replacement Markdown, and optional guards such as
  `expected_hash` or `descriptor_version`. Validate the descriptor path,
  confirm hash alignment, and persist only the targeted section through
  `PromptOverridesStore.upsert`. Require a `confirm` flag and surface precise
  validation errors.
- **`wink.delete_section_override`** — Accepts namespace, prompt key, tag, and
  section path. Remove the specific section while leaving other overrides
  intact. Delete the override file if it becomes empty, but treat missing
  entries as no-ops that still return structured warnings.
- **`wink.get_tool_override`** — Accepts namespace, prompt key, tag, and tool
  name. Return the override payload (description, parameter metadata, contract
  hash, backing path) or descriptor defaults when no override exists.
- **`wink.write_tool_override`** — Accepts namespace, prompt key, tag, tool
  name, override payload, and an optional `expected_contract_hash`. Validate
  against the descriptor, merge the payload without disturbing other entries,
  persist via `PromptOverridesStore.upsert`, and require the same `confirm`
  semantics as section writes.
- **`wink.delete_tool_override`** — Accepts namespace, prompt key, tag, and
  tool name. Remove only the targeted tool override, deleting the file when it
  becomes empty. Missing entries should return structured warnings rather than
  errors.

## Client integration

- Publish a sample configuration for Codex/Claude that invokes
  `wink mcp --config ~/.config/wink/config.toml`, sets `WINK_OVERRIDES_DIR`, and
  pins the working directory. Ensure the server shuts down cleanly once the
  client disconnects.
- Keep tool responses compact to stay within MCP token budgets, and always
  include namespace, prompt key, and tag context so conversational clients can
  chain follow-up calls.

## Operations

- Provide sensible default locations for override directories and config
  files, while allowing environment variable overrides for multi-client
  setups.
- Emit clear diagnostics when validation fails or when an override cannot be
  located; MCP clients pass these messages directly to users.
- Surface backing file paths in mutation responses so users can version-control
  overrides.

## Extensibility

Keep the initial surface area lean. Additional helpers (diff generation,
descriptor discovery, streaming notifications) can arrive later without
disrupting the base MCP workflow.
