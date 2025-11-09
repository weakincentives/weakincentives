# wink MCP specification

## Purpose and goal

`wink` is an MCP-only surface. Its sole responsibility is to start and host the
wink Model Context Protocol (MCP) server that exposes prompt override workflows
implemented inside `weakincentives`. The server must be built with the FastMCP
server framework running on anyio so the project follows its async conventions
while remaining portable across runtimes.

## Scope

- Define the packaging, configuration, runtime, and tool contract for the
  FastMCP+anyio server.
- Document the expectations clients have when interacting with the MCP tools
  that proxy `LocalPromptOverridesStore`.

## Packaging

- Ship the server entry point and every dependency required to host the
  FastMCP+anyio runtime under a single optional extra (for example,
  `[project.optional-dependencies.wink]` in `pyproject.toml`). Installing
  `weakincentives[wink]` must provide the `wink` executable and all MCP
  transport/bindings so the server comes up without additional installs.

## Invocation and configuration

- `wink [GLOBAL OPTIONS]` starts the server, advertises its tools, and blocks
  until the client disconnects.
- Provide a `--config PATH` flag that points to a TOML or YAML file describing
  the override directories, environment switches, client auth, and logging
  preferences.
- Support environment variables—such as `WINK_OVERRIDES_DIR`, `WINK_CONFIG`, and
  `WINK_ENV`—that override file-based settings for deterministic headless
  launches.
- Emit structured, single-line logs to stdout/stderr, map failures onto MCP
  error semantics, and return process exit code `0` on graceful shutdowns.

## MCP server runtime

1. Boot the FastMCP server under anyio, wiring the configured prompt overrides
   store and descriptor loaders into each tool handler.
1. Resolve configuration from CLI flags, config files, and environment variables
   (in that precedence order) so deployments can stay declarative.
1. Register the wink namespace, advertise tools, and hold the event loop open
   until the connected MCP client disconnects.

## MCP tools

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
  `expected_hash` or `descriptor_version`. Validate the descriptor path, confirm
  hash alignment, and persist only the targeted section through
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
- **`wink.delete_tool_override`** — Accepts namespace, prompt key, tag, and tool
  name. Remove only the targeted tool override, deleting the file when it
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

- Provide sensible default locations for override directories and config files,
  while allowing environment variable overrides for multi-client setups.
- Emit clear diagnostics when validation fails or when an override cannot be
  located; MCP clients pass these messages directly to users.
- Surface backing file paths in mutation responses so users can
  version-control overrides.

## Extensibility

Keep the initial surface area lean. Additional helpers (diff generation,
descriptor discovery, streaming notifications) can arrive later without
disrupting the base MCP workflow.

## Implementation status

Outstanding work items tracked here until the MCP server ships:

- Add the FastMCP + anyio dependencies (and any runtime glue) to the `wink`
  optional extra so installing `weakincentives[wink]` brings up the full stack.
- Simplify the CLI to launch directly as `wink [GLOBAL OPTIONS]` with no `mcp`
  subcommand, and ensure the entry point calls into the new runtime.
- Replace the `run_mcp_server` stub with a real FastMCP server that wires up
  configuration loading, prompt registries, the overrides store, and shutdown
  handling under anyio.
- Register the MCP tool definitions (`wink.list_overrides`, `wink.get_*`,
  `wink.write_*`, `wink.delete_*`) with FastMCP, delegating to the existing
  helper functions and surfacing MCP-compliant error payloads.
- Implement the `wink.list_overrides` helper plus any remaining store adapters,
  covering the metadata contract required by the spec.
- Document and check in a sample configuration file that demonstrates
  `--config`, environment overrides, and listener/auth settings for common MCP
  clients.
