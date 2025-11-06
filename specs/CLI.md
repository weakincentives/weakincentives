# wink MCP Server Specification

## Purpose

The `wink` CLI now exists solely to expose a Model Context Protocol (MCP)
server. Instead of mediating interactions directly, the CLI spins up a server
that can be registered in Codex, Claude Desktop, or any other MCP-compatible
client. The MCP surface gives downstream chat interfaces the ability to inspect
and mutate prompt overrides while relying on `weakincentives` for the heavy
lifting.

## Execution Model

- `wink mcp` (or an equivalent entry point) should start the server, advertise
  its capabilities, and block until the hosting client disconnects.
- Configuration is file-based: read overrides storage paths, environment
  settings, and authentication (if any) from a static config file or CLI flags
  so that Codex/Claude can launch the server deterministically.
- Logging must default to structured, single-line entries that downstream
  clients can tail for debugging without overflowing tokens.

## MCP Capabilities

Expose the minimum viable tool set required for working with prompt overrides.

### `wink.list_overrides`

- **Input**: optional namespace filter.
- **Behavior**: enumerate namespaces, prompt keys, and tags known to the
  configured `PromptOverridesStore`. Return lightweight metadata so MCP clients
  can present pickers before making a specific request.
- **Errors**: bubble storage or permission issues using MCP error semantics so
  clients can display actionable messages.

### `wink.get_override`

- **Input**: namespace (`ns`), prompt key (`prompt`), and tag (`tag`, default
  `latest`).
- **Behavior**: resolve the override via `PromptOverridesStore`, returning the
  JSON payload, descriptor hints, and backing file path.
- **Usage**: primary read primitive invoked by Codex/Claude when a user asks to
  inspect an override.

### `wink.upsert_override`

- **Input**: namespace, prompt key, tag (default `latest`), and a replacement
  JSON payload.
- **Behavior**: validate the payload against descriptor expectations when
  available, persist the change through `PromptOverridesStore.upsert`, and
  return the updated metadata (including the file path).
- **Safety**: require an explicit `confirm` flag so clients must opt in before
  mutations occur.

## Client Integration Notes

- Provide a sample Codex/Claude MCP configuration that points at the CLI entry
  point, including expected environment variables and working directory
  assumptions. For example:

  ```json
  {
    "command": "wink",
    "args": ["mcp", "--config", "~/.config/wink/config.toml"],
    "env": {
      "WINK_OVERRIDES_DIR": "~/weakincentives/overrides"
    },
    "cwd": "~/weakincentives"
  }
  ```

- Keep outputs compact; Claude and Codex impose strict token budgets on MCP tool
  responses.

- Make sure every response includes enough context (namespace, key, tag) so chat
  clients can craft follow-up prompts without additional round-trips.

## Operational Guidance

- Ship with sensible defaults for override directories and config file
  locations, but allow overrides through environment variables so multiple
  clients can target different stores.
- Emit explicit diagnostics when validation fails or when an override cannot be
  found; MCP clients surface these errors verbatim to users.
- Encourage downstream users to version control their overrides by surfacing the
  backing file paths in every mutation response.

## Extensibility

- Additional tools (for example, diff generation or descriptor discovery) can be
  layered on later, but keep the default surface minimal to reduce client setup
  complexity.
- Consider advertising text-streaming or notification channels once the MCP
  ecosystem stabilizes; for now, stick to synchronous tool calls.
