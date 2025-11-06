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
- Any helper routines that need an LLM session must default to the `gpt-5`
  model through the OpenAI adapter, while still allowing explicit overrides via
  configuration.

## MCP Capabilities

Expose the minimum viable tool set required for working with prompt overrides.
All read and write operations must target individual sections or tools so
clients can manage fine-grained changes without clobbering entire prompt
payloads.

### `wink.list_overrides`

- **Input**: optional namespace filter.
- **Behavior**: enumerate namespaces, prompt keys, tags, and the section/tool
  handles currently overridden within each prompt. Return just enough metadata
  (paths, hashes, last-modified timestamps) for MCP clients to render pickers
  without loading full payloads.
- **Errors**: bubble storage or permission issues using MCP error semantics so
  clients can display actionable messages.

### `wink.get_section_override`

- **Input**: namespace (`ns`), prompt key (`prompt`), tag (`tag`, default
  `latest`), and section path (`section_path`, slash-delimited).
- **Behavior**: retrieve a single section override, including the rendered body,
  expected hash, descriptor metadata, and backing file path. Return `null`
  values when the section exists but is currently unmodified so clients can
  seed new overrides without extra calls.
- **Usage**: preferred read primitive for Codex/Claude when a user is editing a
  specific section of a prompt.

### `wink.write_section_override`

- **Input**: namespace, prompt key, tag (default `latest`), section path, and
  replacement markdown body. Accept optional `expected_hash` or
  `descriptor_version` inputs so clients can guard against stale edits.
- **Behavior**: validate the section path against the prompt descriptor,
  confirm hash alignment, and persist only the targeted section body via
  `PromptOverridesStore.upsert`. Preserve untouched sections and tool entries in
  the existing override file.
- **Safety**: require an explicit `confirm` flag so clients must opt in before
  mutations occur. Surface precise error messages when validation fails so
  conversational clients can offer corrective guidance.

### `wink.delete_section_override`

- **Input**: namespace, prompt key, tag (default `latest`), and section path.
- **Behavior**: remove the section entry from the override file while leaving
  other sections and tool overrides intact. If the last override is removed,
  delete the file entirely so future reads fall back to prompt defaults.
- **Errors**: treat missing entries as no-ops but return structured messages so
  clients can inform users when there was nothing to delete.

### `wink.get_tool_override`

- **Input**: namespace, prompt key, tag (default `latest`), and tool name.
- **Behavior**: load a single tool override including description, parameter
  metadata, expected contract hash, and backing file path. Return descriptor
  defaults when the tool has no override yet.

### `wink.write_tool_override`

- **Input**: namespace, prompt key, tag (default `latest`), tool name, and the
  override payload (description plus parameter descriptions). Accept optional
  `expected_contract_hash` to protect against stale descriptors.
- **Behavior**: validate the tool against the prompt descriptor, merge the new
  payload into the existing override file without disturbing other entries, and
  persist via `PromptOverridesStore.upsert`.
- **Safety**: require the same `confirm` flag semantics as section writes.

### `wink.delete_tool_override`

- **Input**: namespace, prompt key, tag (default `latest`), and tool name.
- **Behavior**: remove the specific tool override while keeping other entries
  untouched. When the override file becomes empty, delete it to signal that the
  prompt has returned to defaults.
- **Errors**: missing tool entries should not error; report a structured
  warning to the client instead.

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
