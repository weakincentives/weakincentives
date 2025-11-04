# wink CLI Specification

## Purpose

`wink` is the command-line interface that ships with `weakincentives`. It
provides a developer-focused surface for inspecting and manipulating prompt
override files managed by the local overrides store without directly editing the
filesystem. The CLI targets day-to-day prompt curation workflows and mirrors the
Git-style editor integration developers already expect.

## Scope

- Define the user-facing contract for the initial CLI release.
- Cover the CRUD feature set for prompt overrides that are stored via
  `LocalPromptOverridesStore`.
- Describe how the tool integrates with editors, configuration, and structured
  output.

Out of scope: remote stores, prompt execution, optimizer pipelines, and
non-interactive scripting utilities. Future revisions can layer those behaviors
behind additional subcommands.

## Command Overview

### Invocation Pattern

```
wink [GLOBAL OPTIONS] <command> [ARGS]
```

- Commands return exit code `0` on success and `1` on recoverable validation
  errors. Unexpected failures propagate non-zero exit codes surfaced by Python
  exceptions.
- Global options MUST appear before the subcommand.

### Global Options

| Flag | Description |
| ---- | ----------- |
| `--root PATH` | Override the project root auto-discovery described in the local store spec. The flag mirrors the `root_path` constructor argument and is resolved before any filesystem access. |
| `--format {table,json}` | Select output renderer. `table` (default) prints human-readable tables; `json` emits machine-friendly JSON. |
| `--editor CMD` | Override the editor invocation for the current command. Defaults to `$WINK_EDITOR`, then `$VISUAL`, then `$EDITOR`, finally `vi`. |
| `--yes` | Assume "yes" for destructive confirmations (e.g., delete). |
| `-q/--quiet` | Suppress non-error informational output (useful for scripting). |

Global options propagate to subcommands via a shared configuration object.

### Shared Arguments

CRUD commands accept the identifier triple that maps directly to the underlying
store:

- `--ns TEXT` (required)
- `--prompt TEXT` (required; corresponds to `prompt_key`)
- `--tag TEXT` (defaults to `latest`)

For commands that support listing, the identifiers can be used as filters (e.g.
`wink list --ns demo`).

## Subcommand Specifications

### `wink list`

Lists overrides discovered under the current root.

- Without filters, emits one row per override file with the columns `ns`,
  `prompt`, `tag`, `sections`, `tools`, and `path`.
- `sections` and `tools` show counts, not full payloads.
- Supports `--ns`, `--prompt`, and `--tag` filters. Filters apply after walking
  the filesystem so the command remains stateless.
- When `--format json` is requested, the output is a list of objects containing
  the full `PromptOverride` metadata, including relative paths and hash summary
  information. JSON output uses snake_case keys to align with the dataclass
  names.

### `wink show`

Prints the contents of a single override.

- Requires the identifier triple.
- The CLI resolves the override via `PromptOverridesStore.resolve`. Missing
  overrides cause a `PromptNotFound` style error message and exit code `1`.
- Default output renders:
  - Header: `ns/prompt:tag` with the backing file path.
  - Section table: one row per section path, showing the current hash and the
    first line of the body.
  - Tool table: one row per tool override.
  - Full bodies follow after a separator unless `--summary-only` is set.
- JSON output dumps the full override payload.

### `wink edit`

Creates or updates an override in the system editor.

1. Resolve the descriptor via `PromptRegistry` so the CLI can call
   `seed_if_necessary` when no override exists. Failure to locate the prompt
   descriptor returns an error instructing the user to verify the namespace and
   key.
2. Materialize the current override payload as JSON in a temporary file. When
   invoked with `--create-only`, the CLI seeds an override and exits without
   opening the editor.
3. Determine the editor command in priority order: `--editor`, `$WINK_EDITOR`,
   `$VISUAL`, `$EDITOR`, fallback to `vi`. If no editor can be resolved, abort
   with instructions to set `$EDITOR`.
4. Launch the editor in blocking mode. The CLI must:
   - Preserve `$EDITOR` semantics (respect quoted arguments, etc.).
   - Detect when the user exits without modifications and print `No changes`.
   - Re-open the editor if the buffer contains invalid JSON and the user agrees
     to retry; otherwise abort with the invalid content path for manual recovery.
5. Parse the modified JSON into `PromptOverride`, validate section and tool
   hashes against the descriptor, and call `PromptOverridesStore.upsert`.
6. Print a success message describing the stored file path and any warnings about
   skipped sections (e.g., stale hashes filtered by the store).

### `wink delete`

Removes an override file.

- Requires the identifier triple.
- Prints the target file path and prompts for confirmation unless `--yes` is
  provided.
- Calls `PromptOverridesStore.delete`. Missing files are treated as a warning and
  return exit code `0` so the command is idempotent.

### `wink diff`

Shows a diff between the override and the current prompt descriptor materialized
from code. Although not strictly required for CRUD, the diff helps validate
updates before committing them.

- Requires the identifier triple.
- Internally renders the descriptor sections and tool descriptions using the
  same seeding logic as `seed_if_necessary` to produce the baseline.
- When `--editor` or `$EDITOR` references a visual diff tool, the CLI MUST honor
  the setting; otherwise fall back to `unified diff` in the terminal using
  `difflib.unified_diff`.
- Return exit code `1` if the override is missing or if no differences are
  detected (enabling use in scripts that gate on new changes).

## Editor Integration

- `wink` MUST propagate environment variables (`EDITOR`, `VISUAL`, etc.) to the
  spawned process unmodified.
- Temporary files live under `$TMPDIR` (or platform equivalents) and include the
  prompt identifier in the filename for discoverability.
- The CLI deletes temp files after a successful save. On failure, temp files are
  retained and the path is printed for debugging.
- When the editor process returns a non-zero exit code, the CLI treats the run as
  aborted and does not write to disk.

## Configuration Resolution

In addition to CLI flags and environment variables, wink supports a project
configuration file located at `<root>/.weakincentives/wink.toml`.

- `wink` reads the config lazily on first access.
- Supported keys:
  - `[ui] default_format = "json" | "table"`
  - `[ui] confirm_deletes = true | false`
  - `[paths] editor = "code --wait"`
- CLI flags override config values; config overrides environment defaults.

## Error Handling & Messaging

- Validation issues (unknown prompt, invalid JSON, hash mismatch) emit concise
  error summaries plus a hint about remediation. Use stderr for errors, stdout
  for structured output.
- Tracebacks are suppressed by default. `WINK_DEBUG=1` enables full tracebacks
  to aid development.
- Subcommands respect `--quiet` by suppressing success banners while still
  emitting warnings.

## Testing Hooks

- Provide high-level integration tests that exercise editor flows using a fake
  editor script. The script should write deterministic edits to the temp file so
  the CLI can run non-interactively.
- List and show commands should have snapshot-style tests for both table and JSON
  formats to prevent regressions in the text interface.
- Editor fallback ordering and temp file cleanup should be covered by unit tests
  around the helper that resolves the editor command and manages subprocess
  invocation.

