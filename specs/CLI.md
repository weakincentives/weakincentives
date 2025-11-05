# wink CLI Specification

## Purpose

`wink` is the command-line interface that ships with `weakincentives` to manage
prompt overrides stored by the local overrides store. The CLI exists to make it
simple to create, inspect, update, and remove overrides without hand-editing
files.

## Scope

This specification covers only the CRUD workflow for prompt overrides backed by
`LocalPromptOverridesStore`.

## Invocation

```
wink [GLOBAL OPTIONS] <command> [ARGS]
```

- Exit code `0` indicates success; user-facing validation failures return `1`.
- Global options appear before the subcommand.

### Global Options

| Flag | Description |
| ---- | ----------- |
| `--root PATH` | Override automatic project root discovery. |
| `--format {table,json}` | Choose human-readable (`table`) or machine-readable (`json`) output. |
| `--editor CMD` | Override the editor used by commands that open files. |
| `--yes` | Skip confirmation prompts on destructive actions. |

### Shared Identifiers

CRUD commands accept the identifier triple that maps to the override path:

- `--ns TEXT` (required)
- `--prompt TEXT` (required; maps to `prompt_key`)
- `--tag TEXT` (defaults to `latest`)

Filters reuse the same identifiers.

## Subcommands

### `wink list`

List overrides within the resolved root.

- Without filters, show one row per override with `ns`, `prompt`, `tag`, and the
  relative `path`.
- Accept `--ns`, `--prompt`, and `--tag` filters.
- JSON output returns an array of objects containing the same fields.

### `wink show`

Display the contents of a single override.

- Requires the identifier triple.
- On success, print a header (`ns/prompt:tag`) and the full JSON body.
- Missing overrides return a validation error.

### `wink edit`

Create or update an override using the system editor.

1. Resolve the prompt descriptor; fail fast when it cannot be found.
1. Seed a temporary JSON file with the current override (or descriptor default).
1. Launch the editor defined by `--editor`, `$WINK_EDITOR`, `$VISUAL`, `$EDITOR`,
   or `vi`.
1. After the editor exits, parse the JSON and call
   `PromptOverridesStore.upsert`.
1. Report the stored file path.

### `wink delete`

Remove an override file.

- Requires the identifier triple.
- Prompt for confirmation unless `--yes` is set.
- Call `PromptOverridesStore.delete`. Missing files exit with code `0`.

## Output Guidelines

- Errors emit short messages on stderr.
- Success output goes to stdout and respects `--format`.
- No structured diff or non-CRUD functionality is required in this revision.
