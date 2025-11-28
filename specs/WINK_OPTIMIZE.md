# Wink CLI Optimize Mode

`wink optimize` is a thin frontend over the local prompt overrides store. It
loads overrides from a snapshot and provides an in-browser editor to tweak and
persist those overrides back to disk—no runtime mutation, no remote services.

## Goals and Scope

- Add a `wink optimize <snapshot-path>` subcommand that reads the existing
  snapshot format used by `wink debug` and exposes the prompt override store for
  editing.
- Present each prompt's `PromptDescriptor` with its associated local override
  entries, allowing per-field edits and saving.
- Keep all behavior confined to the CLI package and the local filesystem; the
  command is purely a frontend on top of the stored overrides.

## Non-Goals

- No live session capture or runtime state changes beyond editing overrides in
  the snapshot-derived store.
- No new persistence backend; edits are saved back to the same snapshot file.
- No authentication or remote access; the experience targets local manual
  tuning.

## CLI Contract (wink optimize)

- Invocation: `uv run --extra wink wink optimize snapshots/<id>.json [options]`.
- Required argument:
  - `snapshot_path`: path to a snapshot JSON file, validated exactly like `wink debug` (resolve relative paths, fail fast on missing/invalid files).
- Options:
  - `--host` (default `127.0.0.1`) and `--port` (default `8000`).
  - `--open-browser/--no-open-browser` (default on) to launch the UI after
    startup.
  - Inherit logging flags (`--log-level`, `--json-logs`).
- Exit codes mirror `wink debug` (`0` success, `2` validation errors, `3` server
  failures).

## Data Loading and Override Store

- Load the snapshot via `Snapshot.from_json`; reuse the same schema as `wink debug` without adding slices.
- Seed the editable list from `PromptRendered` entries because they carry the
  `PromptDescriptor` payload; `PromptExecuted` events do not include descriptor
  or override data.
- Populate the in-memory override store from the snapshot's override slice when
  present; otherwise initialize empty overrides keyed by descriptor identity so
  the UI still renders descriptors even when no overrides were captured.
- Key the store by prompt identity (descriptor name/version plus execution
  index) so edits remain traceable to specific prompt executions.
- Preserve the original execution order when listing prompts to maintain the
  provenance of each override set.

## Override Editing Flow

- The UI edits only the local override store; untouched fields remain
  unchanged.
- Supported override fields include model selection, temperature, system text,
  tool allowance, tone, and other prompt-local knobs already present in the
  snapshot. Reject unknown fields.
- A per-prompt "Save" persists the edited overrides into the in-memory store
  immediately so subsequent API calls and UI renders show the new values.
- A global "Save snapshot" writes the entire snapshot (with updated override
  store) back to `snapshot_path`, replacing the original file after validating
  write permissions.
- A "Reset changes" control reloads the snapshot from disk to repopulate the
  override store, discarding unsaved edits.

## Server and API Surface

- Implement a dedicated FastAPI app in `weakincentives.cli` (e.g.,
  `optimize_app.py`). Keep argument parsing separate from request handling.
- Routes under `/api` operate directly on the in-memory override store:
  - `GET /api/prompts` → list prompt executions with descriptor metadata and
    override fields present in the store.
  - `GET /api/prompts/{prompt_id}` → return the `PromptDescriptor`, current
    overrides, and original snapshot values for comparison.
  - `POST /api/prompts/{prompt_id}/overrides` → accept partial override updates,
    apply them to the store, and return the updated state; reject invalid fields
    with a 400.
  - `POST /api/save` → serialize the snapshot (including the updated override
    store) to disk and return confirmation; errors retain in-memory edits and
    respond with 400.
  - `POST /api/reset` → reload the snapshot, rebuild the override store, and
    return the refreshed prompts list.
- Keep responses compact; avoid streaming large transcripts unless required for
  context.

## Web UI Expectations

- Focused on editing the local override store quickly:
  - Prompt list panel showing descriptor name/version and a badge for prompts
    with unsaved override edits.
  - Detail view with side-by-side current override values and original snapshot
    defaults, with inline editing controls (text inputs, selects, toggles as
    appropriate).
  - Per-prompt Save (updates the override store) plus global Save Snapshot and
    Reset buttons.
- Keep the UI lightweight (vanilla JS or minimal helper) and reuse `wink debug`
  styling patterns for consistency.
- Surface success/error toasts or inline messages for save/reset operations.

## Testing and Observability

- Unit tests:
  - Argument parsing for `wink optimize` and parity with `wink debug`
    validations.
  - Snapshot loading that builds the override store from `PromptRendered`
    records (and tolerates missing override slices).
  - API routes for listing prompts, retrieving details, applying override
    updates, saving, and resetting (including invalid-field and IO error
    paths).
  - Serialization of the snapshot with the updated override store without
    altering unrelated slices.
- Structured logs for lifecycle events (`optimize.server.start`,
  `optimize.server.save`, `optimize.server.reset`, `optimize.server.error`),
  honoring configured logging flags.

## Packaging and Operations

- Ship via the existing `"wink"` optional dependency group; avoid expanding the
  dependency footprint beyond what `wink debug` uses.
- Keep the `wink` console entrypoint unchanged; register `optimize` alongside
  `debug`.
- Document the new command in CLI docs with an example and a note that it is a
  frontend for the local prompt override store built from the snapshot format.
