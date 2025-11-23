# Wink CLI Debug Server

## Goals and Scope

- Add a `wink debug <snapshot-path>` subcommand that loads a session snapshot and
  serves a read-only FastAPI app for inspecting its contents.
- Optimize for the demo snapshots stored under `snapshots/`, but accept any valid
  snapshot JSON path (absolute or relative).
- Keep the feature self-contained in the CLI package; it should not alter
  runtime session behavior or snapshot serialization formats.
- FastAPI (plus the ASGI server) ships via the `"wink"` extra so the base install
  remains dependency-light.

## Non-Goals

- No live session capture or write-back into snapshots; the server is read-only.
- No remote deployment concerns (auth, TLS, reverse proxies). The server binds to
  localhost by default and targets local debugging only.
- No front-end build step or new asset pipeline; ship static HTML/JS served from
  the FastAPI app.

## CLI Contract (wink debug)

- Invocation: `uv run --extra wink wink debug snapshots/<id>.json [options]`.
- Required argument:
  - `snapshot_path`: path to a snapshot JSON file. Resolve relative paths against
    the current working directory. Surface a clear error and exit non-zero if the
    file is missing or invalid.
- Options:
  - `--host` (default `127.0.0.1`) and `--port` (default `8000`) for the server
    bind address.
  - `--open-browser/--no-open-browser` (default on): open the default browser to
    the UI root after the server starts; log the URL if opening fails.
  - Inherit existing logging flags (`--log-level`, `--json-logs`) and apply them
    to both CLI startup and FastAPI/uvicorn logs.
- Exit codes:
  - `0` on clean shutdown.
  - `2` on argument/validation errors (bad path, unreadable JSON).
  - `3` on server startup failures (port in use, FastAPI import errors).

## Snapshot Loading and Validation

- Reuse `Snapshot.from_json` and `SnapshotPayload` from
  `weakincentives.runtime.session.snapshots` to parse the provided file.
- Validate at startup before binding the server:
  - File must exist and be readable.
  - JSON must match the snapshot schema version; surface the underlying
    `SnapshotRestoreError` message to the user.
- Store the parsed `Snapshot` in memory for fast API responses. Provide a manual
  reload endpoint that re-reads the file and replaces the cached snapshot so
  users can iterate without restarting the server. Reload failures should return
  a 400 response with the validation message and leave the prior snapshot intact.

## Server and API Surface

- Build a FastAPI app in `weakincentives.cli` (e.g., `debug_app.py`) to avoid
  mixing parsing logic with `argparse` wiring.
- Mount routes under `/api`:
  - `GET /api/meta` → `{created_at, slices:[{slice_type, item_type, count}]}`.
  - `GET /api/slices/{slice_type}` → `{slice_type, item_type, items:[...]}`
    where `slice_type` is URL-encoded (use `urllib.parse.quote`). Allow
    `?limit=` and `?offset=` query params for large slices.
  - `GET /api/raw` → the entire snapshot payload as loaded (for download).
  - `POST /api/reload` → reloads from disk and returns the new meta payload.
- Serve a static `index.html` (and a small CSS/JS bundle) from `/` that consumes
  the API. No additional build tooling; ship assets as strings or files under
  `src/weakincentives/cli/static/`.
- Run the app via uvicorn inside the `debug` command to avoid an extra entry
  point. Gracefully shut down on SIGINT/SIGTERM.

## Web UI Expectations

- Purpose-built for exploring snapshot contents quickly:
  - A metadata header with snapshot path, creation timestamp, and schema version.
  - Slice list panel showing `slice_type`, `item_type`, and item counts with a
    search filter.
  - Detail view that renders items as formatted JSON with copy-to-clipboard and
    collapsible long fields. Preserve ordering from the snapshot.
  - Raw JSON download button plus a reload button that calls `/api/reload` and
    refreshes the views.
- Keep the UI responsive without a build step: vanilla JS or a tiny helper like
  htmx/htmx-style fetch patterns inlined in the page. Avoid adding new frontend
  dependencies to the Python extras.
- Style lightly for readability (monospace for JSON blocks, wrapping for long
  text bodies, sticky slice list on wide screens).

## Packaging and Operations

- Add `fastapi` and `uvicorn` to the `"wink"` optional dependency group. Tests
  using `--all-extras` already install extras; ensure the new extra keeps that
  contract.
- Keep the `wink` console script target unchanged; subcommands live inside the
  existing entrypoint module.
- Document the new command in `README.md` (or a dedicated CLI section) with an
  example using `snapshots/<id>.json`.

## Testing and Observability

- Unit tests:
  - `argparse` coverage for required path and options.
  - Snapshot load failure paths (missing file, invalid JSON, schema mismatch).
  - FastAPI app tested with `TestClient` for meta, slice detail, raw, and reload
    routes (including pagination parameters).
  - Browser-opening helper mocked so tests stay headless.
- Keep coverage at 100% across the new module(s). Prefer small helpers so tests
  avoid spinning up uvicorn.
- Emit structured logs for server lifecycle events (`debug.server.start`,
  `debug.server.reload`, `debug.server.error`) respecting the existing logging
  configuration.
