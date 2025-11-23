# Podman Sandbox Specification

## Overview

`PodmanSandboxSection` exposes a Podman-backed execution surface so agents can
run shell commands, manipulate files, and evaluate small Python snippets inside
an isolated Linux container. The section mirrors the ergonomics of the VFS and
ASTEval tools while mapping every call onto a shared Podman workspace rooted at
`/workspace`. This document describes how the section is composed, which session
state reducers back it, and how the sandbox interacts with subagents running at
different isolation levels.

## Rationale

- **Safety-first isolation** – Podman offers a rootless, reproducible runtime
  that keeps experiments and generated patches contained, protecting both the
  host and higher-trust agents.
- **Parity with in-memory tooling** – By mirroring VFS and ASTEval contracts,
  Podman executions behave like the in-process tools the rest of the platform
  relies on, minimizing cognitive overhead when swapping surfaces.
- **Traceable orchestration** – The section threads workspace metadata through
  reducers so orchestrators, subagents, and prompt templates can always explain
  where files live and which container executed a command.

## Scope

- Applies to the Podman-backed sandbox section (`PodmanSandboxSection`) and its
  helper dataclasses (`PodmanWorkspace`, `PodmanShellParams`, etc.).
- Covers runtime configuration (image, limits, mounts), session reducer wiring,
  VFS-compatible tools, and the evaluate_python shim described below.
- Excludes broader prompt composition rules or non-Podman runtimes; see other
  specs (e.g., `specs/VFS_TOOLS.md`, `specs/AST_EVAL.md`) for those.

## Guiding Principles

- **Predictability over flexibility** – Defaults (image, limits, user, mount
  layout) are fixed so `shell_execute` and the VFS tools behave the same across
  orchestrators. Override only through the documented configuration knobs in the
  workspace lifecycle and operational notes sections.
- **Single source of truth** – The reducers enumerated in the session state
  section back every file mutation and execution record. Avoid shadow state
  outside these slices so prompt sections and subagents stay consistent.
- **Minimal host risk** – All tooling resolves paths through `_host_path_for`
  and the hydration pipeline to prevent escapes. Retain the copy-then-exec
  model for writes and removals so binary safety and include/exclude guards stay
  enforced.
- **Reference-friendly** – Treat this document as the operational guide for the
  sandbox: the module layout lists where to dig into code, workspace lifecycle
  details how containers are shaped, and operational notes enumerate the
  supported configuration and caveats.

## Goals

- **Deterministic Container Sandbox** – Provide a podman-managed environment
  with fixed resource limits (1 CPU, 1 GiB RAM, readonly host, no network) so
  shell tooling behaves the same across orchestrators.
- **Unified Filesystem Contract** – Mirror the `VfsToolsSection` APIs so agents
  can interact with container files through the familiar `ls`, `read_file`,
  `write_file`, `glob`, `grep`, `rm`, and `evaluate_python` tools.
- **Reducer-Friendly State** – Keep session slices (`PodmanWorkspace`,
  `VirtualFileSystem`, etc.) in sync with container activity so higher-level
  automation (subagents, prompt overrides, planning tooling) can consume the
  same data structures regardless of backend.
- **Subagent Compatibility** – Ensure that delegated runs inherit or isolate the
  workspace according to `SubagentIsolationLevel`, matching the session cloning
  rules in `specs/SUBAGENTS.md`.

## Non-Goals

- Running privileged containers or attaching to arbitrary host paths outside the
  configured overlay.
- Providing long-running background services; the sandbox is optimized for
  short, synchronous tool invocations (≤120 seconds).
- Supporting network access or custom Podman images at runtime; image selection
  is static when constructing the section.

## Module Layout

- Implementation: `src/weakincentives/tools/podman.py` (exports
  `PodmanSandboxSection`, `PodmanShellParams`, `PodmanShellResult`,
  `PodmanWorkspace`).
- Section registration: `weakincentives.tools.__init__` re-exports the public
  dataclasses and section.
- Tests: `tests/tools/test_podman_tools.py` plus
  `integration-tests/test_podman_shell.py`.
- Prompt helpers: `code_reviewer_example.py` demonstrates composing the section
  with other tools.

### Reference Map

- **How it works**: See "Workspace Lifecycle" and "Session Reducers & State".
- **What to call**: Tooling surface summaries live in "Tooling Surface" and
  "evaluate_python".
- **Where to configure**: Environment and resource knobs are captured in
  "Operational Notes"; host mounts follow "Host Mounts".
- **What to watch out for**: Timeouts, truncation, and isolation expectations
  are documented in "Goals", "Workspace Lifecycle", and "Subagent Isolation
  Requirements".

## External Dependencies

- **`podman` Python SDK** – Imported lazily when constructing a client. Users
  must install `weakincentives[podman]`.
- **Podman CLI** – Used for `podman exec` and `podman cp`. The section resolves
  the CLI connection via `PODMAN_BASE_URL`, `PODMAN_IDENTITY`,
  `PODMAN_CONNECTION`, or by shelling out to
  `podman system connection list --format json`.

## Workspace Lifecycle

1. **Overlay Root** – Each session is assigned an overlay inside
   `${WEAKINCENTIVES_CACHE:-$HOME/.cache/weakincentives/podman}/<session_id>`.
   Host mounts (if any) are hydrated by copying files into this overlay before
   the container starts; hydration is skipped when the overlay already contains
   files so agent edits persist across restarts.
1. **Container Creation** – The section lazily creates a container the first
   time any tool runs. It uses the default image `python:3.12-bookworm`, sets
   user `65534:65534`, disables swap, limits memory to 1 GiB, and bind-mounts
   the overlay into `/workspace`. A tmpfs is mounted at the host temp directory
   to prevent accidental writes outside `/workspace`.
1. **Startup & Health** – Containers start with `sleep infinity`; the section
   immediately runs `test -d /workspace` via `exec_run` to confirm mounts are
   available. Failure raises `ToolValidationError`.
1. **Reuse & Touch** – Subsequence calls reuse the container and overlay. Every
   tool invocation updates `PodmanWorkspace.last_used_at` so idle cleanup
   heuristics can stop/remove the container when needed.
1. **Teardown** – When the section is garbage collected or explicitly closed,
   the container is stopped and removed via the Podman REST API.

## Session Reducers & State

`PodmanSandboxSection` integrates with `Session` reducers so orchestrators can
inspect workspace state without shelling into the container:

- `PodmanWorkspace` (dataclass) captures container id, name, image, workdir,
  overlay path, environment, and timestamps. The section seeds `Session` with
  this slice via `replace_latest` whenever the workspace handle changes.
- `VirtualFileSystem` – The section seeds the VFS slice with either an empty
  snapshot or the hydrated host mounts. Write/delete reducers (`make_write_reducer`
  and `make_delete_reducer`) keep the snapshot in sync with container writes so
  prompt sections can display up-to-date file metadata.
- `WriteFile`, `DeleteEntry`, `EvalResult` – Reducers (shared with the VFS and
  ASTEval specs) are registered to drive the Virtual File System slice and track
  evaluate_python output.

When a subagent runs with `SubagentIsolationLevel.NO_ISOLATION`, it inherits the
parent `Session` (per `specs/SUBAGENTS.md`). The child therefore shares the same
`PodmanWorkspace` slice and executes inside the same container, guaranteeing
that file and shell state are visible across agent boundaries. With
`SubagentIsolationLevel.FULL_ISOLATION`, the orchestrator must clone the parent
session; the clone receives a new session id, which in turn yields a fresh
overlay path and container when the sandbox section is accessed.

### Cloning

`PodmanSandboxSection.clone(session=..., bus=...)` returns a fresh section wired
to the provided runtime. The clone MUST re-register reducers on the supplied
session, publish tool telemetry to the new event bus, and resolve a distinct
overlay and container so no filesystem or process state leaks from the source
section. Host mounts hydrate against the new session id, mirroring constructor
behavior.

## Tooling Surface

### `shell_execute`

```python
Tool[PodmanShellParams, PodmanShellResult](
    name="shell_execute",
    description="Run a short command inside the Podman workspace.",
    handler=PodmanSandboxSection._shell_suite.run_shell,
)
```

- Commands must be ASCII, non-empty, and ≤4,096 characters combined.
- `cwd` is normalized relative to `/workspace`, forbidding `.`/`..`.
- Environment overrides are validated (ASCII, ≤64 entries, ≤80-char keys).
- `stdin` payloads are limited to the VFS write limit (48,000 characters).
- Execution uses `podman exec --workdir <cwd>` with interactive mode when
  stdin is provided. `capture_output=False` returns the sentinel "capture
  disabled" for both stdio streams.
- Timeouts clamp to [1, 120] seconds; a timeout surfaces exit code 124.
- Results are truncated to ≤32 KiB to keep tool responses focused.

### VFS-Compatible Tools

`PodmanSandboxSection` wires the same handlers defined in
`specs/VFS_TOOLS.md` onto the Podman workspace:

- `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `rm`.
- Paths are normalized via `vfs.normalize_string_path` and mapped into the
  overlay using `_host_path_for`.
- File writes happen by buffering the payload locally and pushing it into the
  container via `podman cp`. Append mode pre-reads the existing file to keep
  behavior identical to the in-memory VFS.
- Removals run `python3 -c` scripts inside the container to safely delete files
  or directories and avoid escaping the sandbox.
- All handlers call `touch_workspace()` so `last_used_at` remains accurate.

### `evaluate_python`

- Mirrors the ASTEval contract (`EvalParams`/`EvalResult`), but execution takes
  place via `python3 -c` inside the container with a five-second timeout.
- `reads`/`writes` payloads are not supported because the container already has
  direct access to `/workspace`. The handler enforces that reads, writes, and
  globals dictionaries are empty, ensuring that agents interact with files via
  the dedicated VFS tools instead.
- Unlike the in-memory ASTEval tool, the Podman variant does **not** enforce a
  2,000-character cap on `code`. Scripts can be arbitrarily large (bounded only
  by Podman execution timing and stdout/stderr truncation), making it suitable
  for maintenance macros or generated patches.

## Host Mounts

- `HostMount` definitions reuse the `specs/VFS_TOOLS.md` schema.
- Allowed roots are normalized via `vfs.normalize_host_root`. Each mount
  resolves to a concrete host path and renders a preview block appended to the
  section template.
- Hydration copies files verbatim (binary-safe) into the overlay prior to
  container creation. Include/exclude globs and `max_bytes` budgets are enforced
  during the copy. Failures raise `ToolValidationError` so callers can adjust
  mount settings.

## Prompt Template

The section renders the following template (with an optional host-mount block at
the end):

```
Podman Workspace
----------------
You have access to an isolated Linux container powered by Podman. The `ls`,
`read_file`, `write_file`, `glob`, `grep`, and `rm` tools mirror the virtual
filesystem interface but operate on `/workspace` inside the container. The
`evaluate_python` tool is a thin wrapper around `python3 -c` (≤5 seconds); read
and edit files directly from the workspace. `shell_execute` runs short commands
(≤120 seconds) in the shared environment. No network access or privileged
operations are available. Do not assume files outside `/workspace` exist.
```

`PodmanSandboxSection` shares the `accepts_overrides` flag with other sections so
prompt overrides can swap handlers without editing the root template.

## Telemetry & Error Handling

- All writes/removals capture stderr/stdout from the CLI and fold failures into
  `ToolValidationError` with descriptive messages.
- Command invocations log through `StructuredLogger` with the context
  `component=tools.podman`.
- Connection failures (missing CLI, invalid identity, REST API issues) return
  failing `ToolResult`s so agents can retry or ask for human assistance.

## Subagent Isolation Requirements

- In **No Isolation**, child subagents *must* reuse the existing section object.
  Since they share the same `Session`, they see the same `PodmanWorkspace` and
  `VirtualFileSystem` slices, guaranteeing that file edits made by children are
  visible to the parent after delegation completes.
- In **Full Isolation**, cloning the session produces a new `session_id`. When
  the cloned prompt renders, `PodmanSandboxSection` resolves to a new overlay
  and container. No data leaks between parent and child; host mounts are
  rehydrated independently for each isolated session.

Refer to `specs/SUBAGENTS.md` for the orchestration mechanics around session
cloning and event bus provisioning.

## Testing Expectations

- Unit tests (`tests/tools/test_podman_tools.py`) cover:
  - Command validation, timeout handling, stdout truncation.
  - VFS integration (write/edit/append/remove/glob/grep) and host mount
    hydration, including binary files and byte budgets.
  - Evaluate Python guardrails (rejecting unsupported payloads, timeout paths).
  - Host mount resolver behavior (globs, allowed roots).
- Integration tests (`integration-tests/test_podman_shell.py`) run against a
  real Podman daemon when the `podman` marker is enabled, verifying that shell
  commands, file writes, and evaluate_python work end-to-end.
- Every change touching the sandbox must run `make check` so coverage stays at
  100% across `src/weakincentives`.

## Operational Notes

- Environment variables:
  - `PODMAN_BASE_URL`, `PODMAN_IDENTITY`, `PODMAN_CONNECTION` override CLI
    detection.
  - `WEAKINCENTIVES_CACHE` redirects the overlay root.
- Resource limits are hard-coded; adjust in `podman.py` under `_CPU_PERIOD`,
  `_CPU_QUOTA`, `_MEMORY_LIMIT`, `_TMPFS_SIZE`.
- Containers are named `wink-<session_id>` to simplify debugging with `podman ps`.
