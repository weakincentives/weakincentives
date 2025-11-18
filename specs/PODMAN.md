# Podman Container Tooling Specification

## Snapshot

- Ships a Podman-native `PodmanSandboxSection` under `weakincentives.tools.podman`.
- Default container image: `python:3.12-bookworm`, started via the Podman REST API
  using `podman.PodmanClient` (extra dependency `weakincentives[podman]`).
- Every session owns an isolated container + overlay at `${cache}/podman/<session>`;
  containers run as `65534:65534`, with network disabled, 1 GiB memory, 1 CPU, and
  `/workspace` as the only writable mount.
- Host mounts reuse the `HostMount` contract from `specs/VFS_TOOLS.md`; when
  configured they hydrate `/workspace` before the container starts and seed the
  `VirtualFileSystem` snapshot so shell and filesystem tooling see the same data.
- `PodmanSandboxSection` now exposes `evaluate_python` alongside the VFS tools and
  `shell_execute`, mirroring the ASTEval contract but executing scripts inside
  the container with the real filesystem.
- Work delivers in two phases: Phase 1 exposes only `shell_execute`; Phase 2 ports
  the Virtual File System (VFS) surface so filesystem and shell tooling share
  the same Podman workspace.

## Delivery Phases

| Phase | Scope | Notes |
| --- | --- | --- |
| 1 – `shell_execute` | Container lifecycle + shell tool. | `/workspace` starts empty; agents create files solely via shell commands. Prompt copy, telemetry, and cleanup must all work without VFS. |
| 2 – VFS parity | Reuse VFS contracts (`ls`, `read_file`, `write_file`, etc.) on the Podman workspace. | Mirrors `specs/VFS_TOOLS.md`; every handler operates inside the existing container and keeps reducers in sync. |

## Container Lifecycle

1. **Creation** – On the first Podman tool call, resolve the session id, pull the
   configured image if missing, and create the container with the mounts and
   resource caps above. Containers start with `sleep infinity` so exec sessions
   can be attached for shell or filesystem ops.
1. **Startup & Health** – Immediately `start()` the container and verify
   `/workspace` exists via `exec_run(["test", "-d", "/workspace"])`. Record
   `PodmanWorkspace.started_at` and update `last_used_at` after every call.
1. **Runtime** – Every tool call reuses the same workspace tuple
   `{container_id, image, workdir="/workspace", env}`. The shell and (later)
   VFS tools therefore see consistent state.
1. **Cleanup** – Session finalizers call `container.remove(force=True)` and delete
   the overlay directory. Opportunistic idle cleanup removes workspaces unused
   for >15 minutes.

Failures from `podman.errors.APIError`, transport issues, or timeout kills map to
`ToolValidationError` (for validation problems) or failed `ToolResult`s (for runtime
issues). Unexpected container exits trigger recreation on the next tool call.

## Tooling Surface

### Phase 1 – `shell_execute`

```python
@dataclass(slots=True, frozen=True)
class PodmanShellParams:
    command: tuple[str, ...]  # required, ≤4,096 bytes total
    cwd: str | None = None    # relative to /workspace
    env: Mapping[str, str] = field(default_factory=dict)  # ASCII keys/values
    stdin: str | None = None  # ≤48,000 chars
    timeout_seconds: float = 30.0  # clamped to [1,120]
    capture_output: bool = True

@dataclass(slots=True, frozen=True)
class PodmanShellResult:
    command: tuple[str, ...]
    cwd: str
    exit_code: int
    stdout: str  # ≤32 KiB (truncated w/ "[truncated]")
    stderr: str  # ≤32 KiB (truncated)
    duration_ms: int
    timed_out: bool  # exit_code forced to 124 when True
```

```
run_shell_tool = Tool(
    name="shell_execute",
    description="Run a short command inside the Podman workspace.",
    handler=PodmanSandboxSection.run_shell,
)
```

Key semantics:

- Reject empty commands, non-ASCII env, invalid relative paths, or cwd pointing
  outside `/workspace`.
- Commands run via `container.exec_run`, inherit the base environment plus the
  validated overrides, and respect the timeout watchdog (SIGTERM → SIGKILL).
- When `capture_output` is `False`, stdout/stderr return the fixed string
  "capture disabled".
- After each shell invocation, parse optional `FILE_CHANGE:` footers to restat
  touched files so the `VirtualFileSystem` reducer stays consistent once Phase 2
  arrives.

### Phase 2 – VFS Tools

- Reuse the data classes, validation helpers, and reducer contracts defined in
  `specs/VFS_TOOLS.md` (`ListDirectoryParams`, `ReadFileParams`, etc.).
- Path normalization, ASCII limits, 2,000 result truncation, and 48,000-character
  write budgets all carry over unchanged.
- Host mounts share the `HostMount` schema; they require explicit
  `allowed_host_roots`, honor include/exclude globs, enforce byte budgets, seed
  the `VirtualFileSystem` slice, and hydrate `/workspace` before the first
  container starts.
- Handler differences vs. the in-memory VFS:
  - Paths resolve to `/workspace/<segments>` inside the container—no host disk
    access or parent traversal is possible.
  - Mutations (`write_file`, `edit_file`, `rm`) execute via Podman exec sessions
    (e.g., piping into `tee`) so the container remains authoritative.
  - After a mutation, sync the reducer snapshot to whatever Podman reports to
    keep prompts deterministic.

### Evaluate Python

- Reuses the `EvalParams`/`EvalResult` dataclasses and reducer contract from
  `specs/ASTEVAL.md`, so prompts and adapters interact with a single evaluation
  interface regardless of backend.
- Execution happens inside the Podman workspace via `python3 -c <script>`, with
  a fixed five-second timeout, captured stdout/stderr (truncated to 4,096
  characters), and access to the same `read_text`, `write_text`, and
  `vfs_reads` helpers described in the ASTEval spec.
- `reads` entries are materialised from the overlay and injected into the
  script payload, while staged writes (`writes` parameters or `write_text`
  calls) are validated, templated, and then applied via the VFS reducers so the
  `VirtualFileSystem` snapshot and container stay in sync.
- Validation mirrors ASTEval: ASCII-only writes, 2,000-character code limit,
  no overlapping read/write paths, duplicate detection, and descriptive
  `ToolValidationError`s when template variables are missing or paths are
  invalid. Pending writes are discarded automatically when execution fails.

## Workspace & Prompt Copy

- `/workspace` starts empty unless host mounts are configured. When mounts are
  present, Podman copies the hydrated files (subject to glob filters and byte
  ceilings) into the session's overlay before the first container starts.
  Subsequent recreations reuse the existing overlay and skip hydration when
  files already exist so agent edits persist.
- Prompt section text:
  ```
  Podman Workspace
  ----------------
  You have access to an isolated Linux container powered by Podman. The `ls`,
  `read_file`, `write_file`, `glob`, `grep`, and `rm` tools mirror the virtual
  filesystem interface but operate on `/workspace` inside the container. The
  `shell_execute` tool runs short commands (≤120 seconds) in the same environment.
  No network access or privileged operations are available. Do not assume files
  outside `/workspace` exist.
  ```
- When host mounts are configured, append the rendered mount table just like
  `VfsToolsSection` so agents know what data is available.

## Eventing, Testing, and Reliability

- Emit `ToolEvent` records for every Podman REST call, including exec exit codes
  and durations. Reuse the existing ToolEvent schema—no new event types are
  required. Workspace lifecycle events use the `runtime.podman.*` namespace
  (`created`, `started`, `idle_cleanup`, `recreated_after_failure`).
- Tests:
  - Unit tests stub `PodmanClient` to validate parameter guards without needing
    a real daemon.
  - Integration tests (guarded by the `podman` marker and `PODMAN_AVAILABLE=1`)
    exercise both shell and VFS flows, ensuring the `VirtualFileSystem` snapshot
    mirrors container state.
  - Threadstress tests run concurrent `shell_execute` (and later VFS) calls to
    confirm locking. Smoke tests ensure the default image can run
    `uv run pytest` inside the container via `shell_execute`.
- Failures (image pull issues, socket errors, resource ceiling hits) surface as
  failed `ToolResult`s that instruct agents to retry or contact an operator; the
  section also schedules exponential backoff before attempting new pulls.
