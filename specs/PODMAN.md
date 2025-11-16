# Podman Container Tooling Specification

## Overview

Weak Incentives now ships a Podman-native tool section so agents can inspect and
mutate a reproducible Linux environment without touching the host. The section
keeps the Virtual File System (VFS) surface documented in `specs/VFS_TOOLS.md`,
but every operation targets a running Podman container instead of an in-memory
snapshot. The section also introduces a dedicated shell execution tool so agents
can run short commands inside the same container that backs the filesystem.

## Goals

- **Container isolation** – Run all filesystem mutations and shell commands in a
  short-lived Podman container scoped to the current session.
- **VFS parity** – Reuse the existing VFS tool contracts (names, dataclasses,
  validation rules) so flows and prompts do not fork between virtual and
  Podman-backed workspaces.
- **Single source of truth** – Keep the container filesystem as the canonical
  state while mirroring a trimmed snapshot back into session reducers for
  determinism and auditing.
- **Shell affordance** – Provide a strongly typed shell tool bound to the same
  container so file edits and commands see a consistent workspace.
- **Security & quotas** – Enforce strict mounts, resource ceilings, and
  timeouts so Podman tasks remain predictable on shared runners.

## Non-Goals

- Supporting Docker or other container runtimes (the work focuses solely on the
  Podman REST service v5+).
- Allowing privileged containers, host networking, or arbitrary device mounts.
- Emulating long-running daemons; every command remains short lived and tied to
  one session.

## Module Layout

- Implementation lives in `weakincentives.tools.podman`.
- `PodmanToolsSection` mirrors `VfsToolsSection` and is the only public entry
  point.
- Helpers that orchestrate Podman invocations (workspace mounts, exec wrapper,
  copy helpers) live in `weakincentives.tools._podman`.
- Tests belong under `tests/tools/test_podman_section.py` with fixtures in
  `tests/helpers/podman.py` for container lifecycle orchestration.
- The module exports `PodmanWorkspace` and `PodmanShellResult` dataclasses so
  reducers/tests can introspect state without depending on concrete handler
  classes.
- The section depends on the
  [`podman`](https://github.com/containers/podman-py) Python library to talk to
  the REST API. Declare it behind a `[project.optional-dependencies].podman`
  extra, import it lazily, and raise a `ToolValidationError` that instructs
  operators to install `weakincentives[podman]` when the extra is missing.

## Terminology

- **Workspace root** – The POSIX path (default `/workspace`) that agents see as
  the root of the filesystem tools. All VFS paths resolve relative to this root.
- **Workspace mount** – A host directory (usually the repo under test) mounted
  into the container at `/workspace`. The mount is read/write but scoped to the
  session-specific overlay to avoid changing the host filesystem.
- **Podman workspace** – Tuple of `{container_id, image, workdir, env}` stored in
  session state to identify an active container.

## Container Lifecycle

### Creation

1. Containers are created lazily. The first Podman filesystem or shell tool call
   resolves the session id and ensures a corresponding `PodmanWorkspace` exists.

1. `PodmanWorkspace` is provisioned through the Podman REST API using the
   `podman.PodmanClient`. The section pulls the image when missing and then
   calls `client.containers.create(...)` with the equivalent options that the
   CLI flags would have provided:

   ```python
   from podman import PodmanClient

   client = PodmanClient(base_url=section.socket, identity=section.identity)
   client.images.pull(image)
   container = client.containers.create(
       image=image,
       name=f"wink-{session_id}",
       command=["sleep", "infinity"],
       workdir="/workspace",
       user="65534:65534",
       network_disabled=True,
       mem_limit="1g",
       memswap_limit="1g",
       cpu_period=100_000,
       cpu_quota=100_000,
       env=resolved_env,
       mounts=[
           {"Target": "/tmp", "Type": "tmpfs", "TmpfsOptions": {"SizeBytes": 268_435_456}},
           {
               "Target": "/workspace",
               "Source": session_overlay,
               "Type": "bind",
               "Options": ["rbind", "rw"],
           },
       ],
       remove=True,
   )
   ```

   The `command` keeps the container idle and ready to receive exec sessions
   from filesystem and shell tools.

1. `image` defaults to `ghcr.io/weakincentives/agent-workspace:rolling` and is
   configurable via `PodmanToolsSection(image=...)`.

1. The section keeps a per-session overlay directory under
   `${cache}/podman/${session_id}` so writes never land on the host checkout.

### Startup & Health

- Once created, the container is started via `container.start()` on the REST
  client and left running until the session ends or the idle timeout (15 minutes)
  elapses.
- A readiness probe (`container.exec_run(["test", "-d", "/workspace"])`) runs
  before serving the first tool call. Failures bubble up as
  `ToolValidationError`s.
- The section records `PodmanWorkspace.started_at` and `last_used_at` timestamps
  for cleanup heuristics.

### Cleanup

- Session finalizers stop and remove the container via
  `container.remove(force=True)` and delete
  the overlay directory.
- Idle cleanup runs opportunistically when a new container is created; any
  workspace whose `last_used_at` is older than 15 minutes is force removed.
- Failures during cleanup propagate to the event log but do not crash the user
  session.

## Filesystem Tooling

### Surface

The Podman section registers the same tool list and dataclasses as
`VfsToolsSection`:

| Tool name | Dataclasses | Podman behavior |
| --- | --- | --- |
| `ls` | `ListDirectoryParams` → `FileInfo[]` | Lists container entries under `/workspace/<path>`.
| `read_file` | `ReadFileParams` → `ReadFileResult` | Reads UTF-8 content from the container.
| `write_file` | `WriteFileParams` → `WriteFile` | Writes UTF-8 text directly into the container filesystem.
| `edit_file` | `EditFileParams` → `WriteFile` | Applies string replacements inside the container file.
| `glob` | `GlobParams` → `GlobMatch[]` | Expands patterns relative to `/workspace` using Podman's shell.
| `grep` | `GrepParams` → `GrepMatch[]` | Runs regex searches via `rg` inside the container.
| `rm` | `RemoveParams` → `DeleteEntry` | Deletes files or directories recursively in the container.

### Semantics

- Paths remain relative, ASCII, ≤16 levels deep, and are normalized by the VFS
  helpers before invoking Podman.
- All file operations convert the normalized path to
  `/workspace/<segments joined>`. No parent directory traversal is possible.
- Read/write size limits, encoding guards, and validation errors stay identical
  to the VFS spec. Podman handlers reuse `_normalize_string_path`, `_clamp_read`
  and related helpers directly.
- `write_file` and `edit_file` open exec sessions through
  `container.exec_run(["/bin/sh", "-c", ...])` (e.g. piping into `tee`) for
  atomic writes and record the resulting metadata (size, timestamps, version) by
  stat-ing the file inside the container.
- `ls`, `glob`, and `grep` never read more than 2_000 matches; extra entries are
  truncated with an ellipsis message mirroring VFS behavior.
- Each handler updates the session-scoped `VirtualFileSystem` snapshot by taking
  the authoritative view from the container after the mutation. The snapshot
  only tracks files touched via the tools (to avoid walking the entire root) and
  leverages the same reducer functions for determinism.

### Error Handling

- Podman REST failures (`APIError`, non-zero exec exit codes, missing files)
  map to `ToolValidationError` with stderr attached to the message.
- Transport errors (Podman daemon unavailable, container missing) return failed
  `ToolResult`s with `success=False` and instruct the agent to contact an
  operator.
- If the container dies unexpectedly, the section tears it down and recreates it
  on the next call while emitting a warning event.

## Shell Execution Tool

### Contract

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from weakincentives.prompt.tool import Tool, ToolContext, ToolResult
from weakincentives.tools.errors import ToolValidationError

@dataclass(slots=True, frozen=True)
class PodmanShellParams:
    command: tuple[str, ...]
    cwd: str | None = field(default=None)
    env: Mapping[str, str] = field(default_factory=dict)
    stdin: str | None = field(default=None)
    timeout_seconds: float = 30.0
    capture_output: bool = True


@dataclass(slots=True, frozen=True)
class PodmanShellResult:
    command: tuple[str, ...]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool
```

```python
run_shell_tool = Tool[PodmanShellParams, PodmanShellResult](
    name="podman_shell",
    description="Run a short command inside the Podman workspace.",
    handler=PodmanToolsSection.run_shell,
)
```

### Parameter Semantics

- `command` – Required tuple where the first entry is the executable. Empty
  tuples raise `ToolValidationError`. Commands longer than 4_096 bytes combined
  are rejected.
- `cwd` – Optional relative path resolved under `/workspace`. Defaults to the
  workspace root. Validation rejects paths pointing outside the workspace or to
  files.
- `env` – Additional environment variables merged with the base workspace env.
  Values must be ASCII and ≤512 characters. Keys are uppercased automatically to
  match container conventions.
- `stdin` – Optional UTF-8 payload piped into the command. Bounded to 48_000
  characters.
- `timeout_seconds` – Wall-clock timeout capped between 1 and 120 seconds. The
  handler enforces it by wrapping the REST exec session in a watchdog that sends
  `container.kill(signal="SIGTERM")` followed by `SIGKILL` if the command runs
  past the deadline.
- `capture_output` – When `False`, stdout/stderr are truncated to a fixed string
  noting that capture was disabled.

### Result Semantics

- `stdout` and `stderr` capture up to 32 KiB each. Longer streams are truncated
  with a `"[truncated]"` sentinel at the end.
- `duration_ms` measures time between starting the REST exec session and
  receiving the exit code.
- `timed_out` indicates whether the handler force-killed the process because of
  the timeout guard. When `True`, `exit_code` is set to 124.
- The result payload always echoes the resolved `cwd` so agents can reason about
  relative paths.

### Shared Workspace

- The shell tool and filesystem tools share the same `PodmanWorkspace`. Files
  created via shell commands are immediately visible to `ls`/`read_file` as soon
  as the command completes. Conversely, files written via the VFS tools show up
  in subsequent shell commands because they operate on the same overlay mount.
- After each shell execution, the handler re-stat's any files mentioned in the
  stdout footer `FILE_CHANGE:` markers to keep the `VirtualFileSystem` snapshot
  in sync. (Future work may infer changes via `podman diff`.)

## Prompt & Documentation Copy

`PodmanToolsSection` renders a markdown block directly under "Built-in Tools"
with the following template:

```
Podman Workspace
----------------
You have access to an isolated Linux container powered by Podman. The `ls`,
`read_file`, `write_file`, `glob`, `grep`, and `rm` tools mirror the virtual
filesystem interface but operate on `/workspace` inside the container. The
`podman_shell` tool runs short commands (≤120 seconds) in the same environment.
No network access or privileged operations are available. Do not assume files
outside `/workspace` exist.
```

The section automatically appends a table describing the mounted host paths (if
any) and the default image name so agents know what environment they are using.

## Eventing & Logging

- Every Podman REST invocation publishes `ToolEvent` records with the method,
  path, duration, and exit status (exec calls include the returned exit code).
- Workspace lifecycle events (`created`, `started`, `idle_cleanup`,
  `recreated_after_failure`) emit structured payloads under the namespace
  `runtime.podman`.
- Shell executions log stdout/stderr previews (first 200 characters) for audit
  trails but redact stdin.

## Testing

- Unit tests stub the REST client (`PodmanClient`) with a fake transport that
  records exec calls and simulates exit codes. Fixtures verify validation logic
  without requiring Podman on CI.
- Integration tests run behind the `podman` pytest marker and require
  `PODMAN_AVAILABLE=1`. They bootstrap a container, run a matrix of filesystem
  and shell commands, and assert the `VirtualFileSystem` snapshot stays in sync.
- Threadstress tests reuse the existing plugin to run concurrent `ls` and
  `podman_shell` calls, ensuring lock contention is handled correctly.
- A smoke test under `integration-tests/` ensures the default image can install
  Python dependencies and run `uv run pytest` when invoked through
  `podman_shell`.

## Failure Modes & Recovery

- Missing Podman REST socket or Python client: the section raises
  `ToolValidationError` during initialization and disables itself at prompt
  render time.
- Image pull failure: handlers return a failed `ToolResult` instructing the user
  to retry; the section schedules a backoff before attempting another pull.
- Container crash: the section tears down the workspace, logs an event, and
  recreates the container transparently on the next call.
- Resource exhaustion: exceeding the memory or timeout limits raises a failed
  `ToolResult` with stderr describing the limit that was hit.
