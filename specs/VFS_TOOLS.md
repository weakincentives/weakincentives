# Virtual Filesystem Tool Suite Specification

## Overview

The virtual filesystem tool suite gives agents a deterministic, session-scoped surface for file reads and writes backed by
an orchestrator-managed temporary directory on disk. The directory is created on first use, isolated per session, and
eligible to be mounted into downstream containers when the runtime needs to share the staged workspace. Agents can list
directories, inspect file metadata, edit file contents, and preload selected host folders during initialisation while the
session stays isolated from the primary host filesystem. Snapshots are copy-on-write, start empty automatically, and
disappear when the session ends; no state persists across conversations beyond the orchestrator's snapshot copies.

## Module Surface

- VFS code lives in `weakincentives.tools.vfs`.
- Tool validation errors reuse `ToolValidationError` from `weakincentives.tools.errors`.
- `VfsToolsSection` is the public entry point. It wires the tool definitions, prompt copy, reducer registrations, and
  optional host folder mounts so orchestrators add the section once per session.
- Host folder mounts are declared via the `HostMount` dataclass and supplied through the section's `mounts` argument.

## Session Integration

- `VfsToolsSection` receives the active `Session` during construction.
- The section initialises reducers immediately after construction: it registers `replace_latest` for the
  `VirtualFileSystem` slice, installs the write/delete reducers, and seeds the slice with the mount snapshot. During this
  pass it also provisions a per-session temporary directory on disk (under the operating system's default temporary
  directory) and records the absolute path alongside the snapshot metadata so other components can mount it when needed.
- Tool handlers verify that the `ToolContext.session` supplied at invocation time matches the configured session and raise
  `ToolValidationError` when adapters attempt to reuse the tools with a different session.
- If the session state is cleared (for example via `Session.reset()`), the next tool invocation re-provisions a fresh
  temporary directory and replays the mount initialisation so the VFS remains usable without manual intervention.
- Reducers always clone the current `VirtualFileSystem` before applying changes to keep snapshots immutable to callers.
- Orchestrators retrieve the active snapshot with `select_latest(session, VirtualFileSystem)` and should treat `None`
  as an empty filesystem until the first write occurs. When callers need direct disk access they should consult the
  stored temporary directory path and mount it into subprocess containers explicitly.
- Creating a session snapshot clones the temporary directory into a snapshot artifact before persisting the
  `VirtualFileSystem` payload. Restoring from a snapshot copies that artifact into a fresh temporary directory and
  updates the stored path so follow-up writes act on new files without mutating the saved snapshot.
- Host folder mounts configured on the section are materialised when the section is constructed and merged into the
  initial snapshot for every session during lazy initialisation. Mounted files are streamed into the session's temporary
  directory so the virtual filesystem view matches the on-disk representation.
- Tool handlers run normalization and validation outside the reducer, returning informative status messages alongside
  the params dataclass used for the update.

## Data Model

Schemas are frozen dataclasses and store normalized POSIX-style paths. Paths are relative, ASCII-only, and split into
segments that exclude `.` and `..`. Handlers collapse duplicate slashes and reject empty segments so reducers can trust
the canonical form. File payloads are tracked as raw bytes with an accompanying `encoding` hint so tool handlers can
serialise text directly and fall back to base64 (or other codecs) for binary content.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

FileEncoding = Literal["utf-8", "binary"] | str | None
WriteMode = Literal["create", "overwrite", "append"]


@dataclass(slots=True, frozen=True)
class VfsPath:
    # Relative path segments (e.g. ("src", "main.py"))
    segments: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class VfsFile:
    path: VfsPath
    encoding: FileEncoding
    size_bytes: int
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True, frozen=True)
class VirtualFileSystem:
    root_path: str
    files: tuple[VfsFile, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class ListDirectory:
    path: VfsPath | None = None  # None lists root


@dataclass(slots=True, frozen=True)
class ReadFile:
    path: VfsPath


@dataclass(slots=True, frozen=True)
class WriteFile:
    path: VfsPath
    content: bytes
    mode: WriteMode = "create"
    encoding: FileEncoding | None = "utf-8"


@dataclass(slots=True, frozen=True)
class DeleteFile:
    path: VfsPath


@dataclass(slots=True, frozen=True)
class FileReadResult:
    file: VfsFile
    content: bytes


@dataclass(slots=True, frozen=True)
class HostMount:
    host_path: str
    mount_path: VfsPath | None = None
    include_glob: tuple[str, ...] = field(default_factory=tuple)
    exclude_glob: tuple[str, ...] = field(default_factory=tuple)
    max_bytes: int | None = None
    follow_symlinks: bool = False
```

`HostMount` instances are provided when constructing `VfsToolsSection`; they are never exposed as interactive tools to
agents.

Implementation detail notes:

- `VirtualFileSystem.files` stores files as a tuple sorted lexicographically by path segments for stable diffs.
- `VirtualFileSystem.root_path` records the absolute path to the session-scoped temporary directory on disk.
- Reducers maintain monotonically increasing `version` numbers per file per snapshot. When `WriteFile.mode` is
  `append`, the reducer increments the version and updates metadata based on the new payload length.
- `created_at` and `updated_at` are stored as timezone-aware UTC timestamps (truncated to millisecond precision). The
  reducer sets both timestamps during creation and only `updated_at` on subsequent writes.
- If no snapshot exists when a tool runs, reducers start from an empty `VirtualFileSystem` snapshot automatically.
- The suite enforces size and path guards:
  - Maximum content length per write: 48_000 characters for UTF-8 text; binary writes enforce the same limit on decoded
    bytes.
  - Maximum path depth: 16 segments; maximum segment length: 80 characters.
  - File contents are persisted on disk only. Snapshots retain metadata and the encoding hint describing how to
    interpret bytes when returning a `FileReadResult`. Path segments remain ASCII-only.
- Each `HostMount` requires an ASCII `host_path`, resolves it against an allow-listed root configured by the
  orchestrator, normalises it to an absolute path, and rejects traversal outside the approved surface. Included files
  are streamed into memory as bytes, filtered through the glob include/exclude lists, and mounted under
  `mount_path` (or the host relative path when `mount_path` is `None`). Existing VFS entries at the target paths are
  overwritten, directory structures merge, and `max_bytes` caps the aggregate content pulled from disk. Mounting occurs
  once during `VfsToolsSection` initialisation.
- `DeleteFile` removes files under the exact path. Directories are implicit; removing a directory path deletes all
  files with that prefix.
- Directory listings render shallow entries only (files or immediate directories below the target path). The handler
  performs the projection so reducers only deal with structural updates.

## Tool Contracts

Every tool validates its parameters and raises `ToolValidationError` on failure. Successful invocations publish the new
`VirtualFileSystem` snapshot via the reducer pipeline.

| Tool | Summary | Parameters | Result | Behaviour highlights |
| ---- | ------- | ---------- | ------ | -------------------- |
| `list_directory` | Enumerate directory contents | `ListDirectory` | `dict` | Returns a JSON-serialisable summary with files and directories for the given path; raises if the path resolves to a file. |
| `read_file` | Retrieve file contents | `ReadFile` | `FileReadResult` | Returns file metadata alongside raw bytes. Binary payloads are base64-encoded when serialised. Raises if the path is missing. |
| `write_file` | Create or modify a file | `WriteFile` | `WriteFile` | Accepts text or binary payloads, validates size after decoding, honours the encoding hint, and updates version/timestamps. |
| `delete_file` | Remove file(s) | `DeleteFile` | `DeleteFile` | Deletes the matching file or directory subtree; no-op when nothing matches if `allow_missing` flag is passed (future extension). |

## Prompt Template Guidance

`VfsToolsSection` emits markdown instructing agents to:

1. Remember the virtual filesystem starts empty aside from any host mounts configured by the orchestrator; create files
   via `write_file` when needed.
1. Host mounts are configuration-time only; agents cannot import additional host directories during a session.
1. Use `list_directory` to explore before reading or writing specific files; keep listings targeted to minimise
   output volume.
1. Fetch file contents with `read_file` when context is needed, and operate on the returned version to avoid stale
   edits.
1. Apply edits with `write_file`, keeping updates concise. Binary payloads are supported but should remain small and
   intentional; prefer overwriting complete files instead of issuing multiple appends unless streaming logs.
1. Clean up obsolete files or directories via `delete_file` once work completes to keep the snapshot tight.
1. Avoid mirroring large repositories or binary assets; orchestrated host mounts should stay focused, and the enforced
   size limit still applies.

The section follows the shared prompt conventions (`specs/PROMPTS.md`) and adds all VFS tool definitions to
`RenderedPrompt.tools`.

## Usage Sketch

```python
from dataclasses import dataclass
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.prompt import MarkdownSection, Prompt
from weakincentives.runtime.session import Session
from weakincentives.runtime.session.selectors import select_latest
from weakincentives.tools import vfs

bus = InProcessEventBus()
session = Session(bus=bus)

prompt = Prompt(
    ns="agents/background",
    key="workspace",
    name="workspace-tools",
    sections=[
        MarkdownSection(
            title="Behaviour",
            template="Use the virtual filesystem to stage edits before applying them to the host workspace.",
        ),
        vfs.VfsToolsSection(
            session=session,
            mounts=(
                vfs.HostMount(
                    host_path="docs/",
                    mount_path=vfs.VfsPath(("docs",)),
                    include_glob=("*.md",),
                ),
            ),
        ),
    ],
)

rendered = prompt.render(None)
# Snapshots appear after the first tool invocation; select_latest returns None
# until a handler observes the session.
```

Host mounts are optional; omit the `mounts` argument to start from an entirely empty virtual filesystem.

Adapters emit `ToolInvoked` events through the shared bus; reducers inside `VfsToolsSection` handle persistence
automatically.

## Telemetry

No additional telemetry is required beyond the default event stream. Tool handlers include structured fields in
`ToolInvoked` payloads (path, mode, result size) so downstream analytics can observe usage patterns without custom
instrumentation.

## Testing Checklist

- Unit tests for path normalization and validation, including rejection of absolute paths, `..`, and non-ASCII segments.
- Reducer tests covering create/overwrite/append flows, version increments, and timestamp updates.
- Host mount tests validating allow-list enforcement, glob inclusion/exclusion, `max_bytes` capping, duplicate path
  handling, and deterministic merging when mounts overlap.
- Tests ensuring directory deletion removes all nested files while leaving unrelated paths untouched.
- Prompt snapshot tests verifying `VfsToolsSection` renders instructions and advertises all tools.
- End-to-end scenario: start from the empty VFS (optionally with configured host mounts), list directories, read/write
  files, and confirm session snapshots update with each tool call.

## Documentation Tasks

- Add `vfs_tools_example.py` showing typical usage alongside the planning tools.
- Update `README.md` to mention the virtual filesystem capabilities and link to this specification.
- Generate API reference entries for `VirtualFileSystem`, tool dataclasses, and the handlers in
  `weakincentives.tools.vfs`.
- Document the host mount allow list configuration and security considerations in `AGENTS.md`.
