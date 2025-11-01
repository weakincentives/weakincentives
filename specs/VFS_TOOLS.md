# Virtual Filesystem Tool Suite Specification

## Overview

The virtual filesystem tool suite gives agents a deterministic, session-scoped surface for file reads and writes without
touching the host disk. It supports lightweight workspace snapshots: list directories, inspect file metadata, edit file
contents, and preload selected host folders during initialisation while the session stays isolated. Snapshots are
copy-on-write, start empty automatically, and disappear when the session ends; no state persists across conversations.

## Module Surface

- VFS code lives in `weakincentives.tools.vfs`.
- Tool validation errors reuse `ToolValidationError` from `weakincentives.tools.errors`.
- `VfsToolsSection` is the public entry point. It wires the tool definitions, prompt copy, reducer registrations, and
  optional host folder mounts so orchestrators add the section once per session.
- Host folder mounts are declared via the `HostMount` dataclass and supplied through the section's `mounts` argument.

## Session Integration

- `VfsToolsSection` requires a `Session` from `weakincentives.session`.
- During initialisation the section registers `replace_latest` for the `VirtualFileSystem` slice so every tool write
  publishes a fresh snapshot. A missing snapshot implicitly resolves to an empty filesystem.
- Reducers always clone the current `VirtualFileSystem` before applying changes to keep snapshots immutable to callers.
- Orchestrators retrieve the active snapshot with `select_latest(session, VirtualFileSystem)` and should treat `None`
  as an empty filesystem until the first write occurs.
- Host folder mounts configured on the section are materialised exactly once during initialisation and merged into the
  empty snapshot before the first tool call executes.
- Tool handlers run normalization and validation outside the reducer, returning informative status messages alongside
  the params dataclass used for the update.

## Data Model

Schemas are frozen dataclasses and store normalized POSIX-style paths. Paths are relative, ASCII-only, and split into
segments that exclude `.` and `..`. Handlers collapse duplicate slashes and reject empty segments so reducers can trust
the canonical form.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

FileEncoding = Literal["utf-8"]
WriteMode = Literal["create", "overwrite", "append"]


@dataclass(slots=True, frozen=True)
class VfsPath:
    # Relative path segments (e.g. ("src", "main.py"))
    segments: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class VfsFile:
    path: VfsPath
    content: str
    encoding: FileEncoding
    size_bytes: int
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True, frozen=True)
class VirtualFileSystem:
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
    content: str
    mode: WriteMode = "create"
    encoding: FileEncoding = "utf-8"


@dataclass(slots=True, frozen=True)
class DeleteEntry:
    path: VfsPath


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
- Reducers maintain monotonically increasing `version` numbers per file per snapshot. When `WriteFile.mode` is
  `append`, the reducer concatenates new content and increments the version.
- `created_at` and `updated_at` are stored as timezone-aware UTC timestamps (truncated to millisecond precision). The
  reducer sets both timestamps during creation and only `updated_at` on subsequent writes.
- If no snapshot exists when a tool runs, reducers start from an empty `VirtualFileSystem()` instance automatically.
- The suite enforces size and path guards:
  - Maximum content length per write: 48_000 characters.
  - Maximum path depth: 16 segments; maximum segment length: 80 characters.
  - File content must be valid UTF-8 text; path segments remain ASCII-only.
- Each `HostMount` requires an ASCII `host_path`, resolves it against an allow-listed root configured by the
  orchestrator, normalises it to an absolute path, and rejects traversal outside the approved surface. Included files
  are streamed into memory, decoded as UTF-8, filtered through the glob include/exclude lists, and mounted under
  `mount_path` (or the host relative path when `mount_path` is `None`). Existing VFS entries at the target paths are
  overwritten, directory structures merge, and `max_bytes` caps the aggregate content pulled from disk. Mounting occurs
  once during `VfsToolsSection` initialisation.
- `DeleteEntry` removes files under the exact path. Directories are implicit; removing a directory path deletes all
  files with that prefix.
- Directory listings render shallow entries only (files or immediate directories below the target path). The handler
  performs the projection so reducers only deal with structural updates.

## Tool Contracts

Every tool validates its parameters and raises `ToolValidationError` on failure. Successful invocations publish the new
`VirtualFileSystem` snapshot via the reducer pipeline.

| Tool | Summary | Parameters | Result | Behaviour highlights |
| ---- | ------- | ---------- | ------ | -------------------- |
| `vfs.list_directory` | Enumerate directory contents | `ListDirectory` | `dict` | Returns a JSON-serialisable summary with files and directories for the given path; raises if the path resolves to a file. |
| `vfs.read_file` | Retrieve file contents | `ReadFile` | `dict` | Returns content, version, size, and timestamps; raises if path missing. |
| `vfs.write_file` | Create or modify a file | `WriteFile` | `WriteFile` | Validates mode (`create`, `overwrite`, `append`), trims trailing whitespace only when explicitly requested by caller, and updates version/timestamps. |
| `vfs.delete_entry` | Remove file(s) | `DeleteEntry` | `DeleteEntry` | Deletes the matching file or directory subtree; no-op when nothing matches if `allow_missing` flag is passed (future extension). |

## Prompt Template Guidance

`VfsToolsSection` emits markdown instructing agents to:

1. Remember the virtual filesystem starts empty aside from any host mounts configured by the orchestrator; create files
   via `vfs.write_file` when needed.
1. Host mounts are configuration-time only; agents cannot import additional host directories during a session.
1. Use `vfs.list_directory` to explore before reading or writing specific files; keep listings targeted to minimise
   output volume.
1. Fetch file contents with `vfs.read_file` when context is needed, and operate on the returned version to avoid stale
   edits.
1. Apply edits with `vfs.write_file`, keeping updates concise and UTF-8-only; prefer overwriting complete files instead
   of issuing multiple appends unless streaming logs.
1. Clean up obsolete files or directories via `vfs.delete_entry` once work completes to keep the snapshot tight.
1. Avoid mirroring large repositories or binary assets; orchestrated host mounts should stay focused, and the enforced
   size limit still applies.

The section follows the shared prompt conventions (`specs/PROMPTS.md`) and adds all VFS tool definitions to
`RenderedPrompt.tools`.

## Usage Sketch

```python
from dataclasses import dataclass
from weakincentives.events import InProcessEventBus
from weakincentives.prompt import MarkdownSection, Prompt
from weakincentives.session import Session
from weakincentives.session.selectors import select_latest
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
snapshot = select_latest(session, vfs.VirtualFileSystem)
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
