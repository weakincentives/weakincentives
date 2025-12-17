# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contributed tool suites for specific agent styles.

This package provides domain-specific tools that extend the core primitives:

- **Filesystem**: Protocol and backends for workspace file operations
- **Planning tools**: Session-scoped todo list for background agents
- **VFS tools**: Virtual filesystem with glob, grep, and file operations
- **Asteval tools**: Sandboxed Python expression evaluation
- **Podman tools**: Containerized shell execution
- **Workspace digest**: Caching and optimization for workspace sections

Example usage::

    from weakincentives.contrib.tools import (
        Filesystem,
        InMemoryFilesystem,
        PlanningToolsSection,
        VfsToolsSection,
        AstevalSection,
        PodmanSandboxSection,
    )
"""

# pyright: reportImportCycles=false

from __future__ import annotations

from importlib import import_module
from typing import Any

from .asteval import (
    AstevalConfig,
    AstevalSection,
    EvalFileRead,
    EvalFileWrite,
    EvalParams,
    EvalResult,
)
from .digests import (
    WorkspaceDigest,
    WorkspaceDigestSection,
    clear_workspace_digest,
    latest_workspace_digest,
    set_workspace_digest,
)
from .filesystem import (
    READ_ENTIRE_FILE,
    FileEncoding,
    FileEntry,
    FileStat,
    Filesystem,
    GlobMatch as FilesystemGlobMatch,
    GrepMatch as FilesystemGrepMatch,
    HostFilesystem,
    InMemoryFilesystem,
    ReadResult,
    WriteMode,
    WriteResult,
)
from .planning import (
    AddStep,
    Plan,
    PlanningConfig,
    PlanningStrategy,
    PlanningToolsSection,
    PlanStatus,
    PlanStep,
    ReadPlan,
    SetupPlan,
    StepStatus,
    UpdateStep,
)
from .vfs import (
    DeleteEntry,
    EditFileParams,
    FileInfo,
    GlobMatch,
    GlobParams,
    GrepMatch,
    GrepParams,
    HostMount,
    ListDirectory,
    ListDirectoryParams,
    ListDirectoryResult,
    ReadFile,
    ReadFileParams,
    ReadFileResult,
    RemoveParams,
    VfsConfig,
    VfsFile,
    VfsPath,
    VfsToolsSection,
    WriteFile,
    WriteFileParams,
)
from .workspace import ToolSuiteSection, WorkspaceSection

PodmanSandboxConfig: Any
PodmanSandboxSection: Any
PodmanShellParams: Any
PodmanShellResult: Any
PodmanWorkspace: Any

__all__ = [
    "READ_ENTIRE_FILE",
    "AddStep",
    "AstevalConfig",
    "AstevalSection",
    "DeleteEntry",
    "EditFileParams",
    "EvalFileRead",
    "EvalFileWrite",
    "EvalParams",
    "EvalResult",
    "FileEncoding",
    "FileEntry",
    "FileInfo",
    "FileStat",
    "Filesystem",
    "FilesystemGlobMatch",
    "FilesystemGrepMatch",
    "GlobMatch",
    "GlobParams",
    "GrepMatch",
    "GrepParams",
    "HostFilesystem",
    "HostMount",
    "InMemoryFilesystem",
    "ListDirectory",
    "ListDirectoryParams",
    "ListDirectoryResult",
    "Plan",
    "PlanStatus",
    "PlanStep",
    "PlanningConfig",
    "PlanningStrategy",
    "PlanningToolsSection",
    "PodmanSandboxConfig",
    "PodmanSandboxSection",
    "PodmanShellParams",
    "PodmanShellResult",
    "PodmanWorkspace",
    "ReadFile",
    "ReadFileParams",
    "ReadFileResult",
    "ReadPlan",
    "ReadResult",
    "RemoveParams",
    "SetupPlan",
    "StepStatus",
    "ToolSuiteSection",
    "UpdateStep",
    "VfsConfig",
    "VfsFile",
    "VfsPath",
    "VfsToolsSection",
    "WorkspaceDigest",
    "WorkspaceDigestSection",
    "WorkspaceSection",
    "WriteFile",
    "WriteFileParams",
    "WriteMode",
    "WriteResult",
    "clear_workspace_digest",
    "latest_workspace_digest",
    "set_workspace_digest",
]

_PODMAN_EXPORTS = {
    "PodmanSandboxConfig",
    "PodmanShellParams",
    "PodmanShellResult",
    "PodmanSandboxSection",
    "PodmanWorkspace",
}
_TOOLS_PACKAGE = __name__


def __getattr__(name: str) -> object:
    if name in _PODMAN_EXPORTS:
        module = import_module(f"{_TOOLS_PACKAGE}.podman")
        value = getattr(module, name)
        globals()[name] = value
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
