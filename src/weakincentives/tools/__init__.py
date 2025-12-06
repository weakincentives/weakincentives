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

"""Public surface for built-in tool suites."""

# pyright: reportImportCycles=false

from __future__ import annotations

from importlib import import_module
from typing import Any

from .asteval import (
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
from .errors import DeadlineExceededError, ToolValidationError
from .planning import (
    AddStep,
    Plan,
    PlanningStrategy,
    PlanningToolsSection,
    PlanStatus,
    PlanStep,
    ReadPlan,
    SetupPlan,
    StepStatus,
    UpdateStep,
)
from .subagents import (
    DispatchSubagentsParams,
    SubagentIsolationLevel,
    SubagentResult,
    SubagentsSection,
    SubagentTask,
    build_dispatch_subagents_tool,
    dispatch_subagents,
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
    VfsFile,
    VfsPath,
    VfsToolsSection,
    VirtualFileSystem,
    WriteFile,
    WriteFileParams,
)

PodmanSandboxConfig: Any
PodmanSandboxSection: Any
PodmanShellParams: Any
PodmanShellResult: Any
PodmanWorkspace: Any

__all__ = [
    "AddStep",
    "AstevalSection",
    "DeadlineExceededError",
    "DeleteEntry",
    "DispatchSubagentsParams",
    "EditFileParams",
    "EvalFileRead",
    "EvalFileWrite",
    "EvalParams",
    "EvalResult",
    "FileInfo",
    "GlobMatch",
    "GlobParams",
    "GrepMatch",
    "GrepParams",
    "HostMount",
    "ListDirectory",
    "ListDirectoryParams",
    "ListDirectoryResult",
    "Plan",
    "PlanStatus",
    "PlanStep",
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
    "RemoveParams",
    "SetupPlan",
    "StepStatus",
    "SubagentIsolationLevel",
    "SubagentResult",
    "SubagentTask",
    "SubagentsSection",
    "ToolValidationError",
    "UpdateStep",
    "VfsFile",
    "VfsPath",
    "VfsToolsSection",
    "VirtualFileSystem",
    "WorkspaceDigest",
    "WorkspaceDigestSection",
    "WriteFile",
    "WriteFileParams",
    "build_dispatch_subagents_tool",
    "clear_workspace_digest",
    "dispatch_subagents",
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
