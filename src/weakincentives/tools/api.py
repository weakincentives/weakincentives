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

"""Public surface for built-in tool suites.

Podman support is optional; install ``weakincentives[podman]`` to enable the
sandbox types exported below. The symbols are always present for import and type
checking, but resolve to ``None`` at runtime when the optional dependency is
missing.
"""

# pyright: reportImportCycles=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

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
    ClearPlan,
    MarkStep,
    NewPlanStep,
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
    build_dispatch_subagents_tool,
    dispatch_subagents,
)

if TYPE_CHECKING:
    from .podman import (
        PodmanSandboxConfig,
        PodmanSandboxSection,
        PodmanShellParams,
        PodmanShellResult,
        PodmanWorkspace,
    )
else:
    try:
        from .podman import (
            PodmanSandboxConfig,
            PodmanSandboxSection,
            PodmanShellParams,
            PodmanShellResult,
            PodmanWorkspace,
        )
    except ImportError:
        PodmanSandboxConfig = cast(type[Any], None)
        PodmanSandboxSection = cast(type[Any], None)
        PodmanShellParams = cast(type[Any], None)
        PodmanShellResult = cast(type[Any], None)
        PodmanWorkspace = cast(type[Any], None)

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

__all__ = [
    "AddStep",
    "AstevalSection",
    "ClearPlan",
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
    "MarkStep",
    "NewPlanStep",
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
