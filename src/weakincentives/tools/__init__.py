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

"""Tools namespace."""

# pyright: reportImportCycles=false

from __future__ import annotations

from typing import NoReturn

from . import asteval, digests, errors, planning, podman, subagents, vfs
from .asteval import AstevalSection, EvalFileRead, EvalFileWrite, EvalParams, EvalResult
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
from .podman import (
    PodmanSandboxConfig,
    PodmanSandboxSection,
    PodmanShellParams,
    PodmanShellResult,
    PodmanWorkspace,
)
from .subagents import (
    DispatchSubagentsParams,
    SubagentIsolationLevel,
    SubagentResult,
    SubagentsSection,
    build_dispatch_subagents_tool,
    dispatch_subagents,
)
from .vfs import (
    MAX_WRITE_LENGTH,
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
    ensure_ascii,
    find_file,
    format_directory_message,
    format_edit_message,
    format_glob_message,
    format_grep_message,
)

__all__ = [  # noqa: RUF022
    "AddStep",
    "asteval",
    "AstevalSection",
    "build_dispatch_subagents_tool",
    "ClearPlan",
    "DeadlineExceededError",
    "DeleteEntry",
    "digests",
    "dispatch_subagents",
    "DispatchSubagentsParams",
    "EditFileParams",
    "ensure_ascii",
    "errors",
    "EvalFileRead",
    "EvalFileWrite",
    "EvalParams",
    "EvalResult",
    "FileInfo",
    "find_file",
    "format_directory_message",
    "format_edit_message",
    "format_glob_message",
    "format_grep_message",
    "GlobMatch",
    "GlobParams",
    "GrepMatch",
    "GrepParams",
    "HostMount",
    "ListDirectory",
    "ListDirectoryParams",
    "ListDirectoryResult",
    "MarkStep",
    "MAX_WRITE_LENGTH",
    "NewPlanStep",
    "Plan",
    "PlanningStrategy",
    "planning",
    "PlanningToolsSection",
    "PlanStatus",
    "PlanStep",
    "podman",
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
    "subagents",
    "SubagentsSection",
    "ToolValidationError",
    "UpdateStep",
    "vfs",
    "VfsFile",
    "VfsPath",
    "VfsToolsSection",
    "VirtualFileSystem",
    "WriteFile",
    "WriteFileParams",
]


def __getattr__(name: str) -> NoReturn:  # pragma: no cover - dynamic helper
    raise AttributeError(name)


def __dir__() -> list[str]:  # pragma: no cover - convenience shim
    return sorted({*globals().keys(), *__all__})
