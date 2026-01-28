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

"""Domain-specific tool suites for LLM agent prompts.

This package provides production-ready tool sections that extend the core
weakincentives primitives with filesystem operations, code execution,
container sandboxing, and planning capabilities.

Tool Categories
---------------

**Planning Tools** (``PlanningToolsSection``)
    Session-scoped task management for multi-step agent workflows. Provides
    tools to create plans, add steps, update progress, and track completion.
    Supports multiple planning strategies (ReAct, Plan-Act-Reflect, GDRS).

**Virtual Filesystem Tools** (``VfsToolsSection``)
    File operations on an in-memory or host-backed filesystem. Provides
    ls, read_file, write_file, edit_file, glob, grep, and rm tools with
    path normalization and safety constraints.

**Python Evaluation** (``AstevalSection``)
    Sandboxed Python expression evaluation via the ``asteval`` library.
    Provides a restricted execution environment with safe builtins, VFS
    file access, and configurable timeouts.

    Requires: ``pip install weakincentives[asteval]``

**Podman Sandbox** (``PodmanSandboxSection``)
    Containerized shell execution via Podman. Provides isolated Linux
    environment with mounted workspaces, shell_execute command, and
    filesystem tools operating on the container overlay.

    Requires: ``pip install weakincentives[podman]``

**Workspace Digest** (``WorkspaceDigestSection``)
    Caching layer for workspace summaries. Renders cached digest content
    from session state, with dynamic visibility (summary vs full body).

Public Exports
--------------

Planning Tools
~~~~~~~~~~~~~~

PlanningToolsSection
    Prompt section exposing planning tools. Binds to a Session for state
    persistence. Supports configurable planning strategies.

PlanningConfig
    Configuration dataclass for PlanningToolsSection.

PlanningStrategy
    Enum of planning strategies: REACT, PLAN_ACT_REFLECT,
    GOAL_DECOMPOSE_ROUTE_SYNTHESISE.

Plan, PlanStep
    Immutable state objects representing plans and their steps.

PlanStatus, StepStatus
    Type aliases for plan/step lifecycle states.

SetupPlan, AddStep, UpdateStep, ReadPlan
    Event dataclasses for plan mutations.

Virtual Filesystem
~~~~~~~~~~~~~~~~~~

VfsToolsSection
    Prompt section exposing VFS tools (ls, read_file, write_file, edit_file,
    glob, grep, rm). Can mount host directories with filtering.

VfsConfig
    Configuration dataclass for VfsToolsSection.

VfsPath, VfsFile
    Path and file descriptor types for VFS operations.

HostMount
    Configuration for mounting host directories into the VFS.

ListDirectoryParams, ListDirectoryResult, ListDirectory
    Types for directory listing operations.

ReadFileParams, ReadFileResult, ReadFile
    Types for file reading operations.

WriteFileParams, WriteFile
    Types for file writing operations.

EditFileParams
    Parameters for string replacement in files.

GlobParams, GlobMatch
    Types for glob pattern matching.

GrepParams, GrepMatch
    Types for content search operations.

RemoveParams, DeleteEntry
    Types for file/directory deletion.

FileInfo
    File metadata returned by listing operations.

InMemoryFilesystem
    In-memory implementation of the Filesystem protocol.

Python Evaluation
~~~~~~~~~~~~~~~~~

AstevalSection
    Prompt section exposing the evaluate_python tool. Runs code in a
    sandboxed interpreter with VFS file access.

AstevalConfig
    Configuration dataclass for AstevalSection.

EvalParams, EvalResult
    Input/output types for Python evaluation.

EvalFileRead, EvalFileWrite
    File operation descriptors for evaluation context.

Podman Sandbox
~~~~~~~~~~~~~~

PodmanSandboxSection
    Prompt section exposing shell_execute and filesystem tools in a
    Podman container. Lazily creates containers on first tool use.

PodmanSandboxConfig
    Configuration dataclass for PodmanSandboxSection.

PodmanShellParams, PodmanShellResult
    Input/output types for shell command execution.

PodmanWorkspace
    State object representing an active container session.

Workspace Digest
~~~~~~~~~~~~~~~~

WorkspaceDigestSection
    Prompt section that renders cached workspace digests from session
    state. Supports SUMMARY/FULL visibility modes.

WorkspaceDigest
    State object containing digest summary and body.

set_workspace_digest, clear_workspace_digest, latest_workspace_digest
    Session state management functions for digests.

Example Usage
-------------

Planning tools for multi-step tasks::

    from weakincentives.contrib.tools import (
        PlanningToolsSection,
        PlanningConfig,
        PlanningStrategy,
    )
    from weakincentives.runtime.session import Session

    session = Session()
    config = PlanningConfig(strategy=PlanningStrategy.PLAN_ACT_REFLECT)
    planning = PlanningToolsSection(session=session, config=config)

    # Add to prompt template sections
    template = PromptTemplate(
        ns="myapp",
        key="task-runner",
        sections=(planning, ...),
    )

Virtual filesystem with host mounts::

    from weakincentives.contrib.tools import (
        VfsToolsSection,
        VfsConfig,
        HostMount,
    )

    config = VfsConfig(
        mounts=(
            HostMount(
                host_path="src",
                include_glob=("**/*.py",),
                exclude_glob=("**/__pycache__/**",),
            ),
        ),
        allowed_host_roots=("/home/user/project",),
    )
    vfs = VfsToolsSection(session=session, config=config)

Sandboxed Python evaluation::

    from weakincentives.contrib.tools import AstevalSection, AstevalConfig

    config = AstevalConfig(accepts_overrides=True)
    asteval = AstevalSection(session=session, config=config)

    # Agent can now use evaluate_python tool:
    # evaluate_python(code="sum(range(10))")

Containerized execution::

    from weakincentives.contrib.tools import (
        PodmanSandboxSection,
        PodmanSandboxConfig,
        HostMount,
    )

    config = PodmanSandboxConfig(
        image="python:3.12-bookworm",
        mounts=(HostMount(host_path="src"),),
        allowed_host_roots=("/home/user/project",),
    )
    podman = PodmanSandboxSection(session=session, config=config)

    # Agent can now use shell_execute, ls, read_file, etc. in container

Workspace digest caching::

    from weakincentives.contrib.tools import (
        WorkspaceDigestSection,
        set_workspace_digest,
    )

    digest_section = WorkspaceDigestSection(session=session)

    # Populate digest (typically done by WorkspaceDigestOptimizer)
    set_workspace_digest(
        session,
        section_key="workspace-digest",
        body="Full project analysis...",
        summary="Python web app with FastAPI backend.",
    )

    # Section now renders the cached digest

Architecture Notes
------------------

All tool sections follow consistent patterns:

- **Session binding**: Sections bind to a Session for state management
- **Configuration dataclasses**: Use frozen config objects (e.g., VfsConfig)
- **Tool handlers**: Implement ``handler(params, *, context) -> ToolResult``
- **Cloning support**: Sections implement ``clone(**kwargs)`` for optimizer use
- **Resource contribution**: Sections can contribute resources (e.g., Filesystem)

Lazy Loading
------------

``PodmanSandboxSection`` and related types are lazily loaded to avoid
importing the optional ``podman`` dependency at package import time.
Access these types directly or via attribute access on the module.

See Also
--------

- ``weakincentives.prompt.section``: Base Section class
- ``weakincentives.prompt.tool``: Tool and ToolResult types
- ``weakincentives.filesystem``: Filesystem protocol
- ``weakincentives.runtime.session``: Session state management
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
from .filesystem_memory import InMemoryFilesystem
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
PodmanSandboxConfig: Any
PodmanSandboxSection: Any
PodmanShellParams: Any
PodmanShellResult: Any
PodmanWorkspace: Any

__all__ = [
    "AddStep",
    "AstevalConfig",
    "AstevalSection",
    "DeleteEntry",
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
    "RemoveParams",
    "SetupPlan",
    "StepStatus",
    "UpdateStep",
    "VfsConfig",
    "VfsFile",
    "VfsPath",
    "VfsToolsSection",
    "WorkspaceDigest",
    "WorkspaceDigestSection",
    "WriteFile",
    "WriteFileParams",
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
