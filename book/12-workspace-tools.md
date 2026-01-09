# Chapter 12: Workspace Tools

> **Canonical Reference**: See [specs/WORKSPACE.md](/specs/WORKSPACE.md) for the complete specification.

## Introduction

Background agents need to inspect and manipulate code, files, and execution environments. But giving agents unrestricted filesystem or shell access is dangerous—they can corrupt production code, leak credentials, or accidentally delete critical files.

WINK's workspace tools solve this with a **sandbox-first** design:

- **Virtual Filesystem (VFS)** - Copy-on-write in-memory filesystem for safe file operations
- **Planning Tools** - Explicit plan management in session state
- **Workspace Digest** - Cached repository summaries for efficient context
- **Asteval** - Safe Python expression evaluation without imports or IO
- **Podman Sandbox** - Full Linux container isolation for arbitrary code execution

These tools share common patterns:

1. **Session-scoped** - Each tool suite is bound to a specific session
2. **Pure handlers** - All mutations go through reducers, handlers remain side-effect-free
3. **Sandbox isolation** - Host writes disabled by default
4. **Progressive disclosure** - Start with summaries, expand on demand

The result: agents can safely explore and modify code without risking production systems.

## Architecture Overview

```mermaid
flowchart TB
    subgraph Agent["Agent Context"]
        Prompt["Prompt<br/>(with workspace sections)"]
        Session["Session<br/>(state container)"]
    end

    subgraph Tools["Workspace Tool Suites"]
        Planning["PlanningToolsSection<br/>setup_plan, add_step,<br/>update_step, read_plan"]
        VFS["VfsToolsSection<br/>ls, read_file, write_file,<br/>edit_file, glob, grep, rm"]
        Digest["WorkspaceDigestSection<br/>(read-only context)"]
        Asteval["AstevalSection<br/>evaluate_python"]
        Podman["PodmanSandboxSection<br/>shell_execute, file ops"]
    end

    subgraph State["Session State Slices"]
        PlanSlice["Plan<br/>tuple[Plan, ...]"]
        VfsSlice["VirtualFileSystem<br/>tuple[VfsFile, ...]"]
        DigestSlice["WorkspaceDigest<br/>(cached summary)"]
    end

    subgraph Backends["Execution Backends"]
        Memory["In-Memory VFS"]
        Host["Host Mounts<br/>(read-only)"]
        Container["Podman Container<br/>(isolated)"]
        Evaluator["Asteval Interpreter<br/>(restricted)"]
    end

    Prompt --> Tools
    Tools --> Session

    Planning --> PlanSlice
    VFS --> VfsSlice
    Digest --> DigestSlice

    VfsSlice --> Memory
    Memory -.->|hydrate| Host
    Podman --> Container
    Asteval --> Evaluator

    style Agent fill:#e1f5ff
    style State fill:#fff4e1
    style Backends fill:#e1ffe1
```

**Key concepts:**

1. **Tool sections** expose tools to the model and manage state
2. **Session slices** store typed, immutable state (plans, files, digests)
3. **Backends** provide actual execution (in-memory, containers, interpreters)
4. **Host mounts** hydrate VFS from disk but remain read-only

See [Chapter 4: Tools](/book/04-tools.md) for tool fundamentals and [Chapter 5: Sessions](/book/05-sessions.md) for state management.

## 12.1 PlanningToolsSection

Many models perform better when they can externalize their plan into structured memory. `PlanningToolsSection` provides tools for explicit plan management stored in session state.

### Tools Provided

| Tool | Parameters | Description |
|------|------------|-------------|
| `planning_setup_plan` | `steps: list[dict]` | Initialize plan with steps |
| `planning_add_step` | `title, details, after_step_id` | Add new step to plan |
| `planning_update_step` | `step_id, status, details` | Update existing step |
| `planning_read_plan` | - | Read current plan state |

### Data Model

```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class PlanStep:
    id: str
    title: str
    details: str
    status: Literal["pending", "in_progress", "completed", "skipped"]

@dataclass(slots=True, frozen=True)
class Plan:
    steps: tuple[PlanStep, ...] = ()
```

### Usage Example

```python
from weakincentives.contrib.tools import PlanningToolsSection, PlanningStrategy
from weakincentives.runtime import Session, InProcessDispatcher

bus = InProcessDispatcher()
session = Session(bus=bus)

# Add planning tools to prompt
planning = PlanningToolsSection(
    session=session,
    strategy=PlanningStrategy.REACT,  # or CHAIN_OF_THOUGHT
    key="planning",
)

template = PromptTemplate(
    ns="demo",
    key="task-solver",
    sections=(
        MarkdownSection(
            title="Task",
            key="task",
            template="Solve: ${objective}",
        ),
        planning,  # Planning tools available
    ),
)
```

When the model calls `planning_setup_plan`:

```json
{
  "name": "planning_setup_plan",
  "parameters": {
    "steps": [
      {"title": "Analyze requirements", "details": "Review user input"},
      {"title": "Implement solution", "details": "Write code"},
      {"title": "Test solution", "details": "Run tests"}
    ]
  }
}
```

The plan is stored in session state:

```python
from weakincentives.contrib.tools import Plan

# Query current plan
current_plan = session[Plan].latest()

print(current_plan.steps[0].title)
# Output: "Analyze requirements"

# Check step status
for step in current_plan.steps:
    print(f"{step.title}: {step.status}")
```

### Planning Strategies

**REACT** - Model plans incrementally as it works:
- Setup initial plan
- Execute step
- Update step status
- Add new steps as needed
- Repeat

**CHAIN_OF_THOUGHT** - Model plans everything upfront:
- Setup complete plan
- Execute steps sequentially
- Mark each complete when done

Choose REACT for exploratory tasks where the solution path isn't clear. Choose CHAIN_OF_THOUGHT for well-defined tasks with known steps.

## 12.2 VfsToolsSection

The Virtual Filesystem (VFS) provides copy-on-write file operations without touching the host disk. It's the default workspace for "repo agent" patterns.

### Core Design

```mermaid
flowchart TB
    subgraph Hydration["Initialization (Read-Only)"]
        HostDisk["Host Filesystem<br/>/path/to/repo"]
        Mounts["HostMount Config<br/>include/exclude globs"]
        LoadFiles["Load matching files<br/>into memory"]
    end

    subgraph VFS["Virtual Filesystem (In-Memory)"]
        Files["VfsFile objects<br/>path, content, version"]
        FileIndex["Path index<br/>tuple[VfsFile, ...]"]
    end

    subgraph Operations["Tool Operations"]
        Read["read_file<br/>(reads from memory)"]
        Write["write_file<br/>(creates new VfsFile)"]
        Edit["edit_file<br/>(string replacement)"]
        Delete["rm<br/>(marks deleted)"]
    end

    subgraph Session["Session State"]
        VfsSlice["VirtualFileSystem slice"]
        Reducers["Pure reducers"]
        Events["WriteFile, EditFile,<br/>DeletePath events"]
    end

    HostDisk --> Mounts
    Mounts --> LoadFiles
    LoadFiles --> Files
    Files --> FileIndex

    Read --> FileIndex
    Write --> Events
    Edit --> Events
    Delete --> Events

    Events --> Reducers
    Reducers --> VfsSlice
    VfsSlice --> FileIndex

    style HostDisk fill:#ffe1e1
    style VFS fill:#e1f5ff
    style Session fill:#fff4e1
```

**Key properties:**

- **Copy-on-write** - Host files copied to memory at mount time
- **No host writes** - All writes go to in-memory copy
- **Version tracking** - Each file has a version number that increments on edits
- **Event-driven** - All mutations flow through reducers

### Tools Provided

| Tool | Parameters | Description |
|------|------------|-------------|
| `ls` | `path: str` | List directory entries |
| `read_file` | `file_path, offset, limit` | Read file with pagination |
| `write_file` | `file_path, content` | Create new file |
| `edit_file` | `file_path, old_string, new_string, replace_all` | String replacement editing |
| `glob` | `pattern, path` | Match files by glob pattern |
| `grep` | `pattern, path, glob` | Regex search across files |
| `rm` | `path` | Remove file or directory |

### Configuration

```python
from weakincentives.contrib.tools import VfsToolsSection, VfsConfig, HostMount

# Define what to mount from host
mounts = (
    HostMount(
        host_path="src",           # Host directory
        mount_path=None,           # Mount at /src (None = preserve path)
        include_glob=("*.py",),    # Only Python files
        exclude_glob=(
            "__pycache__/*",
            "*.pyc",
        ),
        max_bytes=10_000_000,      # 10 MB limit per mount
        follow_symlinks=False,     # Don't follow symlinks
    ),
    HostMount(
        host_path="README.md",
        mount_path=None,
    ),
)

vfs = VfsToolsSection(
    session=session,
    config=VfsConfig(
        mounts=mounts,
        allowed_host_roots=(".",),  # Security: only allow mounting from cwd
    ),
    key="vfs",
)
```

### Host Mount Safety

The `allowed_host_roots` parameter prevents accidental mounting of sensitive directories:

```python
# This is safe
VfsConfig(
    mounts=(HostMount(host_path="./src"),),
    allowed_host_roots=(".",),  # Can only mount from current directory
)

# This will raise an error
VfsConfig(
    mounts=(HostMount(host_path="/etc/passwd"),),
    allowed_host_roots=(".",),  # /etc is not under current directory
)
```

Always set `allowed_host_roots` to the narrowest path that contains your mounts.

### Usage Example

```python
from weakincentives.contrib.tools import VirtualFileSystem

# Model calls write_file
# → Dispatch WriteFile event
# → Reducer creates new VfsFile
# → Added to VirtualFileSystem slice

# Query VFS state
vfs_state = session[VirtualFileSystem].latest()

# Find specific file
files = vfs_state.files
readme = next((f for f in files if str(f.path) == "/README.md"), None)

if readme:
    print(f"Content: {readme.content}")
    print(f"Version: {readme.version}")
    print(f"Size: {readme.size_bytes} bytes")
```

### Copy-on-Write Semantics

```python
# Initial state: Host file loaded into VFS
# /src/main.py (version 0, content from disk)

# Model calls edit_file
adapter.evaluate(prompt.bind({
    "task": "Add a docstring to main.py"
}))

# Model invokes: edit_file(file_path="/src/main.py", ...)
# → EditFile event dispatched
# → Reducer creates NEW VfsFile with version 1
# → Original host file untouched

# Query updated state
vfs = session[VirtualFileSystem].latest()
main_file = next(f for f in vfs.files if str(f.path) == "/src/main.py")

assert main_file.version == 1  # Version incremented
# Host disk /src/main.py is unchanged
```

This isolation is critical for safety: agents can freely experiment with code changes without risking production files.

### Limits and Constraints

**Content limits:**
- Maximum file size: 48,000 characters per write
- Maximum path depth: 16 segments
- Maximum segment length: 80 characters
- Encoding: UTF-8 text only (no binary files)

**Path normalization:**
- All paths are POSIX-style (`/src/main.py`)
- Relative paths resolved against VFS root (`/`)
- No `..` traversal allowed
- No absolute host paths

See [specs/WORKSPACE.md](/specs/WORKSPACE.md#vfs-limits) for complete limits.

## 12.3 WorkspaceDigestSection

Large repositories can't fit in context windows. `WorkspaceDigestSection` provides a **cached, structured summary** of the repository that models can explore progressively.

### Digest Structure

```python nocheck
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class WorkspaceDigest:
    """Structured summary of repository."""
    summary: str                  # High-level overview
    file_tree: str                # Tree structure of files
    key_files: dict[str, str]     # Important files with content
    patterns: list[str]           # Detected patterns (frameworks, tools)
    total_files: int
    total_size_bytes: int
```

### Visibility Modes

The digest section supports progressive disclosure:

```python
from weakincentives.contrib.tools import WorkspaceDigestSection, DigestVisibility

# Summary only (default)
digest = WorkspaceDigestSection(
    session=session,
    visibility=DigestVisibility.SUMMARY,
)

# Full context (file tree + key files)
digest = WorkspaceDigestSection(
    session=session,
    visibility=DigestVisibility.FULL,
)
```

**SUMMARY mode** includes:
- High-level summary (project type, tech stack)
- File count and total size
- Key patterns detected

**FULL mode** adds:
- Complete file tree
- Contents of key files (README, main modules)
- Dependency manifests (package.json, requirements.txt)

### Usage Example

```python
from weakincentives.contrib.tools import WorkspaceDigest

# Pre-compute digest (expensive, do once)
digest = WorkspaceDigest.from_directory(
    path="/path/to/repo",
    include_glob=("*.py", "*.md"),
    exclude_glob=("__pycache__/*", "*.pyc"),
)

# Store in session
session[WorkspaceDigest].seed(digest)

# Add to prompt
template = PromptTemplate(
    ns="repo",
    key="analyzer",
    sections=(
        MarkdownSection(
            title="Task",
            key="task",
            template="Analyze this repository: ${question}",
        ),
        WorkspaceDigestSection(
            session=session,
            visibility=DigestVisibility.SUMMARY,
        ),
        VfsToolsSection(...),  # Model can explore files via VFS
    ),
)
```

The model sees:

```markdown
## Repository Context

**Summary:**
Python library for building background agents. Uses pytest for testing,
ruff for linting, and pyright for type checking.

**Statistics:**
- Total files: 247
- Total size: 1.2 MB
- Primary language: Python

**Key patterns:**
- Testing: pytest
- Linting: ruff
- Type checking: pyright
- Dependencies: managed via uv
```

The model can then use VFS tools to explore specific files:

```json
{
  "name": "read_file",
  "parameters": {
    "file_path": "/src/weakincentives/runtime/session.py"
  }
}
```

### Benefits

- **Reduced token usage** - Show summaries instead of full file contents
- **Faster startup** - Pre-computed digest loads instantly
- **Better comprehension** - Models see structure before details
- **Cache-friendly** - Digest updates only when files change

## 12.4 AstevalSection

`asteval` provides **safe-ish Python expression evaluation** without imports, file IO, or network access. It's useful for small transformations (string formatting, arithmetic) without granting full shell access.

### What's Allowed

- Basic Python expressions
- String operations
- Math operations
- List/dict comprehensions
- Variable assignment

### What's Forbidden

- `import` statements
- File operations (`open`, `read`, `write`)
- Network access (`socket`, `urllib`)
- Subprocess spawning (`os.system`, `subprocess`)
- Dangerous builtins (`eval`, `exec`, `compile`)

### Tools Provided

| Tool | Parameters | Description |
|------|------------|-------------|
| `evaluate_python` | `code: str` | Execute Python code, return result + stdout/stderr |

### Usage Example

```python
from weakincentives.contrib.tools import AstevalSection

# Add to prompt
template = PromptTemplate(
    ns="demo",
    key="calculator",
    sections=(
        MarkdownSection(
            title="Task",
            key="task",
            template="Perform calculation: ${expression}",
        ),
        AstevalSection(session=session),
    ),
)
```

Model can execute safe computations:

```json
{
  "name": "evaluate_python",
  "parameters": {
    "code": "result = sum(range(1, 101))\nprint(f'Sum: {result}')"
  }
}
```

Result:

```json
{
  "success": true,
  "result": 5050,
  "stdout": "Sum: 5050\n",
  "stderr": ""
}
```

### Installation

```bash
pip install "weakincentives[asteval]"
```

### Security Considerations

While `asteval` blocks many dangerous operations, it's not a perfect sandbox:

- **CPU exhaustion** - Infinite loops can hang execution
- **Memory exhaustion** - Large data structures can consume RAM
- **Logic bugs** - New asteval versions may have bypasses

**Use asteval for:**
- String formatting and manipulation
- Arithmetic and data transformations
- Prototype validation logic

**Don't use asteval for:**
- Untrusted user code
- Long-running computations
- Critical security boundaries

For stronger isolation, use `PodmanSandboxSection` instead.

## 12.5 PodmanSandboxSection

For strict isolation and reproducible execution, `PodmanSandboxSection` runs commands inside a Podman container with networking disabled.

### Container Lifecycle

```mermaid
stateDiagram-v2
    [*] --> SessionStart: Create session

    SessionStart --> CreateOverlay: Allocate cache dir
    CreateOverlay --> ContainerCreated: podman create

    ContainerCreated --> ContainerRunning: podman start
    ContainerRunning --> ToolCall: execute command

    ToolCall --> ContainerRunning: reuse container
    ContainerRunning --> Stopped: session.close()

    Stopped --> Cleanup: podman rm
    Cleanup --> [*]

    note right of ContainerRunning
        Container stays alive
        for entire session.
        Commands execute via
        podman exec.
    end note
```

**Key properties:**

- **One container per session** - Container created on first tool call
- **Persistent state** - Files written in container persist across tool calls
- **Isolated execution** - No network, limited CPU/memory
- **Automatic cleanup** - Container removed when session closes

### Configuration

```python
from weakincentives.contrib.tools import PodmanSandboxSection, PodmanConfig

podman = PodmanSandboxSection(
    session=session,
    config=PodmanConfig(
        image="python:3.12-slim",  # Base image
        cpu_limit=1.0,              # 1 CPU core
        memory_limit="1g",          # 1 GB RAM
        network_enabled=False,      # Disable networking
        timeout=30.0,               # 30s command timeout
        workspace_root="/workspace",
    ),
)
```

### Tools Provided

| Tool | Parameters | Description |
|------|------------|-------------|
| `shell_execute` | `command: str` | Run shell command in container |
| `ls` | `path: str` | List directory entries |
| `read_file` | `file_path, offset, limit` | Read file from container |
| `write_file` | `file_path, content` | Write file to container |
| `edit_file` | `file_path, old_string, new_string` | Edit file in container |
| `glob` | `pattern, path` | Match files by pattern |
| `grep` | `pattern, path, glob` | Search files by regex |
| `rm` | `path` | Remove file or directory |

### Usage Example

```python
template = PromptTemplate(
    ns="demo",
    key="test-runner",
    sections=(
        MarkdownSection(
            title="Task",
            key="task",
            template="Run tests in isolated container.",
        ),
        PodmanSandboxSection(session=session),
    ),
)
```

Model can execute shell commands:

```json
{
  "name": "shell_execute",
  "parameters": {
    "command": "pytest tests/ -v"
  }
}
```

Result includes stdout, stderr, and exit code:

```json
{
  "success": true,
  "exit_code": 0,
  "stdout": "===== test session starts =====\ntests/test_foo.py::test_bar PASSED\n",
  "stderr": ""
}
```

### Sandboxing Guarantees

**What's isolated:**
- Filesystem (writes don't touch host)
- Network (disabled by default)
- CPU and memory (cgroup limits)
- Process table (container PID namespace)

**What's NOT isolated:**
- Host kernel (containers share kernel)
- Side channels (timing, CPU cache)
- Container escape vulnerabilities

**Use Podman for:**
- Running tests in clean environment
- Executing linters and formatters
- Building code without affecting host
- Reproducible command execution

**Don't use Podman for:**
- Untrusted adversarial code
- Critical security boundaries
- Performance-sensitive workloads (adds latency)

### Installation

```bash
# Install Podman
# macOS: brew install podman
# Linux: apt-get install podman

# Install WINK with Podman support
pip install "weakincentives[podman]"
```

### Performance Considerations

Container creation adds ~1-2 seconds of latency to the first tool call. Subsequent calls reuse the running container and are fast (~100ms).

For better performance:
- Use lightweight base images (`python:3.12-slim` vs `python:3.12`)
- Pre-pull images before session starts
- Keep containers warm across sessions (advanced)

See [specs/WORKSPACE.md](/specs/WORKSPACE.md#podman-performance) for optimization strategies.

## 12.6 Wiring a Workspace into a Prompt

Workspace sections compose naturally. Here's a practical pattern used by production agents:

```python nocheck
from typing import Any
from dataclasses import dataclass
from weakincentives.contrib.tools import (
    PlanningToolsSection,
    PlanningStrategy,
    VfsToolsSection,
    VfsConfig,
    WorkspaceDigestSection,
    DigestVisibility,
    WorkspaceDigest,
)
from weakincentives.contrib.tools.vfs_types import HostMount
from weakincentives.prompt import PromptTemplate, MarkdownSection
from weakincentives.runtime import Session

@dataclass(slots=True, frozen=True)
class RepoAnalysisParams:
    objective: str
    repo_path: str

def build_repo_agent_template(
    *,
    session: Session,
    repo_path: str,
) -> PromptTemplate[Any]:
    """Build prompt for repository analysis agent."""

    # 1. Pre-compute workspace digest (cached summary)
    digest = WorkspaceDigest.from_directory(
        path=repo_path,
        include_glob=("*.py", "*.md", "*.json", "*.toml"),
        exclude_glob=("__pycache__/*", "*.pyc", ".venv/*"),
    )
    session[WorkspaceDigest].seed(digest)

    # 2. Configure VFS mounts
    vfs_mounts = (
        HostMount(
            host_path=f"{repo_path}/src",
            mount_path=None,  # Preserve /src path
            include_glob=("*.py",),
            exclude_glob=("__pycache__/*",),
        ),
        HostMount(
            host_path=f"{repo_path}/README.md",
        ),
        HostMount(
            host_path=f"{repo_path}/pyproject.toml",
        ),
    )

    vfs = VfsToolsSection(
        session=session,
        config=VfsConfig(
            mounts=vfs_mounts,
            allowed_host_roots=(repo_path,),
        ),
        key="vfs",
    )

    # 3. Assemble prompt with all workspace tools
    return PromptTemplate(
        ns="examples",
        key="repo-analyzer",
        name="repo_analyzer",
        sections=(
            MarkdownSection(
                title="Objective",
                key="objective",
                template="Analyze this repository: ${objective}",
            ),
            WorkspaceDigestSection(
                session=session,
                visibility=DigestVisibility.SUMMARY,
                key="context",
            ),
            PlanningToolsSection(
                session=session,
                strategy=PlanningStrategy.REACT,
                key="planning",
            ),
            vfs,  # File operations
            MarkdownSection(
                title="Guidelines",
                key="guidelines",
                template=(
                    "Start by reading the README and main modules. "
                    "Use planning tools to organize your analysis. "
                    "Focus on code quality, architecture, and test coverage."
                ),
            ),
        ),
    )
```

### Usage

```python
from weakincentives.runtime import Session, InProcessDispatcher
from weakincentives.adapters.openai import OpenAIAdapter

bus = InProcessDispatcher()
session = Session(bus=bus)

# Build prompt with workspace tools
template = build_repo_agent_template(
    session=session,
    repo_path="/path/to/repo",
)

# Bind params
prompt = template.bind(RepoAnalysisParams(
    objective="Identify potential bugs in error handling",
    repo_path="/path/to/repo",
))

# Evaluate
adapter = OpenAIAdapter(model="gpt-4o")
response = adapter.evaluate(prompt, session=session)

# Agent can now:
# - Call planning_setup_plan to create analysis plan
# - Call read_file to inspect source code
# - Call grep to find error handling patterns
# - Call write_file to document findings (in VFS, not host)
```

### Execution Flow

```mermaid
sequenceDiagram
    participant Agent as LLM Agent
    participant Planning as PlanningToolsSection
    participant Digest as WorkspaceDigestSection
    participant VFS as VfsToolsSection
    participant Session as Session State

    Note over Agent: Sees SUMMARY digest in context

    Agent->>Planning: planning_setup_plan(steps=[...])
    Planning->>Session: Dispatch InitializeSlice(Plan)
    Session-->>Planning: Plan stored

    Agent->>Digest: (reads from rendered section)
    Note over Agent: Understands repo structure

    Agent->>VFS: read_file("/README.md")
    VFS->>Session: Query VirtualFileSystem
    Session-->>VFS: File content (from mount)
    VFS-->>Agent: File content

    Agent->>VFS: grep("try:", glob="*.py")
    VFS->>Session: Query VirtualFileSystem
    Session-->>VFS: Matching files
    VFS-->>Agent: Search results

    Agent->>VFS: write_file("/analysis.md", content="...")
    VFS->>Session: Dispatch WriteFile event
    Session->>Session: Reducer creates VfsFile
    Session-->>VFS: Success

    Agent->>Planning: planning_update_step(step_id="1", status="completed")
    Planning->>Session: Dispatch UpdateStep event
    Session-->>Planning: Plan updated

    Note over Session: All mutations in audit trail
```

### Key Design Patterns

1. **Session-first** - All tool sections receive the same session instance
2. **Pre-seed state** - Digest computed once, before prompt evaluation
3. **Immutable config** - VFS mounts and limits set at construction
4. **Progressive disclosure** - Start with SUMMARY, expand via tools
5. **Pure handlers** - Tool handlers remain side-effect-free (mutations via reducers)

The important idea: **workspace sections are built with the active session**. Each evaluation gets its own session with its own tool sections. This ensures isolation between concurrent agent runs.

## Combining Workspace Backends

You can mix workspace backends for different isolation needs:

```python
template = PromptTemplate(
    ns="advanced",
    key="multi-backend",
    sections=(
        # VFS for safe file exploration
        VfsToolsSection(session=session, config=vfs_config),

        # Asteval for quick calculations
        AstevalSection(session=session),

        # Podman for running tests
        PodmanSandboxSection(session=session, config=podman_config),
    ),
)
```

The agent can choose the appropriate backend:

- **VFS** - Fast, in-memory file operations
- **Asteval** - Safe Python expressions
- **Podman** - Full shell commands with strong isolation

Each backend maintains independent state in the session.

## Best Practices

### 1. Choose the Right Workspace

| Use Case | Recommended Tool | Rationale |
|----------|------------------|-----------|
| Code review | VFS | Fast, safe, no side effects |
| Running tests | Podman | Isolated, reproducible environment |
| Data transformation | Asteval | Quick calculations without overhead |
| Repo analysis | VFS + Digest | Efficient context + file exploration |
| CI/CD tasks | Podman | Full shell access with isolation |

### 2. Limit Host Mounts

Only mount what the agent needs:

```python
# ❌ Bad: Mount entire repo
HostMount(host_path=".", include_glob=("*",))

# ✅ Good: Mount specific directories
mounts = (
    HostMount(host_path="src", include_glob=("*.py",)),
    HostMount(host_path="tests", include_glob=("test_*.py",)),
    HostMount(host_path="README.md"),
)
```

### 3. Use Progressive Disclosure

Start with summaries, expand on demand:

```python
# Initial context: SUMMARY mode
WorkspaceDigestSection(
    session=session,
    visibility=DigestVisibility.SUMMARY,
)

# Agent can explore details via VFS tools
VfsToolsSection(session=session, config=config)
```

### 4. Set Resource Limits

Prevent runaway execution:

```python
# VFS limits
VfsConfig(
    mounts=mounts,
    max_total_bytes=50_000_000,  # 50 MB total
)

# Podman limits
PodmanConfig(
    cpu_limit=1.0,      # 1 CPU core
    memory_limit="1g",  # 1 GB RAM
    timeout=30.0,       # 30s per command
)
```

### 5. Audit Trail via Session Events

All workspace mutations flow through the session:

```python
from weakincentives.runtime import Session

session = Session(bus=bus)

# Agent uses workspace tools
# → Events dispatched
# → Reducers update state
# → Events logged in session timeline

# Review audit trail
for event in session.events:
    if isinstance(event, (WriteFile, EditFile, DeletePath)):
        print(f"File mutation: {event}")
```

This provides:
- **Debugging** - See exactly what the agent changed
- **Testing** - Assert on specific mutations
- **Compliance** - Audit trail for review

### 6. Test Workspace Handlers

Tool handlers are pure functions—easy to test:

```python
from weakincentives.contrib.tools.vfs_handlers import read_file_handler
from weakincentives.contrib.tools import VirtualFileSystem, VfsFile, VfsPath
from weakincentives.prompt import ToolContext

# Create fake VFS state
vfs = VirtualFileSystem(
    files=(
        VfsFile(
            path=VfsPath.from_string("/test.txt"),
            content="Hello, world!",
            version=0,
        ),
    ),
)

# Create mock session
session[VirtualFileSystem].seed(vfs)

# Call handler directly
result = read_file_handler(
    {"file_path": "/test.txt"},
    context=ToolContext(session=session, resources=resources),
)

assert result.success
assert result.data["content"] == "Hello, world!"
```

No model needed—test business logic in isolation.

## Summary

WINK's workspace tools provide:

- **Safe exploration** - VFS copy-on-write prevents host corruption
- **Strong isolation** - Podman containers sandbox arbitrary code
- **Progressive disclosure** - Digest summaries → detailed file reads
- **Explicit planning** - Structured plan management in session state
- **Audit trail** - All mutations recorded as events
- **Composability** - Mix backends for different isolation needs

The key insight: **sandbox first, host writes opt-in**. Default to safe operations (VFS, Asteval), use strong isolation (Podman) when needed, never grant unrestricted host access.

Next, explore debugging and observability in [Chapter 13: Debugging and Observability](/book/13-debugging-observability.md).

## Further Reading

- [specs/WORKSPACE.md](/specs/WORKSPACE.md) - Complete workspace specification
- [Chapter 4: Tools](/book/04-tools.md) - Tool fundamentals
- [Chapter 5: Sessions](/book/05-sessions.md) - State management and reducers
- [code_reviewer_example.py](/code_reviewer_example.py) - Production workspace usage
