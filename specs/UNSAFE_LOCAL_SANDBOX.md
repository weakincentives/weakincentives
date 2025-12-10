# Unsafe Local Sandbox Specification

## Purpose

The Unsafe Local Sandbox provides the same tool surface as `PodmanSandboxSection`
but operates directly on the local filesystem using a temporary directory. This
section is intended for environments where the agent already runs inside an
isolated container (e.g., Docker, Kubernetes pod, or Podman), eliminating the
need for nested container orchestration.

## Guiding Principles

- **API parity with Podman**: Identical tool names, parameters, and result types
  so prompts can swap between sandbox backends without modification.
- **Zero nested isolation**: Relies on external container isolation; does not
  spawn subprocesses in sandboxed environments.
- **Ephemeral workspace**: All files live in a session-scoped temporary directory
  that is cleaned up on section close.
- **Explicit opt-in**: The "unsafe" prefix signals that host-level isolation is
  the caller's responsibility.

## When to Use

Use `UnsafeLocalSandboxSection` when:

- The agent process already runs inside an isolated container
- Podman or Docker is unavailable in the execution environment
- Nested container overhead is unacceptable for latency-sensitive workloads
- CI/CD pipelines already provide job-level isolation

Do **not** use this section when:

- The agent runs on a shared host without container isolation
- Untrusted code may escape the workspace directory
- Network access must be disabled (local sandbox does not restrict networking)

## Data Model

### UnsafeLocalWorkspace

Session state tracking the active workspace:

```python
@dataclass(slots=True, frozen=True)
class UnsafeLocalWorkspace:
    workspace_path: str          # Absolute path to temp directory
    workdir: str                 # Logical working directory (e.g., "/workspace")
    env: tuple[tuple[str, str], ...]  # Base environment variables
    started_at: datetime
    last_used_at: datetime
```

### Configuration

```python
@dataclass(slots=True, frozen=True)
class UnsafeLocalSandboxConfig:
    mounts: Sequence[HostMount] = ()
    allowed_host_roots: Sequence[os.PathLike[str] | str] = ()
    base_environment: Mapping[str, str] | None = None
    workspace_root: os.PathLike[str] | str | None = None  # Override temp location
    clock: Callable[[], datetime] | None = None
    accepts_overrides: bool = False
```

## Workspace Lifecycle

1. **Directory Creation** - Session-specific directory under `tempfile.gettempdir()`
   or configured `workspace_root`
2. **Mount Hydration** - Copy host mounts into workspace (same as Podman)
3. **Tool Execution** - All operations target the workspace directory
4. **Teardown** - Directory removed recursively on section close or garbage
   collection

### Directory Structure

```
/tmp/wink-<session-id>/
├── workspace/           # Logical /workspace root
│   ├── src/             # Hydrated from host mounts
│   └── notes.txt        # Created via write_file
└── .wink-metadata/      # Internal bookkeeping (optional)
```

## Tools

`UnsafeLocalSandboxSection` exposes the same tools as `PodmanSandboxSection`:

| Tool | Parameters | Description |
|------|------------|-------------|
| `ls` | `path: str` | List directory entries |
| `read_file` | `file_path`, `offset`, `limit` | Read file with pagination |
| `write_file` | `file_path`, `content` | Create new file |
| `edit_file` | `file_path`, `old_string`, `new_string`, `replace_all` | String replacement |
| `glob` | `pattern`, `path` | Match files by pattern |
| `grep` | `pattern`, `path`, `glob` | Regex search |
| `rm` | `path` | Remove file or directory |
| `shell_execute` | `command`, `cwd`, `env`, `stdin`, `timeout_seconds` | Run command locally |
| `evaluate_python` | `code` | Execute Python via `python3 -c` |

### Tool Behavior Differences

While the API is identical, implementation details differ:

| Aspect | Podman | Unsafe Local |
|--------|--------|--------------|
| Execution | `podman exec` | `subprocess.run` |
| Isolation | Container namespace | None (host process) |
| Resource limits | cgroups (1 CPU, 1 GiB) | None |
| Network | Disabled | Host network |
| User | nobody (65534:65534) | Current process user |
| Timeout | Container-enforced | `subprocess` timeout |

### Shell Execution

```python
@dataclass(slots=True, frozen=True)
class LocalShellParams:
    command: tuple[str, ...]
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    stdin: str | None = None
    timeout_seconds: float = 30.0
    capture_output: bool = True
```

Commands execute via `subprocess.run()` with:

- `cwd` resolved relative to workspace root
- `env` merged with base environment
- `shell=False` for security (no shell injection)
- Timeout enforced via `subprocess` timeout parameter

Limits (matching Podman):

- Commands: ASCII, ≤4,096 chars combined
- Timeout: 1-120 seconds
- Output: Truncated to 32 KiB

### Python Evaluation

```python
@dataclass(slots=True, frozen=True)
class LocalEvalParams:
    code: str  # ≤2,000 chars
```

Executes via `subprocess.run(["python3", "-c", code], ...)` with:

- Timeout: 5 seconds
- Working directory: workspace root
- Stdout/stderr captured and truncated to 4,096 chars

Note: Unlike `AstevalSection`, this runs full Python with no AST restrictions.
The lack of sandboxing is intentional—external container isolation is assumed.

## Host Mounts

Host mounts behave identically to `PodmanSandboxSection`:

```python
@dataclass(slots=True, frozen=True)
class HostMount:
    host_path: str
    mount_path: VfsPath | None = None
    include_glob: tuple[str, ...] = ()
    exclude_glob: tuple[str, ...] = ()
    max_bytes: int | None = None
    follow_symlinks: bool = False
```

Files are copied (not symlinked) into the workspace at section construction.
The workspace directory must be within the allowed host roots.

## Session Integration

```python
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session
from weakincentives.tools.unsafe_local import (
    UnsafeLocalSandboxSection,
    UnsafeLocalSandboxConfig,
    UnsafeLocalWorkspace,
)

bus = InProcessEventBus()
session = Session(bus=bus)

config = UnsafeLocalSandboxConfig(
    mounts=(HostMount(host_path="src/"),),
    allowed_host_roots=("/path/to/project",),
)
section = UnsafeLocalSandboxSection(session=session, config=config)

# After tool invocations
workspace = session.query(UnsafeLocalWorkspace).latest()
```

### Reducer Registration

The section automatically registers:

- `UnsafeLocalWorkspace` reducer for workspace state
- `VirtualFileSystem` reducer for VFS tracking (mirrors Podman behavior)
- `WriteFile` and `DeleteEntry` reducers for file mutations

## Prompt Template

```python
_UNSAFE_LOCAL_TEMPLATE: Final[str] = """\
You have access to a local workspace directory. The `ls`, `read_file`,
`write_file`, `glob`, `grep`, and `rm` tools operate on `/workspace` inside
the temporary directory. The `evaluate_python` tool executes Python via
`python3 -c` (≤5 seconds). `shell_execute` runs commands directly on the host
(≤120 seconds). No container isolation is applied—this sandbox relies on
external isolation. Do not assume files outside `/workspace` are accessible."""
```

## Usage Example

```python
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session
from weakincentives.prompt import Prompt, MarkdownSection
from weakincentives.tools.unsafe_local import (
    UnsafeLocalSandboxSection,
    UnsafeLocalSandboxConfig,
)
from weakincentives.tools.vfs import HostMount

bus = InProcessEventBus()
session = Session(bus=bus)

prompt = Prompt(
    ns="agents/workspace",
    key="local-sandbox",
    sections=[
        MarkdownSection(
            title="Instructions",
            key="instructions",
            template="Execute tasks in the local workspace.",
        ),
        UnsafeLocalSandboxSection(
            session=session,
            config=UnsafeLocalSandboxConfig(
                mounts=(HostMount(host_path="src/", include_glob=("*.py",)),),
                allowed_host_roots=("/app",),
            ),
        ),
    ],
)
```

## Cloning

`UnsafeLocalSandboxSection` supports `clone(session=..., bus=...)`:

- Creates a new workspace directory for the cloned session
- Re-registers reducers on the new session
- Reapplies host mount hydration
- Fully decoupled from the original section

## Security Considerations

### What This Section Does NOT Provide

- **Process isolation**: Commands run as the current user with full host access
- **Filesystem isolation**: Only workspace paths are validated; escape is possible
  via symlinks or `..` in executed commands
- **Network isolation**: Full host network access
- **Resource limits**: No CPU, memory, or disk quotas

### Required External Isolation

When using `UnsafeLocalSandboxSection`, ensure:

1. **Container runtime**: Run the agent inside Docker, Podman, or similar
2. **Read-only root**: Mount host filesystem read-only where possible
3. **Network policies**: Apply container-level network restrictions
4. **User namespaces**: Run as non-root user with limited capabilities
5. **Seccomp/AppArmor**: Apply security profiles to the container

### Example Docker Configuration

```dockerfile
FROM python:3.12-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash agent
USER agent
WORKDIR /home/agent

# Copy application
COPY --chown=agent:agent . /app

# Run with minimal capabilities
CMD ["python", "-m", "your_agent"]
```

```bash
docker run --rm \
  --read-only \
  --tmpfs /tmp:size=256M \
  --network none \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  your-agent-image
```

## Configuration Reference

### Environment Variables

| Variable | Description |
|----------|-------------|
| `WEAKINCENTIVES_WORKSPACE_ROOT` | Override temporary directory location |
| `WEAKINCENTIVES_SHELL_TIMEOUT` | Default shell timeout (seconds) |

### Limits

| Limit | Value | Configurable |
|-------|-------|--------------|
| Content per write | 48,000 chars | No |
| Path depth | 16 segments | No |
| Segment length | 80 chars | No |
| Command length | 4,096 chars | No |
| Shell timeout | 1-120 seconds | Yes (per-call) |
| Eval timeout | 5 seconds | No |
| Output truncation | 32 KiB | No |

## Error Handling

### ToolValidationError Conditions

- Path escapes workspace boundary
- File not found or not readable
- Command timeout exceeded
- Content exceeds size limits
- Invalid path characters (non-ASCII)

### Subprocess Failures

Shell and eval failures return structured results:

```python
@dataclass(slots=True, frozen=True)
class LocalShellResult:
    command: tuple[str, ...]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool
```

Non-zero exit codes do not raise exceptions; callers inspect `exit_code` and
`stderr` to diagnose failures.

## Comparison with Other Sections

| Feature | VfsToolsSection | PodmanSandboxSection | UnsafeLocalSandboxSection |
|---------|-----------------|----------------------|---------------------------|
| Storage | In-memory | Container overlay | Local temp directory |
| Shell access | No | Yes | Yes |
| Python eval | No (use AstevalSection) | Yes (python3 -c) | Yes (python3 -c) |
| Isolation | N/A | Container | None (external) |
| Dependencies | None | podman-py | None |
| Startup time | Instant | Container pull/start | Instant |
| Host mounts | Copy to memory | Copy to overlay | Copy to temp |

## Migration Guide

### From PodmanSandboxSection

```python
# Before
from weakincentives.tools.podman import PodmanSandboxSection, PodmanSandboxConfig

section = PodmanSandboxSection(
    session=session,
    config=PodmanSandboxConfig(
        mounts=(HostMount(host_path="src/"),),
        allowed_host_roots=("/project",),
    ),
)

# After
from weakincentives.tools.unsafe_local import (
    UnsafeLocalSandboxSection,
    UnsafeLocalSandboxConfig,
)

section = UnsafeLocalSandboxSection(
    session=session,
    config=UnsafeLocalSandboxConfig(
        mounts=(HostMount(host_path="src/"),),
        allowed_host_roots=("/project",),
    ),
)
```

Tool calls require no changes—parameter and result types are identical.

## Limitations

- **No container isolation**: Security depends entirely on external measures
- **Ephemeral state**: Workspace destroyed when section closes
- **Text-only files**: Binary content rejected (same as VFS)
- **No resource limits**: CPU/memory usage unbounded
- **Host network**: Cannot disable network access
- **Symlink risks**: Malicious symlinks may escape workspace validation
