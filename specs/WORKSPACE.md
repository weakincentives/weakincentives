# Workspace Tools Specification

## Purpose

Workspace tools provide deterministic, session-scoped surfaces for file
operations, code execution, and repository context. Covers virtual filesystem
(VFS), Podman sandbox, Python evaluation (asteval), and workspace digest.

**Implementation:** `src/weakincentives/contrib/tools/`

## Guiding Principles

- **Sandbox first**: VFS in-memory; Podman no network; host access limited to mounts
- **Predictable paths**: Normalized POSIX-style, ASCII-only, relative to session root
- **Single source of state**: Reducers own mutations; handlers remain pure
- **VFS-compatible surface**: All backends expose same tools

## Virtual Filesystem

Session-scoped file operations without writing to host disk. Content hydrated
from explicit host mounts at section construction.

**Implementation:** `src/weakincentives/contrib/tools/vfs.py`

### Data Model

| Type | Fields | Description |
|------|--------|-------------|
| `VfsPath` | `segments: tuple[str, ...]` | Path as tuple |
| `VfsFile` | `path`, `content`, `encoding`, `size_bytes`, `version`, timestamps | File metadata |

### Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| `ls` | `path: str` | List directory |
| `read_file` | `file_path`, `offset`, `limit` | Read with pagination |
| `write_file` | `file_path`, `content` | Create new file |
| `edit_file` | `file_path`, `old_string`, `new_string`, `replace_all` | String replacement |
| `glob` | `pattern`, `path` | Match by pattern |
| `grep` | `pattern`, `path`, `glob` | Regex search |
| `rm` | `path` | Remove file/directory |

### Limits

| Limit | Value |
|-------|-------|
| Content per write | 48,000 characters |
| Path depth | 16 segments |
| Segment length | 80 characters |
| Encoding | UTF-8 text only |

### Host Mounts

| Field | Type | Description |
|-------|------|-------------|
| `host_path` | `str` | Source path |
| `mount_path` | `VfsPath \| None` | Target path |
| `include_glob` | `tuple[str, ...]` | Include patterns |
| `exclude_glob` | `tuple[str, ...]` | Exclude patterns |
| `max_bytes` | `int \| None` | Size limit |
| `follow_symlinks` | `bool` | Follow symlinks |

### Backend Integration

VFS tools operate through a backend that handles file operations. Host mounts
hydrate initial content at section construction:

```python
section = VfsToolsSection(
    mounts=(HostMount(host_path="docs/", include_glob=("*.md",)),),
    allowed_host_roots=("/path/to/project",),
)
```

## Podman Sandbox

Isolated Linux container for shell commands and file operations.

**Implementation:** `src/weakincentives/contrib/tools/podman.py`

### Workspace Lifecycle

1. **Overlay Root** - Session-specific directory under cache
2. **Container Creation** - Python 3.12 image, 1 CPU, 1 GiB RAM, no network
3. **Startup** - `sleep infinity` with health check
4. **Reuse** - Subsequent calls share container
5. **Teardown** - Stopped and removed on section close

### Tools

All VFS tools plus:

| Tool | Description |
|------|-------------|
| `shell_execute` | Run command in container (<=120s) |
| `evaluate_python` | Execute Python via `python3 -c` (<=5s) |

### Shell Execution

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `command` | `tuple[str, ...]` | - | Commands to execute |
| `cwd` | `str \| None` | `None` | Working directory |
| `env` | `Mapping[str, str]` | `{}` | Environment variables |
| `stdin` | `str \| None` | `None` | Standard input |
| `timeout_seconds` | `float` | `30.0` | Timeout |
| `capture_output` | `bool` | `True` | Capture stdout/stderr |

### Limits

- Commands: ASCII, <=4,096 chars combined
- Timeout: 1-120 seconds
- Output: Truncated to 32 KiB

### Configuration

Environment variables: `PODMAN_BASE_URL`, `PODMAN_IDENTITY`, `PODMAN_CONNECTION`,
`WEAKINCENTIVES_CACHE`

## Python Evaluation (ASTEval)

Sandboxed Python expression evaluation.

**Implementation:** `src/weakincentives/contrib/tools/asteval.py`

### Tool Contract

| Type | Fields |
|------|--------|
| `EvalParams` | `code`, `globals`, `reads`, `writes` |
| `EvalResult` | `value_repr`, `stdout`, `stderr`, `globals`, `reads`, `writes` |

### Sandbox Environment

- `asteval.Interpreter(use_numpy=False, minimal=True)`
- Whitelisted symtable: math, statistics, `read_text`, `write_text`, `print`
- Disabled: import, exec, eval, `ALL_DISALLOWED` nodes
- Timeout: 5 seconds

### VFS Integration

Reads resolve from session VFS snapshot. Writes queue through reducer pipeline.

### Installation

```bash
pip install weakincentives[asteval]
```

## Workspace Digest

Task-agnostic repository summaries.

**Implementation:** `src/weakincentives/contrib/optimizers/workspace_digest.py`

### Resolution Order

1. **Session snapshot** - `latest_workspace_digest(session, key)`
2. **Override fallback** - From `PromptOverridesStore`
3. **Placeholder** - Default text with warning log

### Data Captured

- Repository layout and notable directories
- Tooling commands (tests, linting, formatting)
- Known caveats and recurring pitfalls

### Optimization Workflow

```python
context = OptimizationContext(
    adapter=adapter,
    dispatcher=session.dispatcher,
    overrides_store=store,
    overrides_tag="v1",
)
optimizer = WorkspaceDigestOptimizer(context, store_scope=PersistenceScope.SESSION)
result = optimizer.optimize(prompt, session=session)
```

| Scope | Behavior |
|-------|----------|
| `SESSION` | Stores in session slice only |
| `GLOBAL` | Persists to overrides store |

### Result Types

| Type | Fields |
|------|--------|
| `OptimizationResult[T]` | `response`, `artifact`, `metadata` |
| `WorkspaceDigestResult` | `response`, `digest`, `scope`, `section_key` |

## Cloning

All workspace sections support `clone(session=..., dispatcher=...)`:
- Re-registers reducers on new session
- Binds telemetry to new event dispatcher
- Reapplies host mount hydration
- Fully decoupled from original section

## Limitations

- **Ephemeral state**: All workspace data dies with session
- **Text-only VFS**: Binary content rejected
- **No network**: Podman containers have no network access
- **Cooperative timeout**: ASTEval interrupts cooperatively only
- **Synchronized clocks**: Timestamps require UTC synchronization
