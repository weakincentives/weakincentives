# Claude Agent SDK Adapter Specification

> **SDK Version**: This specification targets `claude-agent-sdk>=0.1.15`.

## Purpose

The Claude Agent SDK adapter enables weakincentives prompts to leverage Claude's
full agentic capabilities through the official `claude-agent-sdk` Python package.
This adapter uses the SDK's Hook system to synchronize state bidirectionally
between the SDK's internal execution and the weakincentives Session, preserving
the event-driven architecture while delegating tool execution to Claude Code.

## Guiding Principles

- **Hook-driven state synchronization**: Use SDK hooks to bridge between Claude's
  agentic execution and weakincentives Session state, publishing events and
  enforcing constraints at tool boundaries.
- **Session as source of truth**: The weakincentives Session remains the canonical
  state store; hooks read from and write to it during SDK execution.
- **Native structured output**: Leverage the SDK's `output_format` JSON schema
  support rather than tool-based workarounds.
- **Full SDK power**: Embrace `ClaudeSDKClient` for hooks, custom tools, interrupts,
  and multi-turn conversations.
- **Bidirectional flow**: Session state influences SDK behavior (via PreToolUse),
  and SDK results update Session state (via PostToolUse).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ClaudeAgentSDKAdapter                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │  Prompt     │    │              ClaudeSDKClient                     │   │
│  │  Rendering  │───▶│  ┌────────────────────────────────────────────┐  │   │
│  └─────────────┘    │  │           SDK Agentic Loop                 │  │   │
│                     │  │                                            │  │   │
│  ┌─────────────┐    │  │  ┌──────────┐    ┌──────────┐             │  │   │
│  │  Session    │◀───┼──┼──│PreToolUse│───▶│Tool Exec │             │  │   │
│  │  (State)    │    │  │  │  Hook    │    │ (Native) │             │  │   │
│  │             │───▶│  │  └──────────┘    └────┬─────┘             │  │   │
│  │  - Events   │    │  │        ▲              │                   │  │   │
│  │  - Budget   │    │  │        │              ▼                   │  │   │
│  │  - Deadline │    │  │  ┌─────┴────┐   ┌──────────┐              │  │   │
│  │  - Slices   │    │  │  │ Session  │◀──│PostTool  │              │  │   │
│  └─────────────┘    │  │  │  Sync    │   │Use Hook  │              │  │   │
│                     │  │  └──────────┘   └──────────┘              │  │   │
│                     │  │                                            │  │   │
│                     │  └────────────────────────────────────────────┘  │   │
│                     └──────────────────────────────────────────────────┘   │
│                                                                             │
│  Output: PromptResponse[OutputT] with structured output + events published  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## SDK API Selection

The adapter exclusively uses `ClaudeSDKClient` (not `query()`) because only the
client supports hooks and custom tools:

| Feature | `query()` | `ClaudeSDKClient` |
|---------|-----------|-------------------|
| Hooks | ❌ | ✅ |
| Custom Tools | ❌ | ✅ |
| Interrupts | ❌ | ✅ |
| Multi-turn | ❌ | ✅ |
| State Control | ❌ | ✅ |

## Hook Integration Architecture

### Hook Event Flow

```
User Prompt
    │
    ▼
┌───────────────────┐
│ UserPromptSubmit  │──▶ Inject session context into prompt
│ Hook              │    Query session slices for state
└───────────────────┘
    │
    ▼
┌───────────────────┐
│ PreToolUse Hook   │──▶ Check deadline/budget
│                   │    Query session for tool permissions
│                   │    Modify tool input from session state
│                   │    Block if constraints violated
└───────────────────┘
    │
    ▼
┌───────────────────┐
│ Tool Execution    │    (SDK handles natively)
│ (Claude Code)     │
└───────────────────┘
    │
    ▼
┌───────────────────┐
│ PostToolUse Hook  │──▶ Publish ToolInvoked event
│                   │    Update session state
│                   │    Record tool result for context
└───────────────────┘
    │
    ▼
┌───────────────────┐
│ Stop Hook         │──▶ Finalize budget tracking
│                   │    Publish PromptExecuted event
└───────────────────┘
```

### Hook Implementations

#### PreToolUse Hook

Enforces constraints and injects session state before tool execution:

```python
async def pre_tool_use_hook(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: HookContext,
) -> dict[str, Any]:
    """Intercept tool calls to enforce constraints and inject state."""

    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    # 1. Check deadline
    if adapter._deadline and adapter._deadline.is_expired():
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Deadline exceeded",
            }
        }

    # 2. Check budget
    if adapter._budget_tracker:
        remaining = adapter._budget_tracker.remaining()
        if remaining.total_tokens <= 0:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Token budget exhausted",
                }
            }

    # 3. Query session state for tool-specific behavior
    #    Example: inject workspace paths from session
    if tool_name in ("Read", "Write", "Edit"):
        workspace = session.query(Workspace).latest()
        if workspace and workspace.root:
            # Ensure paths are relative to workspace
            if "file_path" in tool_input:
                tool_input["file_path"] = _resolve_workspace_path(
                    tool_input["file_path"],
                    workspace.root,
                )
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "updatedInput": tool_input,
                    }
                }

    # 4. Check session-based tool permissions
    permissions = session.query(ToolPermissions).latest()
    if permissions and tool_name in permissions.blocked:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Tool {tool_name} blocked by session policy",
            }
        }

    return {}  # Allow tool execution
```

#### PostToolUse Hook

Records tool execution to session and updates state:

```python
async def post_tool_use_hook(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: HookContext,
) -> dict[str, Any]:
    """Capture tool results and publish to session."""

    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})
    tool_output = input_data.get("tool_output", {})
    tool_error = input_data.get("tool_error")

    # 1. Publish ToolInvoked event to session bus
    event = ToolInvoked(
        tool_name=tool_name,
        params=tool_input,
        result=ToolResult(
            message=_extract_message(tool_output),
            value=tool_output,
            success=tool_error is None,
        ),
        success=tool_error is None,
        tool_use_id=tool_use_id,
    )
    session.bus.publish(event)

    # 2. Update session state based on tool results
    #    Example: track files modified
    if tool_name in ("Write", "Edit") and tool_error is None:
        file_path = tool_input.get("file_path", "")
        session.mutate(ModifiedFiles).dispatch(
            FileModified(path=file_path, tool=tool_name)
        )

    # 3. Update plan progress if planning tools used
    if tool_name == "TodoWrite":
        todos = tool_input.get("todos", [])
        session.mutate(Plan).dispatch(PlanUpdated(todos=todos))

    return {}
```

#### UserPromptSubmit Hook

Injects session context into prompts:

```python
async def user_prompt_submit_hook(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: HookContext,
) -> dict[str, Any]:
    """Augment user prompts with session context."""

    original_prompt = input_data.get("prompt", "")

    # Query session for context to inject
    context_parts = []

    # Inject current plan state
    plan = session.query(Plan).latest()
    if plan and plan.steps:
        in_progress = [s for s in plan.steps if s.status == "in_progress"]
        if in_progress:
            context_parts.append(
                f"Current task: {in_progress[0].description}"
            )

    # Inject workspace context
    workspace = session.query(Workspace).latest()
    if workspace:
        context_parts.append(f"Working directory: {workspace.root}")

    if context_parts:
        context_prefix = "\n".join(f"[Context] {c}" for c in context_parts)
        return {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "updatedPrompt": f"{context_prefix}\n\n{original_prompt}",
            }
        }

    return {}
```

#### Stop Hook

Finalizes execution and publishes completion event:

```python
async def stop_hook(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: HookContext,
) -> dict[str, Any]:
    """Handle execution completion."""

    stop_reason = input_data.get("stopReason", "end_turn")

    # Record final state
    adapter._stop_reason = stop_reason

    # Publish completion event will happen in evaluate() after collecting ResultMessage

    return {}
```

## Configuration

### ClaudeAgentSDKClientConfig

```python
@FrozenDataclass()
class ClaudeAgentSDKClientConfig:
    """Client-level configuration for Claude Agent SDK."""

    permission_mode: PermissionMode = "bypassPermissions"
    cwd: str | None = None
    add_dirs: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    setting_sources: tuple[SettingSource, ...] = ()
    sandbox: SandboxSettings | None = None
    max_turns: int | None = None
    include_partial_messages: bool = False


PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]
SettingSource = Literal["user", "project", "local"]
```

**Fields:**

| Field | Default | Description |
|-------|---------|-------------|
| `permission_mode` | `"bypassPermissions"` | Tool permission handling |
| `cwd` | `None` | Working directory for SDK operations |
| `add_dirs` | `()` | Additional accessible directories |
| `env` | `None` | Environment variables passed to SDK |
| `setting_sources` | `()` | Config file sources (empty = isolated) |
| `sandbox` | `None` | Sandboxing configuration |
| `max_turns` | `None` | Maximum conversation turns |
| `include_partial_messages` | `False` | Include streaming partial messages |

### ClaudeAgentSDKModelConfig

```python
@FrozenDataclass()
class ClaudeAgentSDKModelConfig(LLMConfig):
    """Model-level configuration for Claude Agent SDK."""

    model: str = "claude-sonnet-4-5-20250929"
```

### SandboxSettings

```python
@FrozenDataclass()
class SandboxSettings:
    """Sandboxing configuration for SDK execution."""

    enabled: bool = False
    auto_allow_bash_if_sandboxed: bool = False
    excluded_commands: tuple[str, ...] = ()
    allow_unsandboxed_commands: bool = False
    network: SandboxNetworkConfig | None = None


@FrozenDataclass()
class SandboxNetworkConfig:
    """Network-specific sandbox configuration."""

    allow_local_binding: bool = False
    allow_unix_sockets: tuple[str, ...] = ()
```

## ClaudeAgentWorkspace

The `ClaudeAgentWorkspace` provides a VFS-compatible abstraction over a temporary
directory where the Claude Agent SDK operates. It mirrors the `VfsToolsSection`
API while materializing files to disk for native SDK tool access.

### Design Principles

- **VFS API compatibility**: Same `HostMount` configuration as `VfsToolsSection`
- **Temporary isolation**: All operations occur in a session-scoped temp directory
- **Automatic materialization**: Host mounts are copied to temp folder at construction
- **Bidirectional sync**: Changes made by SDK tools sync back to session state
- **Deterministic cleanup**: Temp directory removed on workspace finalization

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ClaudeAgentWorkspace                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐         ┌─────────────────────────────────────────┐   │
│  │   HostMount     │         │         Temporary Directory             │   │
│  │   Config        │────────▶│         /tmp/wink-sdk-XXXXXX/           │   │
│  │                 │  copy   │                                         │   │
│  │  - host_path    │         │  ├── src/                               │   │
│  │  - mount_path   │         │  │   └── main.py     (from host)        │   │
│  │  - include_glob │         │  ├── tests/                             │   │
│  │  - exclude_glob │         │  │   └── test_main.py (from host)       │   │
│  │  - max_bytes    │         │  └── output/         (SDK-created)      │   │
│  └─────────────────┘         │                                         │   │
│                              └─────────────────────────────────────────┘   │
│         │                                    │                              │
│         │                                    │                              │
│         ▼                                    ▼                              │
│  ┌─────────────────┐         ┌─────────────────────────────────────────┐   │
│  │ VirtualFileSystem│◀────────│      sync_workspace_to_vfs()           │   │
│  │ (Session State)  │ on-     │      (Optional post-execution sync)    │   │
│  │                  │ demand  │                                         │   │
│  │ Immutable        │         │      SDK operates directly on temp_dir  │   │
│  │ snapshots        │         │      VFS sync only when explicitly      │   │
│  └─────────────────┘         │      requested                          │   │
│                              └─────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Model

```python
@FrozenDataclass()
class ClaudeAgentWorkspace:
    """Workspace state for Claude Agent SDK execution."""

    temp_dir: Path
    mount_previews: tuple[HostMountPreview, ...]
    vfs_snapshot: VirtualFileSystem
    created_at: datetime
    session_id: str | None = None


@FrozenDataclass()
class HostMount:
    """Configuration for mounting host files into the workspace."""

    host_path: str
    mount_path: VfsPath | None = None
    include_glob: tuple[str, ...] = ()
    exclude_glob: tuple[str, ...] = ()
    max_bytes: int | None = None
    follow_symlinks: bool = False


@FrozenDataclass()
class HostMountPreview:
    """Summary of a materialized host mount."""

    host_path: str
    resolved_host: Path
    mount_path: VfsPath
    entries: tuple[str, ...]
    is_directory: bool
    bytes_copied: int
```

### Workspace Lifecycle

#### Construction

```python
class ClaudeAgentWorkspaceSection(Section[EmptyParams]):
    """Section that manages a temporary workspace for SDK execution."""

    def __init__(
        self,
        *,
        session: SessionProtocol,
        mounts: Sequence[HostMount] = (),
        allowed_host_roots: Sequence[os.PathLike[str] | str] = (),
        temp_dir_prefix: str = "wink-sdk-",
        cleanup_on_finalize: bool = True,
    ) -> None:
        """Initialize workspace with host mounts.

        Args:
            session: Session for state management.
            mounts: Host paths to copy into temp directory.
            allowed_host_roots: Security boundary for host path resolution.
            temp_dir_prefix: Prefix for temporary directory name.
            cleanup_on_finalize: Remove temp dir when section finalizes.
        """
```

#### Materialization

```python
def _materialize_workspace(
    self,
    mounts: Sequence[HostMount],
    allowed_roots: Sequence[Path],
) -> ClaudeAgentWorkspace:
    """Create temp directory and copy host files."""

    # 1. Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix=self._temp_dir_prefix))

    # 2. Process each mount
    previews = []
    vfs_files = []

    for mount in mounts:
        # Resolve and validate host path
        resolved = _resolve_mount_path(mount.host_path, allowed_roots)

        # Determine target path in temp dir
        mount_path = mount.mount_path or VfsPath.from_string(mount.host_path)
        target = temp_dir / mount_path.as_posix()

        # Copy files with glob filtering
        copied_files, preview = _copy_mount_to_temp(
            source=resolved,
            target=target,
            mount=mount,
        )

        previews.append(preview)
        vfs_files.extend(copied_files)

    # 3. Create initial VFS snapshot
    vfs = VirtualFileSystem(files=tuple(sorted(vfs_files, key=lambda f: f.path.segments)))

    # 4. Seed session state
    self._session.mutate(VirtualFileSystem).seed(vfs)

    return ClaudeAgentWorkspace(
        temp_dir=temp_dir,
        mount_previews=tuple(previews),
        vfs_snapshot=vfs,
        created_at=_now(),
    )
```

#### File Copying

```python
def _copy_mount_to_temp(
    source: Path,
    target: Path,
    mount: HostMount,
) -> tuple[list[VfsFile], HostMountPreview]:
    """Copy files from host to temp directory with filtering."""

    copied_files = []
    entries = []
    bytes_copied = 0

    if source.is_file():
        # Single file mount
        target.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_text(encoding="utf-8")
        target.write_text(content, encoding="utf-8")

        vfs_path = VfsPath.from_path(target.relative_to(target.parent.parent))
        copied_files.append(VfsFile(
            path=vfs_path,
            content=content,
            encoding="utf-8",
            size_bytes=len(content.encode("utf-8")),
            version=1,
            created_at=_now(),
            updated_at=_now(),
        ))
        bytes_copied = len(content.encode("utf-8"))
        entries.append(source.name)

    else:
        # Directory mount with glob filtering
        for root, dirs, files in source.walk(follow_symlinks=mount.follow_symlinks):
            rel_root = root.relative_to(source)

            for file_name in files:
                rel_path = rel_root / file_name

                # Apply glob filters
                if not _matches_globs(str(rel_path), mount.include_glob, mount.exclude_glob):
                    continue

                # Check byte budget
                file_path = root / file_name
                content = file_path.read_text(encoding="utf-8")
                file_bytes = len(content.encode("utf-8"))

                if mount.max_bytes and bytes_copied + file_bytes > mount.max_bytes:
                    raise WorkspaceBudgetExceededError(
                        f"Mount exceeds byte budget: {bytes_copied + file_bytes} > {mount.max_bytes}"
                    )

                # Copy to temp
                dest = target / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")

                # Track in VFS
                vfs_path = VfsPath(tuple((target.name, *rel_path.parts)))
                copied_files.append(VfsFile(
                    path=vfs_path,
                    content=content,
                    encoding="utf-8",
                    size_bytes=file_bytes,
                    version=1,
                    created_at=_now(),
                    updated_at=_now(),
                ))

                bytes_copied += file_bytes
                entries.append(str(rel_path))

    return copied_files, HostMountPreview(
        host_path=mount.host_path,
        resolved_host=source,
        mount_path=VfsPath.from_path(target.relative_to(target.parent)),
        entries=tuple(entries[:20]),  # Preview limit
        is_directory=source.is_dir(),
        bytes_copied=bytes_copied,
    )
```

### SDK Integration

#### Adapter Configuration

When a `ClaudeAgentWorkspaceSection` is present, the adapter configures the SDK
to use the temp directory:

```python
async def _evaluate_async(self, prompt, *, session, **kwargs):
    # ...

    # Check for workspace section
    workspace = session.query(ClaudeAgentWorkspace).latest()

    if workspace:
        # Configure SDK to operate in temp directory
        options = ClaudeAgentOptions(
            cwd=str(workspace.temp_dir),
            add_dirs=[str(workspace.temp_dir)],
            # ... other options
        )
```

#### On-Demand VFS Sync

The SDK operates directly on `temp_dir`. VFS sync is **not** performed during tool
execution—instead, call `sync_workspace_to_vfs()` after SDK execution completes
if you need the final state reflected in the session's `VirtualFileSystem` slice:

```python
def sync_workspace_to_vfs(
    session: SessionProtocol,
    workspace: ClaudeAgentWorkspace,
) -> int:
    """Sync temp directory state to VFS after SDK execution.

    Returns:
        Number of files synced (new or modified).
    """
    current_vfs = session.query(VirtualFileSystem).latest()
    known_files = {f.path: f for f in current_vfs.files}
    synced = 0

    for path in workspace.temp_dir.rglob("*"):
        if not path.is_file():
            continue

        rel_path = path.relative_to(workspace.temp_dir)
        vfs_path = VfsPath(rel_path.parts)

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # Skip binary files

        existing = known_files.get(vfs_path)
        if existing is None:
            # New file
            session.mutate(VirtualFileSystem).dispatch(
                WriteFile(path=vfs_path, content=content, mode="create")
            )
            synced += 1
        elif existing.content != content:
            # Modified file
            session.mutate(VirtualFileSystem).dispatch(
                WriteFile(path=vfs_path, content=content, mode="overwrite")
            )
            synced += 1

    return synced
```

This approach avoids per-tool-call overhead and lets callers decide when (or if)
they need VFS state synchronized.

### Cleanup

```python
class ClaudeAgentWorkspaceSection:
    def __init__(self, ...):
        # ...
        if cleanup_on_finalize:
            weakref.finalize(self, self._cleanup, self._workspace.temp_dir)

    @staticmethod
    def _cleanup(temp_dir: Path) -> None:
        """Remove temporary directory on finalization."""
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    def cleanup(self) -> None:
        """Explicitly remove temporary directory."""
        if self._workspace and self._workspace.temp_dir.exists():
            shutil.rmtree(self._workspace.temp_dir, ignore_errors=True)
```

### Usage Example

```python
from weakincentives import Prompt, MarkdownSection
from weakincentives.runtime import Session, InProcessEventBus
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentWorkspaceSection,
    ClaudeAgentWorkspace,
    HostMount,
    sync_workspace_to_vfs,
)
from weakincentives.tools.vfs import VfsPath

# Setup session
bus = InProcessEventBus()
session = Session(bus=bus)

# Create workspace with host mounts
workspace_section = ClaudeAgentWorkspaceSection(
    session=session,
    mounts=(
        HostMount(
            host_path="src",
            mount_path=VfsPath(("project", "src")),
            include_glob=("*.py",),
            max_bytes=1_000_000,  # 1 MB budget
        ),
        HostMount(
            host_path="tests",
            mount_path=VfsPath(("project", "tests")),
            include_glob=("test_*.py",),
        ),
    ),
    allowed_host_roots=("/home/user/myproject",),
)

# Define prompt using workspace
prompt = Prompt[RefactorResult](
    ns="refactor",
    key="codebase",
    sections=[
        workspace_section,
        MarkdownSection(
            title="Task",
            key="task",
            template="Refactor the code in project/src to improve readability",
        ),
    ],
)

# Adapter uses workspace temp_dir as cwd
adapter = ClaudeAgentSDKAdapter(model="claude-sonnet-4-5-20250929")
response = adapter.evaluate(prompt, session=session)

# Optionally sync temp directory changes to VFS (on-demand, not automatic)
workspace = session.query(ClaudeAgentWorkspace).latest()
synced_count = sync_workspace_to_vfs(session, workspace)
print(f"Synced {synced_count} modified files to VFS")

# Now VFS state reflects SDK modifications
vfs = session.query(VirtualFileSystem).latest()
for file in vfs.files:
    print(f"{file.path.as_posix()}: {file.size_bytes} bytes (v{file.version})")

# Cleanup when done (or let finalizer handle it)
workspace_section.cleanup()
```

### Extracting Results

After SDK execution, modified files can be extracted from the workspace:

```python
def extract_modified_files(
    session: SessionProtocol,
    workspace: ClaudeAgentWorkspace,
) -> dict[str, str]:
    """Get all files modified during SDK execution."""

    vfs = session.query(VirtualFileSystem).latest()
    modified = {}

    for file in vfs.files:
        # Check if file was modified (version > 1) or created by SDK
        if file.version > 1 or file.path not in workspace.vfs_snapshot.files:
            modified[file.path.as_posix()] = file.content

    return modified


def write_back_to_host(
    session: SessionProtocol,
    workspace: ClaudeAgentWorkspace,
    target_dir: Path,
) -> list[Path]:
    """Write modified files back to host filesystem."""

    modified = extract_modified_files(session, workspace)
    written = []

    for rel_path, content in modified.items():
        dest = target_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        written.append(dest)

    return written
```

## Structured Output

The SDK natively supports JSON schema validation via `output_format`:

```python
class OutputFormat(TypedDict):
    type: Literal["json_schema"]
    schema: dict[str, Any]
```

The adapter automatically generates this from the prompt's output type:

```python
def _build_output_format(
    self,
    output_type: type[OutputT],
) -> OutputFormat | None:
    """Generate SDK output format from prompt output type."""

    if output_type is type(None):
        return None

    return {
        "type": "json_schema",
        "schema": schema(output_type, mode="serialization"),
    }
```

### Extracting Structured Output

The SDK provides structured output via the `structured_output` attribute on
`ResultMessage` when `output_format` is configured:

```python
def _extract_structured_output(
    self,
    messages: list[Message],
    output_type: type[OutputT],
) -> OutputT | None:
    """Parse structured output from SDK result."""

    # Find the ResultMessage with structured_output
    for message in reversed(messages):
        if isinstance(message, ResultMessage):
            structured = getattr(message, "structured_output", None)
            if structured is not None:
                # Already parsed by SDK, validate against our type
                return parse(output_type, structured, extra="ignore")

    return None
```

**Note**: When `output_format` is configured and the agent cannot produce valid
output matching the schema, the SDK returns an error result with
`subtype: 'error_max_structured_output_retries'`.

## Adapter Implementation

### Constructor

```python
class ClaudeAgentSDKAdapter(ProviderAdapter[OutputT]):
    """Adapter using Claude Agent SDK with hook-based state synchronization."""

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-5-20250929",
        client_config: ClaudeAgentSDKClientConfig | None = None,
        model_config: ClaudeAgentSDKModelConfig | None = None,
        allowed_tools: tuple[str, ...] | None = None,
        disallowed_tools: tuple[str, ...] = (),
    ) -> None:
        """Initialize the Claude Agent SDK adapter.

        Args:
            model: Claude model identifier.
            client_config: SDK client configuration.
            model_config: Model parameter configuration.
            allowed_tools: Tools Claude can use (None = all available).
            disallowed_tools: Tools to explicitly block.
        """
```

### evaluate Method

```python
def evaluate(
    self,
    prompt: Prompt[OutputT],
    *,
    session: SessionProtocol,
    deadline: Deadline | None = None,
    visibility_overrides: Mapping[SectionPath, SectionVisibility] | None = None,
    budget: Budget | None = None,
    budget_tracker: BudgetTracker | None = None,
) -> PromptResponse[OutputT]:
    """Evaluate prompt using Claude Agent SDK with hook-based state sync."""

    return asyncio.run(
        self._evaluate_async(
            prompt,
            session=session,
            deadline=deadline,
            visibility_overrides=visibility_overrides,
            budget=budget,
            budget_tracker=budget_tracker,
        )
    )
```

### Async Implementation

```python
async def _evaluate_async(
    self,
    prompt: Prompt[OutputT],
    *,
    session: SessionProtocol,
    deadline: Deadline | None = None,
    visibility_overrides: Mapping[SectionPath, SectionVisibility] | None = None,
    budget: Budget | None = None,
    budget_tracker: BudgetTracker | None = None,
) -> PromptResponse[OutputT]:
    """Async implementation with full hook integration."""

    # Store references for hook access
    self._session = session
    self._deadline = deadline
    self._budget_tracker = budget_tracker or (
        BudgetTracker(budget) if budget else None
    )

    # 1. Render prompt
    rendered = prompt.render(
        visibility_overrides=visibility_overrides,
    )

    # 2. Publish PromptRendered event
    session.bus.publish(PromptRendered(
        namespace=prompt.ns,
        key=prompt.key,
        adapter_name=self.name,
        text=rendered.text,
        tools=rendered.tools,
    ))

    # 3. Build hook configuration
    hooks = self._build_hooks(session)

    # 4. Build MCP server for weakincentives tools (if any)
    mcp_servers = {}
    allowed_tools = list(self._allowed_tools or [])

    if rendered.tools:
        mcp_config = self._create_mcp_bridge(rendered.tools, session)
        mcp_servers["wink"] = mcp_config
        allowed_tools.extend(
            f"mcp__wink__{tool.name}" for tool in rendered.tools
        )

    # 5. Build output format for structured output
    output_format = self._build_output_format(prompt.output_type)

    # 6. Configure SDK options
    options = ClaudeAgentOptions(
        system_prompt=rendered.text,
        model=self._model,
        permission_mode=self._client_config.permission_mode,
        cwd=self._client_config.cwd,
        add_dirs=list(self._client_config.add_dirs),
        env=dict(self._client_config.env or {}),
        setting_sources=list(self._client_config.setting_sources),
        sandbox=self._sandbox_to_dict(self._client_config.sandbox),
        max_turns=self._client_config.max_turns,
        allowed_tools=allowed_tools,
        disallowed_tools=list(self._disallowed_tools),
        mcp_servers=mcp_servers,
        hooks=hooks,
        output_format=output_format,
    )

    # 7. Execute via ClaudeSDKClient
    messages: list[Message] = []
    result_message: ResultMessage | None = None

    async with ClaudeSDKClient(options=options) as client:
        # Initial prompt is empty since system_prompt contains rendered content
        # User prompt comes from any dynamic input
        user_prompt = self._extract_user_prompt(prompt)
        await client.query(user_prompt)

        async for message in client.receive_response():
            messages.append(message)
            if isinstance(message, ResultMessage):
                result_message = message

    # 8. Process result
    if result_message is None:
        raise PromptEvaluationError(
            message="No result message received from SDK",
            prompt_name=prompt.name,
            phase="response",
        )

    # 9. Update budget tracker from usage
    if self._budget_tracker and result_message.usage:
        self._budget_tracker.record(
            input_tokens=result_message.usage.get("input_tokens", 0),
            output_tokens=result_message.usage.get("output_tokens", 0),
        )

    # 10. Extract structured output
    output = self._extract_structured_output(
        messages,
        prompt.output_type,
    )

    # 11. Extract text response
    text = self._extract_text_response(messages)

    # 12. Publish PromptExecuted event
    session.bus.publish(PromptExecuted(
        namespace=prompt.ns,
        key=prompt.key,
        adapter_name=self.name,
        text=text,
        output=output,
        duration_ms=result_message.duration_ms,
        input_tokens=result_message.usage.get("input_tokens") if result_message.usage else None,
        output_tokens=result_message.usage.get("output_tokens") if result_message.usage else None,
        total_cost_usd=result_message.total_cost_usd,
        session_id=result_message.session_id,
    ))

    return PromptResponse(
        prompt_name=prompt.name,
        text=text,
        output=output,
    )
```

### Building Hooks

```python
def _build_hooks(
    self,
    session: SessionProtocol,
) -> dict[HookEvent, list[HookMatcher]]:
    """Construct hook configuration for state synchronization."""

    return {
        "PreToolUse": [
            HookMatcher(
                hooks=[
                    self._make_pre_tool_use_hook(session),
                ],
                timeout=30.0,
            ),
        ],
        "PostToolUse": [
            HookMatcher(
                hooks=[
                    self._make_post_tool_use_hook(session),
                ],
                timeout=30.0,
            ),
        ],
        "UserPromptSubmit": [
            HookMatcher(
                hooks=[
                    self._make_user_prompt_submit_hook(session),
                ],
                timeout=10.0,
            ),
        ],
        "Stop": [
            HookMatcher(
                hooks=[
                    self._make_stop_hook(session),
                ],
                timeout=10.0,
            ),
        ],
    }
```

## Custom Tool Bridging

For weakincentives tools that need to execute locally (not via Claude Code),
the adapter creates an MCP server bridge:

```python
def _create_mcp_bridge(
    self,
    tools: tuple[Tool[Any, Any], ...],
    session: SessionProtocol,
) -> McpSdkServerConfig:
    """Create MCP server exposing weakincentives tools."""

    sdk_tools = []

    for tool in tools:
        if tool.handler is None:
            continue  # Skip tools without handlers

        sdk_tool = self._wrap_tool(tool, session)
        sdk_tools.append(sdk_tool)

    return create_sdk_mcp_server(
        name="wink",
        version="1.0.0",
        tools=sdk_tools,
    )


def _wrap_tool(
    self,
    tool: Tool[ParamsT, ResultT],
    session: SessionProtocol,
) -> SdkMcpTool[Any]:
    """Wrap a weakincentives tool as an SDK MCP tool."""

    input_schema = (
        schema(tool.params_type, mode="serialization")
        if tool.params_type is not type(None)
        else {}
    )

    @sdk_tool(tool.name, tool.description, input_schema)
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        # Parse arguments
        if tool.params_type is type(None):
            params = None
        else:
            params = parse(tool.params_type, args, extra="forbid")

        # Build context
        context = ToolContext(
            prompt=self._current_prompt,
            rendered_prompt=self._current_rendered,
            adapter=self,
            session=session,
            deadline=self._deadline,
            budget_tracker=self._budget_tracker,
        )

        # Execute handler
        try:
            result = tool.handler(params, context=context)
            return {
                "content": [{"type": "text", "text": result.message}],
                "isError": not result.success,
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }

    return handler
```

## Error Handling

### SDK Exception Mapping

```python
def _normalize_sdk_error(
    self,
    error: Exception,
    prompt_name: str,
) -> PromptEvaluationError:
    """Convert SDK exceptions to weakincentives error types."""

    if isinstance(error, CLINotFoundError):
        return PromptEvaluationError(
            message="Claude Code CLI not found. Install: npm install -g @anthropic-ai/claude-code",
            prompt_name=prompt_name,
            phase="request",
        )

    if isinstance(error, CLIConnectionError):
        return ThrottleError(
            message=str(error),
            prompt_name=prompt_name,
            phase="request",
            details=ThrottleDetails(
                kind=ThrottleKind.TIMEOUT,
                retry_after=None,
                attempts=1,
                retry_safe=True,
                provider_payload=None,
            ),
        )

    if isinstance(error, ProcessError):
        return PromptEvaluationError(
            message=f"Claude Code process failed: {error.stderr}",
            prompt_name=prompt_name,
            phase="request",
            provider_payload={
                "exit_code": error.exit_code,
                "stderr": error.stderr,
            },
        )

    if isinstance(error, CLIJSONDecodeError):
        return PromptEvaluationError(
            message=f"Failed to parse SDK response: {error}",
            prompt_name=prompt_name,
            phase="response",
        )

    return PromptEvaluationError(
        message=str(error),
        prompt_name=prompt_name,
        phase="request",
    )
```

### Hook Error Handling

Hooks should not raise exceptions; errors are returned as hook responses:

```python
async def safe_hook_wrapper(
    hook_fn: HookCallback,
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: HookContext,
) -> dict[str, Any]:
    """Wrap hook to catch exceptions and convert to responses."""
    try:
        return await hook_fn(input_data, tool_use_id, context)
    except DeadlineExceededError:
        return {
            "hookSpecificOutput": {
                "hookEventName": input_data.get("hookEventName", "PreToolUse"),
                "permissionDecision": "deny",
                "permissionDecisionReason": "Deadline exceeded",
            }
        }
    except BudgetExhaustedError:
        return {
            "hookSpecificOutput": {
                "hookEventName": input_data.get("hookEventName", "PreToolUse"),
                "permissionDecision": "deny",
                "permissionDecisionReason": "Budget exhausted",
            }
        }
    except Exception as e:
        logger.exception("Hook error", extra={"error": str(e)})
        return {}  # Allow execution to continue
```

## Budget and Deadline Integration

### Deadline Enforcement

Deadlines are enforced at two points:

1. **Before SDK invocation**: Fail fast if already expired
2. **In PreToolUse hook**: Block tools if deadline exceeded during execution

```python
async def _evaluate_async(self, prompt, *, deadline, **kwargs):
    if deadline and deadline.is_expired():
        raise DeadlineExceededError(
            message="Deadline expired before SDK invocation",
            prompt_name=prompt.name,
        )
    # ... execution continues with hook-based enforcement
```

### Budget Tracking

Budget is tracked from `ResultMessage.usage` and enforced via PreToolUse:

```python
# In PostToolUse hook or after ResultMessage
if result_message.usage:
    budget_tracker.record(
        input_tokens=result_message.usage.get("input_tokens", 0),
        output_tokens=result_message.usage.get("output_tokens", 0),
    )

# In PreToolUse hook
if budget_tracker.remaining().total_tokens <= 0:
    return {"hookSpecificOutput": {"permissionDecision": "deny", ...}}
```

## Telemetry

### Events Published

| Event | When | Source |
|-------|------|--------|
| `PromptRendered` | After prompt render | `evaluate()` |
| `ToolInvoked` | Each tool execution | `PostToolUse` hook |
| `PromptExecuted` | After SDK completion | `evaluate()` |

### Extended Event Data

`PromptExecuted` includes SDK-specific fields:

```python
@dataclass
class PromptExecuted:
    # Standard fields
    namespace: str
    key: str
    adapter_name: str
    text: str | None
    output: Any

    # SDK-specific fields
    duration_ms: int
    duration_api_ms: int | None
    input_tokens: int | None
    output_tokens: int | None
    total_cost_usd: float | None
    session_id: str
    num_turns: int
```

### Logging

```python
logger.info(
    "claude_agent_sdk.evaluate.start",
    extra={
        "event": "sdk.evaluate.start",
        "prompt_name": prompt.name,
        "model": self._model,
        "stateful": self._stateful,
        "tool_count": len(rendered.tools),
        "has_structured_output": output_format is not None,
    },
)

logger.info(
    "claude_agent_sdk.evaluate.complete",
    extra={
        "event": "sdk.evaluate.complete",
        "prompt_name": prompt.name,
        "duration_ms": result_message.duration_ms,
        "input_tokens": result_message.usage.get("input_tokens") if result_message.usage else None,
        "output_tokens": result_message.usage.get("output_tokens") if result_message.usage else None,
        "total_cost_usd": result_message.total_cost_usd,
        "session_id": result_message.session_id,
        "num_turns": result_message.num_turns,
    },
)
```

## File Structure

```
src/weakincentives/adapters/
├── claude_agent_sdk/
│   ├── __init__.py           # Public exports
│   ├── adapter.py            # ClaudeAgentSDKAdapter
│   ├── config.py             # Configuration dataclasses
│   ├── workspace.py          # ClaudeAgentWorkspaceSection
│   ├── _hooks.py             # Hook implementations
│   ├── _bridge.py            # MCP tool bridge
│   ├── _workspace_sync.py    # VFS sync utilities
│   ├── _async_utils.py       # Async/sync bridging
│   └── _errors.py            # Error normalization
```

## Usage Examples

### Basic Evaluation with Session State

```python
from weakincentives import Prompt, MarkdownSection
from weakincentives.runtime import Session, InProcessEventBus
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
)

# Setup session
bus = InProcessEventBus()
session = Session(bus=bus)

# Configure adapter
adapter = ClaudeAgentSDKAdapter(
    model="claude-sonnet-4-5-20250929",
    client_config=ClaudeAgentSDKClientConfig(
        permission_mode="acceptEdits",
        cwd="/home/user/project",
    ),
)

# Define prompt
prompt = Prompt[CodeReview](
    ns="review",
    key="code",
    sections=[
        MarkdownSection(
            title="Task",
            key="task",
            template="Review the code in $file_path",
        ),
    ],
)

# Evaluate - hooks sync state to session automatically
response = adapter.evaluate(prompt, session=session)

# Session now contains all tool invocation events
for event in session.events():
    if isinstance(event, ToolInvoked):
        print(f"Tool used: {event.tool_name}")
```

### With Custom Tools and Session Slices

```python
from weakincentives import Tool, ToolResult, ToolContext
from weakincentives.runtime import Session

# Define session slice for tracking
@dataclass(slots=True, frozen=True)
class AnalysisState:
    files_analyzed: tuple[str, ...] = ()
    issues_found: int = 0

# Register reducer
session.mutate(AnalysisState).register(
    FileAnalyzed,
    lambda state, event: AnalysisState(
        files_analyzed=(*state.files_analyzed, event.path),
        issues_found=state.issues_found + event.issues,
    ),
)

# Define tool that reads session state
@dataclass(slots=True, frozen=True)
class AnalyzeParams:
    file_path: str

def analyze_handler(
    params: AnalyzeParams,
    *,
    context: ToolContext,
) -> ToolResult[None]:
    # Query session state
    state = context.session.query(AnalysisState).latest()
    if params.file_path in state.files_analyzed:
        return ToolResult(
            message=f"Already analyzed: {params.file_path}",
            value=None,
            success=True,
        )

    # Perform analysis...
    issues = do_analysis(params.file_path)

    # Update session state
    context.session.mutate(AnalysisState).dispatch(
        FileAnalyzed(path=params.file_path, issues=len(issues))
    )

    return ToolResult(
        message=f"Found {len(issues)} issues",
        value=None,
        success=True,
    )

analyze_tool = Tool[AnalyzeParams, None](
    name="analyze",
    description="Analyze a source file for issues",
    handler=analyze_handler,
)

# Tool is bridged to SDK and state flows through session
prompt = Prompt[AnalysisReport](
    ns="analysis",
    key="codebase",
    sections=[
        MarkdownSection(
            title="Task",
            key="task",
            template="Analyze all Python files in the project",
            tools=(analyze_tool,),
        ),
    ],
)

response = adapter.evaluate(prompt, session=session)

# Query final state
final_state = session.query(AnalysisState).latest()
print(f"Analyzed {len(final_state.files_analyzed)} files")
print(f"Found {final_state.issues_found} total issues")
```

### With Deadline and Budget

```python
from weakincentives.deadlines import Deadline
from weakincentives.budget import Budget, BudgetTracker
from datetime import timedelta

deadline = Deadline.from_now(timedelta(minutes=5))
budget = Budget(max_total_tokens=50000)
tracker = BudgetTracker(budget)

try:
    response = adapter.evaluate(
        prompt,
        session=session,
        deadline=deadline,
        budget=budget,
        budget_tracker=tracker,
    )
except DeadlineExceededError:
    print("Task took too long")
except BudgetExhaustedError:
    print("Used too many tokens")

# Check remaining budget
remaining = tracker.remaining()
print(f"Tokens remaining: {remaining.total_tokens}")
```

## Limitations

- **CLI dependency**: Requires Claude Code CLI (`@anthropic-ai/claude-code`)
- **Async overhead**: `asyncio.run()` creates new event loop per call
- **Hook latency**: Each tool call incurs hook overhead
- **No streaming in evaluate()**: Results collected after completion
- **Session hooks not supported**: `SessionStart`, `SessionEnd`, `Notification`
  hooks are not available in the Python SDK

## Testing

### Unit Tests

- Mock `ClaudeSDKClient` to test hook wiring
- Verify state synchronization flows
- Test error normalization for all exception types
- Validate structured output extraction

### Integration Tests

- Require Claude Code CLI installation
- Test full hook→session event flow
- Verify multi-turn conversation state
- Test budget enforcement via hooks

### Fixtures

- `tests/fixtures/claude_agent_sdk/` contains sample message sequences
- `tests/helpers/claude_agent_sdk.py` provides mock client and hooks

## Dependencies

```toml
[project.optional-dependencies]
claude-agent-sdk = [
    "claude-agent-sdk>=0.1.15",
]
```
