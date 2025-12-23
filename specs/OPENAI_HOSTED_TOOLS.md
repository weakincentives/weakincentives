# OpenAI Hosted Tools Specification

> **SDK**: `openai>=1.60.0`

## Purpose

This specification extends the OpenAI adapter to support hosted tools (Code
Interpreter, File Search, Web Search), container-backed workspaces, and
server-side context compaction. The design mirrors the Claude Agent SDK
adapter's workspace API so user code remains portable across providers.

## Requirements

- Python: `pip install 'weakincentives[openai]'`
- OpenAI API key with Responses API access
- Models supporting hosted tools (gpt-4o, gpt-5.x, o3, o4-mini)

## Design Principles

- **API parity**: `OpenAIContainerWorkspaceSection` accepts the same `HostMount`
  configuration as `ClaudeAgentWorkspaceSection`.
- **Encapsulation**: Container lifecycle, file upload/download, and compaction
  are hidden behind familiar abstractions.
- **Filesystem protocol**: `ContainerFilesystem` implements `Filesystem` so
  tools work unchanged.
- **Lazy materialization**: Containers are created on first use, not at
  section construction.

## Module Structure

```
src/weakincentives/adapters/openai/
├── __init__.py              # Public exports
├── adapter.py               # Extended OpenAIAdapter
├── config.py                # OpenAIHostedToolsConfig, CompactionConfig
├── workspace.py             # OpenAIContainerWorkspaceSection
├── container.py             # ContainerManager
├── container_fs.py          # ContainerFilesystem
└── compaction.py            # CompactionManager, CompactionState
```

## Configuration

### OpenAIHostedToolsConfig

Controls which hosted tools are enabled for the adapter.

```python
@FrozenDataclass()
class OpenAIHostedToolsConfig:
    """Configuration for OpenAI's hosted tools.

    Attributes:
        code_interpreter: Enable Code Interpreter tool. Required for
            container-backed workspaces.
        file_search: File search configuration. None disables file search.
        web_search: Enable web search tool.
    """

    code_interpreter: bool = False
    file_search: FileSearchConfig | None = None
    web_search: bool = False
```

### ContainerConfig

Controls container resource allocation and behavior.

```python
@FrozenDataclass()
class ContainerConfig:
    """Configuration for Code Interpreter containers.

    Attributes:
        memory_limit: Container memory limit (e.g., "1g", "4g", "8g").
            Defaults to "4g".
        mode: Container creation mode. "auto" reuses active containers
            from previous turns. "explicit" creates a dedicated container.
    """

    memory_limit: str = "4g"
    mode: Literal["auto", "explicit"] = "explicit"
```

### CompactionConfig

Controls server-side context compaction behavior.

```python
@FrozenDataclass()
class CompactionConfig:
    """Server-side context compaction configuration.

    Compaction compresses conversation history into encrypted, opaque items
    that preserve task-relevant information while reducing token footprint.
    Enables multi-hour sessions that exceed single context window limits.

    Attributes:
        enabled: Enable automatic compaction. Defaults to True.
        threshold_tokens: Trigger compaction when context exceeds this size.
            Defaults to 100,000 tokens.
        zdr_mode: Zero Data Retention mode. When True, uses encrypted_content
            items that are never persisted on OpenAI servers.
    """

    enabled: bool = True
    threshold_tokens: int = 100_000
    zdr_mode: bool = False
```

### FileSearchConfig

Configuration for the file search hosted tool.

```python
@FrozenDataclass()
class FileSearchConfig:
    """Configuration for file search tool.

    Attributes:
        vector_store_ids: Pre-existing vector store IDs to search.
        max_results: Maximum results per search. Defaults to 20.
    """

    vector_store_ids: tuple[str, ...] = ()
    max_results: int = 20
```

## Workspace Section

### OpenAIContainerWorkspaceSection

Prompt section that mounts host files into an OpenAI container. Uses the same
`HostMount` type as `ClaudeAgentWorkspaceSection` for configuration parity.

```python
class OpenAIContainerWorkspaceSection(MarkdownSection[_WorkspaceSectionParams]):
    """Workspace section that mounts host files into an OpenAI container.

    Files are copied to a local temp directory (respecting include/exclude
    globs and byte limits), tarballed, and uploaded to a container on first
    adapter evaluation. The container provides an isolated execution
    environment where Code Interpreter can read and modify files.

    Example:
        >>> workspace = OpenAIContainerWorkspaceSection(
        ...     session=session,
        ...     mounts=(
        ...         HostMount(
        ...             host_path="/path/to/repo",
        ...             mount_path="repo",
        ...             exclude_glob=(".git/*", "__pycache__/*"),
        ...             max_bytes=10_000_000,
        ...         ),
        ...     ),
        ...     allowed_host_roots=("/path/to",),
        ... )
    """

    def __init__(
        self,
        *,
        session: Session,
        mounts: Sequence[HostMount] = (),
        allowed_host_roots: Sequence[os.PathLike[str] | str] = (),
        container_config: ContainerConfig | None = None,
        sync_on_cleanup: bool = False,
        accepts_overrides: bool = False,
    ) -> None:
        """Initialize the workspace section.

        Args:
            session: Session for state management.
            mounts: Host mount configurations. Uses the same HostMount type
                as ClaudeAgentWorkspaceSection.
            allowed_host_roots: Security boundary for host paths. Paths
                outside these roots raise WorkspaceSecurityError.
            container_config: Container resource configuration. None uses
                defaults (4g memory, explicit mode).
            sync_on_cleanup: If True, download modified files back to
                temp_dir on cleanup. Defaults to False.
            accepts_overrides: Whether section accepts prompt overrides.
        """
        ...

    @property
    def temp_dir(self) -> Path:
        """Path to the temporary workspace directory."""
        ...

    @property
    def mount_previews(self) -> tuple[HostMountPreview, ...]:
        """Summaries of each materialized mount."""
        ...

    @property
    def filesystem(self) -> Filesystem:
        """Filesystem for this workspace.

        Returns a ContainerFilesystem that implements the Filesystem protocol
        using OpenAI's container file API. Operations are lazy - the container
        is created on first filesystem access.
        """
        ...

    @property
    def container_id(self) -> str | None:
        """Container ID once materialized, None before first use."""
        ...

    def cleanup(self) -> None:
        """Clean up workspace resources.

        If sync_on_cleanup is True, downloads all files from the container
        to temp_dir before cleanup. Always removes the local temp directory.

        Note: Containers expire automatically after 20 minutes of inactivity.
        This method does not explicitly delete the container.
        """
        ...
```

### HostMount (Reused)

The existing `HostMount` type from the Claude Agent SDK adapter is reused
without modification.

```python
@FrozenDataclass()
class HostMount:
    """Configuration for mounting host files into the workspace.

    Attributes:
        host_path: Absolute or relative path to the host file or directory.
        mount_path: Relative path within the workspace. Defaults to the
            basename of host_path.
        include_glob: Glob patterns to include (empty = include all).
        exclude_glob: Glob patterns to exclude.
        max_bytes: Maximum total bytes to copy. None means unlimited.
        follow_symlinks: Whether to follow symbolic links when copying.
    """

    host_path: str
    mount_path: str | None = None
    include_glob: tuple[str, ...] = ()
    exclude_glob: tuple[str, ...] = ()
    max_bytes: int | None = None
    follow_symlinks: bool = False
```

## Container Filesystem

### ContainerFilesystem

Implements the `Filesystem` protocol using OpenAI's container file API.

```python
class ContainerFilesystem(Filesystem):
    """Filesystem backed by an OpenAI container.

    Provides the standard Filesystem interface over OpenAI's container API.
    The container is created lazily on first operation. If the container
    expires (20 min idle timeout), it is recreated and files re-uploaded
    transparently.

    This class is not typically instantiated directly - use
    OpenAIContainerWorkspaceSection which manages the lifecycle.
    """

    def __init__(
        self,
        *,
        client: OpenAI,
        container_config: ContainerConfig,
        tarball_path: Path | None = None,
        local_root: Path | None = None,
    ) -> None:
        """Initialize the container filesystem.

        Args:
            client: OpenAI client instance.
            container_config: Container configuration.
            tarball_path: Path to tarball for initial upload. None means
                empty container.
            local_root: Local directory mirroring container contents. Used
                for sync operations.
        """
        ...

    @property
    def container_id(self) -> str | None:
        """Container ID, or None if not yet created."""
        ...

    def ensure_container(self) -> str:
        """Create container if needed, upload initial files, return ID.

        Idempotent - returns existing container_id if already created.
        Handles expired containers by creating a new one and re-uploading.
        """
        ...

    def read(self, path: str) -> bytes:
        """Read file contents from the container."""
        ...

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read file as text from the container."""
        ...

    def write(self, path: str, content: bytes) -> None:
        """Write file contents to the container."""
        ...

    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        """Write text to file in the container."""
        ...

    def exists(self, path: str) -> bool:
        """Check if path exists in the container."""
        ...

    def is_file(self, path: str) -> bool:
        """Check if path is a file in the container."""
        ...

    def is_dir(self, path: str) -> bool:
        """Check if path is a directory in the container."""
        ...

    def list_dir(self, path: str) -> list[str]:
        """List directory contents in the container."""
        ...

    def delete(self, path: str) -> None:
        """Delete file from the container."""
        ...

    def glob(self, pattern: str) -> list[str]:
        """Find files matching glob pattern in the container."""
        ...

    def download_all(self, dest: Path) -> None:
        """Download all container files to local directory.

        Used by sync_on_cleanup to retrieve modified files.
        """
        ...

    def cleanup(self) -> None:
        """Release resources. Container expires automatically."""
        ...
```

## Compaction

### CompactionState

Session slice tracking compaction state across turns.

```python
@FrozenDataclass()
class CompactionState:
    """Session state for context compaction.

    Stored as a session slice, tracks encrypted items from previous
    compaction passes and usage statistics.

    Attributes:
        encrypted_items: Opaque encrypted content from /responses/compact.
            Prepended to input on subsequent requests.
        last_compaction_tokens: Token count at last compaction.
        compaction_count: Number of compaction passes performed.
    """

    encrypted_items: tuple[dict[str, Any], ...] = ()
    last_compaction_tokens: int = 0
    compaction_count: int = 0
```

### CompactionManager

Manages the compaction lifecycle within the adapter.

```python
class CompactionManager:
    """Manages server-side context compaction.

    Tracks token usage, triggers compaction when threshold is exceeded,
    and stores encrypted items in session state for replay.

    Example:
        >>> manager = CompactionManager(client, config, session)
        >>> messages = manager.inject_encrypted_items(messages)
        >>> # ... run model ...
        >>> if manager.should_compact(response.usage.total_tokens):
        ...     messages = manager.compact(messages)
    """

    def __init__(
        self,
        client: OpenAI,
        config: CompactionConfig,
        session: Session,
    ) -> None: ...

    def should_compact(self, current_tokens: int) -> bool:
        """Check if compaction should be triggered."""
        ...

    def compact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compact messages via /responses/compact endpoint.

        Calls the OpenAI compaction endpoint, stores encrypted items in
        session state, and returns the compacted message list.

        Returns:
            Compacted messages ready for next request.
        """
        ...

    def inject_encrypted_items(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prepend stored encrypted items to message list.

        Called before each request to restore compacted context.
        """
        ...
```

## Adapter Integration

### Extended OpenAIAdapter

The adapter detects workspace sections and wires up container/compaction.

```python
class OpenAIAdapter(ProviderAdapter[Any]):
    """Adapter for OpenAI's Responses API with hosted tools support.

    Extends the base OpenAI adapter with:
    - Hosted tools (Code Interpreter, File Search, Web Search)
    - Container-backed workspaces via OpenAIContainerWorkspaceSection
    - Server-side context compaction

    Example:
        >>> adapter = OpenAIAdapter(
        ...     model="gpt-5.2-codex",
        ...     hosted_tools=OpenAIHostedToolsConfig(code_interpreter=True),
        ...     compaction=CompactionConfig(threshold_tokens=100_000),
        ... )
    """

    def __init__(
        self,
        *,
        model: str,
        model_config: OpenAIModelConfig | None = None,
        client_config: OpenAIClientConfig | None = None,
        client: OpenAI | None = None,
        hosted_tools: OpenAIHostedToolsConfig | None = None,
        compaction: CompactionConfig | None = None,
        tool_choice: ToolChoice = "auto",
    ) -> None:
        """Initialize the adapter.

        Args:
            model: Model identifier (e.g., "gpt-5.2-codex", "o3").
            model_config: Model generation parameters.
            client_config: Client configuration. Mutually exclusive with client.
            client: Pre-configured OpenAI client.
            hosted_tools: Hosted tools configuration. None disables all
                hosted tools.
            compaction: Compaction configuration. None disables compaction.
            tool_choice: Tool selection directive.
        """
        ...

    def evaluate(
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        resources: ResourceRegistry | None = None,
    ) -> PromptResponse[OutputT]:
        """Evaluate a prompt with hosted tools and compaction support.

        Lifecycle:
        1. Detect OpenAIContainerWorkspaceSection in prompt sections
        2. If found, ensure container is materialized with uploaded files
        3. Inject encrypted items from previous compaction (if any)
        4. Build tool configuration with hosted tools + function tools
        5. Run inner loop
        6. Check compaction threshold, compact if needed
        7. Return response

        The container_id is injected into the Code Interpreter tool
        configuration automatically when a workspace section is present.
        """
        ...
```

### Hosted Tool Wire Format

When hosted tools are enabled, the adapter includes them in the request:

```python
def _build_hosted_tools(
    self,
    workspace: OpenAIContainerWorkspaceSection | None,
) -> list[dict[str, Any]]:
    """Build hosted tool specifications for the request."""
    tools: list[dict[str, Any]] = []

    if self._hosted_tools.code_interpreter:
        tool_spec: dict[str, Any] = {"type": "code_interpreter"}
        if workspace is not None:
            container_id = workspace.filesystem.ensure_container()
            tool_spec["container"] = container_id
        tools.append(tool_spec)

    if self._hosted_tools.file_search is not None:
        tools.append({
            "type": "file_search",
            "vector_store_ids": list(
                self._hosted_tools.file_search.vector_store_ids
            ),
            "max_num_results": self._hosted_tools.file_search.max_results,
        })

    if self._hosted_tools.web_search:
        tools.append({"type": "web_search"})

    return tools
```

## User Stories

### Story 1: Portable Code Review

As a developer, I want to switch between Claude Agent SDK and OpenAI without
changing my workspace configuration.

```python
from dataclasses import dataclass

# Shared workspace config - works with both adapters
def create_workspace(session: Session) -> PromptSection:
    return OpenAIContainerWorkspaceSection(  # or ClaudeAgentWorkspaceSection
        session=session,
        mounts=(
            HostMount(
                host_path="/repos/myproject",
                mount_path="project",
                exclude_glob=(".git/*", "node_modules/*", "*.pyc"),
                max_bytes=20_000_000,
            ),
        ),
        allowed_host_roots=("/repos",),
    )


@dataclass(frozen=True)
class ReviewResult:
    summary: str
    issues: list[str]
    suggestions: list[str]


session = Session(bus=InProcessDispatcher())
workspace = create_workspace(session)

try:
    adapter = OpenAIAdapter(
        model="gpt-5.2-codex",
        hosted_tools=OpenAIHostedToolsConfig(code_interpreter=True),
    )

    template = PromptTemplate[ReviewResult](
        ns="review",
        key="code",
        sections=[
            MarkdownSection(
                title="Task",
                key="task",
                template=(
                    "Review the Python code in project/src/ for:\n"
                    "- Security vulnerabilities\n"
                    "- Performance issues\n"
                    "- Code style problems\n\n"
                    "Use Python to read and analyze files systematically."
                ),
            ),
            workspace,
        ],
    )

    response = adapter.evaluate(Prompt(template), session=session)
    print(response.output)
finally:
    workspace.cleanup()
```

### Story 2: Long-Running Analysis with Compaction

As a data scientist, I want to run multi-hour analysis sessions that exceed
single context window limits.

```python
adapter = OpenAIAdapter(
    model="gpt-5.2-codex",
    hosted_tools=OpenAIHostedToolsConfig(code_interpreter=True),
    compaction=CompactionConfig(
        threshold_tokens=100_000,
        zdr_mode=True,  # Compliance requirement
    ),
)

workspace = OpenAIContainerWorkspaceSection(
    session=session,
    mounts=(
        HostMount(
            host_path="/data/analysis",
            mount_path="data",
            max_bytes=50_000_000,
        ),
    ),
    allowed_host_roots=("/data",),
    sync_on_cleanup=True,  # Get results back
)

# Session can run for hours - compaction handles context growth
for task in analysis_tasks:
    response = adapter.evaluate(
        create_analysis_prompt(task, workspace),
        session=session,
    )
    process_results(response.output)

workspace.cleanup()  # Downloads generated files
```

### Story 3: Hybrid Tools (Hosted + Custom)

As a platform developer, I want to combine OpenAI's hosted tools with my
custom function tools.

```python
@dataclass(frozen=True)
class QueryParams:
    sql: str


@dataclass(frozen=True)
class QueryResult:
    rows: list[dict[str, Any]]
    row_count: int


def run_query(params: QueryParams, *, context: ToolContext) -> ToolResult[QueryResult]:
    """Execute SQL against internal database."""
    db = context.resources.get(DatabaseConnection)
    rows = db.execute(params.sql)
    return ToolResult(
        message=f"Query returned {len(rows)} rows",
        value=QueryResult(rows=rows, row_count=len(rows)),
    )


query_tool = Tool[QueryParams, QueryResult](
    name="run_query",
    description="Execute SQL query against the analytics database",
    handler=run_query,
)

adapter = OpenAIAdapter(
    model="gpt-5.2-codex",
    hosted_tools=OpenAIHostedToolsConfig(
        code_interpreter=True,  # For data analysis
        web_search=True,        # For documentation lookup
    ),
)

template = PromptTemplate[AnalysisResult](
    ns="analytics",
    key="report",
    sections=[
        MarkdownSection(
            title="Task",
            key="task",
            template="Analyze sales trends using SQL queries and Python.",
            tools=(query_tool,),  # Custom tool alongside hosted tools
        ),
        workspace,
    ],
)
```

## Container Lifecycle

### Creation

Containers are created lazily on first filesystem operation or when the
adapter detects a workspace section. The creation flow:

1. Workspace section collects files per `HostMount` configuration
2. Files are copied to temp directory (respecting globs/byte limits)
3. Temp directory is tarballed
4. `ContainerFilesystem.ensure_container()` is called
5. Container is created via `POST /v1/containers`
6. Tarball is uploaded via `POST /v1/containers/{id}/files`
7. Container ID is stored and returned

### Expiration

OpenAI containers expire after 20 minutes of inactivity. The adapter handles
this transparently:

1. On filesystem operation, check if container still exists
2. If expired (404 response), create new container
3. Re-upload tarball from local cache
4. Continue operation

### Cleanup

On `workspace.cleanup()`:

1. If `sync_on_cleanup=True`, download all files from container
2. Remove local temp directory
3. Container expires naturally (no explicit deletion needed)

## Compaction Flow

### Automatic Compaction

When `CompactionConfig.enabled=True`:

1. After each response, check `usage.total_tokens`
2. If exceeds `threshold_tokens`, trigger compaction
3. Call `POST /v1/responses/compact` with current messages
4. Store returned `encrypted_content` items in session
5. Replace message history with compacted form

### ZDR Mode

When `CompactionConfig.zdr_mode=True`:

1. All requests include `store=false`
2. Encrypted items are returned instead of stored server-side
3. Client manages encrypted content across requests
4. No conversation state persists on OpenAI servers

### Replay

On each new request:

1. Check session for stored `CompactionState`
2. If encrypted items exist, prepend to message list
3. Continue with normal request flow

## Parity with Claude Agent SDK

| Feature | Claude Agent SDK | OpenAI Hosted Tools |
|---------|------------------|---------------------|
| `HostMount` configuration | ✅ | ✅ Same type |
| `allowed_host_roots` | ✅ | ✅ |
| `include_glob` / `exclude_glob` | ✅ | ✅ |
| `max_bytes` | ✅ | ✅ |
| `mount_previews` property | ✅ | ✅ |
| `filesystem` property | ✅ `HostFilesystem` | ✅ `ContainerFilesystem` |
| `cleanup()` method | ✅ | ✅ + optional sync |
| Network isolation | `NetworkPolicy` | Container VM isolation |
| File tools | Read/Write/Edit/Glob/Grep | Code Interpreter |
| Bash execution | Native Bash tool | Code Interpreter subprocess |
| Context scaling | Client-side snapshots | Server-side compaction |
| MCP bridging | ✅ Native | ❌ Function tools only |

### Intentional Differences

1. **Tool interface**: Claude uses discrete file tools. OpenAI uses Python via
   Code Interpreter. Both achieve the same result with different UX.

2. **Execution model**: Claude tools run in bwrap sandbox. OpenAI runs in
   container VM. Container provides stronger isolation.

3. **State management**: Claude session snapshots are client-side. OpenAI
   compaction is server-side with optional ZDR.

## Error Handling

### ContainerExpiredError

Raised when a container operation fails due to expiration and re-creation
also fails.

```python
class ContainerExpiredError(WinkError):
    """Container expired and could not be recreated."""

    container_id: str
    original_error: Exception
```

### CompactionError

Raised when compaction fails.

```python
class CompactionError(WinkError):
    """Server-side compaction failed."""

    token_count: int
    original_error: Exception
```

### Transparent Recovery

Most transient errors are handled transparently:

- Container expiration → recreate and re-upload
- Rate limits → exponential backoff (existing throttle policy)
- Network errors → retry per adapter policy

## Testing

### Unit Tests

- `ContainerFilesystem` operations against mock container API
- `CompactionManager` state machine transitions
- `OpenAIContainerWorkspaceSection` mount processing (reuse existing tests)
- Hosted tool wire format generation

### Integration Tests

- End-to-end workspace mount → container → file operations
- Compaction trigger and replay across multiple turns
- Container expiration recovery
- Hybrid hosted + function tool execution

### Fixtures

- `tests/fixtures/containers/` - Mock container API responses
- `tests/helpers/openai.py` - Mock OpenAI client with container support

## Migration Guide

### From VfsToolsSection

If using `VfsToolsSection` with OpenAI adapter today:

```python
# Before: VFS tools with InMemoryFilesystem
workspace = VfsToolsSection(filesystem=InMemoryFilesystem())

# After: Container workspace with real file access
workspace = OpenAIContainerWorkspaceSection(
    session=session,
    mounts=(HostMount(host_path="/data", mount_path="data"),),
    allowed_host_roots=("/data",),
)
```

### From Claude Agent SDK

Direct replacement - same `HostMount` configuration:

```python
# Before: Claude Agent SDK
workspace = ClaudeAgentWorkspaceSection(
    session=session,
    mounts=(HostMount(...),),
    allowed_host_roots=(...),
)
adapter = ClaudeAgentSDKAdapter(...)

# After: OpenAI with hosted tools
workspace = OpenAIContainerWorkspaceSection(
    session=session,
    mounts=(HostMount(...),),  # Same config!
    allowed_host_roots=(...),
)
adapter = OpenAIAdapter(
    model="gpt-5.2-codex",
    hosted_tools=OpenAIHostedToolsConfig(code_interpreter=True),
)
```
