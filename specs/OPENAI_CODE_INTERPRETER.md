# OpenAI Code Interpreter Tool

## Scope and intent

This spec explains how to add native OpenAI `code_interpreter` support to the
existing tool surface without changing the prompt wiring already defined in
`weakincentives.prompt.tool`. The design keeps handlers typed, keeps
`ToolContext` untouched, and introduces a provider-aware tool shim so provider
adapters (notably `OpenAIAdapter` in `weakincentives.adapters.openai`) can map
provider-specific payloads into the existing `ToolResult` flow.

## Provider-aware tools

### Shape

A provider-aware tool reuses `Tool` but adds two fields:

- `provider: Literal["openai", ...]` – the adapter must match this to its own
  provider name before exposing the tool.
- `handle_provider_output: Callable[[ResultT, ToolContext], ToolResult[ResultT]] | None` – optional hook invoked when the adapter already holds a typed
  provider payload. The hook receives the decoded `ResultT` instance and the
  same `ToolContext` the normal handler would see. When absent, adapters invoke
  the standard handler after translating provider payloads into `ParamsT`.

Everything else (type arguments, handler signature, examples, validation rules)
remains exactly as in `Tool`.

### Adapter expectations

- Registration mirrors `Tool` registration; adapters reject provider mismatches
  early with the same DbC-style validation used elsewhere in
  `weakincentives.adapters.shared`.
- When a provider returns a tool call, adapters first try
  `handle_provider_output` (if present). A `ToolResult` returned from this hook
  ends the flow; otherwise the adapter falls back to the normal handler.
- The hook must not rely on `type[ResultT]`; only the concrete result payload and
  `ToolContext` are passed.

## OpenAI code interpreter contract

### Configuration

`OpenAICodeInterpreterTool` binds `provider="openai"` and emits
`{"type": "code_interpreter", "container": ...}` entries in Responses API
payloads. The `container` field accepts:

- **Auto**: `{ "type": "auto", "memory_limit"?: "1g|4g|16g|64g", "file_ids"?: list[str], "host_mounts"?: list[HostMountPayload] }`
- **Existing**: `{ "id": str, "memory_limit"?: "1g|4g|16g|64g", "host_mounts"?: list[HostMountPayload] }`

`HostMountPayload` matches the VFS `HostMount` dataclass in
`weakincentives.tools.vfs`: serialize `mount_path` with leading `/` and forward
`include_glob`, `exclude_glob`, `max_bytes`, and `follow_symlinks` untouched.
Adapters should reject invalid tiers or missing IDs during request construction.

### Parameters and results

- **ParamsT**: empty; the model authors Python directly. The tool description
  must clarify that no user arguments are accepted.
- **ResultT**: typed provider payload representing python output (stdout/stderr
  text plus any container citations). Adapters hand this to
  `handle_provider_output` when present, otherwise they wrap it in `ToolResult`
  using the normal handler path.

### Request/response mapping

- Requests use `client.responses.create` with `tools=[{"type": "code_interpreter", "container": <config>}]`. `tool_choice="required"` forces invocation; otherwise models may choose.
- Any `file_ids` in the prompt input are automatically uploaded to the container
  by OpenAI; host projections come from `host_mounts` as above.
- Responses surface generated files as `container_file_citation` annotations on
  the subsequent assistant message. Adapters should forward these identifiers in
  the `ToolResult.value` payload so callers can fetch artifacts via the
  Containers API.

### Failure modes

- Container creation or reuse failures (invalid tier, expired container) should
  fail fast before issuing the request.
- Python execution errors return `success=False` with stderr preserved; do not
  drop citations even when execution fails.

## Usage example (Python only)

The example mirrors the VFS host-mount wiring from `code_reviewer_example.py`
and shows a pass-through provider-output handler.

```python
from pathlib import Path

from openai import OpenAI
from weakincentives.prompt import ToolContext
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.tools import HostMount, OpenAICodeInterpreterTool, VfsPath

client = OpenAI()

host_mount = HostMount(
    host_path="sunfish",
    mount_path=VfsPath(("workspace", "sunfish")),
    include_glob=("*.py", "*.md", "*.txt"),
    exclude_glob=("**/__pycache__/**",),
    max_bytes=600_000,
)


def to_openai_mount(mount: HostMount) -> dict[str, object]:
    mount_path = "/" + "/".join(mount.mount_path.segments)
    return {
        "host_path": mount.host_path,
        "mount_path": mount_path,
        "include_glob": list(mount.include_glob),
        "exclude_glob": list(mount.exclude_glob),
        "max_bytes": mount.max_bytes,
        "follow_symlinks": mount.follow_symlinks,
    }


def identity_provider_output(result: dict[str, object], context: ToolContext) -> ToolResult[dict[str, object]]:
    return ToolResult(message=str(result.get("output_text", "")), value=result, success=True)


code_interpreter_tool = OpenAICodeInterpreterTool(
    name="openai_python",
    description="Executes model-authored Python in a sandboxed container.",
    container={
        "type": "auto",
        "memory_limit": "4g",
        "file_ids": ["file_csv_upload"],
        "host_mounts": [to_openai_mount(host_mount)],
    },
    handle_provider_output=identity_provider_output,
)

response = client.responses.create(
    model="gpt-4.1",
    tools=[{"type": "code_interpreter", "container": code_interpreter_tool.container}],
    tool_choice="required",
    input=(
        "Load /workspace/sunfish/data/input.csv, calculate the mean of the `value` "
        "column, and generate a histogram PNG."
    ),
)

for content in response.output or []:
    for item in content.get("annotations", []) or []:
        if item.get("type") == "container_file_citation":
            downloaded = client.container_files.content(
                container_id=item["container_id"],
                file_id=item["file_id"],
            )
            Path(item["filename"]).write_bytes(downloaded)
```

## Testing expectations

- Unit tests: validate provider-name matching, handler dispatch preference for
  `handle_provider_output`, and HostMount serialization parity with the VFS
  tools.
- Integration (when `OPENAI_API_KEY` is available): issue a minimal
  `code_interpreter` call, assert artifact citations are preserved, and verify
  `ToolResult.success` mirrors provider execution status.
