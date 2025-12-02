# OpenAI Code Interpreter Tool

## Purpose

This specification introduces `ProviderTool` as a provider-aware extension of the
existing `Tool` abstraction and defines `OpenAICodeInterpreterTool`, a
provider-specialized tool for the native OpenAI **code_interpreter** integration
("python tool"). It documents request/response mapping for the Responses API,
container lifecycle expectations, file-handling rules, and runtime hooks for
post-processing provider outputs using the existing `ToolContext` contract.

## Goals

- **Provider-aware tools**: Allow tools to declare the provider they bind to and
  to register optional provider-output handlers while preserving the existing
  `Tool` handler lifecycle and `ToolContext` injection.
- **Native Code Interpreter support**: Add first-class support for OpenAI's
  `type: "code_interpreter"` tool, including container configuration, tool
  choice semantics, and file handling.
- **Consistent orchestration**: Keep prompt/runtime contracts intact (DbC,
  telemetry, context propagation) while enabling provider-specific plumbing and
  result handling.

## Non-Goals

- Adding streaming support for code interpreter responses; only blocking
  Responses calls are in scope.
- Changing general prompt rendering, `Tool` registration APIs, or the
  `ToolResult` success/failure semantics.
- Persisting container state beyond provider guarantees; containers remain
  ephemeral.

## ProviderTool abstraction

### Definition

`ProviderTool[ParamsT, ResultT, ConfigT]` extends `Tool` with two additions:

- **`provider: Literal[...]`** – identifies the provider this tool definition is
  valid for (e.g., `"openai"`). Adapters MAY reject provider tools that do not
  match their provider name during registration.
- **`handle_provider_output: Callable[[ResultT, ToolContext], ToolResult[ResultT]] | None`** – optional hook that processes provider tool
  outputs (e.g., OpenAI tool-call content parts) into a normalized `ToolResult`.
  Provider payloads are mapped into `ResultT` before invocation, and the typed
  result instance is passed alongside `ToolContext` so handlers can construct
  typed payloads or delegate to the default handler. When absent, adapters fall
  back to the default behavior of pushing raw provider payloads through the
  registered handler.

`ProviderTool` retains the base invariants:

- Params/results remain strongly typed dataclasses with serde compatibility.
- Handlers accept `(params: ParamsT, *, context: ToolContext)` and return
  `ToolResult[ResultT]`.
- `ToolContext` continues to expose prompt metadata, adapter reference, session
  info, deadlines, and event bus access so provider output handlers can perform
  secondary prompts or telemetry.

### Registration rules

1. **Declaration**: Providers register via the same section-based mechanism as
   `Tool`, but adapters MUST validate `tool.provider` against their own provider
   name before exposing it to the LLM.
1. **Handler selection**: When a provider returns a tool call, the adapter SHOULD
   first invoke `handle_provider_output` (if present) to translate provider
   payloads into a `ToolResult`. If it is `None`, the adapter SHOULD decode the
   provider payload into the declared `ParamsT` and dispatch the standard
   handler.
1. **Context reuse**: Both the provider-output handler and the standard handler
   receive the same `ToolContext` instance so downstream logic can access prompt
   IDs, deadlines, or emit events.

## OpenAICodeInterpreterTool contract

### Overview

`OpenAICodeInterpreterTool` is a `ProviderTool` implementation bound to
`provider="openai"` and `type="code_interpreter"`. It exposes typed
configuration for containers and enforces OpenAI Responses payload shapes while
remaining compatible with the runtime's tool registry.

### Configuration schema (`ConfigT`)

- `container: AutoContainerConfig | ExistingContainerRef`
  - **AutoContainerConfig**: `{ "type": "auto", "memory_limit": "1g|4g|16g|64g", "file_ids"?: list[str] }`
  - **ExistingContainerRef**: `{ "id": str, "memory_limit"?: "1g|4g|16g|64g" }`
- Default `memory_limit` is `"1g"` unless explicitly set to `"4g"`, `"16g"`, or
  `"64g"`.
- Adapters MUST reject invalid tiers or missing container references during
  DbC validation before issuing a request.

### Input/Output schemas

- **Params (`ParamsT`)**: none. Code Interpreter tool calls do not expect
  user-supplied arguments; the model writes and executes Python directly inside
  the container. Adapters SHOULD register an empty-params schema with a clear
  description that the tool executes Python code authored by the model.
- **Result (`ResultT`)**: model-generated Python output rendered as text plus
  any file citations. The canonical `ToolResult.message` SHOULD contain the
  provider-rendered text output; `value` MAY carry a structured summary of
  stdout/stderr and generated file metadata if available.

### Request construction

1. **Tool declaration**: Register the tool with `{"type": "code_interpreter", "container": <config>}` in the `tools` array when building
   `client.responses.create(...)` payloads.
1. **Tool choice**: Honor prompt metadata; setting `tool_choice="required"`
   forces the model to invoke the python tool. Otherwise, allow the model to
   choose based on instructions.
1. **Input envelope**: Encode instructions and user inputs in the Responses
   `input` array. Do **not** use legacy `messages` envelopes.
1. **Structured outputs**: When the prompt also requests native structured
   output, attach `text.format` alongside the `code_interpreter` tool; ensure
   both coexist in the request body.

### Response handling

1. **Tool call detection**: Parse `response.output[0].content` entries of type
   `"output_text"` and `"input_text"` with `tool_calls` to detect
   `code_interpreter` invocations. Normalize provider output into the declared
   `ResultT` prior to handler dispatch.
1. **Handler invocation**: If `handle_provider_output` exists on
   `OpenAICodeInterpreterTool`, invoke it with the typed `ResultT` payload and
   the active `ToolContext`. Otherwise, decode the tool-call payload (if any)
   and dispatch the default handler.
1. **Result assembly**: Map provider outputs into `ToolResult`:
   - `message`: human-readable python output.
   - `value`: optional structured payload containing stdout, stderr, exit code,
     and citations for generated files.
   - `success`: `False` if provider reports execution failure or container
     errors; otherwise `True`.
1. **Citations**: Preserve `container_file_citation` annotations on the next
   assistant message so downstream consumers can download generated artifacts.

### Container lifecycle and files

- Containers are **ephemeral** and expire after ~20 minutes of inactivity. When
  expired, the provider rejects reuse; adapters MUST surface a clear error and
  SHOULD allow callers to provision a new container.
- Any file included in the model input is automatically uploaded to the active
  container. Additional uploads happen through the `container_files` endpoints;
  adapters SHOULD provide helper utilities for attaching `file_id` references
  and SHOULD expose host paths using the shared `HostMount` VFS API when a
  runtime needs to project local files into the provider's container
  configuration.
- Model-generated files appear as `container_file_citation` annotations with
  `container_id`, `file_id`, and `filename`. The runtime SHOULD propagate these
  identifiers in `ToolResult.value` when available.
- Supported file types include common code, document, image, archive, and data
  formats (e.g., `.csv`, `.png`, `.zip`, `.pdf`, `.py`, `.ipynb`). Reject
  unsupported types during upload validation.

### Error handling

- Treat container provisioning failures, invalid memory tiers, and expired
  container references as configuration errors surfaced before issuing the
  provider call.
- Execution-time failures (e.g., Python exceptions) SHOULD mark
  `ToolResult.success=False` and include stderr excerpts in `value` while still
  relaying provider annotations.
- Content filtering or safety blocks from the provider map to the standard
  finish-reason normalization path; adapters SHOULD raise assertion errors when
  required output is filtered to maintain DbC guarantees.

## Usage examples

### Registering `OpenAICodeInterpreterTool`

The VFS spec defines `HostMount` (see `specs/VFS_TOOLS.md` and
`code_reviewer_example.py`) as the canonical way to project host files into
tools. The OpenAI adapter expects the same structure when wiring
`host_mounts` for the code interpreter. The snippet below shows how to wire an
`OpenAICodeInterpreterTool` with a pass-through provider-output handler that
returns the provider-rendered text while preserving the structured payload.

```python
from pathlib import Path

from openai import OpenAI
from weakincentives.prompt import ToolContext
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.tools import HostMount, OpenAICodeInterpreterTool, VfsPath

client = OpenAI()

# Reuse the VFS HostMount API (same shape used in the code reviewer example)
host_mount = HostMount(
    host_path="sunfish",  # Relative to the allowed host roots configured by the orchestrator
    mount_path=VfsPath(("workspace", "sunfish")),
    include_glob=("*.py", "*.md", "*.txt"),
    exclude_glob=("**/__pycache__/**",),
    max_bytes=600_000,
)

# HostMount uses VfsPath; serialize it into the provider payload
def _to_openai_mount(mount: HostMount) -> dict[str, object]:
    mount_path = "/" + "/".join(mount.mount_path.segments) if mount.mount_path else "/"
    return {
        "host_path": mount.host_path,
        "mount_path": mount_path,
        "include_glob": list(mount.include_glob),
        "exclude_glob": list(mount.exclude_glob),
        "max_bytes": mount.max_bytes,
        "follow_symlinks": mount.follow_symlinks,
    }

def identity_provider_output(result: dict[str, object], context: ToolContext) -> ToolResult[dict[str, object]]:
    """Pass through provider output while preserving structure."""

    return ToolResult(
        message=str(result.get("output_text") or result.get("message") or ""),
        value=result,
        success=True,
    )

code_interpreter_tool = OpenAICodeInterpreterTool(
    name="openai_python",
    description="Executes model-authored Python in a sandboxed container.",
    container={
        "type": "auto",
        "memory_limit": "4g",
        # File IDs come from prior uploads
        "file_ids": ["file_csv_upload"],
        # Host projections use the same HostMount definition as the VFS tools
        "host_mounts": [_to_openai_mount(host_mount)],
    },
    handle_provider_output=identity_provider_output,
)

# Register the tool in your prompt/tool registry before issuing the request.
container_config = code_interpreter_tool.container

response = client.responses.create(
    model="gpt-4.1",
    tools=[{"type": "code_interpreter", "container": container_config}],
    tool_choice="required",
    input=(
        "Load /workspace/sunfish/data/input.csv, calculate the mean of the `value` "
        "column, and generate a histogram PNG."
    ),
)

print(response.output_text)

# Download generated files referenced in container_file_citation annotations
for content in response.output or []:
    for item in content.get("annotations", []) or []:
        if item["type"] == "container_file_citation":
            downloaded = client.container_files.content(
                container_id=item["container_id"],
                file_id=item["file_id"],
            )
            Path(item["filename"]).write_bytes(downloaded)
```

## Testing and observability

- Add unit coverage for `ProviderTool` registration, provider-name validation,
  and `handle_provider_output` dispatch ordering. Mock OpenAI responses to
  produce code interpreter tool calls and ensure handlers receive `ToolContext`.
- Extend integration tests (gated by `OPENAI_API_KEY`) to issue real
  `code_interpreter` calls that execute simple Python (e.g., math operations or
  plotting). Validate container reuse, file citation parsing, and
  `ToolResult` mapping.
- Maintain existing telemetry fields (request IDs, model name, token limits) and
  log container IDs plus memory tiers for debugging. Do not introduce provider
  side effects outside the standard event bus.
