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

"""Command line entry points for the ``wink`` executable."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import socket
from collections.abc import (
    Awaitable,
    Callable,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import mcp.types as mcp_types
import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.shared.exceptions import McpError
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

from ..prompt.overrides import (
    LocalPromptOverridesStore,
    OverrideFileMetadata,
    PromptOverridesError,
    iter_override_files,
)
from ..runtime.logging import (
    StructuredLogger,
    configure_logging,
    get_logger,
)
from .config import load_config
from .wink_overrides import (
    WinkOverridesError,
    apply_section_override,
    apply_tool_override,
    fetch_section_override,
    fetch_tool_override,
    remove_section_override,
    remove_tool_override,
)


class _ListOverridesParams(BaseModel):
    """Input payload for ``wink.list_overrides``."""

    model_config = ConfigDict(extra="forbid")

    ns: str | None = Field(
        default=None,
        description="Filter overrides by namespace (exact match).",
    )


class _ListOverridesEntry(BaseModel):
    """Structured entry describing a persisted override file."""

    model_config = ConfigDict(extra="forbid")

    ns: str
    prompt: str
    tag: str
    section_count: int
    tool_count: int
    backing_file_path: str
    relative_path: str
    updated_at: datetime | None
    content_hash: str


class _ListOverridesResponse(BaseModel):
    """Structured response for ``wink.list_overrides``."""

    model_config = ConfigDict(extra="forbid")

    overrides: list[_ListOverridesEntry]


class _SectionOverrideParams(BaseModel):
    """Shared identifier fields for section override operations."""

    model_config = ConfigDict(extra="forbid")

    ns: str
    prompt: str
    tag: str = Field(
        default="latest",
        description="Override tag to inspect (defaults to 'latest').",
    )
    section_path: str = Field(
        description="Slash-delimited section path (for example: 'intro/body').",
    )


class _WriteSectionOverrideParams(_SectionOverrideParams):
    """Input payload for ``wink.write_section_override``."""

    body: str = Field(description="Markdown content to persist for the section.")
    expected_hash: str | None = Field(
        default=None,
        description="Descriptor hash guard for the target section.",
    )
    descriptor_version: int | None = Field(
        default=None,
        description="Descriptor version guard (optional).",
    )
    confirm: bool = Field(
        description="Explicit confirmation flag required for writes.",
    )


class _DeleteSectionOverrideParams(_SectionOverrideParams):
    """Input payload for ``wink.delete_section_override``."""

    descriptor_version: int | None = Field(
        default=None,
        description="Descriptor version guard (optional).",
    )


class _ToolOverrideParams(BaseModel):
    """Shared identifier fields for tool override operations."""

    model_config = ConfigDict(extra="forbid")

    ns: str
    prompt: str
    tag: str = Field(
        default="latest",
        description="Override tag to inspect (defaults to 'latest').",
    )
    tool_name: str = Field(description="Tool name to inspect or mutate.")


class _WriteToolOverrideParams(_ToolOverrideParams):
    """Input payload for ``wink.write_tool_override``."""

    description: str | None = Field(
        default=None,
        description="Override description for the tool (optional).",
    )
    param_descriptions: Mapping[str, str] | None = Field(
        default=None,
        description="Parameter description overrides (mapping of name to text).",
    )
    expected_contract_hash: str | None = Field(
        default=None,
        description="Descriptor hash guard for the tool contract.",
    )
    descriptor_version: int | None = Field(
        default=None,
        description="Descriptor version guard (optional).",
    )
    confirm: bool = Field(
        description="Explicit confirmation flag required for writes.",
    )


class _DeleteToolOverrideParams(_ToolOverrideParams):
    """Input payload for ``wink.delete_tool_override``."""

    descriptor_version: int | None = Field(
        default=None,
        description="Descriptor version guard (optional).",
    )


class _SectionOverrideSnapshotPayload(BaseModel):
    """Structured response payload for section lookups."""

    model_config = ConfigDict(extra="forbid")

    ns: str
    prompt: str
    tag: str
    section_path: list[str]
    expected_hash: str
    override_body: str | None
    default_body: str | None
    descriptor_version: int
    backing_file_path: str


class _SectionOverrideMutationPayload(BaseModel):
    """Structured response payload for section mutations."""

    model_config = ConfigDict(extra="forbid")

    ns: str
    prompt: str
    tag: str
    section_path: list[str]
    expected_hash: str
    override_body: str | None
    descriptor_version: int
    backing_file_path: str
    updated_at: datetime | None
    warnings: list[str]


class _ToolOverrideSnapshotPayload(BaseModel):
    """Structured response payload for tool override lookups."""

    model_config = ConfigDict(extra="forbid")

    ns: str
    prompt: str
    tag: str
    tool_name: str
    expected_contract_hash: str
    override_description: str | None
    override_param_descriptions: Mapping[str, str]
    default_description: str
    default_param_descriptions: Mapping[str, str]
    description: str
    param_descriptions: Mapping[str, str]
    descriptor_version: int
    backing_file_path: str


class _ToolOverrideMutationPayload(BaseModel):
    """Structured response payload for tool override mutations."""

    model_config = ConfigDict(extra="forbid")

    ns: str
    prompt: str
    tag: str
    tool_name: str
    expected_contract_hash: str
    override_description: str | None
    override_param_descriptions: Mapping[str, str]
    description: str
    param_descriptions: Mapping[str, str]
    descriptor_version: int
    backing_file_path: str
    updated_at: datetime | None
    warnings: list[str]


@dataclass(frozen=True)
class _ToolExecutionResult:
    """Normalised tool execution outcome."""

    message: str
    payload: Mapping[str, Any] | BaseModel


_ToolHandler = Callable[[BaseModel], Awaitable[_ToolExecutionResult]]


@dataclass(frozen=True)
class _ToolSpec:
    """Tool definition paired with its execution handler."""

    definition: mcp_types.Tool
    args_model: type[BaseModel]
    handler: _ToolHandler


_SSE_PATH = "/sse"
_MESSAGES_PATH = "/messages"


def _model_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return the JSON schema for ``model``."""

    return model.model_json_schema(by_alias=True)


def _normalise_payload(payload: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-serialisable mapping for ``payload``."""

    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json", by_alias=True)
    return dict(payload)


def _build_call_result(result: _ToolExecutionResult) -> mcp_types.CallToolResult:
    """Convert an execution result into an MCP ``CallToolResult``."""

    structured = _normalise_payload(result.payload)
    message = result.message or "Operation completed successfully."
    text_block = mcp_types.TextContent(type="text", text=message)
    content: list[mcp_types.ContentBlock] = [text_block]
    return mcp_types.CallToolResult(
        content=content,
        structuredContent=structured,
        isError=False,
    )


def _mcp_error(
    *, code: int, message: str, data: Mapping[str, Any] | None = None
) -> McpError:
    """Build an MCP error payload."""

    return McpError(
        mcp_types.ErrorData(code=code, message=message, data=dict(data or {}))
    )


def _override_identity(path: Path) -> tuple[str, str, str]:
    """Return the (ns, prompt, tag) tuple encoded in ``path``."""

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:  # pragma: no cover - surfaced in practice
        msg = f"Failed to read override file: {path}"
        raise PromptOverridesError(msg) from error

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:  # pragma: no cover - surfaced in practice
        msg = f"Failed to parse prompt override JSON: {path}"
        raise PromptOverridesError(msg) from error

    if not isinstance(payload, MutableMapping):
        msg = f"Prompt override payload must be a mapping: {path}"
        raise PromptOverridesError(msg)

    metadata = cast(MutableMapping[str, object], payload)

    ns = metadata.get("ns")
    prompt_key = metadata.get("prompt_key")
    tag = metadata.get("tag")
    if (
        not isinstance(ns, str)
        or not isinstance(prompt_key, str)
        or not isinstance(tag, str)
    ):
        msg = f"Override file missing metadata fields: {path}"
        raise PromptOverridesError(msg)
    return ns, prompt_key, tag


def _build_override_entry(
    *,
    metadata: OverrideFileMetadata,
    ns: str,
    prompt: str,
    tag: str,
) -> _ListOverridesEntry:
    """Create a list entry for ``metadata``."""

    updated_at = datetime.fromtimestamp(metadata.modified_time, tz=UTC)
    relative_path = "/".join(metadata.relative_segments)
    return _ListOverridesEntry(
        ns=ns,
        prompt=prompt,
        tag=tag,
        section_count=metadata.section_count,
        tool_count=metadata.tool_count,
        backing_file_path=str(metadata.path),
        relative_path=relative_path,
        updated_at=updated_at,
        content_hash=metadata.content_hash,
    )


def _format_host_for_display(host: str) -> str:
    """Return a host string suitable for human-readable output."""

    if host in {"0.0.0.0", "::"}:  # nosec B104 - normalising bind-all display
        return "127.0.0.1"
    return host


def _format_base_url(host: str, port: int, *, sse_path: str) -> tuple[str, str, str]:
    """Return base URLs for SSE and message endpoints."""

    display_host = _format_host_for_display(host)
    if ":" in display_host and not display_host.startswith("["):
        authority = f"[{display_host}]:{port}"
    else:
        authority = f"{display_host}:{port}"
    base = f"http://{authority}"
    return base, f"{base}{sse_path}", f"{base}/messages"


def _connection_instructions(host: str, port: int, *, sse_path: str) -> str:
    """Build human-friendly connection instructions."""

    base, sse_endpoint, post_endpoint = _format_base_url(host, port, sse_path=sse_path)
    lines = [
        f"wink MCP server ready at {sse_endpoint}",
        f"Claude Desktop: add a custom server with base URL {base}",
        f"Codex CLI: npx @modelcontextprotocol/cli connect {base}",
        f"POST endpoint for messages: {post_endpoint}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the wink CLI."""

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    configure_logging(level=args.log_level, json_mode=args.json_logs)
    logger = get_logger(__name__)

    handler = getattr(args, "handler", None)
    if handler is None:  # pragma: no cover - defensive guard
        parser.print_help()
        return 2

    return handler(args=args, logger=logger)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wink",
        description="Command line interface for the wink MCP server.",
    )

    _ = parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the wink configuration file.",
    )
    _ = parser.add_argument(
        "--overrides-dir",
        type=Path,
        default=None,
        help="Directory containing override configuration fragments.",
    )
    _ = parser.add_argument(
        "--log-level",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        default=None,
        help="Override the log level emitted by the CLI.",
    )
    _ = parser.add_argument(
        "--json-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit structured JSON logs (disable with --no-json-logs).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Start the wink Model Context Protocol server.",
    )
    mcp_parser.set_defaults(handler=_handle_mcp_command)

    return parser


def _handle_mcp_command(*, args: argparse.Namespace, logger: StructuredLogger) -> int:
    logger.info(
        "Starting wink MCP server.",
        event="wink.mcp.start",
    )
    run_mcp_server(
        config=args.config,
        overrides_dir=args.overrides_dir,
    )
    return 0


def run_mcp_server(
    *, config: Path | None, overrides_dir: Path | None
) -> None:  # pragma: no cover - integration tested separately
    """Run the wink MCP server using ``config`` and ``overrides_dir``."""

    cli_overrides: dict[str, object] = {}
    if overrides_dir is not None:
        cli_overrides["overrides_dir"] = overrides_dir

    server_config = load_config(config, cli_overrides=cli_overrides)

    logger = get_logger(
        __name__,
        context={
            "component": "wink.mcp",
            "workspace_root": str(server_config.workspace_root),
            "overrides_dir": str(server_config.overrides_dir),
        },
    )

    store = LocalPromptOverridesStore(
        root_path=server_config.workspace_root,
        overrides_relative_path=server_config.overrides_dir,
    )

    async def _list_overrides(params: _ListOverridesParams) -> _ToolExecutionResult:
        def _impl() -> _ToolExecutionResult:
            entries: list[_ListOverridesEntry] = []
            for metadata in iter_override_files(
                root_path=server_config.workspace_root,
                overrides_relative_path=server_config.overrides_dir,
            ):
                ns, prompt_key, tag = _override_identity(metadata.path)
                if params.ns and ns != params.ns:
                    continue
                entries.append(
                    _build_override_entry(
                        metadata=metadata,
                        ns=ns,
                        prompt=prompt_key,
                        tag=tag,
                    )
                )

            payload = _ListOverridesResponse(overrides=entries)
            count = len(entries)
            message = f"Found {count} override{'s' if count != 1 else ''}."
            return _ToolExecutionResult(message=message, payload=payload)

        return await asyncio.to_thread(_impl)

    async def _get_section(params: _SectionOverrideParams) -> _ToolExecutionResult:
        def _impl() -> _ToolExecutionResult:
            snapshot = fetch_section_override(
                config=server_config,
                store=store,
                ns=params.ns,
                prompt=params.prompt,
                tag=params.tag,
                section_path=params.section_path,
            )
            payload = _SectionOverrideSnapshotPayload(
                ns=snapshot.ns,
                prompt=snapshot.prompt,
                tag=snapshot.tag,
                section_path=list(snapshot.section_path),
                expected_hash=snapshot.expected_hash,
                override_body=snapshot.override_body,
                default_body=snapshot.default_body,
                descriptor_version=snapshot.descriptor_version,
                backing_file_path=str(snapshot.backing_file_path),
            )
            message = f"Loaded section override for {snapshot.ns}/{snapshot.prompt}:{snapshot.tag}."
            return _ToolExecutionResult(message=message, payload=payload)

        return await asyncio.to_thread(_impl)

    async def _write_section(
        params: _WriteSectionOverrideParams,
    ) -> _ToolExecutionResult:
        def _impl() -> _ToolExecutionResult:
            mutation = apply_section_override(
                config=server_config,
                store=store,
                ns=params.ns,
                prompt=params.prompt,
                tag=params.tag,
                section_path=params.section_path,
                body=params.body,
                expected_hash=params.expected_hash,
                descriptor_version=params.descriptor_version,
                confirm=params.confirm,
            )
            payload = _SectionOverrideMutationPayload(
                ns=mutation.ns,
                prompt=mutation.prompt,
                tag=mutation.tag,
                section_path=list(mutation.section_path),
                expected_hash=mutation.expected_hash,
                override_body=mutation.override_body,
                descriptor_version=mutation.descriptor_version,
                backing_file_path=str(mutation.backing_file_path),
                updated_at=mutation.updated_at,
                warnings=list(mutation.warnings),
            )
            message = f"Persisted section override for {mutation.ns}/{mutation.prompt}:{mutation.tag}."
            return _ToolExecutionResult(message=message, payload=payload)

        return await asyncio.to_thread(_impl)

    async def _delete_section(
        params: _DeleteSectionOverrideParams,
    ) -> _ToolExecutionResult:
        def _impl() -> _ToolExecutionResult:
            mutation = remove_section_override(
                config=server_config,
                store=store,
                ns=params.ns,
                prompt=params.prompt,
                tag=params.tag,
                section_path=params.section_path,
                descriptor_version=params.descriptor_version,
            )
            payload = _SectionOverrideMutationPayload(
                ns=mutation.ns,
                prompt=mutation.prompt,
                tag=mutation.tag,
                section_path=list(mutation.section_path),
                expected_hash=mutation.expected_hash,
                override_body=mutation.override_body,
                descriptor_version=mutation.descriptor_version,
                backing_file_path=str(mutation.backing_file_path),
                updated_at=mutation.updated_at,
                warnings=list(mutation.warnings),
            )
            message = f"Removed section override for {mutation.ns}/{mutation.prompt}:{mutation.tag}."
            return _ToolExecutionResult(message=message, payload=payload)

        return await asyncio.to_thread(_impl)

    async def _get_tool(params: _ToolOverrideParams) -> _ToolExecutionResult:
        def _impl() -> _ToolExecutionResult:
            snapshot = fetch_tool_override(
                config=server_config,
                store=store,
                ns=params.ns,
                prompt=params.prompt,
                tag=params.tag,
                tool_name=params.tool_name,
            )
            payload = _ToolOverrideSnapshotPayload(
                ns=snapshot.ns,
                prompt=snapshot.prompt,
                tag=snapshot.tag,
                tool_name=snapshot.tool_name,
                expected_contract_hash=snapshot.expected_contract_hash,
                override_description=snapshot.override_description,
                override_param_descriptions=dict(snapshot.override_param_descriptions),
                default_description=snapshot.default_description,
                default_param_descriptions=dict(snapshot.default_param_descriptions),
                description=snapshot.description,
                param_descriptions=dict(snapshot.param_descriptions),
                descriptor_version=snapshot.descriptor_version,
                backing_file_path=str(snapshot.backing_file_path),
            )
            message = f"Loaded tool override for {snapshot.ns}/{snapshot.prompt}:{snapshot.tag}."
            return _ToolExecutionResult(message=message, payload=payload)

        return await asyncio.to_thread(_impl)

    async def _write_tool(
        params: _WriteToolOverrideParams,
    ) -> _ToolExecutionResult:
        def _impl() -> _ToolExecutionResult:
            mutation = apply_tool_override(
                config=server_config,
                store=store,
                ns=params.ns,
                prompt=params.prompt,
                tag=params.tag,
                tool_name=params.tool_name,
                description=params.description,
                param_descriptions=dict(params.param_descriptions or {}),
                expected_contract_hash=params.expected_contract_hash,
                descriptor_version=params.descriptor_version,
                confirm=params.confirm,
            )
            payload = _ToolOverrideMutationPayload(
                ns=mutation.ns,
                prompt=mutation.prompt,
                tag=mutation.tag,
                tool_name=mutation.tool_name,
                expected_contract_hash=mutation.expected_contract_hash,
                override_description=mutation.override_description,
                override_param_descriptions=dict(mutation.override_param_descriptions),
                description=mutation.description,
                param_descriptions=dict(mutation.param_descriptions),
                descriptor_version=mutation.descriptor_version,
                backing_file_path=str(mutation.backing_file_path),
                updated_at=mutation.updated_at,
                warnings=list(mutation.warnings),
            )
            message = f"Persisted tool override for {mutation.ns}/{mutation.prompt}:{mutation.tag}."
            return _ToolExecutionResult(message=message, payload=payload)

        return await asyncio.to_thread(_impl)

    async def _delete_tool(
        params: _DeleteToolOverrideParams,
    ) -> _ToolExecutionResult:
        def _impl() -> _ToolExecutionResult:
            mutation = remove_tool_override(
                config=server_config,
                store=store,
                ns=params.ns,
                prompt=params.prompt,
                tag=params.tag,
                tool_name=params.tool_name,
                descriptor_version=params.descriptor_version,
            )
            payload = _ToolOverrideMutationPayload(
                ns=mutation.ns,
                prompt=mutation.prompt,
                tag=mutation.tag,
                tool_name=mutation.tool_name,
                expected_contract_hash=mutation.expected_contract_hash,
                override_description=mutation.override_description,
                override_param_descriptions=dict(mutation.override_param_descriptions),
                description=mutation.description,
                param_descriptions=dict(mutation.param_descriptions),
                descriptor_version=mutation.descriptor_version,
                backing_file_path=str(mutation.backing_file_path),
                updated_at=mutation.updated_at,
                warnings=list(mutation.warnings),
            )
            message = f"Removed tool override for {mutation.ns}/{mutation.prompt}:{mutation.tag}."
            return _ToolExecutionResult(message=message, payload=payload)

        return await asyncio.to_thread(_impl)

    tool_specs: dict[str, _ToolSpec] = {
        "wink.list_overrides": _ToolSpec(
            definition=mcp_types.Tool(
                name="wink.list_overrides",
                description="List persisted prompt override files.",
                inputSchema=_model_schema(_ListOverridesParams),
                outputSchema=_model_schema(_ListOverridesResponse),
            ),
            args_model=_ListOverridesParams,
            handler=cast(_ToolHandler, _list_overrides),
        ),
        "wink.get_section_override": _ToolSpec(
            definition=mcp_types.Tool(
                name="wink.get_section_override",
                description="Load a rendered section override.",
                inputSchema=_model_schema(_SectionOverrideParams),
                outputSchema=_model_schema(_SectionOverrideSnapshotPayload),
            ),
            args_model=_SectionOverrideParams,
            handler=cast(_ToolHandler, _get_section),
        ),
        "wink.write_section_override": _ToolSpec(
            definition=mcp_types.Tool(
                name="wink.write_section_override",
                description="Persist an updated section override.",
                inputSchema=_model_schema(_WriteSectionOverrideParams),
                outputSchema=_model_schema(_SectionOverrideMutationPayload),
            ),
            args_model=_WriteSectionOverrideParams,
            handler=cast(_ToolHandler, _write_section),
        ),
        "wink.delete_section_override": _ToolSpec(
            definition=mcp_types.Tool(
                name="wink.delete_section_override",
                description="Remove a persisted section override.",
                inputSchema=_model_schema(_DeleteSectionOverrideParams),
                outputSchema=_model_schema(_SectionOverrideMutationPayload),
            ),
            args_model=_DeleteSectionOverrideParams,
            handler=cast(_ToolHandler, _delete_section),
        ),
        "wink.get_tool_override": _ToolSpec(
            definition=mcp_types.Tool(
                name="wink.get_tool_override",
                description="Load metadata for a tool override.",
                inputSchema=_model_schema(_ToolOverrideParams),
                outputSchema=_model_schema(_ToolOverrideSnapshotPayload),
            ),
            args_model=_ToolOverrideParams,
            handler=cast(_ToolHandler, _get_tool),
        ),
        "wink.write_tool_override": _ToolSpec(
            definition=mcp_types.Tool(
                name="wink.write_tool_override",
                description="Persist metadata for a tool override.",
                inputSchema=_model_schema(_WriteToolOverrideParams),
                outputSchema=_model_schema(_ToolOverrideMutationPayload),
            ),
            args_model=_WriteToolOverrideParams,
            handler=cast(_ToolHandler, _write_tool),
        ),
        "wink.delete_tool_override": _ToolSpec(
            definition=mcp_types.Tool(
                name="wink.delete_tool_override",
                description="Remove a persisted tool override.",
                inputSchema=_model_schema(_DeleteToolOverrideParams),
                outputSchema=_model_schema(_ToolOverrideMutationPayload),
            ),
            args_model=_DeleteToolOverrideParams,
            handler=cast(_ToolHandler, _delete_tool),
        ),
    }

    server = Server(
        name="wink",
        instructions="Inspect and manage weakincentives prompt overrides.",
    )

    @server.list_tools()
    async def _list_tools(_: mcp_types.ListToolsRequest) -> mcp_types.ListToolsResult:
        def _impl() -> mcp_types.ListToolsResult:
            return mcp_types.ListToolsResult(
                tools=[spec.definition for spec in tool_specs.values()]
            )

        return await asyncio.to_thread(_impl)

    @server.call_tool()
    async def _call_tool(
        tool_name: str, arguments: Mapping[str, Any] | None
    ) -> mcp_types.CallToolResult:
        spec = tool_specs.get(tool_name)
        if spec is None:
            raise _mcp_error(
                code=mcp_types.INVALID_PARAMS,
                message=f"Unknown tool: {tool_name}",
                data={"tool": tool_name},
            )

        raw_arguments = dict(arguments or {})
        try:
            parsed = spec.args_model.model_validate(raw_arguments)
        except ValidationError as error:
            raise _mcp_error(
                code=mcp_types.INVALID_PARAMS,
                message=f"Invalid parameters for {tool_name}.",
                data={"tool": tool_name, "errors": error.errors()},
            ) from error

        log_context: dict[str, Any] = {"tool": tool_name}
        try:
            request_context = server.request_context
        except LookupError:
            request_context = None
        if request_context is not None:
            log_context["request_id"] = request_context.request_id

        logger.info(
            "Handling MCP tool request.",
            event="wink.mcp.tool.request",
            context=log_context,
        )

        try:
            result = await spec.handler(parsed)
        except WinkOverridesError as error:
            raise _mcp_error(
                code=mcp_types.INVALID_PARAMS,
                message=str(error),
                data={"tool": tool_name, "error": type(error).__name__},
            ) from error
        except PromptOverridesError as error:
            raise _mcp_error(
                code=mcp_types.INTERNAL_ERROR,
                message=str(error),
                data={"tool": tool_name, "error": type(error).__name__},
            ) from error
        except OSError as error:
            raise _mcp_error(
                code=mcp_types.INTERNAL_ERROR,
                message=f"Filesystem error: {error.strerror or error}",
                data={"tool": tool_name, "errno": getattr(error, "errno", None)},
            ) from error
        except ValueError as error:
            raise _mcp_error(
                code=mcp_types.INVALID_PARAMS,
                message=str(error),
                data={"tool": tool_name, "error": type(error).__name__},
            ) from error
        except McpError:
            raise
        except Exception as error:  # pragma: no cover - defensive guard
            raise _mcp_error(
                code=mcp_types.INTERNAL_ERROR,
                message="Unexpected server failure.",
                data={"tool": tool_name, "error": type(error).__name__},
            ) from error

        response = _build_call_result(result)
        logger.info(
            "Completed MCP tool request.",
            event="wink.mcp.tool.response",
            context={
                **log_context,
                "has_structured": response.structuredContent is not None,
            },
        )
        return response

    _ = (_list_tools, _call_tool)

    async def _serve() -> None:
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        uvicorn_server: uvicorn.Server | None = None

        def _signal_handler(signum: int) -> None:
            if not stop_event.is_set():
                logger.info(
                    "Signal received. Shutting down wink MCP server.",
                    event="wink.mcp.signal",
                    context={"signal": signal.Signals(signum).name},
                )
                stop_event.set()
                if uvicorn_server is not None:
                    uvicorn_server.should_exit = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                _ = loop.add_signal_handler(sig, _signal_handler, sig)
            except NotImplementedError:  # pragma: no cover - platform fallback
                _ = signal.signal(
                    sig, lambda _sig, _frame, sig=sig: _signal_handler(sig)
                )

        transport = SseServerTransport(_MESSAGES_PATH)

        async def handle_sse(scope: Scope, receive: Receive, send: Send) -> None:
            logger.info(
                "Accepted wink MCP client connection.",
                event="wink.mcp.session.start",
            )
            try:
                async with transport.connect_sse(scope, receive, send) as streams:
                    await server.run(
                        streams[0],
                        streams[1],
                        server.create_initialization_options(),
                    )
            finally:
                logger.info(
                    "Client disconnected from wink MCP server.",
                    event="wink.mcp.session.stop",
                )
                if not stop_event.is_set():
                    stop_event.set()
                if uvicorn_server is not None:
                    uvicorn_server.should_exit = True

        async def sse_endpoint(request: Request) -> Response:
            send_callable: Send = request._send  # pyright: ignore[reportPrivateUsage]
            await handle_sse(request.scope, request.receive, send_callable)
            return Response()

        app = Starlette(
            debug=False,
            routes=[
                Route(_SSE_PATH, endpoint=sse_endpoint, methods=["GET"]),
                Mount(_MESSAGES_PATH, app=transport.handle_post_message),
            ],
        )

        config = uvicorn.Config(
            app,
            host=server_config.listen_host,
            port=server_config.listen_port,
            log_level="warning",
            access_log=False,
            loop="asyncio",
            lifespan="off",
        )
        uvicorn_server = uvicorn.Server(config)

        server_task = asyncio.create_task(uvicorn_server.serve())
        started_event = getattr(uvicorn_server, "started", None)
        if isinstance(started_event, asyncio.Event):
            _ = await started_event.wait()

        actual_host = server_config.listen_host
        actual_port = server_config.listen_port
        if uvicorn_server.servers:
            raw_sockets = uvicorn_server.servers[0].sockets or []
            sockets = cast(Sequence[socket.socket], raw_sockets)
            if sockets:
                sockname_obj = sockets[0].getsockname()
                if isinstance(sockname_obj, (tuple, list)):
                    sockname_seq = cast(Sequence[object], sockname_obj)
                    if len(sockname_seq) >= 2:
                        host_part = sockname_seq[0]
                        port_part = sockname_seq[1]
                        actual_host = str(host_part)
                        if isinstance(port_part, int):
                            actual_port = port_part
                        elif isinstance(port_part, str) and port_part.isdigit():
                            actual_port = int(port_part)

        instructions = _connection_instructions(
            actual_host,
            actual_port,
            sse_path=_SSE_PATH,
        )
        print(instructions, flush=True)
        logger.info(
            "wink MCP server ready.",
            event="wink.mcp.ready",
            context={
                "listen_host": actual_host,
                "listen_port": actual_port,
                "sse_path": _SSE_PATH,
                "messages_path": _MESSAGES_PATH,
            },
        )

        stop_waiter = asyncio.create_task(stop_event.wait())
        done, _pending = await asyncio.wait(
            {server_task, stop_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stop_waiter in done:
            uvicorn_server.should_exit = True
            await server_task
        else:
            _ = stop_waiter.cancel()
            with suppress(asyncio.CancelledError):
                await stop_waiter

        logger.info(
            "wink MCP server stopped.",
            event="wink.mcp.stop",
        )

    asyncio.run(_serve())
