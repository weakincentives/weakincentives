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
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, cast

import anyio
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import AuthProvider, StaticTokenVerifier

from ..prompt.overrides import LocalPromptOverridesStore, PromptOverridesError
from ..runtime.logging import (
    StructuredLogger,
    configure_logging,
    get_logger,
)
from .config import ConfigError, MCPServerConfig, load_config
from .wink_overrides import (
    OverrideListEntry,
    OverridesInspectionError,
    SectionOverrideMutationResult,
    SectionOverrideSnapshot,
    ToolOverrideMutationResult,
    ToolOverrideSnapshot,
    WinkOverridesError,
    apply_section_override,
    apply_tool_override,
    fetch_section_override,
    fetch_tool_override,
    list_overrides,
    remove_section_override,
    remove_tool_override,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the wink CLI."""

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    env_map = os.environ
    configure_logging(
        level=args.log_level,
        json_mode=args.json_logs,
        env=env_map,
    )
    logger = get_logger(
        __name__,
        context={"component": "wink.cli"},
    )

    config_path = args.config
    if config_path is None:
        env_config = env_map.get("WINK_CONFIG")
        if env_config:
            config_path = Path(env_config).expanduser()

    logger.info(
        "Starting wink MCP server.",
        event="wink.mcp.start",
    )

    try:
        run_mcp_server(
            config=config_path,
            overrides_dir=args.overrides_dir,
            env=env_map,
            logger=logger,
        )
    except Exception as error:  # pragma: no cover - defensive guard
        logger.exception(
            "Failed to run wink MCP server.",
            event="wink.mcp.error",
            context={"error": repr(error)},
        )
        return 1

    return 0


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

    return parser


def run_mcp_server(
    *,
    config: Path | None,
    overrides_dir: Path | None,
    env: Mapping[str, str] | None = None,
    logger: StructuredLogger | None = None,
) -> None:
    """Run the wink MCP server using ``config`` and ``overrides_dir``."""

    env_map = dict(os.environ if env is None else env)
    cli_overrides: dict[str, object] = {}
    if overrides_dir is not None:
        cli_overrides["overrides_dir"] = overrides_dir

    resolved_logger = get_logger(
        __name__,
        logger_override=logger,
        context={"component": "wink.mcp"},
    )

    try:
        server_config = load_config(
            config,
            cli_overrides=cli_overrides,
            env=env_map,
        )
    except (ConfigError, FileNotFoundError) as error:
        resolved_logger.exception(
            "Failed to load wink MCP configuration.",
            event="wink.mcp.config_error",
            context={"error": str(error)},
        )
        raise

    store, _ = _build_store(server_config)
    auth_provider = _build_auth_provider(server_config)

    resolved_logger.info(
        "Resolved wink MCP configuration.",
        event="wink.mcp.config_resolved",
        context={
            "workspace_root": str(server_config.workspace_root),
            "overrides_dir": str(server_config.overrides_dir),
            "environment": server_config.environment,
            "prompt_registry_modules": list(server_config.prompt_registry_modules),
            "listen_host": server_config.listen_host,
            "listen_port": server_config.listen_port,
            "auth_token_count": len(server_config.auth_tokens),
        },
    )

    async def _serve() -> None:
        server = FastMCP(
            name="wink",
            auth=auth_provider,
        )

        _register_tools(
            server=server,
            config=server_config,
            store=store,
            logger=resolved_logger,
        )

        try:
            await server.run_http_async(
                show_banner=False,
                host=server_config.listen_host,
                port=server_config.listen_port,
                log_level=None,
            )
        finally:
            resolved_logger.info(
                "wink MCP runtime stopped.",
                event="wink.mcp.runtime_stop",
            )

    resolved_logger.info(
        "Starting wink MCP runtime.",
        event="wink.mcp.runtime_start",
        context={
            "listen_host": server_config.listen_host,
            "listen_port": server_config.listen_port,
        },
    )

    try:
        anyio.run(_serve)
    except Exception as error:
        resolved_logger.exception(
            "wink MCP runtime failed.",
            event="wink.mcp.runtime_failure",
            context={"error": repr(error)},
        )
        raise


def _build_store(
    config: MCPServerConfig,
) -> tuple[LocalPromptOverridesStore, Path]:
    overrides_dir = config.overrides_dir
    if overrides_dir.is_absolute():
        store = LocalPromptOverridesStore(
            root_path=overrides_dir,
            overrides_relative_path=Path(),
        )
        root = overrides_dir.resolve()
    else:
        store = LocalPromptOverridesStore(
            root_path=config.workspace_root,
            overrides_relative_path=overrides_dir,
        )
        root = (config.workspace_root / overrides_dir).resolve()
    return store, root


def _build_auth_provider(config: MCPServerConfig) -> AuthProvider | None:
    if not config.auth_tokens:
        return None

    tokens: dict[str, dict[str, Any]] = {}
    for client_id, token in config.auth_tokens.items():
        tokens[token] = {
            "client_id": client_id,
            "scopes": [],
        }
    return StaticTokenVerifier(tokens)


def _register_tools(
    *,
    server: FastMCP,
    config: MCPServerConfig,
    store: LocalPromptOverridesStore,
    logger: StructuredLogger,
) -> None:
    async def _list_overrides(namespace: str | None = None) -> dict[str, object]:
        entries = cast(
            Iterable[OverrideListEntry],
            await _call_tool(
                partial(list_overrides, config=config, namespace=namespace),
                logger=logger,
                event="wink.mcp.list_overrides.error",
                context={"namespace": namespace} if namespace else {},
            ),
        )
        return {
            "overrides": [_serialize_override_entry(entry) for entry in entries],
        }

    async def _get_section_override(
        ns: str,
        prompt: str,
        tag: str = "latest",
        section_path: str = "",
    ) -> dict[str, object]:
        snapshot = cast(
            SectionOverrideSnapshot,
            await _call_tool(
                partial(
                    fetch_section_override,
                    config=config,
                    store=store,
                    ns=ns,
                    prompt=prompt,
                    tag=tag,
                    section_path=section_path,
                ),
                logger=logger,
                event="wink.mcp.get_section_override.error",
                context={
                    "ns": ns,
                    "prompt": prompt,
                    "tag": tag,
                    "section_path": section_path,
                },
            ),
        )
        return _serialize_section_snapshot(snapshot)

    async def _write_section_override(
        ns: str,
        prompt: str,
        tag: str = "latest",
        section_path: str = "",
        body: str | None = None,
        expected_hash: str | None = None,
        descriptor_version: int | None = None,
        confirm: bool = False,
    ) -> dict[str, object]:
        context = {
            "ns": ns,
            "prompt": prompt,
            "tag": tag,
            "section_path": section_path,
            "confirm": confirm,
        }
        if body is None:
            error = ValueError(
                "Override body must be provided to write a section override.",
            )
            _log_tool_error(
                logger,
                event="wink.mcp.write_section_override.error",
                error=error,
                context=context,
            )
            raise ToolError(str(error)) from error
        result = cast(
            SectionOverrideMutationResult,
            await _call_tool(
                partial(
                    apply_section_override,
                    config=config,
                    store=store,
                    ns=ns,
                    prompt=prompt,
                    tag=tag,
                    section_path=section_path,
                    body=body,
                    expected_hash=expected_hash,
                    descriptor_version=descriptor_version,
                    confirm=confirm,
                ),
                logger=logger,
                event="wink.mcp.write_section_override.error",
                context=context,
            ),
        )
        return _serialize_section_mutation(result)

    async def _delete_section_override(
        ns: str,
        prompt: str,
        tag: str = "latest",
        section_path: str = "",
        descriptor_version: int | None = None,
    ) -> dict[str, object]:
        result = cast(
            SectionOverrideMutationResult,
            await _call_tool(
                partial(
                    remove_section_override,
                    config=config,
                    store=store,
                    ns=ns,
                    prompt=prompt,
                    tag=tag,
                    section_path=section_path,
                    descriptor_version=descriptor_version,
                ),
                logger=logger,
                event="wink.mcp.delete_section_override.error",
                context={
                    "ns": ns,
                    "prompt": prompt,
                    "tag": tag,
                    "section_path": section_path,
                },
            ),
        )
        return _serialize_section_mutation(result)

    async def _get_tool_override(
        ns: str,
        prompt: str,
        tag: str = "latest",
        tool_name: str = "",
    ) -> dict[str, object]:
        snapshot = cast(
            ToolOverrideSnapshot,
            await _call_tool(
                partial(
                    fetch_tool_override,
                    config=config,
                    store=store,
                    ns=ns,
                    prompt=prompt,
                    tag=tag,
                    tool_name=tool_name,
                ),
                logger=logger,
                event="wink.mcp.get_tool_override.error",
                context={
                    "ns": ns,
                    "prompt": prompt,
                    "tag": tag,
                    "tool_name": tool_name,
                },
            ),
        )
        return _serialize_tool_snapshot(snapshot)

    async def _write_tool_override(
        ns: str,
        prompt: str,
        tag: str = "latest",
        tool_name: str = "",
        description: str | None = None,
        param_descriptions: Mapping[str, str] | None = None,
        expected_contract_hash: str | None = None,
        descriptor_version: int | None = None,
        confirm: bool = False,
    ) -> dict[str, object]:
        result = cast(
            ToolOverrideMutationResult,
            await _call_tool(
                partial(
                    apply_tool_override,
                    config=config,
                    store=store,
                    ns=ns,
                    prompt=prompt,
                    tag=tag,
                    tool_name=tool_name,
                    description=description,
                    param_descriptions=param_descriptions,
                    expected_contract_hash=expected_contract_hash,
                    descriptor_version=descriptor_version,
                    confirm=confirm,
                ),
                logger=logger,
                event="wink.mcp.write_tool_override.error",
                context={
                    "ns": ns,
                    "prompt": prompt,
                    "tag": tag,
                    "tool_name": tool_name,
                    "confirm": confirm,
                },
            ),
        )
        return _serialize_tool_mutation(result)

    async def _delete_tool_override(
        ns: str,
        prompt: str,
        tag: str = "latest",
        tool_name: str = "",
        descriptor_version: int | None = None,
    ) -> dict[str, object]:
        result = cast(
            ToolOverrideMutationResult,
            await _call_tool(
                partial(
                    remove_tool_override,
                    config=config,
                    store=store,
                    ns=ns,
                    prompt=prompt,
                    tag=tag,
                    tool_name=tool_name,
                    descriptor_version=descriptor_version,
                ),
                logger=logger,
                event="wink.mcp.delete_tool_override.error",
                context={
                    "ns": ns,
                    "prompt": prompt,
                    "tag": tag,
                    "tool_name": tool_name,
                },
            ),
        )
        return _serialize_tool_mutation(result)

    _ = server.tool(_list_overrides, name="wink.list_overrides")
    _ = server.tool(_get_section_override, name="wink.get_section_override")
    _ = server.tool(_write_section_override, name="wink.write_section_override")
    _ = server.tool(_delete_section_override, name="wink.delete_section_override")
    _ = server.tool(_get_tool_override, name="wink.get_tool_override")
    _ = server.tool(_write_tool_override, name="wink.write_tool_override")
    _ = server.tool(_delete_tool_override, name="wink.delete_tool_override")


async def _call_tool(
    func: Callable[[], object],
    *,
    logger: StructuredLogger,
    event: str,
    context: Mapping[str, object],
) -> object:
    try:
        return await anyio.to_thread.run_sync(func)  # type: ignore[attr-defined]
    except (WinkOverridesError, OverridesInspectionError) as error:
        _log_tool_error(logger, event=event, error=error, context=context)
        raise ToolError(str(error)) from error
    except PromptOverridesError as error:
        _log_tool_error(logger, event=event, error=error, context=context)
        raise ToolError(str(error)) from error
    except ValueError as error:
        _log_tool_error(logger, event=event, error=error, context=context)
        raise ToolError(str(error)) from error
    except Exception as error:  # pragma: no cover - defensive guard
        _log_tool_error(
            logger,
            event=event,
            error=error,
            context=context,
            fallback="Unexpected wink MCP failure.",
        )
        raise ToolError("Unexpected wink MCP failure.") from error


def _log_tool_error(
    logger: StructuredLogger,
    *,
    event: str,
    error: Exception,
    context: Mapping[str, object],
    fallback: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "error": str(error),
    }
    for key, value in context.items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    logger.error(
        fallback or "wink MCP tool failure.",
        event=event,
        context=payload,
    )


def _serialize_override_entry(entry: OverrideListEntry) -> dict[str, object]:
    return {
        "ns": entry.ns,
        "prompt": entry.prompt,
        "tag": entry.tag,
        "section_count": entry.section_count,
        "tool_count": entry.tool_count,
        "content_hash": entry.content_hash,
        "backing_file_path": str(entry.backing_file_path),
        "updated_at": entry.updated_at.isoformat() if entry.updated_at else None,
    }


def _serialize_section_snapshot(
    snapshot: SectionOverrideSnapshot,
) -> dict[str, object]:
    return {
        "ns": snapshot.ns,
        "prompt": snapshot.prompt,
        "tag": snapshot.tag,
        "section_path": list(snapshot.section_path),
        "expected_hash": snapshot.expected_hash,
        "override_body": snapshot.override_body,
        "default_body": snapshot.default_body,
        "backing_file_path": str(snapshot.backing_file_path),
        "descriptor_version": snapshot.descriptor_version,
    }


def _serialize_tool_snapshot(snapshot: ToolOverrideSnapshot) -> dict[str, object]:
    return {
        "ns": snapshot.ns,
        "prompt": snapshot.prompt,
        "tag": snapshot.tag,
        "tool_name": snapshot.tool_name,
        "expected_contract_hash": snapshot.expected_contract_hash,
        "override_description": snapshot.override_description,
        "override_param_descriptions": snapshot.override_param_descriptions,
        "default_description": snapshot.default_description,
        "default_param_descriptions": snapshot.default_param_descriptions,
        "description": snapshot.description,
        "param_descriptions": snapshot.param_descriptions,
        "backing_file_path": str(snapshot.backing_file_path),
        "descriptor_version": snapshot.descriptor_version,
    }


def _serialize_section_mutation(
    result: SectionOverrideMutationResult,
) -> dict[str, object]:
    return {
        "ns": result.ns,
        "prompt": result.prompt,
        "tag": result.tag,
        "section_path": list(result.section_path),
        "expected_hash": result.expected_hash,
        "override_body": result.override_body,
        "descriptor_version": result.descriptor_version,
        "backing_file_path": str(result.backing_file_path),
        "updated_at": result.updated_at.isoformat() if result.updated_at else None,
        "warnings": list(result.warnings),
    }


def _serialize_tool_mutation(
    result: ToolOverrideMutationResult,
) -> dict[str, object]:
    return {
        "ns": result.ns,
        "prompt": result.prompt,
        "tag": result.tag,
        "tool_name": result.tool_name,
        "expected_contract_hash": result.expected_contract_hash,
        "override_description": result.override_description,
        "override_param_descriptions": result.override_param_descriptions,
        "description": result.description,
        "param_descriptions": result.param_descriptions,
        "descriptor_version": result.descriptor_version,
        "backing_file_path": str(result.backing_file_path),
        "updated_at": result.updated_at.isoformat() if result.updated_at else None,
        "warnings": list(result.warnings),
    }
