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
from collections.abc import Sequence
from pathlib import Path

from ..runtime.logging import (
    StructuredLogger,
    configure_logging,
    get_logger,
)


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


def run_mcp_server(*, config: Path | None, overrides_dir: Path | None) -> None:
    """Run the wink MCP server using ``config`` and ``overrides_dir``."""

    raise NotImplementedError("MCP server integration pending.")  # pragma: no cover
