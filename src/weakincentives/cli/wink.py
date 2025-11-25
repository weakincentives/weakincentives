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

from ..runtime.logging import StructuredLogger, configure_logging, get_logger
from . import debug_app


def main(argv: Sequence[str] | None = None) -> int:
    """Run the wink CLI."""

    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:  # argparse exits with code 2 on errors
        code = exc.code if isinstance(exc.code, int) else 2
        return int(code)

    configure_logging(level=args.log_level, json_mode=args.json_logs)
    logger = get_logger(__name__)

    if args.command == "debug":
        return _run_debug(args, logger)

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wink",
        description="wink CLI entry point.",
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

    subcommands = parser.add_subparsers(dest="command", required=True)

    debug_parser = subcommands.add_parser(
        "debug",
        help="Start a debug server that renders a snapshot inspection UI.",
    )
    _ = debug_parser.add_argument(
        "snapshot_path",
        help="Path to a session snapshot JSONL file or a directory containing snapshots.",
    )
    _ = debug_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind the debug server to (default: 127.0.0.1).",
    )
    _ = debug_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the debug server to (default: 8000).",
    )
    _ = debug_parser.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the default browser to the UI (disable with --no-open-browser).",
    )

    return parser


def _run_debug(args: argparse.Namespace, logger: StructuredLogger) -> int:
    snapshot_path = Path(args.snapshot_path)

    def _bootstrap_loader(path: Path) -> tuple[debug_app.LoadedSnapshot, ...]:
        return debug_app.load_snapshot(path)

    try:
        store = debug_app.SnapshotStore(
            snapshot_path,
            loader=_bootstrap_loader,
            logger=logger,
        )
    except debug_app.SnapshotLoadError as error:
        logger.exception(
            "Snapshot validation failed",
            event="wink.debug.snapshot_error",
            context={"path": str(snapshot_path), "error": str(error)},
        )
        return 2

    app = debug_app.build_debug_app(store, logger=logger)

    try:
        return debug_app.run_debug_server(
            app,
            host=args.host,
            port=args.port,
            open_browser=args.open_browser,
            logger=logger,
        )
    except Exception as error:  # pragma: no cover - defensive guard
        logger.exception(
            "Debug server failed to start",
            event="wink.debug.server_error",
            context={"error": repr(error)},
        )
        return 3
