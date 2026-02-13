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
import sys
from collections.abc import Sequence
from pathlib import Path

from ..debug import BundleValidationError, DebugBundle
from ..runtime.logging import StructuredLogger, configure_logging, get_logger
from . import debug_app
from ._docs import handle_docs
from .query import (
    QueryDatabase,
    QueryError,
    export_jsonl,
    format_as_json,
    format_as_table,
    open_query_database,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the wink CLI."""
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 2
        return int(code)

    if args.command == "docs":
        return handle_docs(args)

    configure_logging(level=args.log_level, json_mode=args.json_logs)
    logger = get_logger(__name__)

    if args.command == "debug":
        return _run_debug(args, logger)

    if args.command == "query":
        return _run_query(args, logger)

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
        help="Start a debug server that renders a bundle inspection UI.",
    )
    _ = debug_parser.add_argument(
        "bundle_path",
        help="Path to a debug bundle .zip file or a directory containing bundles.",
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

    _build_docs_parser(subcommands)
    _build_query_parser(subcommands)

    return parser


def _build_docs_parser(
    subcommands: argparse._SubParsersAction[argparse.ArgumentParser],  # pyright: ignore[reportPrivateUsage]
) -> None:
    """Build the docs subcommand parser."""
    docs_parser = subcommands.add_parser(
        "docs",
        help="Access bundled documentation",
        description="Search and read WINK documentation from the command line.",
    )

    docs_subcommands = docs_parser.add_subparsers(dest="docs_command")

    # list subcommand
    list_parser = docs_subcommands.add_parser(
        "list",
        help="List available documents with descriptions",
    )
    _ = list_parser.add_argument(
        "category",
        nargs="?",
        choices=["specs", "guides"],
        help="Category to list (specs or guides). Lists all if omitted.",
    )

    # search subcommand
    search_parser = docs_subcommands.add_parser(
        "search",
        help="Search documentation for a pattern",
    )
    _ = search_parser.add_argument(
        "pattern",
        help="Pattern to search for",
    )
    _ = search_parser.add_argument(
        "--specs",
        action="store_true",
        help="Search only specs",
    )
    _ = search_parser.add_argument(
        "--guides",
        action="store_true",
        help="Search only guides",
    )
    _ = search_parser.add_argument(
        "--context",
        type=int,
        default=2,
        help="Number of context lines to show (default: 2)",
    )
    _ = search_parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Maximum number of results (default: 20)",
    )
    _ = search_parser.add_argument(
        "--regex",
        action="store_true",
        help="Treat pattern as a regular expression",
    )

    # toc subcommand
    toc_parser = docs_subcommands.add_parser(
        "toc",
        help="Show table of contents for a document",
    )
    _ = toc_parser.add_argument(
        "type",
        choices=["spec", "guide"],
        help="Document type (spec or guide)",
    )
    _ = toc_parser.add_argument(
        "name",
        help="Document name (e.g., SESSIONS or quickstart)",
    )

    # read subcommand
    read_parser = docs_subcommands.add_parser(
        "read",
        help="Read a specific document",
    )
    _ = read_parser.add_argument(
        "type",
        choices=["reference", "changelog", "example", "spec", "guide"],
        help="Document type to read",
    )
    _ = read_parser.add_argument(
        "name",
        nargs="?",
        help="Document name (required for spec and guide types)",
    )


def _run_debug(args: argparse.Namespace, logger: StructuredLogger) -> int:
    bundle_path = Path(args.bundle_path)

    try:
        store = debug_app.BundleStore(
            bundle_path,
            logger=logger,
        )
    except debug_app.BundleLoadError as error:
        logger.exception(
            "Bundle validation failed",
            event="wink.debug.bundle_error",
            context={"path": str(bundle_path), "error": str(error)},
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


def _build_query_parser(
    subcommands: argparse._SubParsersAction[argparse.ArgumentParser],  # pyright: ignore[reportPrivateUsage]
) -> None:
    """Build the query subcommand parser."""
    query_parser = subcommands.add_parser(
        "query",
        help="Query debug bundles via SQL",
        description="Load debug bundle into SQLite and execute SQL queries.",
    )
    _ = query_parser.add_argument(
        "bundle_path",
        help="Path to a debug bundle .zip file.",
    )
    _ = query_parser.add_argument(
        "sql",
        nargs="?",
        help="SQL query to execute. If omitted, --schema must be provided.",
    )
    _ = query_parser.add_argument(
        "--schema",
        action="store_true",
        help="Output schema as JSON and exit.",
    )
    _ = query_parser.add_argument(
        "--table",
        action="store_true",
        help="Output as ASCII table (default: JSON).",
    )
    _ = query_parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Disable column truncation in table output.",
    )
    _ = query_parser.add_argument(
        "--export-jsonl",
        nargs="?",
        const="logs",
        choices=["logs", "session"],
        help="Export raw JSONL (logs or session) to stdout.",
    )


def _run_query(args: argparse.Namespace, logger: StructuredLogger) -> int:
    """Run the query command."""
    bundle_path = Path(args.bundle_path)

    if not bundle_path.exists():
        print(f"Error: Bundle not found: {bundle_path}", file=sys.stderr)
        return 2

    # Handle --export-jsonl without SQL layer
    export_source = getattr(args, "export_jsonl", None)
    if export_source is not None:
        return _handle_export_jsonl(bundle_path, export_source)

    return _handle_sql_query(args, bundle_path, logger)


def _handle_export_jsonl(bundle_path: Path, source: str) -> int:
    """Handle --export-jsonl flag."""
    try:
        bundle = DebugBundle.load(bundle_path)
        content = export_jsonl(bundle, source)
    except BundleValidationError as e:
        print(f"Error: Failed to load bundle: {e}", file=sys.stderr)
        return 2

    if content:
        _ = sys.stdout.write(content)
        if not content.endswith("\n"):
            _ = sys.stdout.write("\n")
        return 0
    print(f"Error: No {source} content in bundle", file=sys.stderr)
    return 1


def _handle_sql_query(
    args: argparse.Namespace, bundle_path: Path, logger: StructuredLogger
) -> int:
    """Handle SQL query execution."""
    try:
        db = open_query_database(bundle_path)
    except QueryError as e:
        logger.exception(
            "Failed to open query database",
            event="wink.query.open_error",
            context={"path": str(bundle_path), "error": str(e)},
        )
        print(f"Error: {e}", file=sys.stderr)
        return 2

    try:
        return _execute_query_command(args, db)
    except QueryError as e:
        logger.exception(
            "Query execution failed",
            event="wink.query.execution_error",
            context={"sql": args.sql, "error": str(e)},
        )
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        db.close()


def _execute_query_command(
    args: argparse.Namespace,
    db: QueryDatabase,
) -> int:
    """Execute the query command logic.

    Returns exit code.
    """
    if args.schema:
        schema = db.get_schema()
        print(schema.to_json())
        return 0

    if not args.sql:
        print("Error: SQL query required (or use --schema)", file=sys.stderr)
        return 1

    results = db.execute_query(args.sql)

    if args.table:
        no_truncate = getattr(args, "no_truncate", False)
        print(format_as_table(results, truncate=not no_truncate))
    else:
        print(format_as_json(results))

    return 0
