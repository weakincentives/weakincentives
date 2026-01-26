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
import re
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from ..debug.bundle import BundleValidationError, DebugBundle
from ..runtime.logging import StructuredLogger, configure_logging, get_logger
from . import debug_app
from .docs_metadata import GUIDE_DESCRIPTIONS, SPEC_DESCRIPTIONS
from .query import (
    QueryDatabase,
    QueryError,
    export_jsonl,
    format_as_json,
    format_as_table,
    open_query_database,
)


def _read_doc(name: str) -> str:
    """Read a documentation file from the package."""
    doc_files = files("weakincentives.docs")
    return doc_files.joinpath(name).read_text(encoding="utf-8")


def _read_example() -> str:
    """Read the code review example and format as markdown documentation."""
    doc_files = files("weakincentives.docs")
    source = doc_files.joinpath("code_reviewer_example.py").read_text(encoding="utf-8")

    intro = """\
# Code Review Agent Example

This example demonstrates a complete code review agent built with WINK.
It showcases several key features of the library:

- **AgentLoop integration**: Background worker processing requests from a mailbox
- **Prompt composition**: Structured prompts with multiple sections and tools
- **Session management**: Persistent state across multiple review requests
- **Provider adapters**: Support for OpenAI and Claude Agent SDK
- **Planning tools**: Plan-Act-Reflect workflow for structured investigations
- **Workspace tools**: VFS, Podman, and Claude Agent SDK sandbox modes

## Running the Example

```bash
# With OpenAI adapter (default)
OPENAI_API_KEY=... python code_reviewer_example.py

# With Podman sandbox
OPENAI_API_KEY=... python code_reviewer_example.py --podman

# With Claude Agent SDK
ANTHROPIC_API_KEY=... python code_reviewer_example.py --claude-agent

# With workspace optimization
python code_reviewer_example.py --optimize
```

## Source Code

```python
"""

    return f"{intro}{source}\n```\n"


def _normalize_doc_name(name: str, available: list[str]) -> str | None:
    """Find the actual document name using case-insensitive matching.

    Returns the correctly-cased name if found, None otherwise.
    """
    # Strip .md extension if present for comparison
    lookup = name.removesuffix(".md").casefold()
    for doc_name in available:
        if doc_name.casefold() == lookup:
            return doc_name
    return None


def _read_spec(name: str) -> str:
    """Read a single spec file by name (case-insensitive)."""
    specs_dir = files("weakincentives.docs.specs")
    available = sorted(
        entry.name.removesuffix(".md")
        for entry in specs_dir.iterdir()
        if entry.name.endswith(".md")
    )

    normalized = _normalize_doc_name(name, available)
    if normalized is None:
        available_list = ", ".join(available)
        msg = f"Spec '{name}' not found. Available specs: {available_list}"
        raise FileNotFoundError(msg)

    filename = f"{normalized}.md"
    content = specs_dir.joinpath(filename).read_text(encoding="utf-8")
    header = f"<!-- specs/{filename} -->"
    return f"{header}\n{content}"


def _read_guide(name: str) -> str:
    """Read a single guide file by name (case-insensitive)."""
    guides_dir = files("weakincentives.docs.guides")
    available = sorted(
        entry.name.removesuffix(".md")
        for entry in guides_dir.iterdir()
        if entry.name.endswith(".md")
    )

    normalized = _normalize_doc_name(name, available)
    if normalized is None:
        available_list = ", ".join(available)
        msg = f"Guide '{name}' not found. Available guides: {available_list}"
        raise FileNotFoundError(msg)

    filename = f"{normalized}.md"
    content = guides_dir.joinpath(filename).read_text(encoding="utf-8")
    header = f"<!-- guides/{filename} -->"
    return f"{header}\n{content}"


def _list_specs() -> list[str]:
    """List all available spec names."""
    specs_dir = files("weakincentives.docs.specs")
    return sorted(
        entry.name.removesuffix(".md")
        for entry in specs_dir.iterdir()
        if entry.name.endswith(".md")
    )


def _list_guides() -> list[str]:
    """List all available guide names."""
    guides_dir = files("weakincentives.docs.guides")
    return sorted(
        entry.name.removesuffix(".md")
        for entry in guides_dir.iterdir()
        if entry.name.endswith(".md")
    )


def _format_doc_list(
    names: list[str], descriptions: dict[str, str], category: str
) -> str:
    """Format a list of documents with descriptions."""
    lines = [f"{category} ({len(names)} documents)", "─" * len(category)]

    max_name_len = max(len(name) for name in names) if names else 0
    for name in names:
        desc = descriptions.get(name, "")
        lines.append(f"{name:<{max_name_len}}  {desc}")

    return "\n".join(lines)


def _handle_list(args: argparse.Namespace) -> int:
    """Handle the list subcommand."""
    category = args.category if hasattr(args, "category") else None

    if category == "specs":
        specs = _list_specs()
        print(_format_doc_list(specs, SPEC_DESCRIPTIONS, "SPECS"))
    elif category == "guides":
        guides = _list_guides()
        print(_format_doc_list(guides, GUIDE_DESCRIPTIONS, "GUIDES"))
    else:
        # List all
        specs = _list_specs()
        guides = _list_guides()
        print(_format_doc_list(specs, SPEC_DESCRIPTIONS, "SPECS"))
        print()
        print(_format_doc_list(guides, GUIDE_DESCRIPTIONS, "GUIDES"))

    return 0


def _extract_headings(content: str) -> list[str]:
    """Extract markdown headings from content."""
    return [line for line in content.splitlines() if line.startswith("#")]


def _handle_toc(args: argparse.Namespace) -> int:
    """Handle the toc subcommand."""
    doc_type = args.type
    name = args.name

    try:
        content = _read_spec(name) if doc_type == "spec" else _read_guide(name)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    # Extract path from the header comment (e.g., "<!-- specs/SESSIONS.md -->")
    first_line = content.split("\n", 1)[0]
    path = first_line.removeprefix("<!-- ").removesuffix(" -->")

    headings = _extract_headings(content)
    print(f"{path} - Table of Contents")
    print("─" * len(path))
    for heading in headings:
        print(heading)

    return 0


def _iter_all_docs() -> Iterator[tuple[str, str, str]]:
    """Iterate over all documents yielding (path, name, content)."""
    specs_dir = files("weakincentives.docs.specs")
    for entry in specs_dir.iterdir():
        if entry.name.endswith(".md"):  # pragma: no branch
            content = entry.read_text(encoding="utf-8")
            yield f"specs/{entry.name}", entry.name.removesuffix(".md"), content

    guides_dir = files("weakincentives.docs.guides")
    for entry in guides_dir.iterdir():
        if entry.name.endswith(".md"):  # pragma: no branch
            content = entry.read_text(encoding="utf-8")
            yield f"guides/{entry.name}", entry.name.removesuffix(".md"), content


def _iter_specs() -> Iterator[tuple[str, str, str]]:
    """Iterate over spec documents yielding (path, name, content)."""
    specs_dir = files("weakincentives.docs.specs")
    for entry in specs_dir.iterdir():  # pragma: no branch
        if entry.name.endswith(".md"):  # pragma: no branch
            content = entry.read_text(encoding="utf-8")
            yield f"specs/{entry.name}", entry.name.removesuffix(".md"), content


def _iter_guides() -> Iterator[tuple[str, str, str]]:
    """Iterate over guide documents yielding (path, name, content)."""
    guides_dir = files("weakincentives.docs.guides")
    for entry in guides_dir.iterdir():
        if entry.name.endswith(".md"):  # pragma: no branch
            content = entry.read_text(encoding="utf-8")
            yield f"guides/{entry.name}", entry.name.removesuffix(".md"), content


def _build_match_fn(
    pattern: str, use_regex: bool
) -> tuple[Callable[[str], bool], None] | tuple[None, str]:
    """Build a match function for the given pattern.

    Returns (match_fn, None) on success or (None, error_message) on failure.
    """
    if use_regex:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return None, f"Invalid regex pattern: {e}"
        return (lambda line: regex.search(line) is not None), None
    pattern_lower = pattern.lower()
    return (lambda line: pattern_lower in line.lower()), None


def _select_doc_iterator(
    specs_only: bool, guides_only: bool
) -> Iterator[tuple[str, str, str]]:
    """Select the appropriate document iterator based on flags."""
    if specs_only:
        return _iter_specs()
    if guides_only:
        return _iter_guides()
    return _iter_all_docs()


@dataclass(slots=True, frozen=True)
class SearchOptions:
    """Options for document search."""

    specs_only: bool = False
    guides_only: bool = False
    context_lines: int = 2
    max_results: int = 20
    use_regex: bool = False


def _search_docs(
    pattern: str,
    opts: SearchOptions,
) -> list[tuple[str, int, list[str]]]:
    """Search documents for pattern.

    Returns list of (path, line_number, context_lines) tuples.
    """
    match_fn, error = _build_match_fn(pattern, opts.use_regex)
    if error:
        raise ValueError(error)
    assert match_fn is not None  # Type narrowing: error was None  # nosec B101

    results: list[tuple[str, int, list[str]]] = []
    doc_iter = _select_doc_iterator(opts.specs_only, opts.guides_only)

    for path, _name, content in doc_iter:
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if match_fn(line):
                start = max(0, i - opts.context_lines)
                end = min(len(lines), i + opts.context_lines + 1)
                results.append((path, i + 1, lines[start:end]))
                if len(results) >= opts.max_results:
                    return results

    return results


def _handle_search(args: argparse.Namespace) -> int:
    """Handle the search subcommand."""
    opts = SearchOptions(
        specs_only=args.specs,
        guides_only=args.guides,
        context_lines=args.context,
        max_results=args.max_results,
        use_regex=args.regex,
    )

    try:
        results = _search_docs(args.pattern, opts)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not results:
        print(f'No matches found for "{args.pattern}"')
        return 0

    print(f'Found {len(results)} matches for "{args.pattern}"')
    print()

    for path, line_num, context in results:
        print(f"{path}:{line_num}")
        for ctx_line in context:
            print(f"  {ctx_line}")
        print()

    return 0


def _read_named_doc(args: argparse.Namespace, doc_type: str) -> str | None:
    """Read a named document (spec or guide), returning content or None on error."""
    if not hasattr(args, "name") or args.name is None:
        print(f"Error: {doc_type} name required", file=sys.stderr)
        return None
    return _read_spec(args.name) if doc_type == "spec" else _read_guide(args.name)


def _handle_read(args: argparse.Namespace) -> int:
    """Handle the read subcommand."""
    doc_type = args.type
    readers = {
        "reference": lambda: _read_doc("llms.md"),
        "changelog": lambda: _read_doc("CHANGELOG.md"),
        "example": _read_example,
    }

    try:
        if doc_type in readers:
            print(readers[doc_type]())
        else:  # spec or guide
            content = _read_named_doc(args, doc_type)
            if content is None:
                return 1
            print(content)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    return 0


def _handle_docs(args: argparse.Namespace) -> int:
    """Handle the docs subcommand."""
    docs_command = getattr(args, "docs_command", None)

    handlers = {
        "list": _handle_list,
        "search": _handle_search,
        "toc": _handle_toc,
        "read": _handle_read,
    }

    if docs_command in handlers:
        return handlers[docs_command](args)

    print("Usage: wink docs {list,search,toc,read} ...")
    print()
    print("Subcommands:")
    print("  list    List available documents with descriptions")
    print("  search  Search documentation for a pattern")
    print("  toc     Show table of contents for a document")
    print("  read    Read a specific document")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the wink CLI."""
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 2
        return int(code)

    if args.command == "docs":
        return _handle_docs(args)

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
