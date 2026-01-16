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
from importlib.resources import files
from pathlib import Path

from ..runtime.logging import StructuredLogger, configure_logging, get_logger
from . import debug_app


def _read_doc(name: str) -> str:
    """Read a documentation file from the package."""
    doc_files = files("weakincentives.docs")
    return doc_files.joinpath(name).read_text(encoding="utf-8")


def _read_example() -> str:
    """Read the code review example and format as markdown documentation.

    Converts the code_reviewer_example.py into a markdown document with
    a brief introduction explaining the example's purpose and structure.
    """
    doc_files = files("weakincentives.docs")
    source = doc_files.joinpath("code_reviewer_example.py").read_text(encoding="utf-8")

    intro = """\
# Code Review Agent Example

This example demonstrates a complete code review agent built with WINK.
It showcases several key features of the library:

- **MainLoop integration**: Background worker processing requests from a mailbox
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


def _read_spec(name: str) -> str:
    """Read a single spec file by name.

    Args:
        name: Spec filename with or without .md extension (e.g., "ADAPTERS" or "ADAPTERS.md")

    Returns:
        Spec content with header comment

    Raises:
        FileNotFoundError: If spec file does not exist
    """
    specs_dir = files("weakincentives.docs.specs")

    # Normalize name to include .md extension
    filename = name if name.endswith(".md") else f"{name}.md"

    # Check if the spec file exists
    spec_path = specs_dir.joinpath(filename)
    try:
        content = spec_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # List available specs for helpful error message
        available = sorted(
            entry.name.removesuffix(".md")
            for entry in specs_dir.iterdir()
            if entry.name.endswith(".md")
        )
        available_list = ", ".join(available)
        msg = f"Spec '{name}' not found. Available specs: {available_list}"
        raise FileNotFoundError(msg) from None

    header = f"<!-- specs/{filename} -->"
    return f"{header}\n{content}"


def _read_specs() -> str:
    """Read all spec files, concatenated with headers."""
    specs_dir = files("weakincentives.docs.specs")
    parts: list[str] = []

    # Get all .md files, sorted alphabetically by name
    spec_entries = sorted(
        (entry for entry in specs_dir.iterdir() if entry.name.endswith(".md")),
        key=lambda entry: entry.name,
    )

    for entry in spec_entries:
        header = f"<!-- specs/{entry.name} -->"
        content = entry.read_text(encoding="utf-8")
        parts.append(f"{header}\n{content}")

    return "\n\n".join(parts)


def _read_guides() -> str:
    """Read all guide files, concatenated with headers."""
    guides_dir = files("weakincentives.docs.guides")
    parts: list[str] = []

    # Get all .md files, sorted alphabetically by name
    guide_entries = sorted(
        (entry for entry in guides_dir.iterdir() if entry.name.endswith(".md")),
        key=lambda entry: entry.name,
    )

    for entry in guide_entries:
        header = f"<!-- guides/{entry.name} -->"
        content = entry.read_text(encoding="utf-8")
        parts.append(f"{header}\n{content}")

    return "\n\n".join(parts)


def _load_docs(args: argparse.Namespace) -> list[str]:
    """Load documentation content based on args."""
    parts: list[str] = []
    if args.reference:
        parts.append(_read_doc("llms.md"))
    if args.guide:
        parts.append(_read_guides())
    if args.spec:
        parts.append(_read_spec(args.spec))
    if args.specs:
        parts.append(_read_specs())
    if args.changelog:
        parts.append(_read_doc("CHANGELOG.md"))
    if args.example:
        parts.append(_read_example())
    return parts


def _handle_docs(args: argparse.Namespace) -> int:
    """Handle the docs subcommand."""
    has_flag = (
        args.reference
        or args.guide
        or args.spec
        or args.specs
        or args.changelog
        or args.example
    )
    if not has_flag:
        print(
            "Error: At least one of --reference, --guide, --spec, --specs, --changelog, or --example required"
        )
        print(
            "Usage: wink docs [--reference] [--guide] [--spec NAME] [--specs] [--changelog] [--example]"
        )
        return 1

    try:
        parts = _load_docs(args)
    except FileNotFoundError as e:
        print(f"Error: Documentation not found: {e}", file=sys.stderr)
        print("This may indicate a packaging error.", file=sys.stderr)
        return 2

    print("\n---\n".join(parts))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the wink CLI."""

    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:  # argparse exits with code 2 on errors
        code = exc.code if isinstance(exc.code, int) else 2
        return int(code)

    if args.command == "docs":
        return _handle_docs(args)

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

    docs_parser = subcommands.add_parser(
        "docs",
        help="Print bundled documentation",
        description="Access WINK documentation from the command line.",
    )
    _ = docs_parser.add_argument(
        "--reference",
        action="store_true",
        help="Print API reference (llms.md)",
    )
    _ = docs_parser.add_argument(
        "--guide",
        action="store_true",
        help="Print usage guides (guides/*.md)",
    )
    _ = docs_parser.add_argument(
        "--spec",
        metavar="NAME",
        help="Print a single spec by name (e.g., ADAPTERS or ADAPTERS.md)",
    )
    _ = docs_parser.add_argument(
        "--specs",
        action="store_true",
        help="Print all specification files",
    )
    _ = docs_parser.add_argument(
        "--changelog",
        action="store_true",
        help="Print changelog (CHANGELOG.md)",
    )
    _ = docs_parser.add_argument(
        "--example",
        action="store_true",
        help="Print code review example as markdown documentation",
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
