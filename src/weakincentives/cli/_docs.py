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

"""Documentation read, list, search, and TOC handlers for ``wink docs``."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Iterator
from importlib.resources import files

from ..dataclasses import FrozenDataclass
from .docs_metadata import GUIDE_DESCRIPTIONS, SPEC_DESCRIPTIONS


def _read_doc(name: str) -> str:
    """Read a documentation file from the package."""
    doc_files = files("weakincentives.docs")
    return doc_files.joinpath(name).read_text(encoding="utf-8")


def _read_example() -> str:
    """Return a minimal example showing core WINK concepts."""
    return """\
# Minimal WINK Example

This example demonstrates core WINK concepts: prompt composition,
session management, and adapter integration.

## Features

- **Prompt composition**: Structured prompts with sections
- **Session management**: Typed state slices with reducers
- **Provider adapters**: Support for Claude Agent SDK
- **Workspace digest**: Cached codebase summaries

## Source Code

```python


from weakincentives import Prompt
from weakincentives.prompt import MarkdownSection, PromptTemplate
from weakincentives.runtime import Session
from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter


@FrozenDataclass()
class ReviewResponse:
    \"\"\"Structured output from the review agent.\"\"\"
    summary: str
    next_steps: list[str]


@FrozenDataclass()
class ReviewParams:
    \"\"\"Parameters for a review request.\"\"\"
    request: str


# Define the prompt template
template = PromptTemplate[ReviewResponse](
    ns="examples",
    key="review-agent",
    name="review_agent",
    sections=(
        MarkdownSection(
            title="Guide",
            key="guide",
            template="You are a helpful code review assistant.",
        ),
        MarkdownSection(
            title="Request",
            key="request",
            template="${request}",
        ),
    ),
)

# Create session and prompt
session = Session()
prompt = Prompt(template).bind(ReviewParams(request="Review the main function."))

# Evaluate with Claude Agent SDK
adapter = ClaudeAgentSDKAdapter()
response = adapter.evaluate(prompt, session=session)

# Access typed output
if response.output:
    print(f"Summary: {response.output.summary}")
```

## Running

```bash
ANTHROPIC_API_KEY=... python example.py
```
"""


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


@FrozenDataclass()
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


def handle_docs(args: argparse.Namespace) -> int:
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
