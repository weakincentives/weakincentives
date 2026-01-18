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

"""Shared utilities for the verification toolchain.

Provides git operations, AST analysis, and markdown parsing.
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# Git utilities
# =============================================================================


def git_tracked_files(
    root: Path,
    pattern: str = "*",
    exclude: frozenset[str] | None = None,
) -> list[Path]:
    """Get git-tracked files matching a pattern.

    Args:
        root: Repository root directory.
        pattern: Glob pattern (e.g., "*.md", "*.py").
        exclude: Directory names to exclude (e.g., {"test-repositories"}).

    Returns:
        List of absolute paths to matching tracked files.
    """
    result = subprocess.run(
        ["git", "ls-files", pattern],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return []

    exclude_set = exclude or frozenset()
    files: list[Path] = []

    for line in result.stdout.splitlines():
        if not line:  # pragma: no cover
            continue
        path = Path(line)
        if exclude_set.intersection(path.parts):
            continue
        full = root / path
        if full.exists():  # pragma: no branch
            files.append(full)

    return sorted(files)


# =============================================================================
# AST utilities
# =============================================================================


@dataclass(frozen=True, slots=True)
class ImportInfo:
    """Information about an import statement."""

    module: str  # Module containing the import
    imported_from: str  # Module being imported
    lineno: int


def extract_imports(source: str, module_name: str) -> list[ImportInfo]:
    """Extract imports from Python source code."""
    tree = ast.parse(source)
    imports: list[ImportInfo] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    ImportInfo(
                        module=module_name,
                        imported_from=alias.name.split(".")[0],
                        lineno=node.lineno,
                    )
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.level > 0:
                # Resolve relative import
                base = module_name.split(".")
                base = base[: -node.level] if node.level <= len(base) else []
                if node.module:  # pragma: no branch - always True from outer condition
                    base.append(node.module)
                imported_from = ".".join(base)
            else:
                imported_from = node.module

            imports.append(
                ImportInfo(
                    module=module_name,
                    imported_from=imported_from,
                    lineno=node.lineno,
                )
            )

    return imports


def path_to_module(path: Path, src_dir: Path) -> str:
    """Convert file path to module name."""
    try:
        rel = path.relative_to(src_dir)
    except ValueError:
        rel = path

    parts = list(rel.parts)
    if parts and parts[-1].endswith(".py"):  # pragma: no branch
        parts[-1] = parts[-1][:-3]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]

    return ".".join(parts)


def get_subpackage(module: str, root: str = "weakincentives") -> str | None:
    """Get the first subpackage under root (e.g., 'contrib' from 'weakincentives.contrib.tools')."""
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != root:
        return None
    return parts[1]


# =============================================================================
# Markdown utilities
# =============================================================================

FENCE_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)```(?P<lang>\w+)?(?P<meta>[^\n]*)\n(?P<code>.*?)\n(?P=indent)```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)

LINK_PATTERN = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

SKIP_MARKERS = frozenset({"nocheck", "skip", "output", "result", "shell", "cli", "console"})


@dataclass(frozen=True, slots=True)
class CodeBlock:
    """A fenced code block from a markdown file."""

    file: Path
    start_line: int
    language: str
    code: str
    meta: str


@dataclass(frozen=True, slots=True)
class Link:
    """A markdown link."""

    file: Path
    line: int
    text: str
    target: str


def extract_code_blocks(
    file: Path,
    languages: frozenset[str] | None = None,
) -> list[CodeBlock]:
    """Extract fenced code blocks from markdown."""
    content = file.read_text(encoding="utf-8")
    blocks: list[CodeBlock] = []

    for match in FENCE_PATTERN.finditer(content):
        lang = match.group("lang") or ""
        meta = match.group("meta").strip().lower()
        code = match.group("code")

        if languages and lang.lower() not in languages:
            continue
        if any(marker in meta for marker in SKIP_MARKERS):
            continue

        start_pos = match.start()
        start_line = content[:start_pos].count("\n") + 2

        blocks.append(CodeBlock(file=file, start_line=start_line, language=lang, code=code, meta=meta))

    return blocks


def extract_links(file: Path) -> list[Link]:
    """Extract local file links from markdown."""
    content = file.read_text(encoding="utf-8")
    lines = content.splitlines()
    links: list[Link] = []

    in_fence = False
    for line_num, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        # Remove inline code
        line_clean = re.sub(r"`[^`]+`", "", line)

        for match in LINK_PATTERN.finditer(line_clean):
            target = match.group(2)
            # Skip URLs and anchors
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            links.append(Link(file=file, line=line_num, text=match.group(1), target=target))

    return links


def is_shell_output(code: str) -> bool:
    """Check if code looks like shell output."""
    first = code.strip().split("\n")[0] if code.strip() else ""
    return first.startswith(("$", ">", ">>>", "..."))


# =============================================================================
# Bandit compatibility shim
# =============================================================================


def patch_ast_for_bandit() -> None:
    """Restore AST nodes removed in Python 3.14 for bandit compatibility."""
    constant = getattr(ast, "Constant", None)
    if constant is None:  # pragma: no cover
        return

    for name in ("Num", "Str", "Bytes", "NameConstant", "Ellipsis"):
        if not hasattr(ast, name):  # pragma: no cover
            setattr(ast, name, constant)

    def _make_prop() -> property:
        return property(lambda self: self.value)

    for attr in ("n", "s", "b"):
        if not hasattr(constant, attr):
            setattr(constant, attr, _make_prop())
