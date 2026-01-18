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

"""Markdown parsing utilities for the verification toolbox."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# Pattern to match fenced code blocks
# Captures: indent, language, meta (e.g., "nocheck"), and code content
FENCE_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)```(?P<lang>\w+)?(?P<meta>[^\n]*)\n"
    r"(?P<code>.*?)"
    r"\n(?P=indent)```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)

# Pattern to match markdown links: [text](target)
LINK_PATTERN = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

# Pattern for inline code spans (to exclude from link checking)
INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")


@dataclass(frozen=True, slots=True)
class CodeBlock:
    """A fenced code block from a markdown file.

    Attributes:
        file: The file containing the code block.
        start_line: Line number where the code starts (1-indexed).
        end_line: Line number where the code ends (1-indexed).
        language: The language identifier (e.g., "python").
        code: The code content.
        meta: Additional metadata after the language (e.g., "nocheck").
    """

    file: Path
    start_line: int
    end_line: int
    language: str
    code: str
    meta: str


@dataclass(frozen=True, slots=True)
class Link:
    """A markdown link.

    Attributes:
        file: The file containing the link.
        line: Line number of the link (1-indexed).
        text: The link text.
        target: The link target (URL or path).
        is_local: Whether this is a local file link (not a URL).
    """

    file: Path
    line: int
    text: str
    target: str
    is_local: bool


def extract_code_blocks(
    file: Path,
    *,
    languages: frozenset[str] | None = None,
    skip_markers: frozenset[str] = frozenset({"nocheck", "skip", "output", "result", "shell", "cli", "console"}),
) -> tuple[CodeBlock, ...]:
    """Extract fenced code blocks from a markdown file.

    Args:
        file: The markdown file to parse.
        languages: If set, only extract blocks with these languages.
                   Defaults to None (extract all).
        skip_markers: Meta markers that indicate a block should be skipped.

    Returns:
        A tuple of CodeBlock objects.
    """
    content = file.read_text(encoding="utf-8")
    blocks: list[CodeBlock] = []

    for match in FENCE_PATTERN.finditer(content):
        lang = match.group("lang") or ""
        meta = match.group("meta").strip().lower()
        code = match.group("code")

        # Filter by language if specified
        if languages is not None and lang.lower() not in languages:
            continue

        # Skip blocks with skip markers
        if any(marker in meta for marker in skip_markers):
            continue

        # Calculate line numbers
        start_pos = match.start()
        start_line = content[:start_pos].count("\n") + 2  # +2 for fence line
        end_line = start_line + code.count("\n")

        blocks.append(
            CodeBlock(
                file=file,
                start_line=start_line,
                end_line=end_line,
                language=lang,
                code=code,
                meta=meta,
            )
        )

    return tuple(blocks)


def extract_links(file: Path) -> tuple[Link, ...]:
    """Extract all links from a markdown file.

    Ignores links inside fenced code blocks and inline code spans.

    Args:
        file: The markdown file to parse.

    Returns:
        A tuple of Link objects.
    """
    content = file.read_text(encoding="utf-8")
    lines = content.splitlines()
    links: list[Link] = []

    in_fence = False
    fence_marker: str | None = None

    for line_num, line in enumerate(lines, start=1):
        # Track fenced code blocks
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif fence_marker and stripped.startswith(fence_marker * 3):
                in_fence = False
                fence_marker = None
            continue

        if in_fence:
            continue

        # Strip inline code spans before checking for links
        line_without_code = INLINE_CODE_PATTERN.sub("", line)

        # Find all links on this line
        for match in LINK_PATTERN.finditer(line_without_code):
            link_text = match.group(1)
            target = match.group(2)

            is_local = _is_local_link(target)

            links.append(
                Link(
                    file=file,
                    line=line_num,
                    text=link_text,
                    target=target,
                    is_local=is_local,
                )
            )

    return tuple(links)


def _is_local_link(target: str) -> bool:
    """Check if a link target is a local file reference.

    Args:
        target: The link target to check.

    Returns:
        True if this is a local file link, False if it's a URL or anchor.
    """
    # Skip URLs
    if target.startswith(("http://", "https://", "mailto:", "ftp://")):
        return False
    # Skip pure anchors
    if target.startswith("#"):
        return False
    return True


def extract_file_path(target: str) -> str:
    """Extract the file path from a link target, stripping anchors.

    Args:
        target: The link target (may include anchor).

    Returns:
        The file path portion of the target.
    """
    if "#" in target:
        return target.split("#")[0]
    return target


def is_python_code_block(block: CodeBlock) -> bool:
    """Check if a code block contains Python code.

    Args:
        block: The code block to check.

    Returns:
        True if this is a Python code block.
    """
    return block.language.lower() in {"python", "py"}


def is_shell_output(code: str) -> bool:
    """Check if code appears to be shell output rather than code.

    Args:
        code: The code content to check.

    Returns:
        True if this looks like shell output.
    """
    first_line = code.strip().split("\n")[0] if code.strip() else ""
    return first_line.startswith(("$", ">", ">>>", "..."))
