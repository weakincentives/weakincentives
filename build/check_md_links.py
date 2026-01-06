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

"""Check local links in Markdown files.

Scans all tracked Markdown files for local links and verifies the targets exist.
Fenced code blocks are ignored to avoid false positives from example snippets.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Exclude test repos and bundled docs (copies of root-level files where links don't apply)
EXCLUDED_PARTS = {"test-repositories"}
EXCLUDED_PREFIXES = (Path("src") / "weakincentives" / "docs",)

# Match markdown links: [text](target)
LINK_PATTERN = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

# Match fenced code block delimiters
FENCE_PATTERN = re.compile(r"^(`{3,}|~{3,})")

# Match inline code spans to exclude them from link checking
INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")


def _collect_markdown_files(root: Path) -> list[Path]:
    """Collect all tracked Markdown files, excluding test repositories."""
    result = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git ls-files failed: {message}")

    files: list[Path] = []
    for line in result.stdout.splitlines():
        if not line:
            continue

        path = Path(line)
        if EXCLUDED_PARTS.intersection(path.parts):
            continue
        if any(path.is_relative_to(prefix) for prefix in EXCLUDED_PREFIXES):
            continue

        candidate = root / path
        if not candidate.exists():
            continue
        files.append(candidate)

    return sorted(files)


def _is_local_link(target: str) -> bool:
    """Check if a link target is a local file reference."""
    # Skip URLs
    if target.startswith(("http://", "https://", "mailto:")):
        return False
    # Skip pure anchors
    return not target.startswith("#")


def _extract_file_path(target: str) -> str:
    """Extract the file path from a link target, stripping anchors."""
    # Remove anchor if present
    if "#" in target:
        target = target.split("#")[0]
    return target


def _check_file_links(md_file: Path, root: Path) -> list[tuple[int, str, str]]:  # noqa: C901
    """Check all local links in a Markdown file.

    Returns a list of (line_number, link_text, target) for broken links.
    """
    broken: list[tuple[int, str, str]] = []
    content = md_file.read_text(encoding="utf-8")
    lines = content.splitlines()

    in_fence = False
    fence_marker: str | None = None

    for line_num, line in enumerate(lines, start=1):
        # Track fenced code blocks
        fence_match = FENCE_PATTERN.match(line.lstrip())
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker[0]  # ` or ~
            elif fence_marker and line.lstrip().startswith(fence_marker * 3):
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

            if not _is_local_link(target):
                continue

            file_path = _extract_file_path(target)
            if not file_path:
                # Pure anchor after stripping
                continue

            # Resolve relative to the markdown file's directory
            resolved = (md_file.parent / file_path).resolve()

            if not resolved.exists():
                broken.append((line_num, link_text, target))

    return broken


def main() -> int:
    """Check all local links in tracked Markdown files."""
    project_root = Path(__file__).resolve().parent.parent
    markdown_files = _collect_markdown_files(project_root)

    if not markdown_files:
        return 0

    all_broken: list[tuple[Path, int, str, str]] = []

    for md_file in markdown_files:
        broken = _check_file_links(md_file, project_root)
        for line_num, link_text, target in broken:
            all_broken.append((md_file, line_num, link_text, target))

    if all_broken:
        print("Broken local links found:", file=sys.stderr)
        for md_file, line_num, link_text, target in all_broken:
            rel_path = md_file.relative_to(project_root)
            print(f"  {rel_path}:{line_num}: [{link_text}]({target})", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
