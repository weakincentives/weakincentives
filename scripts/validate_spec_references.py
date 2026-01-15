#!/usr/bin/env python3
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

"""Validate file path references in specification documents.

Scans all .md files in specs/ for file path references and verifies they exist.
Helps catch reference drift during refactors.

Usage:
    python scripts/validate_spec_references.py
    python scripts/validate_spec_references.py --fix  # Suggest corrections
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Patterns to match file references in specs
FILE_REF_PATTERNS = [
    re.compile(r"`(src/weakincentives/[^`\s:]+\.py)`"),
    re.compile(r"`(tests/[^`\s:]+\.py)`"),
    re.compile(r"`(scripts/[^`\s:]+\.py)`"),
    re.compile(r"\*\*Implementation:\*\*\s*`([^`]+)`"),
    re.compile(r"\(see\s+`([^`]+\.py)`\)"),
]

# Known exceptions - files intentionally referenced but may not exist
KNOWN_EXCEPTIONS: set[str] = set()


def find_file_references(spec_path: Path) -> list[tuple[int, str]]:
    """Extract file path references from a spec file."""
    references: list[tuple[int, str]] = []
    content = spec_path.read_text()

    for line_num, line in enumerate(content.splitlines(), start=1):
        for pattern in FILE_REF_PATTERNS:
            for match in pattern.finditer(line):
                path_str = match.group(1)
                for path in path_str.split(","):
                    path = path.strip().strip("`")
                    if "(" in path or ")" in path:
                        continue
                    if any(path.startswith(p) for p in ("src/", "tests/", "scripts/")):
                        references.append((line_num, path))

    return references


def find_similar_files(missing_path: str, root: Path) -> list[str]:
    """Find files with similar names that might be the correct reference."""
    filename = Path(missing_path).name
    stem = Path(missing_path).stem

    suggestions = [str(f.relative_to(root)) for f in root.rglob(f"*{filename}")]

    if not suggestions:
        suggestions = [str(f.relative_to(root)) for f in root.rglob(f"*{stem}*.py")]

    return suggestions[:5]


def _is_valid_reference(file_path: str, root: Path) -> bool:
    """Check if a file reference is valid."""
    full_path = root / file_path
    if full_path.is_dir() or full_path.exists():
        return True
    return file_path in KNOWN_EXCEPTIONS


def _collect_errors(
    specs_dir: Path, root: Path, fix: bool
) -> list[tuple[Path, int, str, list[str]]]:
    """Collect all invalid file references."""
    errors: list[tuple[Path, int, str, list[str]]] = []

    for spec_file in sorted(specs_dir.glob("*.md")):
        for line_num, file_path in find_file_references(spec_file):
            if not _is_valid_reference(file_path, root):
                suggestions = find_similar_files(file_path, root) if fix else []
                errors.append((spec_file, line_num, file_path, suggestions))

    return errors


def _report_errors(errors: list[tuple[Path, int, str, list[str]]]) -> None:
    """Print error report to stderr."""
    print(f"Found {len(errors)} invalid file reference(s):\n", file=sys.stderr)

    current_spec = None
    for spec_file, line_num, file_path, suggestions in errors:
        if spec_file != current_spec:
            current_spec = spec_file
            print(f"{spec_file.name}:", file=sys.stderr)

        print(f"  Line {line_num}: {file_path}", file=sys.stderr)

        if suggestions:
            print("    Suggestions:", file=sys.stderr)
            for suggestion in suggestions:
                print(f"      - {suggestion}", file=sys.stderr)


def validate_specs(root: Path, fix: bool = False) -> int:
    """Validate all spec file references. Returns 0 on success, 1 on failure."""
    specs_dir = root / "specs"
    if not specs_dir.exists():
        print(f"Error: specs directory not found at {specs_dir}", file=sys.stderr)
        return 1

    errors = _collect_errors(specs_dir, root, fix)

    if not errors:
        spec_count = len(list(specs_dir.glob("*.md")))
        print(f"✓ All file references valid in {spec_count} specs")
        return 0

    _report_errors(errors)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate file path references in specification documents"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Show suggestions for fixing invalid references",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Repository root directory",
    )
    args = parser.parse_args()

    return validate_specs(args.root, fix=args.fix)


if __name__ == "__main__":
    sys.exit(main())
