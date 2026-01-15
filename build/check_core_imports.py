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

"""Check that core modules do not import from contrib.

This enforces the architectural constraint that core library code
(weakincentives.*) should never depend on contrib code (weakincentives.contrib.*).
Contrib is "batteries included" that builds on core, not the other way around.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _is_contrib_import(node: ast.Import | ast.ImportFrom) -> tuple[bool, str]:
    """Check if an import statement imports from contrib."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            if "weakincentives.contrib" in alias.name or alias.name.startswith(
                "weakincentives.contrib"
            ):
                return True, alias.name
    elif isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if "contrib" in module:
            return True, module
    return False, ""


def _check_file(filepath: Path) -> list[str]:
    """Check a single file for contrib imports. Returns list of violations."""
    violations: list[str] = []

    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return violations

    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            is_contrib, import_name = _is_contrib_import(node)
            if is_contrib:
                line = getattr(node, "lineno", "?")
                violations.append(f"{filepath}:{line}: imports {import_name}")

    return violations


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    src_path = project_root / "src" / "weakincentives"
    contrib_path = src_path / "contrib"
    docs_path = src_path / "docs"

    violations: list[str] = []

    for filepath in src_path.rglob("*.py"):
        # Skip contrib directory - those are allowed to import from contrib
        if contrib_path in filepath.parents or filepath.parent == contrib_path:
            continue

        # Skip docs directory - contains bundled example files, not core library code
        if docs_path in filepath.parents or filepath.parent == docs_path:
            continue

        violations.extend(_check_file(filepath))

    if violations:
        print("Core → Contrib import violations detected:")
        print()
        print("Core modules (weakincentives.*) must not import from contrib")
        print("(weakincentives.contrib.*). Contrib builds on core, not vice versa.")
        print()
        for violation in sorted(violations):
            print(f"  {violation}")
        print()
        print(f"Found {len(violations)} violation(s)")
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - invoked via Makefile target
    raise SystemExit(main())
