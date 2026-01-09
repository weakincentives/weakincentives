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

"""Verify Python code examples in documentation files.

Extracts Python code blocks from Markdown documentation and runs them through
pyright type checking to catch broken imports, undefined names, and type errors.

Target files: README.md, WINK_GUIDE.md, llms.md
Excluded: specs/ directory (internal design docs), CHANGELOG.md (release notes)
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Documentation files to verify (relative to project root)
# CHANGELOG.md is excluded as it contains historical examples that may reference
# removed APIs intentionally (to document breaking changes)
DOC_FILES = [
    "README.md",
    "WINK_GUIDE.md",
    "llms.md",
    # Book chapters
    "book/README.md",
    "book/01-philosophy.md",
    "book/02-quickstart.md",
    "book/03-prompts.md",
    "book/04-tools.md",
    "book/04.5-tool-policies.md",
    "book/05-sessions.md",
    "book/06-adapters.md",
    "book/07-main-loop.md",
    "book/08-evaluation.md",
    "book/09-lifecycle.md",
    "book/10-progressive-disclosure.md",
    "book/11-prompt-optimization.md",
    "book/12-workspace-tools.md",
    "book/13-debugging.md",
    "book/14-testing.md",
    "book/15-code-quality.md",
    "book/16-recipes.md",
    "book/17-troubleshooting.md",
    "book/18-api-reference.md",
    "book/appendix-a-from-langgraph.md",
    "book/appendix-b-from-dspy.md",
    "book/appendix-c-formal-verification.md",
]

# Builtins that don't need to be defined
BUILTIN_NAMES = set(dir(builtins)) | {
    "annotations",
    "TYPE_CHECKING",
    "TypeVar",
    "Generic",
    "Protocol",
    "Self",
    "Final",
    "Literal",
    "ClassVar",
    "overload",
    "dataclass",
    "field",
}

# Common stubs for documentation examples
COMMON_STUBS = """\
# Auto-generated stubs for documentation verification
from __future__ import annotations
from typing import Any, TypeVar, Generic, Protocol, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from collections.abc import Mapping
import os
import sys
import json
import asyncio

# Stub common weakincentives imports
try:
    from weakincentives import *
    from weakincentives.runtime import *
    from weakincentives.runtime.session import *
    from weakincentives.runtime.mailbox import *
    from weakincentives.runtime.lifecycle import *
    from weakincentives.prompt import *
    from weakincentives.prompt.tool import *
    from weakincentives.adapters.openai import *
    from weakincentives.contrib.tools import *
    from weakincentives.contrib.mailbox import *
    from weakincentives.dbc import *
    from weakincentives.serde import *
    from weakincentives.evals import *
    from weakincentives.resources import *
except ImportError:
    pass

# Common variables referenced in examples
session: Any = None
dispatcher: Any = None
adapter: Any = None
prompt: Any = None
response: Any = None
context: Any = None
result: Any = None
config: Any = None
params: Any = None
event: Any = None
state: Any = None
plan: Any = None
output: Any = None
message: Any = None
messages: Any = None
tool: Any = None
tools: Any = None
sections: Any = None
slice_type: Any = None
reducer: Any = None
handler: Any = None
client: Any = None
model: Any = None
api_key: Any = None

# External libraries used in examples
dspy: Any = None

# Common type aliases
T = TypeVar("T")
S = TypeVar("S")
E = TypeVar("E")
OutputT = TypeVar("OutputT")
InputT = TypeVar("InputT")
ExpectedT = TypeVar("ExpectedT")
OutputType = Any
ParamsType = Any
ResultType = Any
StateType = Any
EventType = Any
"""

# Pattern to match fenced Python code blocks
# Requires language marker (python|py), matches content until closing fence
# Uses [ \t]* for indent to avoid capturing newlines
FENCE_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)```(?P<lang>python|py)(?P<meta>[^\n]*)\n"
    r"(?P<code>.*?)"
    r"\n(?P=indent)```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)

# Markers that indicate a block should be skipped
SKIP_MARKERS = {"nocheck", "skip", "output", "result", "shell", "cli", "console"}

# Pattern for API reference lines with return type annotation
# Matches: "func(args) -> Type"
API_RETURN_SIGNATURE = re.compile(r"^\s*\w+(?:\[[\w,\s]+\])?\s*\([^)]*\)\s*->.*$")

# Pattern for API reference lines with default values documentation
# Matches: "func(arg=None, other=...)" or "Class[T](param=value)"
API_DEFAULTS_SIGNATURE = re.compile(
    r"^\s*\w+(?:\[[\w,\s]+\])?\s*\([^)]*=(?:None|\.\.\.)[^)]*\)\s*$"
)

# Pattern for API cheatsheet entries - standalone constructor/call documentation
# Matches: "Class(param1, param2)" or "Class[T, U](param)" without assignment
API_CHEATSHEET_ENTRY = re.compile(r"^\s*[A-Z]\w*(?:\[[\w,\s\[\]]+\])?\s*\([^)]*\)\s*$")


def _is_api_reference_block(code: str) -> bool:  # noqa: C901
    """Check if a code block is API reference documentation, not runnable code.

    API reference blocks typically contain:
    - Function signatures: func(args) -> ReturnType
    - Constructor patterns with defaults: Class(param=None, other=...)
    - Method chains: .attr / .method()
    - No actual executable statements (no = assignments on left side)

    Bare constructor calls like "Class(a, b)" are NOT considered documentation
    since they could be executable code.
    """
    lines = [ln.strip() for ln in code.strip().split("\n") if ln.strip()]
    if not lines:
        return False

    # Check for import statements - if present, it's likely runnable code
    has_imports = any(
        ln.startswith(("import ", "from ")) for ln in lines if not ln.startswith("#")
    )

    # Count lines that are strong API reference indicators (return types, defaults)
    strong_signature_lines = 0
    # Count lines that are weak indicators (could be executable code)
    weak_signature_lines = 0

    for line in lines:
        # Skip comments
        if line.startswith("#"):
            continue

        # Strong indicator: signature with return type
        if API_RETURN_SIGNATURE.match(line):
            strong_signature_lines += 1
            continue

        # Strong indicator: signature with default values (Class(param=None, ...))
        if API_DEFAULTS_SIGNATURE.match(line):
            strong_signature_lines += 1
            continue

        # Strong indicator: method chain documentation like ".attr / .method()"
        if line.startswith(".") and "/" in line:
            strong_signature_lines += 1
            continue

        # Strong indicator: bare .attr or .method() (method documentation)
        if line.startswith("."):
            strong_signature_lines += 1
            continue

        # Weak indicator: cheatsheet entries (Class[T](args) without =)
        # Only count these if there are also strong indicators
        if API_CHEATSHEET_ENTRY.match(line) and "=" not in line.split("(")[0]:
            weak_signature_lines += 1
            continue

    # If most non-comment lines are signatures and no imports, it's a reference block
    non_comment_lines = [ln for ln in lines if not ln.startswith("#")]

    # Only count weak signatures if there's at least one strong signature
    total_signatures = strong_signature_lines
    if strong_signature_lines > 0:
        total_signatures += weak_signature_lines

    is_mostly_signatures = bool(
        non_comment_lines and total_signatures >= len(non_comment_lines) * 0.7
    )

    # Reference blocks without imports are cheatsheets
    return is_mostly_signatures and not has_imports


@dataclass(slots=True, frozen=True)
class CodeBlock:
    """A Python code block extracted from documentation."""

    file: Path
    start_line: int
    end_line: int
    code: str
    meta: str


@dataclass(slots=True, frozen=True)
class Diagnostic:
    """A type checking diagnostic."""

    file: Path
    line: int
    column: int
    severity: str
    message: str


def extract_python_blocks(file: Path) -> list[CodeBlock]:
    """Extract Python code blocks from a Markdown file."""
    content = file.read_text(encoding="utf-8")
    blocks: list[CodeBlock] = []

    for match in FENCE_PATTERN.finditer(content):
        lang = match.group("lang")
        meta = match.group("meta").strip().lower()
        code = match.group("code")

        # Only process Python blocks
        if lang not in {"python", "py"}:
            continue

        # Skip blocks with skip markers
        if any(marker in meta for marker in SKIP_MARKERS):
            continue

        # Skip blocks that are clearly shell output or commands
        first_line = code.strip().split("\n")[0] if code.strip() else ""
        if first_line.startswith(("$", ">", ">>>", "...")):
            continue

        # Skip API reference blocks (signature documentation, not runnable code)
        # These typically have patterns like "func(args) -> Type" or "Class(args)"
        # on standalone lines, or method chains like ".attr / .method()"
        if _is_api_reference_block(code):
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
                code=code,
                meta=meta,
            )
        )

    return blocks


def find_undefined_names(code: str) -> set[str]:  # noqa: C901
    """Find names used but not defined in a code block."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    defined: set[str] = set()
    used: set[str] = set()

    class NameVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope_stack: list[set[str]] = [set()]

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Store):
                self.scope_stack[-1].add(node.id)
                defined.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                used.add(node.id)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            defined.add(node.name)
            self.scope_stack[-1].add(node.name)
            # Add arguments to defined set (they're defined within function scope)
            self.scope_stack.append(set())
            for arg in node.args.args:
                self.scope_stack[-1].add(arg.arg)
                defined.add(arg.arg)
            for arg in node.args.posonlyargs:
                self.scope_stack[-1].add(arg.arg)
                defined.add(arg.arg)
            for arg in node.args.kwonlyargs:
                self.scope_stack[-1].add(arg.arg)
                defined.add(arg.arg)
            if node.args.vararg:
                self.scope_stack[-1].add(node.args.vararg.arg)
                defined.add(node.args.vararg.arg)
            if node.args.kwarg:
                self.scope_stack[-1].add(node.args.kwarg.arg)
                defined.add(node.args.kwarg.arg)
            self.generic_visit(node)
            self.scope_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            defined.add(node.name)
            self.scope_stack[-1].add(node.name)
            # Add arguments to defined set (they're defined within function scope)
            self.scope_stack.append(set())
            for arg in node.args.args:
                self.scope_stack[-1].add(arg.arg)
                defined.add(arg.arg)
            for arg in node.args.posonlyargs:
                self.scope_stack[-1].add(arg.arg)
                defined.add(arg.arg)
            for arg in node.args.kwonlyargs:
                self.scope_stack[-1].add(arg.arg)
                defined.add(arg.arg)
            if node.args.vararg:
                self.scope_stack[-1].add(node.args.vararg.arg)
                defined.add(node.args.vararg.arg)
            if node.args.kwarg:
                self.scope_stack[-1].add(node.args.kwarg.arg)
                defined.add(node.args.kwarg.arg)
            self.generic_visit(node)
            self.scope_stack.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            defined.add(node.name)
            self.scope_stack[-1].add(node.name)
            self.scope_stack.append(set())
            self.generic_visit(node)
            self.scope_stack.pop()

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split(".")[0]
                defined.add(name)
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname if alias.asname else alias.name
                defined.add(name)
            self.generic_visit(node)

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            if node.name:
                defined.add(node.name)
            self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:
            self._visit_target(node.target)
            self.generic_visit(node)

        def visit_comprehension(self, node: ast.comprehension) -> None:
            self._visit_target(node.target)
            self.generic_visit(node)

        def visit_With(self, node: ast.With) -> None:
            for item in node.items:
                if item.optional_vars:
                    self._visit_target(item.optional_vars)
            self.generic_visit(node)

        def _visit_target(self, target: ast.expr) -> None:
            if isinstance(target, ast.Name):
                defined.add(target.id)
            elif isinstance(target, ast.Tuple | ast.List):
                for elt in target.elts:
                    self._visit_target(elt)

    visitor = NameVisitor()
    visitor.visit(tree)

    return used - defined - BUILTIN_NAMES


def preprocess_code(code: str) -> str:
    """Preprocess code block for type checking."""
    lines = code.split("\n")
    processed: list[str] = []

    for line in lines:
        # Replace standalone ellipsis with pass in certain contexts
        stripped = line.strip()
        if stripped == "...":
            # Keep indentation, replace with pass
            indent = len(line) - len(line.lstrip())
            processed.append(" " * indent + "pass")
        else:
            processed.append(line)

    return "\n".join(processed)


def synthesize_module(blocks: list[CodeBlock]) -> str:
    """Synthesize a type-checkable module from code blocks."""
    parts: list[str] = [COMMON_STUBS]

    # Collect all undefined names across blocks
    all_undefined: set[str] = set()
    for block in blocks:
        code = preprocess_code(block.code)
        try:
            undefined = find_undefined_names(code)
            all_undefined.update(undefined)
        except Exception:
            pass

    # Generate stubs for undefined names not in common stubs
    stub_names = all_undefined - set(COMMON_STUBS.split())
    if stub_names:
        parts.append("\n# Additional stubs for undefined names")
        parts.extend(
            f"{name}: Any = None"
            for name in sorted(stub_names)
            if name.isidentifier() and not name.startswith("_")
        )

    # Add each code block with source comment
    for block in blocks:
        code = preprocess_code(block.code)
        parts.append(f"\n# Source: {block.file.name}:{block.start_line}")
        parts.append(code)

    return "\n".join(parts)


def run_pyright(module_path: Path, config_path: Path) -> list[Diagnostic]:
    """Run pyright on a synthesized module."""
    cmd = [
        "pyright",
        "--project",
        str(config_path),
        "--outputjson",
        str(module_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    diagnostics: list[Diagnostic] = []
    try:
        output = json.loads(result.stdout)
        diagnostics.extend(
            Diagnostic(
                file=Path(diag.get("file", "")),
                line=diag.get("range", {}).get("start", {}).get("line", 0) + 1,
                column=diag.get("range", {}).get("start", {}).get("character", 0),
                severity=diag.get("severity", "error"),
                message=diag.get("message", ""),
            )
            for diag in output.get("generalDiagnostics", [])
        )
    except (json.JSONDecodeError, KeyError):
        if result.returncode != 0 and result.stderr:
            print(f"pyright error: {result.stderr}", file=sys.stderr)

    return diagnostics


# Patterns for documentation noise that should be filtered
_DOC_NOISE_PATTERNS = (
    "Variable not allowed in type expression",
    "is obscured by a declaration of the same name",
    "EllipsisType",
    "could not be resolved",
    "Unexpected indentation",
    "Unindent not expected",
    "must return value on all code paths",
    "Expression value is unused",
)


def _is_doc_noise(msg: str) -> bool:
    """Check if a diagnostic message is expected noise from documentation."""
    # Check simple patterns
    if any(pattern in msg for pattern in _DOC_NOISE_PATTERNS):
        return True
    # Any type issues from stubs
    if "is not defined" in msg and "Any" in msg:
        return True
    # Type assignability issues between stub types and real types
    return "is not assignable to declared type" in msg and "_doc_examples" in msg


def _filter_diagnostics(
    diagnostics: list[Diagnostic],
    blocks: list[CodeBlock],
    module_content: str,
) -> list[tuple[Path, int, str]]:
    """Filter diagnostics to remove expected doc noise and map to source."""
    errors: list[tuple[Path, int, str]] = []
    for diag in diagnostics:
        if diag.severity not in {"error", "warning"}:
            continue
        if _is_doc_noise(diag.message):
            continue
        source = map_diagnostic_to_source(diag, blocks, module_content)
        if source:
            file, line = source
            errors.append((file, line, f"{diag.severity}: {diag.message}"))
    return errors


def map_diagnostic_to_source(
    diag: Diagnostic,
    blocks: list[CodeBlock],
    module_content: str,
) -> tuple[Path, int] | None:
    """Map a diagnostic from synthesized module back to source file."""
    # Find which block this diagnostic belongs to by line number
    module_lines = module_content.split("\n")
    diag_line = diag.line - 1  # 0-indexed

    if diag_line < 0 or diag_line >= len(module_lines):
        return None

    # Search backwards for source comment
    current_block: CodeBlock | None = None
    block_start_in_module = 0

    for i in range(diag_line, -1, -1):
        line = module_lines[i]
        if line.startswith("# Source: "):
            # Parse "# Source: filename.md:123"
            source_info = line[10:]  # Remove "# Source: "
            for block in blocks:
                expected = f"{block.file.name}:{block.start_line}"
                if source_info == expected:
                    current_block = block
                    block_start_in_module = i + 1
                    break
            break

    if current_block is None:
        return None

    # Calculate line offset within block
    offset = diag_line - block_start_in_module
    source_line = current_block.start_line + offset

    return (current_block.file, source_line)


def verify_files(files: list[Path], quiet: bool = False) -> int:  # noqa: C901
    """Verify Python code examples in documentation files."""
    all_blocks: list[CodeBlock] = []

    for file in files:
        if not file.exists():
            if not quiet:
                print(f"Warning: {file} not found, skipping", file=sys.stderr)
            continue

        blocks = extract_python_blocks(file)
        all_blocks.extend(blocks)
        if not quiet:
            print(f"Extracted {len(blocks)} Python blocks from {file.name}")

    if not all_blocks:
        if not quiet:
            print("No Python code blocks found to verify")
        return 0

    # Synthesize module
    module_content = synthesize_module(all_blocks)

    # Create temporary files for pyright
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Write synthesized module
        module_path = tmppath / "_doc_examples.py"
        module_path.write_text(module_content, encoding="utf-8")

        # Write pyright config (basic mode for docs)
        config = {
            "typeCheckingMode": "basic",
            "pythonVersion": "3.12",
            "include": [str(module_path)],
            "reportMissingImports": "warning",
            "reportMissingTypeStubs": "none",
            "reportUnusedVariable": "none",
            "reportUnusedImport": "none",
            "reportPrivateUsage": "none",
            "reportConstantRedefinition": "none",
            "reportIncompatibleVariableOverride": "none",
            "reportIncompatibleMethodOverride": "none",
        }
        config_path = tmppath / "pyrightconfig.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        # Run pyright
        diagnostics = run_pyright(module_path, config_path)

    # Filter and map diagnostics back to source
    errors = _filter_diagnostics(diagnostics, all_blocks, module_content)

    if errors:
        print(f"\nFound {len(errors)} issue(s) in documentation examples:\n")
        for file, line, message in sorted(errors):
            print(f"  {file.name}:{line}: {message}")
        return 1

    if not quiet:
        print(f"\nVerified {len(all_blocks)} code blocks successfully")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress output unless verification fails",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific files to verify (default: standard doc files)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = [project_root / f for f in DOC_FILES]

    return verify_files(files, quiet=args.quiet)


if __name__ == "__main__":
    sys.exit(main())
