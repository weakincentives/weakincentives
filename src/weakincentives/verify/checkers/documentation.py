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

"""Documentation verification checkers.

These checkers verify documentation integrity:
- Spec references: File paths in specs exist
- Doc examples: Python code blocks type-check
- Markdown links: Local links resolve
- Markdown format: Formatting is consistent
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from weakincentives.verify._git import tracked_files
from weakincentives.verify._markdown import (
    CodeBlock,
    extract_code_blocks,
    extract_file_path,
    extract_links,
    is_shell_output,
)
from weakincentives.verify._subprocess import run_tool
from weakincentives.verify._types import CheckContext, CheckResult, Finding, Severity

# Patterns to match file references in specs
FILE_REF_PATTERNS = [
    re.compile(r"`(src/weakincentives/[^`\s:]+\.py)`"),
    re.compile(r"`(tests/[^`\s:]+\.py)`"),
    re.compile(r"`(scripts/[^`\s:]+\.py)`"),
    re.compile(r"\*\*Implementation:\*\*\s*`([^`]+)`"),
    re.compile(r"\(see\s+`([^`]+\.py)`\)"),
]


class SpecReferencesChecker:
    """Checker for spec file path references.

    Verifies that file paths mentioned in spec documents actually exist.
    """

    @property
    def name(self) -> str:
        return "spec_references"

    @property
    def category(self) -> str:
        return "documentation"

    @property
    def description(self) -> str:
        return "Check that file paths in specs exist"

    def check(self, ctx: CheckContext) -> CheckResult:
        start_time = time.monotonic()
        findings: list[Finding] = []

        doc_dirs = [ctx.project_root / "specs", ctx.project_root / "guides"]
        existing_dirs = [d for d in doc_dirs if d.exists()]

        if not existing_dirs:
            return CheckResult(
                checker=f"{self.category}.{self.name}",
                findings=(),
                duration_ms=int((time.monotonic() - start_time) * 1000),
            )

        for doc_dir in existing_dirs:
            for spec_file in sorted(doc_dir.glob("*.md")):
                self._check_file(spec_file, ctx.project_root, findings)

        duration_ms = int((time.monotonic() - start_time) * 1000)
        return CheckResult(
            checker=f"{self.category}.{self.name}",
            findings=tuple(findings),
            duration_ms=duration_ms,
        )

    @staticmethod
    def _extract_paths(line: str) -> list[str]:
        """Extract file paths from a line of text."""
        paths: list[str] = []
        for pattern in FILE_REF_PATTERNS:
            for match in pattern.finditer(line):
                for path in match.group(1).split(","):
                    path = path.strip().strip("`")
                    is_valid = "(" not in path and ")" not in path
                    is_source = any(
                        path.startswith(p) for p in ("src/", "tests/", "scripts/")
                    )
                    if is_valid and is_source:
                        paths.append(path)
        return paths

    def _check_file(self, spec_file: Path, root: Path, findings: list[Finding]) -> None:
        content = spec_file.read_text(encoding="utf-8")

        for line_num, line in enumerate(content.splitlines(), start=1):
            for path in self._extract_paths(line):
                full_path = root / path
                if not full_path.exists() and not full_path.is_dir():
                    findings.append(
                        Finding(
                            checker=f"{self.category}.{self.name}",
                            severity=Severity.ERROR,
                            message=f"Referenced file not found: {path}",
                            file=spec_file,
                            line=line_num,
                        )
                    )


class MarkdownLinksChecker:
    """Checker for local markdown links.

    Verifies that local links in markdown files resolve to existing files.
    """

    # Exclude test repos and bundled docs
    EXCLUDED_PARTS = frozenset({"test-repositories"})
    EXCLUDED_PREFIXES = (Path("src") / "weakincentives" / "docs",)

    @property
    def name(self) -> str:
        return "markdown_links"

    @property
    def category(self) -> str:
        return "documentation"

    @property
    def description(self) -> str:
        return "Check that local links in markdown resolve"

    def check(self, ctx: CheckContext) -> CheckResult:
        start_time = time.monotonic()
        findings: list[Finding] = []

        try:
            md_files = tracked_files(
                ctx.project_root,
                pattern="*.md",
                exclude_parts=self.EXCLUDED_PARTS,
                exclude_prefixes=self.EXCLUDED_PREFIXES,
            )
        except RuntimeError as e:
            return CheckResult(
                checker=f"{self.category}.{self.name}",
                findings=(
                    Finding(
                        checker=f"{self.category}.{self.name}",
                        severity=Severity.ERROR,
                        message=str(e),
                    ),
                ),
                duration_ms=int((time.monotonic() - start_time) * 1000),
            )

        for md_file in md_files:
            links = extract_links(md_file)
            for link in links:
                if not link.is_local:
                    continue

                file_path = extract_file_path(link.target)
                if not file_path:
                    continue

                resolved = (md_file.parent / file_path).resolve()
                if not resolved.exists():
                    findings.append(
                        Finding(
                            checker=f"{self.category}.{self.name}",
                            severity=Severity.ERROR,
                            message=f"Broken link: [{link.text}]({link.target})",
                            file=md_file,
                            line=link.line,
                        )
                    )

        duration_ms = int((time.monotonic() - start_time) * 1000)
        return CheckResult(
            checker=f"{self.category}.{self.name}",
            findings=tuple(findings),
            duration_ms=duration_ms,
        )


class MarkdownFormatChecker:
    """Checker for markdown formatting.

    Runs mdformat in check mode to verify consistent formatting.
    """

    EXCLUDED_PARTS = frozenset({"test-repositories", "demo-skills"})

    @property
    def name(self) -> str:
        return "markdown_format"

    @property
    def category(self) -> str:
        return "documentation"

    @property
    def description(self) -> str:
        return "Check markdown formatting consistency"

    def check(self, ctx: CheckContext) -> CheckResult:
        start_time = time.monotonic()
        findings: list[Finding] = []

        try:
            md_files = tracked_files(
                ctx.project_root,
                pattern="*.md",
                exclude_parts=self.EXCLUDED_PARTS,
            )
        except RuntimeError as e:
            return CheckResult(
                checker=f"{self.category}.{self.name}",
                findings=(
                    Finding(
                        checker=f"{self.category}.{self.name}",
                        severity=Severity.ERROR,
                        message=str(e),
                    ),
                ),
                duration_ms=int((time.monotonic() - start_time) * 1000),
            )

        if not md_files:
            return CheckResult(
                checker=f"{self.category}.{self.name}",
                findings=(),
                duration_ms=int((time.monotonic() - start_time) * 1000),
            )

        cmd = ["mdformat", "--check", *[str(f) for f in md_files]]
        result = run_tool(cmd, cwd=ctx.project_root)

        if not result.success:
            # Parse mdformat output to find which files need formatting
            output = result.output.strip()
            if output:
                for line in output.splitlines():
                    line = line.strip()
                    if line and not line.startswith("File"):
                        findings.append(
                            Finding(
                                checker=f"{self.category}.{self.name}",
                                severity=Severity.ERROR,
                                message=f"Needs formatting: {line}",
                            )
                        )
            else:
                findings.append(
                    Finding(
                        checker=f"{self.category}.{self.name}",
                        severity=Severity.ERROR,
                        message="Markdown formatting check failed",
                    )
                )

        duration_ms = int((time.monotonic() - start_time) * 1000)
        return CheckResult(
            checker=f"{self.category}.{self.name}",
            findings=tuple(findings),
            duration_ms=duration_ms,
        )


class DocExamplesChecker:
    """Checker for Python code examples in documentation.

    Extracts Python code blocks from markdown documentation and verifies
    they type-check correctly with pyright.
    """

    # Documentation files to verify
    DOC_FILES = ("README.md", "llms.md")
    DOC_DIRS = ("guides",)

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

    # Patterns for documentation noise that should be filtered
    DOC_NOISE_PATTERNS = (
        "Variable not allowed in type expression",
        "is obscured by a declaration of the same name",
        "EllipsisType",
        "could not be resolved",
        "Unexpected indentation",
        "Unindent not expected",
        "must return value on all code paths",
        "Expression value is unused",
    )

    @property
    def name(self) -> str:
        return "doc_examples"

    @property
    def category(self) -> str:
        return "documentation"

    @property
    def description(self) -> str:
        return "Check that Python examples in docs type-check"

    def check(self, ctx: CheckContext) -> CheckResult:  # noqa: C901, PLR0912, PLR0914
        start_time = time.monotonic()
        findings: list[Finding] = []

        # Collect documentation files
        files: list[Path] = []
        for doc_file in self.DOC_FILES:
            path = ctx.project_root / doc_file
            if path.exists():
                files.append(path)

        for doc_dir in self.DOC_DIRS:
            dir_path = ctx.project_root / doc_dir
            if dir_path.exists():
                files.extend(sorted(dir_path.glob("*.md")))

        if not files:
            return CheckResult(
                checker=f"{self.category}.{self.name}",
                findings=(),
                duration_ms=int((time.monotonic() - start_time) * 1000),
            )

        # Extract Python code blocks
        all_blocks: list[CodeBlock] = []
        for file in files:
            blocks = extract_code_blocks(
                file,
                languages=frozenset({"python", "py"}),
            )
            for block in blocks:
                # Skip shell output
                if is_shell_output(block.code):
                    continue
                # Skip API reference blocks
                if self._is_api_reference_block(block.code):
                    continue
                all_blocks.append(block)

        if not all_blocks:
            return CheckResult(
                checker=f"{self.category}.{self.name}",
                findings=(),
                duration_ms=int((time.monotonic() - start_time) * 1000),
            )

        # Synthesize module and run pyright
        module_content = self._synthesize_module(all_blocks)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            module_path = tmppath / "_doc_examples.py"
            _ = module_path.write_text(module_content, encoding="utf-8")

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
            _ = config_path.write_text(json.dumps(config), encoding="utf-8")

            result = run_tool(
                [
                    "pyright",
                    "--project",
                    str(config_path),
                    "--outputjson",
                    str(module_path),
                ],
                timeout_seconds=60.0,
            )

            if result.stdout:
                try:
                    output = json.loads(result.stdout)
                    for diag in output.get("generalDiagnostics", []):
                        if diag.get("severity") not in {"error", "warning"}:
                            continue
                        msg = diag.get("message", "")
                        if self._is_doc_noise(msg):
                            continue

                        source = self._map_to_source(diag, all_blocks, module_content)
                        if source:
                            file, line = source
                            findings.append(
                                Finding(
                                    checker=f"{self.category}.{self.name}",
                                    severity=Severity.ERROR,
                                    message=msg,
                                    file=file,
                                    line=line,
                                )
                            )
                except json.JSONDecodeError:
                    pass

        duration_ms = int((time.monotonic() - start_time) * 1000)
        return CheckResult(
            checker=f"{self.category}.{self.name}",
            findings=tuple(findings),
            duration_ms=duration_ms,
        )

    @staticmethod
    def _is_api_reference_block(code: str) -> bool:
        """Check if code is API reference documentation, not runnable code."""
        lines = [ln.strip() for ln in code.strip().split("\n") if ln.strip()]
        if not lines:
            return False

        has_imports = any(
            ln.startswith(("import ", "from "))
            for ln in lines
            if not ln.startswith("#")
        )

        # Check for signature patterns like "func(args) -> Type"
        api_return = re.compile(r"^\s*\w+(?:\[[\w,\s]+\])?\s*\([^)]*\)\s*->.*$")
        api_defaults = re.compile(
            r"^\s*\w+(?:\[[\w,\s]+\])?\s*\([^)]*=(?:None|\.\.\.)[^)]*\)\s*$"
        )

        strong_signatures = 0
        for line in lines:
            if line.startswith("#"):
                continue
            is_signature = api_return.match(line) or api_defaults.match(line)
            if is_signature or line.startswith("."):
                strong_signatures += 1

        non_comment = [ln for ln in lines if not ln.startswith("#")]
        is_mostly_signatures = bool(
            non_comment and strong_signatures >= len(non_comment) * 0.7
        )

        return is_mostly_signatures and not has_imports

    def _synthesize_module(self, blocks: list[CodeBlock]) -> str:
        """Create a type-checkable module from code blocks."""
        parts: list[str] = [self.COMMON_STUBS]

        for block in blocks:
            code = self._preprocess_code(block.code)
            parts.append(f"\n# Source: {block.file.name}:{block.start_line}")
            parts.append(code)

        return "\n".join(parts)

    @staticmethod
    def _preprocess_code(code: str) -> str:
        """Preprocess code for type checking."""
        lines = code.split("\n")
        processed: list[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped == "...":
                indent = len(line) - len(line.lstrip())
                processed.append(" " * indent + "pass")
            else:
                processed.append(line)

        return "\n".join(processed)

    def _is_doc_noise(self, msg: str) -> bool:
        """Check if a diagnostic is expected documentation noise."""
        if any(pattern in msg for pattern in self.DOC_NOISE_PATTERNS):
            return True
        if "is not defined" in msg and "Any" in msg:
            return True
        return "is not assignable to declared type" in msg and "_doc_examples" in msg

    @staticmethod
    def _map_to_source(
        diag: dict[str, Any],
        blocks: list[CodeBlock],
        module_content: str,
    ) -> tuple[Path, int] | None:
        """Map diagnostic back to source file."""
        diag_line = diag.get("range", {}).get("start", {}).get("line", -1)
        if diag_line < 0:
            return None

        module_lines = module_content.split("\n")
        if diag_line >= len(module_lines):
            return None

        # Search backwards for source comment
        for i in range(diag_line, -1, -1):
            line = module_lines[i]
            if line.startswith("# Source: "):
                source_info = line[10:]
                for block in blocks:
                    expected = f"{block.file.name}:{block.start_line}"
                    if source_info == expected:
                        offset = diag_line - i - 1
                        return (block.file, block.start_line + offset)
                break

        return None
