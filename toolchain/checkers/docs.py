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

"""Documentation verification checker.

Verifies documentation integrity:
- Python code examples type-check correctly
- Local markdown links resolve to existing files
- File paths referenced in specs exist
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..result import CheckResult, Diagnostic, Location
from ..utils import CodeBlock, extract_code_blocks, extract_links, git_tracked_files, is_shell_output

# Patterns to match file references in specs
FILE_REF_PATTERNS = [
    re.compile(r"`(src/weakincentives/[^`\s:]+\.py)`"),
    re.compile(r"`(tests/[^`\s:]+\.py)`"),
]


@dataclass
class DocsChecker:
    """Checker for documentation integrity."""

    root: Path | None = None

    @property
    def name(self) -> str:
        return "docs"

    @property
    def description(self) -> str:
        return "Check documentation (examples, links, references)"

    def run(self) -> CheckResult:
        start = time.monotonic()
        root = self.root or Path.cwd()
        diagnostics: list[Diagnostic] = []

        # Check Python examples in docs
        diagnostics.extend(self._check_examples(root))

        # Check local markdown links
        diagnostics.extend(self._check_links(root))

        # Check spec file references
        diagnostics.extend(self._check_spec_refs(root))

        return CheckResult(
            name=self.name,
            status="passed" if not diagnostics else "failed",
            duration_ms=int((time.monotonic() - start) * 1000),
            diagnostics=tuple(diagnostics),
        )

    def _check_examples(self, root: Path) -> list[Diagnostic]:
        """Check Python code examples type-check correctly."""
        diagnostics: list[Diagnostic] = []

        # Collect doc files
        files: list[Path] = []
        for name in ("README.md", "llms.md"):
            p = root / name
            if p.exists():
                files.append(p)
        for d in ("guides",):
            dp = root / d
            if dp.exists():
                files.extend(sorted(dp.glob("*.md")))

        if not files:
            return []

        # Extract Python code blocks
        blocks: list[CodeBlock] = []
        for f in files:
            for block in extract_code_blocks(f, languages=frozenset({"python", "py"})):
                if is_shell_output(block.code):
                    continue
                if self._is_api_reference(block.code):
                    continue
                blocks.append(block)

        if not blocks:
            return []

        # Synthesize module and run pyright
        module = self._synthesize_module(blocks)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mod_path = tmp / "_doc_examples.py"
            mod_path.write_text(module, encoding="utf-8")

            config = {
                "typeCheckingMode": "basic",
                "pythonVersion": "3.12",
                "include": [str(mod_path)],
                "reportMissingImports": "warning",
                "reportMissingTypeStubs": "none",
                "reportUnusedVariable": "none",
                "reportUnusedImport": "none",
                "reportPrivateUsage": "none",
            }
            cfg_path = tmp / "pyrightconfig.json"
            cfg_path.write_text(json.dumps(config), encoding="utf-8")

            result = subprocess.run(
                ["pyright", "--project", str(cfg_path), "--outputjson", str(mod_path)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            if result.stdout:
                try:
                    output = json.loads(result.stdout)
                    for diag in output.get("generalDiagnostics", []):
                        if diag.get("severity") not in ("error", "warning"):
                            continue
                        msg = diag.get("message", "")
                        if self._is_noise(msg):
                            continue
                        source = self._map_diagnostic(diag, blocks, module)
                        if source:
                            diagnostics.append(
                                Diagnostic(
                                    message=msg,
                                    location=Location(file=str(source[0]), line=source[1]),
                                )
                            )
                except json.JSONDecodeError:
                    pass

        return diagnostics

    def _check_links(self, root: Path) -> list[Diagnostic]:
        """Check local markdown links resolve."""
        diagnostics: list[Diagnostic] = []
        exclude = frozenset({"test-repositories"})

        for md_file in git_tracked_files(root, "*.md", exclude):
            # Skip bundled docs
            if "src/weakincentives/docs" in str(md_file):
                continue

            for link in extract_links(md_file):
                target = link.target.split("#")[0]  # Strip anchor
                if not target:
                    continue
                resolved = (md_file.parent / target).resolve()
                if not resolved.exists():
                    diagnostics.append(
                        Diagnostic(
                            message=f"Broken link: [{link.text}]({link.target})",
                            location=Location(file=str(md_file), line=link.line),
                        )
                    )

        return diagnostics

    def _check_spec_refs(self, root: Path) -> list[Diagnostic]:
        """Check file paths referenced in specs exist."""
        diagnostics: list[Diagnostic] = []

        for d in ("specs", "guides"):
            dp = root / d
            if not dp.exists():
                continue

            for spec in sorted(dp.glob("*.md")):
                content = spec.read_text(encoding="utf-8")
                for line_num, line in enumerate(content.splitlines(), 1):
                    for pattern in FILE_REF_PATTERNS:
                        for match in pattern.finditer(line):
                            path = match.group(1)
                            if not (root / path).exists():
                                diagnostics.append(
                                    Diagnostic(
                                        message=f"Referenced file not found: {path}",
                                        location=Location(file=str(spec), line=line_num),
                                    )
                                )

        return diagnostics

    @staticmethod
    def _is_api_reference(code: str) -> bool:
        """Check if code is API reference, not runnable."""
        lines = [ln.strip() for ln in code.strip().split("\n") if ln.strip() and not ln.strip().startswith("#")]
        if not lines:
            return False

        has_imports = any(ln.startswith(("import ", "from ")) for ln in lines)
        sig_pattern = re.compile(r"^\s*\w+(?:\[[\w,\s]+\])?\s*\([^)]*\)\s*->.*$")

        signatures = sum(1 for ln in lines if sig_pattern.match(ln) or ln.startswith("."))
        return not has_imports and signatures >= len(lines) * 0.7

    def _synthesize_module(self, blocks: list[CodeBlock]) -> str:
        """Create type-checkable module from code blocks."""
        stubs = """\
from __future__ import annotations
from typing import Any, TypeVar
from dataclasses import dataclass
try:
    from weakincentives import *
    from weakincentives.runtime import *
    from weakincentives.runtime.session import *
    from weakincentives.prompt import *
    from weakincentives.adapters.openai import *
    from weakincentives.contrib.tools import *
    from weakincentives.dbc import *
    from weakincentives.serde import *
    from weakincentives.evals import *
    from weakincentives.resources import *
except ImportError:
    pass
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
dspy: Any = None
T = TypeVar("T")
OutputType = Any
ParamsType = Any
"""
        parts = [stubs]

        for block in blocks:
            code = block.code.replace("...", "pass")
            parts.append(f"\n# Source: {block.file.name}:{block.start_line}")
            parts.append(code)

        return "\n".join(parts)

    @staticmethod
    def _is_noise(msg: str) -> bool:
        """Filter expected noise from doc examples."""
        noise = (
            "Variable not allowed in type expression",
            "is obscured by a declaration",
            "could not be resolved",
            "Unexpected indentation",
            "must return value",
            "Expression value is unused",
        )
        return any(n in msg for n in noise)

    @staticmethod
    def _map_diagnostic(diag: dict[str, Any], blocks: list[CodeBlock], module: str) -> tuple[Path, int] | None:
        """Map pyright diagnostic back to source file."""
        diag_line = diag.get("range", {}).get("start", {}).get("line", -1)
        if diag_line < 0:
            return None

        lines = module.split("\n")
        if diag_line >= len(lines):
            return None

        for i in range(diag_line, -1, -1):
            if lines[i].startswith("# Source: "):
                info = lines[i][10:]
                for block in blocks:
                    if info == f"{block.file.name}:{block.start_line}":
                        return (block.file, block.start_line + diag_line - i - 1)
                break

        return None
