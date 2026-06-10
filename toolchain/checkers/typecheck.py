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

"""Combined ty + pyright type checking.

Both type checkers always run, even when the first one fails, so a single
round-trip reports every type error. Diagnostics that ty and pyright raise
for the same file and line are merged into one entry tagged [ty+pyright].
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..checker import merge_output, parser_drift_warning
from ..parsers import parse_pyright, parse_ty, relativize_path
from ..result import CheckResult, Diagnostic


def _default_ty_command() -> list[str]:
    return ["uv", "run", "ty", "check", "--error-on-warning", "src"]


def _default_pyright_command() -> list[str]:
    return ["uv", "run", "pyright"]


def _location_key(diagnostic: Diagnostic) -> tuple[str, int | None] | None:
    """Dedupe key: (relative file, line), or None without a location."""
    if diagnostic.location is None:
        return None
    return (relativize_path(diagnostic.location.file), diagnostic.location.line)


def _tagged(diagnostic: Diagnostic, tag: str) -> Diagnostic:
    """Prefix the message with the tool tag and relativize the path."""
    location = diagnostic.location
    if location is not None and Path(location.file).is_absolute():
        location = replace(location, file=relativize_path(location.file))
    return Diagnostic(
        message=f"{tag} {diagnostic.message}",
        location=location,
        severity=diagnostic.severity,
    )


def _merge_diagnostics(
    ty_diagnostics: tuple[Diagnostic, ...],
    pyright_diagnostics: tuple[Diagnostic, ...],
) -> list[Diagnostic]:
    """Merge per-tool diagnostics, deduplicating same-line findings.

    When both tools flag the same file and line, only ty's entry survives,
    tagged [ty+pyright] so the agreement remains visible.
    """
    pyright_keys = {_location_key(d) for d in pyright_diagnostics} - {None}
    ty_keys = {_location_key(d) for d in ty_diagnostics} - {None}

    merged: list[Diagnostic] = []
    for diagnostic in ty_diagnostics:
        tag = "[ty+pyright]" if _location_key(diagnostic) in pyright_keys else "[ty]"
        merged.append(_tagged(diagnostic, tag))
    for diagnostic in pyright_diagnostics:
        if _location_key(diagnostic) in ty_keys:
            continue
        merged.append(_tagged(diagnostic, "[pyright]"))
    return merged


@dataclass
class TypecheckChecker:
    """Checker that runs ty and pyright and merges their diagnostics.

    Unlike chaining the tools with `&&`, both always run: one failing tool
    does not hide the other's errors, so all type errors are fixable in a
    single iteration.
    """

    ty_command: list[str] = field(default_factory=_default_ty_command)
    pyright_command: list[str] = field(default_factory=_default_pyright_command)
    timeout: int = 300

    @property
    def name(self) -> str:
        return "typecheck"

    @property
    def description(self) -> str:
        return "Check types with ty and pyright (both always run)"

    def run(self) -> CheckResult:
        """Run both type checkers and merge their results."""
        start = time.monotonic()
        ty_output, ty_code, ty_error = self._run_tool(self.ty_command)
        pyright_output, pyright_code, pyright_error = self._run_tool(
            self.pyright_command
        )
        duration_ms = int((time.monotonic() - start) * 1000)

        failed = ty_code != 0 or pyright_code != 0
        diagnostics: list[Diagnostic] = [
            error for error in (ty_error, pyright_error) if error is not None
        ]
        diagnostics.extend(
            _merge_diagnostics(
                parse_ty(ty_output, ty_code),
                parse_pyright(pyright_output, pyright_code),
            )
        )

        output = (
            f"$ {' '.join(self.ty_command)}\n{ty_output.strip()}\n\n"
            f"$ {' '.join(self.pyright_command)}\n{pyright_output.strip()}"
        ).strip()
        if failed and not diagnostics:
            diagnostics = [parser_drift_warning(self.name, output)]

        reproduce = f"{' '.join(self.ty_command)}; {' '.join(self.pyright_command)}"
        return CheckResult(
            name=self.name,
            status="failed" if failed else "passed",
            duration_ms=duration_ms,
            diagnostics=tuple(diagnostics),
            output=output,
            command=("bash", "-c", reproduce),
        )

    def _run_tool(self, command: list[str]) -> tuple[str, int, Diagnostic | None]:
        """Run one type checker, mapping harness errors to a diagnostic."""
        cmd_str = " ".join(command)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            msg = (
                f"Timed out after {self.timeout}s\n"
                f"Command: {cmd_str}\n"
                f"Fix: Increase timeout or investigate hanging process\n"
                f"Run manually: {cmd_str}"
            )
            return "", 1, Diagnostic(msg)
        except FileNotFoundError as e:
            msg = (
                f"Command not found: {e.filename}\n"
                f"Attempted: {cmd_str}\n"
                f"Fix: Ensure dependencies are installed\n"
                f"Run: uv sync && ./install-hooks.sh"
            )
            return "", 1, Diagnostic(msg)

        output = merge_output(result.stdout, result.stderr)
        return output, result.returncode, None
