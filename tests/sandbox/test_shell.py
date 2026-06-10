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

"""Tests for the LocalShell facet and CommandResult."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from weakincentives.clock import FakeClock
from weakincentives.sandbox import (
    DEFAULT_SHELL_TIMEOUT_S,
    MAX_COMMAND_OUTPUT_BYTES,
    CommandResult,
    LocalShell,
    Shell,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


def _python(*code: str) -> list[str]:
    return [sys.executable, "-c", "\n".join(code)]


class TestShellProtocol:
    def test_local_shell_implements_shell(self, root: Path) -> None:
        assert isinstance(LocalShell(root), Shell)

    def test_defaults_exported(self) -> None:
        assert DEFAULT_SHELL_TIMEOUT_S == 60.0
        assert MAX_COMMAND_OUTPUT_BYTES == 1024 * 1024


class TestCommandResult:
    def test_ok_for_zero_exit(self) -> None:
        result = CommandResult(
            exit_code=0, stdout=b"", stderr=b"", truncated=False, duration_s=0.1
        )
        assert result.ok

    def test_not_ok_for_nonzero_exit(self) -> None:
        result = CommandResult(
            exit_code=1, stdout=b"", stderr=b"", truncated=False, duration_s=0.1
        )
        assert not result.ok

    def test_not_ok_when_timed_out(self) -> None:
        result = CommandResult(
            exit_code=0,
            stdout=b"",
            stderr=b"",
            truncated=False,
            duration_s=0.1,
            timed_out=True,
        )
        assert not result.ok


class TestLocalShellRun:
    def test_captures_stdout_and_exit_code(self, root: Path) -> None:
        result = LocalShell(root).run(_python("print('hello')"))
        assert result.ok
        assert result.exit_code == 0
        assert result.stdout == b"hello\n"
        assert result.stderr == b""
        assert not result.truncated
        assert not result.timed_out

    def test_captures_stderr_and_nonzero_exit(self, root: Path) -> None:
        result = LocalShell(root).run(
            _python("import sys", "sys.stderr.write('boom')", "sys.exit(3)")
        )
        assert result.exit_code == 3
        assert result.stderr == b"boom"
        assert not result.ok

    def test_no_shell_interpretation(self, root: Path) -> None:
        result = LocalShell(root).run([sys.executable, "-c", "print('$HOME *')"])
        assert result.stdout == b"$HOME *\n"

    def test_runs_at_root_by_default(self, root: Path) -> None:
        result = LocalShell(root).run(_python("import os", "print(os.getcwd())"))
        assert result.stdout.decode().strip() == str(root.resolve())

    def test_stdin_piped(self, root: Path) -> None:
        result = LocalShell(root).run(
            _python("import sys", "sys.stdout.write(sys.stdin.read())"),
            stdin=b"piped input",
        )
        assert result.stdout == b"piped input"

    def test_duration_measured_with_injected_clock(self, root: Path) -> None:
        result = LocalShell(root, clock=FakeClock()).run(_python("pass"))
        assert result.duration_s == 0.0

    def test_duration_non_negative_with_system_clock(self, root: Path) -> None:
        result = LocalShell(root).run(_python("pass"))
        assert result.duration_s >= 0.0

    def test_empty_argv_rejected(self, root: Path) -> None:
        with pytest.raises(ValueError, match="argv must not be empty"):
            LocalShell(root).run([])

    def test_root_property_resolved(self, root: Path) -> None:
        assert LocalShell(root).root == str(root.resolve())


class TestLocalShellCwd:
    def test_relative_cwd_inside_root(self, root: Path) -> None:
        (root / "sub").mkdir()
        result = LocalShell(root).run(
            _python("import os", "print(os.getcwd())"), cwd="sub"
        )
        assert result.stdout.decode().strip() == str((root / "sub").resolve())

    def test_absolute_cwd_rejected(self, root: Path) -> None:
        with pytest.raises(ValueError, match="must be relative"):
            LocalShell(root).run(_python("pass"), cwd=str(root))

    def test_escaping_cwd_rejected(self, root: Path) -> None:
        with pytest.raises(PermissionError, match="escapes the sandbox root"):
            LocalShell(root).run(_python("pass"), cwd="../outside")

    def test_missing_cwd_rejected(self, root: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            LocalShell(root).run(_python("pass"), cwd="nope")

    def test_dot_cwd_is_root(self, root: Path) -> None:
        result = LocalShell(root).run(
            _python("import os", "print(os.getcwd())"), cwd="."
        )
        assert result.stdout.decode().strip() == str(root.resolve())


class TestLocalShellEnv:
    def test_git_variables_stripped(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GIT_DIR", "/tmp/should-not-leak")
        monkeypatch.setenv("WINK_SHELL_KEEP", "kept")
        shell = LocalShell(root)
        result = shell.run(
            _python(
                "import os",
                "print('GIT_DIR' in os.environ, os.environ.get('WINK_SHELL_KEEP'))",
            )
        )
        assert result.stdout.decode().strip() == "False kept"

    def test_constructor_env_overlays_base(self, root: Path) -> None:
        shell = LocalShell(root, env={"WINK_SHELL_VAR": "ctor"})
        result = shell.run(_python("import os", "print(os.environ['WINK_SHELL_VAR'])"))
        assert result.stdout.decode().strip() == "ctor"

    def test_run_env_overlays_constructor_env(self, root: Path) -> None:
        shell = LocalShell(root, env={"WINK_SHELL_VAR": "ctor"})
        result = shell.run(
            _python("import os", "print(os.environ['WINK_SHELL_VAR'])"),
            env={"WINK_SHELL_VAR": "run"},
        )
        assert result.stdout.decode().strip() == "run"

    def test_environment_captured_at_construction(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        shell = LocalShell(root)
        monkeypatch.setenv("WINK_SHELL_LATE", "late")
        result = shell.run(
            _python("import os", "print(os.environ.get('WINK_SHELL_LATE'))")
        )
        assert result.stdout.decode().strip() == "None"


class TestLocalShellTimeout:
    def test_timeout_kills_and_reports(self, root: Path) -> None:
        result = LocalShell(root).run(
            _python("import time", "time.sleep(30)"), timeout_s=0.2
        )
        assert result.timed_out
        assert result.exit_code == 124
        assert not result.ok

    def test_timeout_preserves_partial_output(self, root: Path) -> None:
        result = LocalShell(root).run(
            _python(
                "import sys, time",
                "print('partial', flush=True)",
                "time.sleep(30)",
            ),
            timeout_s=1.0,
        )
        assert result.timed_out
        assert b"partial" in result.stdout

    def test_default_timeout_applies_when_none(self, root: Path) -> None:
        shell = LocalShell(root, default_timeout_s=0.2)
        result = shell.run(_python("import time", "time.sleep(30)"))
        assert result.timed_out

    def test_non_positive_timeout_rejected(self, root: Path) -> None:
        with pytest.raises(ValueError, match="timeout_s must be positive"):
            LocalShell(root).run(_python("pass"), timeout_s=0)

    def test_non_positive_default_timeout_rejected(self, root: Path) -> None:
        with pytest.raises(ValueError, match="default_timeout_s must be positive"):
            LocalShell(root, default_timeout_s=0)


class TestLocalShellOutputCaps:
    def test_stdout_capped_and_flagged(self, root: Path) -> None:
        shell = LocalShell(root, max_output_bytes=16)
        result = shell.run(_python("print('x' * 100)"))
        assert result.truncated
        assert len(result.stdout) == 16

    def test_stderr_capped_and_flagged(self, root: Path) -> None:
        shell = LocalShell(root, max_output_bytes=16)
        result = shell.run(_python("import sys", "sys.stderr.write('e' * 100)"))
        assert result.truncated
        assert len(result.stderr) == 16

    def test_output_under_cap_not_flagged(self, root: Path) -> None:
        shell = LocalShell(root, max_output_bytes=16)
        result = shell.run(_python("print('ok')"))
        assert not result.truncated

    def test_non_positive_cap_rejected(self, root: Path) -> None:
        with pytest.raises(ValueError, match="max_output_bytes must be positive"):
            LocalShell(root, max_output_bytes=0)


class TestLocalShellLaunchFailures:
    def test_missing_executable_exits_127(self, root: Path) -> None:
        result = LocalShell(root).run(["wink-definitely-not-a-command"])
        assert result.exit_code == 127
        assert b"command not found" in result.stderr

    def test_non_executable_file_exits_126(self, root: Path) -> None:
        script = root / "script.sh"
        script.write_text("#!/bin/sh\necho hi\n")
        script.chmod(0o644)
        result = LocalShell(root).run([str(script)])
        assert result.exit_code == 126
        assert b"permission denied" in result.stderr
