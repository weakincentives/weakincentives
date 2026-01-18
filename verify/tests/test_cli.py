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

"""Tests for verify CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from cli import main


class TestVerifyCLI:
    """Tests for verify CLI main function."""

    def test_list_checkers(self, capsys: pytest.CaptureFixture[str]) -> None:
        """List available checkers."""
        result = main(["--list"])

        assert result == 0
        captured = capsys.readouterr()
        assert "architecture" in captured.out
        assert "documentation" in captured.out
        assert "layer_violations" in captured.out

    def test_invalid_project_root(self, tmp_path: Path) -> None:
        """Error with invalid project root."""
        # Create an empty directory with no pyproject.toml
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = main(["--root", str(empty_dir)])

        assert result == 2  # Error exit code

    def test_no_checkers_selected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Error when no checkers match filter."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")

        result = main(["--root", str(tmp_path), "nonexistent_checker"])

        assert result == 2
        captured = capsys.readouterr()
        assert "No checkers selected" in captured.err
