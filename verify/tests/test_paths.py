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

"""Tests for path utilities."""

from __future__ import annotations

from pathlib import Path

import pytest
from paths import (
    find_project_root,
    get_guides_dir,
    get_package_dir,
    get_specs_dir,
    get_tests_dir,
)


class TestFindProjectRoot:
    """Tests for find_project_root."""

    def test_finds_project_root(self, tmp_path: Path) -> None:
        """Find project root with pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        subdir = tmp_path / "src" / "pkg"
        subdir.mkdir(parents=True)

        root = find_project_root(subdir)
        assert root == tmp_path

    def test_raises_when_not_found(self, tmp_path: Path) -> None:
        """Raise error when pyproject.toml not found."""
        subdir = tmp_path / "some" / "deep" / "path"
        subdir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match=r"pyproject\.toml"):
            find_project_root(subdir)


class TestGetPackageDir:
    """Tests for get_package_dir."""

    def test_gets_package_dir(self, tmp_path: Path) -> None:
        """Get existing package directory."""
        pkg_dir = tmp_path / "src" / "mypackage"
        pkg_dir.mkdir(parents=True)

        result = get_package_dir(tmp_path, "mypackage")
        assert result == pkg_dir

    def test_raises_when_not_found(self, tmp_path: Path) -> None:
        """Raise error when package directory not found."""
        with pytest.raises(FileNotFoundError, match="Package directory not found"):
            get_package_dir(tmp_path, "nonexistent")


class TestGetTestsDir:
    """Tests for get_tests_dir."""

    def test_gets_tests_dir(self, tmp_path: Path) -> None:
        """Get existing tests directory."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        result = get_tests_dir(tmp_path)
        assert result == tests_dir

    def test_raises_when_not_found(self, tmp_path: Path) -> None:
        """Raise error when tests directory not found."""
        with pytest.raises(FileNotFoundError, match="Tests directory not found"):
            get_tests_dir(tmp_path)


class TestGetSpecsDir:
    """Tests for get_specs_dir."""

    def test_gets_specs_dir(self, tmp_path: Path) -> None:
        """Get existing specs directory."""
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        result = get_specs_dir(tmp_path)
        assert result == specs_dir

    def test_raises_when_not_found(self, tmp_path: Path) -> None:
        """Raise error when specs directory not found."""
        with pytest.raises(FileNotFoundError, match="Specs directory not found"):
            get_specs_dir(tmp_path)


class TestGetGuidesDir:
    """Tests for get_guides_dir."""

    def test_gets_guides_dir(self, tmp_path: Path) -> None:
        """Get existing guides directory."""
        guides_dir = tmp_path / "guides"
        guides_dir.mkdir()

        result = get_guides_dir(tmp_path)
        assert result == guides_dir

    def test_raises_when_not_found(self, tmp_path: Path) -> None:
        """Raise error when guides directory not found."""
        with pytest.raises(FileNotFoundError, match="Guides directory not found"):
            get_guides_dir(tmp_path)
