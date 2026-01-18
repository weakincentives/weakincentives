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

"""Tests for checker registry."""

from __future__ import annotations

from core_types import Checker
from registry import (
    get_all_checkers,
    get_categories,
    get_checker,
    get_checkers_by_category,
)


class TestRegistry:
    """Tests for checker registry functions."""

    def test_get_all_checkers(self) -> None:
        """Get all registered checkers."""
        checkers = get_all_checkers()
        assert len(checkers) > 0
        assert all(isinstance(c, Checker) for c in checkers)

    def test_get_checker_existing(self) -> None:
        """Get an existing checker by name."""
        checker = get_checker("layer_violations")
        assert checker is not None
        assert checker.name == "layer_violations"

    def test_get_checker_nonexistent(self) -> None:
        """Get nonexistent checker returns None."""
        checker = get_checker("nonexistent_checker")
        assert checker is None

    def test_get_checkers_by_category(self) -> None:
        """Get checkers by category."""
        arch_checkers = get_checkers_by_category("architecture")
        assert len(arch_checkers) > 0
        assert all(c.category == "architecture" for c in arch_checkers)

    def test_get_checkers_by_nonexistent_category(self) -> None:
        """Get checkers for nonexistent category returns empty."""
        checkers = get_checkers_by_category("nonexistent")
        assert checkers == ()

    def test_get_categories(self) -> None:
        """Get all unique categories."""
        categories = get_categories()
        assert len(categories) > 0
        assert "architecture" in categories
        assert "documentation" in categories

    def test_all_checkers_have_required_properties(self) -> None:
        """All checkers have required properties."""
        for checker in get_all_checkers():
            assert isinstance(checker.name, str)
            assert len(checker.name) > 0
            assert isinstance(checker.category, str)
            assert len(checker.category) > 0
            assert isinstance(checker.description, str)
            assert len(checker.description) > 0
            assert callable(checker.check)
