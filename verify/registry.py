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

"""Checker registry for the verification toolbox."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core_types import Checker


def _load_all_checkers() -> tuple[Checker, ...]:
    """Lazily load all checker instances."""
    # Import here to avoid circular imports
    from checkers.architecture import (
        CoreContribSeparationChecker,
        LayerViolationsChecker,
    )
    from checkers.dependencies import DeptryChecker
    from checkers.documentation import (
        DocExamplesChecker,
        MarkdownFormatChecker,
        MarkdownLinksChecker,
        SpecReferencesChecker,
    )
    from checkers.security import BanditChecker, PipAuditChecker
    from checkers.tests import PytestChecker
    from checkers.types import (
        IntegrationTypesChecker,
        TypeCoverageChecker,
    )

    return (
        # Architecture
        LayerViolationsChecker(),
        CoreContribSeparationChecker(),
        # Documentation
        SpecReferencesChecker(),
        DocExamplesChecker(),
        MarkdownLinksChecker(),
        MarkdownFormatChecker(),
        # Security
        BanditChecker(),
        PipAuditChecker(),
        # Dependencies
        DeptryChecker(),
        # Types
        TypeCoverageChecker(),
        IntegrationTypesChecker(),
        # Tests
        PytestChecker(),
    )


# Cached checker instances (mutable cache, hence lowercase)
_cached_checkers: tuple[Checker, ...] | None = None


def get_all_checkers() -> tuple[Checker, ...]:
    """Get all registered checkers.

    Returns:
        A tuple of all checker instances.
    """
    global _cached_checkers
    if _cached_checkers is None:
        _cached_checkers = _load_all_checkers()
    return _cached_checkers


def get_checker(name: str) -> Checker | None:
    """Get a checker by name.

    Args:
        name: The checker name (e.g., "layer_violations").

    Returns:
        The checker instance, or None if not found.
    """
    for checker in get_all_checkers():
        if checker.name == name:
            return checker
    return None


def get_checkers_by_category(category: str) -> tuple[Checker, ...]:
    """Get all checkers in a category.

    Args:
        category: The category name (e.g., "architecture").

    Returns:
        A tuple of checker instances in that category.
    """
    return tuple(c for c in get_all_checkers() if c.category == category)


def get_categories() -> tuple[str, ...]:
    """Get all unique checker categories.

    Returns:
        A tuple of category names.
    """
    seen: set[str] = set()
    categories: list[str] = []
    for checker in get_all_checkers():
        if checker.category not in seen:
            seen.add(checker.category)
            categories.append(checker.category)
    return tuple(categories)
