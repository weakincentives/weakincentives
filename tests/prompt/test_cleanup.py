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

"""Tests for Section.cleanup() and Prompt.cleanup() lifecycle."""

from __future__ import annotations

from typing import Self

from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.prompt.section import Section
from weakincentives.types.dataclass import SupportsDataclass


class _TrackingSection(Section[None]):
    """Section that records cleanup calls for testing."""

    cleanup_count: int

    def __init__(
        self,
        *,
        key: str,
        children: list[Section[SupportsDataclass]] | None = None,
    ) -> None:
        super().__init__(
            title=f"Tracking ({key})",
            key=key,
            children=children,
        )
        self.cleanup_count = 0

    def cleanup(self) -> None:
        self.cleanup_count += 1

    def clone(self: Self, **kwargs: object) -> Self:
        del kwargs
        return self  # pragma: no cover


# =============================================================================
# Section.cleanup() Tests
# =============================================================================


def test_section_cleanup_is_noop() -> None:
    """Section.cleanup() default implementation does nothing."""
    section = MarkdownSection(title="Test", template="hello", key="test")
    # Should not raise
    section.cleanup()
    section.cleanup()  # idempotent


# =============================================================================
# Prompt.cleanup() Tests
# =============================================================================


def test_prompt_cleanup_calls_sections() -> None:
    """Prompt.cleanup() calls cleanup() on each root section."""
    s1 = _TrackingSection(key="sec-a")
    s2 = _TrackingSection(key="sec-b")

    template = PromptTemplate(ns="test", key="cleanup-test", sections=[s1, s2])
    prompt = Prompt(template)

    prompt.cleanup()

    assert s1.cleanup_count == 1
    assert s2.cleanup_count == 1


def test_prompt_cleanup_is_depth_first() -> None:
    """Prompt.cleanup() recurses into children depth-first."""
    grandchild = _TrackingSection(key="grandchild")
    child = _TrackingSection(key="child", children=[grandchild])
    root = _TrackingSection(key="root", children=[child])

    template = PromptTemplate(ns="test", key="deep-cleanup", sections=[root])
    prompt = Prompt(template)

    prompt.cleanup()

    assert root.cleanup_count == 1
    assert child.cleanup_count == 1
    assert grandchild.cleanup_count == 1


def test_prompt_cleanup_idempotent() -> None:
    """Prompt.cleanup() can be called multiple times safely."""
    s1 = _TrackingSection(key="sec-a")

    template = PromptTemplate(ns="test", key="idem-cleanup", sections=[s1])
    prompt = Prompt(template)

    prompt.cleanup()
    prompt.cleanup()

    assert s1.cleanup_count == 2


def test_prompt_cleanup_no_sections() -> None:
    """Prompt.cleanup() is safe with no sections."""
    template = PromptTemplate(ns="test", key="empty-cleanup")
    prompt = Prompt(template)

    # Should not raise
    prompt.cleanup()
