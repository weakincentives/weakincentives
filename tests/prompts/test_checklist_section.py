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

"""Tests for ChecklistSection and domain-specific checklist builders."""

from __future__ import annotations

import pytest

from weakincentives.prompt import (
    ChecklistItem,
    ChecklistParams,
    ChecklistSection,
    SectionVisibility,
)

# =============================================================================
# ChecklistItem Tests
# =============================================================================


def test_checklist_item_defaults() -> None:
    """ChecklistItem has sensible defaults."""
    item = ChecklistItem(text="Check something")

    assert item.text == "Check something"
    assert item.category is None
    assert item.severity == "medium"


def test_checklist_item_with_all_fields() -> None:
    """ChecklistItem accepts all fields."""
    item = ChecklistItem(
        text="Verify encryption",
        category="Security",
        severity="critical",
    )

    assert item.text == "Verify encryption"
    assert item.category == "Security"
    assert item.severity == "critical"


def test_checklist_item_is_frozen() -> None:
    """ChecklistItem is immutable."""
    item = ChecklistItem(text="Test")

    with pytest.raises(AttributeError):
        item.text = "Modified"  # type: ignore[misc]


# =============================================================================
# ChecklistParams Tests
# =============================================================================


def test_checklist_params_defaults() -> None:
    """ChecklistParams has sensible defaults."""
    params = ChecklistParams()

    assert params.domain == ""
    assert params.item_count == 0


def test_checklist_params_with_values() -> None:
    """ChecklistParams accepts custom values."""
    params = ChecklistParams(domain="security", item_count=10)

    assert params.domain == "security"
    assert params.item_count == 10


# =============================================================================
# ChecklistSection Basic Tests
# =============================================================================


def test_checklist_section_basic_construction() -> None:
    """ChecklistSection can be constructed with basic arguments."""
    items = [
        ChecklistItem(text="Item 1"),
        ChecklistItem(text="Item 2"),
    ]

    section = ChecklistSection(
        title="Test Checklist",
        key="test-checklist",
        domain="test",
        items=items,
    )

    assert section.title == "Test Checklist"
    assert section.key == "test-checklist"
    assert section.domain == "test"
    assert len(section.items) == 2


def test_checklist_section_stores_items_as_tuple() -> None:
    """Items are stored as an immutable tuple."""
    items = [ChecklistItem(text="Item")]
    section = ChecklistSection(
        title="Test",
        key="test",
        domain="test",
        items=items,
    )

    assert isinstance(section.items, tuple)


def test_checklist_section_preamble() -> None:
    """ChecklistSection stores preamble text."""
    section = ChecklistSection(
        title="Test",
        key="test",
        domain="test",
        items=[ChecklistItem(text="Item")],
        preamble="Review these items carefully.",
    )

    assert section.preamble == "Review these items carefully."


def test_checklist_section_default_params() -> None:
    """ChecklistSection sets default params with domain and count."""
    items = [ChecklistItem(text="A"), ChecklistItem(text="B"), ChecklistItem(text="C")]
    section = ChecklistSection(
        title="Test",
        key="test",
        domain="testing",
        items=items,
    )

    assert section.default_params is not None
    assert section.default_params.domain == "testing"
    assert section.default_params.item_count == 3


def test_checklist_section_generates_summary() -> None:
    """ChecklistSection auto-generates summary with item count."""
    items = [ChecklistItem(text="Item 1"), ChecklistItem(text="Item 2")]
    section = ChecklistSection(
        title="Test",
        key="test",
        domain="security",
        items=items,
    )

    assert section.summary is not None
    assert "2 security review items" in section.summary


def test_checklist_section_summary_includes_category_count() -> None:
    """Summary mentions category count when multiple categories exist."""
    items = [
        ChecklistItem(text="Item 1", category="Cat A"),
        ChecklistItem(text="Item 2", category="Cat B"),
        ChecklistItem(text="Item 3", category="Cat C"),
    ]
    section = ChecklistSection(
        title="Test",
        key="test",
        domain="test",
        items=items,
    )

    assert section.summary is not None
    assert "3 categories" in section.summary


def test_checklist_section_default_visibility_is_full() -> None:
    """Default visibility is FULL."""
    section = ChecklistSection(
        title="Test",
        key="test",
        domain="test",
        items=[ChecklistItem(text="Item")],
    )

    assert section.visibility == SectionVisibility.FULL


def test_checklist_section_custom_visibility() -> None:
    """ChecklistSection accepts custom visibility."""
    section = ChecklistSection(
        title="Test",
        key="test",
        domain="test",
        items=[ChecklistItem(text="Item")],
        visibility=SectionVisibility.SUMMARY,
    )

    assert section.visibility == SectionVisibility.SUMMARY


# =============================================================================
# ChecklistSection Rendering Tests
# =============================================================================


def test_checklist_section_render_full_basic() -> None:
    """Full render includes heading and items."""
    items = [
        ChecklistItem(text="Check A"),
        ChecklistItem(text="Check B"),
    ]
    section = ChecklistSection(
        title="Review",
        key="review",
        domain="test",
        items=items,
    )

    rendered = section.render(None, depth=0, number="1")

    assert "## 1. Review" in rendered
    assert "- [ ] Check A" in rendered
    assert "- [ ] Check B" in rendered


def test_checklist_section_render_full_with_preamble() -> None:
    """Full render includes preamble when provided."""
    section = ChecklistSection(
        title="Review",
        key="review",
        domain="test",
        items=[ChecklistItem(text="Item")],
        preamble="Important guidelines:",
    )

    rendered = section.render(None, depth=0, number="1")

    assert "Important guidelines:" in rendered


def test_checklist_section_render_full_with_categories() -> None:
    """Full render groups items by category."""
    items = [
        ChecklistItem(text="Auth check", category="Authentication"),
        ChecklistItem(text="Encrypt check", category="Encryption"),
        ChecklistItem(text="Another auth", category="Authentication"),
    ]
    section = ChecklistSection(
        title="Security",
        key="security",
        domain="security",
        items=items,
    )

    rendered = section.render(None, depth=0, number="1")

    assert "**Authentication**" in rendered
    assert "**Encryption**" in rendered


def test_checklist_section_render_full_with_severity() -> None:
    """Full render shows severity markers for non-medium items."""
    items = [
        ChecklistItem(text="Critical item", severity="critical"),
        ChecklistItem(text="High item", severity="high"),
        ChecklistItem(text="Medium item", severity="medium"),
        ChecklistItem(text="Low item", severity="low"),
    ]
    section = ChecklistSection(
        title="Review",
        key="review",
        domain="test",
        items=items,
    )

    rendered = section.render(None, depth=0, number="1")

    assert "[CRITICAL] Critical item" in rendered
    assert "[HIGH] High item" in rendered
    assert "- [ ] Medium item" in rendered  # No marker for medium
    assert "[LOW] Low item" in rendered


def test_checklist_section_render_summary() -> None:
    """Summary render shows condensed view."""
    section = ChecklistSection(
        title="Review",
        key="review",
        domain="security",
        items=[ChecklistItem(text="Item")],
        visibility=SectionVisibility.SUMMARY,
    )

    rendered = section.render(None, depth=0, number="1")

    assert "## 1. Review" in rendered
    assert "1 security review items" in rendered
    assert "- [ ]" not in rendered  # No checkboxes in summary


def test_checklist_section_render_respects_visibility_override() -> None:
    """Visibility override takes precedence."""
    section = ChecklistSection(
        title="Review",
        key="review",
        domain="test",
        items=[ChecklistItem(text="Check this")],
        visibility=SectionVisibility.SUMMARY,
    )

    # Override to FULL
    rendered = section.render(
        None, depth=0, number="1", visibility=SectionVisibility.FULL
    )

    assert "- [ ] Check this" in rendered


def test_checklist_section_render_depth_affects_heading() -> None:
    """Heading level increases with depth."""
    section = ChecklistSection(
        title="Nested",
        key="nested",
        domain="test",
        items=[ChecklistItem(text="Item")],
    )

    depth_0 = section.render(None, depth=0, number="1")
    depth_1 = section.render(None, depth=1, number="1.1")
    depth_2 = section.render(None, depth=2, number="1.1.1")

    assert "## 1. Nested" in depth_0
    assert "### 1.1. Nested" in depth_1
    assert "#### 1.1.1. Nested" in depth_2


def test_checklist_section_render_includes_path() -> None:
    """Rendered heading includes path when provided."""
    section = ChecklistSection(
        title="Review",
        key="review",
        domain="test",
        items=[ChecklistItem(text="Item")],
    )

    rendered = section.render(None, depth=0, number="1", path=("parent", "review"))

    assert "(parent.review)" in rendered


# =============================================================================
# ChecklistSection Clone Tests
# =============================================================================


def test_checklist_section_clone_basic() -> None:
    """Clone creates independent copy."""
    items = [ChecklistItem(text="Original")]
    section = ChecklistSection(
        title="Test",
        key="test",
        domain="test",
        items=items,
        preamble="Original preamble",
    )

    cloned = section.clone()

    assert cloned.title == section.title
    assert cloned.key == section.key
    assert cloned.domain == section.domain
    assert cloned.items == section.items
    assert cloned.preamble == section.preamble
    assert cloned is not section


def test_checklist_section_clone_preserves_visibility() -> None:
    """Clone preserves visibility settings."""
    section = ChecklistSection(
        title="Test",
        key="test",
        domain="test",
        items=[ChecklistItem(text="Item")],
        visibility=SectionVisibility.SUMMARY,
    )

    cloned = section.clone()

    assert cloned.visibility == SectionVisibility.SUMMARY


def test_checklist_section_clone_with_children() -> None:
    """Clone properly clones children."""
    child = ChecklistSection(
        title="Child",
        key="child",
        domain="child",
        items=[ChecklistItem(text="Child item")],
    )
    section = ChecklistSection(
        title="Parent",
        key="parent",
        domain="parent",
        items=[ChecklistItem(text="Parent item")],
        children=[child],
    )

    cloned = section.clone()

    assert len(cloned.children) == 1
    assert cloned.children[0] is not child
    assert cloned.children[0].title == "Child"


def test_checklist_section_original_body_template() -> None:
    """Original body template returns stable representation."""
    items = [
        ChecklistItem(text="Item A"),
        ChecklistItem(text="Item B"),
    ]
    section = ChecklistSection(
        title="Test",
        key="test",
        domain="test",
        items=items,
        preamble="Preamble text",
    )

    template = section.original_body_template()

    assert template is not None
    assert "Preamble text" in template
    assert "- Item A" in template
    assert "- Item B" in template
