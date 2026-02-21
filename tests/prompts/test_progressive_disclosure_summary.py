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

"""Tests for build_summary_suffix, has_summarized_sections, and VisibilityExpansionRequired."""

from __future__ import annotations

import pytest

from weakincentives.prompt import (
    PromptValidationError,
    SectionVisibility,
    VisibilityExpansionRequired,
)
from weakincentives.prompt.progressive_disclosure import (
    build_summary_suffix,
    has_summarized_sections,
)
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session, SetVisibilityOverride

from .conftest import make_pd_registry, make_pd_section

# Tests for build_summary_suffix


def test_build_summary_suffix_without_children() -> None:
    """Suffix without child sections."""
    suffix = build_summary_suffix("my-section", ())
    assert 'key "my-section"' in suffix
    assert "subsections" not in suffix
    assert "---" in suffix


def test_build_summary_suffix_with_children() -> None:
    """Suffix includes child section names."""
    suffix = build_summary_suffix("parent", ("child1", "child2"))
    assert 'key "parent"' in suffix
    assert "subsections: child1, child2" in suffix


# Tests for has_summarized_sections


def test_has_summarized_sections_no_summarized() -> None:
    """Returns False when all sections are FULL visibility."""
    section = make_pd_section(visibility=SectionVisibility.FULL)
    registry = make_pd_registry((section,))
    snapshot = registry.snapshot()

    assert has_summarized_sections(snapshot) is False


def test_has_summarized_sections_without_summary_text() -> None:
    """Raises when SUMMARY visibility but no summary text."""
    section = make_pd_section(
        visibility=SectionVisibility.SUMMARY,
        summary=None,
    )
    registry = make_pd_registry((section,))
    snapshot = registry.snapshot()

    with pytest.raises(PromptValidationError) as excinfo:
        has_summarized_sections(snapshot)

    assert "SUMMARY visibility requested" in str(excinfo.value)


def test_has_summarized_sections_with_summary_text() -> None:
    """Returns True when SUMMARY visibility with summary text."""
    section = make_pd_section(
        visibility=SectionVisibility.SUMMARY,
        summary="Brief summary",
    )
    registry = make_pd_registry((section,))
    snapshot = registry.snapshot()

    assert has_summarized_sections(snapshot) is True


def test_has_summarized_sections_with_overrides() -> None:
    """Visibility overrides via session state can expand summarized sections."""
    section = make_pd_section(
        key="sec",
        visibility=SectionVisibility.SUMMARY,
        summary="Brief summary",
    )
    registry = make_pd_registry((section,))
    snapshot = registry.snapshot()

    # Without override, it's summarized
    assert has_summarized_sections(snapshot) is True

    # With override to FULL via session state, no summarized sections
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher)
    session.dispatch(
        SetVisibilityOverride(path=("sec",), visibility=SectionVisibility.FULL)
    )
    assert has_summarized_sections(snapshot, session=session) is False


# Tests for VisibilityExpansionRequired exception


def test_visibility_expansion_required_str_representation() -> None:
    """Exception has informative string representation."""
    exc = VisibilityExpansionRequired(
        "Expansion needed",
        requested_overrides={("sec1",): SectionVisibility.FULL},
        reason="Need more info",
        section_keys=("sec1",),
    )

    str_repr = str(exc)
    assert "sec1" in str_repr
    assert "Need more info" in str_repr


def test_visibility_expansion_required_attributes() -> None:
    """Exception stores provided attributes."""
    overrides = {
        ("a",): SectionVisibility.FULL,
        ("b", "c"): SectionVisibility.FULL,
    }
    exc = VisibilityExpansionRequired(
        "Expansion needed",
        requested_overrides=overrides,
        reason="Testing",
        section_keys=("a", "b.c"),
    )

    assert exc.requested_overrides == overrides
    assert exc.reason == "Testing"
    assert exc.section_keys == ("a", "b.c")


# Tests for build_summary_suffix with has_tools parameter


def test_build_summary_suffix_with_tools() -> None:
    """Suffix directs to open_sections when section has tools."""
    suffix = build_summary_suffix("my-section", (), has_tools=True)
    assert "open_sections" in suffix
    assert "read_section" not in suffix


def test_build_summary_suffix_without_tools() -> None:
    """Suffix directs to read_section when section has no tools."""
    suffix = build_summary_suffix("my-section", (), has_tools=False)
    assert "read_section" in suffix
    assert "open_sections" not in suffix


def test_build_summary_suffix_with_children_and_tools() -> None:
    """Suffix includes children and directs to open_sections."""
    suffix = build_summary_suffix("parent", ("child1", "child2"), has_tools=True)
    assert "open_sections" in suffix
    assert "subsections: child1, child2" in suffix


def test_build_summary_suffix_with_children_without_tools() -> None:
    """Suffix includes children and directs to read_section."""
    suffix = build_summary_suffix("parent", ("child1", "child2"), has_tools=False)
    assert "read_section" in suffix
    assert "subsections: child1, child2" in suffix
