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

"""Tests for progressive disclosure functionality."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast
from unittest.mock import MagicMock

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    PromptValidationError,
    SectionVisibility,
    SetVisibilityOverride,
    VisibilityExpansionRequired,
)
from weakincentives.prompt._types import SupportsDataclass
from weakincentives.prompt.progressive_disclosure import (
    OpenSectionsParams,
    build_summary_suffix,
    compute_current_visibility,
    create_open_sections_handler,
    has_summarized_sections,
)
from weakincentives.prompt.registry import PromptRegistry
from weakincentives.prompt.section import Section
from weakincentives.prompt.tool import ToolContext
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session


@dataclass
class _TestParams:
    name: str = "test"


def _make_section(
    *,
    key: str = "test-section",
    summary: str | None = None,
    visibility: SectionVisibility
    | Callable[[_TestParams], SectionVisibility]
    | Callable[[], SectionVisibility] = SectionVisibility.FULL,
) -> MarkdownSection[_TestParams]:
    return MarkdownSection[_TestParams](
        title="Test Section",
        template="Content: ${name}",
        key=key,
        summary=summary,
        visibility=visibility,
        default_params=_TestParams(),
    )


def _make_registry(
    sections: tuple[MarkdownSection[_TestParams], ...],
) -> PromptRegistry:
    registry = PromptRegistry()
    for section in sections:
        registry.register_section(
            cast(Section[SupportsDataclass], section),
            path=(section.key,),
            depth=0,
        )
    return registry


def _make_tool_context() -> ToolContext:
    """Create a mock ToolContext."""
    return ToolContext(
        prompt=MagicMock(),
        rendered_prompt=None,
        adapter=MagicMock(),
        session=MagicMock(),
    )


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
    section = _make_section(visibility=SectionVisibility.FULL)
    registry = _make_registry((section,))
    snapshot = registry.snapshot()

    assert has_summarized_sections(snapshot) is False


def test_has_summarized_sections_without_summary_text() -> None:
    """Returns False when SUMMARY visibility but no summary text."""
    section = _make_section(
        visibility=SectionVisibility.SUMMARY,
        summary=None,
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()

    assert has_summarized_sections(snapshot) is False


def test_has_summarized_sections_with_summary_text() -> None:
    """Returns True when SUMMARY visibility with summary text."""
    section = _make_section(
        visibility=SectionVisibility.SUMMARY,
        summary="Brief summary",
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()

    assert has_summarized_sections(snapshot) is True


def test_has_summarized_sections_with_overrides() -> None:
    """Visibility overrides via session state can expand summarized sections."""
    section = _make_section(
        key="sec",
        visibility=SectionVisibility.SUMMARY,
        summary="Brief summary",
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()

    # Without override, it's summarized
    assert has_summarized_sections(snapshot) is True

    # With override to FULL via session state, no summarized sections
    bus = InProcessEventBus()
    session = Session(bus=bus)
    session.broadcast(
        SetVisibilityOverride(path=("sec",), visibility=SectionVisibility.FULL)
    )
    assert has_summarized_sections(snapshot, session=session) is False


# Tests for compute_current_visibility


def test_compute_current_visibility_default() -> None:
    """Returns default visibility when no overrides."""
    section = _make_section(
        key="sec",
        visibility=SectionVisibility.SUMMARY,
        summary="Summary",
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()

    result = compute_current_visibility(snapshot)
    assert result["sec",] == SectionVisibility.SUMMARY


def test_compute_current_visibility_with_overrides() -> None:
    """Applies visibility overrides from session state."""
    section = _make_section(
        key="sec",
        visibility=SectionVisibility.SUMMARY,
        summary="Summary",
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()

    bus = InProcessEventBus()
    session = Session(bus=bus)
    session.broadcast(
        SetVisibilityOverride(path=("sec",), visibility=SectionVisibility.FULL)
    )
    result = compute_current_visibility(snapshot, session=session)
    assert result["sec",] == SectionVisibility.FULL


def test_compute_current_visibility_uses_params_for_callable() -> None:
    """Visibility selectors can depend on section parameters."""

    def selector(params: _TestParams) -> SectionVisibility:
        return (
            SectionVisibility.SUMMARY
            if params.name == "summarize"
            else SectionVisibility.FULL
        )

    section = _make_section(
        key="sec",
        summary="Summary",
        visibility=selector,
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()

    summarized = compute_current_visibility(
        snapshot, param_lookup={_TestParams: _TestParams(name="summarize")}
    )
    assert summarized["sec",] == SectionVisibility.SUMMARY

    expanded = compute_current_visibility(
        snapshot, param_lookup={_TestParams: _TestParams(name="full")}
    )
    assert expanded["sec",] == SectionVisibility.FULL


# Tests for open_sections handler


def test_open_sections_raises_visibility_expansion_required() -> None:
    """Handler raises VisibilityExpansionRequired with correct overrides."""
    section = _make_section(
        key="summarized",
        visibility=SectionVisibility.SUMMARY,
        summary="Summary text",
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()
    current_visibility = {("summarized",): SectionVisibility.SUMMARY}

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
    )

    params = OpenSectionsParams(
        section_keys=("summarized",),
        reason="Need details",
    )

    with pytest.raises(VisibilityExpansionRequired) as exc_info:
        tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]

    exc = cast(VisibilityExpansionRequired, exc_info.value)
    assert exc.section_keys == ("summarized",)
    assert exc.reason == "Need details"
    assert exc.requested_overrides["summarized",] == SectionVisibility.FULL


def test_open_sections_rejects_empty_keys() -> None:
    """Handler rejects empty section_keys."""
    section = _make_section(key="sec")
    registry = _make_registry((section,))
    snapshot = registry.snapshot()
    current_visibility = compute_current_visibility(snapshot)

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
    )

    params = OpenSectionsParams(
        section_keys=(),
        reason="Empty",
    )

    with pytest.raises(PromptValidationError, match="At least one section key"):
        tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]


def test_open_sections_rejects_nonexistent_section() -> None:
    """Handler rejects keys for sections that don't exist."""
    section = _make_section(key="exists")
    registry = _make_registry((section,))
    snapshot = registry.snapshot()
    current_visibility = compute_current_visibility(snapshot)

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
    )

    params = OpenSectionsParams(
        section_keys=("does-not-exist",),
        reason="Invalid",
    )

    with pytest.raises(PromptValidationError, match="does not exist"):
        tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]


def test_open_sections_rejects_already_expanded() -> None:
    """Handler rejects keys for sections already at FULL visibility."""
    section = _make_section(
        key="expanded",
        visibility=SectionVisibility.FULL,
    )
    registry = _make_registry((section,))
    snapshot = registry.snapshot()
    current_visibility = {("expanded",): SectionVisibility.FULL}

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
    )

    params = OpenSectionsParams(
        section_keys=("expanded",),
        reason="Already expanded",
    )

    with pytest.raises(PromptValidationError, match="already expanded"):
        tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]


def test_open_sections_nested_key_parsing() -> None:
    """Handler correctly parses dot-notation section keys."""
    # Create parent section with child
    parent = MarkdownSection[_TestParams](
        title="Parent",
        template="Parent: ${name}",
        key="parent",
        default_params=_TestParams(),
        children=[
            MarkdownSection[_TestParams](
                title="Child",
                template="Child: ${name}",
                key="child",
                summary="Child summary",
                visibility=SectionVisibility.SUMMARY,
                default_params=_TestParams(),
            )
        ],
    )

    registry = PromptRegistry()
    registry.register_sections((cast(Section[SupportsDataclass], parent),))
    snapshot = registry.snapshot()
    current_visibility = {
        ("parent",): SectionVisibility.FULL,
        ("parent", "child"): SectionVisibility.SUMMARY,
    }

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
    )

    params = OpenSectionsParams(
        section_keys=("parent.child",),
        reason="Need child details",
    )

    with pytest.raises(VisibilityExpansionRequired) as exc_info:
        tool.handler(params, context=_make_tool_context())  # type: ignore[arg-type]

    exc = cast(VisibilityExpansionRequired, exc_info.value)
    assert ("parent", "child") in exc.requested_overrides


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
