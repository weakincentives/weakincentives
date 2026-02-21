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

"""Tests for compute_current_visibility and open_sections handler."""

from __future__ import annotations

from typing import cast

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    PromptValidationError,
    SectionVisibility,
    VisibilityExpansionRequired,
)
from weakincentives.prompt.progressive_disclosure import (
    OpenSectionsParams,
    compute_current_visibility,
    create_open_sections_handler,
)
from weakincentives.prompt.registry import PromptRegistry
from weakincentives.prompt.section import Section
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session, SetVisibilityOverride
from weakincentives.types.dataclass import SupportsDataclass

from .conftest import (
    PDTestParams,
    make_pd_registry,
    make_pd_section,
    make_pd_tool_context,
)

# Tests for compute_current_visibility


def test_compute_current_visibility_default() -> None:
    """Returns default visibility when no overrides."""
    section = make_pd_section(
        key="sec",
        visibility=SectionVisibility.SUMMARY,
        summary="Summary",
    )
    registry = make_pd_registry((section,))
    snapshot = registry.snapshot()

    result = compute_current_visibility(snapshot)
    assert result["sec",] == SectionVisibility.SUMMARY


def test_compute_current_visibility_with_overrides() -> None:
    """Applies visibility overrides from session state."""
    section = make_pd_section(
        key="sec",
        visibility=SectionVisibility.SUMMARY,
        summary="Summary",
    )
    registry = make_pd_registry((section,))
    snapshot = registry.snapshot()

    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher)
    session.dispatch(
        SetVisibilityOverride(path=("sec",), visibility=SectionVisibility.FULL)
    )
    result = compute_current_visibility(snapshot, session=session)
    assert result["sec",] == SectionVisibility.FULL


def test_compute_current_visibility_uses_params_for_callable() -> None:
    """Visibility selectors can depend on section parameters."""

    def selector(params: PDTestParams) -> SectionVisibility:
        return (
            SectionVisibility.SUMMARY
            if params.name == "summarize"
            else SectionVisibility.FULL
        )

    section = make_pd_section(
        key="sec",
        summary="Summary",
        visibility=selector,
    )
    registry = make_pd_registry((section,))
    snapshot = registry.snapshot()

    summarized = compute_current_visibility(
        snapshot, param_lookup={PDTestParams: PDTestParams(name="summarize")}
    )
    assert summarized["sec",] == SectionVisibility.SUMMARY

    expanded = compute_current_visibility(
        snapshot, param_lookup={PDTestParams: PDTestParams(name="full")}
    )
    assert expanded["sec",] == SectionVisibility.FULL


# Tests for open_sections handler


def test_open_sections_raises_visibility_expansion_required() -> None:
    """Handler raises VisibilityExpansionRequired with correct overrides."""
    section = make_pd_section(
        key="summarized",
        visibility=SectionVisibility.SUMMARY,
        summary="Summary text",
    )
    registry = make_pd_registry((section,))
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
        tool.handler(params, context=make_pd_tool_context())  # type: ignore[arg-type]

    exc = cast(VisibilityExpansionRequired, exc_info.value)
    assert exc.section_keys == ("summarized",)
    assert exc.reason == "Need details"
    assert exc.requested_overrides["summarized",] == SectionVisibility.FULL


def test_open_sections_rejects_empty_keys() -> None:
    """Handler rejects empty section_keys."""
    section = make_pd_section(key="sec")
    registry = make_pd_registry((section,))
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
        tool.handler(params, context=make_pd_tool_context())  # type: ignore[arg-type]


def test_open_sections_rejects_nonexistent_section() -> None:
    """Handler rejects keys for sections that don't exist."""
    section = make_pd_section(key="exists")
    registry = make_pd_registry((section,))
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
        tool.handler(params, context=make_pd_tool_context())  # type: ignore[arg-type]


def test_open_sections_rejects_already_expanded() -> None:
    """Handler rejects keys for sections already at FULL visibility."""
    section = make_pd_section(
        key="expanded",
        visibility=SectionVisibility.FULL,
    )
    registry = make_pd_registry((section,))
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
        tool.handler(params, context=make_pd_tool_context())  # type: ignore[arg-type]


def test_open_sections_rejects_multiple_already_expanded() -> None:
    """Handler rejects when all requested sections are already expanded."""
    section1 = make_pd_section(
        key="expanded1",
        visibility=SectionVisibility.FULL,
    )
    section2 = make_pd_section(
        key="expanded2",
        visibility=SectionVisibility.FULL,
    )
    registry = make_pd_registry((section1, section2))
    snapshot = registry.snapshot()
    current_visibility = {
        ("expanded1",): SectionVisibility.FULL,
        ("expanded2",): SectionVisibility.FULL,
    }

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
    )

    params = OpenSectionsParams(
        section_keys=("expanded1", "expanded2"),
        reason="Both already expanded",
    )

    with pytest.raises(PromptValidationError, match="All requested sections"):
        tool.handler(params, context=make_pd_tool_context())  # type: ignore[arg-type]


def test_open_sections_skips_already_expanded_in_mixed_request() -> None:
    """Handler skips already-expanded sections and expands the rest."""
    expanded_section = make_pd_section(
        key="expanded",
        visibility=SectionVisibility.FULL,
    )
    summarized_section = make_pd_section(
        key="summarized",
        summary="Needs expansion",
        visibility=SectionVisibility.SUMMARY,
    )
    registry = make_pd_registry((expanded_section, summarized_section))
    snapshot = registry.snapshot()
    current_visibility = {
        ("expanded",): SectionVisibility.FULL,
        ("summarized",): SectionVisibility.SUMMARY,
    }

    tool = create_open_sections_handler(
        registry=snapshot,
        current_visibility=current_visibility,
    )

    # Request both expanded and summarized sections
    params = OpenSectionsParams(
        section_keys=("expanded", "summarized"),
        reason="Mixed request",
    )

    # Should raise VisibilityExpansionRequired only for the summarized section
    with pytest.raises(VisibilityExpansionRequired) as exc_info:
        tool.handler(params, context=make_pd_tool_context())  # type: ignore[arg-type]

    exc = cast(VisibilityExpansionRequired, exc_info.value)
    # Only the summarized section should be in the overrides
    assert ("summarized",) in exc.requested_overrides
    assert ("expanded",) not in exc.requested_overrides
    assert exc.requested_overrides["summarized",] == SectionVisibility.FULL


def test_open_sections_nested_key_parsing() -> None:
    """Handler correctly parses dot-notation section keys."""
    # Create parent section with child
    parent = MarkdownSection[PDTestParams](
        title="Parent",
        template="Parent: ${name}",
        key="parent",
        default_params=PDTestParams(),
        children=[
            MarkdownSection[PDTestParams](
                title="Child",
                template="Child: ${name}",
                key="child",
                summary="Child summary",
                visibility=SectionVisibility.SUMMARY,
                default_params=PDTestParams(),
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
        tool.handler(params, context=make_pd_tool_context())  # type: ignore[arg-type]

    exc = cast(VisibilityExpansionRequired, exc_info.value)
    assert ("parent", "child") in exc.requested_overrides
