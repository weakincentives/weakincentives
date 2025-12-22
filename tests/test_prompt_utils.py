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

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.contrib.tools.digests import (
    WorkspaceDigest,
    WorkspaceDigestSection,
    set_workspace_digest,
)
from weakincentives.prompt import MarkdownSection, PromptTemplate
from weakincentives.prompt._visibility import SectionVisibility
from weakincentives.prompt.overrides import (
    LocalPromptOverridesStore,
    PromptOverridesError,
)
from weakincentives.prompt.overrides.versioning import PromptDescriptor
from weakincentives.runtime.session import Session


def _build_prompt() -> PromptTemplate[str]:
    sections = (
        MarkdownSection(title="One", template="first", key="one"),
        MarkdownSection(title="Two", template="second", key="two"),
    )
    return PromptTemplate(ns="test", key="prompt", sections=sections)


def test_prompt_find_section_by_key_and_candidates() -> None:
    prompt = _build_prompt()

    assert isinstance(prompt.find_section(MarkdownSection), MarkdownSection)
    assert isinstance(
        prompt.find_section((MarkdownSection,)),
        MarkdownSection,
    )
    assert prompt.find_section((MarkdownSection, WorkspaceDigestSection)).key == "one"
    with pytest.raises(TypeError):
        prompt.find_section(())
    with pytest.raises(KeyError):
        prompt.find_section((WorkspaceDigestSection,))


def test_set_section_override_requires_registered_path(tmp_path: Path) -> None:
    prompt = _build_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)

    descriptor = PromptDescriptor.from_prompt(prompt)
    assert any(section.path == ("one",) for section in descriptor.sections)

    with pytest.raises(PromptOverridesError):
        store.set_section_override(prompt, path=("missing",), body="X")


def test_workspace_digest_section_in_descriptor() -> None:
    section = WorkspaceDigestSection(session=Session())
    descriptor = PromptDescriptor.from_prompt(
        PromptTemplate(ns="digest", key="prompt", sections=(section,))
    )

    assert section.original_body_template() == section._placeholder
    assert ("workspace-digest",) in {entry.path for entry in descriptor.sections}
    assert WorkspaceDigest(section_key="a", summary="s", body="b").section_key == "a"
    with pytest.raises(TypeError):
        section.clone()
    clone = section.clone(session=Session())
    assert isinstance(clone, WorkspaceDigestSection)
    heading = section.format_heading(depth=0, number="1.")
    assert heading.startswith("## 1. Workspace Digest")


def test_workspace_digest_section_render_override_with_empty_body() -> None:
    """render_override returns heading only when body resolves to empty string."""
    section = WorkspaceDigestSection(session=Session(), placeholder="")

    # With empty placeholder and no digest, render_override returns heading only
    rendered = section.render_override("   ", None, 0, "1.")

    assert rendered == "## 1. Workspace Digest"


def test_workspace_digest_section_original_summary_template() -> None:
    """original_summary_template returns a static placeholder for hashing."""
    section = WorkspaceDigestSection(session=Session())

    assert section.original_summary_template() == "Workspace digest summary."


def test_workspace_digest_section_render_body_with_full_visibility() -> None:
    """render_body returns full body content when visibility is FULL."""
    session = Session()
    set_workspace_digest(
        session, "workspace-digest", "Full body content", summary="Short summary"
    )
    section = WorkspaceDigestSection(session=session)

    # render_body with FULL visibility returns the full body
    body = section.render_body(None, visibility=SectionVisibility.FULL)

    assert body == "Full body content"


def test_workspace_digest_section_render_body_with_summary_visibility() -> None:
    """render_body returns summary content when visibility is SUMMARY."""
    session = Session()
    set_workspace_digest(
        session, "workspace-digest", "Full body content", summary="Short summary"
    )
    section = WorkspaceDigestSection(session=session)

    # render_body with SUMMARY visibility returns the summary
    body = section.render_body(None, visibility=SectionVisibility.SUMMARY)

    assert body == "Short summary"


def test_workspace_digest_section_no_digest_renders_placeholder() -> None:
    """When no digest exists, render_body returns placeholder regardless of visibility."""
    session = Session()
    section = WorkspaceDigestSection(session=session)

    # No digest exists - should return placeholder
    body = section.render_body(None, visibility=SectionVisibility.FULL)

    assert "Workspace digest unavailable" in body


def test_workspace_digest_section_dynamic_visibility() -> None:
    """Visibility is FULL when no digest, SUMMARY when digest exists."""
    session = Session()
    section = WorkspaceDigestSection(session=session)

    # No digest - visibility should be FULL, summary should be None
    assert section._visibility_selector() == SectionVisibility.FULL
    assert section.summary is None

    # Add a digest
    set_workspace_digest(session, "workspace-digest", "body", summary="summary")

    # Now visibility should be SUMMARY, summary should exist
    assert section._visibility_selector() == SectionVisibility.SUMMARY
    assert section.summary == "summary"


def test_workspace_digest_section_render_override_respects_visibility() -> None:
    """render_override respects visibility even when body override exists."""
    session = Session()
    set_workspace_digest(
        session, "workspace-digest", "Full body content", summary="Short summary"
    )
    section = WorkspaceDigestSection(session=session)

    # render_override should respect the section's visibility (SUMMARY by default)
    # The override_body parameter should be ignored when a digest exists
    rendered = section.render_override("ignored override body", None, 0, "1.")

    # Should render the summary, not the full body or the override body
    assert "Short summary" in rendered
    assert "Full body content" not in rendered
    assert "ignored override body" not in rendered


def test_workspace_digest_section_render_override_with_full_visibility() -> None:
    """render_override returns full body when visibility is set to FULL."""
    from weakincentives.prompt.visibility_overrides import (
        SectionVisibility,
        SetVisibilityOverride,
    )

    session = Session()
    set_workspace_digest(
        session, "workspace-digest", "Full body content", summary="Short summary"
    )
    # Set visibility override to FULL
    session.dispatch(
        SetVisibilityOverride(
            path=("workspace-digest",), visibility=SectionVisibility.FULL
        )
    )
    section = WorkspaceDigestSection(session=session)

    # render_override with FULL visibility should return the full body
    # Note: path must match the visibility override path for it to be looked up
    rendered = section.render_override(
        "ignored override body", None, 0, "1.", path=("workspace-digest",)
    )

    assert "Full body content" in rendered
    assert "Short summary" not in rendered


"""Coverage tests for prompt utilities and workspace digest plumbing."""
