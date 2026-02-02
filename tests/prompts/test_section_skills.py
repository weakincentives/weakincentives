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

"""Tests for section-level skill attachment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from weakincentives.prompt.markdown import MarkdownSection
from weakincentives.prompt.section import Section, SectionVisibility
from weakincentives.skills import SkillMount


@dataclass
class SectionParams:
    title: str = "placeholder"


class _BareSection(Section[SectionParams]):
    def __init__(
        self,
        *,
        title: str,
        key: str,
        skills: Sequence[object] | None = None,
    ) -> None:
        super().__init__(
            title=title,
            key=key,
            skills=skills,
        )

    def clone(self, **kwargs: object) -> _BareSection:
        skills = kwargs.get("skills", self.skills())
        return _BareSection(
            title=self.title,
            key=self.key,
            skills=skills,  # type: ignore[arg-type]
        )

    def render(
        self,
        params: SectionParams,
        depth: int,
        number: str,
        *,
        visibility: SectionVisibility | None = None,
    ) -> str:
        del params, depth, number, visibility
        _ = self.key
        return ""


def test_sections_default_to_no_skills() -> None:
    section = _BareSection(title="Base", key="base")

    assert section.skills() == ()


def test_sections_reject_non_skill_mount_entries() -> None:
    with pytest.raises(TypeError, match="Section skills must be SkillMount instances"):
        _BareSection(title="Invalid", key="invalid", skills=["oops"])


def test_sections_expose_skills_in_order() -> None:
    first = SkillMount(source=Path("/skills/first"))
    second = SkillMount(source=Path("/skills/second"))
    section = _BareSection(
        title="With Skills", key="with-skills", skills=[first, second]
    )

    skills = section.skills()

    assert skills == (first, second)
    assert skills[0] is first
    assert skills[1] is second


def test_text_section_accepts_skills() -> None:
    skill = SkillMount(source=Path("/skills/test"))
    section = MarkdownSection[SectionParams](
        title="Paragraph",
        template="Hello",
        key="paragraph",
        skills=[skill],
    )

    assert section.skills() == (skill,)


def test_section_clone_preserves_skills() -> None:
    skill = SkillMount(source=Path("/skills/test"))
    section = _BareSection(title="Original", key="original", skills=[skill])

    cloned = section.clone()

    assert cloned.skills() == (skill,)


def test_section_clone_accepts_new_skills() -> None:
    original_skill = SkillMount(source=Path("/skills/original"))
    new_skill = SkillMount(source=Path("/skills/new"))
    section = _BareSection(title="Original", key="original", skills=[original_skill])

    cloned = section.clone(skills=[new_skill])

    assert cloned.skills() == (new_skill,)


# ============================================================================
# Registry skill registration tests
# ============================================================================


def test_registry_registers_section_skills(tmp_path: Path) -> None:
    """Skills from sections are registered in the skill_name_registry."""
    from typing import cast

    from weakincentives.prompt.registry import PromptRegistry
    from weakincentives.types import SupportsDataclass

    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    skill = SkillMount(source=skill_dir, name="test-skill")
    section = _BareSection(title="With Skill", key="with-skill", skills=[skill])

    registry = PromptRegistry()
    registry.register_sections((cast(Section[SupportsDataclass], section),))
    snapshot = registry.snapshot()

    assert "test-skill" in snapshot.skill_name_registry
    assert snapshot.skill_name_registry["test-skill"] == ("with-skill",)


def test_registry_detects_duplicate_skill_names(tmp_path: Path) -> None:
    """Duplicate skill names across sections raise PromptValidationError."""
    from typing import cast

    from weakincentives.prompt.errors import PromptValidationError
    from weakincentives.prompt.registry import PromptRegistry
    from weakincentives.types import SupportsDataclass

    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    skill1 = SkillMount(source=skill_dir, name="duplicate-skill")
    skill2 = SkillMount(source=skill_dir, name="duplicate-skill")

    section1 = _BareSection(title="First", key="first", skills=[skill1])
    section2 = _BareSection(title="Second", key="second", skills=[skill2])

    registry = PromptRegistry()
    with pytest.raises(PromptValidationError, match="Duplicate skill name"):
        registry.register_sections(
            (
                cast(Section[SupportsDataclass], section1),
                cast(Section[SupportsDataclass], section2),
            )
        )


def test_registry_subtree_has_skills(tmp_path: Path) -> None:
    """subtree_has_skills correctly tracks skill presence in hierarchy."""
    from typing import cast

    from weakincentives.prompt.registry import PromptRegistry
    from weakincentives.types import SupportsDataclass

    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    skill = SkillMount(source=skill_dir, name="test-skill")
    section_with_skill = _BareSection(
        title="With Skill", key="with-skill", skills=[skill]
    )
    section_no_skill = _BareSection(title="No Skill", key="no-skill")

    registry = PromptRegistry()
    registry.register_sections(
        (
            cast(Section[SupportsDataclass], section_with_skill),
            cast(Section[SupportsDataclass], section_no_skill),
        )
    )
    snapshot = registry.snapshot()

    assert snapshot.subtree_has_skills["with-skill",] is True
    assert snapshot.subtree_has_skills["no-skill",] is False


# ============================================================================
# Rendered prompt skill collection tests
# ============================================================================


def test_rendered_prompt_collects_skills(tmp_path: Path) -> None:
    """RenderedPrompt.skills collects skills from visible sections."""
    from weakincentives.prompt import PromptTemplate

    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    skill = SkillMount(source=skill_dir, name="collected-skill")

    template = PromptTemplate[None](
        ns="test",
        key="prompt-with-skills",
        sections=(
            MarkdownSection(
                title="Section with Skills",
                template="Content",
                key="section-with-skills",
                skills=[skill],
            ),
        ),
    )

    from weakincentives.prompt import Prompt

    prompt = Prompt(template)
    rendered = prompt.render()

    assert rendered.skills == (skill,)


def test_rendered_prompt_skills_empty_without_skills() -> None:
    """RenderedPrompt.skills is empty when no sections have skills."""
    from weakincentives.prompt import Prompt, PromptTemplate

    template = PromptTemplate[None](
        ns="test",
        key="prompt-no-skills",
        sections=(
            MarkdownSection(
                title="Section without Skills",
                template="Content",
                key="section-no-skills",
            ),
        ),
    )

    prompt = Prompt(template)
    rendered = prompt.render()

    assert rendered.skills == ()


# ============================================================================
# Progressive disclosure tests for skills
# ============================================================================


def test_summarized_section_with_skills_uses_open_sections(tmp_path: Path) -> None:
    """Summarized sections with skills get open_sections instruction (not read_section).

    Skills are treated like tools for progressive disclosure - expanding a section
    with skills requires open_sections to activate them.
    """
    from weakincentives.prompt import Prompt, PromptTemplate, SectionVisibility

    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    skill = SkillMount(source=skill_dir, name="test-skill")

    template = PromptTemplate[None](
        ns="test",
        key="skills-progressive-disclosure",
        sections=(
            MarkdownSection(
                title="Section with Skills",
                template="Full content here.",
                summary="Summarized content.",
                key="section-with-skills",
                visibility=SectionVisibility.SUMMARY,
                skills=[skill],  # Skills but no tools
            ),
        ),
    )

    prompt = Prompt(template)
    rendered = prompt.render()

    # Should use open_sections (not read_section) because section has skills
    assert "open_sections" in rendered.text
    assert "read_section" not in rendered.text


def test_summarized_section_without_tools_or_skills_uses_read_section() -> None:
    """Summarized sections without tools or skills get read_section instruction."""
    from weakincentives.prompt import Prompt, PromptTemplate, SectionVisibility

    template = PromptTemplate[None](
        ns="test",
        key="no-tools-no-skills",
        sections=(
            MarkdownSection(
                title="Plain Section",
                template="Full content here.",
                summary="Summarized content.",
                key="plain-section",
                visibility=SectionVisibility.SUMMARY,
                # No tools and no skills
            ),
        ),
    )

    prompt = Prompt(template)
    rendered = prompt.render()

    # Should use read_section because section has no tools or skills
    assert "read_section" in rendered.text
    assert "open_sections" not in rendered.text
