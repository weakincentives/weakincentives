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

"""Unit tests covering chapter expansion policies."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.prompt import (
    Chapter,
    ChaptersExpansionPolicy,
    MarkdownSection,
    Prompt,
    PromptValidationError,
)
from weakincentives.prompt._normalization import (
    COMPONENT_KEY_PATTERN,
    normalize_component_key,
)


@dataclass
class BaseParams:
    summary: str


@dataclass
class ChapterOneParams:
    detail: str


@dataclass
class ChapterTwoParams:
    detail: str


@dataclass
class ToggleParams:
    include: bool


def build_chapter_prompt() -> Prompt:
    base_section = MarkdownSection[BaseParams](
        title="Summary",
        template="Summary: ${summary}",
        key="summary",
    )
    chapter_one_section = MarkdownSection[ChapterOneParams](
        title="Details",
        template="Details: ${detail}",
        key="chapter-one-detail",
    )
    chapter_two_section = MarkdownSection[ChapterTwoParams](
        title="Extras",
        template="Extras: ${detail}",
        key="chapter-two-detail",
    )
    chapter_one = Chapter[ChapterOneParams](
        key="chapter-one",
        title="Chapter One",
        sections=(chapter_one_section,),
    )
    chapter_two = Chapter[ChapterTwoParams](
        key="chapter-two",
        title="Chapter Two",
        sections=(chapter_two_section,),
    )
    return Prompt(
        ns="tests.prompts",
        key="chapter-prompt",
        sections=(base_section,),
        chapters=(chapter_one, chapter_two),
    )


def test_expand_chapters_all_included_opens_every_chapter() -> None:
    prompt = build_chapter_prompt()

    expanded = prompt.expand_chapters(ChaptersExpansionPolicy.ALL_INCLUDED)

    assert expanded is not prompt
    rendered = expanded.render(
        BaseParams(summary="base"),
        ChapterOneParams(detail="one"),
        ChapterTwoParams(detail="two"),
    )

    assert "Summary: base" in rendered.text
    assert "Details: one" in rendered.text
    assert "Extras: two" in rendered.text


def test_expand_chapters_respects_enabled_predicate() -> None:
    prompt = build_chapter_prompt()
    toggle_section = MarkdownSection[ChapterOneParams](
        title="Toggle",
        template="Toggle: ${detail}",
        key="toggle-detail",
    )
    toggle_chapter = Chapter[ToggleParams](
        key="toggle",
        title="Toggle",
        sections=(toggle_section,),
        enabled=lambda params: params.include,
    )
    prompt_with_toggle = Prompt(
        ns="tests.prompts",
        key="chapter-toggle",
        sections=prompt.chapters[0].sections,
        chapters=(toggle_chapter,),
    )

    expanded = prompt_with_toggle.expand_chapters(
        ChaptersExpansionPolicy.ALL_INCLUDED,
        chapter_params={"toggle": ToggleParams(include=False)},
    )

    rendered = expanded.render(ChapterOneParams(detail="content"))
    assert "Toggle" not in rendered.text

    opened = prompt_with_toggle.expand_chapters(
        ChaptersExpansionPolicy.ALL_INCLUDED,
        chapter_params={"toggle": ToggleParams(include=True)},
    )
    rendered_open = opened.render(ChapterOneParams(detail="content"))
    assert "Toggle: content" in rendered_open.text


def test_expand_chapters_requires_params_when_predicate_present() -> None:
    chapter = Chapter[ToggleParams](
        key="needs-params",
        title="Needs Params",
        sections=(),
        enabled=lambda params: params.include,
    )
    prompt = Prompt(
        ns="tests.prompts",
        key="missing-params",
        sections=(),
        chapters=(chapter,),
    )

    with pytest.raises(PromptValidationError):
        prompt.expand_chapters(ChaptersExpansionPolicy.ALL_INCLUDED)


def test_expand_chapters_rejects_unknown_policy() -> None:
    prompt = build_chapter_prompt()

    with pytest.raises(NotImplementedError):
        prompt.expand_chapters(ChaptersExpansionPolicy.LLM_TOOL)


def test_expand_chapters_rejects_unknown_param_key() -> None:
    prompt = build_chapter_prompt()

    with pytest.raises(PromptValidationError):
        prompt.expand_chapters(
            ChaptersExpansionPolicy.ALL_INCLUDED,
            chapter_params={"missing": ToggleParams(include=True)},
        )


def test_expand_chapters_with_no_chapters_returns_original_prompt() -> None:
    base_prompt = Prompt(
        ns="tests.prompts",
        key="no-chapters",
        sections=(
            MarkdownSection[BaseParams](
                title="Base",
                template="Base: ${summary}",
                key="base",
            ),
        ),
    )

    expanded = base_prompt.expand_chapters(ChaptersExpansionPolicy.ALL_INCLUDED)
    assert expanded is base_prompt


def test_expand_chapters_includes_chapter_without_params() -> None:
    base_section = MarkdownSection[BaseParams](
        title="Base",
        template="Base: ${summary}",
        key="base",
    )
    static_section = MarkdownSection(
        title="Static", template="Static chapter", key="static-section"
    )
    static_chapter = Chapter(
        key="static-chapter",
        title="Static Chapter",
        sections=(static_section,),
    )
    prompt = Prompt(
        ns="tests.prompts",
        key="chapter-paramless",
        sections=(base_section,),
        chapters=(static_chapter,),
    )

    expanded = prompt.expand_chapters(ChaptersExpansionPolicy.ALL_INCLUDED)

    rendered = expanded.render(BaseParams(summary="demo"))

    assert "Static chapter" in rendered.text


def test_expand_chapters_rejects_params_for_paramless_chapter() -> None:
    static_section = MarkdownSection(
        title="Static", template="Static", key="static-section"
    )
    chapter = Chapter(key="static", title="Static", sections=(static_section,))
    prompt = Prompt(
        ns="tests.prompts",
        key="chapter-paramless-params",
        sections=(),
        chapters=(chapter,),
    )

    with pytest.raises(PromptValidationError):
        prompt.expand_chapters(
            ChaptersExpansionPolicy.ALL_INCLUDED,
            chapter_params={"static": BaseParams(summary="noop")},
        )


def test_expand_chapters_accepts_none_for_paramless_chapter() -> None:
    static_section = MarkdownSection(
        title="Static", template="Static", key="static-section"
    )
    chapter = Chapter(key="static", title="Static", sections=(static_section,))
    prompt = Prompt(
        ns="tests.prompts",
        key="chapter-paramless-none",
        sections=(),
        chapters=(chapter,),
    )

    expanded = prompt.expand_chapters(
        ChaptersExpansionPolicy.ALL_INCLUDED,
        chapter_params={"static": None},
    )

    rendered = expanded.render()

    assert "Static" in rendered.text


def test_chapter_without_params_defaults_to_none() -> None:
    chapter = Chapter(key="static", title="Static")

    assert chapter.param_type is None
    assert chapter.sections == ()
    assert chapter.is_enabled(None) is True


def test_chapter_without_params_accepts_parameterless_enabled_callable() -> None:
    chapter = Chapter(key="static", title="Static", enabled=lambda: False)

    assert chapter.is_enabled(None) is False


def test_chapter_without_params_enabled_callable_handles_argument() -> None:
    recorded: list[object | None] = []

    def enabled(value: object | None) -> bool:
        recorded.append(value)
        return value is None

    chapter = Chapter(key="static", title="Static", enabled=enabled)

    assert chapter.is_enabled(None) is True
    assert recorded == [None]


def test_chapter_without_params_enabled_callable_handles_non_inspectable_callable() -> (
    None
):
    chapter = Chapter(key="static", title="Static", enabled=bool)

    assert chapter.is_enabled(None) is False


def test_chapter_without_params_rejects_defaults() -> None:
    with pytest.raises(TypeError):
        Chapter(
            key="invalid", title="Invalid", default_params=ToggleParams(include=True)
        )


def test_chapter_rejects_non_section_payloads() -> None:
    with pytest.raises(TypeError):
        Chapter[ChapterOneParams](
            key="bad",
            title="Bad",
            sections=("not-a-section",),  # type: ignore[arg-type]
        )


def test_chapter_validates_default_param_type() -> None:
    with pytest.raises(TypeError):
        Chapter[ChapterOneParams](
            key="mismatch",
            title="Mismatch",
            default_params=ToggleParams(include=True),  # type: ignore[arg-type]
        )


def test_chapter_is_enabled_defaults_to_true() -> None:
    chapter = Chapter[ChapterOneParams](key="noop", title="Noop")
    assert chapter.is_enabled(None)


def test_chapter_is_enabled_requires_params_when_predicate_defined() -> None:
    chapter = Chapter[ToggleParams](
        key="needs-params",
        title="Needs Params",
        enabled=lambda params: params.include,
    )
    with pytest.raises(TypeError):
        chapter.is_enabled(None)


def test_chapter_key_normalization_enforces_format() -> None:
    with pytest.raises(ValueError) as excinfo:
        Chapter[ChapterOneParams]._normalize_key("")

    assert str(excinfo.value) == "Chapter key must be a non-empty string."

    with pytest.raises(ValueError) as excinfo:
        normalize_component_key("", owner="Chapter")

    assert str(excinfo.value) == "Chapter key must be a non-empty string."

    with pytest.raises(ValueError) as excinfo:
        Chapter[ChapterOneParams]._normalize_key("Invalid Key")

    expected_message = f"Chapter key must match {COMPONENT_KEY_PATTERN.pattern}."
    assert str(excinfo.value) == expected_message

    with pytest.raises(ValueError) as excinfo:
        normalize_component_key("Invalid Key", owner="Chapter")

    assert str(excinfo.value) == expected_message


def test_chapter_rejects_tuple_generic_arguments() -> None:
    with pytest.raises(TypeError):
        bad_args = (ChapterOneParams, ToggleParams)
        Chapter[bad_args]  # type: ignore[index]


def test_prompt_rejects_non_chapter_instances() -> None:
    with pytest.raises(PromptValidationError):
        Prompt(
            ns="tests.prompts",
            key="invalid-chapter",
            sections=(),
            chapters=(object(),),  # type: ignore[arg-type]
        )


def test_prompt_rejects_duplicate_chapter_keys() -> None:
    chapter = Chapter[ChapterOneParams](
        key="duplicate",
        title="Duplicate",
    )
    with pytest.raises(PromptValidationError):
        Prompt(
            ns="tests.prompts",
            key="duplicate-chapters",
            sections=(),
            chapters=(chapter, chapter),
        )


def test_expand_chapters_uses_default_params_when_missing() -> None:
    chapter = Chapter[ToggleParams](
        key="defaults",
        title="Defaults",
        sections=(
            MarkdownSection[ChapterOneParams](
                title="Inner",
                template="Inner: ${detail}",
                key="inner",
            ),
        ),
        default_params=ToggleParams(include=True),
        enabled=lambda params: params.include,
    )
    prompt = Prompt(
        ns="tests.prompts",
        key="use-defaults",
        sections=(),
        chapters=(chapter,),
    )

    expanded = prompt.expand_chapters(ChaptersExpansionPolicy.ALL_INCLUDED)
    rendered = expanded.render(ChapterOneParams(detail="value"))
    assert "Inner: value" in rendered.text


def test_expand_chapters_wraps_enabled_errors() -> None:
    chapter = Chapter[ToggleParams](
        key="error-chapter",
        title="Errors",
        sections=(),
        enabled=lambda params: (_ for _ in ()).throw(RuntimeError("boom")),
        default_params=ToggleParams(include=True),
    )
    prompt = Prompt(
        ns="tests.prompts",
        key="error-prompt",
        sections=(),
        chapters=(chapter,),
    )

    with pytest.raises(PromptValidationError):
        prompt.expand_chapters(ChaptersExpansionPolicy.ALL_INCLUDED)


def test_expand_chapters_validates_parameter_types() -> None:
    chapter = Chapter[ChapterOneParams](
        key="type-guard",
        title="Type Guard",
        sections=(),
    )
    prompt = Prompt(
        ns="tests.prompts",
        key="type-prompt",
        sections=(),
        chapters=(chapter,),
    )

    with pytest.raises(PromptValidationError):
        prompt.expand_chapters(
            ChaptersExpansionPolicy.ALL_INCLUDED,
            chapter_params={"type-guard": ToggleParams(include=True)},
        )
