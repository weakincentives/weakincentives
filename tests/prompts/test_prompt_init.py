from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.prompts import Prompt, PromptValidationError, TextSection


@dataclass
class RootParams:
    title: str


@dataclass
class ChildParams:
    detail: str


@dataclass
class SiblingParams:
    note: str


@dataclass
class DuplicateParams:
    value: str


def test_prompt_initialization_flattens_sections_depth_first():
    child = TextSection[ChildParams](
        title="Child",
        body="Child: ${detail}",
    )
    sibling = TextSection[SiblingParams](
        title="Sibling",
        body="Sibling: ${note}",
    )
    root = TextSection[RootParams](
        title="Root",
        body="Root: ${title}",
        children=[child, sibling],
    )

    prompt = Prompt(name="demo", sections=[root])

    titles = [node.section.title for node in prompt.sections]
    depths = [node.depth for node in prompt.sections]
    paths = [node.path for node in prompt.sections]

    assert titles == ["Root", "Child", "Sibling"]
    assert depths == [0, 1, 1]
    assert paths == [
        ("Root",),
        ("Root", "Child"),
        ("Root", "Sibling"),
    ]
    assert prompt.params_types == {RootParams, ChildParams, SiblingParams}
    assert prompt.name == "demo"


def test_prompt_allows_duplicate_param_dataclasses_and_shares_params():
    first = TextSection[DuplicateParams](
        title="First",
        body="First: ${value}",
        defaults=DuplicateParams(value="alpha"),
    )
    second = TextSection[DuplicateParams](
        title="Second",
        body="Second: ${value}",
        defaults=DuplicateParams(value="beta"),
    )

    prompt = Prompt(sections=[first, second])

    rendered = prompt.render()

    assert "First: alpha" in rendered.text
    assert "Second: beta" in rendered.text
    assert prompt.params_types == {DuplicateParams}


def test_prompt_reuses_provided_params_for_duplicate_sections():
    first = TextSection[DuplicateParams](
        title="First",
        body="First: ${value}",
    )
    second = TextSection[DuplicateParams](
        title="Second",
        body="Second: ${value}",
    )

    prompt = Prompt(sections=[first, second])

    rendered = prompt.render(DuplicateParams(value="shared"))

    assert "First: shared" in rendered.text
    assert "Second: shared" in rendered.text


def test_prompt_duplicate_sections_share_type_defaults_when_missing_section_default():
    first = TextSection[DuplicateParams](
        title="First",
        body="First: ${value}",
        defaults=DuplicateParams(value="alpha"),
    )
    second = TextSection[DuplicateParams](
        title="Second",
        body="Second: ${value}",
    )

    prompt = Prompt(sections=[first, second])

    rendered = prompt.render()

    assert "First: alpha" in rendered.text
    assert "Second: alpha" in rendered.text


def test_prompt_validates_text_section_placeholders():
    @dataclass
    class PlaceholderParams:
        value: str

    section = TextSection[PlaceholderParams](
        title="Invalid",
        body="Missing ${oops}",
    )

    with pytest.raises(PromptValidationError) as exc:
        Prompt(sections=[section])

    assert isinstance(exc.value, PromptValidationError)
    assert exc.value.placeholder == "oops"
    assert exc.value.section_path == ("Invalid",)
    assert exc.value.dataclass_type is PlaceholderParams


def test_text_section_rejects_non_section_children():
    @dataclass
    class ParentParams:
        value: str

    with pytest.raises(TypeError) as exc:
        TextSection[ParentParams](
            title="Parent",
            body="${value}",
            children=["not a section"],  # type: ignore[arg-type]
        )

    assert "Section instances" in str(exc.value)
