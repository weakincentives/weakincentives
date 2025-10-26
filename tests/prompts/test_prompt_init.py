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


def test_prompt_initialization_flattens_sections_depth_first():
    child = TextSection(
        title="Child",
        body="Child: ${detail}",
        params=ChildParams,
    )
    sibling = TextSection(
        title="Sibling",
        body="Sibling: ${note}",
        params=SiblingParams,
    )
    root = TextSection(
        title="Root",
        body="Root: ${title}",
        params=RootParams,
        children=[child, sibling],
    )

    prompt = Prompt(sections=[root])

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


def test_prompt_rejects_duplicate_param_dataclasses():
    @dataclass
    class DuplicateParams:
        value: str

    first = TextSection(
        title="First",
        body="First: ${value}",
        params=DuplicateParams,
    )
    second = TextSection(
        title="Second",
        body="Second: ${value}",
        params=DuplicateParams,
    )

    with pytest.raises(PromptValidationError) as exc:
        Prompt(sections=[first, second])

    assert isinstance(exc.value, PromptValidationError)
    assert exc.value.dataclass_type is DuplicateParams
    assert exc.value.section_path == ("Second",)


def test_prompt_validates_text_section_placeholders():
    @dataclass
    class PlaceholderParams:
        value: str

    section = TextSection(
        title="Invalid",
        body="Missing ${oops}",
        params=PlaceholderParams,
    )

    with pytest.raises(PromptValidationError) as exc:
        Prompt(sections=[section])

    assert isinstance(exc.value, PromptValidationError)
    assert exc.value.placeholder == "oops"
    assert exc.value.section_path == ("Invalid",)
    assert exc.value.dataclass_type is PlaceholderParams
