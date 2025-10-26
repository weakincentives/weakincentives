from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from weakincentives.prompts import (
    Prompt,
    PromptRenderError,
    PromptValidationError,
    Section,
    TextSection,
)


@dataclass
class RegisteredParams:
    value: str


@dataclass
class UnregisteredParams:
    value: str


def test_prompt_render_rejects_unregistered_params_type() -> None:
    section = TextSection(
        title="Registered",
        body="Registered: ${value}",
        params=RegisteredParams,
    )
    prompt = Prompt(sections=[section])

    with pytest.raises(PromptValidationError) as exc:
        prompt.render(UnregisteredParams(value="bad"))

    error = cast(PromptValidationError, exc.value)
    assert error.dataclass_type is UnregisteredParams


@dataclass
class NullConstructedParams:
    value: str = "unused"

    def __new__(cls, *args, **kwargs):
        return None


def test_prompt_render_detects_constructor_returning_none() -> None:
    section = TextSection(
        title="Null",
        body="Null body",
        params=NullConstructedParams,
    )
    prompt = Prompt(sections=[section])

    with pytest.raises(PromptRenderError) as exc:
        prompt.render()

    error = cast(PromptRenderError, exc.value)
    assert error.section_path == ("Null",)
    assert error.dataclass_type is NullConstructedParams


@dataclass
class BrokenParams:
    value: str


class BrokenSection(Section[BrokenParams]):
    def __init__(self) -> None:
        super().__init__(title="Broken", params=BrokenParams)

    def render(self, params: BrokenParams, depth: int) -> str:
        raise PromptRenderError("inner", placeholder="value")


def test_prompt_render_wraps_prompt_errors_with_context() -> None:
    section = BrokenSection()
    prompt = Prompt(sections=[section])

    with pytest.raises(PromptRenderError) as exc:
        prompt.render(BrokenParams(value="x"))

    error = cast(PromptRenderError, exc.value)
    assert error.section_path == ("Broken",)
    assert error.dataclass_type is BrokenParams
    assert error.placeholder == "value"


class InvalidParamsSection(Section[int]):
    def __init__(self) -> None:
        super().__init__(title="Invalid", params=int)

    def render(self, params: int, depth: int) -> str:  # pragma: no cover - defensive
        return "invalid"


def test_prompt_register_requires_dataclass_params() -> None:
    section = InvalidParamsSection()

    with pytest.raises(PromptValidationError) as exc:
        Prompt(sections=[section])

    error = cast(PromptValidationError, exc.value)
    assert error.dataclass_type is int
    assert error.section_path == ("Invalid",)


@dataclass
class DefaultsParams:
    value: str


def test_prompt_register_validates_defaults_type() -> None:
    section = TextSection(
        title="Defaults",
        body="Defaults",
        params=DefaultsParams,
        defaults=DefaultsParams,
    )

    with pytest.raises(PromptValidationError) as exc:
        Prompt(sections=[section])

    error = cast(PromptValidationError, exc.value)
    assert error.dataclass_type is DefaultsParams
    assert error.section_path == ("Defaults",)


@dataclass
class DefaultsMismatchParams:
    value: str


@dataclass
class OtherParams:
    value: str


def test_prompt_register_requires_defaults_type_match() -> None:
    section = TextSection(
        title="Mismatch",
        body="Mismatch",
        params=DefaultsMismatchParams,
        defaults=OtherParams(value="x"),
    )

    with pytest.raises(PromptValidationError) as exc:
        Prompt(sections=[section])

    error = cast(PromptValidationError, exc.value)
    assert error.dataclass_type is DefaultsMismatchParams
    assert error.section_path == ("Mismatch",)


@dataclass
class PlaceholderParams:
    value: str


class BareSection(Section[PlaceholderParams]):
    def __init__(self) -> None:
        super().__init__(title="Bare", params=PlaceholderParams)

    def render(self, params: PlaceholderParams, depth: int) -> str:
        return "bare"


def test_section_placeholder_names_default_to_empty_set() -> None:
    section = BareSection()

    assert section.placeholder_names() == set()


@dataclass
class HeadingOnlyParams:
    pass


def test_text_section_returns_heading_when_body_empty() -> None:
    section = TextSection(
        title="Heading",
        body="\n",
        params=HeadingOnlyParams,
    )

    output = section.render(HeadingOnlyParams(), depth=0)

    assert output == "## Heading"


@dataclass
class PlaceholderNamesParams:
    value: str = "v"
    other: str = "o"


def test_text_section_placeholder_names_cover_named_and_braced() -> None:
    section = TextSection(
        title="Placeholders",
        body="Value: $value and ${other}",
        params=PlaceholderNamesParams,
    )

    assert section.placeholder_names() == {"value", "other"}


@dataclass
class ContextParams:
    value: str


class ContextAwareSection(Section[ContextParams]):
    def __init__(self) -> None:
        super().__init__(title="Context", params=ContextParams)

    def render(self, params: ContextParams, depth: int) -> str:
        raise PromptRenderError(
            "context",
            section_path=("Provided",),
            dataclass_type=ContextParams,
            placeholder="kept",
        )


def test_prompt_render_propagates_errors_with_existing_context() -> None:
    section = ContextAwareSection()
    prompt = Prompt(sections=[section])

    with pytest.raises(PromptRenderError) as exc:
        prompt.render(ContextParams(value="x"))

    error = cast(PromptRenderError, exc.value)
    assert error.section_path == ("Provided",)
    assert error.dataclass_type is ContextParams
    assert error.placeholder == "kept"
