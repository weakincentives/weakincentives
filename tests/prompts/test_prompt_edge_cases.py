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

from dataclasses import dataclass
from typing import Any, cast

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    PromptRenderError,
    PromptTemplate,
    PromptValidationError,
    Section,
)


@dataclass
class RegisteredParams:
    value: str


@dataclass
class UnregisteredParams:
    value: str


def test_prompt_render_rejects_unregistered_params_type() -> None:
    section = MarkdownSection[RegisteredParams](
        title="Registered",
        template="Registered: ${value}",
        key="registered",
    )
    prompt = PromptTemplate(
        ns="tests/prompts",
        key="edge-unregistered",
        sections=[section],
    )

    with pytest.raises(PromptValidationError) as exc:
        prompt.render(UnregisteredParams(value="bad"))

    error = cast(PromptValidationError, exc.value)
    assert error.dataclass_type is UnregisteredParams


@dataclass
class NullConstructedParams:
    value: str = "unused"

    def __new__(
        cls,
        *args: object,
        **kwargs: object,
    ) -> NullConstructedParams | None:
        return None


@dataclass
class InvalidConstructedParams:
    value: str = "unused"

    def __new__(
        cls,
        *args: object,
        **kwargs: object,
    ) -> InvalidConstructedParams | object:
        return {"value": "not dataclass"}


def test_prompt_render_detects_constructor_returning_none() -> None:
    section = MarkdownSection[NullConstructedParams](
        title="Null",
        template="Null body",
        key="null",
    )
    prompt = PromptTemplate(
        ns="tests/prompts",
        key="edge-constructor-none",
        sections=[section],
    )

    with pytest.raises(PromptRenderError) as exc:
        prompt.render()

    error = cast(PromptRenderError, exc.value)
    assert error.section_path == ("null",)
    assert error.dataclass_type is NullConstructedParams


def test_prompt_render_detects_constructor_returning_non_dataclass() -> None:
    section = MarkdownSection[InvalidConstructedParams](
        title="Invalid",
        template="Invalid body",
        key="invalid",
    )
    prompt = PromptTemplate(
        ns="tests.prompts",
        key="edge-constructor-invalid",
        sections=[section],
    )

    with pytest.raises(PromptRenderError) as exc:
        prompt.render()

    error = cast(PromptRenderError, exc.value)
    assert error.section_path == ("invalid",)
    assert error.dataclass_type is InvalidConstructedParams


@dataclass
class BrokenParams:
    value: str


class BrokenSection(Section[BrokenParams]):
    def __init__(self) -> None:
        super().__init__(title="Broken", key="broken")

    def render(self, params: BrokenParams, depth: int, number: str) -> str:
        del params, depth, number
        raise PromptRenderError("inner", section_path=(self.key,), placeholder="value")

    def clone(self, **kwargs: object) -> BrokenSection:
        del kwargs
        _ = self.title
        return BrokenSection()


def test_prompt_render_wraps_prompt_errors_with_context() -> None:
    section = BrokenSection()
    prompt = PromptTemplate(
        ns="tests/prompts",
        key="edge-wrap-error",
        sections=[section],
    )

    with pytest.raises(PromptRenderError) as exc:
        prompt.render(BrokenParams(value="x"))

    error = cast(PromptRenderError, exc.value)
    assert error.section_path == ("broken",)
    assert error.dataclass_type is BrokenParams
    assert error.placeholder == "value"


class InvalidParamsSection(Section[int]):  # type: ignore[arg-type]
    def __init__(self) -> None:
        super().__init__(title="Invalid", key="invalid")

    def render(self, params: int, depth: int, number: str) -> str:
        del params, depth, number
        return self.key

    def clone(self, **kwargs: object) -> InvalidParamsSection:
        del kwargs
        _ = self.title
        return InvalidParamsSection()


def test_invalid_params_section_render_stub() -> None:
    section = InvalidParamsSection()
    assert section.render(0, depth=1, number="1") == "invalid"


def test_prompt_register_requires_dataclass_params() -> None:
    section = InvalidParamsSection()

    with pytest.raises(PromptValidationError) as exc:
        PromptTemplate(
            ns="tests/prompts",
            key="edge-dataclass-required",
            sections=[section],
        )

    error = cast(PromptValidationError, exc.value)
    assert error.dataclass_type is int
    assert error.section_path == ("invalid",)


@dataclass
class DefaultsParams:
    value: str


def test_prompt_register_validates_defaults_type() -> None:
    section = MarkdownSection[DefaultsParams](
        title="Defaults",
        template="Defaults",
        default_params=cast(Any, DefaultsParams),
        key="defaults",
    )

    with pytest.raises(PromptValidationError) as exc:
        PromptTemplate(
            ns="tests/prompts",
            key="edge-defaults-type",
            sections=[section],
        )

    error = cast(PromptValidationError, exc.value)
    assert error.dataclass_type is DefaultsParams
    assert error.section_path == ("defaults",)


@dataclass
class DefaultsMismatchParams:
    value: str


@dataclass
class OtherParams:
    value: str


def test_prompt_register_requires_defaults_type_match() -> None:
    section = MarkdownSection[DefaultsMismatchParams](
        title="Mismatch",
        template="Mismatch",
        default_params=cast(Any, OtherParams(value="x")),
        key="mismatch",
    )

    with pytest.raises(PromptValidationError) as exc:
        PromptTemplate(
            ns="tests/prompts",
            key="edge-defaults-mismatch",
            sections=[section],
        )

    error = cast(PromptValidationError, exc.value)
    assert error.dataclass_type is DefaultsMismatchParams
    assert error.section_path == ("mismatch",)


@dataclass
class PlaceholderParams:
    value: str


class BareSection(Section[PlaceholderParams]):
    def __init__(self) -> None:
        super().__init__(title="Bare", key="bare")

    def render(self, params: PlaceholderParams, depth: int, number: str) -> str:
        del params, depth, number
        return self.key

    def clone(self, **kwargs: object) -> BareSection:
        del kwargs
        _ = self.title
        return BareSection()


def test_section_placeholder_names_default_to_empty_set() -> None:
    section = BareSection()

    assert section.placeholder_names() == set()


@dataclass
class HeadingOnlyParams:
    pass


def test_text_section_returns_heading_when_body_empty() -> None:
    section = MarkdownSection[HeadingOnlyParams](
        title="Heading",
        template="\n",
        key="heading",
    )

    output = section.render(HeadingOnlyParams(), depth=0, number="1")

    assert output == "## 1. Heading"


@dataclass
class PlaceholderNamesParams:
    value: str = "v"
    other: str = "o"


def test_text_section_placeholder_names_cover_named_and_braced() -> None:
    section = MarkdownSection[PlaceholderNamesParams](
        title="Placeholders",
        template="Value: $value and ${other}",
        key="placeholders",
    )

    assert section.placeholder_names() == {"value", "other"}


@dataclass
class ContextParams:
    value: str


class ContextAwareSection(Section[ContextParams]):
    def __init__(self) -> None:
        super().__init__(title="Context", key="context")

    def render(self, params: ContextParams, depth: int, number: str) -> str:
        _ = self.title
        del params, depth, number
        raise PromptRenderError(
            "context",
            section_path=("Provided",),
            dataclass_type=ContextParams,
            placeholder="kept",
        )

    def clone(self, **kwargs: object) -> ContextAwareSection:
        del kwargs
        _ = self.key
        return ContextAwareSection()


def test_prompt_render_propagates_errors_with_existing_context() -> None:
    section = ContextAwareSection()
    prompt = PromptTemplate(
        ns="tests/prompts",
        key="edge-preserve-context",
        sections=[section],
    )

    with pytest.raises(PromptRenderError) as exc:
        prompt.render(ContextParams(value="x"))

    error = cast(PromptRenderError, exc.value)
    assert error.section_path == ("Provided",)
    assert error.dataclass_type is ContextParams
    assert error.placeholder == "kept"
