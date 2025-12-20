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
from typing import cast

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    PromptValidationError,
    Section,
    Tool,
)
from weakincentives.prompt.overrides import ToolOverride, hash_text
from weakincentives.prompt.registry import PromptRegistry
from weakincentives.prompt.rendering import PromptRenderer
from weakincentives.types import SupportsDataclass


@dataclass
class _IntroParams:
    title: str


@dataclass
class _ToolParams:
    query: str


@dataclass
class _ToolResult:
    answer: str


def _build_registry() -> tuple[PromptRegistry, MarkdownSection[_IntroParams]]:
    section = MarkdownSection[_IntroParams](
        title="Intro",
        template="Intro: ${title}",
        key="intro",
        tools=[
            Tool[_ToolParams, _ToolResult](
                name="search",
                description="Search for context.",
                handler=None,
            )
        ],
    )
    registry = PromptRegistry()
    registry.register_section(
        cast(Section[SupportsDataclass], section), path=(section.key,), depth=0
    )
    return registry, section


def test_registry_snapshot_clones_defaults() -> None:
    @dataclass
    class _DefaultParams:
        body: str

    section = MarkdownSection[_DefaultParams](
        title="Body",
        template="Body: ${body}",
        key="body",
        default_params=_DefaultParams(body="seed"),
    )
    registry = PromptRegistry()
    registry.register_section(
        cast(Section[SupportsDataclass], section), path=(section.key,), depth=0
    )

    snapshot = registry.snapshot()
    node = snapshot.sections[0]

    first = snapshot.resolve_section_params(node, {})
    assert isinstance(first, _DefaultParams)
    first.body = "mutated"

    second = snapshot.resolve_section_params(node, {})
    assert isinstance(second, _DefaultParams)

    assert second.body == "seed"
    assert second is not first


def test_registry_validates_placeholders() -> None:
    @dataclass
    class _MismatchParams:
        detail: str

    registry = PromptRegistry()
    section = MarkdownSection[_MismatchParams](
        title="Mismatch",
        template="Missing ${unknown}",
        key="mismatch",
    )

    with pytest.raises(PromptValidationError) as exc:
        registry.register_section(
            cast(Section[SupportsDataclass], section),
            path=(section.key,),
            depth=0,
        )

    assert isinstance(exc.value, PromptValidationError)
    assert exc.value.placeholder == "unknown"
    assert exc.value.section_path == ("mismatch",)


def test_renderer_renders_sections_and_tool_overrides() -> None:
    registry, section = _build_registry()
    snapshot = registry.snapshot()
    renderer = PromptRenderer(
        registry=snapshot,
        structured_output=None,
    )

    params_lookup = renderer.build_param_lookup((_IntroParams(title="Hello"),))

    rendered = renderer.render(
        params_lookup,
        overrides={(section.key,): "${title}!"},
        tool_overrides={
            "search": ToolOverride(
                name="search",
                expected_contract_hash=hash_text("hash"),
                description="Updated description.",
                param_descriptions={"query": "New query."},
            )
        },
    )

    assert rendered.text.endswith("Hello!")
    assert rendered.tools[0].description == "Updated description."
    assert rendered.tool_param_descriptions == {"search": {"query": "New query."}}


def test_renderer_tool_override_with_no_description_change() -> None:
    registry, section = _build_registry()
    snapshot = registry.snapshot()
    renderer = PromptRenderer(
        registry=snapshot,
        structured_output=None,
    )

    params_lookup = renderer.build_param_lookup((_IntroParams(title="Test"),))

    original_description = snapshot.sections[0].section.tools()[0].description

    rendered = renderer.render(
        params_lookup,
        tool_overrides={
            "search": ToolOverride(
                name="search",
                expected_contract_hash=hash_text("hash"),
                description=original_description,
                param_descriptions={},
            )
        },
    )

    assert rendered.tools[0].description == original_description
    assert rendered.tool_param_descriptions == {}


def test_renderer_rejects_unregistered_params_types() -> None:
    registry, _ = _build_registry()
    snapshot = registry.snapshot()
    renderer = PromptRenderer(
        registry=snapshot,
        structured_output=None,
    )

    @dataclass
    class _Other:
        value: str

    with pytest.raises(PromptValidationError):
        renderer.build_param_lookup((_Other(value="x"),))
