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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from weakincentives.prompt.overrides.validation import (
    FORMAT_VERSION,
    load_sections,
    load_tools,
    seed_sections,
    seed_tools,
    serialize_sections,
    serialize_tools,
    validate_header,
    validate_sections_for_write,
    validate_tools_for_write,
)
from weakincentives.prompt.overrides.versioning import (
    PromptDescriptor,
    PromptOverridesError,
    SectionDescriptor,
    SectionOverride,
    ToolDescriptor,
    ToolOverride,
)
from weakincentives.prompt.tool import Tool


@dataclass(slots=True)
class _Params:
    value: int = field(metadata={"description": "Value"})


@dataclass(slots=True)
class _Result:
    message: str


@dataclass(slots=True)
class _Section:
    template: str
    accepts_overrides: bool = True
    _tools: tuple[Tool[Any, Any], ...] = ()

    def original_body_template(self) -> str | None:
        return self.template

    def tools(self) -> tuple[Tool[Any, Any], ...]:
        return self._tools


@dataclass(slots=True)
class _SectionNode:
    path: tuple[str, ...]
    section: _Section


@dataclass(slots=True)
class _Prompt:
    ns: str
    key: str
    _section_nodes: tuple[_SectionNode, ...]

    @property
    def sections(self) -> tuple[_SectionNode, ...]:
        return self._section_nodes


def _build_prompt_with_tool() -> tuple[
    _Prompt, PromptDescriptor, Tool[_Params, _Result]
]:
    tool = Tool[_Params, _Result](
        name="demo_tool",
        description="Example tool",
        handler=None,
    )
    section = _Section(
        template="Body template",
        _tools=(tool,),
    )
    node = _SectionNode(path=("intro",), section=section)
    prompt = _Prompt(ns="demo", key="example", _section_nodes=(node,))
    descriptor = PromptDescriptor.from_prompt(prompt)
    return prompt, descriptor, tool


def test_validate_header_accepts_matching_metadata(tmp_path: Path) -> None:
    prompt, descriptor, _ = _build_prompt_with_tool()
    tag = "latest"
    payload = {
        "version": FORMAT_VERSION,
        "ns": prompt.ns,
        "prompt_key": prompt.key,
        "tag": tag,
    }

    validate_header(payload, descriptor, tag, tmp_path / "override.json")


def test_validate_header_rejects_version_mismatch(tmp_path: Path) -> None:
    prompt, descriptor, _ = _build_prompt_with_tool()
    tag = "latest"
    payload = {
        "version": 99,
        "ns": prompt.ns,
        "prompt_key": prompt.key,
        "tag": tag,
    }

    with pytest.raises(PromptOverridesError):
        validate_header(payload, descriptor, tag, tmp_path / "override.json")


def test_load_sections_filters_unknown_entries() -> None:
    descriptor = PromptDescriptor(
        ns="demo",
        key="example",
        sections=[SectionDescriptor(path=("intro",), content_hash="abc123")],
        tools=[],
    )
    payload = {
        "intro": {
            "expected_hash": "abc123",
            "body": "Body",
        },
        "unknown": {
            "expected_hash": "zzz",
            "body": "Ignored",
        },
    }

    overrides = load_sections(payload, descriptor)

    assert list(overrides) == [("intro",)]


def test_load_tools_filters_unknown_entries() -> None:
    descriptor = PromptDescriptor(
        ns="demo",
        key="example",
        sections=[],
        tools=[
            ToolDescriptor(
                path=("intro",),
                name="demo_tool",
                contract_hash="toolhash",
            )
        ],
    )
    payload = {
        "demo_tool": {
            "expected_contract_hash": "toolhash",
            "description": "Updated",
            "param_descriptions": {"value": "Value"},
        },
        "unknown_tool": {
            "expected_contract_hash": "zzz",
        },
    }

    overrides = load_tools(payload, descriptor)

    assert list(overrides) == ["demo_tool"]


def test_validate_sections_for_write_rejects_unknown_path() -> None:
    descriptor = PromptDescriptor(
        ns="demo",
        key="example",
        sections=[SectionDescriptor(path=("intro",), content_hash="abc123")],
        tools=[],
    )
    overrides = {("unknown",): SectionOverride(expected_hash="abc123", body="Body")}

    with pytest.raises(PromptOverridesError):
        validate_sections_for_write(overrides, descriptor)


def test_validate_tools_for_write_rejects_unknown_tool() -> None:
    descriptor = PromptDescriptor(
        ns="demo",
        key="example",
        sections=[],
        tools=[
            ToolDescriptor(
                path=("intro",),
                name="demo_tool",
                contract_hash="toolhash",
            )
        ],
    )
    overrides = {
        "other": ToolOverride(
            name="other",
            expected_contract_hash="toolhash",
        )
    }

    with pytest.raises(PromptOverridesError):
        validate_tools_for_write(overrides, descriptor)


def test_serialization_round_trip_for_sections() -> None:
    overrides = {("intro",): SectionOverride(expected_hash="abc123", body="Body")}

    payload = serialize_sections(overrides)
    restored = load_sections(
        payload,
        PromptDescriptor(
            "demo", "example", [SectionDescriptor(("intro",), "abc123")], []
        ),
    )

    assert restored == overrides


def test_serialization_round_trip_for_tools() -> None:
    overrides = {
        "demo_tool": ToolOverride(
            name="demo_tool",
            expected_contract_hash="toolhash",
            description="Updated",
            param_descriptions={"value": "Value"},
        )
    }
    descriptor = PromptDescriptor(
        ns="demo",
        key="example",
        sections=[],
        tools=[
            ToolDescriptor(
                path=("intro",),
                name="demo_tool",
                contract_hash="toolhash",
            )
        ],
    )

    payload = serialize_tools(overrides)
    restored = load_tools(payload, descriptor)

    assert restored == overrides


def test_seed_sections_and_tools_generate_overrides() -> None:
    prompt, descriptor, tool = _build_prompt_with_tool()

    sections = seed_sections(prompt, descriptor)
    tools = seed_tools(prompt, descriptor)

    assert sections["intro",].body == "Body template"
    assert tools[tool.name].param_descriptions == {"value": "Value"}
