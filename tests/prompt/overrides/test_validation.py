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
from typing import TYPE_CHECKING, Literal, cast

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
    HexDigest,
    PromptDescriptor,
    PromptOverridesError,
    SectionDescriptor,
    SectionOverride,
    ToolDescriptor,
    ToolOverride,
    _tool_contract_hash,
)
from weakincentives.prompt.tool import Tool

if TYPE_CHECKING:
    from pathlib import Path

    from weakincentives.prompt import SupportsDataclassOrNone, SupportsToolResult
    from weakincentives.types import JSONValue

_NONE_TYPE = type(None)


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
    _tools: tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...] = ()

    def original_body_template(self) -> str | None:
        return self.template

    def tools(self) -> tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...]:
        return self._tools


@dataclass(slots=True)
class _SectionNode:
    path: tuple[str, ...]
    number: str
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
        _tools=(cast("Tool[SupportsDataclassOrNone, SupportsToolResult]", tool),),
    )
    node = _SectionNode(path=("intro",), number="1", section=section)
    prompt = _Prompt(ns="demo", key="example", _section_nodes=(node,))
    descriptor = PromptDescriptor.from_prompt(prompt)
    return prompt, descriptor, tool


VALID_DIGEST = HexDigest("a" * 64)
OTHER_DIGEST = HexDigest("b" * 64)


def test_hex_digest_rejects_non_strings() -> None:
    with pytest.raises(TypeError):
        HexDigest(cast("object", 123))


def test_validate_header_accepts_matching_metadata(tmp_path: Path) -> None:
    prompt, descriptor, _ = _build_prompt_with_tool()
    tag = "latest"
    payload: dict[str, JSONValue] = {
        "version": FORMAT_VERSION,
        "ns": prompt.ns,
        "prompt_key": prompt.key,
        "tag": tag,
    }

    validate_header(payload, descriptor, tag, tmp_path / "override.json")


def test_validate_header_rejects_version_mismatch(tmp_path: Path) -> None:
    prompt, descriptor, _ = _build_prompt_with_tool()
    tag = "latest"
    payload: dict[str, JSONValue] = {
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
        sections=[
            SectionDescriptor(path=("intro",), content_hash=VALID_DIGEST, number="1")
        ],
        tools=[],
    )
    payload: dict[str, JSONValue] = {
        "intro": {
            "expected_hash": str(VALID_DIGEST),
            "body": "Body",
        },
        "unknown": {
            "expected_hash": str(OTHER_DIGEST),
            "body": "Ignored",
        },
    }

    overrides = load_sections(payload, descriptor)

    assert list(overrides) == [("intro",)]


def test_load_sections_rejects_invalid_hash_format() -> None:
    descriptor = PromptDescriptor(
        ns="demo",
        key="example",
        sections=[
            SectionDescriptor(path=("intro",), content_hash=VALID_DIGEST, number="1")
        ],
        tools=[],
    )
    payload: dict[str, JSONValue] = {
        "intro": {
            "expected_hash": "deadbeef",
            "body": "Body",
        }
    }

    with pytest.raises(PromptOverridesError):
        load_sections(payload, descriptor)


def test_load_tools_filters_unknown_entries() -> None:
    descriptor = PromptDescriptor(
        ns="demo",
        key="example",
        sections=[],
        tools=[
            ToolDescriptor(
                path=("intro",),
                name="demo_tool",
                contract_hash=VALID_DIGEST,
            )
        ],
    )
    payload: dict[str, JSONValue] = {
        "demo_tool": {
            "expected_contract_hash": str(VALID_DIGEST),
            "description": "Updated",
            "param_descriptions": {"value": "Value"},
        },
        "unknown_tool": {
            "expected_contract_hash": str(OTHER_DIGEST),
        },
    }

    overrides = load_tools(payload, descriptor)

    assert list(overrides) == ["demo_tool"]


def test_load_tools_rejects_invalid_hash_format() -> None:
    descriptor = PromptDescriptor(
        ns="demo",
        key="example",
        sections=[],
        tools=[
            ToolDescriptor(
                path=("intro",),
                name="demo_tool",
                contract_hash=VALID_DIGEST,
            )
        ],
    )
    payload: dict[str, JSONValue] = {
        "demo_tool": {
            "expected_contract_hash": "deadbeef",
        }
    }

    with pytest.raises(PromptOverridesError):
        load_tools(payload, descriptor)


def test_validate_sections_for_write_rejects_unknown_path() -> None:
    descriptor = PromptDescriptor(
        ns="demo",
        key="example",
        sections=[
            SectionDescriptor(path=("intro",), content_hash=VALID_DIGEST, number="1")
        ],
        tools=[],
    )
    overrides = {("unknown",): SectionOverride(expected_hash=VALID_DIGEST, body="Body")}

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
                contract_hash=VALID_DIGEST,
            )
        ],
    )
    overrides = {
        "other": ToolOverride(
            name="other",
            expected_contract_hash=VALID_DIGEST,
        )
    }

    with pytest.raises(PromptOverridesError):
        validate_tools_for_write(overrides, descriptor)


def test_serialization_round_trip_for_sections() -> None:
    overrides = {("intro",): SectionOverride(expected_hash=VALID_DIGEST, body="Body")}

    payload = serialize_sections(overrides)
    restored = load_sections(
        payload,
        PromptDescriptor(
            "demo",
            "example",
            [SectionDescriptor(("intro",), VALID_DIGEST, "1")],
            [],
        ),
    )

    assert restored == overrides


def test_serialization_round_trip_for_tools() -> None:
    overrides = {
        "demo_tool": ToolOverride(
            name="demo_tool",
            expected_contract_hash=VALID_DIGEST,
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
                contract_hash=VALID_DIGEST,
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


@dataclass(slots=True)
class _NoneToolContract:
    name: str = "none_tool"
    description: str = "None params/result tool"
    params_type: type[SupportsDataclassOrNone] | type[None] = _NONE_TYPE
    result_type: type[SupportsDataclassOrNone] = _NONE_TYPE
    result_container: Literal["object", "array"] = "object"
    accepts_overrides: bool = True


def test_tool_contract_hash_handles_none_types() -> None:
    digest = _tool_contract_hash(_NoneToolContract())

    assert len(str(digest)) == 64
