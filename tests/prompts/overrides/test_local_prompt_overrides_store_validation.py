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

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate, Tool
from weakincentives.prompt.overrides import (
    HexDigest,
    LocalPromptOverridesStore,
    PromptDescriptor,
    PromptOverride,
    PromptOverridesError,
    SectionOverride,
    ToolOverride,
)
from weakincentives.prompt.overrides.validation import (
    load_sections,
    load_tools,
)
from weakincentives.types import JSONValue


@dataclass
class _GreetingParams:
    subject: str


@dataclass
class _ToolParams:
    query: str = field(metadata={"description": "User provided keywords."})


@dataclass
class _ToolResult:
    result: str


def _build_prompt() -> Prompt[Any]:
    return Prompt(
        PromptTemplate.create(
            ns="tests.versioning",
            key="versioned-greeting",
            sections=[
                MarkdownSection[_GreetingParams](
                    title="Greeting",
                    template="Greet ${subject} warmly.",
                    key="greeting",
                )
            ],
        )
    )


def _build_prompt_with_tool() -> Prompt[Any]:
    tool = Tool[_ToolParams, _ToolResult].create(
        name="search",
        description="Search stored notes.",
        handler=None,
    )
    return Prompt(
        PromptTemplate.create(
            ns="tests.versioning",
            key="versioned-greeting-tools",
            sections=[
                MarkdownSection[_GreetingParams](
                    title="Greeting",
                    template="Greet ${subject} warmly.",
                    key="greeting",
                    tools=[tool],
                )
            ],
        )
    )


def _override_path(
    tmp_path: Path, descriptor: PromptDescriptor, tag: str = "latest"
) -> Path:
    override_dir = tmp_path
    for segment in descriptor.ns.split("/"):
        override_dir /= segment
    override_dir /= descriptor.key
    return override_dir / f"{tag}.json"


OTHER_DIGEST = HexDigest("b" * 64)


# -- Section payload validation -----------------------------------------------


def test_resolve_section_payload_validation_errors(tmp_path: Path) -> None:
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    section = descriptor.sections[0]
    section_key = "/".join(section.path)

    assert load_sections(None, descriptor) == {}
    empty_sections: Mapping[str, JSONValue] = {}
    assert load_sections(empty_sections, descriptor) == {}

    with pytest.raises(PromptOverridesError):
        load_sections(cast(JSONValue, ["invalid"]), descriptor)

    with pytest.raises(PromptOverridesError):
        load_sections({section_key: {"expected_hash": 123, "body": "Body"}}, descriptor)

    with pytest.raises(PromptOverridesError):
        load_sections({section_key: "invalid"}, descriptor)

    with pytest.raises(PromptOverridesError):
        load_sections(
            {section_key: {"expected_hash": section.content_hash, "body": 123}},
            descriptor,
        )

    with pytest.raises(PromptOverridesError):
        load_sections(cast(JSONValue, []), descriptor)

    with pytest.raises(PromptOverridesError):
        load_sections(cast(JSONValue, {1: {}}), descriptor)


# -- Tool payload validation ---------------------------------------------------


@pytest.fixture()
def tool_prompt_ctx(
    tmp_path: Path,
) -> tuple[PromptDescriptor, LocalPromptOverridesStore, Path, dict[str, JSONValue]]:
    """Shared context for tool payload validation tests."""
    prompt = _build_prompt_with_tool()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)
    override_path = _override_path(tmp_path, descriptor)
    override_path.parent.mkdir(parents=True, exist_ok=True)
    base_payload: dict[str, JSONValue] = {
        "version": 2,
        "ns": descriptor.ns,
        "prompt_key": descriptor.key,
        "tag": "latest",
        "sections": {},
        "tools": {},
    }
    return descriptor, store, override_path, base_payload


def _write_tool_payload(
    override_path: Path,
    base_payload: dict[str, JSONValue],
    tools: JSONValue,
) -> None:
    payload = dict(base_payload)
    payload["tools"] = tools
    override_path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_tools_none_and_empty_returns_empty(
    tool_prompt_ctx: tuple[
        PromptDescriptor, LocalPromptOverridesStore, Path, dict[str, JSONValue]
    ],
) -> None:
    descriptor, _, _, _ = tool_prompt_ctx
    assert load_tools(None, descriptor) == {}
    empty_tools: Mapping[str, JSONValue] = {}
    assert load_tools(empty_tools, descriptor) == {}


def test_resolve_tool_stale_and_unknown_returns_none(
    tool_prompt_ctx: tuple[
        PromptDescriptor, LocalPromptOverridesStore, Path, dict[str, JSONValue]
    ],
) -> None:
    descriptor, store, override_path, base_payload = tool_prompt_ctx
    tool = descriptor.tools[0]
    _write_tool_payload(
        override_path,
        base_payload,
        {
            "unknown": {
                "expected_contract_hash": tool.contract_hash,
                "description": "desc",
            },
            tool.name: {
                "expected_contract_hash": str(OTHER_DIGEST),
                "description": "desc",
                "param_descriptions": {},
            },
        },
    )
    assert store.resolve(descriptor) is None


def test_resolve_tool_non_string_description_raises(
    tool_prompt_ctx: tuple[
        PromptDescriptor, LocalPromptOverridesStore, Path, dict[str, JSONValue]
    ],
) -> None:
    descriptor, store, override_path, base_payload = tool_prompt_ctx
    tool = descriptor.tools[0]
    _write_tool_payload(
        override_path,
        base_payload,
        {
            tool.name: {
                "expected_contract_hash": tool.contract_hash,
                "description": 123,
                "param_descriptions": {},
            }
        },
    )
    with pytest.raises(PromptOverridesError):
        store.resolve(descriptor)


def test_resolve_tool_null_param_descriptions_resolves(
    tool_prompt_ctx: tuple[
        PromptDescriptor, LocalPromptOverridesStore, Path, dict[str, JSONValue]
    ],
) -> None:
    descriptor, store, override_path, base_payload = tool_prompt_ctx
    tool = descriptor.tools[0]
    _write_tool_payload(
        override_path,
        base_payload,
        {
            tool.name: {
                "expected_contract_hash": tool.contract_hash,
                "description": "desc",
                "param_descriptions": None,
            }
        },
    )
    assert store.resolve(descriptor) is not None


def test_resolve_tool_list_param_descriptions_raises(
    tool_prompt_ctx: tuple[
        PromptDescriptor, LocalPromptOverridesStore, Path, dict[str, JSONValue]
    ],
) -> None:
    descriptor, store, override_path, base_payload = tool_prompt_ctx
    tool = descriptor.tools[0]
    _write_tool_payload(
        override_path,
        base_payload,
        {
            tool.name: {
                "expected_contract_hash": tool.contract_hash,
                "description": "desc",
                "param_descriptions": [],
            }
        },
    )
    with pytest.raises(PromptOverridesError):
        store.resolve(descriptor)


def test_resolve_tool_non_string_param_value_raises(
    tool_prompt_ctx: tuple[
        PromptDescriptor, LocalPromptOverridesStore, Path, dict[str, JSONValue]
    ],
) -> None:
    descriptor, store, override_path, base_payload = tool_prompt_ctx
    tool = descriptor.tools[0]
    _write_tool_payload(
        override_path,
        base_payload,
        {
            tool.name: {
                "expected_contract_hash": tool.contract_hash,
                "description": "desc",
                "param_descriptions": {"field": 1},
            }
        },
    )
    with pytest.raises(PromptOverridesError):
        store.resolve(descriptor)


def test_load_tools_list_input_raises(
    tool_prompt_ctx: tuple[
        PromptDescriptor, LocalPromptOverridesStore, Path, dict[str, JSONValue]
    ],
) -> None:
    descriptor, _, _, _ = tool_prompt_ctx
    with pytest.raises(PromptOverridesError):
        load_tools(cast(JSONValue, []), descriptor)


def test_load_tools_non_string_key_raises(
    tool_prompt_ctx: tuple[
        PromptDescriptor, LocalPromptOverridesStore, Path, dict[str, JSONValue]
    ],
) -> None:
    descriptor, _, _, _ = tool_prompt_ctx
    with pytest.raises(PromptOverridesError):
        load_tools(cast(JSONValue, {1: {}}), descriptor)


def test_load_tools_non_dict_entry_raises(
    tool_prompt_ctx: tuple[
        PromptDescriptor, LocalPromptOverridesStore, Path, dict[str, JSONValue]
    ],
) -> None:
    descriptor, _, _, _ = tool_prompt_ctx
    tool = descriptor.tools[0]
    with pytest.raises(PromptOverridesError):
        load_tools({tool.name: "invalid"}, descriptor)


def test_load_tools_non_string_hash_raises(
    tool_prompt_ctx: tuple[
        PromptDescriptor, LocalPromptOverridesStore, Path, dict[str, JSONValue]
    ],
) -> None:
    descriptor, _, _, _ = tool_prompt_ctx
    tool = descriptor.tools[0]
    with pytest.raises(PromptOverridesError):
        load_tools(
            {tool.name: {"expected_contract_hash": 123}},
            descriptor,
        )


# -- Upsert validation --------------------------------------------------------


def _make_section_override(
    section: object,
    *,
    path: tuple[str, ...] | None = None,
    expected_hash: str | None = None,
    body: object = "Body",
) -> dict[tuple[str, ...], SectionOverride]:
    """Build a single-entry sections dict for testing overrides."""
    return {
        path or section.path: SectionOverride(  # type: ignore[union-attr]
            path=path or section.path,  # type: ignore[union-attr]
            expected_hash=expected_hash or section.content_hash,  # type: ignore[union-attr]
            body=body,  # type: ignore[arg-type]
        )
    }


def _make_override(
    descriptor: PromptDescriptor,
    sections: dict[tuple[str, ...], SectionOverride],
    tool_overrides: dict[str, ToolOverride] | None = None,
) -> PromptOverride:
    """Build a PromptOverride for validation testing."""
    kwargs: dict[str, object] = {
        "ns": descriptor.ns,
        "prompt_key": descriptor.key,
        "tag": "latest",
        "sections": sections,
    }
    if tool_overrides is not None:
        kwargs["tool_overrides"] = tool_overrides
    return PromptOverride(**kwargs)  # type: ignore[arg-type]


@pytest.fixture()
def upsert_ctx(
    tmp_path: Path,
) -> tuple[
    PromptDescriptor,
    LocalPromptOverridesStore,
    object,
    object,
    dict[tuple[str, ...], SectionOverride],
]:
    """Shared context for upsert validation tests."""
    prompt = _build_prompt_with_tool()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)
    section = descriptor.sections[0]
    tool = descriptor.tools[0]
    valid_sections = _make_section_override(section)
    return descriptor, store, section, tool, valid_sections


_UpsertCtx = tuple[
    PromptDescriptor,
    LocalPromptOverridesStore,
    object,
    object,
    dict[tuple[str, ...], SectionOverride],
]


def test_upsert_rejects_mismatched_metadata(tmp_path: Path) -> None:
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)
    section = descriptor.sections[0]

    override = PromptOverride(
        ns="other",
        prompt_key=descriptor.key,
        tag="latest",
        sections={
            section.path: SectionOverride(
                path=section.path,
                expected_hash=section.content_hash,
                body="Text",
            )
        },
    )

    with pytest.raises(PromptOverridesError):
        store.upsert(descriptor, override)


def test_upsert_unknown_section_path_raises(upsert_ctx: _UpsertCtx) -> None:
    descriptor, store, section, _, _ = upsert_ctx
    override = _make_override(
        descriptor, _make_section_override(section, path=("unknown",))
    )
    with pytest.raises(PromptOverridesError):
        store.upsert(descriptor, override)


def test_upsert_wrong_section_hash_raises(upsert_ctx: _UpsertCtx) -> None:
    descriptor, store, section, _, _ = upsert_ctx
    override = _make_override(
        descriptor, _make_section_override(section, expected_hash=OTHER_DIGEST)
    )
    with pytest.raises(PromptOverridesError):
        store.upsert(descriptor, override)


def test_upsert_non_string_body_raises(upsert_ctx: _UpsertCtx) -> None:
    descriptor, store, section, _, _ = upsert_ctx
    override = _make_override(descriptor, _make_section_override(section, body=123))
    with pytest.raises(PromptOverridesError):
        store.upsert(descriptor, override)


def test_upsert_missing_tool_name_raises(upsert_ctx: _UpsertCtx) -> None:
    descriptor, store, _, tool, valid_sections = upsert_ctx
    override = _make_override(
        descriptor,
        valid_sections,
        {
            "missing": ToolOverride(
                name="missing",
                expected_contract_hash=tool.contract_hash,  # type: ignore[union-attr]
            )
        },
    )
    with pytest.raises(PromptOverridesError):
        store.upsert(descriptor, override)


def test_upsert_wrong_tool_hash_raises(upsert_ctx: _UpsertCtx) -> None:
    descriptor, store, _, tool, valid_sections = upsert_ctx
    override = _make_override(
        descriptor,
        valid_sections,
        {tool.name: ToolOverride(name=tool.name, expected_contract_hash=OTHER_DIGEST)},  # type: ignore[union-attr]
    )
    with pytest.raises(PromptOverridesError):
        store.upsert(descriptor, override)


def test_upsert_non_string_tool_description_raises(upsert_ctx: _UpsertCtx) -> None:
    descriptor, store, _, tool, valid_sections = upsert_ctx
    override = _make_override(
        descriptor,
        valid_sections,
        {
            tool.name: ToolOverride(  # type: ignore[union-attr]
                name=tool.name,  # type: ignore[union-attr]
                expected_contract_hash=tool.contract_hash,  # type: ignore[union-attr]
                description=123,
            )
        },  # type: ignore[arg-type]
    )
    with pytest.raises(PromptOverridesError):
        store.upsert(descriptor, override)


def test_upsert_non_string_param_values_raises(upsert_ctx: _UpsertCtx) -> None:
    descriptor, store, _, tool, valid_sections = upsert_ctx
    override = _make_override(
        descriptor,
        valid_sections,
        {
            tool.name: ToolOverride(  # type: ignore[union-attr]
                name=tool.name,  # type: ignore[union-attr]
                expected_contract_hash=tool.contract_hash,  # type: ignore[union-attr]
                param_descriptions={"field": 1},
            )
        },  # type: ignore[arg-type]
    )
    with pytest.raises(PromptOverridesError):
        store.upsert(descriptor, override)


def test_upsert_non_dict_param_descriptions_raises(upsert_ctx: _UpsertCtx) -> None:
    descriptor, store, _, tool, valid_sections = upsert_ctx
    override = _make_override(
        descriptor,
        valid_sections,
        {
            tool.name: ToolOverride(  # type: ignore[union-attr]
                name=tool.name,  # type: ignore[union-attr]
                expected_contract_hash=tool.contract_hash,  # type: ignore[union-attr]
                param_descriptions=123,
            )
        },  # type: ignore[arg-type]
    )
    with pytest.raises(PromptOverridesError):
        store.upsert(descriptor, override)


def test_upsert_rejects_non_string_section_hash(tmp_path: Path) -> None:
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)
    section = descriptor.sections[0]

    override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="latest",
        sections={
            section.path: SectionOverride(
                path=section.path,
                expected_hash=cast(Any, 123),
                body="Body",
            )
        },
        tool_overrides={},
    )

    with pytest.raises(PromptOverridesError):
        store.upsert(descriptor, override)


def test_upsert_rejects_non_string_tool_hash(tmp_path: Path) -> None:
    prompt = _build_prompt_with_tool()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)
    section = descriptor.sections[0]
    tool = descriptor.tools[0]

    override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="latest",
        sections={
            section.path: SectionOverride(
                path=section.path,
                expected_hash=section.content_hash,
                body="Body",
            )
        },
        tool_overrides={
            tool.name: ToolOverride(
                name=tool.name,
                expected_contract_hash=cast(Any, 123),
                param_descriptions={},
            )
        },
    )

    with pytest.raises(PromptOverridesError):
        store.upsert(descriptor, override)


def test_upsert_allows_none_tool_description(tmp_path: Path) -> None:
    prompt = _build_prompt_with_tool()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)
    section = descriptor.sections[0]
    tool = descriptor.tools[0]

    override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="latest",
        sections={
            section.path: SectionOverride(
                path=section.path,
                expected_hash=section.content_hash,
                body="Body",
            )
        },
        tool_overrides={
            tool.name: ToolOverride(
                name=tool.name,
                expected_contract_hash=tool.contract_hash,
                description=None,
                param_descriptions={},
            )
        },
    )

    persisted = store.upsert(descriptor, override)
    assert persisted.tool_overrides[tool.name].description is None

    resolved = store.resolve(descriptor)
    assert resolved is not None
    assert resolved.tool_overrides[tool.name].description is None
