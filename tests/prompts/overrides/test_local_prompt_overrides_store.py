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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import pytest

from weakincentives.prompt import MarkdownSection, PromptTemplate, Tool
from weakincentives.prompt.overrides import (
    HexDigest,
    LocalPromptOverridesStore,
    PromptDescriptor,
    PromptOverride,
    PromptOverridesError,
    SectionDescriptor,
    SectionOverride,
    TaskExampleOverride,
    ToolOverride,
)
from weakincentives.prompt.overrides._fs import OverrideFilesystem
from weakincentives.prompt.overrides.validation import (
    load_sections,
    load_tools,
    seed_sections,
    seed_tools,
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


def _build_prompt() -> PromptTemplate:
    return PromptTemplate(
        ns="tests/versioning",
        key="versioned-greeting",
        sections=[
            MarkdownSection[_GreetingParams](
                title="Greeting",
                template="Greet ${subject} warmly.",
                key="greeting",
            )
        ],
    )


def _build_prompt_with_tool() -> PromptTemplate:
    tool = Tool[_ToolParams, _ToolResult](
        name="search",
        description="Search stored notes.",
        handler=None,
    )
    return PromptTemplate(
        ns="tests/versioning",
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


def _override_path(
    tmp_path: Path, descriptor: PromptDescriptor, tag: str = "latest"
) -> Path:
    override_dir = tmp_path / ".weakincentives" / "prompts" / "overrides"
    for segment in descriptor.ns.split("/"):
        override_dir /= segment
    override_dir /= descriptor.key
    return override_dir / f"{tag}.json"


VALID_DIGEST = HexDigest("a" * 64)
OTHER_DIGEST = HexDigest("b" * 64)


def test_upsert_resolve_and_delete_roundtrip(tmp_path: Path) -> None:
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    section = descriptor.sections[0]
    assert store.resolve(descriptor) is None
    override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="latest",
        sections={
            section.path: SectionOverride(
                path=section.path,
                expected_hash=section.content_hash,
                body="Cheer loudly for ${subject}.",
            )
        },
    )

    persisted = store.upsert(descriptor, override)
    assert persisted.sections[section.path].body == "Cheer loudly for ${subject}."

    override_path = _override_path(tmp_path, descriptor)
    assert override_path.is_file()
    payload = json.loads(override_path.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["sections"]["greeting"]["body"] == "Cheer loudly for ${subject}."

    resolved = store.resolve(descriptor)
    assert resolved is not None
    assert resolved.sections[section.path].body == "Cheer loudly for ${subject}."

    store.delete(ns=descriptor.ns, prompt_key=descriptor.key, tag="latest")
    assert not override_path.exists()

    # deleting again should be a no-op
    store.delete(ns=descriptor.ns, prompt_key=descriptor.key, tag="latest")


def test_seed_captures_prompt_content(tmp_path: Path) -> None:
    prompt = _build_prompt_with_tool()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    override = store.seed(prompt, tag="stable")
    section = descriptor.sections[0]

    assert section.path in override.sections
    assert override.sections[section.path].body == "Greet ${subject} warmly."

    tool_descriptor = descriptor.tools[0]
    assert tool_descriptor.name in override.tool_overrides
    tool_override = override.tool_overrides[tool_descriptor.name]
    assert tool_override.description == "Search stored notes."
    assert tool_override.param_descriptions == {"query": "User provided keywords."}

    resolved = store.resolve(descriptor, tag="stable")
    assert resolved is not None
    assert resolved.sections[section.path].body == "Greet ${subject} warmly."


def test_root_detection_manual_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "nested" / "workspace"
    nested.mkdir(parents=True)

    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)

    monkeypatch.setattr(OverrideFilesystem, "_git_toplevel", lambda _self: None)
    monkeypatch.chdir(nested)

    store = LocalPromptOverridesStore()
    override = store.seed(prompt)

    section = descriptor.sections[0]
    assert override.sections[section.path].body == "Greet ${subject} warmly."

    override_path = _override_path(tmp_path, descriptor)
    assert override_path.exists()


def test_resolve_invalid_json_raises_error(tmp_path: Path) -> None:
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    override_path = _override_path(tmp_path, descriptor)
    override_path.parent.mkdir(parents=True, exist_ok=True)
    override_path.write_text("{not-json}", encoding="utf-8")

    with pytest.raises(PromptOverridesError):
        store.resolve(descriptor)


def test_resolve_filters_stale_override_returns_none(tmp_path: Path) -> None:
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    section = descriptor.sections[0]
    override_path = _override_path(tmp_path, descriptor)
    override_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, JSONValue] = {
        "version": 2,
        "ns": descriptor.ns,
        "prompt_key": descriptor.key,
        "tag": "latest",
        "sections": {
            "/".join(section.path): {
                "expected_hash": str(OTHER_DIGEST),
                "body": "Cheer loudly.",
            }
        },
        "tools": {},
    }
    sections_payload = cast(dict[str, JSONValue], payload["sections"])
    sections_payload["unknown/path"] = {
        "expected_hash": section.content_hash,
        "body": "Unused.",
    }
    override_path.write_text(json.dumps(payload), encoding="utf-8")

    assert store.resolve(descriptor) is None


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


def test_resolve_tool_payload_validation_errors(tmp_path: Path) -> None:
    prompt = _build_prompt_with_tool()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)
    tool = descriptor.tools[0]
    override_path = _override_path(tmp_path, descriptor)
    override_path.parent.mkdir(parents=True, exist_ok=True)

    assert load_tools(None, descriptor) == {}
    empty_tools: Mapping[str, JSONValue] = {}
    assert load_tools(empty_tools, descriptor) == {}

    base_payload = {
        "version": 2,
        "ns": descriptor.ns,
        "prompt_key": descriptor.key,
        "tag": "latest",
        "sections": {},
        "tools": {},
    }

    payload = dict(base_payload)
    payload["tools"] = {
        "unknown": {
            "expected_contract_hash": tool.contract_hash,
            "description": "desc",
        },
        tool.name: {
            "expected_contract_hash": str(OTHER_DIGEST),
            "description": "desc",
            "param_descriptions": {},
        },
    }
    override_path.write_text(json.dumps(payload), encoding="utf-8")
    assert store.resolve(descriptor) is None

    payload = dict(base_payload)
    payload["tools"] = {
        tool.name: {
            "expected_contract_hash": tool.contract_hash,
            "description": 123,
            "param_descriptions": {},
        }
    }
    override_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PromptOverridesError):
        store.resolve(descriptor)

    payload = dict(base_payload)
    payload["tools"] = {
        tool.name: {
            "expected_contract_hash": tool.contract_hash,
            "description": "desc",
            "param_descriptions": None,
        }
    }
    override_path.write_text(json.dumps(payload), encoding="utf-8")
    assert store.resolve(descriptor) is not None

    payload = dict(base_payload)
    payload["tools"] = {
        tool.name: {
            "expected_contract_hash": tool.contract_hash,
            "description": "desc",
            "param_descriptions": [],
        }
    }
    override_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PromptOverridesError):
        store.resolve(descriptor)

    payload = dict(base_payload)
    payload["tools"] = {
        tool.name: {
            "expected_contract_hash": tool.contract_hash,
            "description": "desc",
            "param_descriptions": {"field": 1},
        }
    }
    override_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PromptOverridesError):
        store.resolve(descriptor)

    with pytest.raises(PromptOverridesError):
        load_tools(cast(JSONValue, []), descriptor)

    with pytest.raises(PromptOverridesError):
        load_tools(cast(JSONValue, {1: {}}), descriptor)

    with pytest.raises(PromptOverridesError):
        load_tools({tool.name: "invalid"}, descriptor)

    with pytest.raises(PromptOverridesError):
        load_tools(
            {tool.name: {"expected_contract_hash": 123}},
            descriptor,
        )


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


def test_upsert_validation_errors(tmp_path: Path) -> None:
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
            ("unknown",): SectionOverride(
                path=("unknown",),
                expected_hash=section.content_hash,
                body="Body",
            )
        },
    )
    with pytest.raises(PromptOverridesError):
        store.upsert(descriptor, override)

    override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="latest",
        sections={
            section.path: SectionOverride(
                path=section.path,
                expected_hash=OTHER_DIGEST,
                body="Body",
            )
        },
    )
    with pytest.raises(PromptOverridesError):
        store.upsert(descriptor, override)

    override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="latest",
        sections={
            section.path: SectionOverride(
                path=section.path,
                expected_hash=section.content_hash,
                body=123,  # type: ignore[arg-type]
            )
        },
    )
    with pytest.raises(PromptOverridesError):
        store.upsert(descriptor, override)

    override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="latest",
        sections={
            section.path: SectionOverride(
                path=section.path, expected_hash=section.content_hash, body="Body"
            )
        },
        tool_overrides={
            "missing": ToolOverride(
                name="missing",
                expected_contract_hash=tool.contract_hash,
            )
        },
    )
    with pytest.raises(PromptOverridesError):
        store.upsert(descriptor, override)

    override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="latest",
        sections={
            section.path: SectionOverride(
                path=section.path, expected_hash=section.content_hash, body="Body"
            )
        },
        tool_overrides={
            tool.name: ToolOverride(
                name=tool.name,
                expected_contract_hash=OTHER_DIGEST,
            )
        },
    )
    with pytest.raises(PromptOverridesError):
        store.upsert(descriptor, override)

    override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="latest",
        sections={
            section.path: SectionOverride(
                path=section.path, expected_hash=section.content_hash, body="Body"
            )
        },
        tool_overrides={
            tool.name: ToolOverride(
                name=tool.name,
                expected_contract_hash=tool.contract_hash,
                description=123,  # type: ignore[arg-type]
            )
        },
    )
    with pytest.raises(PromptOverridesError):
        store.upsert(descriptor, override)

    override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="latest",
        sections={
            section.path: SectionOverride(
                path=section.path, expected_hash=section.content_hash, body="Body"
            )
        },
        tool_overrides={
            tool.name: ToolOverride(
                name=tool.name,
                expected_contract_hash=tool.contract_hash,
                param_descriptions={"field": 1},  # type: ignore[arg-type]
            )
        },
    )
    with pytest.raises(PromptOverridesError):
        store.upsert(descriptor, override)

    override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="latest",
        sections={
            section.path: SectionOverride(
                path=section.path, expected_hash=section.content_hash, body="Body"
            )
        },
        tool_overrides={
            tool.name: ToolOverride(
                name=tool.name,
                expected_contract_hash=tool.contract_hash,
                param_descriptions=123,  # type: ignore[arg-type]
            )
        },
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


def test_seed_preserves_existing_override(tmp_path: Path) -> None:
    prompt = _build_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)

    first = store.seed(prompt)
    second = store.seed(prompt)

    assert first.sections == second.sections
    assert first.tool_overrides == second.tool_overrides


def test_seed_errors_on_corrupt_file(tmp_path: Path) -> None:
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    store.seed(prompt)
    override_path = _override_path(tmp_path, descriptor)
    override_path.write_text("not-json", encoding="utf-8")

    with pytest.raises(PromptOverridesError):
        store.seed(prompt)


def test_seed_errors_on_stale_override(tmp_path: Path) -> None:
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    override_path = _override_path(tmp_path, descriptor)
    override_path.parent.mkdir(parents=True, exist_ok=True)
    section = descriptor.sections[0]
    payload = {
        "version": 2,
        "ns": descriptor.ns,
        "prompt_key": descriptor.key,
        "tag": "latest",
        "sections": {
            "/".join(section.path): {
                "expected_hash": str(OTHER_DIGEST),
                "body": "Body",
            }
        },
        "tools": {},
    }
    override_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PromptOverridesError):
        store.seed(prompt)


def test_root_detection_git_command_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    nested = repo_root / "nested"
    nested.mkdir()

    prompt = _build_prompt()

    monkeypatch.setattr(
        OverrideFilesystem,
        "_git_toplevel",
        lambda _self: repo_root,
    )
    monkeypatch.chdir(nested)

    store = LocalPromptOverridesStore()
    override = store.seed(prompt)

    assert override.sections
    assert (repo_root / ".weakincentives").exists()


def test_root_detection_without_git_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = tmp_path / "workspace"
    nested.mkdir(parents=True)

    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)

    monkeypatch.setattr(OverrideFilesystem, "_git_toplevel", lambda _self: None)
    monkeypatch.chdir(nested)

    store = LocalPromptOverridesStore()

    with pytest.raises(PromptOverridesError):
        store.resolve(descriptor)


def test_git_toplevel_empty_output_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    nested = repo_root / "nested"
    nested.mkdir()

    prompt = _build_prompt()

    monkeypatch.setattr(OverrideFilesystem, "_git_toplevel", lambda _self: None)
    monkeypatch.chdir(nested)

    store = LocalPromptOverridesStore()
    store.seed(prompt)

    assert (repo_root / ".weakincentives" / "prompts").exists()


def test_identifier_validation_errors(tmp_path: Path) -> None:
    store = LocalPromptOverridesStore(root_path=tmp_path)

    with pytest.raises(PromptOverridesError):
        store.delete(ns="   ", prompt_key="key", tag="latest")

    with pytest.raises(PromptOverridesError):
        store.delete(ns="/", prompt_key="key", tag="latest")

    with pytest.raises(PromptOverridesError):
        store.delete(ns="ns", prompt_key="Key", tag="latest")

    with pytest.raises(PromptOverridesError):
        store.delete(ns="ns", prompt_key=" ", tag="latest")

    with pytest.raises(PromptOverridesError):
        store.delete(ns="ns", prompt_key="key", tag="LATEST")


def test_resolve_header_validation_errors(tmp_path: Path) -> None:
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    section = descriptor.sections[0]
    override_path = _override_path(tmp_path, descriptor)
    override_path.parent.mkdir(parents=True, exist_ok=True)
    bad_version = {
        "version": 99,  # Unsupported version
        "ns": descriptor.ns,
        "prompt_key": descriptor.key,
        "tag": "latest",
        "sections": {
            "/".join(section.path): {
                "expected_hash": section.content_hash,
                "body": "Body",
            }
        },
        "tools": {},
    }
    override_path.write_text(json.dumps(bad_version), encoding="utf-8")
    with pytest.raises(PromptOverridesError):
        store.resolve(descriptor)

    bad_metadata = dict(bad_version)
    bad_metadata["version"] = 2
    bad_metadata["ns"] = "other"
    override_path.write_text(json.dumps(bad_metadata), encoding="utf-8")
    with pytest.raises(PromptOverridesError):
        store.resolve(descriptor)


def test_seed_sections_missing_section_raises(tmp_path: Path) -> None:
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)

    missing_section_descriptor = replace(
        descriptor,
        sections=[
            SectionDescriptor(
                path=("missing",),
                content_hash=descriptor.sections[0].content_hash,
                number="1",
            )
        ],
    )

    with pytest.raises(PromptOverridesError):
        seed_sections(prompt, missing_section_descriptor)


def test_seed_sections_missing_template_raises(tmp_path: Path) -> None:
    _ = tmp_path
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)

    for node in prompt.sections:
        section = cast(Any, node.section)  # type: ignore[union-attr]
        section.original_body_template = lambda: None

    with pytest.raises(PromptOverridesError):
        seed_sections(prompt, descriptor)


def test_seed_tools_missing_tool_raises(tmp_path: Path) -> None:
    _ = tmp_path
    prompt = _build_prompt_with_tool()
    descriptor = PromptDescriptor.from_prompt(prompt)

    for node in prompt.sections:
        section = cast(Any, node.section)  # type: ignore[union-attr]
        section._tools = ()

    with pytest.raises(PromptOverridesError):
        seed_tools(prompt, descriptor)


def test_collect_param_descriptions_without_metadata(tmp_path: Path) -> None:
    @dataclass
    class _PlainParams:
        value: str

    @dataclass
    class _PlainResult:
        value: str

    tool = Tool[_PlainParams, _PlainResult](
        name="plain",
        description="Plain description.",
        handler=None,
    )

    prompt = PromptTemplate(
        ns="tests/versioning",
        key="plain-metadata",
        sections=[
            MarkdownSection[_GreetingParams](
                title="Example",
                template="Example body for ${subject}.",
                key="example",
                tools=[tool],
            )
        ],
    )
    descriptor = PromptDescriptor.from_prompt(prompt)
    tool.params_type = str

    overrides = seed_tools(prompt, descriptor)

    assert overrides[tool.name].param_descriptions == {}


def test_collect_param_descriptions_with_partial_metadata(tmp_path: Path) -> None:
    """Test branch 585->583: fields without description metadata are skipped."""

    @dataclass
    class _MixedParams:
        with_desc: str = field(metadata={"description": "Has a description"})
        no_desc: str = field(metadata={"other": "value"})  # No description key
        empty_desc: str = field(metadata={"description": ""})  # Empty description

    @dataclass
    class _MixedResult:
        value: str

    tool = Tool[_MixedParams, _MixedResult](
        name="mixed",
        description="Mixed metadata description.",
        handler=None,
    )

    prompt = PromptTemplate(
        ns="tests/versioning",
        key="mixed-metadata",
        sections=[
            MarkdownSection[_GreetingParams](
                title="Example",
                template="Example body for ${subject}.",
                key="example",
                tools=[tool],
            )
        ],
    )
    descriptor = PromptDescriptor.from_prompt(prompt)

    overrides = seed_tools(prompt, descriptor)

    # Only the field with a non-empty description should be included
    assert overrides[tool.name].param_descriptions == {"with_desc": "Has a description"}


def test_store_section_override_hash_mismatch(tmp_path: Path) -> None:
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    section = descriptor.sections[0]
    wrong_hash = HexDigest("b" * 64)
    override = SectionOverride(
        path=section.path,
        expected_hash=wrong_hash,
        body="Modified body",
    )

    with pytest.raises(PromptOverridesError, match="Hash mismatch for section"):
        store.store(descriptor, override)


def test_store_tool_override_hash_mismatch(tmp_path: Path) -> None:
    prompt = _build_prompt_with_tool()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    tool = descriptor.tools[0]
    wrong_hash = HexDigest("c" * 64)
    override = ToolOverride(
        name=tool.name,
        expected_contract_hash=wrong_hash,
        description="Modified description",
        param_descriptions={},
    )

    with pytest.raises(PromptOverridesError, match="Hash mismatch for tool"):
        store.store(descriptor, override)


def test_store_task_example_override(tmp_path: Path) -> None:
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    task_override = TaskExampleOverride(
        path=("section", "example"),
        index=0,
        expected_hash=None,
        action="append",
        objective="New objective",
    )

    result = store.store(descriptor, task_override, tag="latest")
    assert len(result.task_example_overrides) == 1
    assert result.task_example_overrides[0] == task_override


def test_store_task_example_override_updates_existing(tmp_path: Path) -> None:
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    task_override1 = TaskExampleOverride(
        path=("section", "example"),
        index=0,
        expected_hash=None,
        action="append",
        objective="First objective",
    )
    task_override2 = TaskExampleOverride(
        path=("section", "example"),
        index=0,
        expected_hash=None,
        action="modify",
        objective="Updated objective",
    )

    store.store(descriptor, task_override1, tag="latest")
    result = store.store(descriptor, task_override2, tag="latest")

    # Should replace the existing override with same path+index
    assert len(result.task_example_overrides) == 1
    assert result.task_example_overrides[0].action == "modify"
    assert result.task_example_overrides[0].objective == "Updated objective"


def test_store_task_example_override_appends_different_index(tmp_path: Path) -> None:
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    task_override1 = TaskExampleOverride(
        path=("section", "example"),
        index=0,
        expected_hash=None,
        action="append",
    )
    task_override2 = TaskExampleOverride(
        path=("section", "example"),
        index=1,
        expected_hash=None,
        action="append",
    )

    store.store(descriptor, task_override1, tag="latest")
    result = store.store(descriptor, task_override2, tag="latest")

    # Should have both overrides since they have different indices
    assert len(result.task_example_overrides) == 2


def test_upsert_and_resolve_section_with_summary_and_visibility(tmp_path: Path) -> None:
    """Test that summary and visibility fields are persisted and resolved."""
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
                expected_hash=section.content_hash,
                body="Full body content for ${subject}.",
                summary="Brief summary for ${subject}.",
                visibility="summary",
            )
        },
    )

    persisted = store.upsert(descriptor, override)
    assert persisted.sections[section.path].body == "Full body content for ${subject}."
    assert persisted.sections[section.path].summary == "Brief summary for ${subject}."
    assert persisted.sections[section.path].visibility == "summary"

    # Check the JSON file contains the new fields
    override_path = _override_path(tmp_path, descriptor)
    payload = json.loads(override_path.read_text(encoding="utf-8"))
    section_data = payload["sections"]["greeting"]
    assert section_data["body"] == "Full body content for ${subject}."
    assert section_data["summary"] == "Brief summary for ${subject}."
    assert section_data["visibility"] == "summary"

    # Resolve and verify
    resolved = store.resolve(descriptor)
    assert resolved is not None
    assert resolved.sections[section.path].body == "Full body content for ${subject}."
    assert resolved.sections[section.path].summary == "Brief summary for ${subject}."
    assert resolved.sections[section.path].visibility == "summary"


def test_resolve_section_without_summary_and_visibility(tmp_path: Path) -> None:
    """Test that sections without summary/visibility default to None."""
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
                expected_hash=section.content_hash,
                body="Body without summary.",
            )
        },
    )

    persisted = store.upsert(descriptor, override)
    assert persisted.sections[section.path].summary is None
    assert persisted.sections[section.path].visibility is None

    resolved = store.resolve(descriptor)
    assert resolved is not None
    assert resolved.sections[section.path].summary is None
    assert resolved.sections[section.path].visibility is None


def test_load_sections_invalid_summary_raises(tmp_path: Path) -> None:
    """Test that non-string summary values raise an error."""
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    section = descriptor.sections[0]
    section_key = "/".join(section.path)

    with pytest.raises(PromptOverridesError, match="summary must be a string"):
        load_sections(
            {
                section_key: {
                    "expected_hash": section.content_hash,
                    "body": "Body",
                    "summary": 123,
                }
            },
            descriptor,
        )


def test_load_sections_invalid_visibility_raises(tmp_path: Path) -> None:
    """Test that invalid visibility values raise an error."""
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    section = descriptor.sections[0]
    section_key = "/".join(section.path)

    with pytest.raises(
        PromptOverridesError, match="visibility must be 'full' or 'summary'"
    ):
        load_sections(
            {
                section_key: {
                    "expected_hash": section.content_hash,
                    "body": "Body",
                    "visibility": "invalid",
                }
            },
            descriptor,
        )


def test_store_section_with_summary_and_visibility(tmp_path: Path) -> None:
    """Test that store() preserves summary and visibility fields."""
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    section = descriptor.sections[0]
    override = SectionOverride(
        path=section.path,
        expected_hash=section.content_hash,
        body="Full content.",
        summary="Short summary.",
        visibility="full",
    )

    result = store.store(descriptor, override)
    stored_section = result.sections[section.path]
    assert stored_section.body == "Full content."
    assert stored_section.summary == "Short summary."
    assert stored_section.visibility == "full"


def test_store_tool_override_success(tmp_path: Path) -> None:
    """Test successful tool override storage via store()."""
    prompt = _build_prompt_with_tool()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    tool = descriptor.tools[0]
    override = ToolOverride(
        name=tool.name,
        expected_contract_hash=tool.contract_hash,
        description="New description",
        param_descriptions={"query": "Updated param description"},
    )

    result = store.store(descriptor, override)
    stored_tool = result.tool_overrides[tool.name]
    assert stored_tool.description == "New description"
    assert stored_tool.param_descriptions == {"query": "Updated param description"}


def test_store_tool_override_unknown_tool(tmp_path: Path) -> None:
    """Test that store() raises error for unknown tool names."""
    prompt = _build_prompt_with_tool()  # Has tools, but we'll look for a different one
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    override = ToolOverride(
        name="unknown_tool",
        expected_contract_hash=VALID_DIGEST,
        description="Some description",
        param_descriptions={},
    )

    with pytest.raises(
        PromptOverridesError, match="not registered in prompt descriptor"
    ):
        store.store(descriptor, override)


def test_store_with_malformed_existing_json(tmp_path: Path) -> None:
    """Test store() raises error when existing file contains malformed JSON."""
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    # Create malformed JSON file
    override_path = _override_path(tmp_path, descriptor)
    override_path.parent.mkdir(parents=True, exist_ok=True)
    override_path.write_text("{ invalid json }", encoding="utf-8")

    section = descriptor.sections[0]
    override = SectionOverride(
        path=section.path,
        expected_hash=section.content_hash,
        body="New content",
    )

    with pytest.raises(PromptOverridesError, match="Failed to parse prompt override"):
        store.store(descriptor, override)


def test_store_when_existing_overrides_are_stale(tmp_path: Path) -> None:
    """Test store() handles case where existing overrides are all stale."""
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    # Create a file with a stale section override (wrong hash)
    override_path = _override_path(tmp_path, descriptor)
    override_path.parent.mkdir(parents=True, exist_ok=True)
    stale_payload = {
        "version": 2,
        "ns": descriptor.ns,
        "prompt_key": descriptor.key,
        "tag": "latest",
        "sections": {
            "greeting": {
                "path": ["greeting"],
                "expected_hash": "0" * 64,  # Wrong hash
                "body": "Stale content",
            }
        },
        "tools": {},
        "task_example_overrides": [],
    }
    override_path.write_text(json.dumps(stale_payload), encoding="utf-8")

    # Store a new section override - should work even though existing is stale
    section = descriptor.sections[0]
    new_override = SectionOverride(
        path=section.path,
        expected_hash=section.content_hash,
        body="Fresh content",
    )

    result = store.store(descriptor, new_override)
    assert result.sections[section.path].body == "Fresh content"
