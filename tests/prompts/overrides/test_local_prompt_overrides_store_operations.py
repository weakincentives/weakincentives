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

"""Tests for store(), upsert(), and load_sections() operations on LocalPromptOverridesStore."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from weakincentives.prompt import MarkdownSection, PromptTemplate, Tool
from weakincentives.prompt.overrides import (
    HexDigest,
    LocalPromptOverridesStore,
    PromptDescriptor,
    PromptOverride,
    PromptOverridesError,
    SectionOverride,
    TaskExampleOverride,
    ToolOverride,
)
from weakincentives.prompt.overrides.validation import (
    load_sections,
)


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


def _build_prompt_with_tool() -> PromptTemplate:
    tool = Tool[_ToolParams, _ToolResult](
        name="search",
        description="Search stored notes.",
        handler=None,
    )
    return PromptTemplate(
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


def _override_path(
    tmp_path: Path, descriptor: PromptDescriptor, tag: str = "latest"
) -> Path:
    """Build expected override path when root_path is explicitly provided.

    When root_path is explicit, it is used directly as the overrides directory
    (no .weakincentives/prompts/overrides prefix).
    """
    override_dir = tmp_path
    for segment in descriptor.ns.split("/"):
        override_dir /= segment
    override_dir /= descriptor.key
    return override_dir / f"{tag}.json"


VALID_DIGEST = HexDigest("a" * 64)
OTHER_DIGEST = HexDigest("b" * 64)


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


def test_store_rejects_prompt_instead_of_descriptor(tmp_path: Path) -> None:
    """Test that store() raises TypeError when passed a Prompt instead of PromptDescriptor."""
    prompt = _build_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)

    descriptor = PromptDescriptor.from_prompt(prompt)
    section = descriptor.sections[0]
    override = SectionOverride(
        path=section.path,
        expected_hash=section.content_hash,
        body="Content",
    )

    with pytest.raises(TypeError, match="requires a PromptDescriptor"):
        store.store(prompt, override)  # type: ignore[arg-type]
