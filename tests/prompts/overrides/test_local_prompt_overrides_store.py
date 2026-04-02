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
    """Build expected override path when root_path is explicitly provided.

    When root_path is explicit, it is used directly as the overrides directory
    (no .weakincentives/prompts/overrides prefix).
    """
    override_dir = tmp_path
    for segment in descriptor.ns.split("/"):
        override_dir /= segment
    override_dir /= descriptor.key
    return override_dir / f"{tag}.json"


def _auto_discovery_override_path(
    repo_root: Path, descriptor: PromptDescriptor, tag: str = "latest"
) -> Path:
    """Build expected override path when using automatic discovery.

    When using automatic discovery (no explicit root_path), the path
    includes the .weakincentives/prompts/overrides prefix.
    """
    override_dir = repo_root / ".weakincentives" / "prompts" / "overrides"
    for segment in descriptor.ns.split("/"):
        override_dir /= segment
    override_dir /= descriptor.key
    return override_dir / f"{tag}.json"


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

    monkeypatch.chdir(nested)

    store = LocalPromptOverridesStore(_git_toplevel_fn=lambda: None)
    override = store.seed(prompt)

    section = descriptor.sections[0]
    assert override.sections[section.path].body == "Greet ${subject} warmly."

    override_path = _auto_discovery_override_path(tmp_path, descriptor)
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


def test_seed_preserves_existing_override(tmp_path: Path) -> None:
    prompt = _build_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)

    first = store.seed(prompt)
    second = store.seed(prompt)

    assert first.sections == second.sections
    assert first.tool_overrides == second.tool_overrides


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
