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

"""CRUD operations, seed, resolve, delete, and root detection tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from weakincentives.prompt.overrides import (
    LocalPromptOverridesStore,
    PromptDescriptor,
    PromptOverride,
    PromptOverridesError,
    SectionOverride,
)
from weakincentives.prompt.overrides._fs import OverrideFilesystem

from .conftest import (
    OTHER_DIGEST,
    auto_discovery_override_path,
    build_prompt,
    build_prompt_with_tool,
    override_path,
)


def test_upsert_resolve_and_delete_roundtrip(tmp_path: Path) -> None:
    prompt = build_prompt()
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

    op = override_path(tmp_path, descriptor)
    assert op.is_file()
    payload = json.loads(op.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["sections"]["greeting"]["body"] == "Cheer loudly for ${subject}."

    resolved = store.resolve(descriptor)
    assert resolved is not None
    assert resolved.sections[section.path].body == "Cheer loudly for ${subject}."

    store.delete(ns=descriptor.ns, prompt_key=descriptor.key, tag="latest")
    assert not op.exists()

    # deleting again should be a no-op
    store.delete(ns=descriptor.ns, prompt_key=descriptor.key, tag="latest")


def test_seed_captures_prompt_content(tmp_path: Path) -> None:
    prompt = build_prompt_with_tool()
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

    prompt = build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)

    monkeypatch.setattr(OverrideFilesystem, "_git_toplevel", lambda _self: None)
    monkeypatch.chdir(nested)

    store = LocalPromptOverridesStore()
    override = store.seed(prompt)

    section = descriptor.sections[0]
    assert override.sections[section.path].body == "Greet ${subject} warmly."

    op = auto_discovery_override_path(tmp_path, descriptor)
    assert op.exists()


def test_seed_preserves_existing_override(tmp_path: Path) -> None:
    prompt = build_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)

    first = store.seed(prompt)
    second = store.seed(prompt)

    assert first.sections == second.sections
    assert first.tool_overrides == second.tool_overrides


def test_seed_errors_on_corrupt_file(tmp_path: Path) -> None:
    prompt = build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    store.seed(prompt)
    op = override_path(tmp_path, descriptor)
    op.write_text("not-json", encoding="utf-8")

    with pytest.raises(PromptOverridesError):
        store.seed(prompt)


def test_seed_errors_on_stale_override(tmp_path: Path) -> None:
    prompt = build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    store = LocalPromptOverridesStore(root_path=tmp_path)

    op = override_path(tmp_path, descriptor)
    op.parent.mkdir(parents=True, exist_ok=True)
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
    op.write_text(json.dumps(payload), encoding="utf-8")

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

    prompt = build_prompt()

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

    prompt = build_prompt()
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

    prompt = build_prompt()

    monkeypatch.setattr(OverrideFilesystem, "_git_toplevel", lambda _self: None)
    monkeypatch.chdir(nested)

    store = LocalPromptOverridesStore()
    store.seed(prompt)

    assert (repo_root / ".weakincentives" / "prompts").exists()


def test_upsert_and_resolve_section_with_summary_and_visibility(tmp_path: Path) -> None:
    """Test that summary and visibility fields are persisted and resolved."""
    prompt = build_prompt()
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
    op = override_path(tmp_path, descriptor)
    payload = json.loads(op.read_text(encoding="utf-8"))
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
    prompt = build_prompt()
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
