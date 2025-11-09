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

"""Tests for wink override helpers."""

from __future__ import annotations

import hashlib
import importlib
import sys
from collections.abc import Iterator, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import Protocol, cast

import pytest

import weakincentives.cli.wink_overrides as wink_overrides_module
from weakincentives.cli.config import MCPServerConfig
from weakincentives.cli.wink_overrides import (
    PROMPT_DESCRIPTOR_VERSION,
    OverrideListEntry,
    OverridesInspectionError,
    PromptNotFoundError,
    PromptRegistryImportError,
    SectionNotFoundError,
    SectionOverrideApplyError,
    SectionOverrideMutationResult,
    SectionOverrideRemoveError,
    SectionOverrideResolutionError,
    SectionOverrideSnapshot,
    SectionOverridesUnavailableError,
    ToolNotFoundError,
    ToolOverrideApplyError,
    ToolOverrideMutationResult,
    ToolOverrideRemoveError,
    ToolOverrideResolutionError,
    ToolOverrideSnapshot,
    ToolOverridesUnavailableError,
    _build_override_file_path,
    _build_override_list_entry,
    _coerce_prompt,
    _extract_override_body,
    _find_section_descriptor,
    _find_section_node,
    _iter_prompts_from_modules,
    _iter_prompts_from_registry,
    _iter_registry_prompts,
    _normalize_registry_key,
    _parse_section_path,
    _split_namespace,
    _validate_identifier,
    apply_section_override,
    apply_tool_override,
    fetch_section_override,
    fetch_tool_override,
    list_overrides,
    remove_section_override,
    remove_tool_override,
)
from weakincentives.prompt import Prompt
from weakincentives.prompt.overrides import (
    LocalPromptOverridesStore,
    PromptDescriptor,
    PromptOverride,
    PromptOverridesError,
    PromptOverridesStore,
    SectionOverride,
    ToolOverride,
)
from weakincentives.prompt.overrides.inspection import OverrideFileMetadata


class _PromptModule(Protocol):
    PROMPTS: Mapping[tuple[str, str], Prompt[object]]


@pytest.fixture
def prompt_workspace(tmp_path: Path) -> Iterator[tuple[Path, str]]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    module_name = "wink_prompts"
    module_dir = workspace / module_name
    module_dir.mkdir()
    module_code = dedent(
        """
        from dataclasses import dataclass, field

        from weakincentives.prompt import MarkdownSection, Prompt
        from weakincentives.prompt.tool import Tool, ToolContext
        from weakincentives.prompt.tool_result import ToolResult


        @dataclass
        class SearchParams:
            query: str = field(metadata={"description": "Search query"})


        @dataclass
        class SearchResult:
            value: str


        @dataclass
        class IndexParams:
            document_id: str = field(
                metadata={"description": "Document identifier"}
            )


        @dataclass
        class IndexResult:
            success: bool


        def _search_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[SearchResult]:
            return ToolResult(message="ok", value=None)


        def _index_handler(
            params: IndexParams, *, context: ToolContext
        ) -> ToolResult[IndexResult]:
            return ToolResult(message="ok", value=None)


        def _disabled_handler(
            params: SearchParams, *, context: ToolContext
        ) -> ToolResult[SearchResult]:
            return ToolResult(message="ok", value=None)


        search_tool = Tool[SearchParams, SearchResult](
            name="search_docs",
            description="Search the documentation.",
            handler=_search_handler,
        )

        index_tool = Tool[IndexParams, IndexResult](
            name="index_docs",
            description="Index documentation.",
            handler=_index_handler,
        )

        disabled_tool = Tool[SearchParams, SearchResult](
            name="search_disabled",
            description="Search disabled.",
            handler=_disabled_handler,
            accepts_overrides=False,
        )


        PROMPTS = {
            ("demo", "example"): Prompt(
                ns="demo",
                key="example",
                sections=(
                    MarkdownSection(
                        title="Example",
                        key="intro",
                        template="Example body",
                        tools=(search_tool, index_tool),
                    ),
                ),
            ),
            ("demo", "disabled"): Prompt(
                ns="demo",
                key="disabled",
                sections=(
                    MarkdownSection(
                        title="Disabled",
                        key="intro",
                        template="Disabled body",
                        accepts_overrides=False,
                    ),
                ),
            ),
            ("demo", "defaultless"): Prompt(
                ns="demo",
                key="defaultless",
                sections=(
                    MarkdownSection(
                        title="Defaultless",
                        key="intro",
                        template="Defaultless body",
                    ),
                ),
            ),
            ("demo", "tool_disabled"): Prompt(
                ns="demo",
                key="tool_disabled",
                sections=(
                    MarkdownSection(
                        title="Tool Disabled",
                        key="intro",
                        template="Tool disabled body",
                        tools=(disabled_tool,),
                    ),
                ),
            ),
        }
        """
    )
    (module_dir / "__init__.py").write_text(module_code)

    sys.path.insert(0, str(workspace))
    importlib.invalidate_caches()

    try:
        yield workspace, module_name
    finally:
        sys.modules.pop(module_name, None)
        with suppress(ValueError):  # pragma: no branch - defensive cleanup
            sys.path.remove(str(workspace))


def _load_prompt(module_name: str, key: tuple[str, str]) -> Prompt[object]:
    module = cast(_PromptModule, importlib.import_module(module_name))
    return module.PROMPTS[key]


def _build_config(workspace: Path, module_name: str) -> MCPServerConfig:
    overrides_dir = workspace / ".wink" / "overrides"
    return MCPServerConfig(
        workspace_root=workspace,
        overrides_dir=overrides_dir,
        prompt_registry_modules=(module_name,),
    )


def _build_store(workspace: Path) -> LocalPromptOverridesStore:
    overrides_relative = Path(".wink") / "overrides"
    return LocalPromptOverridesStore(
        root_path=workspace,
        overrides_relative_path=overrides_relative,
    )


def test_list_overrides_returns_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    overrides_root = workspace / ".wink" / "overrides" / "demo" / "example"
    overrides_root.mkdir(parents=True)

    file_path = overrides_root / "latest.json"
    file_path.write_text(
        '{"sections": {"intro": {}}, "tools": {"search": {}}}',
        encoding="utf-8",
    )

    config = MCPServerConfig(
        workspace_root=workspace,
        overrides_dir=Path(".wink/overrides"),
    )

    entries = list_overrides(config=config)

    assert entries
    [entry] = entries
    assert isinstance(entry, OverrideListEntry)
    assert entry.ns == "demo"
    assert entry.prompt == "example"
    assert entry.tag == "latest"
    assert entry.section_count == 1
    assert entry.tool_count == 1
    assert entry.backing_file_path == file_path.resolve()

    expected_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    assert entry.content_hash == expected_hash

    expected_timestamp = wink_overrides_module._truncate_to_milliseconds(
        datetime.fromtimestamp(file_path.stat().st_mtime, tz=UTC)
    )
    assert entry.updated_at == expected_timestamp


def test_list_overrides_applies_namespace_filter(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    config = MCPServerConfig(
        workspace_root=workspace,
        overrides_dir=Path(".wink/overrides"),
    )

    first = workspace / ".wink" / "overrides" / "demo" / "example"
    second = workspace / ".wink" / "overrides" / "other" / "sample"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "latest.json").write_text(
        '{"sections": {}, "tools": {}}', encoding="utf-8"
    )
    (second / "latest.json").write_text(
        '{"sections": {}, "tools": {}}', encoding="utf-8"
    )

    entries = list_overrides(config=config, namespace="demo")

    assert len(entries) == 1
    assert entries[0].ns == "demo"
    assert entries[0].prompt == "example"


def test_list_overrides_treats_blank_namespace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    config = MCPServerConfig(
        workspace_root=workspace,
        overrides_dir=Path(".wink/overrides"),
    )

    first = workspace / ".wink" / "overrides" / "demo" / "example"
    second = workspace / ".wink" / "overrides" / "other" / "sample"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "latest.json").write_text(
        '{"sections": {}, "tools": {}}', encoding="utf-8"
    )
    (second / "latest.json").write_text(
        '{"sections": {}, "tools": {}}', encoding="utf-8"
    )

    entries = list_overrides(config=config, namespace="   ")

    pairs = sorted((entry.ns, entry.prompt) for entry in entries)
    assert pairs == [("demo", "example"), ("other", "sample")]


def test_list_overrides_supports_absolute_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    overrides_root = tmp_path / "overrides"
    path = overrides_root / "demo" / "example"
    path.mkdir(parents=True)
    file_path = path / "latest.json"
    file_path.write_text('{"sections": {}, "tools": {}}', encoding="utf-8")

    config = MCPServerConfig(
        workspace_root=workspace,
        overrides_dir=overrides_root,
    )

    entries = list_overrides(config=config)

    assert len(entries) == 1
    assert entries[0].backing_file_path == file_path.resolve()


def test_list_overrides_wraps_iterator_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    config = MCPServerConfig(
        workspace_root=workspace,
        overrides_dir=Path(".wink/overrides"),
    )

    def fake_iter_override_files(**_: object) -> Iterator[OverrideFileMetadata]:
        raise PromptOverridesError("boom")

    monkeypatch.setattr(
        wink_overrides_module, "iter_override_files", fake_iter_override_files
    )

    with pytest.raises(OverridesInspectionError):
        list_overrides(config=config)


def test_list_overrides_wraps_prompt_errors(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    config = MCPServerConfig(
        workspace_root=workspace,
        overrides_dir=Path(".wink/overrides"),
    )

    overrides_root = workspace / ".wink" / "overrides" / "demo" / "example"
    overrides_root.mkdir(parents=True)
    (overrides_root / "latest.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(OverridesInspectionError):
        list_overrides(config=config)


def test_list_overrides_wraps_value_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    config = MCPServerConfig(
        workspace_root=workspace,
        overrides_dir=Path(".wink/overrides"),
    )

    metadata = OverrideFileMetadata(
        path=Path("invalid.json"),
        relative_segments=("demo",),
        modified_time=0.0,
        content_hash="hash",
        section_count=0,
        tool_count=0,
    )

    def fake_iter_override_files(**_: object) -> Iterator[OverrideFileMetadata]:
        return iter([metadata])

    monkeypatch.setattr(
        wink_overrides_module, "iter_override_files", fake_iter_override_files
    )

    with pytest.raises(OverridesInspectionError):
        list_overrides(config=config)


def test_build_override_list_entry_requires_namespace_segments() -> None:
    class FalseySegments(tuple[str, ...]):
        def __new__(cls, items: tuple[str, ...]) -> FalseySegments:
            return super().__new__(cls, items)

        def __getitem__(self, index: slice | int) -> object:
            result = super().__getitem__(index)
            if isinstance(index, slice):
                return FalseySegments(cast(tuple[str, ...], result))
            return result

        def __bool__(self) -> bool:
            return False

    metadata = OverrideFileMetadata(
        path=Path("invalid.json"),
        relative_segments=FalseySegments(("demo", "example", "latest.json")),
        modified_time=0.0,
        content_hash="hash",
        section_count=0,
        tool_count=0,
    )

    with pytest.raises(ValueError):
        _build_override_list_entry(metadata)


def _seed_override(
    *,
    store: LocalPromptOverridesStore,
    prompt_obj: Prompt[object],
    body: str,
) -> SectionOverride:
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    section_descriptor = next(section for section in descriptor.sections)
    override = SectionOverride(
        expected_hash=section_descriptor.content_hash,
        body=body,
    )
    prompt_override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="latest",
        sections={section_descriptor.path: override},
    )
    store.upsert(descriptor, prompt_override)
    return override


def _seed_tool_override(
    *,
    store: LocalPromptOverridesStore,
    prompt_obj: Prompt[object],
    tool_name: str,
    description: str,
    param_descriptions: Mapping[str, str],
) -> ToolOverride:
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    tool_descriptor = next(tool for tool in descriptor.tools if tool.name == tool_name)
    existing = store.resolve(descriptor, tag="latest")
    sections = dict(existing.sections) if existing else {}
    tools = dict(existing.tool_overrides) if existing else {}
    override = ToolOverride(
        name=tool_descriptor.name,
        expected_contract_hash=tool_descriptor.contract_hash,
        description=description,
        param_descriptions=dict(param_descriptions),
    )
    tools[tool_name] = override
    prompt_override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="latest",
        sections=sections,
        tool_overrides=tools,
    )
    store.upsert(descriptor, prompt_override)
    return override


def test_fetch_section_override_returns_override(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    store = _build_store(workspace)
    seeded = _seed_override(store=store, prompt_obj=prompt_obj, body="Custom body")

    config = _build_config(workspace, module_name)
    snapshot = fetch_section_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        section_path="intro",
    )

    expected_path = config.overrides_dir / "demo" / "example" / "latest.json"

    assert isinstance(snapshot, SectionOverrideSnapshot)
    assert snapshot.override_body == seeded.body
    assert snapshot.default_body == "Example body"
    assert snapshot.expected_hash == seeded.expected_hash
    assert snapshot.backing_file_path == expected_path
    assert snapshot.descriptor_version == PROMPT_DESCRIPTOR_VERSION


def test_fetch_section_override_returns_none_when_missing_override(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    store = _build_store(workspace)
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    _ = store.resolve(descriptor, tag="latest")  # ensure resolution path works

    config = _build_config(workspace, module_name)
    snapshot = fetch_section_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        section_path="intro",
    )

    assert snapshot.override_body is None
    assert snapshot.default_body == "Example body"


def test_fetch_section_override_raises_when_default_body_missing(
    prompt_workspace: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, module_name = prompt_workspace
    prompt_obj = _load_prompt(module_name, ("demo", "defaultless"))
    section_node = prompt_obj.sections[0]
    monkeypatch.setattr(section_node.section, "original_body_template", lambda: None)

    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    with pytest.raises(SectionOverridesUnavailableError):
        fetch_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="defaultless",
            tag="latest",
            section_path="intro",
        )


def test_fetch_section_override_raises_when_descriptor_missing(
    prompt_workspace: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    def _stub_descriptor(descriptor: PromptDescriptor, path: tuple[str, ...]) -> None:
        return None

    monkeypatch.setattr(
        "weakincentives.cli.wink_overrides._find_section_descriptor",
        _stub_descriptor,
    )

    with pytest.raises(SectionOverridesUnavailableError):
        fetch_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            section_path="intro",
        )


def test_fetch_section_override_raises_when_prompt_missing(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    with pytest.raises(PromptNotFoundError):
        fetch_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="missing",
            tag="latest",
            section_path="intro",
        )


def test_fetch_section_override_raises_when_section_missing(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    with pytest.raises(SectionNotFoundError):
        fetch_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            section_path="missing",
        )


def test_fetch_section_override_raises_when_overrides_disabled(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    with pytest.raises(SectionOverridesUnavailableError):
        fetch_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="disabled",
            tag="latest",
            section_path="intro",
        )


def test_fetch_section_override_maps_store_errors(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)

    class FailingStore(PromptOverridesStore):
        def resolve(
            self, descriptor: PromptDescriptor, tag: str = "latest"
        ) -> PromptOverride | None:
            raise PromptOverridesError("boom")

        def upsert(
            self, descriptor: PromptDescriptor, override: PromptOverride
        ) -> PromptOverride:
            raise NotImplementedError

        def delete(self, *, ns: str, prompt_key: str, tag: str) -> None:
            raise NotImplementedError

        def seed_if_necessary(
            self, prompt: object, *, tag: str = "latest"
        ) -> PromptOverride:
            raise NotImplementedError

    store = FailingStore()

    with pytest.raises(SectionOverrideResolutionError):
        fetch_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            section_path="intro",
        )


def test_fetch_section_override_maps_import_errors(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = MCPServerConfig(
        workspace_root=workspace,
        overrides_dir=workspace / ".wink" / "overrides",
        prompt_registry_modules=("missing.module",),
    )
    store = _build_store(workspace)

    with pytest.raises(PromptRegistryImportError):
        fetch_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            section_path="intro",
        )


def test_apply_section_override_persists_body(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    section_descriptor = next(section for section in descriptor.sections)

    result = apply_section_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        section_path="intro",
        body="Updated body",
        expected_hash=section_descriptor.content_hash,
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
        confirm=True,
    )

    assert isinstance(result, SectionOverrideMutationResult)
    assert result.override_body == "Updated body"
    assert result.section_path == section_descriptor.path
    assert result.expected_hash == section_descriptor.content_hash
    assert result.descriptor_version == PROMPT_DESCRIPTOR_VERSION
    assert result.backing_file_path.exists()
    assert result.updated_at is not None
    assert result.updated_at.tzinfo is UTC

    persisted = store.resolve(descriptor, tag="latest")
    assert persisted is not None
    assert persisted.sections[section_descriptor.path].body == result.override_body


def test_apply_section_override_requires_confirm(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    section_descriptor = next(section for section in descriptor.sections)

    with pytest.raises(SectionOverrideApplyError):
        apply_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            section_path="intro",
            body="Updated body",
            expected_hash=section_descriptor.content_hash,
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=False,
        )


def test_apply_section_override_raises_when_section_missing(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    section_descriptor = next(section for section in descriptor.sections)

    with pytest.raises(SectionNotFoundError):
        apply_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            section_path="missing",
            body="Updated body",
            expected_hash=section_descriptor.content_hash,
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=True,
        )


def test_apply_section_override_rejects_disabled_section(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    with pytest.raises(SectionOverridesUnavailableError):
        apply_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="disabled",
            tag="latest",
            section_path="intro",
            body="Updated body",
            expected_hash="placeholder",
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=True,
        )


def test_apply_section_override_rejects_missing_descriptor(
    prompt_workspace: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    section_descriptor = next(section for section in descriptor.sections)

    monkeypatch.setattr(
        wink_overrides_module,
        "_find_section_descriptor",
        lambda _descriptor, _path: None,
    )

    with pytest.raises(SectionOverridesUnavailableError):
        apply_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            section_path="intro",
            body="Updated body",
            expected_hash=section_descriptor.content_hash,
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=True,
        )


def test_apply_section_override_rejects_descriptor_version_mismatch(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    section_descriptor = next(section for section in descriptor.sections)

    with pytest.raises(SectionOverrideApplyError):
        apply_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            section_path="intro",
            body="Updated body",
            expected_hash=section_descriptor.content_hash,
            descriptor_version=PROMPT_DESCRIPTOR_VERSION + 1,
            confirm=True,
        )


def test_apply_section_override_requires_expected_hash(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    with pytest.raises(SectionOverrideApplyError):
        apply_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            section_path="intro",
            body="Updated body",
            expected_hash=None,
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=True,
        )


def test_apply_section_override_maps_resolve_errors(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)

    class ResolveErrorStore(PromptOverridesStore):
        def resolve(
            self, descriptor: PromptDescriptor, *, tag: str = "latest"
        ) -> PromptOverride | None:
            raise PromptOverridesError("boom")

        def upsert(
            self, descriptor: PromptDescriptor, override: PromptOverride
        ) -> PromptOverride:
            raise NotImplementedError

        def delete(self, *, ns: str, prompt_key: str, tag: str) -> None:
            raise NotImplementedError

        def seed_if_necessary(
            self, prompt: object, *, tag: str = "latest"
        ) -> PromptOverride:
            raise NotImplementedError

    store = ResolveErrorStore()
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    section_descriptor = next(section for section in descriptor.sections)

    with pytest.raises(SectionOverrideApplyError):
        apply_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            section_path="intro",
            body="Updated body",
            expected_hash=section_descriptor.content_hash,
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=True,
        )


def test_apply_section_override_maps_upsert_errors(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)

    class UpsertErrorStore(PromptOverridesStore):
        def resolve(
            self, descriptor: PromptDescriptor, *, tag: str = "latest"
        ) -> PromptOverride | None:
            return None

        def upsert(
            self, descriptor: PromptDescriptor, override: PromptOverride
        ) -> PromptOverride:
            raise PromptOverridesError("boom")

        def delete(self, *, ns: str, prompt_key: str, tag: str) -> None:
            raise NotImplementedError

        def seed_if_necessary(
            self, prompt: object, *, tag: str = "latest"
        ) -> PromptOverride:
            raise NotImplementedError

    store = UpsertErrorStore()
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    section_descriptor = next(section for section in descriptor.sections)

    with pytest.raises(SectionOverrideApplyError):
        apply_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            section_path="intro",
            body="Updated body",
            expected_hash=section_descriptor.content_hash,
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=True,
        )


def test_apply_section_override_detects_missing_persisted_section(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)

    class MissingPersistedStore(PromptOverridesStore):
        def resolve(
            self, descriptor: PromptDescriptor, *, tag: str = "latest"
        ) -> PromptOverride | None:
            return None

        def upsert(
            self, descriptor: PromptDescriptor, override: PromptOverride
        ) -> PromptOverride:
            return PromptOverride(
                ns=descriptor.ns,
                prompt_key=descriptor.key,
                tag=override.tag,
                sections={},
            )

        def delete(self, *, ns: str, prompt_key: str, tag: str) -> None:
            raise NotImplementedError

        def seed_if_necessary(
            self, prompt: object, *, tag: str = "latest"
        ) -> PromptOverride:
            raise NotImplementedError

    store = MissingPersistedStore()
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    section_descriptor = next(section for section in descriptor.sections)

    with pytest.raises(SectionOverrideApplyError):
        apply_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            section_path="intro",
            body="Updated body",
            expected_hash=section_descriptor.content_hash,
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=True,
        )


def test_apply_section_override_uses_now_when_stat_missing(
    prompt_workspace: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    section_descriptor = next(section for section in descriptor.sections)

    sentinel = wink_overrides_module.datetime(2024, 1, 1, tzinfo=UTC)
    monkeypatch.setattr(wink_overrides_module, "_stat_timestamp", lambda _path: None)
    monkeypatch.setattr(wink_overrides_module, "_now", lambda: sentinel)

    result = apply_section_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        section_path="intro",
        body="Updated body",
        expected_hash=section_descriptor.content_hash,
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
        confirm=True,
    )

    assert result.updated_at == sentinel


def test_apply_section_override_rejects_hash_mismatch(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    with pytest.raises(SectionOverrideApplyError):
        apply_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            section_path="intro",
            body="Updated body",
            expected_hash="not-a-real-hash",
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=True,
        )


def test_remove_section_override_deletes_file_when_empty(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    section_descriptor = next(section for section in descriptor.sections)
    _seed_override(store=store, prompt_obj=prompt_obj, body="Custom body")

    result = remove_section_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        section_path="intro",
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
    )

    assert isinstance(result, SectionOverrideMutationResult)
    assert result.override_body is None
    assert result.warnings == ()
    assert result.expected_hash == section_descriptor.content_hash
    assert result.descriptor_version == PROMPT_DESCRIPTOR_VERSION
    assert result.updated_at is not None
    assert result.updated_at.tzinfo is UTC
    assert not result.backing_file_path.exists()

    persisted = store.resolve(descriptor, tag="latest")
    assert persisted is None


def test_remove_section_override_warns_when_missing(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    section_descriptor = next(section for section in descriptor.sections)

    result = remove_section_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        section_path="intro",
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
    )

    assert isinstance(result, SectionOverrideMutationResult)
    assert result.override_body is None
    assert result.warnings
    assert result.expected_hash == section_descriptor.content_hash
    assert result.descriptor_version == PROMPT_DESCRIPTOR_VERSION
    assert result.updated_at is None
    assert not result.backing_file_path.exists()

    persisted = store.resolve(descriptor, tag="latest")
    assert persisted is None


def test_remove_section_override_raises_when_section_missing(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    with pytest.raises(SectionNotFoundError):
        remove_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            section_path="missing",
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
        )


def test_remove_section_override_rejects_disabled_section(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    with pytest.raises(SectionOverridesUnavailableError):
        remove_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="disabled",
            tag="latest",
            section_path="intro",
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
        )


def test_remove_section_override_rejects_missing_descriptor(
    prompt_workspace: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    monkeypatch.setattr(
        wink_overrides_module,
        "_find_section_descriptor",
        lambda _descriptor, _path: None,
    )

    with pytest.raises(SectionOverridesUnavailableError):
        remove_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            section_path="intro",
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
        )


def test_remove_section_override_rejects_descriptor_version_mismatch(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    with pytest.raises(SectionOverrideRemoveError):
        remove_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            section_path="intro",
            descriptor_version=PROMPT_DESCRIPTOR_VERSION + 1,
        )


def test_remove_section_override_maps_resolve_errors(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)

    class ResolveErrorStore(PromptOverridesStore):
        def resolve(
            self, descriptor: PromptDescriptor, *, tag: str = "latest"
        ) -> PromptOverride | None:
            raise PromptOverridesError("boom")

        def upsert(
            self, descriptor: PromptDescriptor, override: PromptOverride
        ) -> PromptOverride:
            raise NotImplementedError

        def delete(self, *, ns: str, prompt_key: str, tag: str) -> None:
            raise NotImplementedError

        def seed_if_necessary(
            self, prompt: object, *, tag: str = "latest"
        ) -> PromptOverride:
            raise NotImplementedError

    store = ResolveErrorStore()

    with pytest.raises(SectionOverrideRemoveError):
        remove_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            section_path="intro",
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
        )


def test_remove_section_override_maps_upsert_errors(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)

    class UpsertErrorStore(PromptOverridesStore):
        def resolve(
            self, descriptor: PromptDescriptor, *, tag: str = "latest"
        ) -> PromptOverride | None:
            return PromptOverride(
                ns=descriptor.ns,
                prompt_key=descriptor.key,
                tag=tag,
                sections={
                    ("intro",): SectionOverride(
                        expected_hash="hash-1",
                        body="Body",
                    ),
                    ("other",): SectionOverride(
                        expected_hash="hash-2",
                        body="Other",
                    ),
                },
            )

        def upsert(
            self, descriptor: PromptDescriptor, override: PromptOverride
        ) -> PromptOverride:
            raise PromptOverridesError("boom")

        def delete(self, *, ns: str, prompt_key: str, tag: str) -> None:
            raise NotImplementedError

        def seed_if_necessary(
            self, prompt: object, *, tag: str = "latest"
        ) -> PromptOverride:
            raise NotImplementedError

    store = UpsertErrorStore()

    with pytest.raises(SectionOverrideRemoveError):
        remove_section_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            section_path="intro",
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
        )


def test_remove_section_override_updates_existing_overrides(
    prompt_workspace: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)

    class MultiSectionStore(PromptOverridesStore):
        def __init__(self) -> None:
            self.persisted: PromptOverride | None = None

        def resolve(
            self, descriptor: PromptDescriptor, *, tag: str = "latest"
        ) -> PromptOverride | None:
            return PromptOverride(
                ns=descriptor.ns,
                prompt_key=descriptor.key,
                tag=tag,
                sections={
                    ("intro",): SectionOverride(
                        expected_hash="hash-1",
                        body="Custom body",
                    ),
                    ("other",): SectionOverride(
                        expected_hash="hash-2",
                        body="Other body",
                    ),
                },
            )

        def upsert(
            self, descriptor: PromptDescriptor, override: PromptOverride
        ) -> PromptOverride:
            self.persisted = override
            return override

        def delete(self, *, ns: str, prompt_key: str, tag: str) -> None:
            raise NotImplementedError

        def seed_if_necessary(
            self, prompt: object, *, tag: str = "latest"
        ) -> PromptOverride:
            raise NotImplementedError

    store = MultiSectionStore()
    sentinel = datetime(2025, 1, 1, tzinfo=UTC)
    monkeypatch.setattr(wink_overrides_module, "_stat_timestamp", lambda _path: None)
    monkeypatch.setattr(wink_overrides_module, "_now", lambda: sentinel)

    result = remove_section_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        section_path="intro",
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
    )

    assert isinstance(store.persisted, PromptOverride)
    assert ("intro",) not in store.persisted.sections
    assert result.updated_at == sentinel


def test_now_truncates_to_milliseconds() -> None:
    timestamp = wink_overrides_module._now()

    assert timestamp.tzinfo is UTC
    assert timestamp.microsecond % 1000 == 0


def test_remove_section_override_uses_now_when_stat_missing_on_delete(
    prompt_workspace: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    _seed_override(store=store, prompt_obj=prompt_obj, body="Custom body")

    sentinel = datetime(2030, 1, 1, tzinfo=UTC)
    monkeypatch.setattr(wink_overrides_module, "_stat_timestamp", lambda _path: None)
    monkeypatch.setattr(wink_overrides_module, "_now", lambda: sentinel)

    result = remove_section_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        section_path="intro",
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
    )

    assert result.updated_at == sentinel


def test_fetch_tool_override_returns_override(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    seeded = _seed_tool_override(
        store=store,
        prompt_obj=prompt_obj,
        tool_name="search_docs",
        description="Override search",
        param_descriptions={"query": "Override query"},
    )

    config = _build_config(workspace, module_name)
    snapshot = fetch_tool_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        tool_name="search_docs",
    )

    expected_path = config.overrides_dir / "demo" / "example" / "latest.json"

    assert isinstance(snapshot, ToolOverrideSnapshot)
    assert snapshot.override_description == seeded.description
    assert snapshot.description == seeded.description
    assert snapshot.override_param_descriptions == seeded.param_descriptions
    assert snapshot.param_descriptions["query"] == "Override query"
    assert snapshot.default_description == "Search the documentation."
    assert snapshot.default_param_descriptions == {"query": "Search query"}
    assert snapshot.backing_file_path == expected_path


def test_fetch_tool_override_returns_defaults_when_missing_override(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    store = _build_store(workspace)
    _ = _load_prompt(module_name, ("demo", "example"))

    config = _build_config(workspace, module_name)
    snapshot = fetch_tool_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        tool_name="search_docs",
    )

    assert snapshot.override_description is None
    assert snapshot.description == "Search the documentation."
    assert snapshot.param_descriptions == {"query": "Search query"}


def test_fetch_tool_override_raises_when_tool_missing(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    store = _build_store(workspace)
    config = _build_config(workspace, module_name)

    with pytest.raises(ToolNotFoundError):
        fetch_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="missing",
        )


def test_fetch_tool_override_raises_when_tool_overrides_disabled(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    store = _build_store(workspace)
    config = _build_config(workspace, module_name)

    with pytest.raises(ToolOverridesUnavailableError):
        fetch_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="tool_disabled",
            tag="latest",
            tool_name="search_disabled",
        )


def test_fetch_tool_override_maps_store_errors(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)

    class FailingStore(PromptOverridesStore):
        def resolve(
            self, descriptor: PromptDescriptor, tag: str = "latest"
        ) -> PromptOverride | None:
            raise PromptOverridesError("boom")

        def upsert(
            self, descriptor: PromptDescriptor, override: PromptOverride
        ) -> PromptOverride:
            raise NotImplementedError

        def delete(self, *, ns: str, prompt_key: str, tag: str) -> None:
            raise NotImplementedError

        def seed_if_necessary(
            self, prompt: object, *, tag: str = "latest"
        ) -> PromptOverride:
            raise NotImplementedError

    store = FailingStore()

    with pytest.raises(ToolOverrideResolutionError):
        fetch_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="search_docs",
        )


def test_fetch_tool_override_raises_when_descriptor_missing(
    prompt_workspace: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, module_name = prompt_workspace
    store = _build_store(workspace)
    config = _build_config(workspace, module_name)

    def _stub_descriptor(descriptor: PromptDescriptor, tool_name: str) -> None:
        return None

    monkeypatch.setattr(
        "weakincentives.cli.wink_overrides._find_tool_descriptor",
        _stub_descriptor,
    )

    with pytest.raises(ToolOverridesUnavailableError):
        fetch_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="search_docs",
        )


def test_apply_tool_override_persists_payload(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    tool_descriptor = next(
        tool for tool in descriptor.tools if tool.name == "search_docs"
    )

    result = apply_tool_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        tool_name="search_docs",
        description="Override search",
        param_descriptions={"query": "Override query"},
        expected_contract_hash=tool_descriptor.contract_hash,
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
        confirm=True,
    )

    override = store.resolve(descriptor, tag="latest")
    assert override is not None
    persisted = override.tool_overrides["search_docs"]

    assert isinstance(result, ToolOverrideMutationResult)
    assert persisted.description == "Override search"
    assert persisted.param_descriptions == {"query": "Override query"}
    assert result.override_description == "Override search"
    assert result.description == "Override search"
    assert result.param_descriptions == {"query": "Override query"}


def test_apply_tool_override_requires_confirm(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    tool_descriptor = next(
        tool for tool in descriptor.tools if tool.name == "search_docs"
    )

    with pytest.raises(ToolOverrideApplyError):
        apply_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="search_docs",
            description="Override search",
            param_descriptions={},
            expected_contract_hash=tool_descriptor.contract_hash,
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=False,
        )


def test_apply_tool_override_raises_when_tool_missing(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    with pytest.raises(ToolNotFoundError):
        apply_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="missing",
            description="Override",
            param_descriptions={},
            expected_contract_hash="hash",
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=True,
        )


def test_apply_tool_override_rejects_disabled_tool(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    with pytest.raises(ToolOverridesUnavailableError):
        apply_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="tool_disabled",
            tag="latest",
            tool_name="search_disabled",
            description="Override",
            param_descriptions={},
            expected_contract_hash="hash",
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=True,
        )


def test_apply_tool_override_rejects_missing_descriptor(
    prompt_workspace: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    def _stub_descriptor(descriptor: PromptDescriptor, tool_name: str) -> None:
        return None

    monkeypatch.setattr(
        "weakincentives.cli.wink_overrides._find_tool_descriptor",
        _stub_descriptor,
    )

    with pytest.raises(ToolOverridesUnavailableError):
        apply_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="search_docs",
            description="Override",
            param_descriptions={},
            expected_contract_hash="hash",
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=True,
        )


def test_apply_tool_override_rejects_descriptor_version_mismatch(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    tool_descriptor = next(
        tool for tool in descriptor.tools if tool.name == "search_docs"
    )

    with pytest.raises(ToolOverrideApplyError):
        apply_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="search_docs",
            description="Override",
            param_descriptions={},
            expected_contract_hash=tool_descriptor.contract_hash,
            descriptor_version=-1,
            confirm=True,
        )


def test_apply_tool_override_requires_expected_hash(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    with pytest.raises(ToolOverrideApplyError):
        apply_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="search_docs",
            description="Override",
            param_descriptions={},
            expected_contract_hash=None,
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=True,
        )


def test_apply_tool_override_validates_contract_hash(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    with pytest.raises(ToolOverrideApplyError):
        apply_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="search_docs",
            description="Override",
            param_descriptions={},
            expected_contract_hash="invalid",
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=True,
        )


def test_apply_tool_override_rejects_invalid_param_mapping(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    tool_descriptor = next(
        tool for tool in descriptor.tools if tool.name == "search_docs"
    )

    with pytest.raises(ToolOverrideApplyError):
        apply_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="search_docs",
            description="Override",
            param_descriptions={1: "bad"},
            expected_contract_hash=tool_descriptor.contract_hash,
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=True,
        )


def test_apply_tool_override_rejects_non_mapping_param_descriptions(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    tool_descriptor = next(
        tool for tool in descriptor.tools if tool.name == "search_docs"
    )

    with pytest.raises(ToolOverrideApplyError):
        apply_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="search_docs",
            description="Override",
            param_descriptions=[("query", "value")],
            expected_contract_hash=tool_descriptor.contract_hash,
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=True,
        )


def test_apply_tool_override_preserves_existing_overrides(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    _seed_tool_override(
        store=store,
        prompt_obj=prompt_obj,
        tool_name="index_docs",
        description="Index override",
        param_descriptions={"document_id": "Override id"},
    )
    tool_descriptor = next(
        tool for tool in descriptor.tools if tool.name == "search_docs"
    )

    _ = apply_tool_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        tool_name="search_docs",
        description="Override search",
        param_descriptions={"query": "Override query"},
        expected_contract_hash=tool_descriptor.contract_hash,
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
        confirm=True,
    )

    persisted = store.resolve(descriptor, tag="latest")
    assert persisted is not None
    assert "index_docs" in persisted.tool_overrides


def test_apply_tool_override_maps_resolve_errors(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    tool_descriptor = next(
        tool for tool in descriptor.tools if tool.name == "search_docs"
    )

    class FailingStore(PromptOverridesStore):
        def resolve(
            self, descriptor: PromptDescriptor, tag: str = "latest"
        ) -> PromptOverride | None:
            raise PromptOverridesError("boom")

        def upsert(
            self, descriptor: PromptDescriptor, override: PromptOverride
        ) -> PromptOverride:
            raise NotImplementedError

        def delete(self, *, ns: str, prompt_key: str, tag: str) -> None:
            raise NotImplementedError

        def seed_if_necessary(
            self, prompt: object, *, tag: str = "latest"
        ) -> PromptOverride:
            raise NotImplementedError

    store = FailingStore()

    with pytest.raises(ToolOverrideApplyError):
        apply_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="search_docs",
            description="Override",
            param_descriptions={},
            expected_contract_hash=tool_descriptor.contract_hash,
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=True,
        )


def test_apply_tool_override_maps_upsert_errors(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    tool_descriptor = next(
        tool for tool in descriptor.tools if tool.name == "search_docs"
    )

    class FailingStore(PromptOverridesStore):
        def resolve(
            self, descriptor: PromptDescriptor, tag: str = "latest"
        ) -> PromptOverride | None:
            return PromptOverride(
                ns=descriptor.ns,
                prompt_key=descriptor.key,
                tag=tag,
            )

        def upsert(
            self, descriptor: PromptDescriptor, override: PromptOverride
        ) -> PromptOverride:
            raise PromptOverridesError("boom")

        def delete(self, *, ns: str, prompt_key: str, tag: str) -> None:
            raise NotImplementedError

        def seed_if_necessary(
            self, prompt: object, *, tag: str = "latest"
        ) -> PromptOverride:
            raise NotImplementedError

    store = FailingStore()

    with pytest.raises(ToolOverrideApplyError):
        apply_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="search_docs",
            description="Override",
            param_descriptions={},
            expected_contract_hash=tool_descriptor.contract_hash,
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=True,
        )


def test_apply_tool_override_detects_missing_persisted_tool(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    tool_descriptor = next(
        tool for tool in descriptor.tools if tool.name == "search_docs"
    )

    class MissingPersistedStore(PromptOverridesStore):
        def resolve(
            self, descriptor: PromptDescriptor, tag: str = "latest"
        ) -> PromptOverride | None:
            return PromptOverride(ns=descriptor.ns, prompt_key=descriptor.key, tag=tag)

        def upsert(
            self, descriptor: PromptDescriptor, override: PromptOverride
        ) -> PromptOverride:
            return PromptOverride(
                ns=descriptor.ns, prompt_key=descriptor.key, tag=override.tag
            )

        def delete(self, *, ns: str, prompt_key: str, tag: str) -> None:
            raise NotImplementedError

        def seed_if_necessary(
            self, prompt: object, *, tag: str = "latest"
        ) -> PromptOverride:
            raise NotImplementedError

    store = MissingPersistedStore()

    with pytest.raises(ToolOverrideApplyError):
        apply_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="search_docs",
            description="Override",
            param_descriptions={},
            expected_contract_hash=tool_descriptor.contract_hash,
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            confirm=True,
        )


def test_apply_tool_override_uses_now_when_stat_missing(
    prompt_workspace: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    tool_descriptor = next(
        tool for tool in descriptor.tools if tool.name == "search_docs"
    )

    sentinel = datetime(2030, 1, 1, tzinfo=UTC)
    monkeypatch.setattr(wink_overrides_module, "_stat_timestamp", lambda _path: None)
    monkeypatch.setattr(wink_overrides_module, "_now", lambda: sentinel)

    result = apply_tool_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        tool_name="search_docs",
        description="Override",
        param_descriptions={},
        expected_contract_hash=tool_descriptor.contract_hash,
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
        confirm=True,
    )

    assert result.updated_at == sentinel


def test_remove_tool_override_removes_entry(prompt_workspace: tuple[Path, str]) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    _seed_tool_override(
        store=store,
        prompt_obj=prompt_obj,
        tool_name="search_docs",
        description="Override search",
        param_descriptions={"query": "Override query"},
    )

    result = remove_tool_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        tool_name="search_docs",
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
    )

    override = store.resolve(descriptor, tag="latest")
    assert override is None or "search_docs" not in override.tool_overrides
    assert result.override_description is None
    assert result.description == "Search the documentation."
    assert result.warnings == ()


def test_remove_tool_override_warns_when_missing(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    result = remove_tool_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        tool_name="search_docs",
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
    )

    assert result.warnings == (
        "No override found for tool search_docs. Nothing to remove.",
    )
    assert result.description == "Search the documentation."


def test_remove_tool_override_warns_when_tool_absent_from_override(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    _seed_tool_override(
        store=store,
        prompt_obj=prompt_obj,
        tool_name="index_docs",
        description="Override index",
        param_descriptions={"document_id": "Override id"},
    )

    result = remove_tool_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        tool_name="search_docs",
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
    )

    assert result.warnings == (
        "No override found for tool search_docs. Nothing to remove.",
    )


def test_remove_tool_override_warns_when_delete_missing_file(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)

    tool_descriptor = next(
        tool for tool in descriptor.tools if tool.name == "search_docs"
    )

    class MissingDeleteStore(PromptOverridesStore):
        def resolve(
            self, descriptor: PromptDescriptor, tag: str = "latest"
        ) -> PromptOverride | None:
            override = PromptOverride(
                ns=descriptor.ns,
                prompt_key=descriptor.key,
                tag=tag,
                tool_overrides={},
            )
            override.tool_overrides["search_docs"] = ToolOverride(
                name="search_docs",
                expected_contract_hash=tool_descriptor.contract_hash,
                description="Override",
                param_descriptions={},
            )
            return override

        def upsert(
            self, descriptor: PromptDescriptor, override: PromptOverride
        ) -> PromptOverride:
            raise NotImplementedError

        def delete(self, *, ns: str, prompt_key: str, tag: str) -> None:
            raise FileNotFoundError

        def seed_if_necessary(
            self, prompt: object, *, tag: str = "latest"
        ) -> PromptOverride:
            raise NotImplementedError

    store = MissingDeleteStore()

    result = remove_tool_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        tool_name="search_docs",
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
    )

    assert result.warnings == (
        "Override file missing while removing tool override. Nothing to delete.",
    )


def test_remove_tool_override_maps_delete_errors(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)

    tool_descriptor = next(
        tool for tool in descriptor.tools if tool.name == "search_docs"
    )

    class FailingDeleteStore(PromptOverridesStore):
        def resolve(
            self, descriptor: PromptDescriptor, tag: str = "latest"
        ) -> PromptOverride | None:
            override = PromptOverride(
                ns=descriptor.ns,
                prompt_key=descriptor.key,
                tag=tag,
                tool_overrides={},
            )
            override.tool_overrides["search_docs"] = ToolOverride(
                name="search_docs",
                expected_contract_hash=tool_descriptor.contract_hash,
                description="Override",
                param_descriptions={},
            )
            return override

        def upsert(
            self, descriptor: PromptDescriptor, override: PromptOverride
        ) -> PromptOverride:
            raise NotImplementedError

        def delete(self, *, ns: str, prompt_key: str, tag: str) -> None:
            raise PromptOverridesError("boom")

        def seed_if_necessary(
            self, prompt: object, *, tag: str = "latest"
        ) -> PromptOverride:
            raise NotImplementedError

    store = FailingDeleteStore()

    with pytest.raises(ToolOverrideRemoveError):
        remove_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="search_docs",
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
        )


def test_remove_tool_override_preserves_other_entries(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)
    _seed_tool_override(
        store=store,
        prompt_obj=prompt_obj,
        tool_name="search_docs",
        description="Override search",
        param_descriptions={"query": "Override query"},
    )
    _seed_tool_override(
        store=store,
        prompt_obj=prompt_obj,
        tool_name="index_docs",
        description="Override index",
        param_descriptions={"document_id": "Override id"},
    )

    _ = remove_tool_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        tool_name="search_docs",
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
    )

    persisted = store.resolve(descriptor, tag="latest")
    assert persisted is not None
    assert "index_docs" in persisted.tool_overrides


def test_remove_tool_override_rejects_missing_descriptor(
    prompt_workspace: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    def _stub_descriptor(descriptor: PromptDescriptor, tool_name: str) -> None:
        return None

    monkeypatch.setattr(
        "weakincentives.cli.wink_overrides._find_tool_descriptor",
        _stub_descriptor,
    )

    with pytest.raises(ToolOverridesUnavailableError):
        remove_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="search_docs",
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
        )


def test_remove_tool_override_uses_now_when_stat_missing(
    prompt_workspace: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    _seed_tool_override(
        store=store,
        prompt_obj=prompt_obj,
        tool_name="search_docs",
        description="Override search",
        param_descriptions={"query": "Override query"},
    )
    _seed_tool_override(
        store=store,
        prompt_obj=prompt_obj,
        tool_name="index_docs",
        description="Override index",
        param_descriptions={"document_id": "Override id"},
    )

    sentinel = datetime(2030, 1, 1, tzinfo=UTC)
    monkeypatch.setattr(wink_overrides_module, "_stat_timestamp", lambda _path: None)
    monkeypatch.setattr(wink_overrides_module, "_now", lambda: sentinel)

    result = remove_tool_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        tool_name="search_docs",
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
    )

    assert result.updated_at == sentinel


def test_remove_tool_override_validates_descriptor_version(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    with pytest.raises(ToolOverrideRemoveError):
        remove_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="search_docs",
            descriptor_version=-1,
        )


def test_remove_tool_override_uses_now_when_stat_missing_on_delete(
    prompt_workspace: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    _seed_tool_override(
        store=store,
        prompt_obj=prompt_obj,
        tool_name="search_docs",
        description="Override search",
        param_descriptions={"query": "Override query"},
    )

    sentinel = datetime(2035, 5, 5, tzinfo=UTC)
    monkeypatch.setattr(wink_overrides_module, "_stat_timestamp", lambda _path: None)
    monkeypatch.setattr(wink_overrides_module, "_now", lambda: sentinel)

    result = remove_tool_override(
        config=config,
        store=store,
        ns="demo",
        prompt="example",
        tag="latest",
        tool_name="search_docs",
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
    )

    assert result.updated_at == sentinel


def test_remove_tool_override_maps_store_errors(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)

    class FailingStore(PromptOverridesStore):
        def resolve(
            self, descriptor: PromptDescriptor, tag: str = "latest"
        ) -> PromptOverride | None:
            raise PromptOverridesError("boom")

        def upsert(
            self, descriptor: PromptDescriptor, override: PromptOverride
        ) -> PromptOverride:
            raise NotImplementedError

        def delete(self, *, ns: str, prompt_key: str, tag: str) -> None:
            raise NotImplementedError

        def seed_if_necessary(
            self, prompt: object, *, tag: str = "latest"
        ) -> PromptOverride:
            raise NotImplementedError

    store = FailingStore()

    with pytest.raises(ToolOverrideRemoveError):
        remove_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="search_docs",
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
        )


def test_remove_tool_override_maps_upsert_errors(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)

    tool_descriptor = next(
        tool for tool in descriptor.tools if tool.name == "search_docs"
    )

    class FailingStore(PromptOverridesStore):
        def __init__(self) -> None:
            self.upsert_calls = 0

        def resolve(
            self, descriptor: PromptDescriptor, tag: str = "latest"
        ) -> PromptOverride | None:
            override = PromptOverride(
                ns=descriptor.ns,
                prompt_key=descriptor.key,
                tag=tag,
                tool_overrides={},
            )
            override.tool_overrides["search_docs"] = ToolOverride(
                name="search_docs",
                expected_contract_hash=tool_descriptor.contract_hash,
                description="Override",
                param_descriptions={},
            )
            override.tool_overrides["index_docs"] = ToolOverride(
                name="index_docs",
                expected_contract_hash=next(
                    tool.contract_hash
                    for tool in descriptor.tools
                    if tool.name == "index_docs"
                ),
                description="Other override",
                param_descriptions={},
            )
            return override

        def upsert(
            self, descriptor: PromptDescriptor, override: PromptOverride
        ) -> PromptOverride:
            self.upsert_calls += 1
            raise PromptOverridesError("boom")

        def delete(self, *, ns: str, prompt_key: str, tag: str) -> None:
            raise NotImplementedError

        def seed_if_necessary(
            self, prompt: object, *, tag: str = "latest"
        ) -> PromptOverride:
            raise NotImplementedError

    store = FailingStore()

    with pytest.raises(ToolOverrideRemoveError):
        remove_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="search_docs",
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
        )

    assert store.upsert_calls == 1


def test_remove_tool_override_rejects_disabled_tool(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    with pytest.raises(ToolOverridesUnavailableError):
        remove_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="tool_disabled",
            tag="latest",
            tool_name="search_disabled",
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
        )


def test_remove_tool_override_rejects_missing_tool(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    config = _build_config(workspace, module_name)
    store = _build_store(workspace)

    with pytest.raises(ToolNotFoundError):
        remove_tool_override(
            config=config,
            store=store,
            ns="demo",
            prompt="example",
            tag="latest",
            tool_name="missing",
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
        )


def test_collect_tool_param_descriptions_returns_empty_when_params_not_dataclass(
    prompt_workspace: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace, module_name = prompt_workspace
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    tool = prompt_obj.sections[0].section.tools()[0]
    original = tool.params_type
    monkeypatch.setattr(tool, "params_type", object, raising=False)

    descriptions = wink_overrides_module._collect_tool_param_descriptions(tool)

    assert descriptions == {}
    monkeypatch.setattr(tool, "params_type", original, raising=False)


def test_find_tool_descriptor_returns_none(
    prompt_workspace: tuple[Path, str],
) -> None:
    _workspace, module_name = prompt_workspace
    prompt_obj = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt_obj)

    result = wink_overrides_module._find_tool_descriptor(descriptor, "missing")

    assert result is None


def test_extract_override_body_returns_none_when_section_missing() -> None:
    override = PromptOverride(
        ns="demo",
        prompt_key="example",
        tag="latest",
        sections={},
    )
    assert _extract_override_body(override, ("intro",)) is None


def test_parse_section_path_raises_on_empty() -> None:
    with pytest.raises(ValueError):
        _parse_section_path(" ")


def test_iter_registry_prompts_returns_empty_when_modules_missing(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = MCPServerConfig(
        workspace_root=workspace,
        overrides_dir=workspace / ".wink" / "overrides",
        prompt_registry_modules=(),
    )
    assert list(_iter_registry_prompts(config)) == []


def test_iter_prompts_from_modules_skips_blank_entries(
    prompt_workspace: tuple[Path, str],
) -> None:
    workspace, module_name = prompt_workspace
    prompts = list(_iter_prompts_from_modules(workspace, ("", module_name)))
    assert prompts


def test_iter_prompts_from_modules_requires_prompts_attribute(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    module_dir = workspace / "broken_prompts"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("__all__ = []\n")

    importlib.invalidate_caches()

    config = MCPServerConfig(
        workspace_root=workspace,
        overrides_dir=workspace / ".wink" / "overrides",
        prompt_registry_modules=("broken_prompts",),
    )

    try:
        with pytest.raises(PromptRegistryImportError):
            list(_iter_registry_prompts(config))
    finally:
        sys.modules.pop("broken_prompts", None)


def test_iter_prompts_from_modules_requires_mapping_prompts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    module_dir = workspace / "invalid_prompts"
    module_dir.mkdir()
    module_code = "PROMPTS = []\n"
    (module_dir / "__init__.py").write_text(module_code)

    importlib.invalidate_caches()

    config = MCPServerConfig(
        workspace_root=workspace,
        overrides_dir=workspace / ".wink" / "overrides",
        prompt_registry_modules=("invalid_prompts",),
    )

    try:
        with pytest.raises(PromptRegistryImportError):
            list(_iter_registry_prompts(config))
    finally:
        sys.modules.pop("invalid_prompts", None)


def test_iter_prompts_from_registry_validates_metadata(
    prompt_workspace: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace, module_name = prompt_workspace
    prompt = _load_prompt(module_name, ("demo", "example"))
    monkeypatch.setattr(prompt, "ns", "other")

    registry: Mapping[object, object] = {("demo", "example"): prompt}
    iterator = _iter_prompts_from_registry(module_name, registry)

    with pytest.raises(PromptRegistryImportError):
        next(iterator)


def test_normalize_registry_key_validations() -> None:
    with pytest.raises(PromptRegistryImportError):
        _normalize_registry_key("module", ("only",))

    with pytest.raises(PromptRegistryImportError):
        _normalize_registry_key("module", ("demo", 1))

    with pytest.raises(PromptRegistryImportError):
        _normalize_registry_key("module", "demo")

    with pytest.raises(PromptRegistryImportError):
        _normalize_registry_key("module", 123)

    with pytest.raises(PromptRegistryImportError):
        _normalize_registry_key("module", ("demo", " "))

    assert _normalize_registry_key("module", "demo/example") == ("demo", "example")


def test_coerce_prompt_validates_type() -> None:
    with pytest.raises(PromptRegistryImportError):
        _coerce_prompt("module", object())


def test_find_section_helpers_return_none(
    prompt_workspace: tuple[Path, str],
) -> None:
    _workspace, module_name = prompt_workspace
    prompt = _load_prompt(module_name, ("demo", "example"))
    descriptor = PromptDescriptor.from_prompt(prompt)

    assert _find_section_descriptor(descriptor, ("missing",)) is None
    assert _find_section_node(prompt, ("missing",)) is None


def test_build_override_file_path_normalizes_relative_paths(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    overrides_dir = Path(".wink") / "overrides"
    config = MCPServerConfig(
        workspace_root=workspace,
        overrides_dir=overrides_dir,
        prompt_registry_modules=(),
    )

    path = _build_override_file_path(config, "demo", "prompt", "tag")
    assert path.parts[-3:] == ("demo", "prompt", "tag.json")


def test_split_namespace_validations() -> None:
    with pytest.raises(ValueError):
        _split_namespace(" ")

    with pytest.raises(ValueError):
        _split_namespace("///")


def test_validate_identifier_validations() -> None:
    with pytest.raises(ValueError):
        _validate_identifier(" ", "label")

    with pytest.raises(ValueError):
        _validate_identifier("Invalid", "label")
