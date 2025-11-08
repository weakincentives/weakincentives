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

import importlib
import sys
from collections.abc import Iterator, Mapping
from contextlib import suppress
from pathlib import Path
from textwrap import dedent
from typing import Protocol, cast

import pytest

from weakincentives.cli.config import MCPServerConfig
from weakincentives.cli.wink_overrides import (
    PROMPT_DESCRIPTOR_VERSION,
    PromptNotFoundError,
    PromptRegistryImportError,
    SectionNotFoundError,
    SectionOverrideResolutionError,
    SectionOverrideSnapshot,
    SectionOverridesUnavailableError,
    _build_override_file_path,
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
    fetch_section_override,
)
from weakincentives.prompt import Prompt
from weakincentives.prompt.overrides import (
    LocalPromptOverridesStore,
    PromptDescriptor,
    PromptOverride,
    PromptOverridesError,
    PromptOverridesStore,
    SectionOverride,
)


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
        from weakincentives.prompt import MarkdownSection, Prompt

        PROMPTS = {
            ("demo", "example"): Prompt(
                ns="demo",
                key="example",
                sections=(
                    MarkdownSection(
                        title="Example",
                        key="intro",
                        template="Example body",
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
