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

"""Helpers for inspecting prompt overrides for the wink MCP server."""

from __future__ import annotations

import importlib
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from ..prompt import Prompt
from ..prompt.overrides import (
    PromptDescriptor,
    PromptLike,
    PromptOverride,
    PromptOverridesError,
    PromptOverridesStore,
    SectionDescriptor,
    SectionOverride,
)
from .config import MCPServerConfig

type PromptAny = Prompt[Any]

PROMPT_DESCRIPTOR_VERSION = 1

_IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"


class _SectionLike(Protocol):
    def original_body_template(self) -> str | None: ...

    accepts_overrides: bool


class _SectionNodeProtocol(Protocol):
    path: tuple[str, ...]
    section: _SectionLike


class WinkOverridesError(Exception):
    """Base class for wink override helper failures."""


class PromptRegistryImportError(WinkOverridesError):
    """Raised when prompt registry modules fail to import or validate."""

    def __init__(
        self, module: str, message: str, *, cause: Exception | None = None
    ) -> None:
        super().__init__(f"{module}: {message}")
        self.module = module
        self.message = message
        self.__cause__ = cause


class PromptNotFoundError(WinkOverridesError):
    """Raised when the requested prompt cannot be located in the registry."""

    def __init__(self, ns: str, prompt_key: str) -> None:
        super().__init__(f"Prompt not found: {ns}/{prompt_key}")
        self.ns = ns
        self.prompt_key = prompt_key


class SectionNotFoundError(WinkOverridesError):
    """Raised when the requested section path is not part of the prompt descriptor."""

    def __init__(self, ns: str, prompt_key: str, section_path: tuple[str, ...]) -> None:
        path_display = "/".join(section_path) or "<root>"
        super().__init__(f"Section not found for {ns}/{prompt_key}: {path_display}")
        self.ns = ns
        self.prompt_key = prompt_key
        self.section_path = section_path


class SectionOverridesUnavailableError(WinkOverridesError):
    """Raised when the target section does not support overrides."""

    def __init__(self, ns: str, prompt_key: str, section_path: tuple[str, ...]) -> None:
        path_display = "/".join(section_path) or "<root>"
        super().__init__(
            f"Section overrides unavailable for {ns}/{prompt_key}: {path_display}"
        )
        self.ns = ns
        self.prompt_key = prompt_key
        self.section_path = section_path


class SectionOverrideResolutionError(WinkOverridesError):
    """Raised when the overrides store cannot resolve the requested section."""

    def __init__(
        self,
        ns: str,
        prompt_key: str,
        tag: str,
        section_path: tuple[str, ...],
        *,
        cause: PromptOverridesError,
    ) -> None:
        path_display = "/".join(section_path) or "<root>"
        super().__init__(
            f"Failed to resolve overrides for {ns}/{prompt_key}:{tag} section {path_display}"
        )
        self.ns = ns
        self.prompt_key = prompt_key
        self.tag = tag
        self.section_path = section_path
        self.__cause__ = cause


@dataclass(frozen=True, slots=True)
class SectionOverrideSnapshot:
    """Structured representation of a section override lookup."""

    ns: str
    prompt: str
    tag: str
    section_path: tuple[str, ...]
    expected_hash: str
    override_body: str | None
    default_body: str | None
    backing_file_path: Path
    descriptor_version: int


@dataclass(frozen=True, slots=True)
class SectionOverrideMutationResult:
    """Result payload for section override mutations."""

    ns: str
    prompt: str
    tag: str
    section_path: tuple[str, ...]
    expected_hash: str
    override_body: str | None
    descriptor_version: int
    backing_file_path: Path
    updated_at: datetime | None
    warnings: tuple[str, ...] = ()


class SectionOverrideApplyError(WinkOverridesError):
    """Raised when applying a section override fails."""


class SectionOverrideRemoveError(WinkOverridesError):
    """Raised when removing a section override fails."""


def fetch_section_override(
    *,
    config: MCPServerConfig,
    store: PromptOverridesStore,
    ns: str,
    prompt: str,
    tag: str,
    section_path: str,
) -> SectionOverrideSnapshot:
    """Resolve a section override for the wink MCP helpers."""

    normalized_path = _parse_section_path(section_path)
    prompt_obj = _load_prompt(ns, prompt, config)
    section_node = _find_section_node(prompt_obj, normalized_path)
    if section_node is None:
        raise SectionNotFoundError(ns, prompt, normalized_path)
    if not getattr(section_node.section, "accepts_overrides", True):
        raise SectionOverridesUnavailableError(ns, prompt, normalized_path)

    default_body = section_node.section.original_body_template()
    if default_body is None:
        raise SectionOverridesUnavailableError(ns, prompt, normalized_path)

    descriptor = PromptDescriptor.from_prompt(cast(PromptLike, prompt_obj))
    descriptor_section = _find_section_descriptor(descriptor, normalized_path)
    if descriptor_section is None:
        raise SectionOverridesUnavailableError(ns, prompt, normalized_path)

    try:
        override_payload = store.resolve(descriptor, tag=tag)
    except PromptOverridesError as error:
        raise SectionOverrideResolutionError(
            ns,
            prompt,
            tag,
            normalized_path,
            cause=error,
        ) from error

    override_body = _extract_override_body(override_payload, normalized_path)
    backing_path = _build_override_file_path(config, ns, prompt, tag)

    return SectionOverrideSnapshot(
        ns=ns,
        prompt=prompt,
        tag=tag,
        section_path=normalized_path,
        expected_hash=descriptor_section.content_hash,
        override_body=override_body,
        default_body=default_body,
        backing_file_path=backing_path,
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
    )


def apply_section_override(
    *,
    config: MCPServerConfig,
    store: PromptOverridesStore,
    ns: str,
    prompt: str,
    tag: str,
    section_path: str,
    body: str,
    expected_hash: str | None,
    descriptor_version: int | None,
    confirm: bool,
) -> SectionOverrideMutationResult:
    """Persist a section override after validating guards."""

    normalized_path = _parse_section_path(section_path)
    section_display = "/".join(normalized_path) or "<root>"

    if not confirm:
        raise SectionOverrideApplyError(
            "Confirmation required to persist section overrides.",
        )

    prompt_obj = _load_prompt(ns, prompt, config)
    section_node = _find_section_node(prompt_obj, normalized_path)
    if section_node is None:
        raise SectionNotFoundError(ns, prompt, normalized_path)
    if not getattr(section_node.section, "accepts_overrides", True):
        raise SectionOverridesUnavailableError(ns, prompt, normalized_path)

    descriptor = PromptDescriptor.from_prompt(cast(PromptLike, prompt_obj))
    descriptor_section = _find_section_descriptor(descriptor, normalized_path)
    if descriptor_section is None:
        raise SectionOverridesUnavailableError(ns, prompt, normalized_path)

    if (
        descriptor_version is not None
        and descriptor_version != PROMPT_DESCRIPTOR_VERSION
    ):
        raise SectionOverrideApplyError(
            f"Descriptor version mismatch. Expected {PROMPT_DESCRIPTOR_VERSION}, received {descriptor_version}."
        )

    if expected_hash is None:
        raise SectionOverrideApplyError(
            "expected_hash must be provided to apply a section override.",
        )
    if expected_hash != descriptor_section.content_hash:
        raise SectionOverrideApplyError(
            f"Hash mismatch for section {section_display}. Expected {descriptor_section.content_hash}, received {expected_hash}."
        )

    try:
        existing_override = store.resolve(descriptor, tag=tag)
    except PromptOverridesError as error:
        raise SectionOverrideApplyError(
            "Failed to load existing overrides before applying section override."
        ) from error

    sections = dict(existing_override.sections) if existing_override else {}
    tools = dict(existing_override.tool_overrides) if existing_override else {}

    sections[normalized_path] = SectionOverride(
        expected_hash=expected_hash,
        body=body,
    )

    override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag=tag,
        sections=sections,
        tool_overrides=tools,
    )

    try:
        persisted = store.upsert(descriptor, override)
    except PromptOverridesError as error:
        raise SectionOverrideApplyError(
            "Failed to persist section override."
        ) from error

    persisted_section = persisted.sections.get(normalized_path)
    if persisted_section is None:
        raise SectionOverrideApplyError(
            "Persisted override missing target section after write.",
        )

    backing_path = _build_override_file_path(config, ns, prompt, tag)
    updated_at = _stat_timestamp(backing_path)
    if updated_at is None:
        updated_at = _now()

    return SectionOverrideMutationResult(
        ns=ns,
        prompt=prompt,
        tag=tag,
        section_path=normalized_path,
        expected_hash=descriptor_section.content_hash,
        override_body=persisted_section.body,
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
        backing_file_path=backing_path,
        updated_at=updated_at,
    )


def remove_section_override(
    *,
    config: MCPServerConfig,
    store: PromptOverridesStore,
    ns: str,
    prompt: str,
    tag: str,
    section_path: str,
    descriptor_version: int | None,
) -> SectionOverrideMutationResult:
    """Remove a single section override, deleting the file when empty."""

    normalized_path = _parse_section_path(section_path)
    section_display = "/".join(normalized_path) or "<root>"

    prompt_obj = _load_prompt(ns, prompt, config)
    section_node = _find_section_node(prompt_obj, normalized_path)
    if section_node is None:
        raise SectionNotFoundError(ns, prompt, normalized_path)
    if not getattr(section_node.section, "accepts_overrides", True):
        raise SectionOverridesUnavailableError(ns, prompt, normalized_path)

    descriptor = PromptDescriptor.from_prompt(cast(PromptLike, prompt_obj))
    descriptor_section = _find_section_descriptor(descriptor, normalized_path)
    if descriptor_section is None:
        raise SectionOverridesUnavailableError(ns, prompt, normalized_path)

    if (
        descriptor_version is not None
        and descriptor_version != PROMPT_DESCRIPTOR_VERSION
    ):
        raise SectionOverrideRemoveError(
            f"Descriptor version mismatch. Expected {PROMPT_DESCRIPTOR_VERSION}, received {descriptor_version}."
        )

    try:
        existing_override = store.resolve(descriptor, tag=tag)
    except PromptOverridesError as error:
        raise SectionOverrideRemoveError(
            "Failed to load existing overrides before removing section override."
        ) from error

    backing_path = _build_override_file_path(config, ns, prompt, tag)
    warnings: list[str] = []

    if existing_override is None or normalized_path not in existing_override.sections:
        warnings.append(
            f"No override found for section {section_display}. Nothing to remove."
        )
        return SectionOverrideMutationResult(
            ns=ns,
            prompt=prompt,
            tag=tag,
            section_path=normalized_path,
            expected_hash=descriptor_section.content_hash,
            override_body=None,
            descriptor_version=PROMPT_DESCRIPTOR_VERSION,
            backing_file_path=backing_path,
            updated_at=_stat_timestamp(backing_path),
            warnings=tuple(warnings),
        )

    sections = dict(existing_override.sections)
    _ = sections.pop(normalized_path, None)
    tools = dict(existing_override.tool_overrides)

    if sections or tools:
        override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=tag,
            sections=sections,
            tool_overrides=tools,
        )
        try:
            _ = store.upsert(descriptor, override)
        except PromptOverridesError as error:
            raise SectionOverrideRemoveError(
                "Failed to persist section removal."
            ) from error
        updated_at = _stat_timestamp(backing_path)
        if updated_at is None:
            updated_at = _now()
    else:
        updated_at = _stat_timestamp(backing_path)
        store.delete(ns=descriptor.ns, prompt_key=descriptor.key, tag=tag)
        if updated_at is None:
            updated_at = _now()

    return SectionOverrideMutationResult(
        ns=ns,
        prompt=prompt,
        tag=tag,
        section_path=normalized_path,
        expected_hash=descriptor_section.content_hash,
        override_body=None,
        descriptor_version=PROMPT_DESCRIPTOR_VERSION,
        backing_file_path=backing_path,
        updated_at=updated_at,
        warnings=tuple(warnings),
    )


def _extract_override_body(
    override: PromptOverride | None, section_path: tuple[str, ...]
) -> str | None:
    if override is None:
        return None
    section_override = override.sections.get(section_path)
    if section_override is None:
        return None
    return section_override.body


def _parse_section_path(path: str) -> tuple[str, ...]:
    parts = [segment.strip() for segment in path.split("/") if segment.strip()]
    if not parts:
        msg = "Section path must contain at least one segment."
        raise ValueError(msg)
    return tuple(parts)


def _load_prompt(ns: str, prompt_key: str, config: MCPServerConfig) -> PromptAny:
    for prompt in _iter_registry_prompts(config):
        if prompt.ns == ns and prompt.key == prompt_key:
            return prompt
    raise PromptNotFoundError(ns, prompt_key)


def _iter_registry_prompts(config: MCPServerConfig) -> Iterator[PromptAny]:
    modules = config.prompt_registry_modules
    if not modules:
        return iter(())
    return _iter_prompts_from_modules(config.workspace_root, modules)


def _iter_prompts_from_modules(
    workspace_root: Path, modules: Sequence[str]
) -> Iterator[PromptAny]:
    with _temporary_sys_path(workspace_root):
        for module_name in modules:
            if not module_name:
                continue
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as error:
                raise PromptRegistryImportError(
                    module_name,
                    "module could not be imported",
                    cause=error,
                ) from error
            except Exception as error:  # pragma: no cover - defensive guard
                raise PromptRegistryImportError(
                    module_name,
                    "module import failed",
                    cause=error,
                ) from error

            registry_obj = getattr(module, "PROMPTS", None)
            if registry_obj is None:
                raise PromptRegistryImportError(
                    module_name,
                    "module must define a PROMPTS mapping",
                )
            if not isinstance(registry_obj, Mapping):
                raise PromptRegistryImportError(
                    module_name,
                    "PROMPTS must be a mapping of identifiers to Prompt instances",
                )

            registry = cast(Mapping[object, object], registry_obj)
            yield from _iter_prompts_from_registry(module_name, registry)


def _iter_prompts_from_registry(
    module_name: str, registry: Mapping[object, object]
) -> Iterator[PromptAny]:
    for key, value in registry.items():
        ns, prompt_key = _normalize_registry_key(module_name, key)
        prompt = _coerce_prompt(module_name, value)
        if prompt.ns != ns or prompt.key != prompt_key:
            raise PromptRegistryImportError(
                module_name,
                (
                    "registry entry key does not match prompt metadata "
                    f"({ns}/{prompt_key} vs {prompt.ns}/{prompt.key})"
                ),
            )
        yield prompt


def _normalize_registry_key(module_name: str, key: object) -> tuple[str, str]:
    if isinstance(key, tuple):
        tuple_key = cast(tuple[Any, ...], key)
        if len(tuple_key) != 2:
            raise PromptRegistryImportError(
                module_name,
                "prompt registry tuple keys must be (namespace, prompt_key)",
            )
        ns_obj: object = tuple_key[0]
        prompt_obj: object = tuple_key[1]
        if isinstance(ns_obj, str) and isinstance(prompt_obj, str):
            ns = ns_obj
            prompt_key = prompt_obj
        else:
            raise PromptRegistryImportError(
                module_name,
                "registry tuple keys must contain strings",
            )
    elif isinstance(key, str):
        segments = [segment.strip() for segment in key.split("/") if segment.strip()]
        if len(segments) != 2:
            raise PromptRegistryImportError(
                module_name,
                "string keys must be formatted as 'namespace/prompt'",
            )
        ns, prompt_key = segments[0], segments[1]
    else:
        raise PromptRegistryImportError(
            module_name,
            "registry keys must be tuples or strings",
        )
    ns_normalized = ns.strip()
    prompt_normalized = prompt_key.strip()
    if not ns_normalized or not prompt_normalized:
        raise PromptRegistryImportError(
            module_name,
            "registry keys must not be empty",
        )
    return ns_normalized, prompt_normalized


def _coerce_prompt(module_name: str, candidate: object) -> PromptAny:
    if isinstance(candidate, Prompt):
        return cast(PromptAny, candidate)  # ty: ignore[redundant-cast]
    raise PromptRegistryImportError(
        module_name,
        "registry values must be Prompt instances",
    )


def _find_section_descriptor(
    descriptor: PromptDescriptor, section_path: tuple[str, ...]
) -> SectionDescriptor | None:
    for section in descriptor.sections:
        if section.path == section_path:
            return section
    return None


def _find_section_node(
    prompt: PromptAny, section_path: tuple[str, ...]
) -> _SectionNodeProtocol | None:
    for node in prompt.sections:
        if node.path == section_path:
            return cast(_SectionNodeProtocol, node)
    return None


def _build_override_file_path(
    config: MCPServerConfig, ns: str, prompt_key: str, tag: str
) -> Path:
    base = config.overrides_dir
    if not base.is_absolute():
        base = (config.workspace_root / base).resolve()
    segments = _split_namespace(ns)
    prompt_component = _validate_identifier(prompt_key, "prompt key")
    tag_component = _validate_identifier(tag, "tag")
    return base.joinpath(*segments, prompt_component, f"{tag_component}.json")


def _split_namespace(ns: str) -> tuple[str, ...]:
    stripped = ns.strip()
    if not stripped:
        raise ValueError("Namespace must be a non-empty string.")
    parts = [segment.strip() for segment in stripped.split("/") if segment.strip()]
    if not parts:
        raise ValueError("Namespace must contain at least one segment.")
    return tuple(_validate_identifier(part, "namespace segment") for part in parts)


def _validate_identifier(value: str, label: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{label.capitalize()} must be a non-empty string.")
    if not re.fullmatch(_IDENTIFIER_PATTERN, stripped):
        raise ValueError(
            f"{label.capitalize()} must match pattern {_IDENTIFIER_PATTERN}."
        )
    return stripped


@contextmanager
def _temporary_sys_path(workspace_root: Path) -> Iterator[None]:
    workspace = str(workspace_root)
    added = False
    if workspace and workspace not in sys.path:
        sys.path.insert(0, workspace)
        added = True
    try:
        yield
    finally:
        if added:
            with suppress(ValueError):  # pragma: no branch - defensive cleanup
                sys.path.remove(workspace)


def _stat_timestamp(path: Path) -> datetime | None:
    try:
        stat_result = path.stat()
    except FileNotFoundError:
        return None
    timestamp = datetime.fromtimestamp(stat_result.st_mtime, tz=UTC)
    return _truncate_to_milliseconds(timestamp)


def _now() -> datetime:
    return _truncate_to_milliseconds(datetime.now(UTC))


def _truncate_to_milliseconds(value: datetime) -> datetime:
    microsecond = value.microsecond - (value.microsecond % 1000)
    return value.replace(microsecond=microsecond, tzinfo=UTC)


__all__ = [
    "PROMPT_DESCRIPTOR_VERSION",
    "PromptNotFoundError",
    "PromptRegistryImportError",
    "SectionNotFoundError",
    "SectionOverrideApplyError",
    "SectionOverrideMutationResult",
    "SectionOverrideRemoveError",
    "SectionOverrideResolutionError",
    "SectionOverrideSnapshot",
    "SectionOverridesUnavailableError",
    "WinkOverridesError",
    "apply_section_override",
    "fetch_section_override",
    "remove_section_override",
]
