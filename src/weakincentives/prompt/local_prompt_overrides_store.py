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
import re
import subprocess  # nosec B404 - git invocation for root discovery
import tempfile
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, cast, overload, override

from ..logging import StructuredLogger, get_logger
from ._types import SupportsDataclass
from .tool import Tool
from .versioning import (
    PromptDescriptor,
    PromptLike,
    PromptOverride,
    PromptOverridesError,
    PromptOverridesStore,
    SectionDescriptor,
    SectionOverride,
    ToolDescriptor,
    ToolOverride,
)

_LOGGER: StructuredLogger = get_logger(
    __name__, context={"component": "prompt_overrides"}
)
_FORMAT_VERSION = 1
_DEFAULT_RELATIVE_PATH = Path(".weakincentives") / "prompts" / "overrides"
_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class LocalPromptOverridesStore(PromptOverridesStore):
    """Persist prompt overrides to disk within the project workspace."""

    def __init__(
        self,
        *,
        root_path: str | Path | None = None,
        overrides_relative_path: str | Path = _DEFAULT_RELATIVE_PATH,
    ) -> None:
        super().__init__()
        self._explicit_root = Path(root_path).resolve() if root_path else None
        self._root: Path | None = None
        self._overrides_relative_path = Path(overrides_relative_path)

    @override
    def resolve(
        self,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None:
        normalized_tag = self._validate_identifier(tag, "tag")
        file_path = self._override_file_path(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=normalized_tag,
        )
        if not file_path.exists():
            _LOGGER.debug(
                "Override file not found.",
                event="prompt_override_missing",
                context={
                    "ns": descriptor.ns,
                    "prompt_key": descriptor.key,
                    "tag": normalized_tag,
                },
            )
            return None

        try:
            with file_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as error:
            raise PromptOverridesError(
                f"Failed to parse prompt override JSON: {file_path}"
            ) from error

        self._validate_header(payload, descriptor, normalized_tag, file_path)

        sections = self._load_sections(payload.get("sections", {}), descriptor)
        tools = self._load_tools(payload.get("tools", {}), descriptor)

        if not sections and not tools:
            _LOGGER.debug(
                "No applicable overrides remain after validation.",
                event="prompt_override_empty",
                context={
                    "ns": descriptor.ns,
                    "prompt_key": descriptor.key,
                    "tag": normalized_tag,
                },
            )
            return None

        override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=tag,
            sections=sections,
            tool_overrides=tools,
        )
        _LOGGER.info(
            "Resolved prompt override.",
            event="prompt_override_resolved",
            context={
                "ns": descriptor.ns,
                "prompt_key": descriptor.key,
                "tag": normalized_tag,
                "section_count": len(sections),
                "tool_count": len(tools),
            },
        )
        return override

    @override
    def upsert(
        self,
        descriptor: PromptDescriptor,
        override: PromptOverride,
    ) -> PromptOverride:
        if override.ns != descriptor.ns or override.prompt_key != descriptor.key:
            raise PromptOverridesError(
                "Override metadata does not match descriptor.",
            )
        normalized_tag = self._validate_identifier(override.tag, "tag")

        file_path = self._override_file_path(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=normalized_tag,
        )

        validated_sections = self._validate_sections_for_write(
            override.sections,
            descriptor,
        )
        validated_tools = self._validate_tools_for_write(
            override.tool_overrides,
            descriptor,
        )

        payload = {
            "version": _FORMAT_VERSION,
            "ns": descriptor.ns,
            "prompt_key": descriptor.key,
            "tag": normalized_tag,
            "sections": self._serialize_sections(validated_sections),
            "tools": self._serialize_tools(validated_tools),
        }

        self._atomic_write(file_path, payload)

        persisted = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=normalized_tag,
            sections=validated_sections,
            tool_overrides=validated_tools,
        )
        _LOGGER.info(
            "Persisted prompt override.",
            event="prompt_override_persisted",
            context={
                "ns": descriptor.ns,
                "prompt_key": descriptor.key,
                "tag": normalized_tag,
                "section_count": len(validated_sections),
                "tool_count": len(validated_tools),
            },
        )
        return persisted

    @override
    def delete(
        self,
        *,
        ns: str,
        prompt_key: str,
        tag: str,
    ) -> None:
        normalized_tag = self._validate_identifier(tag, "tag")
        file_path = self._override_file_path(
            ns=ns,
            prompt_key=prompt_key,
            tag=normalized_tag,
        )
        try:
            file_path.unlink()
        except FileNotFoundError:
            _LOGGER.debug(
                "No override file to delete.",
                event="prompt_override_delete_missing",
                context={
                    "ns": ns,
                    "prompt_key": prompt_key,
                    "tag": normalized_tag,
                },
            )

    @override
    def seed_if_necessary(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
    ) -> PromptOverride:
        descriptor = PromptDescriptor.from_prompt(prompt)
        normalized_tag = self._validate_identifier(tag, "tag")
        file_path = self._override_file_path(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=normalized_tag,
        )

        if file_path.exists():
            existing = self.resolve(descriptor=descriptor, tag=normalized_tag)
            if existing is None:
                raise PromptOverridesError(
                    "Override file exists but could not be resolved."
                )
            return existing

        sections = self._seed_sections(prompt, descriptor)
        tools = self._seed_tools(prompt, descriptor)

        seed_override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=normalized_tag,
            sections=sections,
            tool_overrides=tools,
        )
        return self.upsert(descriptor, seed_override)

    def _resolve_root(self) -> Path:
        if self._root is not None:
            return self._root
        if self._explicit_root is not None:
            self._root = self._explicit_root
            return self._root

        git_root = self._git_toplevel()
        if git_root is not None:
            self._root = git_root
            return self._root

        traversal_root = self._walk_to_git_root()
        if traversal_root is None:
            raise PromptOverridesError(
                "Failed to locate repository root. Provide root_path explicitly."
            )
        self._root = traversal_root
        return self._root

    def _git_toplevel(self) -> Path | None:
        try:
            result = subprocess.run(  # nosec B603 B607 - git is invoked with explicit args
                ["git", "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        path = result.stdout.strip()
        if not path:
            return None
        return Path(path).resolve()

    def _walk_to_git_root(self) -> Path | None:
        current = Path.cwd().resolve()
        for candidate in (current, *current.parents):
            git_dir = candidate / ".git"
            if git_dir.exists():
                return candidate
        return None

    def _overrides_dir(self) -> Path:
        root = self._resolve_root()
        return root / self._overrides_relative_path

    def _override_file_path(
        self,
        *,
        ns: str,
        prompt_key: str,
        tag: str,
    ) -> Path:
        segments = self._split_namespace(ns)
        prompt_component = self._validate_identifier(prompt_key, "prompt key")
        tag_component = self._validate_identifier(tag, "tag")
        directory = self._overrides_dir().joinpath(*segments, prompt_component)
        return directory / f"{tag_component}.json"

    def _split_namespace(self, ns: str) -> tuple[str, ...]:
        stripped = ns.strip()
        if not stripped:
            raise PromptOverridesError("Namespace must be a non-empty string.")
        segments = tuple(part.strip() for part in stripped.split("/") if part.strip())
        if not segments:
            raise PromptOverridesError("Namespace must contain at least one segment.")
        return tuple(
            self._validate_identifier(segment, "namespace segment")
            for segment in segments
        )

    def _validate_identifier(self, value: str, label: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise PromptOverridesError(
                f"{label.capitalize()} must be a non-empty string."
            )
        if not _IDENTIFIER_PATTERN.fullmatch(stripped):
            raise PromptOverridesError(
                f"{label.capitalize()} must match pattern ^[a-z0-9][a-z0-9._-]{{0,63}}$."
            )
        return stripped

    def _validate_header(
        self,
        payload: Mapping[str, Any],
        descriptor: PromptDescriptor,
        tag: str,
        file_path: Path,
    ) -> None:
        version = payload.get("version")
        if version != _FORMAT_VERSION:
            raise PromptOverridesError(
                f"Unsupported override file version {version!r} in {file_path}."
            )
        ns = payload.get("ns")
        prompt_key = payload.get("prompt_key")
        tag_value = payload.get("tag")
        if ns != descriptor.ns or prompt_key != descriptor.key or tag_value != tag:
            raise PromptOverridesError(
                "Override file metadata does not match descriptor inputs."
            )

    def _section_descriptor_index(
        self, descriptor: PromptDescriptor
    ) -> dict[tuple[str, ...], SectionDescriptor]:
        return {section.path: section for section in descriptor.sections}

    def _tool_descriptor_index(
        self, descriptor: PromptDescriptor
    ) -> dict[str, ToolDescriptor]:
        return {tool.name: tool for tool in descriptor.tools}

    @overload
    def _normalize_section_override(
        self,
        *,
        path: tuple[str, ...],
        descriptor_section: SectionDescriptor | None,
        expected_hash: object,
        body: object,
        strict: Literal[True],
        path_display: str,
        body_error_message: str,
    ) -> SectionOverride: ...

    @overload
    def _normalize_section_override(
        self,
        *,
        path: tuple[str, ...],
        descriptor_section: SectionDescriptor | None,
        expected_hash: object,
        body: object,
        strict: Literal[False],
        path_display: str,
        body_error_message: str,
    ) -> SectionOverride | None: ...

    def _normalize_section_override(
        self,
        *,
        path: tuple[str, ...],
        descriptor_section: SectionDescriptor | None,
        expected_hash: object,
        body: object,
        strict: bool,
        path_display: str,
        body_error_message: str,
    ) -> SectionOverride | None:
        if descriptor_section is None:
            if strict:
                raise PromptOverridesError(
                    f"Unknown section path for override: {path_display}"
                )
            _LOGGER.debug(
                "Skipping unknown override section path.",
                event="prompt_override_unknown_section",
                context={"path": path_display},
            )
            return None
        if not isinstance(expected_hash, str):
            raise PromptOverridesError("Section expected_hash must be a string.")
        if expected_hash != descriptor_section.content_hash:
            if strict:
                raise PromptOverridesError(f"Hash mismatch for section {path_display}.")
            _LOGGER.debug(
                "Skipping stale section override.",
                event="prompt_override_stale_section",
                context={
                    "path": path_display,
                    "expected_hash": descriptor_section.content_hash,
                    "found_hash": expected_hash,
                },
            )
            return None
        if not isinstance(body, str):
            raise PromptOverridesError(body_error_message)
        return SectionOverride(
            expected_hash=expected_hash,
            body=body,
        )

    @overload
    def _normalize_tool_override(
        self,
        *,
        name: str,
        descriptor_tool: ToolDescriptor | None,
        expected_hash: object,
        description: object,
        param_descriptions: object,
        strict: Literal[True],
        description_error_message: str,
        param_mapping_error_message: str,
        param_entry_error_message: str,
    ) -> ToolOverride: ...

    @overload
    def _normalize_tool_override(
        self,
        *,
        name: str,
        descriptor_tool: ToolDescriptor | None,
        expected_hash: object,
        description: object,
        param_descriptions: object,
        strict: Literal[False],
        description_error_message: str,
        param_mapping_error_message: str,
        param_entry_error_message: str,
    ) -> ToolOverride | None: ...

    def _normalize_tool_override(
        self,
        *,
        name: str,
        descriptor_tool: ToolDescriptor | None,
        expected_hash: object,
        description: object,
        param_descriptions: object,
        strict: bool,
        description_error_message: str,
        param_mapping_error_message: str,
        param_entry_error_message: str,
    ) -> ToolOverride | None:
        if descriptor_tool is None:
            if strict:
                raise PromptOverridesError(f"Unknown tool override: {name}")
            _LOGGER.debug(
                "Skipping unknown tool override.",
                event="prompt_override_unknown_tool",
                context={"tool": name},
            )
            return None
        if not isinstance(expected_hash, str):
            raise PromptOverridesError("Tool expected_contract_hash must be a string.")
        if expected_hash != descriptor_tool.contract_hash:
            if strict:
                raise PromptOverridesError(f"Hash mismatch for tool override: {name}.")
            _LOGGER.debug(
                "Skipping stale tool override.",
                event="prompt_override_stale_tool",
                context={
                    "tool": name,
                    "expected_hash": descriptor_tool.contract_hash,
                    "found_hash": expected_hash,
                },
            )
            return None
        if description is not None and not isinstance(description, str):
            raise PromptOverridesError(description_error_message)
        if param_descriptions is None:
            param_descriptions = {}
        if not isinstance(param_descriptions, Mapping):
            raise PromptOverridesError(param_mapping_error_message)
        normalized_params: dict[str, str] = {}
        for key, value in cast(Mapping[object, object], param_descriptions).items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise PromptOverridesError(param_entry_error_message)
            normalized_params[key] = value
        if description is None:
            normalized_description: str | None = None
        else:
            normalized_description = description
        return ToolOverride(
            name=name,
            expected_contract_hash=expected_hash,
            description=normalized_description,
            param_descriptions=normalized_params,
        )

    def _load_sections(
        self,
        payload: object | None,
        descriptor: PromptDescriptor,
    ) -> dict[tuple[str, ...], SectionOverride]:
        if payload is None:
            return {}
        if not isinstance(payload, Mapping):
            raise PromptOverridesError("Sections payload must be a mapping.")
        if not payload:
            return {}
        mapping_payload = cast(Mapping[object, object], payload)
        descriptor_index = self._section_descriptor_index(descriptor)
        overrides: dict[tuple[str, ...], SectionOverride] = {}
        for path_key_obj, section_payload_raw in mapping_payload.items():
            if not isinstance(path_key_obj, str):
                raise PromptOverridesError("Section keys must be strings.")
            path_key = path_key_obj
            path = tuple(part for part in path_key.split("/") if part)
            if not isinstance(section_payload_raw, Mapping):
                raise PromptOverridesError("Section payload must be an object.")
            section_payload = cast(Mapping[str, object], section_payload_raw)
            expected_hash = section_payload.get("expected_hash")
            if not isinstance(expected_hash, str):
                raise PromptOverridesError("Section expected_hash must be a string.")
            body = section_payload.get("body")
            section_override = self._normalize_section_override(
                path=path,
                descriptor_section=descriptor_index.get(path),
                expected_hash=expected_hash,
                body=body,
                strict=False,
                path_display=path_key,
                body_error_message="Section body must be a string.",
            )
            if section_override is not None:
                overrides[path] = section_override
        return overrides

    def _load_tools(
        self,
        payload: object | None,
        descriptor: PromptDescriptor,
    ) -> dict[str, ToolOverride]:
        if payload is None:
            return {}
        if not isinstance(payload, Mapping):
            raise PromptOverridesError("Tools payload must be a mapping.")
        if not payload:
            return {}
        mapping_payload = cast(Mapping[object, object], payload)
        descriptor_index = self._tool_descriptor_index(descriptor)
        overrides: dict[str, ToolOverride] = {}
        for tool_name_obj, tool_payload_raw in mapping_payload.items():
            if not isinstance(tool_name_obj, str):
                raise PromptOverridesError("Tool names must be strings.")
            tool_name = tool_name_obj
            if not isinstance(tool_payload_raw, Mapping):
                raise PromptOverridesError("Tool payload must be an object.")
            tool_payload = cast(Mapping[str, object], tool_payload_raw)
            expected_hash = tool_payload.get("expected_contract_hash")
            if not isinstance(expected_hash, str):
                raise PromptOverridesError(
                    "Tool expected_contract_hash must be a string."
                )
            description = tool_payload.get("description")
            param_payload = tool_payload.get("param_descriptions")
            tool_override = self._normalize_tool_override(
                name=tool_name,
                descriptor_tool=descriptor_index.get(tool_name),
                expected_hash=expected_hash,
                description=description,
                param_descriptions=param_payload,
                strict=False,
                description_error_message="Tool description must be a string when set.",
                param_mapping_error_message="Tool param_descriptions must be a mapping when provided.",
                param_entry_error_message="Tool param description entries must be strings.",
            )
            if tool_override is not None:
                overrides[tool_name] = tool_override
        return overrides

    def _validate_sections_for_write(
        self,
        sections: Mapping[tuple[str, ...], SectionOverride],
        descriptor: PromptDescriptor,
    ) -> dict[tuple[str, ...], SectionOverride]:
        descriptor_index = self._section_descriptor_index(descriptor)
        validated: dict[tuple[str, ...], SectionOverride] = {}
        for path, section_override in sections.items():
            path_display = "/".join(path)
            normalized_section = self._normalize_section_override(
                path=path,
                descriptor_section=descriptor_index.get(path),
                expected_hash=cast(Any, section_override).expected_hash,
                body=cast(Any, section_override).body,
                strict=True,
                path_display=path_display,
                body_error_message=f"Section override body must be a string for {path_display}.",
            )
            validated[path] = normalized_section
        return validated

    def _validate_tools_for_write(
        self,
        tools: Mapping[str, ToolOverride],
        descriptor: PromptDescriptor,
    ) -> dict[str, ToolOverride]:
        if not tools:
            return {}
        descriptor_index = self._tool_descriptor_index(descriptor)
        validated: dict[str, ToolOverride] = {}
        for name, tool_override in tools.items():
            normalized_tool = self._normalize_tool_override(
                name=name,
                descriptor_tool=descriptor_index.get(name),
                expected_hash=cast(Any, tool_override).expected_contract_hash,
                description=cast(Any, tool_override).description,
                param_descriptions=cast(Any, tool_override).param_descriptions,
                strict=True,
                description_error_message=f"Tool description override must be a string for {name}.",
                param_mapping_error_message=f"Tool parameter descriptions must be a mapping for {name}.",
                param_entry_error_message=f"Tool parameter descriptions must map strings to strings for {name}.",
            )
            validated[name] = normalized_tool
        return validated

    def _serialize_sections(
        self,
        sections: Mapping[tuple[str, ...], SectionOverride],
    ) -> dict[str, dict[str, str]]:
        serialized: dict[str, dict[str, str]] = {}
        for path, section_override in sections.items():
            key = "/".join(path)
            serialized[key] = {
                "expected_hash": section_override.expected_hash,
                "body": section_override.body,
            }
        return serialized

    def _serialize_tools(
        self,
        tools: Mapping[str, ToolOverride],
    ) -> dict[str, dict[str, Any]]:
        serialized: dict[str, dict[str, Any]] = {}
        for name, tool_override in tools.items():
            serialized[name] = {
                "expected_contract_hash": tool_override.expected_contract_hash,
                "description": tool_override.description,
                "param_descriptions": dict(tool_override.param_descriptions),
            }
        return serialized

    def _atomic_write(self, file_path: Path, payload: Mapping[str, Any]) -> None:
        directory = file_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=directory, delete=False, encoding="utf-8"
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            _ = handle.write("\n")
            temp_name = Path(handle.name)
        _ = Path(temp_name).replace(file_path)

    def _seed_sections(
        self,
        prompt: PromptLike,
        descriptor: PromptDescriptor,
    ) -> dict[tuple[str, ...], SectionOverride]:
        section_lookup = {node.path: node.section for node in prompt.sections}
        seeded: dict[tuple[str, ...], SectionOverride] = {}
        for section in descriptor.sections:
            section_obj = section_lookup.get(section.path)
            if section_obj is None:
                raise PromptOverridesError(
                    f"Prompt missing section for descriptor path {'/'.join(section.path)}."
                )
            template = section_obj.original_body_template()
            if template is None:
                raise PromptOverridesError(
                    "Cannot seed override for section without template."
                )
            seeded[section.path] = SectionOverride(
                expected_hash=section.content_hash,
                body=template,
            )
        return seeded

    def _seed_tools(
        self,
        prompt: PromptLike,
        descriptor: PromptDescriptor,
    ) -> dict[str, ToolOverride]:
        if not descriptor.tools:
            return {}
        tool_lookup: dict[str, Tool[SupportsDataclass, SupportsDataclass]] = {}
        for node in prompt.sections:
            for tool in node.section.tools():
                tool_lookup[tool.name] = tool
        seeded: dict[str, ToolOverride] = {}
        for tool in descriptor.tools:
            tool_obj = tool_lookup.get(tool.name)
            if tool_obj is None:
                raise PromptOverridesError(
                    f"Prompt missing tool for descriptor entry {tool.name}."
                )
            param_descriptions = self._collect_param_descriptions(tool_obj)
            seeded[tool.name] = ToolOverride(
                name=tool.name,
                expected_contract_hash=tool.contract_hash,
                description=tool_obj.description,
                param_descriptions=param_descriptions,
            )
        return seeded

    def _collect_param_descriptions(
        self,
        tool: Tool[SupportsDataclass, SupportsDataclass],
    ) -> dict[str, str]:
        params_type = getattr(tool, "params_type", None)
        if not isinstance(params_type, type) or not is_dataclass(params_type):
            return {}
        descriptions: dict[str, str] = {}
        for field in fields(params_type):
            description = field.metadata.get("description") if field.metadata else None
            if isinstance(description, str) and description:
                descriptions[field.name] = description
        return descriptions


__all__ = ["LocalPromptOverridesStore"]
