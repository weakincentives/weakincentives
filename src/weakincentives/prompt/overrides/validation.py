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

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, cast, overload

from ...runtime.logging import StructuredLogger, get_logger
from .._types import SupportsDataclass
from ..tool import Tool
from .versioning import (
    HexDigest,
    PromptDescriptor,
    PromptLike,
    PromptOverride,
    PromptOverridesError,
    SectionDescriptor,
    SectionOverride,
    ToolDescriptor,
    ToolOverride,
    ensure_hex_digest,
)

_LOGGER: StructuredLogger = get_logger(
    __name__, context={"component": "prompt_overrides"}
)
FORMAT_VERSION = 1


def validate_header(
    payload: Mapping[str, Any],
    descriptor: PromptDescriptor,
    tag: str,
    file_path: Path,
) -> None:
    version = payload.get("version")
    if version != FORMAT_VERSION:
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
    descriptor: PromptDescriptor,
) -> dict[tuple[str, ...], SectionDescriptor]:
    return {section.path: section for section in descriptor.sections}


def _tool_descriptor_index(
    descriptor: PromptDescriptor,
) -> dict[str, ToolDescriptor]:
    return {tool.name: tool for tool in descriptor.tools}


def _format_section_path(path: tuple[str, ...]) -> str:
    return "/".join(path)


def _log_mismatched_override_metadata(
    descriptor: PromptDescriptor,
    override: PromptOverride,
) -> None:
    _LOGGER.debug(
        "Skipping override due to descriptor metadata mismatch.",
        event="prompt_override_mismatched_descriptor",
        context={
            "expected_ns": descriptor.ns,
            "expected_key": descriptor.key,
            "override_ns": override.ns,
            "override_key": override.prompt_key,
        },
    )


@overload
def _normalize_section_override(
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
    expected_digest = ensure_hex_digest(
        cast(HexDigest | str, expected_hash),
        field_name="Section expected_hash",
    )
    if expected_digest != descriptor_section.content_hash:
        if strict:
            raise PromptOverridesError(f"Hash mismatch for section {path_display}.")
        _LOGGER.debug(
            "Skipping stale section override.",
            event="prompt_override_stale_section",
            context={
                "path": path_display,
                "expected_hash": str(descriptor_section.content_hash),
                "found_hash": str(expected_digest),
            },
        )
        return None
    if not isinstance(body, str):
        raise PromptOverridesError(body_error_message)
    return SectionOverride(
        expected_hash=expected_digest,
        body=body,
    )


@overload
def _normalize_tool_override(
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
    expected_digest = ensure_hex_digest(
        cast(HexDigest | str, expected_hash),
        field_name="Tool expected_contract_hash",
    )
    if expected_digest != descriptor_tool.contract_hash:
        if strict:
            raise PromptOverridesError(f"Hash mismatch for tool override: {name}.")
        _LOGGER.debug(
            "Skipping stale tool override.",
            event="prompt_override_stale_tool",
            context={
                "tool": name,
                "expected_hash": str(descriptor_tool.contract_hash),
                "found_hash": str(expected_digest),
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
        expected_contract_hash=expected_digest,
        description=normalized_description,
        param_descriptions=normalized_params,
    )


def load_sections(
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
    descriptor_index = _section_descriptor_index(descriptor)
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
        body = section_payload.get("body")
        section_override = _normalize_section_override(
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


def filter_override_for_descriptor(
    descriptor: PromptDescriptor,
    override: PromptOverride,
) -> tuple[dict[tuple[str, ...], SectionOverride], dict[str, ToolOverride]]:
    if override.ns != descriptor.ns or override.prompt_key != descriptor.key:
        _log_mismatched_override_metadata(descriptor, override)
        return {}, {}

    descriptor_sections = _section_descriptor_index(descriptor)
    descriptor_tools = _tool_descriptor_index(descriptor)

    filtered_sections: dict[tuple[str, ...], SectionOverride] = {}
    for path, section_override in override.sections.items():
        normalized_section = _normalize_section_override(
            path=path,
            descriptor_section=descriptor_sections.get(path),
            expected_hash=section_override.expected_hash,
            body=section_override.body,
            strict=False,
            path_display=_format_section_path(path),
            body_error_message="Section override body must be a string.",
        )
        if normalized_section is not None:
            filtered_sections[path] = normalized_section

    filtered_tools: dict[str, ToolOverride] = {}
    for name, tool_override in override.tool_overrides.items():
        normalized_tool = _normalize_tool_override(
            name=name,
            descriptor_tool=descriptor_tools.get(name),
            expected_hash=tool_override.expected_contract_hash,
            description=tool_override.description,
            param_descriptions=tool_override.param_descriptions,
            strict=False,
            description_error_message="Tool description override must be a string when set.",
            param_mapping_error_message="Tool param_descriptions must be a mapping when provided.",
            param_entry_error_message="Tool param description entries must be strings.",
        )
        if normalized_tool is not None:
            filtered_tools[name] = normalized_tool

    return filtered_sections, filtered_tools


def load_tools(
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
    descriptor_index = _tool_descriptor_index(descriptor)
    overrides: dict[str, ToolOverride] = {}
    for tool_name_obj, tool_payload_raw in mapping_payload.items():
        if not isinstance(tool_name_obj, str):
            raise PromptOverridesError("Tool names must be strings.")
        tool_name = tool_name_obj
        if not isinstance(tool_payload_raw, Mapping):
            raise PromptOverridesError("Tool payload must be an object.")
        tool_payload = cast(Mapping[str, object], tool_payload_raw)
        expected_hash = tool_payload.get("expected_contract_hash")
        description = tool_payload.get("description")
        param_payload = tool_payload.get("param_descriptions")
        tool_override = _normalize_tool_override(
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


def validate_sections_for_write(
    sections: Mapping[tuple[str, ...], SectionOverride],
    descriptor: PromptDescriptor,
) -> dict[tuple[str, ...], SectionOverride]:
    descriptor_index = _section_descriptor_index(descriptor)
    validated: dict[tuple[str, ...], SectionOverride] = {}
    for path, section_override in sections.items():
        path_display = "/".join(path)
        normalized_section = _normalize_section_override(
            path=path,
            descriptor_section=descriptor_index.get(path),
            expected_hash=cast(Any, section_override).expected_hash,
            body=cast(Any, section_override).body,
            strict=True,
            path_display=path_display,
            body_error_message=(
                f"Section override body must be a string for {path_display}."
            ),
        )
        validated[path] = normalized_section
    return validated


def validate_tools_for_write(
    tools: Mapping[str, ToolOverride],
    descriptor: PromptDescriptor,
) -> dict[str, ToolOverride]:
    if not tools:
        return {}
    descriptor_index = _tool_descriptor_index(descriptor)
    validated: dict[str, ToolOverride] = {}
    for name, tool_override in tools.items():
        normalized_tool = _normalize_tool_override(
            name=name,
            descriptor_tool=descriptor_index.get(name),
            expected_hash=cast(Any, tool_override).expected_contract_hash,
            description=cast(Any, tool_override).description,
            param_descriptions=cast(Any, tool_override).param_descriptions,
            strict=True,
            description_error_message=(
                f"Tool description override must be a string for {name}."
            ),
            param_mapping_error_message=(
                f"Tool parameter descriptions must be a mapping for {name}."
            ),
            param_entry_error_message=(
                f"Tool parameter descriptions must map strings to strings for {name}."
            ),
        )
        validated[name] = normalized_tool
    return validated


def serialize_sections(
    sections: Mapping[tuple[str, ...], SectionOverride],
) -> dict[str, dict[str, str]]:
    serialized: dict[str, dict[str, str]] = {}
    for path, section_override in sections.items():
        key = "/".join(path)
        serialized[key] = {
            "expected_hash": str(section_override.expected_hash),
            "body": section_override.body,
        }
    return serialized


def serialize_tools(
    tools: Mapping[str, ToolOverride],
) -> dict[str, dict[str, Any]]:
    serialized: dict[str, dict[str, Any]] = {}
    for name, tool_override in tools.items():
        serialized[name] = {
            "expected_contract_hash": str(tool_override.expected_contract_hash),
            "description": tool_override.description,
            "param_descriptions": dict(tool_override.param_descriptions),
        }
    return serialized


def seed_sections(
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


def seed_tools(
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
        param_descriptions = _collect_param_descriptions(tool_obj)
        seeded[tool.name] = ToolOverride(
            name=tool.name,
            expected_contract_hash=tool.contract_hash,
            description=tool_obj.description,
            param_descriptions=param_descriptions,
        )
    return seeded


def _collect_param_descriptions(
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


__all__ = [
    "FORMAT_VERSION",
    "filter_override_for_descriptor",
    "load_sections",
    "load_tools",
    "seed_sections",
    "seed_tools",
    "serialize_sections",
    "serialize_tools",
    "validate_header",
    "validate_sections_for_write",
    "validate_tools_for_write",
]
