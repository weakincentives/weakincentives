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

from collections.abc import Iterable, Mapping
from dataclasses import MISSING, Field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, cast, overload, get_args, get_origin, get_type_hints

from ...runtime.logging import StructuredLogger, get_logger
from ...types import JSONValue
from .versioning import (
    HexDigest,
    PromptDescriptor,
    PromptLike,
    PromptOverride,
    PromptOverridesError,
    SectionDescriptor,
    SectionOverride,
    ToolContractProtocol,
    ToolDescriptor,
    ToolParamDescription,
    ToolOverride,
    ensure_hex_digest,
)

_LOGGER: StructuredLogger = get_logger(
    __name__, context={"component": "prompt_overrides"}
)
FORMAT_VERSION = 1


def validate_header(
    payload: Mapping[str, JSONValue],
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
    expected_hash: JSONValue,
    body: JSONValue,
    strict: Literal[True],
    path_display: str,
    body_error_message: str,
) -> SectionOverride: ...


@overload
def _normalize_section_override(
    *,
    path: tuple[str, ...],
    descriptor_section: SectionDescriptor | None,
    expected_hash: JSONValue,
    body: JSONValue,
    strict: Literal[False],
    path_display: str,
    body_error_message: str,
) -> SectionOverride | None: ...


def _normalize_section_override(
    *,
    path: tuple[str, ...],
    descriptor_section: SectionDescriptor | None,
    expected_hash: JSONValue,
    body: JSONValue,
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
    expected_hash: JSONValue,
    description: JSONValue,
    param_descriptions: JSONValue,
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
    expected_hash: JSONValue,
    description: JSONValue,
    param_descriptions: JSONValue,
    strict: Literal[False],
    description_error_message: str,
    param_mapping_error_message: str,
    param_entry_error_message: str,
) -> ToolOverride | None: ...


def _normalize_tool_override(
    *,
    name: str,
    descriptor_tool: ToolDescriptor | None,
    expected_hash: JSONValue,
    description: JSONValue,
    param_descriptions: JSONValue,
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
    mapping_params = cast(Mapping[str, JSONValue], param_descriptions)
    normalized_params: dict[str, ToolParamDescription] = {}
    for key, value in mapping_params.items():
        normalized_params[key] = _normalize_param_description_entry(
            field_name=key,
            payload=value,
            entry_error_message=param_entry_error_message,
        )
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
    payload: JSONValue | None,
    descriptor: PromptDescriptor,
) -> dict[tuple[str, ...], SectionOverride]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise PromptOverridesError("Sections payload must be a mapping.")
    if not payload:
        return {}
    mapping_payload = cast(Mapping[object, JSONValue], payload)
    mapping_entries = cast(Iterable[tuple[object, JSONValue]], mapping_payload.items())
    descriptor_index = _section_descriptor_index(descriptor)
    overrides: dict[tuple[str, ...], SectionOverride] = {}
    for path_key_raw, section_payload_raw in mapping_entries:
        if not isinstance(path_key_raw, str):
            raise PromptOverridesError("Section keys must be strings.")
        path_key = path_key_raw
        path = tuple(part for part in path_key.split("/") if part)
        if not isinstance(section_payload_raw, Mapping):
            raise PromptOverridesError("Section payload must be an object.")
        section_payload = cast(Mapping[str, JSONValue], section_payload_raw)
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
            param_entry_error_message=(
                "Tool param description entries must be strings or metadata objects."
            ),
        )
        if normalized_tool is not None:
            filtered_tools[name] = normalized_tool

    return filtered_sections, filtered_tools


def load_tools(
    payload: JSONValue | None,
    descriptor: PromptDescriptor,
) -> dict[str, ToolOverride]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise PromptOverridesError("Tools payload must be a mapping.")
    if not payload:
        return {}
    mapping_payload = cast(Mapping[object, JSONValue], payload)
    mapping_entries = cast(Iterable[tuple[object, JSONValue]], mapping_payload.items())
    descriptor_index = _tool_descriptor_index(descriptor)
    overrides: dict[str, ToolOverride] = {}
    for tool_name_raw, tool_payload_raw in mapping_entries:
        if not isinstance(tool_name_raw, str):
            raise PromptOverridesError("Tool names must be strings.")
        tool_name = tool_name_raw
        if not isinstance(tool_payload_raw, Mapping):
            raise PromptOverridesError("Tool payload must be an object.")
        tool_payload = cast(Mapping[str, JSONValue], tool_payload_raw)
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
            param_entry_error_message=(
                "Tool param description entries must be strings or metadata objects."
            ),
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
            expected_hash=section_override.expected_hash,
            body=section_override.body,
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
            expected_hash=tool_override.expected_contract_hash,
            description=tool_override.description,
            param_descriptions=tool_override.param_descriptions,
            strict=True,
            description_error_message=(
                f"Tool description override must be a string for {name}."
            ),
            param_mapping_error_message=(
                f"Tool parameter descriptions must be a mapping for {name}."
            ),
            param_entry_error_message=(
                f"Tool parameter descriptions must map strings to metadata objects for {name}."
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


def _serialize_param_descriptions(
    descriptions: Mapping[str, ToolParamDescription],
) -> dict[str, dict[str, JSONValue]]:
    serialized: dict[str, dict[str, JSONValue]] = {}
    for field_name, metadata in descriptions.items():
        serialized[field_name] = {
            "description": metadata.description,
            "type": metadata.type_name,
            "has_default": metadata.has_default,
            "default": metadata.default,
        }
    return serialized


def _normalize_param_description_entry(
    *,
    field_name: str,
    payload: JSONValue,
    entry_error_message: str,
) -> ToolParamDescription:
    if isinstance(payload, ToolParamDescription):
        return payload
    if isinstance(payload, str):
        return ToolParamDescription(description=payload)
    if not isinstance(payload, Mapping):
        raise PromptOverridesError(entry_error_message)
    mapping = cast(Mapping[str, JSONValue], payload)
    description_value = mapping.get("description")
    if not isinstance(description_value, str):
        raise PromptOverridesError(
            f"Tool parameter metadata for {field_name} must include a description string."
        )
    type_value = mapping.get("type")
    if type_value is not None and not isinstance(type_value, str):
        raise PromptOverridesError(
            f"Tool parameter metadata for {field_name} must encode the type name as a string when provided."
        )
    has_default_value = mapping.get("has_default")
    if has_default_value is None:
        has_default = mapping.get("default") is not None
    elif isinstance(has_default_value, bool):
        has_default = has_default_value
    else:
        raise PromptOverridesError(
            f"Tool parameter metadata for {field_name} must encode has_default as a boolean when provided."
        )
    default_value = mapping.get("default")
    if default_value is not None and not isinstance(default_value, str):
        raise PromptOverridesError(
            f"Tool parameter metadata for {field_name} must encode the default as a string when provided."
        )
    default_repr: str | None = cast(str | None, default_value)
    if not has_default:
        default_repr = None
    return ToolParamDescription(
        description=description_value,
        type_name=type_value,
        has_default=has_default,
        default=default_repr,
    )


def serialize_tools(
    tools: Mapping[str, ToolOverride],
) -> dict[str, dict[str, JSONValue]]:
    serialized: dict[str, dict[str, JSONValue]] = {}
    for name, tool_override in tools.items():
        serialized[name] = {
            "expected_contract_hash": str(tool_override.expected_contract_hash),
            "description": tool_override.description,
            "param_descriptions": _serialize_param_descriptions(
                tool_override.param_descriptions
            ),
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
    tool_lookup: dict[str, ToolContractProtocol] = {}
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
        param_descriptions = _collect_param_metadata(tool_obj)
        seeded[tool.name] = ToolOverride(
            name=tool.name,
            expected_contract_hash=tool.contract_hash,
            description=tool_obj.description,
            param_descriptions=param_descriptions,
        )
    return seeded


def _collect_param_metadata(tool: ToolContractProtocol) -> dict[str, ToolParamDescription]:
    params_type = getattr(tool, "params_type", None)
    if not isinstance(params_type, type) or not is_dataclass(params_type):
        return {}
    hints = _safe_get_type_hints(params_type)
    catalog: dict[str, ToolParamDescription] = {}
    _collect_dataclass_fields(
        dataclass_type=params_type,
        hints=hints,
        prefix=(),
        catalog=catalog,
        visited=set(),
    )
    return catalog


def _collect_dataclass_fields(
    *,
    dataclass_type: type[Any],
    hints: Mapping[str, Any],
    prefix: tuple[str, ...],
    catalog: dict[str, ToolParamDescription],
    visited: set[type[Any]],
) -> None:
    if dataclass_type in visited:
        return
    visited.add(dataclass_type)
    for field in fields(dataclass_type):
        field_path = _format_param_path((*prefix, field.name))
        description = _field_description(field, field_path)
        annotation = hints.get(field.name, field.type)
        type_name = _format_type_name(annotation)
        has_default = _field_has_default(field)
        default_repr = _field_default_repr(field) if has_default else None
        catalog[field_path] = ToolParamDescription(
            description=description,
            type_name=type_name,
            has_default=has_default,
            default=default_repr,
        )
        nested_dataclass = _resolve_nested_dataclass(annotation)
        if nested_dataclass is not None:
            nested_hints = _safe_get_type_hints(nested_dataclass)
            _collect_dataclass_fields(
                dataclass_type=nested_dataclass,
                hints=nested_hints,
                prefix=(*prefix, field.name),
                catalog=catalog,
                visited=visited,
            )
    visited.remove(dataclass_type)


def _format_param_path(parts: tuple[str, ...]) -> str:
    return ".".join(parts)


def _field_description(field: Field[Any], path: str) -> str:
    if field.metadata:
        description = field.metadata.get("description")
    else:
        description = None
    if isinstance(description, str) and description.strip():
        return description
    return f"Describe the `{path}` parameter."


def _field_has_default(field: Field[Any]) -> bool:
    return field.default is not MISSING or field.default_factory is not MISSING


def _field_default_repr(field: Field[Any]) -> str:
    if field.default is not MISSING:
        return repr(field.default)
    factory = field.default_factory
    factory_name = getattr(factory, "__qualname__", getattr(factory, "__name__", repr(factory)))
    module = getattr(factory, "__module__", None)
    if module:
        return f"<factory {module}.{factory_name}>"
    return f"<factory {factory_name}>"


def _safe_get_type_hints(dataclass_type: type[Any]) -> Mapping[str, Any]:
    try:
        return get_type_hints(dataclass_type, include_extras=True)
    except Exception:
        return {}


def _format_type_name(annotation: Any) -> str | None:
    if annotation is None:
        return None
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type):
            return annotation.__name__
        return repr(annotation)
    if getattr(origin, "__qualname__", "") == "Annotated":
        args = get_args(annotation)
        if args:
            return _format_type_name(args[0])
        return "Annotated"
    args = get_args(annotation)
    origin_name = getattr(origin, "__name__", repr(origin))
    if not args:
        return origin_name
    arg_names = ", ".join(
        filter(None, (_format_type_name(arg) for arg in args))
    )
    return f"{origin_name}[{arg_names}]"


def _resolve_nested_dataclass(annotation: Any) -> type[Any] | None:
    for candidate in _iter_possible_dataclass_types(annotation):
        if isinstance(candidate, type) and is_dataclass(candidate):
            return candidate
    return None


def _iter_possible_dataclass_types(annotation: Any) -> tuple[Any, ...]:
    origin = get_origin(annotation)
    if origin is None:
        return (annotation,)
    if getattr(origin, "__qualname__", "") == "Annotated":
        args = get_args(annotation)
        if args:
            return _iter_possible_dataclass_types(args[0])
        return ()
    args = get_args(annotation)
    if not args:
        return ()
    flattened: list[Any] = []
    for arg in args:
        flattened.extend(_iter_possible_dataclass_types(arg))
    return tuple(flattened)


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
