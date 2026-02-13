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
from dataclasses import dataclass, fields, is_dataclass
from typing import cast

from ...runtime.logging import StructuredLogger, get_logger
from ...types import JSONValue
from .versioning import (
    HexDigest,
    PromptDescriptor,
    PromptLike,
    PromptOverridesError,
    ToolContractProtocol,
    ToolDescriptor,
    ToolOverride,
    ensure_hex_digest,
)

_LOGGER: StructuredLogger = get_logger(
    __name__, context={"component": "prompt_overrides"}
)


def tool_descriptor_index(
    descriptor: PromptDescriptor,
) -> dict[str, ToolDescriptor]:
    return {tool.name: tool for tool in descriptor.tools}


@dataclass(slots=True)
class ToolValidationConfig:
    strict: bool
    description_error_message: str
    param_mapping_error_message: str
    param_entry_error_message: str


def _validate_tool_descriptor(
    *,
    name: str,
    descriptor_tool: ToolDescriptor | None,
    strict: bool,
) -> ToolDescriptor | None:
    if descriptor_tool is None:
        if strict:
            raise PromptOverridesError(f"Unknown tool override: {name}")
        _LOGGER.debug(
            "Skipping unknown tool override.",
            event="prompt_override_unknown_tool",
            context={"tool": name},
        )
        return None
    return descriptor_tool


def _validate_tool_expected_hash(
    *,
    name: str,
    descriptor_tool: ToolDescriptor,
    expected_hash: JSONValue,
    strict: bool,
) -> HexDigest | None:
    expected_digest = ensure_hex_digest(
        cast(HexDigest | str, expected_hash),
        field_name="Tool expected_contract_hash",
    )
    if expected_digest != descriptor_tool.contract_hash:
        if strict:
            msg = (
                f"Hash mismatch for tool override {name}: expected "
                f"{descriptor_tool.contract_hash}, got {expected_digest}."
            )
            raise PromptOverridesError(msg)
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
    return expected_digest


def _normalize_tool_description(
    *,
    description: JSONValue,
    description_error_message: str,
) -> str | None:
    if description is not None and not isinstance(description, str):
        raise PromptOverridesError(description_error_message)
    return description if description is not None else None


def _normalize_param_descriptions(
    *,
    param_descriptions: JSONValue,
    param_mapping_error_message: str,
    param_entry_error_message: str,
) -> dict[str, str]:
    if param_descriptions is None:
        param_descriptions = {}
    if not isinstance(param_descriptions, Mapping):
        raise PromptOverridesError(param_mapping_error_message)
    mapping_params = cast(Mapping[str, JSONValue], param_descriptions)
    normalized_params: dict[str, str] = {}
    for key, value in mapping_params.items():
        if not isinstance(value, str):
            raise PromptOverridesError(param_entry_error_message)
        normalized_params[key] = value
    return normalized_params


def normalize_tool_override(
    *,
    name: str,
    descriptor_tool: ToolDescriptor | None,
    expected_hash: JSONValue,
    description: JSONValue,
    param_descriptions: JSONValue,
    config: ToolValidationConfig,
) -> ToolOverride | None:
    validated_tool = _validate_tool_descriptor(
        name=name,
        descriptor_tool=descriptor_tool,
        strict=config.strict,
    )
    if validated_tool is None:
        return None

    expected_digest = _validate_tool_expected_hash(
        name=name,
        descriptor_tool=validated_tool,
        expected_hash=expected_hash,
        strict=config.strict,
    )
    if expected_digest is None:
        return None

    normalized_description = _normalize_tool_description(
        description=description,
        description_error_message=config.description_error_message,
    )
    normalized_params = _normalize_param_descriptions(
        param_descriptions=param_descriptions,
        param_mapping_error_message=config.param_mapping_error_message,
        param_entry_error_message=config.param_entry_error_message,
    )
    return ToolOverride(
        name=name,
        expected_contract_hash=expected_digest,
        description=normalized_description,
        param_descriptions=normalized_params,
    )


def _load_tool_override_entry(
    *,
    tool_name_raw: object,
    tool_payload_raw: JSONValue,
    descriptor_index: Mapping[str, ToolDescriptor],
) -> tuple[str, ToolOverride] | None:
    if not isinstance(tool_name_raw, str):
        raise PromptOverridesError("Tool names must be strings.")
    if not isinstance(tool_payload_raw, Mapping):
        raise PromptOverridesError("Tool payload must be an object.")
    tool_name = tool_name_raw
    tool_payload = cast(Mapping[str, JSONValue], tool_payload_raw)
    expected_hash = tool_payload.get("expected_contract_hash")
    description = tool_payload.get("description")
    param_payload = tool_payload.get("param_descriptions")
    config = ToolValidationConfig(
        strict=False,
        description_error_message="Tool description must be a string when set.",
        param_mapping_error_message=(
            "Tool param_descriptions must be a mapping when provided."
        ),
        param_entry_error_message="Tool param description entries must be strings.",
    )
    tool_override = normalize_tool_override(
        name=tool_name,
        descriptor_tool=descriptor_index.get(tool_name),
        expected_hash=expected_hash,
        description=description,
        param_descriptions=param_payload,
        config=config,
    )
    if tool_override is None:
        return None
    return tool_name, tool_override


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
    descriptor_index = tool_descriptor_index(descriptor)
    overrides: dict[str, ToolOverride] = {}
    for tool_name_raw, tool_payload_raw in mapping_entries:
        normalized_tool = _load_tool_override_entry(
            tool_name_raw=tool_name_raw,
            tool_payload_raw=tool_payload_raw,
            descriptor_index=descriptor_index,
        )
        if normalized_tool is not None:
            name, tool_override = normalized_tool
            overrides[name] = tool_override
    return overrides


def validate_tools_for_write(
    tools: Mapping[str, ToolOverride],
    descriptor: PromptDescriptor,
) -> dict[str, ToolOverride]:
    if not tools:
        return {}
    descriptor_index = tool_descriptor_index(descriptor)
    validated: dict[str, ToolOverride] = {}
    for name, tool_override in tools.items():
        tool_config = ToolValidationConfig(
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
        normalized_tool = normalize_tool_override(
            name=name,
            descriptor_tool=descriptor_index.get(name),
            expected_hash=tool_override.expected_contract_hash,
            description=tool_override.description,
            param_descriptions=tool_override.param_descriptions,
            config=tool_config,
        )
        validated[name] = cast(ToolOverride, normalized_tool)
    return validated


def serialize_tools(
    tools: Mapping[str, ToolOverride],
) -> dict[str, dict[str, JSONValue]]:
    serialized: dict[str, dict[str, JSONValue]] = {}
    for name, tool_override in tools.items():
        serialized[name] = {
            "expected_contract_hash": str(tool_override.expected_contract_hash),
            "description": tool_override.description,
            "param_descriptions": dict(tool_override.param_descriptions),
        }
    return serialized


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
        param_descriptions = _collect_param_descriptions(tool_obj)
        seeded[tool.name] = ToolOverride(
            name=tool.name,
            expected_contract_hash=tool.contract_hash,
            description=tool_obj.description,
            param_descriptions=param_descriptions,
        )
    return seeded


def _collect_param_descriptions(tool: ToolContractProtocol) -> dict[str, str]:
    params_type = getattr(tool, "params_type", None)
    if not isinstance(params_type, type) or not is_dataclass(params_type):
        return {}
    descriptions: dict[str, str] = {}
    for field in fields(params_type):
        description = field.metadata.get("description") if field.metadata else None
        if isinstance(description, str) and description:
            descriptions[field.name] = description
    return descriptions
