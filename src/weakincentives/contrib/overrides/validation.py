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
from pathlib import Path
from typing import Literal, cast

from weakincentives.prompt.overrides import (
    HexDigest,
    PromptDescriptor,
    PromptLike,
    PromptOverride,
    PromptOverridesError,
    SectionDescriptor,
    SectionOverride,
    TaskExampleOverride,
    TaskStepOverride,
    ToolContractProtocol,
    ToolDescriptor,
    ToolOverride,
    ensure_hex_digest,
)
from weakincentives.runtime.logging import StructuredLogger, get_logger
from weakincentives.types import JSONValue

_LOGGER: StructuredLogger = get_logger(
    __name__, context={"component": "prompt_overrides"}
)
FORMAT_VERSION = 2


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


@dataclass(slots=True)
class SectionValidationConfig:
    strict: bool
    path_display: str
    body_error_message: str


def _normalize_section_override(
    *,
    path: tuple[str, ...],
    descriptor_section: SectionDescriptor | None,
    expected_hash: JSONValue,
    body: JSONValue,
    summary: str | None,
    visibility: Literal["full", "summary"] | None,
    config: SectionValidationConfig,
) -> SectionOverride | None:
    if descriptor_section is None:
        if config.strict:
            raise PromptOverridesError(
                f"Unknown section path for override: {config.path_display}"
            )
        _LOGGER.debug(
            "Skipping unknown override section path.",
            event="prompt_override_unknown_section",
            context={"path": config.path_display},
        )
        return None
    expected_digest = ensure_hex_digest(
        cast(HexDigest | str, expected_hash),
        field_name="Section expected_hash",
    )
    if expected_digest != descriptor_section.content_hash:
        if config.strict:
            msg = (
                f"Hash mismatch for section {config.path_display}: expected "
                f"{descriptor_section.content_hash}, got {expected_digest}."
            )
            raise PromptOverridesError(msg)
        _LOGGER.debug(
            "Skipping stale section override.",
            event="prompt_override_stale_section",
            context={
                "path": config.path_display,
                "expected_hash": str(descriptor_section.content_hash),
                "found_hash": str(expected_digest),
            },
        )
        return None
    if not isinstance(body, str):
        raise PromptOverridesError(config.body_error_message)
    return SectionOverride(
        path=path,
        expected_hash=expected_digest,
        body=body,
        summary=summary,
        visibility=visibility,
    )


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


def _normalize_tool_override(
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


def _parse_section_summary(
    section_payload: Mapping[str, JSONValue],
) -> str | None:
    """Parse and validate the summary field from a section override payload."""
    summary = section_payload.get("summary")
    if summary is not None and not isinstance(summary, str):
        raise PromptOverridesError("Section summary must be a string.")
    return summary


def _parse_section_visibility(
    section_payload: Mapping[str, JSONValue],
) -> Literal["full", "summary"] | None:
    """Parse and validate the visibility field from a section override payload."""
    visibility = section_payload.get("visibility")
    if visibility is None:
        return None
    if visibility not in {"full", "summary"}:
        raise PromptOverridesError(
            f"Section visibility must be 'full' or 'summary', got {visibility!r}."
        )
    return visibility  # type: ignore[return-value]


def _load_section_override_entry(
    *,
    path_key_raw: object,
    section_payload_raw: JSONValue,
    descriptor_index: Mapping[tuple[str, ...], SectionDescriptor],
) -> tuple[tuple[str, ...], SectionOverride] | None:
    if not isinstance(path_key_raw, str):
        raise PromptOverridesError("Section keys must be strings.")
    path_key = path_key_raw
    path = tuple(part for part in path_key.split("/") if part)
    if not isinstance(section_payload_raw, Mapping):
        raise PromptOverridesError("Section payload must be an object.")
    section_payload = cast(Mapping[str, JSONValue], section_payload_raw)
    expected_hash = section_payload.get("expected_hash")
    body = section_payload.get("body")
    summary = _parse_section_summary(section_payload)
    visibility = _parse_section_visibility(section_payload)
    config = SectionValidationConfig(
        strict=False,
        path_display=path_key,
        body_error_message="Section body must be a string.",
    )
    section_override = _normalize_section_override(
        path=path,
        descriptor_section=descriptor_index.get(path),
        expected_hash=expected_hash,
        body=body,
        summary=summary,
        visibility=visibility,
        config=config,
    )
    if section_override is None:
        return None
    return path, section_override


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
    tool_override = _normalize_tool_override(
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
        normalized_section = _load_section_override_entry(
            path_key_raw=path_key_raw,
            section_payload_raw=section_payload_raw,
            descriptor_index=descriptor_index,
        )
        if normalized_section is not None:
            path, section_override = normalized_section
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
        section_config = SectionValidationConfig(
            strict=False,
            path_display=_format_section_path(path),
            body_error_message="Section override body must be a string.",
        )
        normalized_section = _normalize_section_override(
            path=path,
            descriptor_section=descriptor_sections.get(path),
            expected_hash=section_override.expected_hash,
            body=section_override.body,
            summary=section_override.summary,
            visibility=section_override.visibility,
            config=section_config,
        )
        if normalized_section is not None:
            filtered_sections[path] = normalized_section

    filtered_tools: dict[str, ToolOverride] = {}
    for name, tool_override in override.tool_overrides.items():
        tool_config = ToolValidationConfig(
            strict=False,
            description_error_message="Tool description override must be a string when set.",
            param_mapping_error_message="Tool param_descriptions must be a mapping when provided.",
            param_entry_error_message="Tool param description entries must be strings.",
        )
        normalized_tool = _normalize_tool_override(
            name=name,
            descriptor_tool=descriptor_tools.get(name),
            expected_hash=tool_override.expected_contract_hash,
            description=tool_override.description,
            param_descriptions=tool_override.param_descriptions,
            config=tool_config,
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
        normalized_tool = _load_tool_override_entry(
            tool_name_raw=tool_name_raw,
            tool_payload_raw=tool_payload_raw,
            descriptor_index=descriptor_index,
        )
        if normalized_tool is not None:
            name, tool_override = normalized_tool
            overrides[name] = tool_override
    return overrides


def validate_sections_for_write(
    sections: Mapping[tuple[str, ...], SectionOverride],
    descriptor: PromptDescriptor,
) -> dict[tuple[str, ...], SectionOverride]:
    descriptor_index = _section_descriptor_index(descriptor)
    validated: dict[tuple[str, ...], SectionOverride] = {}
    for path, section_override in sections.items():
        path_display = "/".join(path)
        section_config = SectionValidationConfig(
            strict=True,
            path_display=path_display,
            body_error_message=(
                f"Section override body must be a string for {path_display}."
            ),
        )
        normalized_section = _normalize_section_override(
            path=path,
            descriptor_section=descriptor_index.get(path),
            expected_hash=section_override.expected_hash,
            body=section_override.body,
            summary=section_override.summary,
            visibility=section_override.visibility,
            config=section_config,
        )
        validated[path] = cast(SectionOverride, normalized_section)
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
        normalized_tool = _normalize_tool_override(
            name=name,
            descriptor_tool=descriptor_index.get(name),
            expected_hash=tool_override.expected_contract_hash,
            description=tool_override.description,
            param_descriptions=tool_override.param_descriptions,
            config=tool_config,
        )
        validated[name] = cast(ToolOverride, normalized_tool)
    return validated


def serialize_sections(
    sections: Mapping[tuple[str, ...], SectionOverride],
) -> dict[str, dict[str, object]]:
    serialized: dict[str, dict[str, object]] = {}
    for path, section_override in sections.items():
        key = "/".join(path)
        entry: dict[str, object] = {
            "path": list(path),
            "expected_hash": str(section_override.expected_hash),
            "body": section_override.body,
        }
        if section_override.summary is not None:
            entry["summary"] = section_override.summary
        if section_override.visibility is not None:
            entry["visibility"] = section_override.visibility
        serialized[key] = entry
    return serialized


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


def _parse_task_example_path(item_map: Mapping[str, JSONValue]) -> tuple[str, ...]:
    """Parse and validate the path field from a task example override."""
    path_raw = item_map.get("path")
    if not isinstance(path_raw, list):
        raise PromptOverridesError("Task example override path must be a list.")
    return tuple(str(p) for p in path_raw)


def _parse_task_example_index(item_map: Mapping[str, JSONValue]) -> int:
    """Parse and validate the index field from a task example override."""
    index = item_map.get("index")
    if not isinstance(index, int):
        raise PromptOverridesError("Task example override index must be an integer.")
    return index


def _parse_task_example_hash(
    item_map: Mapping[str, JSONValue],
) -> HexDigest | None:
    """Parse and validate the expected_hash field from a task example override."""
    expected_hash_raw = item_map.get("expected_hash")
    if expected_hash_raw is None:
        return None
    if not isinstance(expected_hash_raw, str):
        raise PromptOverridesError("Task example expected_hash must be a string.")
    return ensure_hex_digest(expected_hash_raw, field_name="Task example expected_hash")


def _parse_task_example_action(
    item_map: Mapping[str, JSONValue],
) -> Literal["modify", "remove", "append"]:
    """Parse and validate the action field from a task example override."""
    action = item_map.get("action")
    if action not in {"modify", "remove", "append"}:
        raise PromptOverridesError(
            f"Task example override action must be 'modify', 'remove', or 'append', got {action!r}."
        )
    return action  # type: ignore[return-value]


def _parse_steps_to_remove(item_map: Mapping[str, JSONValue]) -> tuple[int, ...]:
    """Parse and validate the steps_to_remove field."""
    steps_to_remove_raw = item_map.get("steps_to_remove")
    if steps_to_remove_raw is None:
        return ()
    if not isinstance(steps_to_remove_raw, list):
        raise PromptOverridesError("steps_to_remove must be a list.")
    return tuple(i for i in steps_to_remove_raw if isinstance(i, int))


def _parse_task_example_entry(
    item_map: Mapping[str, JSONValue],
) -> TaskExampleOverride:
    """Parse a single task example override entry."""
    path = _parse_task_example_path(item_map)
    index = _parse_task_example_index(item_map)
    expected_hash = _parse_task_example_hash(item_map)
    action = _parse_task_example_action(item_map)

    objective = item_map.get("objective")
    if objective is not None and not isinstance(objective, str):
        raise PromptOverridesError("Task example objective must be a string.")
    outcome = item_map.get("outcome")
    if outcome is not None and not isinstance(outcome, str):
        raise PromptOverridesError("Task example outcome must be a string.")

    return TaskExampleOverride(
        path=path,
        index=index,
        expected_hash=expected_hash,
        action=action,
        objective=objective,
        outcome=outcome,
        step_overrides=_load_step_overrides(item_map.get("step_overrides")),
        steps_to_remove=_parse_steps_to_remove(item_map),
        steps_to_append=_load_step_overrides(item_map.get("steps_to_append")),
    )


def load_task_example_overrides(
    payload: JSONValue | None,
) -> tuple[TaskExampleOverride, ...]:
    """Load task example overrides from JSON payload."""
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise PromptOverridesError("Task example overrides must be a list.")
    overrides: list[TaskExampleOverride] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise PromptOverridesError("Task example override entry must be an object.")
        item_map = cast(Mapping[str, JSONValue], item)
        overrides.append(_parse_task_example_entry(item_map))
    return tuple(overrides)


def _parse_optional_string(
    item_map: Mapping[str, JSONValue],
    field: str,
    error_msg: str,
) -> str | None:
    """Parse an optional string field from a mapping."""
    value = item_map.get(field)
    if value is not None and not isinstance(value, str):
        raise PromptOverridesError(error_msg)
    return value


def _parse_step_override_entry(item_map: Mapping[str, JSONValue]) -> TaskStepOverride:
    """Parse a single step override entry."""
    index = item_map.get("index")
    if not isinstance(index, int):
        raise PromptOverridesError("Step override index must be an integer.")
    return TaskStepOverride(
        index=index,
        tool_name=_parse_optional_string(
            item_map, "tool_name", "Step override tool_name must be a string."
        ),
        description=_parse_optional_string(
            item_map, "description", "Step override description must be a string."
        ),
        input_json=_parse_optional_string(
            item_map, "input_json", "Step override input_json must be a string."
        ),
        output_json=_parse_optional_string(
            item_map, "output_json", "Step override output_json must be a string."
        ),
    )


def _load_step_overrides(
    payload: JSONValue | None,
) -> tuple[TaskStepOverride, ...]:
    """Load step overrides from JSON payload."""
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise PromptOverridesError("Step overrides must be a list.")
    steps: list[TaskStepOverride] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise PromptOverridesError("Step override entry must be an object.")
        item_map = cast(Mapping[str, JSONValue], item)
        steps.append(_parse_step_override_entry(item_map))
    return tuple(steps)


def serialize_task_example_overrides(
    overrides: tuple[TaskExampleOverride, ...],
) -> list[dict[str, JSONValue]]:
    """Serialize task example overrides to JSON-compatible format."""
    serialized: list[dict[str, JSONValue]] = []
    for override in overrides:
        entry: dict[str, JSONValue] = {
            "path": list(override.path),
            "index": override.index,
            "expected_hash": str(override.expected_hash)
            if override.expected_hash
            else None,
            "action": override.action,
        }
        if override.objective is not None:
            entry["objective"] = override.objective
        if override.outcome is not None:
            entry["outcome"] = override.outcome
        if override.step_overrides:
            entry["step_overrides"] = _serialize_step_overrides(override.step_overrides)
        if override.steps_to_remove:
            entry["steps_to_remove"] = list(override.steps_to_remove)
        if override.steps_to_append:
            entry["steps_to_append"] = _serialize_step_overrides(
                override.steps_to_append
            )
        serialized.append(entry)
    return serialized


def _serialize_step_overrides(
    steps: tuple[TaskStepOverride, ...],
) -> list[dict[str, JSONValue]]:
    """Serialize step overrides to JSON-compatible format."""
    serialized: list[dict[str, JSONValue]] = []
    for step in steps:
        entry: dict[str, JSONValue] = {"index": step.index}
        if step.tool_name is not None:
            entry["tool_name"] = step.tool_name
        if step.description is not None:
            entry["description"] = step.description
        if step.input_json is not None:
            entry["input_json"] = step.input_json
        if step.output_json is not None:
            entry["output_json"] = step.output_json
        serialized.append(entry)
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
            path=section.path,
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


__all__ = [
    "FORMAT_VERSION",
    "filter_override_for_descriptor",
    "load_sections",
    "load_task_example_overrides",
    "load_tools",
    "seed_sections",
    "seed_tools",
    "serialize_sections",
    "serialize_task_example_overrides",
    "serialize_tools",
    "validate_header",
    "validate_sections_for_write",
    "validate_tools_for_write",
]
