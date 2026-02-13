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
from pathlib import Path

from ...runtime.logging import StructuredLogger, get_logger
from ...types import JSONValue
from ._section_overrides import (
    SectionValidationConfig,
    format_section_path,
    load_sections,
    normalize_section_override,
    section_descriptor_index,
    seed_sections,
    serialize_sections,
    validate_sections_for_write,
)
from ._task_example_overrides import (
    load_task_example_overrides,
    serialize_task_example_overrides,
)
from ._tool_overrides import (
    ToolValidationConfig,
    load_tools,
    normalize_tool_override,
    seed_tools,
    serialize_tools,
    tool_descriptor_index,
    validate_tools_for_write,
)
from .versioning import (
    PromptDescriptor,
    PromptOverride,
    PromptOverridesError,
    SectionOverride,
    ToolOverride,
)

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


def filter_override_for_descriptor(
    descriptor: PromptDescriptor,
    override: PromptOverride,
) -> tuple[dict[tuple[str, ...], SectionOverride], dict[str, ToolOverride]]:
    if override.ns != descriptor.ns or override.prompt_key != descriptor.key:
        _log_mismatched_override_metadata(descriptor, override)
        return {}, {}

    descriptor_sections = section_descriptor_index(descriptor)
    descriptor_tools = tool_descriptor_index(descriptor)

    filtered_sections: dict[tuple[str, ...], SectionOverride] = {}
    for path, section_override in override.sections.items():
        section_config = SectionValidationConfig(
            strict=False,
            path_display=format_section_path(path),
            body_error_message="Section override body must be a string.",
        )
        normalized_section = normalize_section_override(
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
        normalized_tool = normalize_tool_override(
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
