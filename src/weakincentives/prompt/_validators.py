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

# pyright: reportImportCycles=false
"""Validation helpers for :mod:`weakincentives.prompt.registry`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any, cast

from ..types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)
from .errors import PromptValidationError, SectionPath
from .tool import Tool

if TYPE_CHECKING:
    from .registry import PromptRegistry, SectionNode
    from .section import Section


# ---------------------------------------------------------------------------
# Section registration validators
# ---------------------------------------------------------------------------


def validate_section_params_type(
    section: Section[SupportsDataclass],
    path: SectionPath,
) -> type[SupportsDataclass] | None:
    """Validate and return the section's params type, or *None*."""

    params_type = section.params_type
    if params_type is not None and not is_dataclass(params_type):
        raise PromptValidationError(
            "Section params must be a dataclass.",
            section_path=path,
            dataclass_type=params_type,
        )
    return params_type


def validate_section_defaults(
    default_value: object,
    path: SectionPath,
    params_type: type[SupportsDataclass],
) -> None:
    """Validate that *default_value* is a dataclass instance matching *params_type*."""

    if isinstance(default_value, type) or not is_dataclass(default_value):
        raise PromptValidationError(
            "Section defaults must be dataclass instances.",
            section_path=path,
            dataclass_type=params_type,
        )
    if type(default_value) is not params_type:
        raise PromptValidationError(
            "Section defaults must match section params type.",
            section_path=path,
            dataclass_type=params_type,
        )


def validate_section_placeholders(
    placeholder_names: set[str],
    path: SectionPath,
    params_type: type[SupportsDataclass] | None,
) -> None:
    """Validate placeholder names are backed by fields in *params_type*."""

    if params_type is None:
        if placeholder_names:
            placeholder = sorted(placeholder_names)[0]
            raise PromptValidationError(
                "Section does not accept parameters but declares placeholders.",
                section_path=path,
                placeholder=placeholder,
            )
        return

    param_fields = {field.name for field in fields(params_type)}
    unknown_placeholders = placeholder_names - param_fields
    if unknown_placeholders:
        placeholder = sorted(unknown_placeholders)[0]
        raise PromptValidationError(
            "Template references unknown placeholder.",
            section_path=path,
            dataclass_type=params_type,
            placeholder=placeholder,
        )


def validate_tool_instance(
    tool: object,
    path: SectionPath,
    params_type: type[SupportsDataclass] | None,
) -> None:
    """Validate that *tool* is a :class:`Tool` instance."""

    if not isinstance(tool, Tool):
        raise PromptValidationError(
            "Section tools must be Tool instances.",
            section_path=path,
            dataclass_type=params_type,
        )


def validate_tool_registration(
    tool: Tool[SupportsDataclassOrNone, SupportsToolResult],
    path: SectionPath,
    tool_name_registry: Mapping[str, SectionPath],
) -> None:
    """Validate tool params type and uniqueness within the registry."""

    params_type = tool.params_type
    if params_type is not type(None) and not is_dataclass(params_type):
        raise PromptValidationError(
            "Tool parameters must be dataclass types.",
            section_path=path,
            dataclass_type=params_type,
        )

    existing_path = tool_name_registry.get(tool.name)
    if existing_path is not None:
        raise PromptValidationError(
            "Duplicate tool name registered for prompt.",
            section_path=path,
            dataclass_type=tool.params_type,
        )


def validate_skill_registration(
    name: str,
    path: SectionPath,
    skill_name_registry: Mapping[str, SectionPath],
) -> None:
    """Validate that skill *name* is unique within the registry."""

    existing_path = skill_name_registry.get(name)
    if existing_path is not None:
        raise PromptValidationError(
            f"Duplicate skill name '{name}' registered for prompt.",
            section_path=path,
        )


# ---------------------------------------------------------------------------
# Invariant callbacks for PromptRegistry
# ---------------------------------------------------------------------------


def registry_paths_are_registered(
    registry: PromptRegistry,
) -> tuple[bool, str] | bool:
    """Ensure internal registries only reference known section nodes."""

    node_by_path = {
        node.path: node
        for node in registry._section_nodes  # pyright: ignore[reportPrivateUsage]
    }
    validations = (
        _validate_default_paths(registry, node_by_path),
        _validate_placeholders(registry, node_by_path),
        _validate_tool_registry(registry, node_by_path),
        _validate_defaults_by_type(registry),
    )

    for validation in validations:
        if validation is not True:
            return validation

    return True


def params_registry_is_consistent(
    registry: PromptRegistry,
) -> tuple[bool, str] | bool:
    """Ensure params registry entries point at known nodes with matching types."""

    section_nodes = list(registry._section_nodes)  # pyright: ignore[reportPrivateUsage]
    params_registry = registry._params_registry  # pyright: ignore[reportPrivateUsage]
    for params_type, nodes in params_registry.items():
        validation = _validate_param_nodes(section_nodes, params_type, nodes)
        if validation is not True:
            return validation

    return True


# ---------------------------------------------------------------------------
# Helpers for invariant callbacks
# ---------------------------------------------------------------------------


def _validate_default_paths(
    registry: PromptRegistry,
    node_by_path: Mapping[SectionPath, SectionNode[SupportsDataclass]],
) -> tuple[bool, str] | bool:
    defaults_by_path = registry._defaults_by_path  # pyright: ignore[reportPrivateUsage]
    unknown_default_paths = [
        path for path in defaults_by_path if path not in node_by_path
    ]
    if unknown_default_paths:
        return (
            False,
            f"defaults reference unknown paths: {sorted(unknown_default_paths)!r}",
        )

    for path, default in defaults_by_path.items():
        node = node_by_path[path]
        params_type = node.section.params_type
        if params_type is None:
            return False, f"section at {path!r} does not accept params but has defaults"
        if type(default) is not params_type:
            return False, (
                "default params type mismatch for path "
                f"{path!r}: expected {params_type.__name__}, got {type(default).__name__}"
            )

    return True


def _validate_placeholders(
    registry: PromptRegistry,
    node_by_path: Mapping[SectionPath, SectionNode[SupportsDataclass]],
) -> tuple[bool, str] | bool:
    placeholders = registry._placeholders  # pyright: ignore[reportPrivateUsage]
    unknown_placeholder_paths = [
        path for path in placeholders if path not in node_by_path
    ]
    if unknown_placeholder_paths:
        return False, (
            "placeholders reference unknown paths: "
            f"{sorted(unknown_placeholder_paths)!r}"
        )

    return True


def _validate_tool_registry(
    registry: PromptRegistry,
    node_by_path: Mapping[SectionPath, SectionNode[SupportsDataclass]],
) -> tuple[bool, str] | bool:
    tool_name_registry = registry._tool_name_registry  # pyright: ignore[reportPrivateUsage]
    unknown_tool_paths = [
        path for path in tool_name_registry.values() if path not in node_by_path
    ]
    if unknown_tool_paths:
        return False, f"tools reference unknown paths: {sorted(unknown_tool_paths)!r}"

    return True


def _validate_defaults_by_type(
    registry: PromptRegistry,
) -> tuple[bool, str] | bool:
    defaults_by_type = registry._defaults_by_type  # pyright: ignore[reportPrivateUsage]
    for params_type, default in defaults_by_type.items():
        if type(default) is not params_type:
            return False, (
                "default by type mismatch for "
                f"{params_type.__name__}: got {type(default).__name__}"
            )

    return True


def _validate_param_nodes(
    section_nodes: Sequence[SectionNode[SupportsDataclass]],
    params_type: type[SupportsDataclass],
    nodes: Sequence[SectionNode[SupportsDataclass]],
) -> tuple[bool, str] | bool:
    for node in nodes:
        if node not in section_nodes:
            return False, (
                "params registry references unknown node at path "
                f"{node.path!r} for {params_type.__name__}"
            )
        node_params_type = node.section.params_type
        if node_params_type is None:
            return False, (
                "params registry references section without params at path "
                f"{node.path!r}"
            )
        if node_params_type is not params_type:
            return False, (
                "params registry type mismatch for path "
                f"{node.path!r}: expected {params_type.__name__}, "
                f"found {node_params_type.__name__}"
            )

    return True


# ---------------------------------------------------------------------------
# Task-example validation
# ---------------------------------------------------------------------------


def validate_task_examples(
    section_nodes: Sequence[SectionNode[SupportsDataclass]],
    tool_name_registry: Mapping[str, SectionPath],
    structured_output_type: type[SupportsDataclass] | None = None,
) -> None:
    """Validate task example tool references and type coherence."""
    # Import here to avoid circular imports
    from .task_examples import TaskExample

    # Build a map from tool name to Tool instance
    tool_instances: dict[str, Tool[SupportsDataclassOrNone, SupportsToolResult]] = {}
    for node in section_nodes:
        for tool in node.section.tools():
            tool_instances[tool.name] = tool

    # Find all TaskExample sections and validate their steps
    for node in section_nodes:
        if not isinstance(node.section, TaskExample):
            continue

        task_example = cast(TaskExample[Any], node.section)
        _validate_task_example_steps(
            task_example,
            node.path,
            tool_name_registry,
            tool_instances,
        )
        _validate_task_example_outcome(
            task_example,
            node.path,
            structured_output_type,
        )


def _validate_task_example_outcome(
    task_example: object,
    path: SectionPath,
    structured_output_type: type[SupportsDataclass] | None,
) -> None:
    """Validate that task example outcome matches prompt's output type."""
    outcome = getattr(task_example, "outcome", None)

    if structured_output_type is None:
        # Prompt has no structured output - outcome must be a string
        if not isinstance(outcome, str):
            msg = (
                f"Task example outcome must be a string when prompt has no "
                f"structured output. Got: {type(outcome).__name__}."
            )
            raise PromptValidationError(
                msg,
                section_path=path,
                placeholder="outcome",
            )
    elif type(outcome) is not structured_output_type:
        # Prompt has structured output - outcome must be instance of that type
        expected = structured_output_type.__name__
        actual = type(outcome).__name__
        msg = (
            f"Task example outcome type mismatch. Expected: {expected}, got: {actual}."
        )
        raise PromptValidationError(
            msg,
            section_path=path,
            placeholder="outcome",
        )


def _validate_task_example_steps(
    task_example: object,
    path: SectionPath,
    tool_name_registry: Mapping[str, SectionPath],
    tool_instances: Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
) -> None:
    """Validate steps in a task example."""
    available_tools = sorted(tool_name_registry.keys())
    steps = getattr(task_example, "steps", ())

    for step_idx, step in enumerate(steps):
        tool_name = getattr(step, "tool_name", "")

        # Check tool name exists
        if tool_name not in tool_name_registry:
            available_str = ", ".join(available_tools) if available_tools else "none"
            msg = (
                f'Unknown tool "{tool_name}" in task example step {step_idx}. '
                f"Available tools: {available_str}."
            )
            raise PromptValidationError(
                msg,
                section_path=path,
                placeholder="steps",
            )

        # Validate type coherence - tool_instances contains all registered tools
        tool = tool_instances[tool_name]
        _validate_step_type_coherence(
            step,
            step_idx,
            tool,
            path,
        )


def _validate_step_type_coherence(
    step: object,
    step_idx: int,
    tool: Tool[SupportsDataclassOrNone, SupportsToolResult],
    path: SectionPath,
) -> None:
    """Validate that step example types match the tool's types."""
    example = getattr(step, "example", None)
    tool_name = getattr(step, "tool_name", "unknown")

    _validate_step_input_type(example, step_idx, tool_name, tool.params_type, path)
    _validate_step_output_type(
        example, step_idx, tool_name, tool.result_type, tool.result_container, path
    )


def _validate_step_input_type(
    example: object,
    step_idx: int,
    tool_name: str,
    expected_type: type[SupportsDataclass] | type[None],
    path: SectionPath,
) -> None:
    """Validate step input matches expected params type."""
    example_input = getattr(example, "input", None)

    if expected_type is type(None):  # pragma: no cover
        if example_input is not None:
            msg = (
                f'Task example step {step_idx} input type mismatch for tool "{tool_name}". '
                f"Expected: None, got: {type(example_input).__name__}."
            )
            raise PromptValidationError(msg, section_path=path, placeholder="steps")
        return

    if example_input is None:
        msg = (
            f'Task example step {step_idx} input type mismatch for tool "{tool_name}". '
            f"Expected: {expected_type.__name__}, got: None."
        )
        raise PromptValidationError(msg, section_path=path, placeholder="steps")
    if type(example_input) is not expected_type:
        msg = (
            f'Task example step {step_idx} input type mismatch for tool "{tool_name}". '
            f"Expected: {expected_type.__name__}, got: {type(example_input).__name__}."
        )
        raise PromptValidationError(msg, section_path=path, placeholder="steps")


def _validate_step_output_type(  # noqa: PLR0913, PLR0917
    example: object,
    step_idx: int,
    tool_name: str,
    expected_type: type[SupportsDataclass] | type[None],
    container: str,
    path: SectionPath,
) -> None:
    """Validate step output matches expected result type."""
    example_output = getattr(example, "output", None)

    if expected_type is type(None):  # pragma: no cover
        if example_output is not None:
            msg = (
                f'Task example step {step_idx} output type mismatch for tool "{tool_name}". '
                f"Expected: None, got: {type(example_output).__name__}."
            )
            raise PromptValidationError(msg, section_path=path, placeholder="steps")
        return

    if container == "array":  # pragma: no cover
        _validate_array_output(example_output, step_idx, tool_name, expected_type, path)
        return

    if example_output is None:
        msg = (
            f'Task example step {step_idx} output type mismatch for tool "{tool_name}". '
            f"Expected: {expected_type.__name__}, got: None."
        )
        raise PromptValidationError(msg, section_path=path, placeholder="steps")
    if type(example_output) is not expected_type:
        msg = (
            f'Task example step {step_idx} output type mismatch for tool "{tool_name}". '
            f"Expected: {expected_type.__name__}, got: {type(example_output).__name__}."
        )
        raise PromptValidationError(msg, section_path=path, placeholder="steps")


def _validate_array_output(
    output: object,
    step_idx: int,
    tool_name: str,
    element_type: type[SupportsDataclass],
    path: SectionPath,
) -> None:
    """Validate array output contains correct element types."""
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes, bytearray)):
        msg = (
            f'Task example step {step_idx} output type mismatch for tool "{tool_name}". '
            f"Expected: sequence of {element_type.__name__}, got: {type(output).__name__}."
        )
        raise PromptValidationError(msg, section_path=path, placeholder="steps")
    for item in cast(Sequence[object], output):
        if type(item) is not element_type:
            msg = (
                f'Task example step {step_idx} output type mismatch for tool "{tool_name}". '
                f"Expected: sequence of {element_type.__name__}, "
                f"got item of type: {type(item).__name__}."
            )
            raise PromptValidationError(msg, section_path=path, placeholder="steps")
