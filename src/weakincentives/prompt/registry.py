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

"""Section registration helpers for :mod:`weakincentives.prompt`."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import fields, is_dataclass, replace
from types import MappingProxyType
from typing import Any, cast

from ..dataclasses import FrozenDataclass
from ..dbc import invariant
from ._types import SupportsDataclass, SupportsDataclassOrNone, SupportsToolResult
from .errors import PromptRenderError, PromptValidationError, SectionPath
from .section import Section
from .tool import Tool


@FrozenDataclass()
class SectionNode[ParamsT: SupportsDataclass]:
    """Flattened view of a section within a prompt."""

    section: Section[ParamsT]
    depth: int
    path: SectionPath
    number: str


@FrozenDataclass()
class RegistrySnapshot:
    """Immutable view over registered prompt sections."""

    sections: tuple[SectionNode[SupportsDataclass], ...]
    params_registry: Mapping[
        type[SupportsDataclass], tuple[SectionNode[SupportsDataclass], ...]
    ]
    defaults_by_path: Mapping[SectionPath, SupportsDataclass]
    defaults_by_type: Mapping[type[SupportsDataclass], SupportsDataclass]
    placeholders: Mapping[SectionPath, frozenset[str]]
    tool_name_registry: Mapping[str, SectionPath]

    def resolve_section_params(
        self,
        node: SectionNode[SupportsDataclass],
        param_lookup: MutableMapping[type[SupportsDataclass], SupportsDataclass],
    ) -> SupportsDataclass | None:
        """Return parameters for a section, applying defaults when necessary."""

        params_type = node.section.params_type
        if params_type is None:
            return None

        section_params = param_lookup.get(params_type)
        if section_params is None:
            section_params = self._default_or_construct_params(node, params_type)

        return self._ensure_dataclass_params(section_params, node, params_type)

    def _default_or_construct_params(
        self,
        node: SectionNode[SupportsDataclass],
        params_type: type[SupportsDataclass],
    ) -> SupportsDataclass | None:
        default_value = self.defaults_by_path.get(node.path)
        if default_value is not None:
            return clone_dataclass(default_value)

        type_default = self.defaults_by_type.get(params_type)
        if type_default is not None:
            return clone_dataclass(type_default)

        return self._construct_section_params(params_type, node)

    @staticmethod
    def _construct_section_params(
        params_type: type[SupportsDataclass],
        node: SectionNode[SupportsDataclass],
    ) -> SupportsDataclass | None:
        try:
            constructor = cast(Callable[[], SupportsDataclass | None], params_type)
            return constructor()
        except TypeError as error:
            raise PromptRenderError(
                "Missing parameters for section.",
                section_path=node.path,
                dataclass_type=params_type,
            ) from error

    @staticmethod
    def _ensure_dataclass_params(
        params: SupportsDataclass | None,
        node: SectionNode[SupportsDataclass],
        params_type: type[SupportsDataclass],
    ) -> SupportsDataclass:
        if params is None or not is_dataclass(params):
            raise PromptRenderError(
                "Section constructor must return a dataclass instance.",
                section_path=node.path,
                dataclass_type=params_type,
            )

        return params

    @property
    def params_types(self) -> set[type[SupportsDataclass]]:
        """Return the set of parameter dataclasses registered for sections."""

        return set(self.params_registry.keys())

    @property
    def section_paths(self) -> frozenset[SectionPath]:
        """Return the set of all registered section paths."""

        return frozenset(node.path for node in self.sections)


def _registry_paths_are_registered(
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


def _params_registry_is_consistent(
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


@invariant(
    _registry_paths_are_registered,
    _params_registry_is_consistent,
)
class PromptRegistry:
    """Collect and validate prompt sections prior to rendering."""

    def __init__(self) -> None:
        super().__init__()
        self._section_nodes: list[SectionNode[SupportsDataclass]] = []
        self._params_registry: dict[
            type[SupportsDataclass], list[SectionNode[SupportsDataclass]]
        ] = {}
        self._defaults_by_path: dict[SectionPath, SupportsDataclass] = {}
        self._defaults_by_type: dict[type[SupportsDataclass], SupportsDataclass] = {}
        self._placeholders: dict[SectionPath, set[str]] = {}
        self._tool_name_registry: dict[str, SectionPath] = {}
        self._numbering_stack: list[int] = []

    def register_sections(self, sections: Sequence[Section[SupportsDataclass]]) -> None:
        """Register the provided root sections."""

        for section in sections:
            self._register_section(section, path=(section.key,), depth=0)

    def register_section(
        self,
        section: Section[SupportsDataclass],
        *,
        path: SectionPath,
        depth: int,
    ) -> None:
        """Register a single section at the supplied path and depth."""

        self._register_section(section, path=path, depth=depth)

    def _register_section(
        self,
        section: Section[SupportsDataclass],
        *,
        path: SectionPath,
        depth: int,
    ) -> None:
        params_type = self._validate_section_params(section, path)
        node = self._register_section_node(section, path, depth)
        self._register_params_registry(params_type, node)
        self._register_section_defaults(section, path, params_type)
        self._register_placeholders(section, path, params_type)
        self._register_section_tools_if_present(section, path, params_type)
        self._register_child_sections(section, path, depth)

    @staticmethod
    def _validate_section_params(
        section: Section[SupportsDataclass],
        path: SectionPath,
    ) -> type[SupportsDataclass] | None:
        params_type = section.params_type
        if params_type is not None and not is_dataclass(params_type):
            raise PromptValidationError(
                "Section params must be a dataclass.",
                section_path=path,
                dataclass_type=params_type,
            )
        return params_type

    def _register_section_node(
        self,
        section: Section[SupportsDataclass],
        path: SectionPath,
        depth: int,
    ) -> SectionNode[SupportsDataclass]:
        number = self._next_section_number(depth)
        node: SectionNode[SupportsDataclass] = SectionNode(
            section=section, depth=depth, path=path, number=number
        )
        self._section_nodes.append(node)
        return node

    def _register_params_registry(
        self,
        params_type: type[SupportsDataclass] | None,
        node: SectionNode[SupportsDataclass],
    ) -> None:
        if params_type is not None:
            self._params_registry.setdefault(params_type, []).append(node)

    def _register_section_defaults(
        self,
        section: Section[SupportsDataclass],
        path: SectionPath,
        params_type: type[SupportsDataclass] | None,
    ) -> None:
        if params_type is None or section.default_params is None:
            return

        default_value = section.default_params
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
        self._defaults_by_path[path] = default_value
        _ = self._defaults_by_type.setdefault(params_type, default_value)

    def _register_placeholders(
        self,
        section: Section[SupportsDataclass],
        path: SectionPath,
        params_type: type[SupportsDataclass] | None,
    ) -> None:
        section_placeholders = section.placeholder_names()
        self._placeholders[path] = set(section_placeholders)
        if params_type is None:
            if section_placeholders:
                placeholder = sorted(section_placeholders)[0]
                raise PromptValidationError(
                    "Section does not accept parameters but declares placeholders.",
                    section_path=path,
                    placeholder=placeholder,
                )
            return

        param_fields = {field.name for field in fields(params_type)}
        unknown_placeholders = section_placeholders - param_fields
        if unknown_placeholders:
            placeholder = sorted(unknown_placeholders)[0]
            raise PromptValidationError(
                "Template references unknown placeholder.",
                section_path=path,
                dataclass_type=params_type,
                placeholder=placeholder,
            )

    def _register_section_tools_if_present(
        self,
        section: Section[SupportsDataclass],
        path: SectionPath,
        params_type: type[SupportsDataclass] | None,
    ) -> None:
        section_tools = cast(tuple[object, ...], section.tools())
        if not section_tools:
            return

        for tool in section_tools:
            if not isinstance(tool, Tool):
                raise PromptValidationError(
                    "Section tools must be Tool instances.",
                    section_path=path,
                    dataclass_type=params_type,
                )
            typed_tool = cast(Tool[SupportsDataclassOrNone, SupportsToolResult], tool)
            self._register_section_tools(
                typed_tool,
                path,
            )

    def _register_child_sections(
        self,
        section: Section[SupportsDataclass],
        path: SectionPath,
        depth: int,
    ) -> None:
        for child in section.children:
            child_path = (*path, child.key)
            self._register_section(child, path=child_path, depth=depth + 1)

    def _next_section_number(self, depth: int) -> str:
        while len(self._numbering_stack) > depth + 1:
            _ = self._numbering_stack.pop()

        if len(self._numbering_stack) <= depth:
            while len(self._numbering_stack) < depth + 1:
                self._numbering_stack.append(1)
        else:
            self._numbering_stack[-1] += 1

        return ".".join(str(value) for value in self._numbering_stack)

    def _register_section_tools[
        ParamsT: SupportsDataclassOrNone,
        ResultT: SupportsToolResult,
    ](
        self,
        tool: Tool[ParamsT, ResultT],
        path: SectionPath,
    ) -> None:
        params_type = tool.params_type
        if params_type is not type(None) and not is_dataclass(params_type):
            raise PromptValidationError(
                "Tool parameters must be dataclass types.",
                section_path=path,
                dataclass_type=params_type,
            )

        existing_path = self._tool_name_registry.get(tool.name)
        if existing_path is not None:
            raise PromptValidationError(
                "Duplicate tool name registered for prompt.",
                section_path=path,
                dataclass_type=tool.params_type,
            )

        self._tool_name_registry[tool.name] = path

    def snapshot(
        self,
        structured_output_type: type[SupportsDataclass] | None = None,
    ) -> RegistrySnapshot:
        """Return an immutable snapshot of the registered sections."""

        # Validate task examples after all sections are registered
        self._validate_task_examples(structured_output_type)

        params_registry: dict[
            type[SupportsDataclass], tuple[SectionNode[SupportsDataclass], ...]
        ] = {
            params_type: tuple(nodes)
            for params_type, nodes in self._params_registry.items()
        }
        defaults_by_path = MappingProxyType(dict(self._defaults_by_path))
        defaults_by_type = MappingProxyType(dict(self._defaults_by_type))
        placeholders = MappingProxyType(
            {path: frozenset(names) for path, names in self._placeholders.items()}
        )
        tool_name_registry = MappingProxyType(dict(self._tool_name_registry))

        return RegistrySnapshot(
            sections=tuple(self._section_nodes),
            params_registry=MappingProxyType(params_registry),
            defaults_by_path=defaults_by_path,
            defaults_by_type=defaults_by_type,
            placeholders=placeholders,
            tool_name_registry=tool_name_registry,
        )

    def _validate_task_examples(
        self,
        structured_output_type: type[SupportsDataclass] | None = None,
    ) -> None:
        """Validate task example tool references and type coherence."""
        # Import here to avoid circular imports
        from .task_examples import TaskExample

        # Build a map from tool name to Tool instance
        tool_instances: dict[
            str, Tool[SupportsDataclassOrNone, SupportsToolResult]
        ] = {}
        for node in self._section_nodes:
            for tool in node.section.tools():
                tool_instances[tool.name] = tool

        # Find all TaskExample sections and validate their steps
        for node in self._section_nodes:
            if not isinstance(node.section, TaskExample):
                continue

            task_example = cast(TaskExample[Any], node.section)
            self._validate_task_example_steps(
                task_example,
                node.path,
                tool_instances,
            )
            PromptRegistry._validate_task_example_outcome(
                task_example,
                node.path,
                structured_output_type,
            )

    @staticmethod
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
                f"Task example outcome type mismatch. "
                f"Expected: {expected}, got: {actual}."
            )
            raise PromptValidationError(
                msg,
                section_path=path,
                placeholder="outcome",
            )

    def _validate_task_example_steps(
        self,
        task_example: object,
        path: SectionPath,
        tool_instances: dict[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
    ) -> None:
        """Validate steps in a task example."""
        available_tools = sorted(self._tool_name_registry.keys())
        steps = getattr(task_example, "steps", ())

        for step_idx, step in enumerate(steps):
            tool_name = getattr(step, "tool_name", "")

            # Check tool name exists
            if tool_name not in self._tool_name_registry:
                available_str = (
                    ", ".join(available_tools) if available_tools else "none"
                )
                msg = (
                    f'Unknown tool "{tool_name}" in task example step {step_idx}. '
                    f"Available tools: {available_str}."
                )
                raise PromptValidationError(
                    msg,
                    section_path=path,
                    placeholder="steps",
                )

            # Validate type coherence
            tool = tool_instances.get(tool_name)
            if tool is not None:
                PromptRegistry._validate_step_type_coherence(
                    step,
                    step_idx,
                    tool,
                    path,
                )

    @staticmethod
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


def clone_dataclass(instance: SupportsDataclass) -> SupportsDataclass:
    """Return a shallow copy of the provided dataclass instance."""

    return cast(SupportsDataclass, replace(cast(Any, instance)))


__all__ = [
    "PromptRegistry",
    "RegistrySnapshot",
    "SectionNode",
    "clone_dataclass",
]
