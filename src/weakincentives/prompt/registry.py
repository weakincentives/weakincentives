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
from ..types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)
from ._indices import build_registry_indices
from ._validators import (
    params_registry_is_consistent,
    registry_paths_are_registered,
    validate_task_examples,
)
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
    skill_name_registry: Mapping[str, SectionPath]
    # Precomputed indices for O(1) lookups
    node_by_path: Mapping[SectionPath, SectionNode[SupportsDataclass]]
    children_by_path: Mapping[SectionPath, tuple[str, ...]]
    subtree_has_tools: Mapping[SectionPath, bool]
    subtree_has_skills: Mapping[SectionPath, bool]

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


@invariant(
    registry_paths_are_registered,
    params_registry_is_consistent,
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
        self._skill_name_registry: dict[str, SectionPath] = {}
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
        self._register_section_skills_if_present(section, path)
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

    def _register_section_skills_if_present(
        self,
        section: Section[SupportsDataclass],
        path: SectionPath,
    ) -> None:
        from ..skills import resolve_skill_name

        section_skills = section.skills()
        if not section_skills:
            return

        # Skills are already validated as SkillMount by Section._normalize_skills
        for mount in section_skills:
            name = resolve_skill_name(mount)
            existing_path = self._skill_name_registry.get(name)
            if existing_path is not None:
                raise PromptValidationError(
                    f"Duplicate skill name '{name}' registered for prompt.",
                    section_path=path,
                )
            self._skill_name_registry[name] = path

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
        validate_task_examples(
            self._section_nodes,
            self._tool_name_registry,
            structured_output_type,
        )

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
        skill_name_registry = MappingProxyType(dict(self._skill_name_registry))

        # Build precomputed indices for O(1) lookups
        sections_tuple = tuple(self._section_nodes)
        node_by_path, children_by_path, subtree_has_tools, subtree_has_skills = (
            build_registry_indices(sections_tuple)
        )

        return RegistrySnapshot(
            sections=sections_tuple,
            params_registry=MappingProxyType(params_registry),
            defaults_by_path=defaults_by_path,
            defaults_by_type=defaults_by_type,
            placeholders=placeholders,
            tool_name_registry=tool_name_registry,
            skill_name_registry=skill_name_registry,
            node_by_path=node_by_path,
            children_by_path=children_by_path,
            subtree_has_tools=subtree_has_tools,
            subtree_has_skills=subtree_has_skills,
        )


def clone_dataclass(instance: SupportsDataclass) -> SupportsDataclass:
    """Return a shallow copy of the provided dataclass instance."""

    return cast(SupportsDataclass, replace(cast(Any, instance)))


__all__ = [
    "PromptRegistry",
    "RegistrySnapshot",
    "SectionNode",
    "clone_dataclass",
]
