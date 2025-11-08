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

"""Section and chapter registration utilities for :mod:`weakincentives.prompt`."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from types import MappingProxyType
from typing import Any, cast

from ._types import SupportsDataclass
from .chapter import Chapter
from .errors import PromptRenderError, PromptValidationError, SectionPath
from .section import Section


def _clone_dataclass(instance: SupportsDataclass) -> SupportsDataclass:
    """Return a shallow copy of the provided dataclass instance."""

    return cast(SupportsDataclass, replace(cast(Any, instance)))


@dataclass(frozen=True, slots=True)
class SectionNode[ParamsT: SupportsDataclass]:
    """Flattened view of a section within a prompt."""

    section: Section[ParamsT]
    depth: int
    path: SectionPath


@dataclass(frozen=True)
class PromptRegistrySnapshot:
    """Immutable view of prompt registration state."""

    section_nodes: tuple[SectionNode[SupportsDataclass], ...]
    params_registry: Mapping[
        type[SupportsDataclass], tuple[SectionNode[SupportsDataclass], ...]
    ]
    defaults_by_path: Mapping[SectionPath, SupportsDataclass]
    defaults_by_type: Mapping[type[SupportsDataclass], SupportsDataclass]
    placeholders: Mapping[SectionPath, frozenset[str]]
    tool_name_registry: Mapping[str, SectionPath]
    base_sections: tuple[Section[SupportsDataclass], ...]
    chapters: tuple[Chapter[SupportsDataclass], ...]
    chapter_key_registry: Mapping[str, Chapter[SupportsDataclass]]
    response_section: Section[SupportsDataclass] | None

    def resolve_section_params(
        self,
        node: SectionNode[SupportsDataclass],
        param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass],
    ) -> SupportsDataclass | None:
        """Return the params object for ``node`` using ``param_lookup`` and defaults."""

        params_type = node.section.param_type
        if params_type is None:
            return None

        section_params: SupportsDataclass | None = param_lookup.get(params_type)

        if section_params is None:
            default_value = self.defaults_by_path.get(node.path)
            if default_value is not None:
                section_params = _clone_dataclass(default_value)
            else:
                type_default = self.defaults_by_type.get(params_type)
                if type_default is not None:
                    section_params = _clone_dataclass(type_default)
                else:
                    try:
                        constructor = cast(
                            Callable[[], SupportsDataclass | None],
                            params_type,
                        )
                        section_params = constructor()
                    except TypeError as error:  # pragma: no cover - defensive
                        raise PromptRenderError(
                            "Missing parameters for section.",
                            section_path=node.path,
                            dataclass_type=params_type,
                        ) from error

        result: SupportsDataclass | None = section_params
        if result is None or not is_dataclass(result):
            raise PromptRenderError(
                "Section constructor must return a dataclass instance.",
                section_path=node.path,
                dataclass_type=params_type,
            )

        return result


class PromptRegistry:
    """Manage prompt sections, chapters, and tool registration."""

    def __init__(
        self,
        *,
        base_sections: Sequence[Section[Any]] | None = None,
        response_section: Section[Any] | None = None,
        chapters: Sequence[object] | None = None,
    ) -> None:
        super().__init__()
        normalized_sections: list[Section[SupportsDataclass]] = [
            cast(Section[SupportsDataclass], section) for section in base_sections or ()
        ]
        self._base_sections: tuple[Section[SupportsDataclass], ...] = tuple(
            normalized_sections
        )
        self._section_nodes: list[SectionNode[SupportsDataclass]] = []
        self._params_registry: dict[
            type[SupportsDataclass], list[SectionNode[SupportsDataclass]]
        ] = {}
        self._defaults_by_path: dict[SectionPath, SupportsDataclass] = {}
        self._defaults_by_type: dict[type[SupportsDataclass], SupportsDataclass] = {}
        self._placeholders: dict[SectionPath, frozenset[str]] = {}
        self._tool_name_registry: dict[str, SectionPath] = {}

        for section in self._base_sections:
            self._register_section(section, path=(section.key,), depth=0)

        self._response_section: Section[SupportsDataclass] | None = None
        if response_section is not None:
            normalized_response = cast(Section[SupportsDataclass], response_section)
            self._response_section = normalized_response
            self._register_section(
                normalized_response,
                path=(normalized_response.key,),
                depth=0,
            )

        normalized_chapters: list[Chapter[SupportsDataclass]] = []
        seen_chapter_keys: set[str] = set()
        for raw_chapter in chapters or ():
            if not isinstance(raw_chapter, Chapter):
                raise PromptValidationError(
                    "Prompt chapters must be Chapter instances.",
                    section_path=(getattr(raw_chapter, "key", "?"),),
                )
            normalized = cast(Chapter[SupportsDataclass], raw_chapter)
            if normalized.key in seen_chapter_keys:
                raise PromptValidationError(
                    "Prompt chapters must use unique keys.",
                    section_path=(normalized.key,),
                )
            seen_chapter_keys.add(normalized.key)
            normalized_chapters.append(normalized)

        self._chapters: tuple[Chapter[SupportsDataclass], ...] = tuple(
            normalized_chapters
        )
        self._chapter_key_registry: dict[str, Chapter[SupportsDataclass]] = {
            chapter.key: chapter for chapter in self._chapters
        }

    @property
    def section_nodes(self) -> tuple[SectionNode[SupportsDataclass], ...]:
        return tuple(self._section_nodes)

    @property
    def param_types(self) -> set[type[SupportsDataclass]]:
        return set(self._params_registry.keys())

    @property
    def placeholders(self) -> Mapping[SectionPath, frozenset[str]]:
        return MappingProxyType(self._placeholders)

    @property
    def chapters(self) -> tuple[Chapter[SupportsDataclass], ...]:
        return self._chapters

    @property
    def chapter_key_registry(self) -> Mapping[str, Chapter[SupportsDataclass]]:
        return MappingProxyType(self._chapter_key_registry)

    @property
    def base_sections(self) -> tuple[Section[SupportsDataclass], ...]:
        return self._base_sections

    @property
    def response_section(self) -> Section[SupportsDataclass] | None:
        return self._response_section

    def snapshot(self) -> PromptRegistrySnapshot:
        """Return an immutable view of the current registration state."""

        params_registry: dict[
            type[SupportsDataclass], tuple[SectionNode[SupportsDataclass], ...]
        ] = {key: tuple(nodes) for key, nodes in self._params_registry.items()}
        return PromptRegistrySnapshot(
            section_nodes=tuple(self._section_nodes),
            params_registry=MappingProxyType(params_registry),
            defaults_by_path=MappingProxyType(dict(self._defaults_by_path)),
            defaults_by_type=MappingProxyType(dict(self._defaults_by_type)),
            placeholders=MappingProxyType(self._placeholders),
            tool_name_registry=MappingProxyType(dict(self._tool_name_registry)),
            base_sections=self._base_sections,
            chapters=self._chapters,
            chapter_key_registry=MappingProxyType(dict(self._chapter_key_registry)),
            response_section=self._response_section,
        )

    def resolve_section_params(
        self,
        node: SectionNode[SupportsDataclass],
        param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass],
    ) -> SupportsDataclass | None:
        """Delegate to the immutable snapshot helper for parameter resolution."""

        snapshot = self.snapshot()
        return snapshot.resolve_section_params(node, param_lookup)

    def _register_section(
        self,
        section: Section[SupportsDataclass],
        *,
        path: SectionPath,
        depth: int,
    ) -> None:
        params_type = section.param_type
        if params_type is not None and not is_dataclass(params_type):
            raise PromptValidationError(
                "Section params must be a dataclass.",
                section_path=path,
                dataclass_type=params_type,
            )

        node: SectionNode[SupportsDataclass] = SectionNode(
            section=section,
            depth=depth,
            path=path,
        )
        self._section_nodes.append(node)
        if params_type is not None:
            self._params_registry.setdefault(params_type, []).append(node)

        if params_type is not None and section.default_params is not None:
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

        section_placeholders = section.placeholder_names()
        if params_type is None:
            if section_placeholders:
                placeholder = sorted(section_placeholders)[0]
                raise PromptValidationError(
                    "Section does not accept parameters but declares placeholders.",
                    section_path=path,
                    placeholder=placeholder,
                )
            self._placeholders[path] = frozenset()
        else:
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
            self._placeholders[path] = frozenset(section_placeholders)

        self._register_section_tools(section, path)

        for child in section.children:
            child_path = (*path, child.key)
            self._register_section(child, path=child_path, depth=depth + 1)

    def _register_section_tools(
        self,
        section: Section[SupportsDataclass],
        path: SectionPath,
    ) -> None:
        section_tools = section.tools()
        if not section_tools:
            return

        for tool in section_tools:
            params_type = cast(
                type[SupportsDataclass] | None, getattr(tool, "params_type", None)
            )
            if not isinstance(params_type, type) or not is_dataclass(params_type):
                dataclass_type = (
                    params_type if isinstance(params_type, type) else section.param_type
                )
                raise PromptValidationError(
                    "Tool params_type must be a dataclass type.",
                    section_path=path,
                    dataclass_type=dataclass_type,
                )
            existing_path = self._tool_name_registry.get(tool.name)
            if existing_path is not None:
                raise PromptValidationError(
                    "Duplicate tool name registered for prompt.",
                    section_path=path,
                    dataclass_type=tool.params_type,
                )

            self._tool_name_registry[tool.name] = path


__all__ = [
    "PromptRegistry",
    "PromptRegistrySnapshot",
    "SectionNode",
]
