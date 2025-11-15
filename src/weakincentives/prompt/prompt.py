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

from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, get_args, get_origin

from ._types import SupportsDataclass
from .chapter import Chapter, ChaptersExpansionPolicy
from .errors import PromptValidationError, SectionPath
from .registry import PromptRegistry, RegistrySnapshot, SectionNode, clone_dataclass
from .rendering import PromptRenderer, RenderedPrompt
from .response_format import ResponseFormatParams, ResponseFormatSection
from .section import Section

if TYPE_CHECKING:
    from .overrides import PromptLike, PromptOverridesStore, ToolOverride


def _format_specialization_argument(argument: object | None) -> str:
    if argument is None:
        return "?"
    if isinstance(argument, type):
        return argument.__name__
    return repr(argument)


class Prompt[OutputT]:
    """Coordinate prompt sections and their parameter bindings."""

    _output_container_spec: ClassVar[Literal["object", "array"] | None] = None
    _output_dataclass_candidate: ClassVar[Any] = None

    def __class_getitem__(cls, item: object) -> type[Prompt[Any]]:
        origin = get_origin(item)
        candidate = item
        container: Literal["object", "array"] | None = "object"

        if origin is list:
            args = get_args(item)
            candidate = args[0] if len(args) == 1 else None
            container = "array"
            label = f"list[{_format_specialization_argument(candidate)}]"
        else:
            container = "object"
            label = _format_specialization_argument(candidate)

        name = f"{cls.__name__}[{label}]"
        namespace = {
            "__module__": cls.__module__,
            "_output_container_spec": container if candidate is not None else None,
            "_output_dataclass_candidate": candidate,
        }
        return type(name, (cls,), namespace)

    def __init__(
        self,
        *,
        ns: str,
        key: str,
        name: str | None = None,
        sections: Sequence[Section[Any]] | None = None,
        chapters: Sequence[Chapter[Any]] | None = None,
        inject_output_instructions: bool = True,
        allow_extra_keys: bool = False,
    ) -> None:
        super().__init__()
        stripped_ns = ns.strip()
        if not stripped_ns:
            raise PromptValidationError("Prompt namespace must be a non-empty string.")
        stripped_key = key.strip()
        if not stripped_key:
            raise PromptValidationError("Prompt key must be a non-empty string.")
        self.ns = stripped_ns
        self.key = stripped_key
        self.name = name
        base_sections: list[Section[SupportsDataclass]] = [
            cast(Section[SupportsDataclass], section) for section in sections or ()
        ]
        self._base_sections: tuple[Section[SupportsDataclass], ...] = tuple(
            base_sections
        )
        self._sections: tuple[Section[SupportsDataclass], ...] = tuple(base_sections)
        self._registry = PromptRegistry()
        self.placeholders: dict[SectionPath, set[str]] = {}
        self._allow_extra_keys_requested = allow_extra_keys

        normalized_chapters: list[Chapter[SupportsDataclass]] = []
        seen_chapter_keys: set[str] = set()
        for chapter in chapters or ():
            if not isinstance(chapter, Chapter):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise PromptValidationError(
                    "Prompt chapters must be Chapter instances.",
                    section_path=(getattr(chapter, "key", "?"),),
                )
            normalized = cast(Chapter[SupportsDataclass], chapter)
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

        self._output_type: type[Any] | None
        self._output_container: Literal["object", "array"] | None
        self._allow_extra_keys: bool | None
        (
            self._output_type,
            self._output_container,
            self._allow_extra_keys,
        ) = self._resolve_output_spec(allow_extra_keys)

        self.inject_output_instructions = inject_output_instructions

        self._registry.register_sections(self._sections)

        self._response_section: ResponseFormatSection | None = None
        if self._output_type is not None and self._output_container is not None:
            response_params = self._build_response_format_params()
            response_section = ResponseFormatSection(
                params=response_params,
                enabled=lambda _params, prompt=self: prompt.inject_output_instructions,
            )
            self._response_section = response_section
            section_for_registry = cast(Section[SupportsDataclass], response_section)
            self._sections += (section_for_registry,)
            self._registry.register_section(
                section_for_registry, path=(response_section.key,), depth=0
            )

        snapshot = self._registry.snapshot()
        self._registry_snapshot: RegistrySnapshot = snapshot
        self.placeholders = {
            path: set(names) for path, names in snapshot.placeholders.items()
        }

        self._renderer = PromptRenderer(
            registry=snapshot,
            output_type=self._output_type,
            output_container=self._output_container,
            allow_extra_keys=self._allow_extra_keys,
            response_section=self._response_section,
        )

    def render(
        self,
        *params: SupportsDataclass,
        overrides_store: PromptOverridesStore | None = None,
        tag: str = "latest",
        inject_output_instructions: bool | None = None,
    ) -> RenderedPrompt[OutputT]:
        """Render the prompt and apply overrides when an overrides store is supplied."""

        overrides: dict[SectionPath, str] | None = None
        tool_overrides: dict[str, ToolOverride] | None = None

        if overrides_store is not None:
            from .overrides import PromptDescriptor

            descriptor = PromptDescriptor.from_prompt(cast("PromptLike", self))
            override = overrides_store.resolve(descriptor=descriptor, tag=tag)

            if override is not None:
                overrides = {
                    path: section_override.body
                    for path, section_override in override.sections.items()
                }
                tool_overrides = dict(override.tool_overrides)

        param_lookup = self._renderer.build_param_lookup(params)
        return self._renderer.render(
            param_lookup,
            overrides,
            tool_overrides,
            inject_output_instructions=inject_output_instructions,
        )

    @property
    def sections(self) -> tuple[SectionNode[SupportsDataclass], ...]:
        return self._registry_snapshot.sections

    @property
    def param_types(self) -> set[type[SupportsDataclass]]:
        return self._registry_snapshot.param_types

    @property
    def chapters(self) -> tuple[Chapter[SupportsDataclass], ...]:
        return self._chapters

    def expand_chapters(
        self,
        policy: ChaptersExpansionPolicy,
        *,
        chapter_params: Mapping[str, SupportsDataclass | None] | None = None,
    ) -> Prompt[OutputT]:
        """Return a prompt snapshot with chapters opened per the supplied policy."""

        if not self._chapters:
            return self

        if policy is ChaptersExpansionPolicy.ALL_INCLUDED:
            return self._expand_chapters_all_included(
                chapter_params=chapter_params or {}
            )

        raise NotImplementedError(
            f"Chapters expansion policy '{policy.value}' is not supported."
        )

    def _expand_chapters_all_included(
        self,
        *,
        chapter_params: Mapping[str, SupportsDataclass | None],
    ) -> Prompt[OutputT]:
        provided_lookup = dict(chapter_params)
        unknown_keys = set(provided_lookup) - set(self._chapter_key_registry.keys())
        if unknown_keys:
            unknown_key = sorted(unknown_keys)[0]
            raise PromptValidationError(
                "Chapter parameters reference unknown chapter key.",
                section_path=(unknown_key,),
            )

        open_sections: list[Section[SupportsDataclass]] = list(self._base_sections)

        for chapter in self._chapters:
            key_present = chapter.key in provided_lookup
            params = provided_lookup.get(chapter.key)
            if key_present:
                params = self._normalize_chapter_params(chapter, params)
            elif chapter.default_params is not None:
                params = clone_dataclass(chapter.default_params)

            if chapter.enabled is not None:
                if params is None and chapter.param_type is not None:
                    raise PromptValidationError(
                        "Chapter requires parameters for enabled predicate.",
                        section_path=(chapter.key,),
                        dataclass_type=chapter.param_type,
                    )
                try:
                    enabled = chapter.is_enabled(cast(Any, params))
                except Exception as error:
                    raise PromptValidationError(
                        "Chapter enabled predicate failed.",
                        section_path=(chapter.key,),
                        dataclass_type=chapter.param_type,
                    ) from error
                if not enabled:
                    continue

            open_sections.extend(chapter.sections)

        prompt_cls = type(self)

        return prompt_cls(
            ns=self.ns,
            key=self.key,
            name=self.name,
            sections=open_sections,
            chapters=(),
            inject_output_instructions=self.inject_output_instructions,
            allow_extra_keys=self._allow_extra_keys_requested,
        )

    def _normalize_chapter_params(
        self,
        chapter: Chapter[SupportsDataclass],
        params: SupportsDataclass | None,
    ) -> SupportsDataclass | None:
        params_type = chapter.param_type
        if params_type is None:
            if params is not None:
                raise PromptValidationError(
                    "Chapter does not accept parameters.",
                    section_path=(chapter.key,),
                )
            return None

        if not isinstance(params, params_type) or not is_dataclass(params):
            raise PromptValidationError(
                "Chapter parameters must be instances of the declared dataclass.",
                section_path=(chapter.key,),
                dataclass_type=params_type,
            )
        return params

    def _resolve_output_spec(
        self, allow_extra_keys: bool
    ) -> tuple[type[Any] | None, Literal["object", "array"] | None, bool | None]:
        candidate = getattr(type(self), "_output_dataclass_candidate", None)
        container = cast(
            Literal["object", "array"] | None,
            getattr(type(self), "_output_container_spec", None),
        )

        if candidate is None or container is None:
            return None, None, None

        if not isinstance(candidate, type):
            candidate_type = cast(type[Any], type(candidate))
            raise PromptValidationError(
                "Prompt output type must be a dataclass.",
                dataclass_type=candidate_type,
            )

        if not is_dataclass(candidate):
            bad_dataclass = cast(type[Any], candidate)
            raise PromptValidationError(
                "Prompt output type must be a dataclass.",
                dataclass_type=bad_dataclass,
            )

        dataclass_type = cast(type[Any], candidate)
        return dataclass_type, container, allow_extra_keys

    def _build_response_format_params(self) -> ResponseFormatParams:
        container = self._output_container
        if container is None:
            raise RuntimeError(
                "Output container missing during response format construction."
            )

        article: Literal["a", "an"] = (
            "an" if container.startswith(("a", "e", "i", "o", "u")) else "a"
        )
        extra_clause = ". Do not add extra keys." if not self._allow_extra_keys else "."
        return ResponseFormatParams(
            article=article,
            container=container,
            extra_clause=extra_clause,
        )


__all__ = ["Prompt", "RenderedPrompt", "SectionNode"]
