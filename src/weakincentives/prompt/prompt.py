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
from dataclasses import dataclass, field, is_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    cast,
    get_args,
    get_origin,
)

from ._normalization import normalize_component_key
from ._overrides_protocols import PromptOverridesStoreProtocol
from ._types import SupportsDataclass
from .chapter import Chapter, ChaptersExpansionPolicy
from .errors import PromptValidationError, SectionPath
from .registry import PromptRegistry, RegistrySnapshot, SectionNode, clone_dataclass
from .rendering import PromptRenderer, RenderedPrompt
from .response_format import ResponseFormatParams, ResponseFormatSection
from .section import Section
from .structured_output import StructuredOutputConfig

if TYPE_CHECKING:
    from .overrides import PromptLike, ToolOverride


def _format_specialization_argument(argument: object | None) -> str:
    if argument is None:
        return "?"
    if isinstance(argument, type):
        return argument.__name__
    return repr(argument)


@dataclass(frozen=True, slots=True, kw_only=True)
class Prompt[OutputT]:
    """Coordinate prompt sections and their parameter bindings."""

    ns: str
    key: str
    name: str | None = None
    _base_sections: tuple[Section[SupportsDataclass], ...] = field(init=False)
    _sections: tuple[Section[SupportsDataclass], ...] = field(init=False)
    _registry: PromptRegistry = field(init=False, repr=False)
    placeholders: dict[SectionPath, set[str]] = field(init=False, default_factory=dict)
    _allow_extra_keys_requested: bool = field(init=False, repr=False, default=False)
    _chapters: tuple[Chapter[SupportsDataclass], ...] = field(init=False, default_factory=tuple)
    _chapter_key_registry: dict[str, Chapter[SupportsDataclass]] = field(
        init=False, repr=False, default_factory=dict
    )
    _chapter_expansion_enabled: bool = field(init=False, repr=False, default=True)
    _structured_output: StructuredOutputConfig[SupportsDataclass] | None = field(
        init=False, repr=False, default=None
    )
    inject_output_instructions: bool = True
    _response_section: ResponseFormatSection | None = field(
        init=False, repr=False, default=None
    )
    _registry_snapshot: RegistrySnapshot = field(init=False, repr=False)
    _renderer: PromptRenderer[OutputT] = field(init=False, repr=False)

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
        sections: Sequence[Section[SupportsDataclass]] | None = None,
        chapters: Sequence[Chapter[SupportsDataclass]] | None = None,
        inject_output_instructions: bool = True,
        allow_extra_keys: bool = False,
        _chapter_expansion_enabled: bool | None = None,
    ) -> None:
        stripped_ns = ns.strip()
        if not stripped_ns:
            raise PromptValidationError("Prompt namespace must be a non-empty string.")
        stripped_key = key.strip()
        if not stripped_key:
            raise PromptValidationError("Prompt key must be a non-empty string.")
        normalized_key = normalize_component_key(stripped_key, owner="Prompt")
        object.__setattr__(self, "ns", stripped_ns)
        object.__setattr__(self, "key", normalized_key)
        object.__setattr__(self, "name", name)
        base_sections = tuple(sections or ())
        object.__setattr__(self, "_base_sections", base_sections)
        object.__setattr__(self, "_sections", base_sections)
        object.__setattr__(self, "_registry", PromptRegistry())
        object.__setattr__(self, "placeholders", {})
        object.__setattr__(self, "_allow_extra_keys_requested", allow_extra_keys)

        seen_chapter_keys: set[str] = set()
        provided_chapters = tuple(chapters or ())
        for chapter in provided_chapters:
            if chapter.key in seen_chapter_keys:
                raise PromptValidationError(
                    "Prompt chapters must use unique keys.",
                    section_path=(chapter.key,),
                )
            seen_chapter_keys.add(chapter.key)
        object.__setattr__(self, "_chapters", provided_chapters)
        object.__setattr__(
            self,
            "_chapter_key_registry",
            {chapter.key: chapter for chapter in provided_chapters},
        )
        expansion_enabled = (
            bool(provided_chapters)
            if _chapter_expansion_enabled is None
            else _chapter_expansion_enabled
        )
        object.__setattr__(self, "_chapter_expansion_enabled", expansion_enabled)

        structured_output = self._resolve_output_spec(allow_extra_keys)
        object.__setattr__(self, "_structured_output", structured_output)

        object.__setattr__(self, "inject_output_instructions", inject_output_instructions)

        registry = self._registry

        sections_to_register: list[Section[SupportsDataclass]] = list(base_sections)

        response_section: ResponseFormatSection | None = None
        if structured_output is not None:
            response_params = self._build_response_format_params()
            response_section = ResponseFormatSection(
                params=response_params,
                enabled=lambda _params, prompt=self: prompt.inject_output_instructions,
            )
            object.__setattr__(self, "_response_section", response_section)
            section_for_registry = cast(Section[SupportsDataclass], response_section)
            sections_to_register.append(section_for_registry)
        else:
            object.__setattr__(self, "_response_section", None)

        object.__setattr__(self, "_sections", tuple(sections_to_register))
        registry.register_sections(self._sections)

        snapshot = registry.snapshot()
        object.__setattr__(self, "_registry_snapshot", snapshot)
        object.__setattr__(
            self,
            "placeholders",
            {path: set(names) for path, names in snapshot.placeholders.items()},
        )

        object.__setattr__(
            self,
            "_renderer",
            PromptRenderer(
                registry=snapshot,
                structured_output=structured_output,
                response_section=response_section,
            ),
        )

    def render(
        self,
        *params: SupportsDataclass,
        overrides_store: PromptOverridesStoreProtocol | None = None,
        tag: str = "latest",
        inject_output_instructions: bool | None = None,
        chapters_expansion_policy: ChaptersExpansionPolicy | None = None,
        chapter_params: Mapping[str, SupportsDataclass | None] | None = None,
    ) -> RenderedPrompt[OutputT]:
        """Render the prompt and apply overrides when an overrides store is supplied."""

        prompt_for_render: Prompt[OutputT] = self
        if chapters_expansion_policy is not None and self._chapter_expansion_enabled:
            prompt_for_render = self.expand_chapters(
                chapters_expansion_policy, chapter_params=chapter_params
            )

        overrides: dict[SectionPath, str] | None = None
        tool_overrides: dict[str, ToolOverride] | None = None
        from .overrides import PromptDescriptor

        descriptor = PromptDescriptor.from_prompt(cast("PromptLike", prompt_for_render))

        if overrides_store is not None:
            override = overrides_store.resolve(descriptor=descriptor, tag=tag)

            if override is not None:
                overrides = {
                    path: section_override.body
                    for path, section_override in override.sections.items()
                }
                tool_overrides = dict(override.tool_overrides)

        param_lookup = prompt_for_render._renderer.build_param_lookup(params)
        return prompt_for_render._renderer.render(
            param_lookup,
            overrides,
            tool_overrides,
            inject_output_instructions=inject_output_instructions,
            descriptor=descriptor,
        )

    @property
    def sections(self) -> tuple[SectionNode[SupportsDataclass], ...]:
        return self._registry_snapshot.sections

    @property
    def param_types(self) -> set[type[SupportsDataclass]]:
        return self._registry_snapshot.param_types

    @property
    def structured_output(self) -> StructuredOutputConfig[SupportsDataclass] | None:
        """Resolved structured output declaration, when present."""

        return self._structured_output

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

        if not self._chapters or not self._chapter_expansion_enabled:
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
            should_open = True

            if chapter.enabled is not None:
                if params is None and chapter.param_type is not None:
                    raise PromptValidationError(
                        "Chapter requires parameters for enabled predicate.",
                        section_path=(chapter.key,),
                        dataclass_type=chapter.param_type,
                    )
                try:
                    enabled = chapter.is_enabled(params)
                except Exception as error:
                    raise PromptValidationError(
                        "Chapter enabled predicate failed.",
                        section_path=(chapter.key,),
                        dataclass_type=chapter.param_type,
                    ) from error
                if not enabled:
                    should_open = False

            if not should_open:
                continue

            open_sections.extend(chapter.sections)

        prompt_cls = type(self)

        expanded = prompt_cls(
            ns=self.ns,
            key=self.key,
            name=self.name,
            sections=open_sections,
            chapters=self._chapters,
            inject_output_instructions=self.inject_output_instructions,
            allow_extra_keys=self._allow_extra_keys_requested,
            _chapter_expansion_enabled=False,
        )
        return expanded

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

        if params is None:
            raise PromptValidationError(
                "Chapter requires parameters.",
                section_path=(chapter.key,),
                dataclass_type=params_type,
            )
        return params

    def _resolve_output_spec(
        self, allow_extra_keys: bool
    ) -> StructuredOutputConfig[SupportsDataclass] | None:
        candidate = getattr(type(self), "_output_dataclass_candidate", None)
        container = cast(
            Literal["object", "array"] | None,
            getattr(type(self), "_output_container_spec", None),
        )

        if candidate is None or container is None:
            return None

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

        dataclass_type = cast(type[SupportsDataclass], candidate)
        return StructuredOutputConfig(
            dataclass_type=dataclass_type,
            container=container,
            allow_extra_keys=allow_extra_keys,
        )

    def _build_response_format_params(self) -> ResponseFormatParams:
        spec = self._structured_output
        if spec is None:
            raise RuntimeError(
                "Output container missing during response format construction."
            )
        container = spec.container

        article: Literal["a", "an"] = (
            "an" if container.startswith(("a", "e", "i", "o", "u")) else "a"
        )
        extra_clause = "." if spec.allow_extra_keys else ". Do not add extra keys."
        return ResponseFormatParams(
            article=article,
            container=container,
            extra_clause=extra_clause,
        )


__all__ = ["Prompt", "RenderedPrompt", "SectionNode"]
