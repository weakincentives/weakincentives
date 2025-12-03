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

from collections.abc import Sequence
from dataclasses import is_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

from ._overrides_protocols import PromptOverridesStore
from ._types import SupportsDataclass
from .errors import PromptValidationError, SectionPath
from .registry import PromptRegistry, RegistrySnapshot, SectionNode
from .rendering import PromptRenderer, RenderedPrompt
from .response_format import ResponseFormatParams, ResponseFormatSection
from .section import Section
from .structured_output import StructuredOutputConfig

if TYPE_CHECKING:
    from .overrides import PromptLike, ToolOverride

OutputT = TypeVar("OutputT", covariant=True)


def _format_specialization_argument(argument: object | None) -> str:
    if argument is None:
        return "?"
    if isinstance(argument, type):
        return argument.__name__
    return repr(argument)


class PromptTemplate(Generic[OutputT]):  # noqa: UP046
    """Coordinate prompt sections and their parameter bindings."""

    _output_container_spec: ClassVar[Literal["object", "array"] | None] = None
    _output_dataclass_candidate: ClassVar[Any] = None

    def __class_getitem__(cls, item: object) -> type[PromptTemplate[Any]]:
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
        base_sections = tuple(sections or ())
        self._base_sections: tuple[Section[SupportsDataclass], ...] = base_sections
        self._sections: tuple[Section[SupportsDataclass], ...] = base_sections
        self._registry = PromptRegistry()
        self.placeholders: dict[SectionPath, set[str]] = {}
        self._allow_extra_keys_requested = allow_extra_keys

        self._structured_output: StructuredOutputConfig[SupportsDataclass] | None
        self._structured_output = self._resolve_output_spec(allow_extra_keys)

        self.inject_output_instructions = inject_output_instructions

        self._registry.register_sections(self._sections)

        self._response_section: ResponseFormatSection | None = None
        if self._structured_output is not None:
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

        self._renderer: PromptRenderer[OutputT] = PromptRenderer(
            registry=snapshot,
            structured_output=self._structured_output,
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
        from .overrides import PromptDescriptor

        descriptor = PromptDescriptor.from_prompt(cast("PromptLike", self))

        if overrides_store is not None:
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
            descriptor=descriptor,
        )

    def bind(
        self,
        *params: SupportsDataclass,
        overrides_store: PromptOverridesStore | None = None,
        tag: str = "latest",
        inject_output_instructions: bool | None = None,
    ) -> Prompt[OutputT]:
        """Return a bound prompt wrapper with the provided parameters."""

        effective_instructions = (
            inject_output_instructions
            if inject_output_instructions is not None
            else self.inject_output_instructions
        )
        return Prompt(
            self,
            overrides_store=overrides_store,
            overrides_tag=tag,
            inject_output_instructions=effective_instructions,
            params=params,
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

    def find_section(
        self,
        selector: type[Section[SupportsDataclass]]
        | tuple[type[Section[SupportsDataclass]], ...],
    ) -> Section[SupportsDataclass]:
        """Return the first section matching ``selector``."""

        if isinstance(selector, tuple):
            if not selector:
                raise TypeError("find_section requires at least one section type.")
            candidates = selector
        else:
            candidates = (selector,)

        for node in self._registry_snapshot.sections:
            if any(isinstance(node.section, candidate) for candidate in candidates):
                return node.section

        raise KeyError(
            f"Section matching {candidates!r} not found in prompt {self.ns}:{self.key}."
        )

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


class Prompt(Generic[OutputT]):  # noqa: UP046
    """Wrap a prompt template with overrides and bound parameters."""

    def __init__(
        self,
        template: PromptTemplate[OutputT],
        *,
        overrides_store: PromptOverridesStore | None = None,
        overrides_tag: str = "latest",
        inject_output_instructions: bool | None = None,
        params: Sequence[SupportsDataclass] | None = None,
    ) -> None:
        super().__init__()
        self.template = template
        self.ns: str = template.ns
        self.key: str = template.key
        self.name: str | None = template.name
        self.overrides_store = overrides_store
        self.overrides_tag = overrides_tag
        self.inject_output_instructions = inject_output_instructions
        self._params: tuple[SupportsDataclass, ...] = tuple(params or ())

    @property
    def params(self) -> tuple[SupportsDataclass, ...]:
        """Return the parameters bound to this prompt instance."""

        return self._params

    @property
    def sections(self) -> tuple[SectionNode[SupportsDataclass], ...]:
        return self.template.sections

    @property
    def structured_output(self) -> StructuredOutputConfig[SupportsDataclass] | None:
        return self.template.structured_output

    def bind(self, *params: SupportsDataclass) -> Prompt[OutputT]:
        """Mutate this prompt's bound parameters; return self for chaining.

        New dataclass instances replace any existing binding of the same
        dataclass type; otherwise they are appended.
        """

        if not params:
            return self

        current = list(self._params)
        for candidate in params:
            replaced = False
            for idx, existing in enumerate(current):
                if type(existing) is type(candidate):
                    current[idx] = candidate
                    replaced = True
                    break
            if not replaced:
                current.append(candidate)

        self._params = tuple(current)
        return self

    def render(
        self, *, inject_output_instructions: bool | None = None
    ) -> RenderedPrompt[OutputT]:
        """Render the underlying template with stored overrides and params."""

        instructions_flag = (
            inject_output_instructions
            if inject_output_instructions is not None
            else self.inject_output_instructions
        )
        tag = self.overrides_tag if self.overrides_tag else "latest"
        return self.template.render(
            *self._params,
            overrides_store=self.overrides_store,
            tag=tag,
            inject_output_instructions=instructions_flag,
        )

    def find_section(
        self,
        selector: type[Section[SupportsDataclass]]
        | tuple[type[Section[SupportsDataclass]], ...],
    ) -> Section[SupportsDataclass]:
        return self.template.find_section(selector)


__all__ = ["Prompt", "PromptTemplate", "RenderedPrompt", "SectionNode"]
