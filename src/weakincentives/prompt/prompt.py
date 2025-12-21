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
from dataclasses import MISSING, field, is_dataclass
from functools import cached_property
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

from ..dataclasses import FrozenDataclass
from ._overrides_protocols import PromptOverridesStore
from ._types import SupportsDataclass
from .errors import PromptValidationError, SectionPath
from .overrides import PromptDescriptor
from .registry import PromptRegistry, SectionNode
from .rendering import PromptRenderer, RenderedPrompt
from .section import Section
from .structured_output import StructuredOutputConfig

if TYPE_CHECKING:
    from ..contrib.tools.filesystem import Filesystem
    from ..runtime.session.protocols import SessionProtocol
    from .overrides import PromptLike, ToolOverride
    from .registry import RegistrySnapshot

OutputT = TypeVar("OutputT", covariant=True)


def _format_specialization_argument(argument: object | None) -> str:
    if argument is None:
        return "?"
    if isinstance(argument, type):
        return argument.__name__
    return repr(argument)


def _resolve_output_spec(
    cls: type[PromptTemplate[Any]], allow_extra_keys: bool
) -> StructuredOutputConfig[SupportsDataclass] | None:
    """Resolve structured output from class-level type specialization."""
    candidate = getattr(cls, "_output_dataclass_candidate", None)
    container = cast(
        Literal["object", "array"] | None,
        getattr(cls, "_output_container_spec", None),
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


@FrozenDataclass(slots=False)
class PromptTemplate(Generic[OutputT]):  # noqa: UP046
    """Coordinate prompt sections and their parameter bindings.

    PromptTemplate is an immutable dataclass that coordinates prompt sections
    and their parameter bindings. All construction logic runs through
    ``__pre_init__`` to derive internal state before the instance is frozen.

    Copy helpers ``update()``, ``merge()``, and ``map()`` are available for
    producing modified copies when needed.
    """

    ns: str
    key: str
    name: str | None = None
    sections: tuple[Section[SupportsDataclass], ...] = ()
    allow_extra_keys: bool = False
    _snapshot: RegistrySnapshot | None = field(init=False, default=None)
    _structured_output: StructuredOutputConfig[SupportsDataclass] | None = field(
        init=False, default=None
    )

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
        return type(name, (cls,), namespace)  # ty: ignore[invalid-return-type]  # dynamic type

    @classmethod
    def __pre_init__(
        cls,
        *,
        ns: str | object = MISSING,
        key: str | object = MISSING,
        name: str | object | None = MISSING,
        sections: Sequence[Section[SupportsDataclass]] | object | None = MISSING,
        allow_extra_keys: bool | object = MISSING,
    ) -> dict[str, Any]:
        """Normalize inputs and derive internal state before construction."""
        # Validate required inputs
        if ns is MISSING:
            raise TypeError("PromptTemplate() missing required argument: 'ns'")
        if key is MISSING:
            raise TypeError("PromptTemplate() missing required argument: 'key'")

        ns_str = cast(str, ns)
        key_str = cast(str, key)
        name_val = cast(str | None, name) if name is not MISSING else None
        sections_input = (
            cast(Sequence[Section[SupportsDataclass]] | None, sections)
            if sections is not MISSING
            else None
        )
        allow_extra = (
            cast(bool, allow_extra_keys) if allow_extra_keys is not MISSING else False
        )

        stripped_ns = ns_str.strip()
        if not stripped_ns:
            raise PromptValidationError("Prompt namespace must be a non-empty string.")
        stripped_key = key_str.strip()
        if not stripped_key:
            raise PromptValidationError("Prompt key must be a non-empty string.")

        sections_tuple = tuple(sections_input or ())
        registry = PromptRegistry()
        registry.register_sections(sections_tuple)

        structured_output = _resolve_output_spec(cls, allow_extra)
        structured_output_type = (
            structured_output.dataclass_type if structured_output is not None else None
        )
        snapshot = registry.snapshot(structured_output_type=structured_output_type)

        return {
            "ns": stripped_ns,
            "key": stripped_key,
            "name": name_val,
            "sections": sections_tuple,
            "allow_extra_keys": allow_extra,
            "_snapshot": snapshot,
            "_structured_output": structured_output,
        }

    @property
    def params_types(self) -> set[type[SupportsDataclass]]:
        """Return the set of parameter types used by this prompt's sections."""
        snapshot = self._snapshot
        if snapshot is None:  # pragma: no cover
            raise RuntimeError("PromptTemplate._snapshot not initialized")
        return snapshot.params_types

    @property
    def nodes(self) -> tuple[SectionNode[SupportsDataclass], ...]:
        """Return the flattened section nodes with depth, path, and numbering metadata."""
        snapshot = self._snapshot
        if snapshot is None:  # pragma: no cover
            raise RuntimeError("PromptTemplate._snapshot not initialized")
        return snapshot.sections

    @cached_property
    def descriptor(self) -> PromptDescriptor:
        """Return the prompt descriptor for this template, computed lazily."""
        return PromptDescriptor.from_prompt(cast("PromptLike", self))

    @property
    def placeholders(self) -> Mapping[SectionPath, frozenset[str]]:
        """Return the placeholders for each section path."""
        snapshot = self._snapshot
        if snapshot is None:  # pragma: no cover
            raise RuntimeError("PromptTemplate._snapshot not initialized")
        return snapshot.placeholders

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

        snapshot = self._snapshot
        if snapshot is None:  # pragma: no cover
            raise RuntimeError("PromptTemplate._snapshot not initialized")
        for node in snapshot.sections:
            if any(isinstance(node.section, candidate) for candidate in candidates):
                return node.section

        raise KeyError(
            f"Section matching {candidates!r} not found in prompt {self.ns}:{self.key}."
        )


class Prompt(Generic[OutputT]):  # noqa: UP046
    """Bind a prompt template with overrides and parameters for rendering.

    Prompt is the only way to render a PromptTemplate. It holds the runtime
    configuration (overrides store, tag, params) and performs all rendering
    and override resolution.
    """

    def __init__(
        self,
        template: PromptTemplate[OutputT],
        overrides_store: PromptOverridesStore | None = None,
        overrides_tag: str = "latest",
    ) -> None:
        super().__init__()
        self.template = template
        self.ns: str = template.ns
        self.key: str = template.key
        self.name: str | None = template.name
        self.overrides_store = overrides_store
        self.overrides_tag = overrides_tag
        self._params: tuple[SupportsDataclass, ...] = ()

    @property
    def params(self) -> tuple[SupportsDataclass, ...]:
        """Return the parameters bound to this prompt instance."""

        return self._params

    @property
    def sections(self) -> tuple[Section[SupportsDataclass], ...]:
        """Return the sections registered in this prompt's template."""
        return self.template.sections

    @property
    def nodes(self) -> tuple[SectionNode[SupportsDataclass], ...]:
        """Return the flattened section nodes with depth, path, and numbering metadata."""
        return self.template.nodes

    @property
    def descriptor(self) -> PromptDescriptor:
        return self.template.descriptor

    @property
    def structured_output(self) -> StructuredOutputConfig[SupportsDataclass] | None:
        return self.template.structured_output

    @cached_property
    def renderer(self) -> PromptRenderer[OutputT]:
        """Return the prompt renderer, created lazily on first access."""
        snapshot = self.template._snapshot  # pyright: ignore[reportPrivateUsage]
        if snapshot is None:  # pragma: no cover
            raise RuntimeError("PromptTemplate._snapshot not initialized")
        return PromptRenderer(
            registry=snapshot,
            structured_output=self.template._structured_output,  # pyright: ignore[reportPrivateUsage]
        )

    def bind(self, *params: SupportsDataclass) -> Prompt[OutputT]:
        """Mutate this prompt's bound parameters; return self for chaining.

        New dataclass instances replace any existing binding of the same
        dataclass type; otherwise they are appended. Passing multiple params
        of the same type in a single bind() call is not allowed - validation
        will raise an error during render().
        """

        if not params:
            return self

        # All new params are appended, but if there's already a param of the
        # same type in existing params, we replace it. Duplicates within the
        # same bind() call are passed through for validation during render().
        current = list(self._params)
        for candidate in params:
            replaced = False
            for idx, existing in enumerate(current):
                # Only replace if the existing param is from a previous bind,
                # not from the current params list. Check by comparing with
                # original self._params length.
                if type(existing) is type(candidate) and idx < len(self._params):
                    current[idx] = candidate
                    replaced = True
                    break
            if not replaced:
                current.append(candidate)

        self._params = tuple(current)
        return self

    def render(
        self,
        *,
        session: SessionProtocol | None = None,
    ) -> RenderedPrompt[OutputT]:
        """Render the prompt with bound parameters and optional overrides.

        Override resolution and rendering are performed here. The template
        provides only metadata and the registry snapshot for rendering.

        Visibility overrides are managed exclusively via Session state using
        the VisibilityOverrides state slice. Use session[VisibilityOverrides]
        to set visibility overrides before rendering.

        Args:
            session: Optional session for visibility callables that inspect state.
                When provided, the session is passed to enabled predicates and
                visibility selectors that accept a `session` keyword argument.
                The session is also used to query VisibilityOverrides for
                section-specific visibility control.
        """
        tag = self.overrides_tag if self.overrides_tag else "latest"

        overrides: dict[SectionPath, str] | None = None
        tool_overrides: dict[str, ToolOverride] | None = None
        descriptor = self.descriptor

        if self.overrides_store is not None:
            override = self.overrides_store.resolve(descriptor=descriptor, tag=tag)

            if override is not None:
                overrides = {
                    path: section_override.body
                    for path, section_override in override.sections.items()
                }
                tool_overrides = dict(override.tool_overrides)

        renderer = self.renderer
        param_lookup = renderer.build_param_lookup(self._params)
        return renderer.render(
            param_lookup,
            overrides,
            tool_overrides,
            descriptor=descriptor,
            session=session,
        )

    def find_section(
        self,
        selector: type[Section[SupportsDataclass]]
        | tuple[type[Section[SupportsDataclass]], ...],
    ) -> Section[SupportsDataclass]:
        return self.template.find_section(selector)

    def filesystem(self) -> Filesystem | None:
        """Return the filesystem from the workspace section, if present.

        Searches the template's section tree for a section implementing
        WorkspaceSection and returns its filesystem property.

        Returns None if no workspace section exists in the template.
        """
        # Import here to avoid circular imports; the protocol is runtime_checkable
        from ..contrib.tools.workspace import WorkspaceSection

        snapshot = self.template._snapshot  # pyright: ignore[reportPrivateUsage]
        if snapshot is None:  # pragma: no cover
            return None

        for node in snapshot.sections:
            section = node.section
            if isinstance(section, WorkspaceSection):
                return section.filesystem
        return None


__all__ = ["Prompt", "PromptTemplate", "RenderedPrompt", "SectionNode"]
