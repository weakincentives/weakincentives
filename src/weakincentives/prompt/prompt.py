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

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import MISSING, field, is_dataclass
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    cast,
    get_args,
    get_origin,
)

from ..dataclasses import FrozenDataclass
from ..resources import ResourceRegistry
from ..resources.builder import RegistryBuilder
from ._normalization import normalize_component_key
from ._overrides_protocols import PromptOverridesStore
from ._prompt_resources import PromptResources
from ._types import SupportsDataclass
from .errors import PromptValidationError, SectionPath
from .feedback import FeedbackProviderConfig
from .overrides import PromptDescriptor
from .policy import ToolPolicy
from .registry import PromptRegistry, SectionNode
from .rendering import PromptRenderer, RenderedPrompt
from .section import Section
from .structured_output import StructuredOutputConfig
from .task_completion import TaskCompletionChecker

if TYPE_CHECKING:
    from ..filesystem import Filesystem
    from ..resources.context import ScopedResourceContext
    from ..runtime.session.protocols import SessionProtocol
    from .overrides import PromptLike, ToolOverride
    from .registry import RegistrySnapshot


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
class PromptTemplate[OutputT]:
    """Coordinate prompt sections and their parameter bindings.

    PromptTemplate is an immutable dataclass that coordinates prompt sections
    and their parameter bindings. All construction logic runs through
    ``__pre_init__`` to derive internal state before the instance is frozen.

    Copy helpers ``update()``, ``merge()``, and ``map()`` are available for
    producing modified copies when needed.

    Resources can be declared at the template level and will be combined with
    resources contributed by individual sections.
    """

    ns: str
    key: str
    name: str | None = None
    # Field accepts Section sequence as input and stores SectionNode tuple
    sections: (
        Sequence[Section[SupportsDataclass]]
        | tuple[SectionNode[SupportsDataclass], ...]
    ) = ()
    policies: Sequence[ToolPolicy] = ()
    feedback_providers: Sequence[FeedbackProviderConfig] = ()
    task_completion_checker: TaskCompletionChecker | None = None
    allow_extra_keys: bool = False
    resources: ResourceRegistry = field(default_factory=ResourceRegistry)
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
        return type(name, (cls,), namespace)

    @classmethod
    def __pre_init__(
        cls,
        *,
        ns: str | object = MISSING,
        key: str | object = MISSING,
        name: str | object | None = MISSING,
        sections: Sequence[Section[SupportsDataclass]]
        | tuple[SectionNode[SupportsDataclass], ...]
        | object
        | None = MISSING,
        policies: Sequence[ToolPolicy] | object = MISSING,
        feedback_providers: Sequence[FeedbackProviderConfig] | object = MISSING,
        task_completion_checker: TaskCompletionChecker | object | None = MISSING,
        allow_extra_keys: bool | object = MISSING,
        resources: ResourceRegistry | object = MISSING,
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
        policies_input: Sequence[ToolPolicy] = (
            cast(Sequence[ToolPolicy], policies) if policies is not MISSING else ()
        )
        feedback_providers_input: Sequence[FeedbackProviderConfig] = (
            cast(Sequence[FeedbackProviderConfig], feedback_providers)
            if feedback_providers is not MISSING
            else ()
        )
        allow_extra = (
            cast(bool, allow_extra_keys) if allow_extra_keys is not MISSING else False
        )
        resources_val = (
            cast(ResourceRegistry, resources)
            if resources is not MISSING
            else ResourceRegistry()
        )

        try:
            stripped_ns = normalize_component_key(ns_str, owner="Prompt namespace")
        except ValueError as exc:
            raise PromptValidationError(str(exc)) from exc
        try:
            stripped_key = normalize_component_key(key_str, owner="Prompt key")
        except ValueError as exc:
            raise PromptValidationError(str(exc)) from exc

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
            "sections": snapshot.sections,
            "policies": tuple(policies_input),
            "feedback_providers": tuple(feedback_providers_input),
            "task_completion_checker": (
                cast(TaskCompletionChecker | None, task_completion_checker)
                if task_completion_checker is not MISSING
                else None
            ),
            "allow_extra_keys": allow_extra,
            "resources": resources_val,
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


class Prompt[OutputT]:
    """Bind a prompt template with overrides and parameters for rendering.

    Prompt is the only way to render a PromptTemplate. It holds the runtime
    configuration (overrides store, tag, params) and performs all rendering
    and override resolution.

    Resource lifecycle is managed via ``prompt.resource_scope()``::

        prompt = Prompt(template).bind(params, resources={Filesystem: fs})

        with prompt.resource_scope():  # Resources initialized
            rendered = prompt.render()
            filesystem = prompt.resources.get(Filesystem)
        # Resources cleaned up
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
        self._bound_resources: ResourceRegistry | None = None
        self._resource_context: ScopedResourceContext | None = None
        self._cached_registry: ResourceRegistry | None = None

    @property
    def params(self) -> tuple[SupportsDataclass, ...]:
        """Return the parameters bound to this prompt instance."""

        return self._params

    @property
    def sections(self) -> tuple[SectionNode[SupportsDataclass], ...]:
        # sections is always SectionNode tuple after construction
        return cast(  # pragma: no cover
            tuple[SectionNode[SupportsDataclass], ...], self.template.sections
        )

    @property
    def descriptor(self) -> PromptDescriptor:
        return self.template.descriptor

    @property
    def structured_output(self) -> StructuredOutputConfig[SupportsDataclass] | None:
        return self.template.structured_output

    @property
    def feedback_providers(self) -> tuple[FeedbackProviderConfig, ...]:
        """Return feedback providers configured on this prompt."""
        return tuple(self.template.feedback_providers)

    @property
    def task_completion_checker(self) -> TaskCompletionChecker | None:
        """Return task completion checker configured on this prompt."""
        return self.template.task_completion_checker

    def policies_for_tool(self, tool_name: str) -> tuple[ToolPolicy, ...]:
        """Collect policies that apply to a tool from sections and template.

        Returns policies in order: section policies first, then prompt-level.
        """
        result: list[ToolPolicy] = []

        # Find section containing the tool and add its policies
        snapshot = self.template._snapshot  # pyright: ignore[reportPrivateUsage]
        if snapshot is not None:  # pragma: no branch - snapshot always initialized
            for node in snapshot.sections:
                section = node.section
                for tool in section.tools():
                    if tool.name == tool_name:
                        result.extend(section.policies())
                        break

        # Add prompt-level policies
        result.extend(self.template.policies)
        return tuple(result)

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

    def bind(
        self,
        *params: SupportsDataclass,
        resources: Mapping[type[object], object] | None = None,
    ) -> Prompt[OutputT]:
        """Mutate this prompt's bound parameters; return self for chaining.

        New dataclass instances replace any existing binding of the same
        dataclass type; otherwise they are appended. Passing multiple params
        of the same type in a single bind() call is not allowed - validation
        will raise an error during render().

        Args:
            *params: Dataclass instances to bind as section parameters.
            resources: Optional runtime resources to merge with template/section
                resources. Pass a dict mapping protocol types to instances.
                These take precedence over template-level resources.

        Example::

            prompt.bind(
                MyParams(value=42),
                resources={Filesystem: fs, BudgetTracker: tracker},
            )
        """
        # Handle params binding
        if params:
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

        # Handle resources binding
        if resources is not None:
            new_registry = ResourceRegistry.build(resources)
            if self._bound_resources is None:
                self._bound_resources = new_registry
            else:
                self._bound_resources = self._bound_resources.merge(
                    new_registry, strict=False
                )
            self._cached_registry = None  # Invalidate cache

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
        return self.template.find_section(selector)  # pragma: no cover

    def filesystem(self) -> Filesystem | None:
        """Return the filesystem from the workspace section, if present.

        Searches the template's section tree for a section implementing
        WorkspaceSectionProtocol and returns its filesystem property.

        Returns None if no workspace section exists in the template.
        """
        from .protocols import WorkspaceSectionProtocol

        snapshot = self.template._snapshot  # pyright: ignore[reportPrivateUsage]
        if snapshot is None:  # pragma: no cover
            return None

        for node in snapshot.sections:
            section = node.section
            if isinstance(section, WorkspaceSectionProtocol):
                return section.filesystem  # pragma: no cover
        return None

    def cleanup(self) -> None:
        """Clean up resources held by prompt sections.

        Calls cleanup() on each section in the prompt. The snapshot
        already contains all sections (including children) in a flat
        list, so no recursion is needed.

        Called by the AgentLoop after debug bundle artifacts have been
        captured.
        """
        snapshot = self.template._snapshot  # pyright: ignore[reportPrivateUsage]
        if snapshot is None:  # pragma: no cover
            return
        for node in snapshot.sections:
            node.section.cleanup()

    def _collected_resources(self) -> ResourceRegistry:
        """Collect resources from template, sections, and bind-time.

        Resources are collected in order (later overrides earlier):
        1. Template-level resources
        2. Section configure() contributions (depth-first)
        3. Bind-time resources

        Results are cached and invalidated when ``bind(resources=...)``
        is called.
        """
        if self._cached_registry is not None:
            return self._cached_registry

        builder = RegistryBuilder()
        self._configure_template_resources(builder)
        self._configure_section_resources(builder)
        result = builder.build()

        # Merge bind-time resources (these override everything)
        if self._bound_resources is not None:
            result = result.merge(self._bound_resources, strict=False)

        self._cached_registry = result
        return result

    def _configure_template_resources(self, builder: RegistryBuilder) -> None:
        """Add template-level resources to the builder."""
        for protocol in self.template.resources:
            b = self.template.resources.binding_for(protocol)
            if b is not None:  # pragma: no branch — always true for own keys
                builder._add(b)  # pyright: ignore[reportPrivateUsage]

    def _configure_section_resources(self, builder: RegistryBuilder) -> None:
        """Recursively configure section resources (depth-first)."""
        snapshot = self.template._snapshot  # pyright: ignore[reportPrivateUsage]
        if snapshot is None:  # pragma: no cover
            return
        for node in snapshot.sections:
            self._configure_section(node.section, builder)

    def _configure_section(
        self,
        section: Section[SupportsDataclass],
        builder: RegistryBuilder,
    ) -> None:
        """Configure a single section and its children."""
        section.configure(builder)
        for child in section.children:
            self._configure_section(child, builder)

    @contextmanager
    def resource_scope(self) -> Iterator[ScopedResourceContext]:
        """Enter the resource lifecycle scope.

        Creates and manages a ``ScopedResourceContext`` for this prompt's
        resources. Eager singletons are initialized on entry; all
        ``Closeable`` resources are disposed on exit.

        Reentrant: if a scope is already active, yields the existing
        context without creating a new one. Only the outermost scope
        performs cleanup on exit.

        Example::

            with prompt.resource_scope() as ctx:
                fs = ctx.get(Filesystem)
                result = adapter.evaluate(prompt, session=session)
            # Resources cleaned up

        Yields:
            Started ``ScopedResourceContext`` for resolving resources.
        """
        if self._resource_context is not None:
            # Reentrant: yield existing context, don't manage lifecycle
            yield self._resource_context
            return

        registry = self._collected_resources()
        ctx = registry._create_context()  # pyright: ignore[reportPrivateUsage]
        ctx.start()
        self._resource_context = ctx
        try:
            yield ctx
        finally:
            ctx.close()
            self._resource_context = None

    def _activate_scope(self) -> None:
        """Enter resource scope without context manager (for tests only).

        This is equivalent to the entry half of ``resource_scope()`` but
        without automatic cleanup. Use only in test setups where
        ``resource_scope()`` context manager is impractical.
        """
        if self._resource_context is not None:
            raise RuntimeError("Resource scope already entered")
        registry = self._collected_resources()
        ctx = registry._create_context()  # pyright: ignore[reportPrivateUsage]
        ctx.start()
        self._resource_context = ctx

    @property
    def resources(self) -> PromptResources:
        """Resource accessor for dependency resolution.

        Provides access to resources within an active ``resource_scope()``::

            with prompt.resource_scope():
                service = prompt.resources.get(MyService)

        Accessing resources outside a scope raises ``RuntimeError``.
        """
        return PromptResources(self)


__all__ = [
    "FeedbackProviderConfig",
    "Prompt",
    "PromptResources",
    "PromptTemplate",
    "RenderedPrompt",
    "SectionNode",
]
