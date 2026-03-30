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
from dataclasses import field, is_dataclass
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

from ..dataclasses import Constructable, FrozenDataclass, allow_construction
from ..resources import ResourceRegistry
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
class PromptTemplate[OutputT](Constructable):
    """Immutable definition of a prompt's structure.

    PromptTemplate stores the declared sections, policies, and metadata.
    All derived state (registry snapshot, structured output config) is
    computed by :class:`Prompt` at bind time.

    Resources can be declared at the template level and will be combined with
    resources contributed by individual sections.
    """

    ns: str
    key: str
    name: str | None = None
    sections: tuple[Section[SupportsDataclass], ...] = ()
    policies: tuple[ToolPolicy, ...] = ()
    feedback_providers: tuple[FeedbackProviderConfig, ...] = ()
    task_completion_checker: TaskCompletionChecker | None = None
    allow_extra_keys: bool = False
    resources: ResourceRegistry = field(default_factory=ResourceRegistry)

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
    def create(
        cls,
        *,
        ns: str,
        key: str,
        name: str | None = None,
        sections: Sequence[Section[SupportsDataclass]] = (),
        policies: Sequence[ToolPolicy] = (),
        feedback_providers: Sequence[FeedbackProviderConfig] = (),
        task_completion_checker: TaskCompletionChecker | None = None,
        allow_extra_keys: bool = False,
        resources: ResourceRegistry | None = None,
    ) -> PromptTemplate[OutputT]:
        """Create a validated PromptTemplate instance."""
        try:
            stripped_ns = normalize_component_key(ns, owner="Prompt namespace")
        except ValueError as exc:
            raise PromptValidationError(str(exc)) from exc
        try:
            stripped_key = normalize_component_key(key, owner="Prompt key")
        except ValueError as exc:
            raise PromptValidationError(str(exc)) from exc

        with allow_construction():
            return cls(
                ns=stripped_ns,
                key=stripped_key,
                name=name,
                sections=tuple(sections),
                policies=tuple(policies),
                feedback_providers=tuple(feedback_providers),
                task_completion_checker=task_completion_checker,
                allow_extra_keys=allow_extra_keys,
                resources=resources if resources is not None else ResourceRegistry(),
            )


class Prompt[OutputT]:
    """Bind a prompt template with overrides and parameters for rendering.

    Prompt is the only way to render a PromptTemplate. It holds the runtime
    configuration (overrides store, tag, params) and computes derived state
    (registry snapshot, structured output) from the template.

    Resource lifecycle is managed via ``prompt.resources``::

        prompt = Prompt(template).bind(params, resources={Filesystem: fs})

        with prompt.resources:  # Resources initialized
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

    @cached_property
    def _snapshot(self) -> RegistrySnapshot:
        """Registry snapshot computed from template sections."""
        registry = PromptRegistry()
        registry.register_sections(self.template.sections)
        structured_output_type = (
            self.structured_output.dataclass_type
            if self.structured_output is not None
            else None
        )
        return registry.snapshot(structured_output_type=structured_output_type)

    @cached_property
    def structured_output(self) -> StructuredOutputConfig[SupportsDataclass] | None:
        """Resolved structured output config from template's type specialization."""
        return _resolve_output_spec(type(self.template), self.template.allow_extra_keys)

    @property
    def params(self) -> tuple[SupportsDataclass, ...]:
        """Return the parameters bound to this prompt instance."""
        return self._params

    @property
    def sections(self) -> tuple[SectionNode[SupportsDataclass], ...]:
        return self._snapshot.sections

    @property
    def params_types(self) -> set[type[SupportsDataclass]]:
        """Return the set of parameter types used by this prompt's sections."""
        return self._snapshot.params_types

    @property
    def placeholders(self) -> Mapping[SectionPath, frozenset[str]]:
        """Return the placeholders for each section path."""
        return self._snapshot.placeholders

    @cached_property
    def descriptor(self) -> PromptDescriptor:
        return PromptDescriptor.from_prompt(cast("PromptLike", self))

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

        for node in self._snapshot.sections:
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
        return PromptRenderer(
            registry=self._snapshot,
            structured_output=self.structured_output,
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
        """Return the first section matching ``selector``."""

        if isinstance(selector, tuple):
            if not selector:
                raise TypeError("find_section requires at least one section type.")
            candidates = selector
        else:
            candidates = (selector,)

        for node in self._snapshot.sections:
            if any(isinstance(node.section, candidate) for candidate in candidates):
                return node.section

        msg = (
            f"Section matching {candidates!r} not found in prompt "
            f"{self.template.ns}:{self.template.key}."
        )
        raise KeyError(msg)

    def filesystem(self) -> Filesystem | None:
        """Return the filesystem from the workspace section, if present.

        Searches the template's section tree for a section implementing
        WorkspaceSectionProtocol and returns its filesystem property.

        Returns None if no workspace section exists in the template.
        """
        from .protocols import WorkspaceSectionProtocol

        for node in self._snapshot.sections:
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
        for node in self._snapshot.sections:
            node.section.cleanup()

    def _collected_resources(self) -> ResourceRegistry:
        """Collect resources from template and all sections.

        Resources are collected in order:
        1. Template-level resources
        2. Section resources (depth-first)
        3. Bind-time resources

        Later sources override earlier on conflicts.
        """
        result = self.template.resources

        # Collect from sections
        def collect_from_section(section: Section[SupportsDataclass]) -> None:
            nonlocal result
            section_resources = section.resources()
            if len(section_resources) > 0:
                result = result.merge(section_resources, strict=False)
            for child in section.children:
                collect_from_section(child)

        for node in self._snapshot.sections:
            collect_from_section(node.section)

        # Merge bind-time resources
        if self._bound_resources is not None:
            result = result.merge(self._bound_resources, strict=False)

        return result

    @property
    def resources(self) -> PromptResources:
        """Resource accessor for lifecycle management and dependency resolution.

        Returns a dual-purpose object that:

        1. Acts as a context manager for resource lifecycle::

            with prompt.resources:
                service = prompt.resources.get(MyService)

        2. Provides direct access to resources within the context::

            with prompt.resources as ctx:
                service = ctx.get(MyService)

        Accessing resources outside the context raises RuntimeError.
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
