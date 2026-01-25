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
    Literal,
    cast,
    get_args,
    get_origin,
)

from ..dataclasses import FrozenDataclass
from ..resources import ResourceRegistry
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
    """Immutable template defining prompt structure, sections, and output type.

    PromptTemplate coordinates prompt sections, parameter bindings, and output
    type declarations. It is an immutable dataclass where all construction logic
    runs through ``__pre_init__`` to derive internal state before freezing.

    Type Parameters:
        OutputT: The structured output type. Use ``PromptTemplate[MyDataclass]``
            for object output or ``PromptTemplate[list[MyDataclass]]`` for array
            output. The dataclass must be decorated with ``@dataclass``.

    Attributes:
        ns: Namespace identifier for the prompt (e.g., "myapp", "evals").
        key: Unique key within the namespace (e.g., "main", "qa-review").
        name: Optional human-readable display name.
        sections: Tuple of Section instances defining prompt content and tools.
        policies: Tool policies applied at the template level.
        feedback_providers: Feedback provider configurations.
        allow_extra_keys: Whether to allow extra keys in structured output.
        resources: ResourceRegistry for dependency injection.

    Example::

        @dataclass
        class ReviewOutput:
            rating: int
            comments: str

        template = PromptTemplate[ReviewOutput](
            ns="evals",
            key="code-review",
            sections=[
                SystemSection(),
                InstructionsSection(),
            ],
        )

        # For array output:
        batch_template = PromptTemplate[list[ReviewOutput]](
            ns="evals",
            key="batch-review",
            sections=[...],
        )
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
    allow_extra_keys: bool = False
    resources: ResourceRegistry = field(default_factory=ResourceRegistry)
    _snapshot: RegistrySnapshot | None = field(init=False, default=None)
    _structured_output: StructuredOutputConfig[SupportsDataclass] | None = field(
        init=False, default=None
    )

    _output_container_spec: ClassVar[Literal["object", "array"] | None] = None
    _output_dataclass_candidate: ClassVar[Any] = None

    def __class_getitem__(cls, item: object) -> type[PromptTemplate[Any]]:
        """Create a specialized PromptTemplate subclass for structured output.

        This method enables type specialization syntax for declaring structured
        output types. The specialization creates a new subclass with metadata
        used during rendering to enforce output schema.

        Args:
            item: A dataclass type for object output, or ``list[DataclassType]``
                for array output. The type must be a dataclass.

        Returns:
            A new PromptTemplate subclass configured for the specified output type.

        Example::

            # Object output - single dataclass instance
            PromptTemplate[ReviewResult]

            # Array output - list of dataclass instances
            PromptTemplate[list[ReviewResult]]

        Raises:
            PromptValidationError: During instantiation if the type is not a dataclass.
        """
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
        sections: Sequence[Section[SupportsDataclass]]
        | tuple[SectionNode[SupportsDataclass], ...]
        | object
        | None = MISSING,
        policies: Sequence[ToolPolicy] | object = MISSING,
        feedback_providers: Sequence[FeedbackProviderConfig] | object = MISSING,
        allow_extra_keys: bool | object = MISSING,
        resources: ResourceRegistry | object = MISSING,
    ) -> dict[str, Any]:
        """Normalize inputs and derive internal state before construction.

        This method is called by the ``@FrozenDataclass`` decorator before
        ``__init__``. It validates inputs, builds the section registry, and
        resolves structured output configuration.

        Args:
            ns: Namespace identifier (required, must be non-empty after stripping).
            key: Prompt key (required, must be non-empty after stripping).
            name: Optional human-readable display name.
            sections: Sequence of Section instances to include in the template.
            policies: Tool policies to apply at the template level.
            feedback_providers: Feedback provider configurations.
            allow_extra_keys: Whether structured output allows extra keys.
            resources: ResourceRegistry for dependency injection.

        Returns:
            Dict of normalized field values for dataclass construction.

        Raises:
            TypeError: If required arguments ``ns`` or ``key`` are missing.
            PromptValidationError: If ``ns`` or ``key`` are empty strings,
                or if structured output type is not a valid dataclass.
        """
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
            "sections": snapshot.sections,
            "policies": tuple(policies_input),
            "feedback_providers": tuple(feedback_providers_input),
            "allow_extra_keys": allow_extra,
            "resources": resources_val,
            "_snapshot": snapshot,
            "_structured_output": structured_output,
        }

    @property
    def params_types(self) -> set[type[SupportsDataclass]]:
        """Return the set of parameter dataclass types required by sections.

        Each section declares a ``_params_type`` that specifies which dataclass
        type it expects for template variable substitution. This property
        collects all unique types across all sections.

        Use this to understand what parameters must be bound before rendering::

            template = PromptTemplate[Output](ns="app", key="main", sections=[...])
            for param_type in template.params_types:
                print(f"Requires: {param_type.__name__}")

        Returns:
            Set of dataclass types that must be provided via ``Prompt.bind()``.

        Raises:
            RuntimeError: If called before template initialization completes.
        """
        snapshot = self._snapshot
        if snapshot is None:  # pragma: no cover
            raise RuntimeError("PromptTemplate._snapshot not initialized")
        return snapshot.params_types

    @cached_property
    def descriptor(self) -> PromptDescriptor:
        """Return the prompt descriptor for override resolution.

        The descriptor uniquely identifies this prompt template and is used
        by the overrides system to look up prompt-specific configuration.
        It captures the namespace, key, and structural fingerprint of the
        template.

        Computed lazily on first access and cached for subsequent calls.

        Returns:
            PromptDescriptor containing identifying information for this template.
        """
        return PromptDescriptor.from_prompt(cast("PromptLike", self))

    @property
    def placeholders(self) -> Mapping[SectionPath, frozenset[str]]:
        """Return template variable placeholders for each section.

        Placeholders are ``{variable_name}`` tokens in section bodies that
        are substituted with values from bound parameters during rendering.
        This mapping shows which placeholders exist in each section.

        Returns:
            Mapping from SectionPath to frozenset of placeholder names found
            in that section's body template.

        Raises:
            RuntimeError: If called before template initialization completes.
        """
        snapshot = self._snapshot
        if snapshot is None:  # pragma: no cover
            raise RuntimeError("PromptTemplate._snapshot not initialized")
        return snapshot.placeholders

    @property
    def structured_output(self) -> StructuredOutputConfig[SupportsDataclass] | None:
        """Return the structured output configuration, if declared.

        When the template is specialized with a type (e.g., ``PromptTemplate[MyOutput]``),
        this property returns the resolved configuration containing the dataclass type,
        container format (object or array), and validation settings.

        Returns:
            StructuredOutputConfig if a type was specified during specialization,
            None otherwise.

        Example::

            template = PromptTemplate[ReviewOutput](ns="app", key="review", ...)
            config = template.structured_output
            if config:
                print(f"Output type: {config.dataclass_type.__name__}")
                print(f"Container: {config.container}")  # "object" or "array"
        """
        return self._structured_output

    def find_section(
        self,
        selector: type[Section[SupportsDataclass]]
        | tuple[type[Section[SupportsDataclass]], ...],
    ) -> Section[SupportsDataclass]:
        """Find and return the first section matching the given type(s).

        Searches the template's section tree for a section that is an instance
        of the specified type(s). Useful for accessing section-specific
        configuration or properties.

        Args:
            selector: A Section subclass type, or a tuple of types to match.
                The first section matching any of the types is returned.

        Returns:
            The first Section instance matching the selector.

        Raises:
            TypeError: If selector is an empty tuple.
            KeyError: If no section matches the selector.

        Example::

            # Find by single type
            workspace = template.find_section(WorkspaceSection)

            # Find by multiple types (first match wins)
            section = template.find_section((SystemSection, InstructionsSection))
        """

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
    """Runtime prompt instance that binds parameters and renders templates.

    Prompt wraps a PromptTemplate with runtime configuration including parameter
    bindings, resource instances, and override settings. It is the only way to
    render a PromptTemplate into messages for LLM consumption.

    Type Parameters:
        OutputT: The structured output type, inherited from the template.

    Attributes:
        template: The underlying PromptTemplate being rendered.
        ns: Namespace identifier (copied from template).
        key: Prompt key (copied from template).
        name: Display name (copied from template).
        overrides_store: Optional store for prompt overrides.
        overrides_tag: Tag for selecting override version (default: "latest").

    Basic Usage::

        template = PromptTemplate[Output](ns="app", key="main", sections=[...])
        prompt = Prompt(template)
        prompt.bind(MyParams(value=42))
        rendered = prompt.render()

    With Resources::

        prompt = Prompt(template).bind(
            params,
            resources={Filesystem: fs, Logger: logger},
        )

        with prompt.resources:  # Initialize resources
            rendered = prompt.render()
            fs = prompt.resources.get(Filesystem)
        # Resources cleaned up automatically

    With Overrides::

        store = FileOverridesStore("/path/to/overrides")
        prompt = Prompt(template, overrides_store=store, overrides_tag="v2")
        rendered = prompt.render()  # Uses overrides from store
    """

    def __init__(
        self,
        template: PromptTemplate[OutputT],
        overrides_store: PromptOverridesStore | None = None,
        overrides_tag: str = "latest",
    ) -> None:
        """Create a Prompt instance from a template.

        Args:
            template: The PromptTemplate to render. Defines structure, sections,
                and output type.
            overrides_store: Optional store for loading prompt overrides. When
                provided, the store is queried during render() to apply section
                body and tool overrides.
            overrides_tag: Version tag for selecting overrides (default: "latest").
                Passed to the overrides store during resolution.
        """
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

    @property
    def params(self) -> tuple[SupportsDataclass, ...]:
        """Return all parameter instances bound to this prompt.

        Parameters are dataclass instances passed to ``bind()`` that provide
        values for template variable substitution in section bodies. Each
        section's ``_params_type`` determines which parameter instance it uses.

        Returns:
            Tuple of bound dataclass instances in binding order.
        """

        return self._params

    @property
    def sections(self) -> tuple[SectionNode[SupportsDataclass], ...]:
        """Return the registered section nodes from the template.

        Each SectionNode wraps a Section with additional registry metadata
        including the section's path, computed placeholders, and child nodes.

        Returns:
            Tuple of SectionNode instances representing the prompt structure.
        """
        # sections is always SectionNode tuple after construction
        return cast(tuple[SectionNode[SupportsDataclass], ...], self.template.sections)

    @property
    def descriptor(self) -> PromptDescriptor:
        """Return the prompt descriptor for override resolution.

        Delegates to the underlying template's descriptor. See
        ``PromptTemplate.descriptor`` for details.

        Returns:
            PromptDescriptor identifying this prompt for the overrides system.
        """
        return self.template.descriptor

    @property
    def structured_output(self) -> StructuredOutputConfig[SupportsDataclass] | None:
        """Return structured output configuration from the template.

        Delegates to the underlying template's structured_output. See
        ``PromptTemplate.structured_output`` for details.

        Returns:
            StructuredOutputConfig if declared, None otherwise.
        """
        return self.template.structured_output

    @property
    def feedback_providers(self) -> tuple[FeedbackProviderConfig, ...]:
        """Return feedback provider configurations from the template.

        Feedback providers generate additional context or guidance based on
        runtime state. They are invoked during prompt rendering to augment
        section content.

        Returns:
            Tuple of FeedbackProviderConfig instances from the template.
        """
        return tuple(self.template.feedback_providers)

    def policies_for_tool(self, tool_name: str) -> tuple[ToolPolicy, ...]:
        """Collect all tool policies that apply to a specific tool.

        Searches for the tool by name in the prompt's sections and collects
        applicable policies from both the containing section and the template.
        Policies are returned in precedence order: section-level first, then
        template-level.

        Args:
            tool_name: The name of the tool to find policies for.

        Returns:
            Tuple of ToolPolicy instances applicable to the tool, ordered by
            precedence (section policies before template policies).

        Example::

            policies = prompt.policies_for_tool("write_file")
            for policy in policies:
                if not policy.allows(tool_call):
                    raise PolicyViolation(policy.message)
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
        """Return the prompt renderer for this template.

        The renderer handles template variable substitution, section body
        assembly, and override application. It is created lazily on first
        access and cached for subsequent renders.

        Returns:
            PromptRenderer configured with the template's registry snapshot
            and structured output settings.

        Raises:
            RuntimeError: If the template was not properly initialized.
        """
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
        """Find and return the first section matching the given type(s).

        Convenience method that delegates to ``template.find_section()``.
        See ``PromptTemplate.find_section()`` for full documentation.

        Args:
            selector: A Section subclass type, or tuple of types to match.

        Returns:
            The first Section instance matching the selector.

        Raises:
            TypeError: If selector is an empty tuple.
            KeyError: If no section matches the selector.
        """
        return self.template.find_section(selector)

    def filesystem(self) -> Filesystem | None:
        """Return the filesystem from the workspace section, if present.

        Searches the template's section tree for a section implementing
        WorkspaceSection and returns its filesystem property.

        Returns None if no workspace section exists in the template.
        """
        from .protocols import WorkspaceSection

        snapshot = self.template._snapshot  # pyright: ignore[reportPrivateUsage]
        if snapshot is None:  # pragma: no cover
            return None

        for node in snapshot.sections:
            section = node.section
            if isinstance(section, WorkspaceSection):
                return section.filesystem
        return None

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

        snapshot = self.template._snapshot  # pyright: ignore[reportPrivateUsage]
        if snapshot is not None:  # pragma: no branch - tested separately
            for node in snapshot.sections:
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
