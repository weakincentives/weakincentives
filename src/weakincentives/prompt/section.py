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

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Self, TypeVar, cast

if TYPE_CHECKING:
    from ..resources import ResourceRegistry
    from ..runtime.session.protocols import SessionProtocol
    from .policy import ToolPolicy
    from .tool import Tool

from ..types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)
from ._enabled_predicate import (
    EnabledPredicate,
    NormalizedEnabledPredicate,
    normalize_enabled_predicate,
)
from ._generic_params_specializer import GenericParamsSpecializer
from ._normalization import normalize_component_key
from ._render_tool_examples import render_tool_examples_block
from ._visibility import (
    NormalizedVisibilitySelector,
    SectionVisibility,
    VisibilitySelector,
    normalize_visibility_selector,
)
from .errors import PromptRenderError, PromptValidationError, SectionPath

SectionParamsT = TypeVar("SectionParamsT", bound=SupportsDataclass, covariant=True)


class Section(GenericParamsSpecializer[SectionParamsT], ABC):
    """Abstract base class for composable prompt sections.

    Sections are the fundamental building blocks for constructing prompts.
    Each section represents a discrete piece of content with:

    - A unique ``key`` for identification and path-based access
    - A ``title`` displayed as the section heading
    - Optional typed ``params`` for template rendering
    - Optional child sections for hierarchical composition
    - Optional tools and policies scoped to the section
    - Visibility controls for conditional rendering

    Subclass this to create custom section types. Common subclasses include
    :class:`MarkdownSection` for template-based content and :class:`DynamicSection`
    for programmatically generated content.

    Example::

        class MySection(Section[MyParams]):
            def render_body(self, params, *, visibility=None, path=(), session=None):
                return f"Hello, {params.name}!"

            def clone(self, **kwargs):
                return MySection(title=self.title, key=self.key, **kwargs)

    Type Parameters:
        SectionParamsT: The dataclass type for section parameters. Use ``None``
            for sections that don't require parameters.

    Attributes:
        key: Unique identifier for this section (validated pattern: ``^[a-z0-9][a-z0-9._-]{0,63}$``).
        title: Display title used in rendered headings.
        children: Tuple of nested child sections.
        default_params: Default parameter values when none provided.
        accepts_overrides: Whether body overrides are allowed for this section.
        summary: Optional summary template for SUMMARY visibility mode.
        visibility: Default visibility selector (constant or callable).
        params_type: The resolved parameter type, or None for parameterless sections.
    """

    _generic_owner_name: ClassVar[str | None] = "Section"

    key: str
    children: tuple[Section[SupportsDataclass], ...]

    def __init__(
        self,
        *,
        title: str,
        key: str,
        default_params: SectionParamsT | None = None,
        children: Sequence[Section[SupportsDataclass]] | None = None,
        enabled: EnabledPredicate | None = None,
        tools: Sequence[object] | None = None,
        policies: Sequence[ToolPolicy] | None = None,
        accepts_overrides: bool = True,
        summary: str | None = None,
        visibility: VisibilitySelector = SectionVisibility.FULL,
    ) -> None:
        """Initialize a section with the given configuration.

        Args:
            title: Display title for the section heading. Rendered as a markdown
                heading with depth-based level (e.g., ``## 1. Title``).
            key: Unique identifier for path-based access and overrides. Must match
                pattern ``^[a-z0-9][a-z0-9._-]{0,63}$``.
            default_params: Default parameter values used when rendering without
                explicit params. Must be None for parameterless sections.
            children: Nested child sections. Children inherit numbering context
                and contribute their tools/resources to the parent prompt.
            enabled: Predicate controlling whether the section renders. Can be:
                - ``None``: Always enabled (default)
                - ``bool``: Static enabled/disabled
                - ``Callable[[params], bool]``: Dynamic based on params
                - ``Callable[[params, session], bool]``: Dynamic with session access
            tools: Tools exposed by this section. Must be :class:`Tool` instances.
                Tools are collected by the prompt and made available to the model.
            policies: Tool policies for this section. Policies control tool behavior
                and constraints.
            accepts_overrides: If True (default), allows body content to be replaced
                via override mechanism. Set to False for sections with fixed content.
            summary: Optional summary template rendered when visibility is SUMMARY.
                Supports the same placeholder syntax as the body template.
            visibility: Controls section rendering mode. Can be:
                - :attr:`SectionVisibility.FULL`: Render complete content (default)
                - :attr:`SectionVisibility.SUMMARY`: Render summary only
                - :attr:`SectionVisibility.HIDDEN`: Skip rendering entirely
                - ``Callable``: Dynamic visibility based on params/session

        Raises:
            TypeError: If ``default_params`` is provided for a parameterless section,
                or if ``children`` contains non-Section instances, or if ``tools``
                contains non-Tool instances.
        """
        super().__init__()
        params_candidate = getattr(self.__class__, "_params_type", None)
        candidate_type = (
            params_candidate if isinstance(params_candidate, type) else None
        )
        params_type = cast(type[SupportsDataclass] | None, candidate_type)

        self.params_type: type[SectionParamsT] | None = cast(
            type[SectionParamsT] | None, params_type
        )
        self.title = title
        self.key = self._normalize_key(key)
        self.default_params = default_params
        self.accepts_overrides = accepts_overrides
        self.summary = summary
        self.visibility = visibility

        if self.params_type is None and self.default_params is not None:
            raise TypeError("Section without parameters cannot define default_params.")

        normalized_children: list[Section[SupportsDataclass]] = []
        raw_children: Sequence[object] = cast(Sequence[object], children or ())
        for child in raw_children:
            if not isinstance(child, Section):
                raise TypeError("Section children must be Section instances.")
            normalized_children.append(cast(Section[SupportsDataclass], child))
        self.children = tuple(normalized_children)
        self._enabled: NormalizedEnabledPredicate | None = normalize_enabled_predicate(
            enabled, params_type
        )
        self._tools = self._normalize_tools(tools)
        self._policies = self._normalize_policies(policies)
        self._visibility: NormalizedVisibilitySelector = normalize_visibility_selector(
            visibility, params_type
        )

    def is_enabled(
        self,
        params: SupportsDataclass | None,
        *,
        session: SessionProtocol | None = None,
    ) -> bool:
        """Determine whether this section should be rendered.

        Evaluates the section's ``enabled`` predicate (if any) to determine
        whether the section should be included in the rendered output. Disabled
        sections are completely skipped during rendering, including their
        children, tools, and resources.

        Args:
            params: The section parameters to pass to the enabled predicate.
                May be None for parameterless sections.
            session: Optional session for enabled predicates that need to
                inspect session state (e.g., checking slice values).

        Returns:
            True if the section should render, False if it should be skipped.
            Always returns True if no enabled predicate was configured.
        """
        if self._enabled is None:
            return True
        return bool(self._enabled(params, session))

    def format_heading(
        self,
        depth: int,
        number: str,
        path: tuple[str, ...] = (),
    ) -> str:
        """Format the section heading with depth, number, and path annotation.

        This helper builds consistent markdown headings for all sections:
        - Root sections (depth 0) get ``##``
        - Each depth level adds one ``#`` (depth 1 = ``###``, depth 2 = ``####``)
        - Path annotation is appended in parentheses when provided

        Args:
            depth: The nesting depth of this section (affects heading level).
            number: The section number prefix (e.g., "1.2.").
            path: The section path as a tuple of keys for annotation.

        Returns:
            Formatted heading string (e.g., ``## 1. Title`` or ``### 1.1. Child (parent.child)``).
        """
        heading_level = "#" * (depth + 2)
        normalized_number = number.rstrip(".")
        path_str = ".".join(path) if path else ""
        title_with_path = (
            f"{self.title.strip()} ({path_str})" if path_str else self.title.strip()
        )
        return f"{heading_level} {normalized_number}. {title_with_path}"

    def render_body(
        self,
        params: SupportsDataclass | None,
        *,
        visibility: SectionVisibility | None = None,
        path: tuple[str, ...] = (),
        session: SessionProtocol | None = None,
    ) -> str:
        """Produce the body content (without heading) for the section.

        Subclasses should override this method to provide their body content.
        The base implementation returns an empty string.

        Args:
            params: The parameters to use when rendering the section body.
            visibility: The effective visibility for rendering.
            path: The section path as a tuple of keys.
            session: Optional session for visibility callables or state access.

        Returns:
            The rendered body content as a string.
        """
        del params, visibility, path, session
        return ""

    def render_tool_examples(self) -> str:
        """Render tool usage examples for tools declared on this section.

        Collects examples from all tools attached to this section and formats
        them into a markdown block. Examples help the model understand correct
        tool usage patterns.

        Returns:
            Formatted markdown block containing tool examples, or empty string
            if no tools are attached or none have examples defined.
        """
        return render_tool_examples_block(self._tools)

    def render(
        self,
        params: SupportsDataclass | None,
        depth: int,
        number: str,
        *,
        path: tuple[str, ...] = (),
        visibility: SectionVisibility | None = None,
    ) -> str:
        """Produce markdown output for the section at the supplied depth.

        The default implementation combines:
        1. Formatted heading via :meth:`format_heading`
        2. Body content via :meth:`render_body`
        3. Tool examples via :meth:`render_tool_examples`

        Subclasses can override :meth:`render_body` for custom body rendering
        while inheriting consistent heading and tool formatting, or override
        this method entirely for complete control.

        Args:
            params: The parameters to use when rendering the section template.
            depth: The nesting depth of this section (affects heading level).
            number: The section number prefix (e.g., "1.2.").
            path: The section path as a tuple of keys (e.g., ("parent", "child")).
            visibility: The effective visibility to use for rendering. When
                called from PromptRenderer, this is the already-computed
                effective visibility (incorporating session state, overrides,
                and user-provided selectors). When called directly, this may
                be None, in which case the section should compute effective
                visibility using its default selector.

        Returns:
            The complete rendered markdown output.
        """
        heading = self.format_heading(depth, number, path)
        body = self.render_body(params, visibility=visibility, path=path)
        rendered_tools = self.render_tool_examples()

        combined_body = body
        if rendered_tools:
            combined_body = f"{body}\n\n{rendered_tools}" if body else rendered_tools

        if combined_body:
            return f"{heading}\n\n{combined_body}"
        return heading

    def render_override(
        self,
        override_body: str,
        params: SupportsDataclass | None,
        depth: int,
        number: str,
        path: tuple[str, ...] = (),
    ) -> str:
        """Render the section using an override body instead of the default template.

        Called by :class:`PromptRenderer` when an override body is provided for
        this section. The default implementation raises an error; subclasses that
        support template-based overrides (like :class:`MarkdownSection`) should
        override this method.

        Args:
            override_body: The override template text to render instead of the
                default body.
            params: The parameters to use when rendering the override template.
            depth: The nesting depth of this section (affects heading level).
            number: The section number prefix (e.g., "1.2.").
            path: The section path as a tuple of keys.

        Returns:
            The rendered markdown content.

        Raises:
            PromptRenderError: When the section does not support override rendering.
        """
        msg = (
            f"Section '{self.key}' does not support override rendering. "
            "Override render_override() in your section subclass to enable this."
        )
        raise PromptRenderError(msg, section_path=path)

    def placeholder_names(self) -> set[str]:
        """Return the set of placeholder names used in the section template.

        Placeholders are template variables (e.g., ``{name}``, ``{config.value}``)
        that get substituted with values from the section parameters during rendering.

        Override this method in subclasses that support templating to return
        the actual placeholder names extracted from the template. The base
        implementation returns an empty set.

        Returns:
            Set of placeholder identifier strings found in the template.
            Empty set if the section has no template or no placeholders.
        """

        return set()

    @abstractmethod
    def clone(self: Self, **kwargs: object) -> Self:
        """Create a deep copy of this section with optional attribute overrides.

        This method must be implemented by all Section subclasses to support
        prompt composition operations that require copying sections (e.g.,
        merging prompts, applying overrides).

        The clone should:
        1. Copy all section attributes to a new instance
        2. Recursively clone all child sections
        3. Apply any provided kwargs as attribute overrides

        Args:
            **kwargs: Attribute overrides to apply to the cloned section.
                Common overrides include ``title``, ``key``, ``default_params``,
                ``children``, ``enabled``, ``tools``, and ``visibility``.

        Returns:
            A new section instance of the same type with copied/overridden attributes.

        Example::

            # Clone with modified title
            new_section = section.clone(title="Updated Title")

            # Clone with different default params
            new_section = section.clone(default_params=MyParams(value=42))
        """

    def tools(self) -> tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...]:
        """Return the tools declared on this section.

        Tools are capabilities exposed to the model for this section's context.
        When a prompt is rendered, tools from all enabled sections are collected
        and made available to the model.

        Returns:
            Tuple of Tool instances attached to this section. Empty tuple if
            no tools are declared. Does not include tools from child sections
            (those are collected separately during prompt rendering).

        See Also:
            :meth:`policies`: Returns tool policies for this section.
        """

        return self._tools

    def policies(self) -> tuple[ToolPolicy, ...]:
        """Return the tool policies declared on this section.

        Policies define constraints and behaviors for tool usage, such as
        requiring confirmation before execution, limiting call frequency,
        or restricting which tools can be used together.

        Returns:
            Tuple of ToolPolicy instances attached to this section. Empty tuple
            if no policies are declared. Does not include policies from child
            sections (those are collected separately during prompt rendering).

        See Also:
            :meth:`tools`: Returns tools declared on this section.
        """

        return self._policies

    def original_body_template(self) -> str | None:
        """Return the raw body template text before parameter substitution.

        This method returns the original template string as defined in the section,
        without any placeholder substitution applied. It is used for:

        - Content hashing to detect template changes
        - Serialization of section definitions
        - Debugging and introspection

        Override this in subclasses that use template-based rendering (e.g.,
        :class:`MarkdownSection`) to return the actual template text.

        Returns:
            The raw template string if this section uses templating, or None
            for sections that generate content programmatically.
        """

        return None

    def original_summary_template(self) -> str | None:
        """Return the raw summary template text before parameter substitution.

        The summary template is used when the section's visibility is set to
        :attr:`SectionVisibility.SUMMARY`. It provides a condensed version of
        the section content, useful for reducing prompt length while preserving
        key information.

        Returns:
            The summary template string if defined, or None if this section
            does not have a summary. When None, requesting SUMMARY visibility
            will raise a :class:`PromptValidationError`.

        See Also:
            :meth:`effective_visibility`: Resolves the visibility for rendering.
        """

        return self.summary

    def resources(self) -> ResourceRegistry:
        """Return the resource registry for dependencies required by this section.

        Resources are dependencies (like filesystems, databases, or external
        services) that tools in this section need to function. The prompt
        automatically collects and merges resources from all sections.

        Override this method in subclasses that need to contribute resources.
        The default implementation returns an empty registry.

        Returns:
            A ResourceRegistry containing bindings for this section's dependencies.
            Empty registry if no resources are required.

        Example::

            def resources(self) -> ResourceRegistry:
                return ResourceRegistry.build({Filesystem: self.filesystem})

        See Also:
            :class:`~weakincentives.resources.ResourceRegistry`: For building registries.
            :meth:`tools`: Tools that may consume these resources.
        """
        from ..resources import ResourceRegistry

        return ResourceRegistry()

    def effective_visibility(
        self,
        override: SectionVisibility | None = None,
        params: SupportsDataclass | None = None,
        *,
        session: SessionProtocol | None = None,
        path: SectionPath | None = None,
    ) -> SectionVisibility:
        """Return the visibility to use for rendering.

        The visibility is resolved in the following order:
        1. Explicit ``override`` parameter (if provided)
        2. Session state override (if session has VisibilityOverrides with path)
        3. User-provided visibility selector/constant

        Args:
            override: Optional visibility override. When provided, this takes
                precedence over all other visibility sources.
            params: Parameters used to render the section, when available.
            session: Optional session for visibility callables that inspect state.
                Also used to query VisibilityOverrides from session state.
            path: Section path used to look up session-based overrides.

        Returns:
            The effective visibility to use.

        Raises:
            PromptValidationError: If SUMMARY visibility is requested but no
                summary template is defined for this section.
        """
        visibility = override

        # Check session state for override if no explicit override provided
        if visibility is None and session is not None and path is not None:
            from ..runtime.session.visibility_overrides import (
                get_session_visibility_override,
            )

            visibility = get_session_visibility_override(session, path)

        # Fall back to user-provided selector
        if visibility is None:
            visibility = self._visibility(params, session)

        # Raise if SUMMARY requested but no summary template
        if visibility == SectionVisibility.SUMMARY and self.summary is None:
            msg = (
                f"SUMMARY visibility requested for section '{self.key}' "
                "but no summary template is defined."
            )
            raise PromptValidationError(msg, section_path=path)
        return visibility

    @staticmethod
    def _normalize_key(key: str) -> str:
        return normalize_component_key(key, owner="Section")

    @staticmethod
    def _normalize_tools(
        tools: Sequence[object] | None,
    ) -> tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...]:
        if not tools:
            return ()

        from .tool import Tool

        normalized: list[Tool[SupportsDataclassOrNone, SupportsToolResult]] = []
        for tool in tools:
            if not isinstance(tool, Tool):
                raise TypeError("Section tools must be Tool instances.")
            normalized.append(
                cast(Tool[SupportsDataclassOrNone, SupportsToolResult], tool)
            )
        return tuple(normalized)

    @staticmethod
    def _normalize_policies(
        policies: Sequence[ToolPolicy] | None,
    ) -> tuple[ToolPolicy, ...]:
        if not policies:
            return ()
        return tuple(policies)


__all__ = ["Section", "SectionVisibility"]
