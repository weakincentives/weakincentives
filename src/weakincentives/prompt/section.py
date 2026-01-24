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
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypeVar, cast

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
    """Abstract building block for prompt content."""

    _generic_owner_name: ClassVar[str | None] = "Section"

    key: str
    children: tuple[Section[Any], ...]

    def __init__(
        self,
        *,
        title: str,
        key: str,
        default_params: SectionParamsT | None = None,
        children: Sequence[Section[Any]] | None = None,
        enabled: EnabledPredicate | None = None,
        tools: Sequence[object] | None = None,
        policies: Sequence[ToolPolicy] | None = None,
        accepts_overrides: bool = True,
        summary: str | None = None,
        visibility: VisibilitySelector = SectionVisibility.FULL,
    ) -> None:
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

        normalized_children: list[Section[Any]] = []
        raw_children: Sequence[object] = cast(Sequence[object], children or ())
        for child in raw_children:
            if not isinstance(child, Section):
                raise TypeError("Section children must be Section instances.")
            normalized_children.append(cast(Section[Any], child))
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
        """Return True when the section should render for the given params.

        Args:
            params: The section parameters.
            session: Optional session for visibility callables that inspect state.
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
        """Render tool examples for this section.

        Returns:
            Formatted tool examples block, or empty string if no tools/examples.
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
        """Return placeholder identifiers used by the section template."""

        return set()

    @abstractmethod
    def clone(self: Self, **kwargs: object) -> Self:
        """Return a deep copy of the section and its children."""

    def tools(self) -> tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...]:
        """Return the tools exposed by this section."""

        return self._tools

    def policies(self) -> tuple[ToolPolicy, ...]:
        """Return the policies declared by this section."""

        return self._policies

    def original_body_template(self) -> str | None:
        """Return the template text that participates in hashing, when available."""

        return None

    def original_summary_template(self) -> str | None:
        """Return the summary template text, when available."""

        return self.summary

    def resources(self) -> ResourceRegistry:
        """Return resources required by this section.

        Override to contribute resources. Default returns empty registry.
        The prompt collects resources from all sections automatically.

        Example::

            def resources(self) -> ResourceRegistry:
                return ResourceRegistry.build({Filesystem: self.filesystem})
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
