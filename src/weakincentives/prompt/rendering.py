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

# pyright: reportImportCycles=false

"""Rendering helpers for :mod:`weakincentives.prompt`."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, MutableMapping
from dataclasses import field, is_dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, override

from ..dataclasses import FrozenDataclass
from ..deadlines import Deadline
from ._types import SupportsDataclass, SupportsDataclassOrNone, SupportsToolResult
from ._visibility import SectionVisibility
from .errors import PromptRenderError, PromptValidationError, SectionPath
from .progressive_disclosure import (
    build_summary_suffix,
    build_vfs_summary_suffix,
    compute_current_visibility,
    compute_vfs_context_path,
    create_open_sections_handler,
    find_workspace_section,
    section_subtree_has_tools,
)
from .registry import RegistrySnapshot, SectionNode
from .section import Section
from .structured_output import StructuredOutputConfig
from .tool import Tool

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .overrides import PromptDescriptor, ToolOverride


_EMPTY_TOOL_PARAM_DESCRIPTIONS: Mapping[str, Mapping[str, str]] = MappingProxyType({})


OutputT_co = TypeVar("OutputT_co", covariant=True)


@FrozenDataclass()
class RenderedPrompt[OutputT_co]:
    """Rendered prompt text paired with structured output metadata."""

    text: str
    structured_output: StructuredOutputConfig[SupportsDataclass] | None = None
    deadline: Deadline | None = None
    descriptor: PromptDescriptor | None = None
    _tools: tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...] = field(
        default_factory=tuple
    )
    _tool_param_descriptions: Mapping[str, Mapping[str, str]] = field(
        default=_EMPTY_TOOL_PARAM_DESCRIPTIONS
    )

    @override
    def __str__(self) -> str:  # pragma: no cover - delegated behaviour
        return self.text

    @property
    def tools(self) -> tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...]:
        """Tools contributed by enabled sections in traversal order."""

        return self._tools

    @property
    def tool_param_descriptions(
        self,
    ) -> Mapping[str, Mapping[str, str]]:
        """Description patches keyed by tool name."""

        return self._tool_param_descriptions

    @property
    def output_type(self) -> type[SupportsDataclass] | None:
        """Return the declared dataclass type for structured output."""

        if self.structured_output is None:
            return None
        return self.structured_output.dataclass_type

    @property
    def container(self) -> Literal["object", "array"] | None:
        """Return the declared container for structured output."""

        if self.structured_output is None:
            return None
        return self.structured_output.container

    @property
    def allow_extra_keys(self) -> bool | None:
        """Return whether extra keys are allowed in structured output."""

        if self.structured_output is None:
            return None
        return self.structured_output.allow_extra_keys


def _freeze_tool_param_descriptions(
    descriptions: Mapping[str, dict[str, str]],
) -> Mapping[str, Mapping[str, str]]:
    if not descriptions:
        return MappingProxyType({})
    frozen: dict[str, Mapping[str, str]] = {}
    for name, field_mapping in descriptions.items():
        frozen[name] = MappingProxyType(dict(field_mapping))
    return MappingProxyType(frozen)


OutputT = TypeVar("OutputT")


class PromptRenderer[OutputT]:
    """Render prompts using a registry snapshot."""

    def __init__(
        self,
        *,
        registry: RegistrySnapshot,
        structured_output: StructuredOutputConfig[SupportsDataclass] | None,
    ) -> None:
        super().__init__()
        self._registry = registry
        self._structured_output = structured_output

    def build_param_lookup(
        self, params: tuple[SupportsDataclass, ...]
    ) -> dict[type[SupportsDataclass], SupportsDataclass]:
        lookup: dict[type[SupportsDataclass], SupportsDataclass] = {}
        for value in params:
            if isinstance(value, type):
                provided_type: type[Any] = value
            else:
                provided_type = type(value)
            if isinstance(value, type) or not is_dataclass(value):
                raise PromptValidationError(
                    "Prompt expects dataclass instances.",
                    dataclass_type=provided_type,
                )
            params_type = cast(type[SupportsDataclass], provided_type)
            if params_type in lookup:
                raise PromptValidationError(
                    "Duplicate params type supplied to prompt.",
                    dataclass_type=params_type,
                )
            if params_type not in self._registry.param_types:
                raise PromptValidationError(
                    "Unexpected params type supplied to prompt.",
                    dataclass_type=params_type,
                )
            lookup[params_type] = value
        return lookup

    def render(  # noqa: PLR0914, C901
        self,
        param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass],
        overrides: Mapping[SectionPath, str] | None = None,
        tool_overrides: Mapping[str, ToolOverride] | None = None,
        *,
        descriptor: PromptDescriptor | None = None,
        visibility_overrides: Mapping[SectionPath, SectionVisibility] | None = None,
    ) -> RenderedPrompt[OutputT]:
        rendered_sections: list[str] = []
        collected_tools: list[Tool[SupportsDataclassOrNone, SupportsToolResult]] = []
        override_lookup = dict(overrides or {})
        tool_override_lookup = dict(tool_overrides or {})
        visibility_override_lookup = dict(visibility_overrides or {})
        field_description_patches: dict[str, dict[str, str]] = {}
        summary_skip_depth: int | None = None
        has_summarized_with_tools = False

        # Find workspace section for auto-rendering tool-free summarized content
        workspace_section = find_workspace_section(self._registry)

        for node, section_params in self._iter_enabled_sections(dict(param_lookup)):
            # Skip children of sections rendered with SUMMARY visibility
            if summary_skip_depth is not None:
                if node.depth > summary_skip_depth:
                    continue
                summary_skip_depth = None

            override_body = (
                override_lookup.get(node.path)
                if getattr(node.section, "accepts_overrides", True)
                else None
            )
            visibility_override = visibility_override_lookup.get(node.path)
            effective_visibility = node.section.effective_visibility(
                visibility_override, section_params
            )

            # When rendering with SUMMARY visibility, skip children
            if effective_visibility == SectionVisibility.SUMMARY:
                summary_skip_depth = node.depth

            rendered = self._render_section(
                node, section_params, override_body, visibility_override
            )

            # Append summary suffix for sections rendered with SUMMARY visibility
            if (
                effective_visibility == SectionVisibility.SUMMARY
                and node.section.summary is not None
                and rendered
            ):
                subtree_has_tools = section_subtree_has_tools(node.section)

                if subtree_has_tools:
                    # Section has tools - use open_sections suffix
                    has_summarized_with_tools = True
                    section_key = ".".join(node.path)
                    child_keys = self._collect_child_keys(node)
                    suffix = build_summary_suffix(section_key, child_keys)
                    rendered += suffix
                elif workspace_section is not None:
                    # Section has no tools and workspace available - auto-render to VFS
                    self._auto_render_to_vfs(
                        node, section_params, workspace_section, dict(param_lookup)
                    )
                    suffix = build_vfs_summary_suffix(node.path)
                    rendered += suffix
                else:
                    # No workspace section - fall back to open_sections
                    has_summarized_with_tools = True
                    section_key = ".".join(node.path)
                    child_keys = self._collect_child_keys(node)
                    suffix = build_summary_suffix(section_key, child_keys)
                    rendered += suffix

            # Don't collect tools when rendering with SUMMARY visibility
            if effective_visibility != SectionVisibility.SUMMARY:
                self._collect_section_tools(
                    node.section,
                    tool_override_lookup,
                    collected_tools,
                    field_description_patches,
                )

            if rendered:
                rendered_sections.append(rendered)

        # Inject open_sections tool only when there are summarized sections with tools
        if has_summarized_with_tools:
            current_visibility = compute_current_visibility(
                self._registry,
                visibility_overrides,
                param_lookup,
            )
            open_sections_tool = create_open_sections_handler(
                registry=self._registry,
                current_visibility=current_visibility,
            )
            collected_tools.append(
                cast(
                    Tool[SupportsDataclassOrNone, SupportsToolResult],
                    open_sections_tool,
                )
            )

        text = "\n\n".join(rendered_sections)

        return RenderedPrompt[OutputT](
            text=text,
            structured_output=self._structured_output,
            descriptor=descriptor,
            _tools=tuple(collected_tools),
            _tool_param_descriptions=_freeze_tool_param_descriptions(
                field_description_patches
            ),
        )

    def _collect_child_keys(
        self, parent_node: SectionNode[SupportsDataclass]
    ) -> tuple[str, ...]:
        """Collect the keys of direct child sections for a parent node."""
        child_keys: list[str] = []
        parent_depth = parent_node.depth
        parent_path = parent_node.path
        in_parent = False

        for node in self._registry.sections:
            if node is parent_node:
                in_parent = True
                continue

            if in_parent:
                # Direct children have depth == parent_depth + 1
                # and their path starts with parent's path
                if node.depth == parent_depth + 1 and node.path[:-1] == parent_path:
                    child_keys.append(node.section.key)
                elif node.depth <= parent_depth:
                    # We've moved past the parent's children
                    break

        return tuple(child_keys)

    def _collect_section_tools(
        self,
        section: Section[SupportsDataclass],
        tool_override_lookup: dict[str, ToolOverride],
        collected_tools: list[Tool[SupportsDataclassOrNone, SupportsToolResult]],
        field_description_patches: dict[str, dict[str, str]],
    ) -> None:
        section_tools = section.tools()
        if not section_tools:
            return

        for tool in section_tools:
            override = (
                tool_override_lookup.get(tool.name) if tool.accepts_overrides else None
            )
            patched_tool = tool
            if override is not None:
                if (
                    override.description is not None
                    and override.description != tool.description
                ):
                    patched_tool = replace(tool, description=override.description)
                if override.param_descriptions:
                    field_description_patches[tool.name] = dict(
                        override.param_descriptions
                    )
            collected_tools.append(patched_tool)

    def _iter_enabled_sections(
        self,
        param_lookup: MutableMapping[type[SupportsDataclass], SupportsDataclass],
    ) -> Iterator[tuple[SectionNode[SupportsDataclass], SupportsDataclass | None]]:
        skip_depth: int | None = None

        for node in self._registry.sections:
            if skip_depth is not None:
                if node.depth > skip_depth:
                    continue
                skip_depth = None

            section_params = self._registry.resolve_section_params(node, param_lookup)

            try:
                enabled = node.section.is_enabled(section_params)
            except Exception as error:  # pragma: no cover - defensive
                raise PromptRenderError(
                    "Section enabled predicate failed.",
                    section_path=node.path,
                    dataclass_type=node.section.param_type,
                ) from error

            if not enabled:
                skip_depth = node.depth
                continue

            yield node, section_params

    def _render_section(
        self,
        node: SectionNode[SupportsDataclass],
        section_params: SupportsDataclass | None,
        override_body: str | None,
        visibility_override: SectionVisibility | None = None,
    ) -> str:
        params_type = node.section.param_type
        try:
            render_override = getattr(node.section, "render_with_template", None)
            if override_body is not None and callable(render_override):
                override_renderer = cast(
                    Callable[
                        [str, SupportsDataclass | None, int, str, tuple[str, ...]], str
                    ],
                    render_override,
                )
                rendered = override_renderer(
                    override_body, section_params, node.depth, node.number, node.path
                )
            else:
                rendered = node.section.render(
                    section_params,
                    node.depth,
                    node.number,
                    path=node.path,
                    visibility=visibility_override,
                )
        except PromptRenderError as error:
            if error.section_path and error.dataclass_type:
                raise
            raise PromptRenderError(
                error.message,
                section_path=node.path,
                dataclass_type=params_type,
                placeholder=error.placeholder,
            ) from error
        except Exception as error:  # pragma: no cover - defensive
            raise PromptRenderError(
                "Section rendering failed.",
                section_path=node.path,
                dataclass_type=params_type,
            ) from error

        return rendered

    def _auto_render_to_vfs(
        self,
        node: SectionNode[SupportsDataclass],
        section_params: SupportsDataclass | None,
        workspace_section: object,
        param_lookup: dict[type[SupportsDataclass], SupportsDataclass],
    ) -> None:
        """Render a section subtree to the VFS as a context file.

        Args:
            node: The section node being rendered with SUMMARY visibility.
            section_params: Parameters for the section.
            workspace_section: The workspace section providing VFS access.
            param_lookup: Parameter lookup for resolving child section params.
        """
        from ..tools.vfs import VfsPath, VirtualFileSystem

        # Render the full content of the section subtree
        full_content = self._render_section_subtree(node, section_params, param_lookup)

        # Compute the VFS path
        vfs_path_str = compute_vfs_context_path(node.path)
        # Remove leading slash and split into segments
        path_segments = tuple(vfs_path_str.lstrip("/").split("/"))
        vfs_path = VfsPath(segments=path_segments)

        # Access the session from the workspace section
        session = getattr(workspace_section, "session", None)
        if session is None:
            return

        # Get the current VFS state and create a new file entry directly
        from datetime import UTC, datetime

        from ..tools.vfs import VfsFile

        vfs_snapshot = session.query(VirtualFileSystem).latest()
        current_files = list(vfs_snapshot.files) if vfs_snapshot else []

        # Check if file already exists and remove it (for overwrite)
        current_files = [f for f in current_files if f.path.segments != path_segments]

        # Create new file entry
        now = datetime.now(UTC)
        size_bytes = len(full_content.encode("utf-8"))
        new_file = VfsFile(
            path=vfs_path,
            content=full_content,
            encoding="utf-8",
            size_bytes=size_bytes,
            version=1,
            created_at=now,
            updated_at=now,
        )
        current_files.append(new_file)

        # Update VFS state directly using seed
        new_vfs = VirtualFileSystem(files=tuple(current_files))
        session.mutate(VirtualFileSystem).seed(new_vfs)

    def _render_section_subtree(
        self,
        node: SectionNode[SupportsDataclass],
        section_params: SupportsDataclass | None,
        param_lookup: dict[type[SupportsDataclass], SupportsDataclass],
    ) -> str:
        """Render a section and all its children as full content.

        Args:
            node: The root section node to render.
            section_params: Parameters for the root section.
            param_lookup: Parameter lookup for resolving child section params.

        Returns:
            The rendered markdown content including all children.
        """
        parts: list[str] = []

        # Render the root section with FULL visibility
        root_rendered = node.section.render(
            section_params,
            node.depth,
            node.number,
            path=node.path,
            visibility=SectionVisibility.FULL,
        )
        if root_rendered:
            parts.append(root_rendered)

        # Render all children recursively
        self._render_children_recursive(
            node.section, node.depth, node.number, node.path, param_lookup, parts
        )

        return "\n\n".join(parts)

    def _render_children_recursive(  # noqa: PLR0913, PLR0917
        self,
        section: Section[SupportsDataclass],
        parent_depth: int,
        parent_number: str,
        parent_path: SectionPath,
        param_lookup: dict[type[SupportsDataclass], SupportsDataclass],
        parts: list[str],
    ) -> None:
        """Recursively render children of a section.

        Args:
            section: The parent section.
            parent_depth: The depth of the parent section.
            parent_number: The section number of the parent.
            parent_path: The path of the parent section.
            param_lookup: Parameter lookup for resolving params.
            parts: List to append rendered content to.
        """
        for idx, child in enumerate(section.children, start=1):
            child_depth = parent_depth + 1
            child_number = f"{parent_number}.{idx}"
            child_path = (*parent_path, child.key)

            # Resolve params for the child section
            child_params = self._resolve_child_params(child, param_lookup)

            # Check if child is enabled
            if not child.is_enabled(child_params):
                continue

            # Render the child with FULL visibility
            child_rendered = child.render(
                child_params,
                child_depth,
                child_number,
                path=child_path,
                visibility=SectionVisibility.FULL,
            )
            if child_rendered:
                parts.append(child_rendered)

            # Recurse into grandchildren
            self._render_children_recursive(
                child, child_depth, child_number, child_path, param_lookup, parts
            )

    def _resolve_child_params(
        self,
        section: Section[SupportsDataclass],
        param_lookup: dict[type[SupportsDataclass], SupportsDataclass],
    ) -> SupportsDataclass | None:
        """Resolve parameters for a child section.

        Args:
            section: The section to resolve parameters for.
            param_lookup: Parameter lookup mapping.

        Returns:
            The resolved parameters or None if the section has no params type.
        """
        params_type = section.param_type
        if params_type is None:
            return None

        # Try to find in param_lookup
        params = param_lookup.get(params_type)
        if params is not None:
            return params

        # Try default_params
        if section.default_params is not None:
            return section.default_params

        # Try to construct with no args
        try:
            constructor = cast(Callable[[], SupportsDataclass | None], params_type)
            return constructor()
        except TypeError:  # pragma: no cover - defensive; registry fails first
            return None


__all__ = ["PromptRenderer", "RenderedPrompt"]
