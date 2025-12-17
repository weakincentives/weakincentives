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

import json
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import fields, is_dataclass
from string import Template
from typing import Any, Literal, Self, TypeVar, cast, override

from ..serde import clone as clone_dataclass, dump
from ..types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)
from .errors import PromptRenderError
from .section import Section, SectionVisibility, VisibilitySelector
from .tool import Tool
from .tool_result import render_tool_payload

MarkdownParamsT = TypeVar(
    "MarkdownParamsT",
    bound=SupportsDataclass,
    covariant=True,
)


class MarkdownSection(Section[MarkdownParamsT]):
    """Render markdown content using :class:`string.Template`."""

    def __init__(
        self,
        *,
        title: str,
        template: str,
        key: str,
        default_params: MarkdownParamsT | None = None,
        children: Sequence[Section[SupportsDataclass]] | None = None,
        enabled: Callable[[SupportsDataclass], bool] | None = None,
        tools: Sequence[object] | None = None,
        accepts_overrides: bool = True,
        summary: str | None = None,
        visibility: VisibilitySelector = SectionVisibility.FULL,
    ) -> None:
        self.template = template
        super().__init__(
            title=title,
            key=key,
            default_params=default_params,
            children=children,
            enabled=enabled,
            tools=tools,
            accepts_overrides=accepts_overrides,
            summary=summary,
            visibility=visibility,
        )

    @override
    def render(
        self,
        params: SupportsDataclass | None,
        depth: int,
        number: str,
        *,
        path: tuple[str, ...] = (),
        visibility: SectionVisibility | None = None,
    ) -> str:
        # Use passed visibility directly when provided (already computed by renderer)
        # Fall back to effective_visibility for direct render() calls without renderer
        effective = (
            visibility
            if visibility is not None
            else self.effective_visibility(override=None, params=params)
        )
        if effective == SectionVisibility.SUMMARY and self.summary is not None:
            return self.render_with_template(self.summary, params, depth, number, path)
        return self.render_with_template(self.template, params, depth, number, path)

    @override
    def render_override(
        self,
        override_body: str,
        params: SupportsDataclass | None,
        depth: int,
        number: str,
        path: tuple[str, ...] = (),
    ) -> str:
        return self.render_with_template(override_body, params, depth, number, path)

    def render_with_template(
        self,
        template_text: str,
        params: SupportsDataclass | None,
        depth: int,
        number: str,
        path: tuple[str, ...] = (),
    ) -> str:
        heading_level = "#" * (depth + 2)
        normalized_number = number.rstrip(".")
        path_str = ".".join(path) if path else ""
        title_with_path = (
            f"{self.title.strip()} ({path_str})" if path_str else self.title.strip()
        )
        heading = f"{heading_level} {normalized_number}. {title_with_path}"
        template = Template(textwrap.dedent(template_text).strip())
        try:
            normalized_params = self._normalize_params(params)
            rendered_body = template.substitute(normalized_params)
        except KeyError as error:
            missing = error.args[0]
            raise PromptRenderError(
                "Missing placeholder during render.",
                placeholder=str(missing),
            ) from error
        rendered_tools = _render_tool_examples_block(self.tools())
        normalized_body = rendered_body.strip()
        combined_body = normalized_body
        if rendered_tools:
            combined_body = (
                f"{normalized_body}\n\n{rendered_tools}"
                if normalized_body
                else rendered_tools
            )

        if combined_body:
            return f"{heading}\n\n{combined_body}"
        return heading

    @override
    def placeholder_names(self) -> set[str]:
        template = Template(textwrap.dedent(self.template).strip())
        placeholders: set[str] = set()
        for match in template.pattern.finditer(template.template):
            named = match.group("named")
            if named:
                placeholders.add(named)
                continue
            braced = match.group("braced")
            if braced:
                placeholders.add(braced)
        return placeholders

    @staticmethod
    def _normalize_params(params: SupportsDataclass | None) -> dict[str, Any]:
        if params is None:
            return {}
        if not is_dataclass(params) or isinstance(params, type):
            raise PromptRenderError(
                "Section params must be a dataclass instance.",
                dataclass_type=type(params),
            )

        return {field.name: getattr(params, field.name) for field in fields(params)}

    @override
    def original_body_template(self) -> str:
        return self.template

    @override
    def clone(self, **kwargs: object) -> Self:
        cloned_children: list[Section[SupportsDataclass]] = []
        for child in self.children:
            if not hasattr(child, "clone"):
                raise TypeError(
                    "Section children must implement clone()."
                )  # pragma: no cover
            cloned_children.append(child.clone(**kwargs))

        cloned_default = (
            clone_dataclass(self.default_params)
            if self.default_params is not None
            else None
        )

        cls: type[Any] = type(self)
        clone = cls(
            title=self.title,
            template=self.template,
            key=self.key,
            default_params=cloned_default,
            children=cloned_children,
            enabled=self._enabled,  # ty: ignore[invalid-argument-type]  # callback arity
            tools=self.tools(),
            accepts_overrides=self.accepts_overrides,
            summary=self.summary,
            visibility=self.visibility,
        )
        return cast(Self, clone)


def _render_tool_examples_block(
    tools: Sequence[Tool[SupportsDataclassOrNone, SupportsToolResult]],
) -> str:
    rendered: list[str] = []
    for tool in tools:
        if not tool.examples:
            continue

        examples_block = _render_examples_for_tool(tool)
        rendered.append(
            "\n".join(
                [
                    f"- {tool.name}: {tool.description}",
                    textwrap.indent(examples_block, "  "),
                ]
            )
        )

    if not rendered:
        return ""

    return "\n".join(["Tools:", *rendered])


def _render_examples_for_tool(
    tool: Tool[SupportsDataclassOrNone, SupportsToolResult],
) -> str:
    lines: list[str] = [f"- {tool.name} examples:"]
    for example in tool.examples:
        rendered_output = _render_example_output(
            example.output, container=tool.result_container
        )
        lines.extend(
            [
                f"  - description: {example.description}",
                "    input:",
                *_render_fenced_block(
                    _render_example_value(example.input),
                    indent="      ",
                    language="json",
                ),
                "    output:",
                *_render_fenced_block(rendered_output, indent="      ", language=None),
            ]
        )
    return "\n".join(lines)


def _render_example_value(value: SupportsDataclass | None) -> str:
    if value is None:
        return "null"

    serialized_value = dump(value, exclude_none=True)

    return json.dumps(serialized_value, ensure_ascii=False)


def _render_example_output(
    value: SupportsToolResult | None,
    *,
    container: Literal["object", "array"],
) -> str:
    if container == "array":
        sequence_value = cast(Sequence[object], value or [])
        return render_tool_payload(list(sequence_value))

    return render_tool_payload(value)


def _render_fenced_block(
    content: str, *, indent: str, language: str | None
) -> list[str]:
    fence = "```" if language is None else f"```{language}"
    indented_lines = content.splitlines() or [""]
    return [
        f"{indent}{fence}",
        *[f"{indent}{line}" for line in indented_lines],
        f"{indent}```",
    ]


__all__ = ["MarkdownSection"]
