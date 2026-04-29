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

"""Hierarchical prompts that bundle typed parameters and tools.

Sections are parameterized by a frozen dataclass so the placeholders in the
template, the field types of the dataclass, and the binding API line up at
construction time.
"""

from __future__ import annotations

import enum
import re
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from string import Template
from typing import Any, final, override

from .errors import PromptError
from .tool import Tool

__all__ = [
    "MarkdownSection",
    "Prompt",
    "RenderedPrompt",
    "Section",
    "SectionVisibility",
]


class SectionVisibility(enum.Enum):
    """How a section renders by default."""

    FULL = "full"
    SUMMARY = "summary"


_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def _check_name(value: str, *, kind: str) -> None:
    if not _NAME_RE.match(value):
        raise PromptError(f"invalid {kind}: {value!r}")


def _placeholders(template: str) -> frozenset[str]:
    compiled = Template(textwrap.dedent(template).strip())
    found: set[str] = set()
    for match in compiled.pattern.finditer(compiled.template):
        named = match.group("named") or match.group("braced")
        if named:
            found.add(named)
    return frozenset(found)


def _dataclass_fields(params_type: type[object] | None) -> frozenset[str]:
    if params_type is None:
        return frozenset()
    if not is_dataclass(params_type):
        raise PromptError(f"section params type must be a dataclass: {params_type!r}")
    return frozenset(f.name for f in fields(params_type))


class Section[ParamsT]:
    """Base class for prompt sections.

    Subclasses define how a section renders its body. The spine ships
    :class:`MarkdownSection`; extras may add others (HTML, JSON-schema, etc.)
    by subclassing :class:`Section`.
    """

    title: str
    key: str
    children: tuple[Section[Any], ...]
    tools: tuple[Tool[Any, Any], ...]
    params_type: type[ParamsT] | None
    default_params: ParamsT | None
    enabled: Callable[[ParamsT | None], bool] | None
    visibility: SectionVisibility
    summary: str | None

    def __init__(
        self,
        *,
        title: str,
        key: str,
        params_type: type[ParamsT] | None = None,
        default_params: ParamsT | None = None,
        children: Sequence[Section[Any]] = (),
        tools: Sequence[Tool[Any, Any]] = (),
        enabled: Callable[[ParamsT | None], bool] | None = None,
        visibility: SectionVisibility = SectionVisibility.FULL,
        summary: str | None = None,
    ) -> None:
        super().__init__()
        _check_name(key, kind="section key")
        self.title = title
        self.key = key
        self.children = tuple(children)
        self.tools = tuple(tools)
        self.params_type = params_type
        self.default_params = default_params
        self.enabled = enabled
        self.visibility = visibility
        self.summary = summary

    def reachable_tools(self) -> tuple[Tool[Any, Any], ...]:
        out: list[Tool[Any, Any]] = list(self.tools)
        for child in self.children:
            out.extend(child.reachable_tools())
        return tuple(out)

    def render_body(self, params: ParamsT | None) -> str:
        del params
        return ""

    def expects(self) -> frozenset[str]:
        return _dataclass_fields(self.params_type)


@final
class MarkdownSection[ParamsT](Section[ParamsT]):
    """Markdown body rendered with :class:`string.Template`."""

    template: str

    def __init__(
        self,
        *,
        title: str,
        key: str,
        template: str = "",
        params_type: type[ParamsT] | None = None,
        default_params: ParamsT | None = None,
        children: Sequence[Section[Any]] = (),
        tools: Sequence[Tool[Any, Any]] = (),
        enabled: Callable[[ParamsT | None], bool] | None = None,
        visibility: SectionVisibility = SectionVisibility.FULL,
        summary: str | None = None,
    ) -> None:
        super().__init__(
            title=title,
            key=key,
            params_type=params_type,
            default_params=default_params,
            children=children,
            tools=tools,
            enabled=enabled,
            visibility=visibility,
            summary=summary,
        )
        self.template = template
        if template:
            placeholders = _placeholders(template)
            allowed = _dataclass_fields(params_type)
            missing = placeholders - allowed
            if params_type is not None and missing:
                raise PromptError(
                    f"section {key!r} template references "
                    f"unknown fields: {sorted(missing)!r}"
                )

    @override
    def render_body(self, params: ParamsT | None) -> str:
        if self.visibility is SectionVisibility.SUMMARY and self.summary is not None:
            return self.summary.strip()
        if not self.template:
            return ""
        compiled = Template(textwrap.dedent(self.template).strip())
        substitutions: dict[str, object] = {}
        resolved = params if params is not None else self.default_params
        if resolved is not None:
            if not is_dataclass(resolved) or isinstance(resolved, type):
                raise PromptError(f"section {self.key!r} expected a dataclass instance")
            for f in fields(resolved):
                substitutions[f.name] = getattr(resolved, f.name)
        try:
            return compiled.substitute(substitutions).strip()
        except KeyError as error:
            placeholder = error.args[0]
            raise PromptError(
                f"missing placeholder {placeholder!r} in section {self.key!r}"
            ) from error


@final
@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """Output of :meth:`Prompt.render`."""

    text: str
    tools: tuple[Tool[Any, Any], ...]
    output_type: type[object] | None = None


def _empty_param_map() -> Mapping[str, object]:
    return {}


@final
@dataclass(frozen=True, slots=True)
class Prompt:
    """A namespaced prompt composed of typed sections.

    ``params`` maps section key to a parameter dataclass instance; each
    instance must match the section's declared ``params_type``.
    ``output_type`` optionally declares the dataclass an adapter should
    coerce the model's response into; helpers in ``weakincentives.serde``
    do the actual parsing so the spine stays JSON-walking-only.
    """

    ns: str
    key: str
    sections: tuple[Section[Any], ...] = ()
    params: Mapping[str, object] = field(default_factory=_empty_param_map)
    output_type: type[object] | None = None

    def __post_init__(self) -> None:
        _check_name(self.ns, kind="namespace")
        _check_name(self.key, kind="key")
        seen: set[str] = set()
        for section in self.sections:
            for tool in section.reachable_tools():
                if tool.name in seen:
                    raise PromptError(f"duplicate tool name: {tool.name}")
                seen.add(tool.name)

    def bind(self, **params: object) -> Prompt:
        """Return a new :class:`Prompt` with overridden section parameters."""
        merged: dict[str, object] = {**self.params, **params}
        return Prompt(
            ns=self.ns,
            key=self.key,
            sections=self.sections,
            params=merged,
            output_type=self.output_type,
        )

    def render(self) -> RenderedPrompt:
        chunks: list[str] = []
        tools: list[Tool[Any, Any]] = []
        for index, section in enumerate(self.sections, start=1):
            self._render_section(section, depth=0, number=str(index), out=chunks)
            tools.extend(section.reachable_tools())
        return RenderedPrompt(
            text="\n\n".join(chunks),
            tools=tuple(tools),
            output_type=self.output_type,
        )

    def _render_section(
        self,
        section: Section[Any],
        *,
        depth: int,
        number: str,
        out: list[str],
    ) -> None:
        params = self.params.get(section.key)
        if section.enabled is not None and not section.enabled(params):
            return
        heading = f"{'#' * (depth + 2)} {number}. {section.title}"
        body = section.render_body(params)
        out.append(f"{heading}\n\n{body}" if body else heading)
        for child_index, child in enumerate(section.children, start=1):
            self._render_section(
                child,
                depth=depth + 1,
                number=f"{number}.{child_index}",
                out=out,
            )
