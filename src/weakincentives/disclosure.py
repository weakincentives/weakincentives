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

"""Progressive disclosure for prompt sections.

A :class:`Prompt` can ship some sections in :class:`SectionVisibility.SUMMARY`
mode so the model sees a short stub instead of the full body. The model
asks to expand them via the ``open_sections`` tool; the runtime layer
calls :func:`apply_visibility_overrides` on the next iteration to
rebuild the prompt with the requested sections expanded.

The state slice :class:`SectionExpansions` lives on the
:class:`Session` so the policy survives across iterations and shows up
in debug bundles.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import cast, final

from weakincentives.core import (
    MarkdownSection,
    Prompt,
    Replace,
    Section,
    SectionVisibility,
    Session,
    Tool,
    ToolContext,
    ToolResult,
    reducer,
)

__all__ = [
    "ExpandSection",
    "SectionExpansions",
    "apply_visibility_overrides",
    "open_sections_tool",
    "read_section_tool",
]


@final
@dataclass(frozen=True, slots=True)
class ExpandSection:
    """Event signalling the model has asked to expand ``section_key``."""

    section_key: str


def _empty_keys() -> frozenset[str]:
    return frozenset()


@final
@dataclass(frozen=True)
class SectionExpansions:
    """Slice tracking which section keys are currently expanded."""

    keys: frozenset[str] = field(default_factory=_empty_keys)

    @reducer(on=ExpandSection)
    def expand(self, event: ExpandSection) -> Replace[SectionExpansions]:
        return Replace((replace(self, keys=self.keys | {event.section_key}),))


# =============================================================================
# Tools
# =============================================================================


def _open_sections_handler(
    params: object, context: ToolContext
) -> ToolResult[tuple[str, ...]]:
    if not isinstance(params, dict):
        return ToolResult[tuple[str, ...]].error(
            "open_sections expects {'keys': [...]}"
        )
    raw_params = cast("dict[str, object]", params)
    raw = raw_params.get("keys")
    if not isinstance(raw, list):
        return ToolResult[tuple[str, ...]].error(
            "'keys' must be an array of section keys"
        )
    keys = [str(item) for item in cast("list[object]", raw)]
    if not keys:
        return ToolResult[tuple[str, ...]].error("'keys' must not be empty")
    for key in keys:
        context.session.dispatch(ExpandSection(section_key=key))
    return ToolResult[tuple[str, ...]].ok(
        tuple(keys), message=f"expanded {len(keys)} section(s)"
    )


def _read_section_handler(params: object, context: ToolContext) -> ToolResult[str]:
    if not isinstance(params, dict):
        return ToolResult[str].error("read_section expects {'key': str}")
    raw_params = cast("dict[str, object]", params)
    key = raw_params.get("key")
    if not isinstance(key, str):
        return ToolResult[str].error("'key' must be a string")
    target = _find_section(context.prompt, key)
    if target is None:
        return ToolResult[str].error(f"unknown section: {key!r}")
    body = _full_body_for(target)
    return ToolResult[str].ok(body)


open_sections_tool: Tool[object, tuple[str, ...]] = Tool(
    name="open_sections",
    description="Expand previously-summarised prompt sections by key.",
    handler=_open_sections_handler,
)

read_section_tool: Tool[object, str] = Tool(
    name="read_section",
    description="Return the full body of a single prompt section.",
    handler=_read_section_handler,
)


# =============================================================================
# Visibility-aware composition
# =============================================================================


def apply_visibility_overrides(prompt: Prompt, session: Session) -> Prompt:
    """Return a copy of ``prompt`` with currently-expanded sections opened."""
    state = session[SectionExpansions].latest()
    expanded: frozenset[str] = state.keys if state is not None else frozenset()
    if not expanded:
        return prompt
    return Prompt(
        ns=prompt.ns,
        key=prompt.key,
        sections=tuple(
            _override_visibility(section, expanded) for section in prompt.sections
        ),
        params=prompt.params,
        output_type=prompt.output_type,
    )


def _override_visibility(
    section: Section[object], expanded: frozenset[str]
) -> Section[object]:
    children = tuple(
        _override_visibility(child, expanded) for child in section.children
    )
    if section.key in expanded and isinstance(section, MarkdownSection):
        cloned: Section[object] = MarkdownSection[object](
            title=section.title,
            key=section.key,
            template=section.template,
            params_type=section.params_type,
            default_params=section.default_params,
            children=children,
            tools=section.tools,
            enabled=section.enabled,
            visibility=SectionVisibility.FULL,
            summary=section.summary,
        )
        return cloned
    if children == section.children:
        return section
    if isinstance(section, MarkdownSection):
        return MarkdownSection[object](
            title=section.title,
            key=section.key,
            template=section.template,
            params_type=section.params_type,
            default_params=section.default_params,
            children=children,
            tools=section.tools,
            enabled=section.enabled,
            visibility=section.visibility,
            summary=section.summary,
        )
    return section  # pragma: no cover - non-markdown sections are not opened


def _find_section(prompt: Prompt, key: str) -> Section[object] | None:
    for section in prompt.sections:
        found = _find_in_subtree(section, key)
        if found is not None:
            return found
    return None


def _find_in_subtree(section: Section[object], key: str) -> Section[object] | None:
    if section.key == key:
        return section
    for child in section.children:
        found = _find_in_subtree(child, key)
        if found is not None:
            return found
    return None


def _full_body_for(section: Section[object]) -> str:
    if isinstance(section, MarkdownSection):
        return section.template.strip()
    return section.render_body(section.default_params)
