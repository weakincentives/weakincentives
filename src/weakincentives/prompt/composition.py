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

"""Utilities for composing delegation prompts from rendered parents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, ClassVar, Generic, TypeVar, cast, override

from ._types import SupportsDataclass
from .errors import PromptRenderError
from .markdown import MarkdownSection
from .prompt import Prompt, RenderedPrompt
from .response_format import ResponseFormatParams, ResponseFormatSection
from .section import Section

ParentOutputT = TypeVar("ParentOutputT")
DelegationOutputT = TypeVar("DelegationOutputT")


@dataclass(slots=True)
class DelegationParams:
    """Delegation summary fields surfaced to the delegated agent."""

    reason: str
    expected_result: str
    may_delegate_further: str
    recap_lines: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.recap_lines = tuple(self.recap_lines)


@dataclass(slots=True)
class ParentPromptParams:
    """Container for the verbatim parent prompt body."""

    body: str


@dataclass(slots=True)
class RecapParams:
    """Bullet-style recap directives rendered after the parent prompt."""

    bullets: tuple[str, ...]


class DelegationSummarySection(MarkdownSection[DelegationParams]):
    """Delegation summary rendered as a fixed bullet list."""

    def __init__(self) -> None:
        super().__init__(
            title="Delegation Summary",
            key="delegation-summary",
            template=(
                "- **Reason** - ${reason}\n"
                "- **Expected result** - ${expected_result}\n"
                "- **May delegate further?** - ${may_delegate_further}"
            ),
        )


class ParentPromptSection(Section[ParentPromptParams]):
    """Embed the parent prompt verbatim between explicit markers."""

    def __init__(
        self,
        *,
        tools: Sequence[object] | None = None,
        default_params: ParentPromptParams | None = None,
    ) -> None:
        super().__init__(
            title="Parent Prompt (Verbatim)",
            key="parent-prompt",
            tools=tools,
            default_params=default_params,
        )

    @override
    def render(self, params: ParentPromptParams | None, depth: int) -> str:
        if params is None:
            raise PromptRenderError(
                "Parent prompt section requires parameters.",
                dataclass_type=ParentPromptParams,
            )
        heading = "#" * (depth + 2)
        prefix = f"{heading} {self.title}"
        body = params.body
        suffix = "" if body.endswith("\n") else "\n"
        return (
            f"{prefix}\n\n"
            "<!-- PARENT PROMPT START -->\n"
            f"{body}{suffix}"
            "<!-- PARENT PROMPT END -->"
        )


class RecapSection(Section[RecapParams]):
    """Render a concise recap of inherited parent prompt directives."""

    def __init__(self) -> None:
        super().__init__(title="Recap", key="recap")

    @override
    def render(self, params: RecapParams | None, depth: int) -> str:
        if params is None:
            raise PromptRenderError(
                "Recap section requires parameters.",
                dataclass_type=RecapParams,
            )
        heading = "#" * (depth + 2)
        prefix = f"{heading} {self.title}"
        bullets = params.bullets
        bullet_lines = "\n".join(f"- {line}" for line in bullets)
        if bullet_lines:
            return f"{prefix}\n\n{bullet_lines}"
        return prefix


class DelegationPrompt(Generic[ParentOutputT, DelegationOutputT]):  # noqa: UP046
    """Wrap a rendered parent prompt for subagent delegation."""

    _parent_output_type: ClassVar[type[Any] | None] = None
    _delegation_output_type: ClassVar[type[Any] | None] = None

    def __class_getitem__(
        cls, item: tuple[type[Any], type[Any]]
    ) -> type["DelegationPrompt[Any, Any]"]:  # noqa: UP037
        parent_output, delegation_output = item

        name = f"{cls.__name__}[{parent_output.__name__}, {delegation_output.__name__}]"
        namespace = {
            "__module__": cls.__module__,
            "_parent_output_type": parent_output,
            "_delegation_output_type": delegation_output,
        }
        specialized = type(name, (cls,), namespace)
        return cast("type[DelegationPrompt[Any, Any]]", specialized)

    def __init__(
        self,
        parent_prompt: Prompt[ParentOutputT],
        rendered_parent: RenderedPrompt[ParentOutputT],
        *,
        include_response_format: bool = False,
    ) -> None:
        super().__init__()
        self._rendered_parent = rendered_parent
        summary_section = DelegationSummarySection()
        parent_section = ParentPromptSection(
            tools=rendered_parent.tools,
            default_params=ParentPromptParams(body=rendered_parent.text),
        )
        sections: list[Section[Any]] = [summary_section]

        if include_response_format:
            response_section = self._build_response_format_section(rendered_parent)
            if response_section is not None:
                sections.append(response_section)

        sections.append(parent_section)

        self._recap_section = RecapSection()
        sections.append(self._recap_section)

        delegation_output_type = self._resolve_delegation_output_type()
        prompt_cls: type[Prompt[DelegationOutputT]] = Prompt[delegation_output_type]
        self._prompt = prompt_cls(
            ns=f"{parent_prompt.ns}.delegation",
            key=f"{parent_prompt.key}-wrapper",
            sections=tuple(sections),
            inject_output_instructions=False,
            allow_extra_keys=bool(rendered_parent.allow_extra_keys),
        )

    def _resolve_delegation_output_type(self) -> type[DelegationOutputT]:
        candidate = getattr(type(self), "_delegation_output_type", None)
        if isinstance(candidate, type):
            return cast(type[DelegationOutputT], candidate)

        msg = "Specialize DelegationPrompt with an explicit output type"
        raise TypeError(msg)

    def _build_response_format_section(
        self,
        rendered_parent: RenderedPrompt[ParentOutputT],
    ) -> ResponseFormatSection | None:
        container = rendered_parent.container
        if container is None:
            return None

        article = "an" if container.startswith(tuple("aeiou")) else "a"
        extra_clause = (
            "." if rendered_parent.allow_extra_keys else ". Do not add extra keys."
        )

        return ResponseFormatSection(
            params=ResponseFormatParams(
                article=article, container=container, extra_clause=extra_clause
            )
        )

    @property
    def prompt(self) -> Prompt[DelegationOutputT]:
        """Expose the composed prompt for direct access when required."""

        return self._prompt

    def render(
        self,
        summary: DelegationParams,
        parent: ParentPromptParams | None = None,
        recap: RecapParams | None = None,
    ) -> RenderedPrompt[DelegationOutputT]:
        params: list[SupportsDataclass] = [summary]
        parent_params = parent or ParentPromptParams(body=self._rendered_parent.text)
        params.append(parent_params)

        default_recap = RecapParams(bullets=tuple(summary.recap_lines))
        recap_params = recap or default_recap
        params.append(recap_params)

        rendered = self._prompt.render(*tuple(params))
        parent_deadline = self._rendered_parent.deadline
        if parent_deadline is not None and rendered.deadline is not parent_deadline:
            rendered = replace(rendered, deadline=parent_deadline)
        merged_descriptions = _merge_tool_param_descriptions(
            self._rendered_parent.tool_param_descriptions,
            rendered.tool_param_descriptions,
        )
        if merged_descriptions is rendered.tool_param_descriptions:
            return rendered
        return replace(rendered, _tool_param_descriptions=merged_descriptions)


def _merge_tool_param_descriptions(
    parent_descriptions: Mapping[str, Mapping[str, str]],
    rendered_descriptions: Mapping[str, Mapping[str, str]],
) -> Mapping[str, Mapping[str, str]]:
    if not parent_descriptions:
        return rendered_descriptions

    merged: dict[str, dict[str, str]] = {
        name: dict(fields) for name, fields in parent_descriptions.items()
    }
    for name, fields in rendered_descriptions.items():
        if name not in merged:
            merged[name] = dict(fields)
        else:
            merged[name].update(fields)

    return MappingProxyType(
        {
            name: MappingProxyType(dict(field_mapping))
            for name, field_mapping in merged.items()
        }
    )


__all__ = [
    "DelegationParams",
    "DelegationPrompt",
    "DelegationSummarySection",
    "ParentPromptParams",
    "ParentPromptSection",
    "RecapParams",
    "RecapSection",
]
