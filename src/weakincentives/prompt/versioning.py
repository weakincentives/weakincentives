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
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Protocol

from ..serde.dataclass_serde import schema
from ._types import SupportsDataclass
from .tool import Tool


def _section_override_mapping_factory() -> dict[tuple[str, ...], SectionOverride]:
    return {}


def _tool_override_mapping_factory() -> dict[str, ToolOverride]:
    return {}


def _param_description_mapping_factory() -> dict[str, str]:
    return {}


class SectionLike(Protocol):
    def original_body_template(self) -> str | None: ...

    def tools(self) -> tuple[Tool[SupportsDataclass, SupportsDataclass], ...]: ...


class SectionNodeLike(Protocol):
    path: tuple[str, ...]
    section: SectionLike


class PromptLike(Protocol):
    ns: str
    key: str

    @property
    def sections(self) -> tuple[SectionNodeLike, ...]: ...


@dataclass(slots=True)
class SectionDescriptor:
    """Hash metadata for a single section within a prompt."""

    path: tuple[str, ...]
    content_hash: str


@dataclass(slots=True)
class ToolDescriptor:
    """Stable metadata describing a tool exposed by a prompt."""

    path: tuple[str, ...]
    name: str
    contract_hash: str


@dataclass(slots=True)
class PromptDescriptor:
    """Stable metadata describing a prompt and its hash-aware sections."""

    ns: str
    key: str
    sections: list[SectionDescriptor]
    tools: list[ToolDescriptor]

    @classmethod
    def from_prompt(cls, prompt: PromptLike) -> PromptDescriptor:
        sections: list[SectionDescriptor] = []
        tools: list[ToolDescriptor] = []
        for node in prompt.sections:
            template = node.section.original_body_template()
            if template is not None:
                content_hash = sha256(template.encode("utf-8")).hexdigest()
                sections.append(SectionDescriptor(node.path, content_hash))
            tool_descriptors = [
                ToolDescriptor(
                    path=node.path,
                    name=tool.name,
                    contract_hash=_tool_contract_hash(tool),
                )
                for tool in node.section.tools()
            ]
            tools.extend(tool_descriptors)
        return cls(prompt.ns, prompt.key, sections, tools)


@dataclass(slots=True)
class SectionOverride:
    """Override payload for a prompt section validated by hash."""

    expected_hash: str
    body: str


@dataclass(slots=True)
class ToolOverride:
    """Description overrides validated against a tool contract hash."""

    name: str
    expected_contract_hash: str
    description: str | None = None
    param_descriptions: dict[str, str] = field(
        default_factory=_param_description_mapping_factory
    )


@dataclass(slots=True)
class PromptOverride:
    """Runtime replacements for prompt sections validated by an overrides store."""

    ns: str
    prompt_key: str
    tag: str
    sections: dict[tuple[str, ...], SectionOverride] = field(
        default_factory=_section_override_mapping_factory
    )
    tool_overrides: dict[str, ToolOverride] = field(
        default_factory=_tool_override_mapping_factory
    )


class PromptOverridesError(Exception):
    """Raised when prompt overrides fail validation or persistence."""


class PromptOverridesStore(Protocol):
    """Lookup interface for resolving prompt overrides at render time."""

    def resolve(
        self,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None: ...

    def upsert(
        self,
        descriptor: PromptDescriptor,
        override: PromptOverride,
    ) -> PromptOverride: ...

    def delete(
        self,
        *,
        ns: str,
        prompt_key: str,
        tag: str,
    ) -> None: ...

    def seed_if_necessary(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
    ) -> PromptOverride: ...


__all__ = [
    "PromptDescriptor",
    "PromptOverride",
    "PromptOverridesError",
    "PromptOverridesStore",
    "SectionDescriptor",
    "SectionOverride",
    "ToolDescriptor",
    "ToolOverride",
]


def _tool_contract_hash(tool: Tool[Any, Any]) -> str:
    description_hash = hash_text(tool.description)
    params_schema_hash = hash_json(schema(tool.params_type, extra="forbid"))
    result_schema_hash = hash_json(schema(tool.result_type, extra="ignore"))
    return hash_text(
        "::".join((description_hash, params_schema_hash, result_schema_hash))
    )


def hash_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def hash_json(value: object) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hash_text(canonical)
