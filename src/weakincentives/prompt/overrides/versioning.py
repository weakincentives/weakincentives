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
import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Protocol, TypeVar, overload

from ...serde.schema import schema
from .._types import SupportsDataclass, SupportsToolResult
from ..tool import Tool


def _section_override_mapping_factory() -> dict[tuple[str, ...], SectionOverride]:
    return {}


def _tool_override_mapping_factory() -> dict[str, ToolOverride]:
    return {}


def _param_description_mapping_factory() -> dict[str, str]:
    return {}


class SectionLike(Protocol):
    def original_body_template(self) -> str | None: ...

    def tools(self) -> tuple[Tool[SupportsDataclass, SupportsToolResult], ...]: ...

    accepts_overrides: bool


class SectionNodeLike(Protocol):
    path: tuple[str, ...]
    section: SectionLike


class PromptLike(Protocol):
    ns: str
    key: str

    @property
    def sections(self) -> tuple[SectionNodeLike, ...]: ...


_HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

_HexDigestT = TypeVar("_HexDigestT", bound="HexDigest")


class HexDigest(str):
    """A validated lowercase hexadecimal SHA-256 digest."""

    __slots__ = ()

    def __new__(cls: type[_HexDigestT], value: object) -> _HexDigestT:
        if not isinstance(value, str):
            msg = "HexDigest value must be a string."
            raise TypeError(msg)
        if not _HEX_DIGEST_RE.fullmatch(value):
            msg = f"Invalid hex digest value: {value!r}"
            raise ValueError(msg)
        return str.__new__(cls, value)


@overload
def ensure_hex_digest(value: HexDigest, *, field_name: str) -> HexDigest: ...


@overload
def ensure_hex_digest(value: str, *, field_name: str) -> HexDigest: ...


def ensure_hex_digest(value: object, *, field_name: str) -> HexDigest:
    """Normalize an object to a :class:`HexDigest` with helpful errors."""

    if isinstance(value, HexDigest):
        return value
    if isinstance(value, str):
        try:
            return HexDigest(value)
        except ValueError as error:
            msg = f"{field_name} must be a 64 character lowercase hex digest."
            raise PromptOverridesError(msg) from error
    msg = f"{field_name} must be a string."
    raise PromptOverridesError(msg)


@dataclass(slots=True)
class SectionDescriptor:
    """Hash metadata for a single section within a prompt."""

    path: tuple[str, ...]
    content_hash: HexDigest


@dataclass(slots=True)
class ToolDescriptor:
    """Stable metadata describing a tool exposed by a prompt."""

    path: tuple[str, ...]
    name: str
    contract_hash: HexDigest


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
            if getattr(node.section, "accepts_overrides", True):
                template = node.section.original_body_template()
                if template is not None:
                    content_hash = hash_text(template)
                    sections.append(SectionDescriptor(node.path, content_hash))
            tool_descriptors = [
                ToolDescriptor(
                    path=node.path,
                    name=tool.name,
                    contract_hash=_tool_contract_hash(tool),
                )
                for tool in node.section.tools()
                if tool.accepts_overrides
            ]
            tools.extend(tool_descriptors)
        return cls(prompt.ns, prompt.key, sections, tools)


@dataclass(slots=True)
class SectionOverride:
    """Override payload for a prompt section validated by hash."""

    expected_hash: HexDigest
    body: str


@dataclass(slots=True)
class ToolOverride:
    """Description overrides validated against a tool contract hash."""

    name: str
    expected_contract_hash: HexDigest
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
    "HexDigest",
    "PromptDescriptor",
    "PromptOverride",
    "PromptOverridesError",
    "PromptOverridesStore",
    "SectionDescriptor",
    "SectionOverride",
    "ToolDescriptor",
    "ToolOverride",
    "ensure_hex_digest",
]


def _tool_contract_hash(tool: Tool[SupportsDataclass, SupportsToolResult]) -> HexDigest:
    description_hash = hash_text(tool.description)
    params_schema_hash = hash_json(schema(tool.params_type, extra="forbid"))
    if getattr(tool, "result_container", "object") == "array":
        item_schema = schema(tool.result_type, extra="ignore")
        result_schema = {
            "title": f"{tool.result_type.__name__}List",
            "type": "array",
            "items": item_schema,
        }
    else:
        result_schema = schema(tool.result_type, extra="ignore")
    result_schema_hash = hash_json(result_schema)
    return hash_text(
        "::".join((description_hash, params_schema_hash, result_schema_hash))
    )


def hash_text(value: str) -> HexDigest:
    return HexDigest(sha256(value.encode("utf-8")).hexdigest())


def hash_json(value: object) -> HexDigest:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hash_text(canonical)
