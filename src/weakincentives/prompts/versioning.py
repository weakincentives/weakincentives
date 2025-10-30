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

from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Final, Protocol

if TYPE_CHECKING:
    from .prompt import Prompt


@dataclass(slots=True)
class SectionDescriptor:
    """Hash metadata for a single section within a prompt."""

    path: tuple[str, ...]
    content_hash: str


@dataclass(slots=True)
class PromptDescriptor:
    """Stable metadata describing a prompt and its hash-aware sections."""

    key: str
    sections: list[SectionDescriptor]

    @classmethod
    def from_prompt(cls, prompt: Prompt[Any]) -> PromptDescriptor:
        sections: list[SectionDescriptor] = []
        for node in prompt.sections:
            template = node.section.original_body_template()
            if template is None:
                continue
            content_hash = sha256(template.encode("utf-8")).hexdigest()
            sections.append(SectionDescriptor(node.path, content_hash))
        return cls(prompt.key, sections)


@dataclass(slots=True)
class PromptOverride:
    """Runtime replacements for prompt sections validated by a version store."""

    prompt_key: str
    tag: str
    overrides: dict[tuple[str, ...], str]


class PromptVersionStore(Protocol):
    """Lookup interface for resolving prompt overrides at render time."""

    def resolve(
        self,
        description: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None: ...


__all__: Final[list[str]] = [
    "PromptDescriptor",
    "PromptOverride",
    "PromptVersionStore",
    "SectionDescriptor",
]
