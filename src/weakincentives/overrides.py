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

"""Drift-tracked prompt overrides.

Iterating on a prompt without redeploying code is a common operations
problem: ops want to tweak a section's body, but they also want to know
when the source has moved on so the override stops applying. This
module records overrides keyed by a content hash of the section being
modified; when the source changes, the hash changes and the override
is dropped automatically.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import final

from weakincentives.core import (
    MarkdownSection,
    Prompt,
    Section,
)

__all__ = [
    "OverrideStore",
    "PromptDescriptor",
    "SectionDescriptor",
    "SectionOverride",
    "apply_overrides",
    "describe_prompt",
]


@final
@dataclass(frozen=True, slots=True)
class SectionDescriptor:
    """Identity record for a single :class:`Section`."""

    key: str
    content_hash: str


@final
@dataclass(frozen=True, slots=True)
class PromptDescriptor:
    """Identity record for a whole :class:`Prompt`."""

    ns: str
    key: str
    sections: tuple[SectionDescriptor, ...]

    def get(self, section_key: str) -> SectionDescriptor | None:
        for section in self.sections:
            if section.key == section_key:
                return section
        return None


@final
@dataclass(frozen=True, slots=True)
class SectionOverride:
    """A replacement template body, fingerprinted against the original."""

    section_key: str
    template: str
    expected_hash: str


# =============================================================================
# Description
# =============================================================================


def describe_prompt(prompt: Prompt) -> PromptDescriptor:
    """Build a :class:`PromptDescriptor` for ``prompt`` and every nested section."""
    descriptors: list[SectionDescriptor] = []
    for section in prompt.sections:
        _walk_descriptors(section, descriptors)
    return PromptDescriptor(ns=prompt.ns, key=prompt.key, sections=tuple(descriptors))


def _walk_descriptors(section: Section[object], out: list[SectionDescriptor]) -> None:
    out.append(_describe_section(section))
    for child in section.children:
        _walk_descriptors(child, out)


def _describe_section(section: Section[object]) -> SectionDescriptor:
    body = ""
    if isinstance(section, MarkdownSection):
        body = section.template
    digest = hashlib.sha256(f"{section.title}\n{body}".encode()).hexdigest()
    return SectionDescriptor(key=section.key, content_hash=digest)


# =============================================================================
# Store
# =============================================================================


@dataclass(slots=True)
class OverrideStore:
    """In-memory store of :class:`SectionOverride` records.

    Overrides are filtered against the current :class:`PromptDescriptor`
    on every :meth:`resolve`; entries whose ``expected_hash`` no longer
    matches the section are dropped silently.
    """

    _by_key: dict[str, SectionOverride] = field(default_factory=lambda: {}, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def upsert(self, override: SectionOverride) -> None:
        with self._lock:
            self._by_key[override.section_key] = override

    def remove(self, section_key: str) -> None:
        with self._lock:
            _ = self._by_key.pop(section_key, None)

    def all(self) -> tuple[SectionOverride, ...]:
        with self._lock:
            return tuple(self._by_key.values())

    def resolve(self, descriptor: PromptDescriptor) -> Mapping[str, SectionOverride]:
        live: dict[str, SectionOverride] = {}
        with self._lock:
            entries = dict(self._by_key)
        for section in descriptor.sections:
            override = entries.get(section.key)
            if override is None:
                continue
            if override.expected_hash != section.content_hash:
                continue
            live[section.key] = override
        return live


# =============================================================================
# Apply
# =============================================================================


def apply_overrides(prompt: Prompt, store: OverrideStore) -> Prompt:
    """Return a copy of ``prompt`` with live overrides applied."""
    descriptor = describe_prompt(prompt)
    overrides = store.resolve(descriptor)
    if not overrides:
        return prompt
    return Prompt(
        ns=prompt.ns,
        key=prompt.key,
        sections=tuple(
            _apply_to_section(section, overrides) for section in prompt.sections
        ),
        params=prompt.params,
        output_type=prompt.output_type,
    )


def _apply_to_section(
    section: Section[object], overrides: Mapping[str, SectionOverride]
) -> Section[object]:
    children = tuple(_apply_to_section(child, overrides) for child in section.children)
    override = overrides.get(section.key)
    if override is not None and isinstance(section, MarkdownSection):
        return MarkdownSection[object](
            title=section.title,
            key=section.key,
            template=override.template,
            params_type=section.params_type,
            default_params=section.default_params,
            children=children,
            tools=section.tools,
            enabled=section.enabled,
            visibility=section.visibility,
            summary=section.summary,
        )
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
    return section  # pragma: no cover - non-markdown sections aren't overridable
