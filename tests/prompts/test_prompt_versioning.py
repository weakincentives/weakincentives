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

from weakincentives.prompts import Prompt, Section, TextSection
from weakincentives.prompts.versioning import (
    PromptDescriptor,
    PromptOverride,
    PromptVersionStore,
    SectionDescriptor,
)


@dataclass
class _GreetingParams:
    subject: str


class _StaticSection(Section[_GreetingParams]):
    def render(self, params: _GreetingParams, depth: int) -> str:
        return f"Depth {depth}: {params.subject}"


def _build_prompt() -> Prompt:
    return Prompt(
        key="versioned-greeting",
        name="greeting",
        sections=[
            TextSection[_GreetingParams](
                title="Greeting",
                body="Greet ${subject} warmly.",
            )
        ],
    )


def test_prompt_descriptor_hashes_text_sections() -> None:
    prompt = _build_prompt()

    descriptor = PromptDescriptor.from_prompt(prompt)

    assert descriptor.key == "versioned-greeting"
    assert descriptor.sections == [
        SectionDescriptor(
            path=("greeting",),
            content_hash=sha256(b"Greet ${subject} warmly.").hexdigest(),
        )
    ]


def test_prompt_descriptor_includes_response_format_section() -> None:
    @dataclass
    class Summary:
        topic: str

    prompt = Prompt[Summary](
        key="versioned-summary",
        sections=[
            TextSection[_GreetingParams](
                title="Task",
                body="Summarize ${subject} succinctly.",
            )
        ],
    )

    descriptor = PromptDescriptor.from_prompt(prompt)
    paths = [section.path for section in descriptor.sections]

    assert ("task",) in paths
    assert ("response-format",) in paths


def test_prompt_descriptor_ignores_non_hash_sections() -> None:
    section = _StaticSection(title="Static")
    prompt = Prompt(key="versioned-static", sections=[section])

    descriptor = PromptDescriptor.from_prompt(prompt)

    assert descriptor.sections == []


class _RecordingStore(PromptVersionStore):
    def __init__(self, override: PromptOverride | None) -> None:
        self.override = override
        self.calls: list[tuple[PromptDescriptor, str]] = []

    def resolve(
        self,
        description: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None:
        self.calls.append((description, tag))
        return self.override


def test_prompt_render_with_overrides_applies_matching_sections() -> None:
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    path = descriptor.sections[0].path

    override = PromptOverride(
        prompt_key=descriptor.key,
        tag="experiment",
        overrides={path: "Cheer loudly for ${subject}."},
    )
    store = _RecordingStore(override)

    rendered = prompt.render_with_overrides(
        _GreetingParams(subject="Operators"),
        version_store=store,
        tag="experiment",
    )

    assert "Cheer loudly for Operators." in rendered.text
    assert store.calls == [(descriptor, "experiment")]


def test_prompt_render_with_overrides_ignores_non_matching_override() -> None:
    prompt = _build_prompt()

    override = PromptOverride(
        prompt_key="other-prompt",
        tag="latest",
        overrides={("other",): "Ignore this."},
    )
    store = _RecordingStore(override)

    rendered = prompt.render_with_overrides(
        _GreetingParams(subject="Operators"),
        version_store=store,
    )

    assert "Greet Operators warmly." in rendered.text


def test_prompt_render_with_overrides_handles_missing_override() -> None:
    prompt = _build_prompt()
    store = _RecordingStore(None)

    rendered = prompt.render_with_overrides(
        _GreetingParams(subject="Operators"),
        version_store=store,
    )

    assert "Greet Operators warmly." in rendered.text
    assert store.calls[0][1] == "latest"
