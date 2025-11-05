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

from weakincentives.prompt import MarkdownSection, Prompt, Section, Tool
from weakincentives.prompt.versioning import (
    PromptDescriptor,
    PromptOverride,
    PromptOverridesStore,
    SectionDescriptor,
    SectionOverride,
    ToolDescriptor,
    ToolOverride,
    hash_json,
    hash_text,
)
from weakincentives.serde.dataclass_serde import schema


@dataclass
class _GreetingParams:
    subject: str


class _StaticSection(Section[_GreetingParams]):
    def render(self, params: _GreetingParams, depth: int) -> str:
        return f"Depth {depth}: {params.subject}"


def _build_prompt() -> Prompt:
    return Prompt(
        ns="tests/versioning",
        key="versioned-greeting",
        name="greeting",
        sections=[
            MarkdownSection[_GreetingParams](
                title="Greeting",
                template="Greet ${subject} warmly.",
                key="greeting",
            )
        ],
    )


@dataclass
class _LookupResult:
    success: bool


@dataclass
class _LookupItem:
    value: str


def _build_tool_prompt(
    *, accepts_overrides: bool = True
) -> tuple[Prompt, Tool[_GreetingParams, _LookupResult]]:
    tool = Tool[_GreetingParams, _LookupResult](
        name="greeter",
        description="Greet the provided subject in a friendly way.",
        handler=None,
        accepts_overrides=accepts_overrides,
    )
    prompt = Prompt(
        ns="tests/versioning",
        key="versioned-greeting-tools",
        name="greeting-tools",
        sections=[
            MarkdownSection[_GreetingParams](
                title="Greeting",
                template="Greet ${subject} warmly.",
                key="greeting",
                tools=[tool],
            )
        ],
    )
    return prompt, tool


def _build_sequence_tool_prompt(
    *, accepts_overrides: bool = True
) -> tuple[Prompt, Tool[_GreetingParams, tuple[_LookupItem, ...]]]:
    tool = Tool[_GreetingParams, tuple[_LookupItem, ...]](
        name="greeter_sequence",
        description="Greet and capture multiple responses.",
        handler=None,
        accepts_overrides=accepts_overrides,
    )
    prompt = Prompt(
        ns="tests/versioning",
        key="versioned-greeting-tools-seq",
        name="greeting-tools-seq",
        sections=[
            MarkdownSection[_GreetingParams](
                title="Greeting",
                template="Greet ${subject} warmly.",
                key="greeting",
                tools=[tool],
            )
        ],
    )
    return prompt, tool


def test_prompt_descriptor_hashes_text_sections() -> None:
    prompt = _build_prompt()

    descriptor = PromptDescriptor.from_prompt(prompt)

    assert descriptor.ns == "tests/versioning"
    assert descriptor.key == "versioned-greeting"
    assert descriptor.sections == [
        SectionDescriptor(
            path=("greeting",),
            content_hash=sha256(b"Greet ${subject} warmly.").hexdigest(),
        )
    ]
    assert descriptor.tools == []


def test_prompt_descriptor_excludes_response_format_section() -> None:
    @dataclass
    class Summary:
        topic: str

    prompt = Prompt[Summary](
        ns="tests/versioning",
        key="versioned-summary",
        sections=[
            MarkdownSection[_GreetingParams](
                title="Task",
                template="Summarize ${subject} succinctly.",
                key="task",
            )
        ],
    )

    descriptor = PromptDescriptor.from_prompt(prompt)
    paths = [section.path for section in descriptor.sections]

    assert ("task",) in paths
    assert ("response-format",) not in paths
    assert descriptor.tools == []
    assert descriptor.ns == "tests/versioning"


def test_prompt_descriptor_ignores_non_hash_sections() -> None:
    section = _StaticSection(title="Static", key="static")
    prompt = Prompt(
        ns="tests/versioning",
        key="versioned-static",
        sections=[section],
    )

    descriptor = PromptDescriptor.from_prompt(prompt)

    assert descriptor.sections == []
    assert descriptor.tools == []


def test_prompt_descriptor_collects_tools() -> None:
    prompt, tool = _build_tool_prompt()

    descriptor = PromptDescriptor.from_prompt(prompt)

    description_hash = hash_text(tool.description)
    params_schema_hash = hash_json(schema(tool.params_type, extra="forbid"))
    result_schema_hash = hash_json(schema(tool.result_type, extra="ignore"))
    expected_contract = hash_text(
        "::".join((description_hash, params_schema_hash, result_schema_hash))
    )

    assert descriptor.tools == [
        ToolDescriptor(
            path=("greeting",), name="greeter", contract_hash=expected_contract
        )
    ]


def test_prompt_descriptor_collects_sequence_tools() -> None:
    prompt, tool = _build_sequence_tool_prompt()

    descriptor = PromptDescriptor.from_prompt(prompt)

    description_hash = hash_text(tool.description)
    params_schema_hash = hash_json(schema(tool.params_type, extra="forbid"))
    item_schema = schema(tool.result_type, extra="ignore")
    result_schema = {
        "title": f"{tool.result_type.__name__}List",
        "type": "array",
        "items": item_schema,
    }
    result_schema_hash = hash_json(result_schema)
    expected_contract = hash_text(
        "::".join((description_hash, params_schema_hash, result_schema_hash))
    )

    assert descriptor.tools == [
        ToolDescriptor(
            path=("greeting",),
            name="greeter_sequence",
            contract_hash=expected_contract,
        )
    ]


def test_prompt_descriptor_skips_tools_without_override_acceptance() -> None:
    prompt, _ = _build_tool_prompt(accepts_overrides=False)

    descriptor = PromptDescriptor.from_prompt(prompt)

    assert descriptor.tools == []


class _RecordingOverridesStore(PromptOverridesStore):
    def __init__(self, override: PromptOverride | None) -> None:
        self.override = override
        self.calls: list[tuple[PromptDescriptor, str]] = []

    def resolve(
        self,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None:
        self.calls.append((descriptor, tag))
        return self.override

    def upsert(
        self,
        descriptor: PromptDescriptor,
        override: PromptOverride,
    ) -> PromptOverride:
        raise NotImplementedError

    def delete(
        self,
        *,
        ns: str,
        prompt_key: str,
        tag: str,
    ) -> None:
        raise NotImplementedError

    def seed_if_necessary(
        self,
        prompt: Prompt,
        *,
        tag: str = "latest",
    ) -> PromptOverride:
        raise NotImplementedError


def test_prompt_render_with_overrides_applies_matching_sections() -> None:
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    section = descriptor.sections[0]
    path = section.path

    override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="experiment",
        sections={
            path: SectionOverride(
                expected_hash=section.content_hash,
                body="Cheer loudly for ${subject}.",
            )
        },
    )
    store = _RecordingOverridesStore(override)

    rendered = prompt.render_with_overrides(
        _GreetingParams(subject="Operators"),
        overrides_store=store,
        tag="experiment",
    )

    assert "Cheer loudly for Operators." in rendered.text
    assert store.calls == [(descriptor, "experiment")]
    assert rendered.tool_param_descriptions == {}


def test_prompt_render_with_overrides_respects_section_acceptance() -> None:
    prompt = _build_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    section = descriptor.sections[0]
    path = section.path

    override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="experiment",
        sections={
            path: SectionOverride(
                expected_hash=section.content_hash,
                body="Cheer loudly for ${subject}.",
            )
        },
    )
    store = _RecordingOverridesStore(override)

    prompt.sections[0].section.accepts_overrides = False  # type: ignore[misc]

    rendered = prompt.render_with_overrides(
        _GreetingParams(subject="Operators"),
        overrides_store=store,
        tag="experiment",
    )

    assert "Greet Operators warmly." in rendered.text


def test_prompt_render_with_overrides_ignores_non_matching_override() -> None:
    prompt = _build_prompt()

    override = PromptOverride(
        ns="tests/versioning",
        prompt_key="other-prompt",
        tag="latest",
        sections={
            ("other",): SectionOverride(
                expected_hash="deadbeef",
                body="Ignore this.",
            )
        },
    )
    store = _RecordingOverridesStore(override)

    rendered = prompt.render_with_overrides(
        _GreetingParams(subject="Operators"),
        overrides_store=store,
    )

    assert "Greet Operators warmly." in rendered.text


def test_prompt_render_with_overrides_handles_missing_override() -> None:
    prompt = _build_prompt()
    store = _RecordingOverridesStore(None)

    rendered = prompt.render_with_overrides(
        _GreetingParams(subject="Operators"),
        overrides_store=store,
    )

    assert "Greet Operators warmly." in rendered.text
    assert store.calls[0][1] == "latest"


def test_prompt_render_with_tool_overrides_updates_description() -> None:
    prompt, tool = _build_tool_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    contract_hash = descriptor.tools[0].contract_hash

    override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="latest",
        sections={},
        tool_overrides={
            tool.name: ToolOverride(
                name=tool.name,
                expected_contract_hash=contract_hash,
                description="Offer a celebratory greeting for the subject.",
                param_descriptions={"subject": "Name of the person to greet."},
            )
        },
    )
    store = _RecordingOverridesStore(override)

    rendered = prompt.render_with_overrides(
        _GreetingParams(subject="Operators"),
        overrides_store=store,
    )

    assert (
        rendered.tools[0].description == "Offer a celebratory greeting for the subject."
    )
    assert rendered.tool_param_descriptions == {
        tool.name: {"subject": "Name of the person to greet."}
    }


def test_prompt_render_with_tool_overrides_respects_acceptance() -> None:
    prompt, tool = _build_tool_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)
    contract_hash = descriptor.tools[0].contract_hash

    override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="latest",
        sections={},
        tool_overrides={
            tool.name: ToolOverride(
                name=tool.name,
                expected_contract_hash=contract_hash,
                description="Offer a celebratory greeting for the subject.",
            )
        },
    )
    store = _RecordingOverridesStore(override)

    prompt.sections[0].section.tools()[0].accepts_overrides = False  # type: ignore[misc]

    rendered = prompt.render_with_overrides(
        _GreetingParams(subject="Operators"),
        overrides_store=store,
    )

    assert rendered.tools[0].description == tool.description


def test_prompt_render_with_tool_override_rejects_mismatched_contract() -> None:
    prompt, tool = _build_tool_prompt()
    descriptor = PromptDescriptor.from_prompt(prompt)

    override = PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="latest",
        sections={},
        tool_overrides={
            tool.name: ToolOverride(
                name=tool.name,
                expected_contract_hash="not-a-match",
                description="This override should not apply.",
            )
        },
    )
    store = _RecordingOverridesStore(override)

    rendered = prompt.render_with_overrides(
        _GreetingParams(subject="Operators"),
        overrides_store=store,
    )

    assert rendered.tools[0].description == tool.description
    assert rendered.tool_param_descriptions == {}


def test_prompt_override_tool_default_factory_is_isolated() -> None:
    baseline = PromptOverride(ns="ns", prompt_key="key", tag="latest")
    successor = PromptOverride(ns="ns", prompt_key="key", tag="next")

    assert baseline.tool_overrides == {}
    assert successor.tool_overrides == {}
    successor.tool_overrides["example"] = ToolOverride(
        name="example",
        expected_contract_hash="hash",
    )

    assert "example" not in baseline.tool_overrides
