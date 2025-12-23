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
from typing import Any

from weakincentives.contrib.overrides import filter_override_for_descriptor
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    Section,
    SectionVisibility,
    Tool,
)
from weakincentives.prompt.overrides import (
    PromptDescriptor,
    PromptOverride,
    PromptOverridesStore,
    SectionDescriptor,
    SectionOverride,
    TaskExampleOverride,
    ToolDescriptor,
    ToolOverride,
    hash_json,
    hash_text,
)
from weakincentives.serde.schema import schema


@dataclass
class _GreetingParams:
    subject: str


class _StaticSection(Section[_GreetingParams]):
    def render(
        self,
        params: _GreetingParams,
        depth: int,
        number: str,
        *,
        visibility: SectionVisibility | None = None,
    ) -> str:
        del visibility
        return f"{self.title}: Depth {depth}: {params.subject} ({number})"

    def clone(self, **kwargs: object) -> _StaticSection:
        return _StaticSection(title=self.title, key=self.key)


def _build_prompt() -> PromptTemplate:
    return PromptTemplate(
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
) -> tuple[PromptTemplate, Tool[_GreetingParams, _LookupResult]]:
    tool = Tool[_GreetingParams, _LookupResult](
        name="greeter",
        description="Greet the provided subject in a friendly way.",
        handler=None,
        accepts_overrides=accepts_overrides,
    )
    prompt = PromptTemplate(
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
) -> tuple[PromptTemplate, Tool[_GreetingParams, tuple[_LookupItem, ...]]]:
    tool = Tool[_GreetingParams, tuple[_LookupItem, ...]](
        name="greeter_sequence",
        description="Greet and capture multiple responses.",
        handler=None,
        accepts_overrides=accepts_overrides,
    )
    prompt = PromptTemplate(
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
            content_hash=hash_text("Greet ${subject} warmly."),
            number="1",
        )
    ]
    assert descriptor.tools == []


def test_prompt_descriptor_ignores_non_hash_sections() -> None:
    section = _StaticSection(title="Static", key="static")
    prompt = PromptTemplate(
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
            path=("greeting",),
            name="greeter",
            contract_hash=expected_contract,
            example_hashes=(),
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
            example_hashes=(),
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
        if self.override is None:
            return None
        filtered_sections, filtered_tools = filter_override_for_descriptor(
            descriptor, self.override
        )
        if not filtered_sections and not filtered_tools:
            return None
        return PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=self.override.tag,
            sections=filtered_sections,
            tool_overrides=filtered_tools,
        )

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

    def seed(
        self,
        prompt: PromptTemplate[Any],
        *,
        tag: str = "latest",
    ) -> PromptOverride:
        raise NotImplementedError

    def store(
        self,
        prompt: PromptTemplate[Any],
        override: SectionOverride | ToolOverride | TaskExampleOverride,
        *,
        tag: str = "latest",
    ) -> PromptOverride:
        raise NotImplementedError


def test_prompt_render_applies_matching_sections() -> None:
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
                path=path,
                expected_hash=section.content_hash,
                body="Cheer loudly for ${subject}.",
            )
        },
    )
    store = _RecordingOverridesStore(override)

    rendered = (
        Prompt(
            prompt,
            overrides_store=store,
            overrides_tag="experiment",
        )
        .bind(_GreetingParams(subject="Operators"))
        .render()
    )

    assert "Cheer loudly for Operators." in rendered.text
    assert store.calls == [(descriptor, "experiment")]
    assert rendered.tool_param_descriptions == {}


def test_prompt_render_respects_section_acceptance() -> None:
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
                path=path,
                expected_hash=section.content_hash,
                body="Cheer loudly for ${subject}.",
            )
        },
    )
    store = _RecordingOverridesStore(override)

    prompt.sections[0].section.accepts_overrides = False  # type: ignore[union-attr]

    rendered = (
        Prompt(
            prompt,
            overrides_store=store,
            overrides_tag="experiment",
        )
        .bind(_GreetingParams(subject="Operators"))
        .render()
    )

    assert "Greet Operators warmly." in rendered.text


def test_prompt_render_ignores_non_matching_override() -> None:
    prompt = _build_prompt()

    override = PromptOverride(
        ns="tests/versioning",
        prompt_key="other-prompt",
        tag="latest",
        sections={
            ("other",): SectionOverride(
                path=("other",),
                expected_hash=hash_text("deadbeef"),
                body="Ignore this.",
            )
        },
    )
    store = _RecordingOverridesStore(override)

    rendered = (
        Prompt(
            prompt,
            overrides_store=store,
        )
        .bind(_GreetingParams(subject="Operators"))
        .render()
    )

    assert "Greet Operators warmly." in rendered.text


def test_prompt_render_handles_missing_override() -> None:
    prompt = _build_prompt()
    store = _RecordingOverridesStore(None)

    rendered = (
        Prompt(
            prompt,
            overrides_store=store,
        )
        .bind(_GreetingParams(subject="Operators"))
        .render()
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

    rendered = (
        Prompt(
            prompt,
            overrides_store=store,
        )
        .bind(_GreetingParams(subject="Operators"))
        .render()
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

    prompt.sections[0].section.tools()[0].accepts_overrides = False  # type: ignore[union-attr]

    rendered = (
        Prompt(
            prompt,
            overrides_store=store,
        )
        .bind(_GreetingParams(subject="Operators"))
        .render()
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
                expected_contract_hash=hash_text("not-a-match"),
                description="This override should not apply.",
            )
        },
    )
    store = _RecordingOverridesStore(override)

    rendered = (
        Prompt(
            prompt,
            overrides_store=store,
        )
        .bind(_GreetingParams(subject="Operators"))
        .render()
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
        expected_contract_hash=hash_text("hash"),
    )

    assert "example" not in baseline.tool_overrides


def test_tool_example_hash_handles_missing_input_attribute() -> None:
    """Test that tool example hashing handles examples without input attribute."""
    from weakincentives.prompt.overrides.versioning import _serialize_example_value

    # Test with None value (covering line 430)
    result = _serialize_example_value(None)
    assert result is None


# ---------------------------------------------------------------------------
# DbC Precondition Tests
# ---------------------------------------------------------------------------


def test_ensure_hex_digest_requires_field_name() -> None:
    """ensure_hex_digest requires non-empty field_name."""
    import pytest

    from weakincentives.dbc import dbc_enabled
    from weakincentives.prompt.overrides.versioning import ensure_hex_digest

    with dbc_enabled():
        with pytest.raises(AssertionError, match="field_name must be non-empty"):
            ensure_hex_digest("a" * 64, field_name="")


def test_descriptor_for_prompt_requires_ns_key() -> None:
    """descriptor_for_prompt requires prompt to have ns and key attributes."""
    import pytest

    from weakincentives.dbc import dbc_enabled
    from weakincentives.prompt.overrides.versioning import descriptor_for_prompt

    class _FakePrompt:
        sections: tuple[object, ...] = ()

    with dbc_enabled():
        with pytest.raises(AssertionError, match="prompt must have ns and key"):
            descriptor_for_prompt(_FakePrompt())  # type: ignore[arg-type]


def test_descriptor_for_prompt_requires_sections() -> None:
    """descriptor_for_prompt requires prompt to have sections attribute."""
    import pytest

    from weakincentives.dbc import dbc_enabled
    from weakincentives.prompt.overrides.versioning import descriptor_for_prompt

    class _FakePrompt:
        ns = "test"
        key = "test"

    with dbc_enabled():
        with pytest.raises(AssertionError, match="prompt must have sections"):
            descriptor_for_prompt(_FakePrompt())  # type: ignore[arg-type]


def test_hash_text_requires_string() -> None:
    """hash_text requires value to be a string."""
    import pytest

    from weakincentives.dbc import dbc_enabled

    with dbc_enabled():
        with pytest.raises(AssertionError, match="value must be a string"):
            hash_text(123)  # type: ignore[arg-type]
