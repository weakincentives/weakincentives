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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptDescriptor,
    PromptTemplate,
    PromptValidationError,
    SectionNode,
)
from weakincentives.prompt.section import Section
from weakincentives.types import SupportsDataclass


@dataclass
class RootParams:
    title: str


@dataclass
class ChildParams:
    detail: str


@dataclass
class SiblingParams:
    note: str


@dataclass
class DuplicateParams:
    value: str


def test_prompt_initialization_flattens_sections_depth_first() -> None:
    child = MarkdownSection[ChildParams](
        title="Child",
        template="Child: ${detail}",
        key="child",
    )
    sibling = MarkdownSection[SiblingParams](
        title="Sibling",
        template="Sibling: ${note}",
        key="sibling",
    )
    root = MarkdownSection[RootParams](
        title="Root",
        template="Root: ${title}",
        key="root",
        children=[child, sibling],
    )

    prompt = PromptTemplate(
        ns="tests/prompts",
        key="prompt-init",
        name="demo",
        sections=[root],
    )

    sections = cast(tuple[SectionNode[Any], ...], prompt.sections)
    titles = [node.section.title for node in sections]
    depths = [node.depth for node in sections]
    paths = [node.path for node in sections]

    assert titles == ["Root", "Child", "Sibling"]
    assert depths == [0, 1, 1]
    assert paths == [
        ("root",),
        ("root", "child"),
        ("root", "sibling"),
    ]
    assert prompt.params_types == {RootParams, ChildParams, SiblingParams}
    assert prompt.name == "demo"


def test_prompt_requires_non_empty_key() -> None:
    section = MarkdownSection[RootParams](
        title="Root", template="Body: ${title}", key="root"
    )

    with pytest.raises(PromptValidationError):
        PromptTemplate(ns="tests/prompts", key="   ", sections=[section])


def test_prompt_requires_non_empty_namespace() -> None:
    section = MarkdownSection[RootParams](
        title="Root", template="Body: ${title}", key="root"
    )

    with pytest.raises(PromptValidationError):
        PromptTemplate(ns="   ", key="prompt-ns", sections=[section])


def test_prompt_allows_duplicate_param_dataclasses_and_shares_params() -> None:
    first = MarkdownSection[DuplicateParams](
        title="First",
        template="First: ${value}",
        key="first",
        default_params=DuplicateParams(value="alpha"),
    )
    second = MarkdownSection[DuplicateParams](
        title="Second",
        template="Second: ${value}",
        key="second",
        default_params=DuplicateParams(value="beta"),
    )

    template = PromptTemplate(
        ns="tests/prompts",
        key="duplicate-defaults",
        sections=[first, second],
    )

    rendered = Prompt(template).render()

    assert "First: alpha" in rendered.text
    assert "Second: beta" in rendered.text
    assert template.params_types == {DuplicateParams}


def test_prompt_reuses_provided_params_for_duplicate_sections() -> None:
    first = MarkdownSection[DuplicateParams](
        title="First",
        template="First: ${value}",
        key="first",
    )
    second = MarkdownSection[DuplicateParams](
        title="Second",
        template="Second: ${value}",
        key="second",
    )

    template = PromptTemplate(
        ns="tests/prompts",
        key="duplicate-shared",
        sections=[first, second],
    )

    rendered = Prompt(template).bind(DuplicateParams(value="shared")).render()

    assert "First: shared" in rendered.text
    assert "Second: shared" in rendered.text


def test_prompt_exposes_placeholders_from_registry_snapshot() -> None:
    section = MarkdownSection[RootParams](
        title="Root", template="Root: ${title}", key="root"
    )

    prompt = PromptTemplate(ns="tests/prompts", key="placeholder", sections=[section])

    assert prompt.placeholders == {("root",): frozenset({"title"})}


def test_prompt_duplicate_sections_share_type_defaults_when_missing_section_default() -> (
    None
):
    first = MarkdownSection[DuplicateParams](
        title="First",
        template="First: ${value}",
        key="first",
        default_params=DuplicateParams(value="alpha"),
    )
    second = MarkdownSection[DuplicateParams](
        title="Second",
        template="Second: ${value}",
        key="second",
    )

    template = PromptTemplate(
        ns="tests/prompts",
        key="duplicate-type-default",
        sections=[first, second],
    )

    rendered = Prompt(template).render()

    assert "First: alpha" in rendered.text
    assert "Second: alpha" in rendered.text


def test_prompt_validates_text_section_placeholders() -> None:
    @dataclass
    class PlaceholderParams:
        value: str

    section = MarkdownSection[PlaceholderParams](
        title="Invalid",
        template="Missing ${oops}",
        key="invalid",
    )

    with pytest.raises(PromptValidationError) as exc:
        PromptTemplate(
            ns="tests/prompts", key="invalid-placeholder", sections=[section]
        )

    assert isinstance(exc.value, PromptValidationError)
    assert exc.value.placeholder == "oops"
    assert exc.value.section_path == ("invalid",)
    assert exc.value.dataclass_type is PlaceholderParams


def test_text_section_rejects_non_section_children() -> None:
    @dataclass
    class ParentParams:
        value: str

    with pytest.raises(TypeError) as exc:
        MarkdownSection[ParentParams](
            title="Parent",
            template="${value}",
            key="parent",
            children=cast(
                Sequence[Section[SupportsDataclass]],
                ["not a section"],
            ),
        )

    assert "Section instances" in str(exc.value)


def test_prompt_template_is_immutable() -> None:
    section = MarkdownSection[RootParams](
        title="Root", template="Root: ${title}", key="root"
    )

    prompt = PromptTemplate(ns="tests/prompts", key="immutable", sections=[section])

    with pytest.raises(AttributeError):
        prompt.ns = "changed"
    with pytest.raises(AttributeError):
        prompt.placeholders = {}  # type: ignore[misc]


def test_prompt_descriptor_cached_on_first_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    section = MarkdownSection[RootParams](
        title="Root", template="Root: ${title}", key="root"
    )

    template = PromptTemplate(ns="tests/prompts", key="descriptor", sections=[section])

    # First access triggers lazy creation and caches the descriptor
    first_descriptor = template.descriptor
    assert first_descriptor.ns == "tests/prompts"

    def _fail(
        cls: type[PromptDescriptor], prompt_like: PromptTemplate[SupportsDataclass]
    ) -> PromptDescriptor:
        raise AssertionError("from_prompt should not be invoked after first access")

    monkeypatch.setattr(PromptDescriptor, "from_prompt", classmethod(_fail))

    # Subsequent accesses should use the cached descriptor
    prompt = Prompt(template).bind(RootParams(title="hello"))
    rendered = prompt.render()

    assert template.descriptor is first_descriptor
    assert prompt.descriptor is template.descriptor
    assert "Root: hello" in rendered.text


def test_prompt_missing_ns_raises_type_error() -> None:
    section = MarkdownSection[RootParams](
        title="Root", template="Root: ${title}", key="root"
    )

    with pytest.raises(TypeError, match="missing required argument: 'ns'"):
        PromptTemplate(key="my-key", sections=[section])  # type: ignore[call-arg]


def test_prompt_missing_key_raises_type_error() -> None:
    section = MarkdownSection[RootParams](
        title="Root", template="Root: ${title}", key="root"
    )

    with pytest.raises(TypeError, match="missing required argument: 'key'"):
        PromptTemplate(ns="my-ns", sections=[section])  # type: ignore[call-arg]


class TestPromptResourceLifecycle:
    """Tests for prompt-bound resource lifecycle."""

    def test_accessing_resources_outside_context_raises_runtime_error(self) -> None:
        """Calling get() outside context manager raises RuntimeError."""
        template = PromptTemplate(ns="tests", key="resource-test")
        prompt = Prompt(template).bind(resources={})

        # Accessing prompt.resources is fine - it returns PromptResources
        resources = prompt.resources
        assert resources is not None

        # But calling get() outside context raises
        with pytest.raises(
            RuntimeError,
            match="Resources accessed outside context",
        ):
            resources.get(object)

    def test_entering_context_twice_raises_runtime_error(self) -> None:
        """Entering resource context twice raises RuntimeError."""
        template = PromptTemplate(ns="tests", key="resource-test")
        prompt = Prompt(template)

        with prompt.resources:
            with pytest.raises(RuntimeError, match="context already entered"):
                prompt.resources.__enter__()

    def test_binding_resources_multiple_times_merges_them(self) -> None:
        """Binding resources multiple times should merge the registries."""

        class Resource1:
            pass

        class Resource2:
            pass

        res1 = Resource1()
        res2 = Resource2()

        template = PromptTemplate(ns="tests", key="resource-test")
        prompt = Prompt(template)
        prompt = prompt.bind(resources={Resource1: res1})
        prompt = prompt.bind(resources={Resource2: res2})

        with prompt.resources:
            assert prompt.resources.get(Resource1) is res1
            assert prompt.resources.get(Resource2) is res2

    def test_resources_property_available_within_context(self) -> None:
        """Resources are available within the context manager."""
        from weakincentives.prompt import PromptResources

        template = PromptTemplate(ns="tests", key="resource-test")
        prompt = Prompt(template).bind(resources={})

        # prompt.resources returns PromptResources
        assert isinstance(prompt.resources, PromptResources)

        # get() works inside context
        with prompt.resources:
            # Should not raise - context is active
            prompt.resources.get_optional(object)

    def test_child_section_resources_collected(self) -> None:
        """Resources from child sections are collected into prompt resources."""
        from typing import Self

        from weakincentives.resources import ResourceRegistry

        class ChildResource:
            pass

        child_res = ChildResource()

        # Create a section that provides resources
        class SectionWithResources(Section[RootParams]):
            def render(
                self,
                params: RootParams,
                heading_level: int,
                visibility: object = None,
            ) -> str:
                return "content"

            def resources(self) -> ResourceRegistry:
                return ResourceRegistry.build({ChildResource: child_res})

            def clone(self, **kwargs: object) -> Self:
                return cast(Self, self)

        parent_section = MarkdownSection[RootParams](
            title="Parent",
            key="parent",
            template="Parent: ${title}",
            children=(
                SectionWithResources(
                    title="Child",
                    key="child",
                ),
            ),
        )

        template = PromptTemplate(
            ns="tests", key="child-resources-test", sections=[parent_section]
        )
        prompt = Prompt(template).bind(RootParams(title="test"))

        with prompt.resources:
            # Child section resources should be accessible
            assert prompt.resources.get(ChildResource) is child_res

    def test_exit_before_enter_is_safe(self) -> None:
        """Calling __exit__ without __enter__ is a no-op."""
        template = PromptTemplate(ns="tests", key="resource-test")
        prompt = Prompt(template)

        # Calling __exit__ without entering should be safe (no-op)
        prompt.resources.__exit__(None, None, None)

    def test_tool_scope_provides_resolver(self) -> None:
        """tool_scope() provides a ResourceResolver within context."""

        class ToolScopedResource:
            pass

        instance = ToolScopedResource()
        template = PromptTemplate(ns="tests", key="resource-test")
        prompt = Prompt(template).bind(resources={ToolScopedResource: instance})

        with prompt.resources:
            with prompt.resources.tool_scope() as resolver:
                # Should be able to resolve resources within tool scope
                assert resolver.get(ToolScopedResource) is instance

    def test_get_optional_factory_binding_within_context(self) -> None:
        """get_optional() resolves factory-constructed resources within context."""
        from weakincentives.resources import Binding

        class FactoryResource:
            pass

        created_instance = FactoryResource()
        # Factory binding (not pre-provided) - passed via dict
        binding = Binding(FactoryResource, provider=lambda _: created_instance)

        template = PromptTemplate(ns="tests", key="factory-test")
        prompt = Prompt(template).bind(resources={FactoryResource: binding})

        with prompt.resources:
            # get_optional should resolve via context for factory bindings
            resolved = prompt.resources.get_optional(FactoryResource)
            assert resolved is created_instance
