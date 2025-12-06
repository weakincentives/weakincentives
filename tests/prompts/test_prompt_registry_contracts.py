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

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest

from weakincentives.dbc import dbc_enabled
from weakincentives.prompt import Section, SectionVisibility
from weakincentives.prompt.registry import PromptRegistry, SectionNode
from weakincentives.serde import clone as clone_dataclass

if TYPE_CHECKING:
    from weakincentives.prompt._types import SupportsDataclass


@dataclass
class ExampleParams:
    value: str = "example"


@dataclass
class OtherParams:
    level: int = 1


class ExampleSection(Section[ExampleParams]):
    def render(
        self,
        params: ExampleParams,
        depth: int,
        number: str,
        *,
        visibility: SectionVisibility | None = None,
    ) -> str:
        del number, visibility
        _ = self.key
        return f"example:{params.value}:{depth}"

    def placeholder_names(self) -> set[str]:
        _ = self.children
        return {"value"}

    def clone(self, **kwargs: object) -> ExampleSection:
        cloned_children = tuple(child.clone(**kwargs) for child in self.children)
        cloned_default = clone_dataclass(self.default_params)
        return ExampleSection(
            title=self.title,
            key=self.key,
            default_params=cloned_default,
            children=cloned_children,
        )


class OtherSection(Section[OtherParams]):
    def render(
        self,
        params: OtherParams,
        depth: int,
        number: str,
        *,
        visibility: SectionVisibility | None = None,
    ) -> str:
        del number, visibility
        _ = self.title
        return f"other:{params.level}:{depth}"

    def placeholder_names(self) -> set[str]:
        _ = self.children
        return {"level"}

    def clone(self, **kwargs: object) -> OtherSection:
        cloned_children = tuple(child.clone(**kwargs) for child in self.children)
        cloned_default = clone_dataclass(self.default_params)
        return OtherSection(
            title=self.title,
            key=self.key,
            default_params=cloned_default,
            children=cloned_children,
        )


class NoParamsSection(Section):
    def render(
        self,
        params: object,
        depth: int,
        number: str,
        *,
        visibility: SectionVisibility | None = None,
    ) -> str:
        del params, depth, number, visibility
        _ = self.key
        return "no-params"

    def clone(self, **kwargs: object) -> NoParamsSection:
        return NoParamsSection(title=self.title, key=self.key)


def _build_registry_with_sections() -> PromptRegistry:
    registry = PromptRegistry()
    child = OtherSection(title="Child", key="child")
    parent = ExampleSection(
        title="Parent",
        key="parent",
        default_params=ExampleParams(value="default"),
        children=[child],
    )
    registry.register_sections((cast("Section[SupportsDataclass]", parent),))
    no_params = NoParamsSection(title="Static", key="static")
    registry.register_section(
        cast("Section[SupportsDataclass]", no_params),
        path=("static",),
        depth=0,
    )
    return registry


def test_prompt_registry_invariants_allow_valid_usage() -> None:
    registry = PromptRegistry()
    child = OtherSection(title="Child", key="child")
    parent = ExampleSection(
        title="Parent",
        key="parent",
        default_params=ExampleParams(value="default"),
        children=[child],
    )
    sibling = OtherSection(title="Sibling", key="sibling")

    with dbc_enabled():
        registry.register_sections((cast("Section[SupportsDataclass]", parent),))
        registry.register_section(
            cast("Section[SupportsDataclass]", sibling),
            path=("sibling",),
            depth=0,
        )
        snapshot = registry.snapshot()

    default_params = cast("ExampleParams", snapshot.defaults_by_path["parent",])
    assert default_params.value == "default"
    assert snapshot.placeholders["parent",] == frozenset({"value"})
    assert snapshot.placeholders["parent", "child"] == frozenset({"level"})
    assert snapshot.tool_name_registry == {}
    assert snapshot.params_registry[ExampleParams][0].path == ("parent",)


def _get_section_path(
    registry: PromptRegistry, section_type: type[object]
) -> tuple[str, ...]:
    snapshot = registry.snapshot()
    for node in snapshot.sections:
        if isinstance(node.section, section_type):
            return node.path
    raise AssertionError(f"{section_type.__name__} not registered")


def _corrupt_defaults(registry: PromptRegistry) -> None:
    registry._defaults_by_path["ghost",] = ExampleParams(value="ghost")


def _corrupt_placeholders(registry: PromptRegistry) -> None:
    registry._placeholders["ghost",] = {"value"}


def _corrupt_tool_registry(registry: PromptRegistry) -> None:
    registry._tool_name_registry["ghost"] = ("ghost",)


def _corrupt_params_registry(registry: PromptRegistry) -> None:
    node = registry._section_nodes[0]
    registry._params_registry[OtherParams] = [node]


def _corrupt_defaults_for_no_params_section(registry: PromptRegistry) -> None:
    path = _get_section_path(registry, NoParamsSection)
    registry._defaults_by_path[path] = ExampleParams(value="ghost")


def _corrupt_defaults_type_mismatch(registry: PromptRegistry) -> None:
    path = _get_section_path(registry, ExampleSection)
    registry._defaults_by_path[path] = OtherParams(level=99)


def _corrupt_defaults_by_type_mismatch(registry: PromptRegistry) -> None:
    registry._defaults_by_type[ExampleParams] = OtherParams(level=42)


def _corrupt_params_registry_with_unknown_node(registry: PromptRegistry) -> None:
    snapshot = registry.snapshot()
    example_node = next(
        node for node in snapshot.sections if isinstance(node.section, ExampleSection)
    )
    ghost_node = SectionNode(
        section=example_node.section,
        depth=example_node.depth,
        path=("ghost",),
        number="999",
    )
    registry._params_registry[ExampleParams].append(ghost_node)


def _corrupt_params_registry_with_no_params_section(registry: PromptRegistry) -> None:
    snapshot = registry.snapshot()
    no_params_node = next(
        node for node in snapshot.sections if isinstance(node.section, NoParamsSection)
    )
    registry._params_registry[ExampleParams].append(no_params_node)


Mutation = Callable[[PromptRegistry], None]


@pytest.mark.parametrize(
    "mutator",
    (
        pytest.param(_corrupt_defaults, id="default-path"),
        pytest.param(_corrupt_placeholders, id="placeholders"),
        pytest.param(_corrupt_tool_registry, id="tools"),
        pytest.param(_corrupt_params_registry, id="params-registry"),
        pytest.param(_corrupt_defaults_for_no_params_section, id="defaults-no-params"),
        pytest.param(_corrupt_defaults_type_mismatch, id="defaults-type-mismatch"),
        pytest.param(
            _corrupt_defaults_by_type_mismatch, id="defaults-by-type-mismatch"
        ),
        pytest.param(
            _corrupt_params_registry_with_unknown_node,
            id="params-registry-unknown-node",
        ),
        pytest.param(
            _corrupt_params_registry_with_no_params_section,
            id="params-registry-no-params",
        ),
    ),
)
def test_prompt_registry_invariants_detect_corruption(mutator: Mutation) -> None:
    registry = _build_registry_with_sections()
    mutator(registry)

    with pytest.raises(AssertionError), dbc_enabled():
        registry.snapshot()
