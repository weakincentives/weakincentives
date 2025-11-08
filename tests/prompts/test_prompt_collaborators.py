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

from weakincentives.prompt import MarkdownSection, Prompt
from weakincentives.prompt.registry import PromptRegistry
from weakincentives.prompt.rendering import PromptRenderer


@dataclass
class _GreetingParams:
    subject: str


@dataclass
class _DefaultableParams:
    topic: str = "default"


def test_prompt_registry_snapshot_resolves_defaults() -> None:
    section = MarkdownSection[_DefaultableParams](
        title="Topic",
        template="Discuss ${topic} in detail.",
        key="topic",
        default_params=_DefaultableParams(topic="registry"),
    )
    registry = PromptRegistry(base_sections=[section])
    snapshot = registry.snapshot()
    node = snapshot.section_nodes[0]

    resolved = snapshot.resolve_section_params(node, {})

    assert isinstance(resolved, _DefaultableParams)
    assert resolved.topic == "registry"
    assert resolved is not section.default_params
    assert snapshot.placeholders[node.path] == {"topic"}
    assert registry.response_section is None
    direct_resolved = registry.resolve_section_params(node, {})
    assert isinstance(direct_resolved, _DefaultableParams)


def test_prompt_renderer_applies_overrides_and_tool_descriptions() -> None:
    prompt = Prompt(
        ns="tests.collaborators",
        key="renderer",
        sections=[
            MarkdownSection[_GreetingParams](
                title="Greeting",
                template="Greet ${subject} warmly.",
                key="greeting",
            )
        ],
    )

    registry = prompt._registry.snapshot()
    renderer = PromptRenderer(
        registry=registry,
        output_type=prompt._output_type,
        output_container=prompt._output_container,
        allow_extra_keys=prompt._allow_extra_keys,
        overrides={registry.section_nodes[0].path: "Custom body."},
    )

    params = renderer.collect_param_lookup((_GreetingParams(subject="agent"),))
    rendered = renderer.render(params)

    assert rendered.text == "## Greeting\n\nCustom body."
    assert rendered.tools == ()
