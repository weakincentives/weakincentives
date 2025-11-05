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

from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    clear_registry,
    get_prompt,
    iter_prompts,
    register_prompt,
    unregister_prompt,
)


@dataclass
class _Params:
    value: str


def test_registry_registers_and_clears() -> None:
    clear_registry()
    prompt = Prompt(
        ns="tests",
        key="demo",
        sections=[
            MarkdownSection[_Params](
                title="Body",
                template="${value}",
                key="body",
            )
        ],
    )

    register_prompt(prompt)
    assert get_prompt("tests", "demo") is prompt
    assert list(iter_prompts()) == [prompt]

    unregister_prompt("tests", "demo")
    assert get_prompt("tests", "demo") is None

    clear_registry()
    assert list(iter_prompts()) == []
