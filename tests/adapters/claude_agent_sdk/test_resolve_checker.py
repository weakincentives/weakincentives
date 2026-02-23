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

"""Tests for resolve_checker: prompt-scoped resolution."""

from __future__ import annotations

from typing import cast

from weakincentives.adapters.claude_agent_sdk._task_completion import resolve_checker
from weakincentives.prompt import FileOutputChecker, Prompt, PromptTemplate
from weakincentives.prompt.protocols import PromptProtocol


class TestResolveChecker:
    """Tests for resolve_checker resolution logic."""

    def test_returns_none_when_prompt_has_no_checker(self) -> None:
        """No checker on prompt => None."""
        prompt: Prompt[object] = Prompt(PromptTemplate(ns="test", key="test"))
        prompt._activate_scope()

        result = resolve_checker(prompt=cast("PromptProtocol[object]", prompt))

        assert result is None

    def test_returns_prompt_checker(self) -> None:
        """Checker on prompt => prompt checker."""
        checker = FileOutputChecker(files=("output.txt",))
        prompt: Prompt[object] = Prompt(
            PromptTemplate(ns="test", key="test", task_completion_checker=checker)
        )
        prompt._activate_scope()

        result = resolve_checker(prompt=cast("PromptProtocol[object]", prompt))

        assert result is checker

    def test_returns_none_when_prompt_is_none(self) -> None:
        """Prompt is None => None."""
        result = resolve_checker(prompt=None)

        assert result is None
