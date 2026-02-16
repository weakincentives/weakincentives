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

"""Tests for resolve_checker: prompt-first resolution with config fallback."""

from __future__ import annotations

import warnings
from typing import cast

from weakincentives.adapters.claude_agent_sdk._task_completion import resolve_checker
from weakincentives.adapters.claude_agent_sdk.config import ClaudeAgentSDKClientConfig
from weakincentives.prompt import FileOutputChecker, Prompt, PromptTemplate
from weakincentives.prompt.protocols import PromptProtocol


class TestResolveChecker:
    """Tests for resolve_checker resolution logic."""

    def test_returns_none_when_neither_source_has_checker(self) -> None:
        """No checker on prompt or config => None."""
        prompt: Prompt[object] = Prompt(PromptTemplate(ns="test", key="test"))
        prompt.resources.__enter__()
        config = ClaudeAgentSDKClientConfig()

        result = resolve_checker(
            prompt=cast("PromptProtocol[object]", prompt),
            client_config=config,
        )

        assert result is None

    def test_returns_prompt_checker_when_only_prompt_has_one(self) -> None:
        """Checker on prompt only => prompt checker."""
        checker = FileOutputChecker(files=("output.txt",))
        prompt: Prompt[object] = Prompt(
            PromptTemplate(ns="test", key="test", task_completion_checker=checker)
        )
        prompt.resources.__enter__()
        config = ClaudeAgentSDKClientConfig()

        result = resolve_checker(
            prompt=cast("PromptProtocol[object]", prompt),
            client_config=config,
        )

        assert result is checker

    def test_returns_config_checker_with_deprecation_warning(self) -> None:
        """Checker on config only => config checker with deprecation warning."""
        checker = FileOutputChecker(files=("output.txt",))
        prompt: Prompt[object] = Prompt(PromptTemplate(ns="test", key="test"))
        prompt.resources.__enter__()
        config = ClaudeAgentSDKClientConfig(task_completion_checker=checker)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = resolve_checker(
                prompt=cast("PromptProtocol[object]", prompt),
                client_config=config,
            )

        assert result is checker
        assert len(caught) == 1
        assert issubclass(caught[0].category, DeprecationWarning)
        assert "deprecated" in str(caught[0].message).lower()

    def test_prompt_checker_takes_priority_over_config(self) -> None:
        """Both prompt and config have checkers => prompt wins, no deprecation warning."""
        prompt_checker = FileOutputChecker(files=("from_prompt.txt",))
        config_checker = FileOutputChecker(files=("from_config.txt",))

        prompt: Prompt[object] = Prompt(
            PromptTemplate(
                ns="test", key="test", task_completion_checker=prompt_checker
            )
        )
        prompt.resources.__enter__()
        config = ClaudeAgentSDKClientConfig(task_completion_checker=config_checker)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = resolve_checker(
                prompt=cast("PromptProtocol[object]", prompt),
                client_config=config,
            )

        assert result is prompt_checker
        # No deprecation warning when prompt provides the checker
        assert not any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_returns_none_when_prompt_is_none(self) -> None:
        """Prompt is None and no config checker => None."""
        config = ClaudeAgentSDKClientConfig()

        result = resolve_checker(prompt=None, client_config=config)

        assert result is None

    def test_returns_config_checker_when_prompt_is_none(self) -> None:
        """Prompt is None but config has checker => config checker with deprecation."""
        checker = FileOutputChecker(files=("output.txt",))
        config = ClaudeAgentSDKClientConfig(task_completion_checker=checker)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = resolve_checker(prompt=None, client_config=config)

        assert result is checker
        assert len(caught) == 1
        assert issubclass(caught[0].category, DeprecationWarning)
