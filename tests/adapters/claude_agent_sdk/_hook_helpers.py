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

"""Shared helpers and mock types for Claude Agent SDK hook tests."""

from __future__ import annotations

from typing import Any

from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
from weakincentives.filesystem import Filesystem
from weakincentives.prompt import (
    Feedback,
    FeedbackProviderConfig,
    FeedbackTrigger,
    Prompt,
    PromptTemplate,
    TaskCompletionChecker,
)


def _make_prompt() -> Prompt[object]:
    """Create a prompt in active context."""
    prompt: Prompt[object] = Prompt(PromptTemplate(ns="tests", key="hooks-test"))
    prompt.resources.__enter__()
    return prompt


def _make_prompt_with_fs(
    fs: InMemoryFilesystem,
    *,
    task_completion_checker: TaskCompletionChecker | None = None,
) -> Prompt[object]:
    """Create a prompt with filesystem bound in active context."""
    prompt: Prompt[object] = Prompt(
        PromptTemplate(
            ns="tests",
            key="hooks-test",
            task_completion_checker=task_completion_checker,
        )
    )
    prompt = prompt.bind(resources={Filesystem: fs})
    prompt.resources.__enter__()
    return prompt


class _AlwaysTriggerProvider:
    """Test provider that always triggers and returns fixed feedback."""

    @property
    def name(self) -> str:
        return "AlwaysTrigger"

    def should_run(
        self,
        *,
        context: Any,
    ) -> bool:
        return True

    def provide(
        self,
        *,
        context: Any,
    ) -> Feedback:
        return Feedback(
            provider_name=self.name,
            summary="Test feedback triggered",
            severity="info",
        )


def _make_prompt_with_feedback_provider() -> Prompt[object]:
    """Create a prompt with a feedback provider that always triggers."""
    provider = _AlwaysTriggerProvider()
    config = FeedbackProviderConfig(
        provider=provider,  # type: ignore[arg-type]
        trigger=FeedbackTrigger(every_n_calls=1),  # Trigger on every call
    )
    template: PromptTemplate[object] = PromptTemplate(
        ns="tests",
        key="hooks-test-feedback",
        name="test_prompt",  # Match the prompt_name used in HookContext
        feedback_providers=(config,),
    )
    prompt: Prompt[object] = Prompt(template)
    prompt.resources.__enter__()
    return prompt
