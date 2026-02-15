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

"""Tier 4 ACK scenarios for feedback provider delivery.

Feedback providers are declared on the prompt and triggered during tool
execution. These scenarios verify that adapters deliver feedback content
to the agent so it can influence subsequent behavior.
"""

from __future__ import annotations

import pytest

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.prompt import (
    Feedback,
    FeedbackProviderConfig,
    FeedbackTrigger,
    MarkdownSection,
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.runtime.session import Session

from ..adapters import AdapterFixture
from . import TransformRequest, TransformResult, make_adapter_ns

pytestmark = pytest.mark.ack_capability("feedback_providers")


class _ReminderProvider:
    """Test feedback provider that injects a magic codeword into the conversation."""

    MAGIC_CODEWORD = "CANARY_FEEDBACK_7742"

    @property
    def name(self) -> str:
        return "reminder_provider"

    def should_run(self, *, context: object) -> bool:
        return True

    def provide(self, *, context: object) -> Feedback:
        return Feedback(
            provider_name=self.name,
            summary=(
                f"Important: Include the exact string '{self.MAGIC_CODEWORD}' "
                "in your final response."
            ),
            severity="info",
        )


def _build_echo_tool() -> Tool[TransformRequest, TransformResult]:
    """Build a simple echo tool to trigger feedback delivery."""

    def handler(
        params: TransformRequest,
        *,
        context: ToolContext,
    ) -> ToolResult[TransformResult]:
        del context
        result = TransformResult(text=params.text)
        return ToolResult.ok(result, message=f"Echo: {result.text}")

    return Tool[TransformRequest, TransformResult](
        name="echo_text",
        description="Return the provided text unchanged.",
        handler=handler,
    )


def test_feedback_provider_delivers_content(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
) -> None:
    """Adapter delivers feedback provider content to the agent after tool use."""
    tool = _build_echo_tool()
    provider = _ReminderProvider()

    section = MarkdownSection[TransformRequest](
        title="Echo Task",
        template=(
            'Call the `echo_text` tool with text "hello". '
            "Then respond with any feedback you received."
        ),
        tools=(tool,),
        key="task",
    )

    template = PromptTemplate(
        ns=make_adapter_ns(adapter_fixture.adapter_name),
        key="ack-feedback",
        name="ack_feedback_provider",
        sections=[section],
        feedback_providers=(
            FeedbackProviderConfig(
                provider=provider,  # type: ignore[arg-type]
                trigger=FeedbackTrigger(every_n_calls=1),
            ),
        ),
    )

    prompt = Prompt(template).bind(TransformRequest(text="hello"))
    response = adapter.evaluate(prompt, session=session)

    # The agent should have received and included the magic codeword
    assert response.text is not None
    assert _ReminderProvider.MAGIC_CODEWORD in response.text
