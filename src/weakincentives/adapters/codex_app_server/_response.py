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

"""Response building for the Codex App Server adapter."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from ...budget import BudgetTracker
from ...prompt import RenderedPrompt
from ...prompt.structured_output import OutputParseError, parse_structured_output
from ...runtime.events import PromptExecuted
from ...runtime.events.types import TokenUsage
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.run_context import RunContext
from ...runtime.session.protocols import SessionProtocol
from ...types import AdapterName
from ..core import PromptEvaluationError, PromptResponse

logger: StructuredLogger = get_logger(
    __name__, context={"component": "codex_app_server"}
)


def parse_structured_output_or_raise[OutputT](
    text: str,
    rendered: RenderedPrompt[OutputT],
    prompt_name: str,
) -> OutputT | None:
    """Parse JSON text into the expected output type."""
    if rendered.output_type is None:
        return None  # pragma: no cover

    try:
        return cast(OutputT, parse_structured_output(text, rendered))
    except (OutputParseError, TypeError, ValueError) as error:
        raise PromptEvaluationError(
            message=f"Failed to parse structured output: {error}",
            prompt_name=prompt_name,
            phase="response",
            provider_payload={"raw_text": text[:2000]},
        ) from error


def build_response[OutputT](  # noqa: PLR0913
    *,
    accumulated_text: str | None,
    usage: TokenUsage | None,
    output_schema: dict[str, Any] | None,
    rendered: RenderedPrompt[OutputT],
    prompt_name: str,
    adapter_name: AdapterName,
    session: SessionProtocol,
    budget_tracker: BudgetTracker | None,
    run_context: RunContext | None,
    start_time: datetime,
    utcnow: datetime,
) -> PromptResponse[OutputT]:
    """Build PromptResponse, dispatch event, and log completion."""
    duration_ms = int((utcnow - start_time).total_seconds() * 1000)

    output: OutputT | None = None
    if output_schema is not None and accumulated_text:
        output = parse_structured_output_or_raise(
            accumulated_text, rendered, prompt_name
        )

    if budget_tracker and usage:
        budget_tracker.record_cumulative(prompt_name, usage)

    response = PromptResponse(
        prompt_name=prompt_name,
        text=accumulated_text,
        output=output,
    )

    _ = session.dispatcher.dispatch(
        PromptExecuted(
            prompt_name=prompt_name,
            adapter=adapter_name,
            result=response,
            session_id=getattr(session, "session_id", None),
            created_at=utcnow,
            usage=usage,
            run_context=run_context,
        )
    )

    logger.info(
        "codex_app_server.evaluate.complete",
        event="evaluate.complete",
        context={
            "prompt_name": prompt_name,
            "duration_ms": duration_ms,
            "input_tokens": usage.input_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
        },
    )

    return response
