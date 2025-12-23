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

"""Provider invocation with throttle retry handling.

This module provides the ProviderCaller class that handles provider invocation
with retry logic for throttling, deadline enforcement, and exponential backoff.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, NoReturn

from ..deadlines import Deadline
from ..runtime.logging import StructuredLogger, get_logger
from .core import (
    PROMPT_EVALUATION_PHASE_REQUEST,
    PromptEvaluationError,
    PromptEvaluationPhase,
)
from .throttle import (
    ThrottleError,
    ThrottlePolicy,
    details_from_error,
    jittered_backoff,
    new_throttle_policy,
    sleep_for,
)
from .utilities import (
    ToolChoice,
    deadline_provider_payload,
)

logger: StructuredLogger = get_logger(
    __name__, context={"component": "adapters.provider_caller"}
)


ProviderCall = Callable[
    [
        list[dict[str, Any]],
        Sequence[Mapping[str, Any]],
        ToolChoice | None,
        Mapping[str, Any] | None,
    ],
    object,
]
"""Callable responsible for invoking the provider with assembled payloads."""


@dataclass(slots=True)
class ProviderCaller:
    """Handles provider invocation with throttle retry logic.

    This class encapsulates the logic for calling a provider with:
    - Deadline enforcement before each call
    - Throttle error handling with exponential backoff
    - Configurable retry policy

    Args:
        call_provider: Callable that invokes the provider
        prompt_name: Name of the prompt (for error context)
        throttle_policy: Policy governing retry behavior
        deadline: Optional deadline for the operation
        log: Logger for diagnostic output
    """

    call_provider: ProviderCall
    prompt_name: str
    throttle_policy: ThrottlePolicy = field(default_factory=new_throttle_policy)
    deadline: Deadline | None = None
    log: StructuredLogger = field(default_factory=lambda: logger)

    def _raise_deadline_error(
        self, message: str, *, phase: PromptEvaluationPhase
    ) -> NoReturn:
        raise PromptEvaluationError(
            message,
            prompt_name=self.prompt_name,
            phase=phase,
            provider_payload=deadline_provider_payload(self.deadline),
        )

    def _ensure_deadline_remaining(
        self, message: str, *, phase: PromptEvaluationPhase
    ) -> None:
        if self.deadline is None:
            return
        if self.deadline.remaining() <= timedelta(0):
            self._raise_deadline_error(message, phase=phase)

    def call(
        self,
        messages: list[dict[str, Any]],
        tool_specs: Sequence[Mapping[str, Any]],
        tool_choice: ToolChoice | None,
        response_format: Mapping[str, Any] | None,
    ) -> object:
        """Invoke the provider with throttle retry handling.

        Args:
            messages: Conversation messages to send
            tool_specs: Tool specifications for the provider
            tool_choice: Directive for tool selection
            response_format: Format specification for structured output

        Returns:
            The raw provider response object

        Raises:
            PromptEvaluationError: If deadline expires or retries exhausted
            ThrottleError: If retry budget is exhausted
        """
        attempts = 0
        total_delay = timedelta(0)

        while True:
            attempts += 1
            self._ensure_deadline_remaining(
                "Deadline expired before provider request.",
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            )

            try:
                return self.call_provider(
                    messages,
                    tool_specs,
                    tool_choice if tool_specs else None,
                    response_format,
                )
            except ThrottleError as error:
                attempts = max(error.attempts, attempts)
                if not error.retry_safe:
                    raise

                if attempts >= self.throttle_policy.max_attempts:
                    raise ThrottleError(
                        "Throttle retry budget exhausted.",
                        prompt_name=self.prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_REQUEST,
                        details=details_from_error(
                            error, attempts=attempts, retry_safe=False
                        ),
                    ) from error

                delay = jittered_backoff(
                    policy=self.throttle_policy,
                    attempt=attempts,
                    retry_after=error.retry_after,
                )

                if self.deadline is not None and self.deadline.remaining() <= delay:
                    raise ThrottleError(
                        "Deadline expired before retrying after throttling.",
                        prompt_name=self.prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_REQUEST,
                        details=details_from_error(
                            error, attempts=attempts, retry_safe=False
                        ),
                    ) from error

                total_delay += delay
                if total_delay > self.throttle_policy.max_total_delay:
                    raise ThrottleError(
                        "Throttle retry window exceeded configured budget.",
                        prompt_name=self.prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_REQUEST,
                        details=details_from_error(
                            error, attempts=attempts, retry_safe=False
                        ),
                    ) from error

                self.log.warning(
                    "Provider throttled request.",
                    event="prompt_throttled",
                    context={
                        "attempt": attempts,
                        "retry_after_seconds": error.retry_after.total_seconds()
                        if error.retry_after
                        else None,
                        "kind": error.kind,
                        "delay_seconds": delay.total_seconds(),
                    },
                )
                sleep_for(delay)


__all__ = [
    "ProviderCall",
    "ProviderCaller",
]
