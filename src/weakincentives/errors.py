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

"""Base exception hierarchy for :mod:`weakincentives`."""

from __future__ import annotations

from typing import Any, Literal


class WinkError(Exception):
    """Base class for all weakincentives exceptions.

    This class serves as the root of the exception hierarchy, allowing callers
    to catch all library-specific exceptions with a single handler::

        try:
            # weakincentives operations
        except WinkError:
            # handle any library error
    """


class ToolValidationError(WinkError, ValueError):
    """Raised when tool parameters fail validation checks."""


class DeadlineExceededError(WinkError, RuntimeError):
    """Raised when tool execution cannot finish before the deadline."""


class SnapshotError(WinkError, RuntimeError):
    """Base class for snapshot-related errors."""


class SnapshotRestoreError(SnapshotError):
    """Raised when restoring from a snapshot fails."""


class TransactionError(WinkError, RuntimeError):
    """Base class for transaction-related errors."""


class RestoreFailedError(TransactionError):
    """Failed to restore from snapshot during transaction rollback."""


PromptEvaluationPhase = Literal["request", "response", "tool", "budget"]
"""Phases where a prompt evaluation error can occur."""

PROMPT_EVALUATION_PHASE_REQUEST: PromptEvaluationPhase = "request"
"""Prompt evaluation failed while issuing the provider request."""

PROMPT_EVALUATION_PHASE_RESPONSE: PromptEvaluationPhase = "response"
"""Prompt evaluation failed while handling the provider response."""

PROMPT_EVALUATION_PHASE_TOOL: PromptEvaluationPhase = "tool"
"""Prompt evaluation failed while handling a tool invocation."""

PROMPT_EVALUATION_PHASE_BUDGET: PromptEvaluationPhase = "budget"
"""Prompt evaluation failed due to budget limits being exceeded."""


class PromptEvaluationError(WinkError, RuntimeError):
    """Raised when evaluation against a provider fails."""

    def __init__(
        self,
        message: str,
        *,
        prompt_name: str,
        phase: PromptEvaluationPhase,
        provider_payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.prompt_name = prompt_name
        self.phase: PromptEvaluationPhase = phase
        self.provider_payload = provider_payload


__all__ = [
    "PROMPT_EVALUATION_PHASE_BUDGET",
    "PROMPT_EVALUATION_PHASE_REQUEST",
    "PROMPT_EVALUATION_PHASE_RESPONSE",
    "PROMPT_EVALUATION_PHASE_TOOL",
    "DeadlineExceededError",
    "PromptEvaluationError",
    "PromptEvaluationPhase",
    "RestoreFailedError",
    "SnapshotError",
    "SnapshotRestoreError",
    "ToolValidationError",
    "TransactionError",
    "WinkError",
]
