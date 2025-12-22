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

"""Deadline handling utilities for provider adapters."""

from __future__ import annotations

from typing import Any, NoReturn

from ..deadlines import Deadline
from .core import (
    PROMPT_EVALUATION_PHASE_TOOL,
    PromptEvaluationError,
)


def deadline_provider_payload(deadline: Deadline | None) -> dict[str, Any] | None:
    """Return a provider payload snippet describing the active deadline."""

    if deadline is None:
        return None
    return {"deadline_expires_at": deadline.expires_at.isoformat()}


def raise_tool_deadline_error(
    *, prompt_name: str, tool_name: str, deadline: Deadline
) -> NoReturn:
    """Raise a PromptEvaluationError for a deadline expired during tool execution."""
    raise PromptEvaluationError(
        f"Deadline expired before executing tool '{tool_name}'.",
        prompt_name=prompt_name,
        phase=PROMPT_EVALUATION_PHASE_TOOL,
        provider_payload=deadline_provider_payload(deadline),
    )


__all__ = [
    "deadline_provider_payload",
    "raise_tool_deadline_error",
]
