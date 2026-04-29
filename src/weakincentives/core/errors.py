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

"""Stable error hierarchy for the WINK spine.

Every error in the spine is a subclass of :class:`WinkError`. Extras and user
code can pattern-match on the categories below; their identity is part of
the published API.
"""

from __future__ import annotations

__all__ = [
    "BudgetExceededError",
    "ContractError",
    "DeadlineExceededError",
    "PolicyDeniedError",
    "PromptError",
    "SessionError",
    "SnapshotError",
    "ToolError",
    "TransactionError",
    "WinkError",
]


class WinkError(Exception):
    """Root of every spine-defined exception."""


class PromptError(WinkError):
    """Raised on prompt construction or render failures."""


class ToolError(WinkError):
    """Raised on tool registration or invocation failures."""


class SessionError(WinkError):
    """Raised on reducer, slice, or dispatch failures."""


class SnapshotError(WinkError):
    """Raised on snapshot serialization or restoration failures."""


class TransactionError(WinkError):
    """Raised when a tool transaction cannot complete or roll back."""


class ContractError(WinkError):
    """Base for runtime contract violations (deadlines, budgets, policies)."""


class DeadlineExceededError(ContractError):
    """Raised when a deadline is reached before work completes."""


class BudgetExceededError(ContractError):
    """Raised when a budget is exhausted."""


class PolicyDeniedError(ContractError):
    """Raised by a tool policy to block an invocation."""
