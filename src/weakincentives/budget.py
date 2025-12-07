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

"""Budget abstraction for token and deadline enforcement."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Literal, override

from .dataclasses import FrozenDataclass
from .deadlines import Deadline
from .errors import WinkError

if TYPE_CHECKING:
    from .runtime.events import TokenUsage

__all__ = [
    "Budget",
    "BudgetExceededDimension",
    "BudgetExceededError",
    "BudgetTracker",
]

BudgetExceededDimension = Literal[
    "deadline", "total_tokens", "input_tokens", "output_tokens"
]
"""Dimension that caused a budget limit to be exceeded."""


@FrozenDataclass()
class Budget:
    """Resource envelope combining time and token limits.

    At least one limit must be set. Token limits must be positive when provided.
    """

    deadline: Deadline | None = None
    max_total_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if (
            self.deadline is None
            and self.max_total_tokens is None
            and self.max_input_tokens is None
            and self.max_output_tokens is None
        ):
            msg = "Budget requires at least one limit to be set."
            raise ValueError(msg)

        if self.max_total_tokens is not None and self.max_total_tokens <= 0:
            msg = "max_total_tokens must be positive."
            raise ValueError(msg)

        if self.max_input_tokens is not None and self.max_input_tokens <= 0:
            msg = "max_input_tokens must be positive."
            raise ValueError(msg)

        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            msg = "max_output_tokens must be positive."
            raise ValueError(msg)


@FrozenDataclass()
class BudgetExceededError(WinkError, RuntimeError):
    """Raised when a budget limit is breached."""

    budget: Budget
    consumed: TokenUsage
    exceeded_dimension: BudgetExceededDimension

    @override
    def __str__(self) -> str:
        return f"Budget exceeded: {self.exceeded_dimension}"


@dataclass
class BudgetTracker:
    """Tracks cumulative TokenUsage per evaluation against a Budget.

    Thread-safe for concurrent execution.
    """

    budget: Budget
    _per_evaluation: dict[str, TokenUsage] = field(default_factory=lambda: {})
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_cumulative(self, evaluation_id: str, usage: TokenUsage) -> None:
        """Record cumulative usage for an evaluation (replaces previous)."""
        with self._lock:
            self._per_evaluation[evaluation_id] = usage

    @property
    def consumed(self) -> TokenUsage:
        """Sum usage across all evaluations."""
        # Import lazily to avoid circular import
        from .runtime.events import TokenUsage as TokenUsageClass

        with self._lock:
            return TokenUsageClass(
                input_tokens=sum(
                    u.input_tokens or 0 for u in self._per_evaluation.values()
                ),
                output_tokens=sum(
                    u.output_tokens or 0 for u in self._per_evaluation.values()
                ),
                cached_tokens=sum(
                    u.cached_tokens or 0 for u in self._per_evaluation.values()
                ),
            )

    def check(self) -> None:
        """Raise BudgetExceededError if any limit is breached."""
        budget = self.budget

        # Check deadline first
        if budget.deadline is not None and budget.deadline.remaining() <= timedelta(0):
            raise BudgetExceededError(
                budget=budget,
                consumed=self.consumed,
                exceeded_dimension="deadline",
            )

        consumed = self.consumed

        # Check token limits
        if budget.max_input_tokens is not None:
            input_total = consumed.input_tokens or 0
            if input_total > budget.max_input_tokens:
                raise BudgetExceededError(
                    budget=budget,
                    consumed=consumed,
                    exceeded_dimension="input_tokens",
                )

        if budget.max_output_tokens is not None:
            output_total = consumed.output_tokens or 0
            if output_total > budget.max_output_tokens:
                raise BudgetExceededError(
                    budget=budget,
                    consumed=consumed,
                    exceeded_dimension="output_tokens",
                )

        if budget.max_total_tokens is not None:
            total = consumed.total_tokens or 0
            if total > budget.max_total_tokens:
                raise BudgetExceededError(
                    budget=budget,
                    consumed=consumed,
                    exceeded_dimension="total_tokens",
                )
