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
"""Dimension that caused a budget limit to be exceeded.

Values:
    - "deadline": Time limit was reached
    - "total_tokens": Combined input + output tokens exceeded the limit
    - "input_tokens": Input token count exceeded the limit
    - "output_tokens": Output token count exceeded the limit
"""


@FrozenDataclass()
class Budget:
    """Resource envelope combining time and token limits.

    A Budget defines constraints on LLM API usage, including time-based deadlines
    and token consumption limits. Use with `BudgetTracker` to enforce limits
    during execution.

    At least one limit must be set. Token limits must be positive when provided.

    Attributes:
        deadline: Optional time limit as a `Deadline` object. When set, operations
            must complete before this deadline or `BudgetExceededError` is raised.
        max_total_tokens: Optional limit on combined input + output tokens.
        max_input_tokens: Optional limit on input (prompt) tokens consumed.
        max_output_tokens: Optional limit on output (completion) tokens generated.

    Raises:
        ValueError: If no limits are set, or if any token limit is non-positive.

    Example:
        >>> from weakincentives.deadlines import Deadline
        >>> # Budget with 1-hour deadline and 100k total tokens
        >>> budget = Budget(
        ...     deadline=Deadline.from_timeout(timedelta(hours=1)),
        ...     max_total_tokens=100_000,
        ... )
        >>> # Budget with only token limits
        >>> budget = Budget(max_input_tokens=50_000, max_output_tokens=10_000)
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
    """Raised when a budget limit is breached.

    This error is raised by `BudgetTracker.check()` when any configured limit
    in the associated `Budget` has been exceeded. The error provides full
    context about which limit was breached and current consumption.

    Attributes:
        budget: The Budget instance whose limit was exceeded.
        consumed: TokenUsage snapshot at the time the limit was breached,
            showing input_tokens, output_tokens, and cached_tokens consumed.
        exceeded_dimension: Which specific limit was breached (deadline,
            total_tokens, input_tokens, or output_tokens).

    Example:
        >>> try:
        ...     tracker.check()
        ... except BudgetExceededError as e:
        ...     print(f"Exceeded {e.exceeded_dimension}: {e.consumed}")
    """

    budget: Budget
    consumed: TokenUsage
    exceeded_dimension: BudgetExceededDimension

    @override
    def __str__(self) -> str:
        return f"Budget exceeded: {self.exceeded_dimension}"


@dataclass
class BudgetTracker:
    """Tracks cumulative token usage across multiple evaluations against a Budget.

    BudgetTracker aggregates TokenUsage from concurrent evaluations and enforces
    the limits defined in the associated Budget. Each evaluation reports its
    cumulative usage via `record_cumulative()`, and `check()` validates against
    all configured limits.

    Thread-safe: all operations are protected by an internal lock, allowing
    concurrent evaluations to safely report usage.

    Attributes:
        budget: The Budget defining the limits to enforce.

    Example:
        >>> budget = Budget(max_total_tokens=100_000)
        >>> tracker = BudgetTracker(budget=budget)
        >>> # Record usage from evaluation "eval-1"
        >>> tracker.record_cumulative("eval-1", TokenUsage(input_tokens=1000))
        >>> # Check if any limits exceeded
        >>> tracker.check()  # Raises BudgetExceededError if over limit
        >>> # Get aggregated consumption across all evaluations
        >>> print(tracker.consumed.total_tokens)
    """

    budget: Budget
    _per_evaluation: dict[str, TokenUsage] = field(default_factory=lambda: {})
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_cumulative(self, evaluation_id: str, usage: TokenUsage) -> None:
        """Record cumulative token usage for a specific evaluation.

        Each call replaces any previously recorded usage for the given evaluation.
        This is designed for evaluations that report their total cumulative usage
        rather than incremental deltas.

        Args:
            evaluation_id: Unique identifier for the evaluation (e.g., "eval-123").
            usage: The cumulative TokenUsage for this evaluation so far.

        Note:
            This method is thread-safe. Multiple evaluations can record
            concurrently without data races.
        """
        with self._lock:
            self._per_evaluation[evaluation_id] = usage

    @property
    def consumed(self) -> TokenUsage:
        """Aggregate token usage across all tracked evaluations.

        Returns:
            TokenUsage: Combined usage with summed input_tokens, output_tokens,
                and cached_tokens from all evaluations. Fields default to 0
                if not set on individual usages.

        Note:
            This property is thread-safe and computes a fresh sum on each access.
        """
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
        """Validate that all budget limits are satisfied.

        Checks all configured limits in the following order:
        1. Deadline (if set) - raises immediately if deadline has passed
        2. Input tokens (if max_input_tokens set)
        3. Output tokens (if max_output_tokens set)
        4. Total tokens (if max_total_tokens set)

        Raises:
            BudgetExceededError: If any configured limit has been exceeded.
                The error's `exceeded_dimension` indicates which limit was
                breached first (in check order above).

        Note:
            Call this periodically during long-running operations to enforce
            budget constraints. The check is thread-safe.
        """
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
