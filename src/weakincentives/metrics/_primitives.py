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

"""Immutable metric primitives for in-memory collection."""

from __future__ import annotations

from dataclasses import replace
from typing import Self

from ..dataclasses import FrozenDataclass

# 18 exponential bucket boundaries: 1, 2, 4, 8, ..., 65536, +Inf (milliseconds)
_BUCKET_BOUNDARIES: tuple[int, ...] = tuple(2**i for i in range(17))
"""Exponential bucket boundaries (1ms to 65536ms), with implicit +Inf."""

_NUM_BUCKETS: int = len(_BUCKET_BOUNDARIES) + 1  # 18 buckets including +Inf


@FrozenDataclass()
class Counter:
    """Monotonically increasing counter metric.

    Attributes:
        name: Metric identifier.
        value: Current count (non-negative).
        labels: Dimension labels as (key, value) pairs.
    """

    name: str
    value: int = 0
    labels: tuple[tuple[str, str], ...] = ()

    def inc(self, delta: int = 1) -> Self:
        """Return a new Counter with incremented value.

        Args:
            delta: Amount to increment (default 1).

        Returns:
            New Counter with updated value.
        """
        return replace(self, value=self.value + delta)


@FrozenDataclass()
class Histogram:
    """Distribution histogram with exponential bucket boundaries.

    Buckets use exponential boundaries (1, 2, 4, 8, ..., 65536, +Inf)
    milliseconds for latency tracking.

    Attributes:
        name: Metric identifier.
        bucket_counts: Counts per bucket (18 buckets).
        total_count: Total number of observations.
        total_sum: Sum of all observed values.
        labels: Dimension labels as (key, value) pairs.
    """

    name: str
    bucket_counts: tuple[int, ...] = tuple(0 for _ in range(_NUM_BUCKETS))
    total_count: int = 0
    total_sum: int = 0
    labels: tuple[tuple[str, str], ...] = ()

    def observe(self, value: int) -> Self:
        """Record an observation in the histogram.

        Args:
            value: The observed value (typically milliseconds).

        Returns:
            New Histogram with updated counts.
        """
        bucket_idx = self._find_bucket(value)
        new_counts = list(self.bucket_counts)
        new_counts[bucket_idx] += 1
        return replace(
            self,
            bucket_counts=tuple(new_counts),
            total_count=self.total_count + 1,
            total_sum=self.total_sum + value,
        )

    def percentile(self, p: float) -> float | None:
        """Estimate a percentile from the histogram distribution.

        Uses linear interpolation within buckets.

        Args:
            p: Percentile to compute (0.0 to 1.0).

        Returns:
            Estimated percentile value, or None if no observations.
        """
        if self.total_count == 0:
            return None

        target = self.total_count * p
        cumulative = 0

        for i, count in enumerate(self.bucket_counts):
            if cumulative + count >= target:
                # Found the bucket containing the percentile
                lower = 0 if i == 0 else _BUCKET_BOUNDARIES[i - 1]
                upper = (
                    _BUCKET_BOUNDARIES[i]
                    if i < len(_BUCKET_BOUNDARIES)
                    else lower * 2  # Extrapolate for +Inf bucket
                )
                if count == 0:
                    return float(lower)
                # Linear interpolation within bucket
                fraction = (target - cumulative) / count
                return lower + fraction * (upper - lower)
            cumulative += count

        # Fallback: return upper bound of last non-infinite bucket
        # This is defensive code - mathematically unreachable with valid histogram data
        return float(_BUCKET_BOUNDARIES[-1])  # pragma: no cover

    @property
    def mean(self) -> float | None:
        """Average of observed values.

        Returns:
            Mean value, or None if no observations.
        """
        if self.total_count == 0:
            return None
        return self.total_sum / self.total_count

    @staticmethod
    def _find_bucket(value: int) -> int:
        """Find the bucket index for a given value."""
        for i, boundary in enumerate(_BUCKET_BOUNDARIES):
            if value <= boundary:
                return i
        return len(_BUCKET_BOUNDARIES)  # +Inf bucket


@FrozenDataclass()
class Gauge:
    """Point-in-time value that can increase or decrease.

    Attributes:
        name: Metric identifier.
        value: Current value.
        labels: Dimension labels as (key, value) pairs.
    """

    name: str
    value: int = 0
    labels: tuple[tuple[str, str], ...] = ()

    def set(self, value: int) -> Self:
        """Return a new Gauge with the specified value.

        Args:
            value: New value to set.

        Returns:
            New Gauge with updated value.
        """
        return replace(self, value=value)

    def inc(self, delta: int = 1) -> Self:
        """Return a new Gauge with incremented value.

        Args:
            delta: Amount to increment (default 1).

        Returns:
            New Gauge with updated value.
        """
        return replace(self, value=self.value + delta)

    def dec(self, delta: int = 1) -> Self:
        """Return a new Gauge with decremented value.

        Args:
            delta: Amount to decrement (default 1).

        Returns:
            New Gauge with updated value.
        """
        return replace(self, value=self.value - delta)


__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
]
