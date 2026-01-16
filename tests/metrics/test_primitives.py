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

"""Tests for metrics primitives."""

from __future__ import annotations

import pytest

from weakincentives.metrics import Counter, Gauge, Histogram


class TestCounter:
    """Tests for Counter primitive."""

    def test_counter_default_value(self) -> None:
        """Counter should initialize with zero value."""
        counter = Counter(name="test")
        assert counter.value == 0

    def test_counter_custom_value(self) -> None:
        """Counter should accept custom initial value."""
        counter = Counter(name="test", value=10)
        assert counter.value == 10

    def test_counter_inc_default(self) -> None:
        """inc() should increment by 1 by default."""
        counter = Counter(name="test")
        new_counter = counter.inc()
        assert new_counter.value == 1
        assert counter.value == 0  # Original unchanged

    def test_counter_inc_delta(self) -> None:
        """inc() should increment by specified delta."""
        counter = Counter(name="test", value=5)
        new_counter = counter.inc(delta=3)
        assert new_counter.value == 8

    def test_counter_with_labels(self) -> None:
        """Counter should preserve labels."""
        labels = (("env", "prod"), ("service", "api"))
        counter = Counter(name="test", labels=labels)
        new_counter = counter.inc()
        assert new_counter.labels == labels

    def test_counter_is_immutable(self) -> None:
        """Counter should be immutable."""
        counter = Counter(name="test")
        with pytest.raises(AttributeError):
            counter.value = 5  # type: ignore[misc]


class TestHistogram:
    """Tests for Histogram primitive."""

    def test_histogram_default_state(self) -> None:
        """Histogram should initialize with zero counts."""
        hist = Histogram(name="test")
        assert hist.total_count == 0
        assert hist.total_sum == 0
        assert len(hist.bucket_counts) == 18  # 17 exponential + 1 for +Inf

    def test_histogram_observe(self) -> None:
        """observe() should record a value."""
        hist = Histogram(name="test")
        new_hist = hist.observe(100)
        assert new_hist.total_count == 1
        assert new_hist.total_sum == 100
        assert hist.total_count == 0  # Original unchanged

    def test_histogram_observe_multiple(self) -> None:
        """Multiple observations should accumulate."""
        hist = Histogram(name="test")
        hist = hist.observe(10)
        hist = hist.observe(20)
        hist = hist.observe(30)
        assert hist.total_count == 3
        assert hist.total_sum == 60

    def test_histogram_bucket_boundaries(self) -> None:
        """Values should fall into correct buckets."""
        hist = Histogram(name="test")

        # Value 1 should go to bucket 0 (boundary 1)
        hist = hist.observe(1)
        assert hist.bucket_counts[0] == 1

        # Value 2 should go to bucket 1 (boundary 2)
        hist = hist.observe(2)
        assert hist.bucket_counts[1] == 1

        # Value 3 should go to bucket 2 (boundary 4)
        hist = hist.observe(3)
        assert hist.bucket_counts[2] == 1

        # Value 100000 should go to +Inf bucket (index 17)
        hist = hist.observe(100000)
        assert hist.bucket_counts[17] == 1

    def test_histogram_mean_empty(self) -> None:
        """mean should return None for empty histogram."""
        hist = Histogram(name="test")
        assert hist.mean is None

    def test_histogram_mean_with_data(self) -> None:
        """mean should return average of observations."""
        hist = Histogram(name="test")
        hist = hist.observe(10)
        hist = hist.observe(20)
        hist = hist.observe(30)
        assert hist.mean == 20.0

    def test_histogram_percentile_empty(self) -> None:
        """percentile should return None for empty histogram."""
        hist = Histogram(name="test")
        assert hist.percentile(0.5) is None

    def test_histogram_percentile_p50(self) -> None:
        """percentile(0.5) should return median estimate."""
        hist = Histogram(name="test")
        # Add values in first bucket (0-1)
        for _ in range(10):
            hist = hist.observe(1)
        p50 = hist.percentile(0.5)
        assert p50 is not None
        assert 0 <= p50 <= 1

    def test_histogram_percentile_p99(self) -> None:
        """percentile(0.99) should return high percentile estimate."""
        hist = Histogram(name="test")
        for i in range(100):
            hist = hist.observe(i + 1)
        p99 = hist.percentile(0.99)
        assert p99 is not None
        assert p99 > 0

    def test_histogram_percentile_p0(self) -> None:
        """percentile(0.0) should return lower bound."""
        hist = Histogram(name="test")
        hist = hist.observe(10)
        p0 = hist.percentile(0.0)
        assert p0 is not None
        assert p0 == 0.0

    def test_histogram_percentile_p100(self) -> None:
        """percentile(1.0) should return upper bound estimate."""
        hist = Histogram(name="test")
        # Add values to the last non-infinite bucket
        for _ in range(10):
            hist = hist.observe(65536)
        p100 = hist.percentile(1.0)
        assert p100 is not None

    def test_histogram_with_labels(self) -> None:
        """Histogram should preserve labels."""
        labels = (("env", "prod"),)
        hist = Histogram(name="test", labels=labels)
        new_hist = hist.observe(100)
        assert new_hist.labels == labels

    def test_histogram_is_immutable(self) -> None:
        """Histogram should be immutable."""
        hist = Histogram(name="test")
        with pytest.raises(AttributeError):
            hist.total_count = 5  # type: ignore[misc]


class TestGauge:
    """Tests for Gauge primitive."""

    def test_gauge_default_value(self) -> None:
        """Gauge should initialize with zero value."""
        gauge = Gauge(name="test")
        assert gauge.value == 0

    def test_gauge_custom_value(self) -> None:
        """Gauge should accept custom initial value."""
        gauge = Gauge(name="test", value=10)
        assert gauge.value == 10

    def test_gauge_set(self) -> None:
        """set() should update value."""
        gauge = Gauge(name="test")
        new_gauge = gauge.set(42)
        assert new_gauge.value == 42
        assert gauge.value == 0  # Original unchanged

    def test_gauge_inc_default(self) -> None:
        """inc() should increment by 1 by default."""
        gauge = Gauge(name="test", value=10)
        new_gauge = gauge.inc()
        assert new_gauge.value == 11

    def test_gauge_inc_delta(self) -> None:
        """inc() should increment by specified delta."""
        gauge = Gauge(name="test", value=10)
        new_gauge = gauge.inc(delta=5)
        assert new_gauge.value == 15

    def test_gauge_dec_default(self) -> None:
        """dec() should decrement by 1 by default."""
        gauge = Gauge(name="test", value=10)
        new_gauge = gauge.dec()
        assert new_gauge.value == 9

    def test_gauge_dec_delta(self) -> None:
        """dec() should decrement by specified delta."""
        gauge = Gauge(name="test", value=10)
        new_gauge = gauge.dec(delta=3)
        assert new_gauge.value == 7

    def test_gauge_negative_values(self) -> None:
        """Gauge should support negative values."""
        gauge = Gauge(name="test", value=0)
        new_gauge = gauge.dec(delta=5)
        assert new_gauge.value == -5

    def test_gauge_with_labels(self) -> None:
        """Gauge should preserve labels."""
        labels = (("env", "prod"),)
        gauge = Gauge(name="test", labels=labels)
        new_gauge = gauge.set(42)
        assert new_gauge.labels == labels

    def test_gauge_is_immutable(self) -> None:
        """Gauge should be immutable."""
        gauge = Gauge(name="test")
        with pytest.raises(AttributeError):
            gauge.value = 5  # type: ignore[misc]
