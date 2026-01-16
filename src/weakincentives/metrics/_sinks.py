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

"""Metrics sinks for external export."""

from __future__ import annotations

import logging
import socket
from typing import Protocol

logger: logging.Logger = logging.getLogger(__name__)


class MetricsSink(Protocol):
    """Protocol for external metrics export.

    Implementations should be thread-safe and handle network failures gracefully.
    """

    def emit_counter(self, name: str, value: int, tags: dict[str, str]) -> None:
        """Emit a counter metric.

        Args:
            name: Metric name.
            value: Delta to add to counter.
            tags: Dimensional labels.
        """
        ...

    def emit_histogram(self, name: str, value: int, tags: dict[str, str]) -> None:
        """Emit a histogram observation.

        Args:
            name: Metric name.
            value: Observed value.
            tags: Dimensional labels.
        """
        ...

    def emit_gauge(self, name: str, value: int, tags: dict[str, str]) -> None:
        """Emit a gauge metric.

        Args:
            name: Metric name.
            value: Current value.
            tags: Dimensional labels.
        """
        ...

    def flush(self) -> None:
        """Flush any buffered metrics."""
        ...


class StatsdSink:
    """StatsD protocol sink for metrics export.

    Compatible with StatsD, DataDog, Telegraf, and other StatsD-compatible receivers.

    Args:
        host: StatsD server hostname.
        port: StatsD server port.
        prefix: Prefix for all metric names.
        sample_rate: Sampling rate (0.0 to 1.0).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8125,
        prefix: str = "wink",
        sample_rate: float = 1.0,
    ) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._prefix = prefix
        self._sample_rate = sample_rate
        self._socket: socket.socket | None = None

    def _get_socket(self) -> socket.socket:
        """Get or create the UDP socket."""
        if self._socket is None:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return self._socket

    @staticmethod
    def _format_tags(tags: dict[str, str]) -> str:
        """Format tags for DogStatsD format."""
        if not tags:
            return ""
        return "|#" + ",".join(f"{k}:{v}" for k, v in sorted(tags.items()))

    def _send(self, data: str) -> None:
        """Send data to StatsD server."""
        try:
            sock = self._get_socket()
            _ = sock.sendto(data.encode("utf-8"), (self._host, self._port))
        except OSError:
            logger.warning(
                "Failed to send metrics to StatsD",
                extra={"host": self._host, "port": self._port},
                exc_info=True,
            )

    def emit_counter(self, name: str, value: int, tags: dict[str, str]) -> None:
        """Emit a counter metric."""
        metric_name = f"{self._prefix}.{name}"
        tag_str = self._format_tags(tags)
        data = f"{metric_name}:{value}|c{tag_str}"
        if self._sample_rate < 1.0:
            data = f"{data}|@{self._sample_rate}"
        self._send(data)

    def emit_histogram(self, name: str, value: int, tags: dict[str, str]) -> None:
        """Emit a histogram observation (as StatsD timing)."""
        metric_name = f"{self._prefix}.{name}"
        tag_str = self._format_tags(tags)
        data = f"{metric_name}:{value}|ms{tag_str}"
        if self._sample_rate < 1.0:
            data = f"{data}|@{self._sample_rate}"
        self._send(data)

    def emit_gauge(self, name: str, value: int, tags: dict[str, str]) -> None:
        """Emit a gauge metric."""
        metric_name = f"{self._prefix}.{name}"
        tag_str = self._format_tags(tags)
        data = f"{metric_name}:{value}|g{tag_str}"
        self._send(data)

    def flush(self) -> None:
        """Flush buffered metrics (no-op for UDP)."""
        pass

    def close(self) -> None:
        """Close the socket."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class DebugSink:
    """Debug sink that logs all metrics.

    Useful for development and testing.

    Args:
        logger: Logger instance to use. Defaults to module logger.
    """

    def __init__(self, log: logging.Logger | None = None) -> None:
        super().__init__()
        self._logger = log or logger

    def emit_counter(self, name: str, value: int, tags: dict[str, str]) -> None:
        """Log a counter metric."""
        self._logger.debug(
            "Counter: %s=%d",
            name,
            value,
            extra={"metric_name": name, "metric_value": value, "tags": tags},
        )

    def emit_histogram(self, name: str, value: int, tags: dict[str, str]) -> None:
        """Log a histogram observation."""
        self._logger.debug(
            "Histogram: %s=%d",
            name,
            value,
            extra={"metric_name": name, "metric_value": value, "tags": tags},
        )

    def emit_gauge(self, name: str, value: int, tags: dict[str, str]) -> None:
        """Log a gauge metric."""
        self._logger.debug(
            "Gauge: %s=%d",
            name,
            value,
            extra={"metric_name": name, "metric_value": value, "tags": tags},
        )

    def flush(self) -> None:
        """Flush logs (no-op)."""
        pass


__all__ = [
    "DebugSink",
    "MetricsSink",
    "StatsdSink",
]
