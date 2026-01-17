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

"""Tests for metrics debug utilities."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from weakincentives.metrics import (
    AdapterCallParams,
    InMemoryMetricsCollector,
    MetricsSnapshot,
    archive_metrics,
    dump_metrics,
)


class TestDumpMetrics:
    """Tests for dump_metrics function."""

    def test_dump_metrics_creates_file(self, tmp_path: Path) -> None:
        """dump_metrics should create a JSON file."""
        snapshot = MetricsSnapshot.empty(worker_id="test")
        target = tmp_path / "metrics.json"

        result = dump_metrics(snapshot, target)

        assert result == target
        assert target.exists()

    def test_dump_metrics_creates_parent_dirs(self, tmp_path: Path) -> None:
        """dump_metrics should create parent directories."""
        snapshot = MetricsSnapshot.empty(worker_id="test")
        target = tmp_path / "deep" / "nested" / "metrics.json"

        result = dump_metrics(snapshot, target)

        assert result == target
        assert target.exists()

    def test_dump_metrics_valid_json(self, tmp_path: Path) -> None:
        """dump_metrics should write valid JSON."""
        collector = InMemoryMetricsCollector(worker_id="test-worker")
        collector.record_adapter_call(
            "openai",
            AdapterCallParams(render_ms=10, call_ms=500, parse_ms=5, tool_ms=100),
        )
        collector.record_tool_call("read_file", latency_ms=25, success=True)
        snapshot = collector.snapshot()

        target = tmp_path / "metrics.json"
        dump_metrics(snapshot, target)

        # Should be valid JSON
        content = target.read_text()
        data = json.loads(content)

        assert "adapters" in data
        assert "tools" in data
        assert "mailboxes" in data
        assert "captured_at" in data
        assert data["worker_id"] == "test-worker"

    def test_dump_metrics_contains_adapter_data(self, tmp_path: Path) -> None:
        """dump_metrics should include adapter metrics."""
        collector = InMemoryMetricsCollector()
        collector.record_adapter_call(
            "openai",
            AdapterCallParams(render_ms=10, call_ms=500, parse_ms=5, tool_ms=100),
        )
        snapshot = collector.snapshot()

        target = tmp_path / "metrics.json"
        dump_metrics(snapshot, target)

        data = json.loads(target.read_text())
        assert len(data["adapters"]) == 1
        assert data["adapters"][0]["adapter"] == "openai"

    def test_dump_metrics_contains_tool_data(self, tmp_path: Path) -> None:
        """dump_metrics should include tool metrics."""
        collector = InMemoryMetricsCollector()
        collector.record_tool_call("read_file", latency_ms=25, success=True)
        collector.record_tool_call(
            "read_file", latency_ms=30, success=False, error_code="NOT_FOUND"
        )
        snapshot = collector.snapshot()

        target = tmp_path / "metrics.json"
        dump_metrics(snapshot, target)

        data = json.loads(target.read_text())
        assert len(data["tools"]) == 1
        tool_data = data["tools"][0]
        assert tool_data["tool_name"] == "read_file"
        assert tool_data["call_count"]["value"] == 2

    def test_dump_metrics_contains_mailbox_data(self, tmp_path: Path) -> None:
        """dump_metrics should include mailbox metrics."""
        collector = InMemoryMetricsCollector()
        collector.record_message_received("requests", delivery_count=1, age_ms=100)
        collector.record_message_ack("requests")
        snapshot = collector.snapshot()

        target = tmp_path / "metrics.json"
        dump_metrics(snapshot, target)

        data = json.loads(target.read_text())
        assert len(data["mailboxes"]) == 1
        mailbox_data = data["mailboxes"][0]
        assert mailbox_data["queue_name"] == "requests"

    def test_dump_metrics_path_expansion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """dump_metrics should expand ~ in path."""
        # Mock expanduser to use tmp_path as home
        monkeypatch.setattr(Path, "expanduser", lambda self: tmp_path / self.name)

        snapshot = MetricsSnapshot.empty()
        target = Path("~/metrics.json")

        result = dump_metrics(snapshot, target)
        assert result.parent == tmp_path


class TestSerializeValue:
    """Tests for _serialize_value function."""

    def test_serialize_unknown_type(self, tmp_path: Path) -> None:
        """_serialize_value should convert unknown types to str."""
        from weakincentives.metrics._debug import _serialize_value

        class CustomObject:
            def __str__(self) -> str:
                return "custom-object"

        result = _serialize_value(CustomObject())
        assert result == "custom-object"


class TestArchiveMetrics:
    """Tests for archive_metrics function."""

    def test_archive_metrics_creates_file(self, tmp_path: Path) -> None:
        """archive_metrics should create a timestamped file."""
        snapshot = MetricsSnapshot.empty(worker_id="worker-1")

        result = archive_metrics(snapshot, base_dir=tmp_path)

        assert result.exists()
        assert ".weakincentives/debug/metrics" in str(result)
        assert "worker-1" in result.name
        assert result.suffix == ".json"

    def test_archive_metrics_timestamp_format(self, tmp_path: Path) -> None:
        """archive_metrics should use ISO timestamp in filename."""
        captured_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        snapshot = MetricsSnapshot(
            adapters=(),
            tools=(),
            mailboxes=(),
            captured_at=captured_at,
            worker_id="worker-1",
        )

        result = archive_metrics(snapshot, base_dir=tmp_path)

        assert "2024-01-15T10:30:00Z" in result.name

    def test_archive_metrics_unknown_worker(self, tmp_path: Path) -> None:
        """archive_metrics should use 'unknown' for missing worker_id."""
        snapshot = MetricsSnapshot.empty(worker_id=None)

        result = archive_metrics(snapshot, base_dir=tmp_path)

        assert "unknown" in result.name

    def test_archive_metrics_multiple_archives(self, tmp_path: Path) -> None:
        """Multiple archives should coexist."""
        snapshot1 = MetricsSnapshot(
            adapters=(),
            tools=(),
            mailboxes=(),
            captured_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            worker_id="worker-1",
        )
        snapshot2 = MetricsSnapshot(
            adapters=(),
            tools=(),
            mailboxes=(),
            captured_at=datetime(2024, 1, 15, 10, 35, 0, tzinfo=UTC),
            worker_id="worker-1",
        )

        result1 = archive_metrics(snapshot1, base_dir=tmp_path)
        result2 = archive_metrics(snapshot2, base_dir=tmp_path)

        assert result1 != result2
        assert result1.exists()
        assert result2.exists()

    def test_archive_metrics_default_base_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """archive_metrics should use cwd by default."""
        monkeypatch.chdir(tmp_path)
        snapshot = MetricsSnapshot.empty(worker_id="test")

        result = archive_metrics(snapshot)

        assert result.exists()
        assert str(tmp_path) in str(result)
