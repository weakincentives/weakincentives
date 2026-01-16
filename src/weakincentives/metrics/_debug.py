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

"""Debug utilities for metrics persistence."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from ..types import JSONValue
from ._snapshot import MetricsSnapshot

logger: logging.Logger = logging.getLogger(__name__)

_DEBUG_DIR = Path(".weakincentives/debug/metrics")


def _serialize_value(value: Any) -> JSONValue:  # noqa: ANN401
    """Serialize a value for JSON output.

    Handles dataclasses, datetimes, tuples, dicts, and primitives.
    """
    if is_dataclass(value) and not isinstance(value, type):
        raw: dict[str, Any] = asdict(value)
        return {k: _serialize_value(v) for k, v in raw.items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        items = cast(tuple[Any, ...], value)
        return [_serialize_value(v) for v in items]
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        return {str(k): _serialize_value(v) for k, v in mapping.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def dump_metrics(snapshot: MetricsSnapshot, path: str | Path) -> Path:
    """Persist a metrics snapshot to a JSON file.

    Args:
        snapshot: The MetricsSnapshot to persist.
        path: Target file path.

    Returns:
        Path to the created file.

    Raises:
        OSError: If the file cannot be written.
    """
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    data = _serialize_value(snapshot)
    payload = json.dumps(data, indent=2, default=str)
    _ = target.write_text(payload, encoding="utf-8")

    logger.info(
        "Metrics snapshot persisted.",
        extra={
            "snapshot_path": str(target),
            "adapter_count": len(snapshot.adapters),
            "tool_count": len(snapshot.tools),
            "mailbox_count": len(snapshot.mailboxes),
        },
    )
    return target


def archive_metrics(snapshot: MetricsSnapshot, *, base_dir: Path | None = None) -> Path:
    """Archive a metrics snapshot to the debug directory.

    Creates a timestamped JSON file in the debug archive:
    ``.weakincentives/debug/metrics/<timestamp>_<worker_id>.json``

    Args:
        snapshot: The MetricsSnapshot to archive.
        base_dir: Optional base directory (defaults to current directory).

    Returns:
        Path to the created archive file.

    Raises:
        OSError: If the archive cannot be written.
    """
    base = base_dir or Path.cwd()
    archive_dir = base / _DEBUG_DIR

    timestamp = snapshot.captured_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    worker = snapshot.worker_id or "unknown"
    filename = f"{timestamp}_{worker}.json"

    return dump_metrics(snapshot, archive_dir / filename)


__all__ = [
    "archive_metrics",
    "dump_metrics",
]
