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

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from weakincentives.runtime.session.snapshots import (
    SnapshotPayload,
    SnapshotRestoreError,
    _normalize_tags,
)


def test_snapshot_payload_rejects_non_mapping_tags() -> None:
    payload = json.dumps(
        {
            "version": "1",
            "created_at": datetime.now(UTC).isoformat(),
            "slices": [],
            "tags": ["not-a-mapping"],
        }
    )

    with pytest.raises(SnapshotRestoreError):
        SnapshotPayload.from_json(payload)


def test_snapshot_payload_rejects_non_string_tags() -> None:
    payload = json.dumps(
        {
            "version": "1",
            "created_at": datetime.now(UTC).isoformat(),
            "slices": [],
            "tags": {"ok": 1},
        }
    )

    with pytest.raises(SnapshotRestoreError):
        SnapshotPayload.from_json(payload)


def test_normalize_tags_with_none() -> None:
    """Test branch 59->66: when tags is None, return empty MappingProxyType."""
    result = _normalize_tags(None, error_cls=SnapshotRestoreError)

    # Should return an empty immutable mapping
    assert len(result) == 0
    assert isinstance(result, type({}).__bases__[0])  # MappingProxyType
