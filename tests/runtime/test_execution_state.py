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

"""Tests for ExecutionState and related classes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import pytest

from weakincentives.contrib.tools.filesystem import (
    Filesystem,
    InMemoryFilesystem,
)
from weakincentives.prompt.tool import ResourceRegistry
from weakincentives.runtime.execution_state import (
    CompositeSnapshot,
    ExecutionState,
    SnapshotMetadata,
)
from weakincentives.runtime.session import Session, Snapshot
from weakincentives.runtime.session.snapshots import SnapshotRestoreError
from weakincentives.runtime.snapshotable import Snapshotable

# --- Test Data ---


@dataclass(frozen=True)
class SamplePlan:
    """Test dataclass for session state."""

    objective: str
    status: str = "active"


# --- SnapshotMetadata Tests ---


class TestSnapshotMetadata:
    def test_default_values(self) -> None:
        metadata = SnapshotMetadata()
        assert metadata.tag is None
        assert metadata.tool_call_id is None
        assert metadata.tool_name is None
        assert metadata.phase == "manual"

    def test_custom_values(self) -> None:
        metadata = SnapshotMetadata(
            tag="before-edit",
            tool_call_id="call_123",
            tool_name="edit_file",
            phase="pre_tool",
        )
        assert metadata.tag == "before-edit"
        assert metadata.tool_call_id == "call_123"
        assert metadata.tool_name == "edit_file"
        assert metadata.phase == "pre_tool"


# --- CompositeSnapshot Tests ---


class TestCompositeSnapshot:
    def test_create_with_defaults(self) -> None:
        session = Session()
        session_snapshot = session.snapshot()

        composite = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session_snapshot,
        )

        assert composite.snapshot_id == UUID("12345678-1234-5678-1234-567812345678")
        assert len(composite.resources) == 0
        assert composite.metadata is None

    def test_create_with_metadata(self) -> None:
        session = Session()
        session_snapshot = session.snapshot()
        metadata = SnapshotMetadata(tag="test", phase="pre_tool")

        composite = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session_snapshot,
            metadata=metadata,
        )

        assert composite.metadata is not None
        assert composite.metadata.tag == "test"
        assert composite.metadata.phase == "pre_tool"

    def test_to_json_roundtrip(self) -> None:
        session = Session()
        session[SamplePlan].seed([SamplePlan(objective="test")])
        session_snapshot = session.snapshot()

        original = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            session=session_snapshot,
            metadata=SnapshotMetadata(
                tag="test-tag",
                tool_call_id="call_123",
                tool_name="edit",
                phase="pre_tool",
            ),
        )

        json_str = original.to_json()
        restored = CompositeSnapshot.from_json(json_str)

        assert restored.snapshot_id == original.snapshot_id
        assert restored.created_at == original.created_at
        assert restored.metadata is not None
        assert restored.metadata.tag == "test-tag"
        assert restored.metadata.tool_call_id == "call_123"
        assert restored.metadata.tool_name == "edit"
        assert restored.metadata.phase == "pre_tool"

    def test_from_json_rejects_invalid_json(self) -> None:
        with pytest.raises(SnapshotRestoreError):
            CompositeSnapshot.from_json("not-json")

    def test_from_json_rejects_non_object(self) -> None:
        with pytest.raises(SnapshotRestoreError):
            CompositeSnapshot.from_json('"string"')

    def test_from_json_rejects_wrong_version(self) -> None:
        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        # Modify version
        import json

        data = json.loads(json_str)
        data["version"] = "999"
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)

        assert "version mismatch" in str(exc_info.value)


# --- ExecutionState Tests ---


class TestExecutionState:
    def test_create_minimal(self) -> None:
        session = Session()
        state = ExecutionState(session=session)

        assert state.session is session
        assert isinstance(state.resources, ResourceRegistry)

    def test_create_with_resources(self) -> None:
        session = Session()
        fs = InMemoryFilesystem()
        resources = ResourceRegistry.build({Filesystem: fs})

        state = ExecutionState(session=session, resources=resources)

        assert state.resources.get(Filesystem) is fs

    def test_snapshot_captures_session(self) -> None:
        session = Session()
        session[SamplePlan].seed([SamplePlan(objective="original")])
        state = ExecutionState(session=session)

        snapshot = state.snapshot(tag="test")

        assert snapshot.metadata is not None
        assert snapshot.metadata.tag == "test"
        assert isinstance(snapshot.session, Snapshot)

    def test_snapshot_captures_filesystem(self) -> None:
        session = Session()
        fs = InMemoryFilesystem()
        fs.write("test.txt", "hello")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        snapshot = state.snapshot(tag="with-fs")

        assert Filesystem in snapshot.resources

    def test_restore_restores_session(self) -> None:
        session = Session()
        session[SamplePlan].seed([SamplePlan(objective="original")])
        state = ExecutionState(session=session)

        snapshot = state.snapshot()

        # Modify session
        session[SamplePlan].seed([SamplePlan(objective="modified")])
        assert session[SamplePlan].latest().objective == "modified"

        # Restore
        state.restore(snapshot)

        assert session[SamplePlan].latest().objective == "original"

    def test_restore_restores_filesystem(self) -> None:
        session = Session()
        fs = InMemoryFilesystem()
        fs.write("test.txt", "original")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        snapshot = state.snapshot()

        # Modify filesystem
        fs.write("test.txt", "modified")
        assert fs.read("test.txt").content == "modified"

        # Restore
        state.restore(snapshot)

        assert fs.read("test.txt").content == "original"

    def test_tool_failure_restores_state(self) -> None:
        """Tool failure does not change state - acceptance criteria."""
        session = Session()
        session[SamplePlan].seed([SamplePlan(objective="test", status="active")])
        fs = InMemoryFilesystem()
        fs.write("file.txt", "original")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        # Capture pre-execution snapshot
        pre_snapshot = state.snapshot()

        # Simulate failed tool execution that modifies state
        fs.write("file.txt", "modified")
        session[SamplePlan].seed([SamplePlan(objective="changed")])

        # Restore on failure
        state.restore(pre_snapshot)

        assert fs.read("file.txt").content == "original"
        assert session[SamplePlan].latest().objective == "test"


# --- Snapshotable Protocol Tests ---


class TestSnapshotableProtocol:
    def test_in_memory_filesystem_is_snapshotable(self) -> None:
        fs = InMemoryFilesystem()
        assert isinstance(fs, Snapshotable)

    def test_resource_registry_finds_snapshotable_resources(self) -> None:
        fs = InMemoryFilesystem()
        resources = ResourceRegistry.build({Filesystem: fs})

        snapshotable = resources.snapshotable_resources()

        assert Filesystem in snapshotable
        assert snapshotable[Filesystem] is fs


# --- Integration Tests ---


class TestExecutionStateIntegration:
    def test_composite_snapshot_roundtrip_with_filesystem(self) -> None:
        """Composite snapshot includes both session and filesystem."""
        session = Session()
        session[SamplePlan].seed([SamplePlan(objective="test")])
        fs = InMemoryFilesystem()
        fs.write("config.py", "DEBUG = True")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        # Snapshot
        snapshot = state.snapshot(tag="pre:tool_call")

        # Modify state
        session[SamplePlan].seed([SamplePlan(objective="changed")])
        fs.write("config.py", "DEBUG = False")

        # Restore
        state.restore(snapshot)

        assert session[SamplePlan].latest().objective == "test"
        assert fs.read("config.py").content == "DEBUG = True"

    def test_multiple_snapshots_isolated(self) -> None:
        """Multiple snapshots don't interfere with each other."""
        session = Session()
        fs = InMemoryFilesystem()
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        # First state
        fs.write("file.txt", "v1")
        snapshot1 = state.snapshot()

        # Second state
        fs.write("file.txt", "v2")
        snapshot2 = state.snapshot()

        # Third state
        fs.write("file.txt", "v3")

        # Restore to first
        state.restore(snapshot1)
        assert fs.read("file.txt").content == "v1"

        # Restore to second
        state.restore(snapshot2)
        assert fs.read("file.txt").content == "v2"


# --- CompositeSnapshot Error Handling Tests ---


class TestCompositeSnapshotFromJsonErrors:
    """Tests for from_json error handling paths."""

    def test_from_json_rejects_invalid_snapshot_id_type(self) -> None:
        """snapshot_id must be a string."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["snapshot_id"] = 12345  # Not a string
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "snapshot_id must be a string" in str(exc_info.value)

    def test_from_json_rejects_invalid_snapshot_id_uuid(self) -> None:
        """snapshot_id must be a valid UUID."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["snapshot_id"] = "not-a-uuid"
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Invalid snapshot_id" in str(exc_info.value)

    def test_from_json_rejects_invalid_created_at_type(self) -> None:
        """created_at must be a string."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["created_at"] = 12345  # Not a string
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "created_at must be a string" in str(exc_info.value)

    def test_from_json_rejects_invalid_created_at_timestamp(self) -> None:
        """created_at must be a valid timestamp."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["created_at"] = "not-a-timestamp"
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Invalid created_at timestamp" in str(exc_info.value)

    def test_from_json_rejects_invalid_session_type(self) -> None:
        """session must be an object."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["session"] = "not-an-object"
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Session snapshot must be an object" in str(exc_info.value)

    def test_from_json_rejects_invalid_session_parse(self) -> None:
        """Session snapshot must be parseable."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["session"] = {"invalid": "session-data"}  # Invalid structure
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Failed to parse session snapshot" in str(exc_info.value)

    def test_from_json_rejects_non_list_resources(self) -> None:
        """resources must be a list."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["resources"] = {"not": "a list"}
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Resources must be a list" in str(exc_info.value)

    def test_from_json_rejects_non_object_resource_entry(self) -> None:
        """Resource entry must be an object."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["resources"] = ["not-an-object"]
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Resource entry must be an object" in str(exc_info.value)

    def test_from_json_rejects_missing_resource_type(self) -> None:
        """Resource entry must have resource_type."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["resources"] = [{"snapshot": {}}]  # Missing resource_type
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Resource type must be a string" in str(exc_info.value)

    def test_from_json_rejects_unknown_resource_type(self) -> None:
        """Resource type must be resolvable."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["resources"] = [
            {"resource_type": "unknown.module:UnknownType", "snapshot": {}}
        ]
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Unknown resource type" in str(exc_info.value)

    def test_from_json_rejects_non_object_resource_snapshot(self) -> None:
        """Resource snapshot must be an object."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["resources"] = [
            {
                "resource_type": "weakincentives.contrib.tools.filesystem:Filesystem",
                "snapshot": "not-an-object",
            }
        ]
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Resource snapshot must be an object" in str(exc_info.value)

    def test_from_json_rejects_missing_type_reference(self) -> None:
        """Resource snapshot must include type reference."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["resources"] = [
            {
                "resource_type": "weakincentives.contrib.tools.filesystem:Filesystem",
                "snapshot": {},  # Missing __type__
            }
        ]
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "must include type reference" in str(exc_info.value)

    def test_from_json_rejects_unknown_snapshot_type(self) -> None:
        """Snapshot type must be resolvable."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["resources"] = [
            {
                "resource_type": "weakincentives.contrib.tools.filesystem:Filesystem",
                "snapshot": {"__type__": "unknown.module:UnknownSnapshot"},
            }
        ]
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Unknown snapshot type" in str(exc_info.value)

    def test_from_json_rejects_invalid_resource_snapshot_parse(self) -> None:
        """Resource snapshot must be parseable."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["resources"] = [
            {
                "resource_type": "weakincentives.contrib.tools.filesystem:Filesystem",
                "snapshot": {
                    "__type__": (
                        "weakincentives.contrib.tools.filesystem:FilesystemSnapshot"
                    ),
                    "invalid_field": "missing required fields",
                },
            }
        ]
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Failed to parse resource snapshot" in str(exc_info.value)

    def test_from_json_rejects_non_object_metadata(self) -> None:
        """metadata must be an object if present."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["metadata"] = "not-an-object"
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Metadata must be an object" in str(exc_info.value)

    def test_from_json_rejects_invalid_metadata_tag(self) -> None:
        """metadata.tag must be a string if present."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["metadata"] = {"tag": 12345}  # Not a string
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Metadata tag must be a string" in str(exc_info.value)

    def test_from_json_rejects_invalid_metadata_tool_call_id(self) -> None:
        """metadata.tool_call_id must be a string if present."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["metadata"] = {"tool_call_id": 12345}  # Not a string
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Metadata tool_call_id must be a string" in str(exc_info.value)

    def test_from_json_rejects_invalid_metadata_tool_name(self) -> None:
        """metadata.tool_name must be a string if present."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["metadata"] = {"tool_name": 12345}  # Not a string
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Metadata tool_name must be a string" in str(exc_info.value)

    def test_from_json_rejects_invalid_metadata_phase(self) -> None:
        """metadata.phase must be valid."""
        import json

        session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
        )
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        data["metadata"] = {"phase": "invalid_phase"}
        modified = json.dumps(data)

        with pytest.raises(SnapshotRestoreError) as exc_info:
            CompositeSnapshot.from_json(modified)
        assert "Metadata phase must be valid" in str(exc_info.value)


class TestCompositeSnapshotToJsonWithResources:
    """Tests for to_json with resources."""

    def test_to_json_serializes_filesystem_snapshot(self) -> None:
        """to_json serializes filesystem snapshots."""
        session = Session()
        fs = InMemoryFilesystem()
        fs.write("test.txt", "content")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        snapshot = state.snapshot()
        json_str = snapshot.to_json()

        # Verify we can roundtrip
        restored = CompositeSnapshot.from_json(json_str)
        assert Filesystem in restored.resources


class TestExecutionStateRestoreErrors:
    """Tests for restore error handling."""

    def test_restore_skips_missing_resource(self) -> None:
        """restore skips resources not in registry."""
        import types

        session = Session()
        state = ExecutionState(session=session)  # No filesystem

        # Create snapshot with filesystem resource
        fs = InMemoryFilesystem()
        fs.write("test.txt", "content")
        fs_snapshot = fs.snapshot()

        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
            resources=types.MappingProxyType({Filesystem: fs_snapshot}),
        )

        # Should not raise - just skips missing resource
        state.restore(snapshot)

    def test_restore_raises_on_resource_restore_failure(self) -> None:
        """restore raises RestoreFailedError on resource restore failure."""
        import types

        from weakincentives.errors import RestoreFailedError
        from weakincentives.runtime.snapshotable import Snapshotable

        class FailingResource:
            """Resource that fails to restore."""

            def snapshot(self, *, tag: str | None = None) -> str:
                return "snapshot"

            def restore(self, snapshot: object) -> None:
                raise RuntimeError("Restore failed")

        assert isinstance(FailingResource(), Snapshotable)

        session = Session()
        resource = FailingResource()
        resources = ResourceRegistry.build({FailingResource: resource})
        state = ExecutionState(session=session, resources=resources)

        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session.snapshot(),
            resources=types.MappingProxyType({FailingResource: "snapshot"}),
        )

        with pytest.raises(RestoreFailedError) as exc_info:
            state.restore(snapshot)
        assert "Failed to restore" in str(exc_info.value)
        assert "FailingResource" in str(exc_info.value)

    def test_restore_raises_on_session_rollback_failure(self) -> None:
        """restore raises RestoreFailedError when session.rollback fails."""
        import types
        from unittest.mock import MagicMock

        from weakincentives.errors import RestoreFailedError

        session = MagicMock()
        session.rollback.side_effect = SnapshotRestoreError("Session rollback failed")

        state = ExecutionState(session=session)

        # Use a real session for the snapshot
        real_session = Session()
        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=real_session.snapshot(),
            resources=types.MappingProxyType({}),
        )

        with pytest.raises(RestoreFailedError) as exc_info:
            state.restore(snapshot)
        assert "Failed to restore session" in str(exc_info.value)


class TestCompositeSnapshotToJsonErrors:
    """Tests for to_json error handling."""

    def test_to_json_raises_on_resource_serialization_error(self) -> None:
        """to_json raises SnapshotSerializationError when resource fails to serialize."""
        import types

        from weakincentives.runtime.session.snapshots import SnapshotSerializationError

        # Create a mock snapshot that can't be serialized
        class UnserializableSnapshot:
            pass

        session = Session()
        session_snapshot = session.snapshot()

        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session_snapshot,
            resources=types.MappingProxyType({Filesystem: UnserializableSnapshot()}),
        )

        with pytest.raises(SnapshotSerializationError) as exc_info:
            snapshot.to_json()
        assert "Failed to serialize resource snapshot" in str(exc_info.value)

    def test_to_json_raises_on_general_error(self) -> None:
        """to_json raises SnapshotSerializationError on general errors."""
        from unittest.mock import patch

        from weakincentives.runtime.session.snapshots import SnapshotSerializationError

        session = Session()
        session_snapshot = session.snapshot()

        snapshot = CompositeSnapshot(
            snapshot_id=UUID("12345678-1234-5678-1234-567812345678"),
            created_at=datetime.now(UTC),
            session=session_snapshot,
        )

        # Mock json.dumps to raise an exception after session serialization
        original_dumps = __import__("json").dumps

        def failing_dumps(obj: object, **kwargs: object) -> str:
            if isinstance(obj, dict) and "version" in obj:
                raise RuntimeError("Unexpected error")
            return original_dumps(obj, **kwargs)

        with patch("json.dumps", side_effect=failing_dumps):
            with pytest.raises(SnapshotSerializationError) as exc_info:
                snapshot.to_json()
            assert "Failed to serialize composite snapshot" in str(exc_info.value)


class TestToolTransactionMethods:
    """Tests for ExecutionState tool transaction methods."""

    def test_begin_tool_execution_creates_snapshot(self) -> None:
        """begin_tool_execution takes a snapshot and stores pending execution."""
        from weakincentives.contrib.tools.filesystem import (
            Filesystem,
            InMemoryFilesystem,
        )
        from weakincentives.prompt.tool import ResourceRegistry

        session = Session()
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "initial")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        state.begin_tool_execution(tool_use_id="abc123", tool_name="Edit")

        # Check pending execution exists
        assert "abc123" in state.pending_tool_executions
        pending = state.pending_tool_executions["abc123"]
        assert pending.tool_use_id == "abc123"
        assert pending.tool_name == "Edit"
        assert pending.snapshot is not None

    def test_end_tool_execution_success_no_restore(self) -> None:
        """end_tool_execution with success=True doesn't restore state."""
        from weakincentives.contrib.tools.filesystem import (
            Filesystem,
            InMemoryFilesystem,
        )
        from weakincentives.prompt.tool import ResourceRegistry

        session = Session()
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "initial")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        state.begin_tool_execution(tool_use_id="abc123", tool_name="Edit")

        # Modify state after snapshot
        fs.write("/test.txt", "modified")

        # End with success
        restored = state.end_tool_execution(tool_use_id="abc123", success=True)

        assert restored is False
        assert "abc123" not in state.pending_tool_executions
        # State should NOT be restored
        assert fs.read("/test.txt").content == "modified"

    def test_end_tool_execution_failure_restores(self) -> None:
        """end_tool_execution with success=False restores state."""
        from weakincentives.contrib.tools.filesystem import (
            Filesystem,
            InMemoryFilesystem,
        )
        from weakincentives.prompt.tool import ResourceRegistry

        session = Session()
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "initial")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        state.begin_tool_execution(tool_use_id="abc123", tool_name="Edit")

        # Modify state after snapshot
        fs.write("/test.txt", "modified")

        # End with failure
        restored = state.end_tool_execution(tool_use_id="abc123", success=False)

        assert restored is True
        assert "abc123" not in state.pending_tool_executions
        # State should be restored
        assert fs.read("/test.txt").content == "initial"

    def test_end_tool_execution_unknown_id_returns_false(self) -> None:
        """end_tool_execution with unknown ID returns False."""
        session = Session()
        state = ExecutionState(session=session)

        restored = state.end_tool_execution(tool_use_id="unknown", success=False)

        assert restored is False

    def test_abort_tool_execution_restores(self) -> None:
        """abort_tool_execution always restores state."""
        from weakincentives.contrib.tools.filesystem import (
            Filesystem,
            InMemoryFilesystem,
        )
        from weakincentives.prompt.tool import ResourceRegistry

        session = Session()
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "initial")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        state.begin_tool_execution(tool_use_id="abc123", tool_name="Edit")

        # Modify state after snapshot
        fs.write("/test.txt", "modified")

        # Abort
        restored = state.abort_tool_execution(tool_use_id="abc123")

        assert restored is True
        assert "abc123" not in state.pending_tool_executions
        # State should be restored
        assert fs.read("/test.txt").content == "initial"

    def test_abort_tool_execution_unknown_id_returns_false(self) -> None:
        """abort_tool_execution with unknown ID returns False."""
        session = Session()
        state = ExecutionState(session=session)

        restored = state.abort_tool_execution(tool_use_id="unknown")

        assert restored is False

    def test_pending_tool_executions_is_read_only(self) -> None:
        """pending_tool_executions returns read-only mapping."""
        import types as types_module

        session = Session()
        state = ExecutionState(session=session)

        state.begin_tool_execution(tool_use_id="abc123", tool_name="Edit")

        pending = state.pending_tool_executions
        assert isinstance(pending, types_module.MappingProxyType)
        # Should not be able to modify
        with pytest.raises(TypeError):
            pending["new"] = None  # type: ignore[index]
