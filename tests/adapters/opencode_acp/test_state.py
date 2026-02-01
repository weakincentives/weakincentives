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

"""Tests for OpenCode ACP session state."""

from __future__ import annotations

from weakincentives.adapters.opencode_acp._state import OpenCodeACPSessionState


class TestOpenCodeACPSessionState:
    def test_construction(self) -> None:
        state = OpenCodeACPSessionState(
            session_id="session-123",
            cwd="/path/to/workspace",
            workspace_fingerprint="abc123",
        )
        assert state.session_id == "session-123"
        assert state.cwd == "/path/to/workspace"
        assert state.workspace_fingerprint == "abc123"

    def test_with_none_fingerprint(self) -> None:
        state = OpenCodeACPSessionState(
            session_id="session-456",
            cwd="/another/path",
            workspace_fingerprint=None,
        )
        assert state.session_id == "session-456"
        assert state.cwd == "/another/path"
        assert state.workspace_fingerprint is None

    def test_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        import pytest

        state = OpenCodeACPSessionState(
            session_id="test",
            cwd="/test",
            workspace_fingerprint=None,
        )

        with pytest.raises(FrozenInstanceError):
            state.session_id = "modified"  # type: ignore[misc]

    def test_can_be_stored_in_session(self) -> None:
        from weakincentives.runtime import InProcessDispatcher, Session

        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)

        state = OpenCodeACPSessionState(
            session_id="session-789",
            cwd="/workspace",
            workspace_fingerprint="fingerprint123",
        )

        # Store in session
        session[OpenCodeACPSessionState].seed(state)

        # Retrieve from session
        retrieved = session[OpenCodeACPSessionState].latest()
        assert retrieved is not None
        assert retrieved.session_id == "session-789"
        assert retrieved.cwd == "/workspace"
        assert retrieved.workspace_fingerprint == "fingerprint123"

    def test_update_creates_new_instance(self) -> None:
        state = OpenCodeACPSessionState(
            session_id="original",
            cwd="/original",
            workspace_fingerprint="original-fp",
        )

        updated = state.update(session_id="updated")
        assert updated is not state
        assert updated.session_id == "updated"
        assert updated.cwd == "/original"  # Unchanged
        assert updated.workspace_fingerprint == "original-fp"  # Unchanged
