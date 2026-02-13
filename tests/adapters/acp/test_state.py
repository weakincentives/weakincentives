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

"""Tests for ACP session state slice."""

from __future__ import annotations

from weakincentives.adapters.acp._state import ACPSessionState


class TestACPSessionState:
    def test_creation(self) -> None:
        state = ACPSessionState(
            session_id="sess-123",
            cwd="/tmp/work",
            workspace_fingerprint="abc123",
        )
        assert state.session_id == "sess-123"
        assert state.cwd == "/tmp/work"
        assert state.workspace_fingerprint == "abc123"

    def test_none_fingerprint(self) -> None:
        state = ACPSessionState(
            session_id="sess-456",
            cwd="/home/user",
            workspace_fingerprint=None,
        )
        assert state.workspace_fingerprint is None

    def test_frozen(self) -> None:
        state = ACPSessionState(
            session_id="sess-789",
            cwd="/tmp",
            workspace_fingerprint=None,
        )
        try:
            state.session_id = "other"  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised
