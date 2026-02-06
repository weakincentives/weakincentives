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

"""Tests for Codex App Server session state slice."""

from __future__ import annotations

from weakincentives.adapters.codex_app_server._state import (
    CodexAppServerSessionState,
)


class TestCodexAppServerSessionState:
    def test_creation(self) -> None:
        state = CodexAppServerSessionState(
            thread_id="thread-abc",
            cwd="/tmp/work",
            workspace_fingerprint="abc123",
        )
        assert state.thread_id == "thread-abc"
        assert state.cwd == "/tmp/work"
        assert state.workspace_fingerprint == "abc123"

    def test_none_fingerprint(self) -> None:
        state = CodexAppServerSessionState(
            thread_id="t1",
            cwd="/",
            workspace_fingerprint=None,
        )
        assert state.workspace_fingerprint is None

    def test_dynamic_tool_names_default(self) -> None:
        state = CodexAppServerSessionState(
            thread_id="t1",
            cwd="/",
            workspace_fingerprint=None,
        )
        assert state.dynamic_tool_names == ()

    def test_dynamic_tool_names_explicit(self) -> None:
        state = CodexAppServerSessionState(
            thread_id="t1",
            cwd="/",
            workspace_fingerprint=None,
            dynamic_tool_names=("alpha", "beta"),
        )
        assert state.dynamic_tool_names == ("alpha", "beta")

    def test_frozen(self) -> None:
        state = CodexAppServerSessionState(
            thread_id="t1", cwd="/", workspace_fingerprint=None
        )
        try:
            state.thread_id = "t2"  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised
