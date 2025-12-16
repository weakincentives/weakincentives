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

"""Tests for LangSmith context management."""

from __future__ import annotations

from uuid import uuid4

from weakincentives.contrib.langsmith._context import (
    TraceContext,
    clear_context,
    clear_current_run_tree,
    get_context,
    get_current_run_tree,
    set_context,
    set_current_run_tree,
)


class TestTraceContext:
    """Tests for TraceContext dataclass."""

    def test_creation(self) -> None:
        """TraceContext can be created with required fields."""
        trace_id = uuid4()
        root_run_id = uuid4()
        session_id = uuid4()

        context = TraceContext(
            trace_id=trace_id,
            root_run_id=root_run_id,
            current_run_id=root_run_id,
            session_id=session_id,
        )

        assert context.trace_id == trace_id
        assert context.root_run_id == root_run_id
        assert context.current_run_id == root_run_id
        assert context.session_id == session_id
        assert context.run_count == 0
        assert context.total_tokens == 0

    def test_mutable(self) -> None:
        """TraceContext is mutable for run tracking."""
        context = TraceContext(
            trace_id=uuid4(),
            root_run_id=uuid4(),
            current_run_id=uuid4(),
            session_id=uuid4(),
        )

        context.run_count = 5
        context.total_tokens = 1000

        assert context.run_count == 5
        assert context.total_tokens == 1000


class TestContextManagement:
    """Tests for context get/set/clear functions."""

    def test_get_context_returns_none_for_unknown_session(self) -> None:
        """get_context returns None for unknown session."""
        session_id = uuid4()
        assert get_context(session_id) is None

    def test_get_context_returns_none_for_none_session(self) -> None:
        """get_context returns None when session_id is None."""
        assert get_context(None) is None

    def test_set_and_get_context(self) -> None:
        """set_context and get_context work together."""
        session_id = uuid4()
        context = TraceContext(
            trace_id=uuid4(),
            root_run_id=uuid4(),
            current_run_id=uuid4(),
            session_id=session_id,
        )

        set_context(session_id, context)

        try:
            retrieved = get_context(session_id)
            assert retrieved is context
        finally:
            clear_context(session_id)

    def test_clear_context_returns_removed_context(self) -> None:
        """clear_context returns the removed context."""
        session_id = uuid4()
        context = TraceContext(
            trace_id=uuid4(),
            root_run_id=uuid4(),
            current_run_id=uuid4(),
            session_id=session_id,
        )

        set_context(session_id, context)
        removed = clear_context(session_id)

        assert removed is context
        assert get_context(session_id) is None

    def test_clear_context_returns_none_for_unknown(self) -> None:
        """clear_context returns None for unknown session."""
        session_id = uuid4()
        assert clear_context(session_id) is None

    def test_clear_context_returns_none_for_none_session(self) -> None:
        """clear_context returns None when session_id is None."""
        assert clear_context(None) is None

    def test_multiple_sessions(self) -> None:
        """Multiple sessions can have independent contexts."""
        session1 = uuid4()
        session2 = uuid4()

        context1 = TraceContext(
            trace_id=uuid4(),
            root_run_id=uuid4(),
            current_run_id=uuid4(),
            session_id=session1,
        )
        context2 = TraceContext(
            trace_id=uuid4(),
            root_run_id=uuid4(),
            current_run_id=uuid4(),
            session_id=session2,
        )

        set_context(session1, context1)
        set_context(session2, context2)

        try:
            assert get_context(session1) is context1
            assert get_context(session2) is context2
        finally:
            clear_context(session1)
            clear_context(session2)


class TestRunTreeManagement:
    """Tests for run tree get/set/clear functions."""

    def test_get_current_run_tree_default_none(self) -> None:
        """get_current_run_tree returns None by default."""
        clear_current_run_tree()
        assert get_current_run_tree() is None

    def test_set_and_get_run_tree(self) -> None:
        """set_current_run_tree and get_current_run_tree work together."""
        mock_tree = {"id": "test-run-tree"}

        set_current_run_tree(mock_tree)

        try:
            assert get_current_run_tree() is mock_tree
        finally:
            clear_current_run_tree()

    def test_clear_run_tree(self) -> None:
        """clear_current_run_tree removes the run tree."""
        set_current_run_tree({"id": "test"})
        clear_current_run_tree()

        assert get_current_run_tree() is None
