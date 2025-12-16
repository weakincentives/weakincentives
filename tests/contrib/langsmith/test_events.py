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

"""Tests for LangSmith events."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from weakincentives.contrib.langsmith import (
    LangSmithTraceCompleted,
    LangSmithTraceStarted,
    LangSmithUploadFailed,
)


class TestLangSmithTraceStarted:
    """Tests for LangSmithTraceStarted event."""

    def test_creation(self) -> None:
        """Event can be created with required fields."""
        trace_id = uuid4()
        session_id = uuid4()
        now = datetime.now(UTC)

        event = LangSmithTraceStarted(
            trace_id=trace_id,
            session_id=session_id,
            project="test-project",
            created_at=now,
        )

        assert event.trace_id == trace_id
        assert event.session_id == session_id
        assert event.project == "test-project"
        assert event.created_at == now
        assert isinstance(event.event_id, UUID)

    def test_session_id_optional(self) -> None:
        """Event accepts None session_id."""
        event = LangSmithTraceStarted(
            trace_id=uuid4(),
            session_id=None,
            project="test",
            created_at=datetime.now(UTC),
        )

        assert event.session_id is None

    def test_frozen(self) -> None:
        """Event is immutable."""
        event = LangSmithTraceStarted(
            trace_id=uuid4(),
            session_id=None,
            project="test",
            created_at=datetime.now(UTC),
        )

        with pytest.raises(AttributeError):
            event.project = "new"


class TestLangSmithTraceCompleted:
    """Tests for LangSmithTraceCompleted event."""

    def test_creation(self) -> None:
        """Event can be created with required fields."""
        trace_id = uuid4()
        now = datetime.now(UTC)

        event = LangSmithTraceCompleted(
            trace_id=trace_id,
            run_count=5,
            total_tokens=1000,
            trace_url="https://smith.langchain.com/trace/123",
            created_at=now,
        )

        assert event.trace_id == trace_id
        assert event.run_count == 5
        assert event.total_tokens == 1000
        assert event.trace_url == "https://smith.langchain.com/trace/123"
        assert event.created_at == now

    def test_trace_url_optional(self) -> None:
        """Event accepts None trace_url."""
        event = LangSmithTraceCompleted(
            trace_id=uuid4(),
            run_count=1,
            total_tokens=100,
            trace_url=None,
            created_at=datetime.now(UTC),
        )

        assert event.trace_url is None


class TestLangSmithUploadFailed:
    """Tests for LangSmithUploadFailed event."""

    def test_creation(self) -> None:
        """Event can be created with required fields."""
        trace_id = uuid4()
        now = datetime.now(UTC)

        event = LangSmithUploadFailed(
            trace_id=trace_id,
            error="Connection refused",
            retry_count=3,
            created_at=now,
        )

        assert event.trace_id == trace_id
        assert event.error == "Connection refused"
        assert event.retry_count == 3
        assert event.created_at == now

    def test_trace_id_optional(self) -> None:
        """Event accepts None trace_id for pre-trace failures."""
        event = LangSmithUploadFailed(
            trace_id=None,
            error="API key missing",
            retry_count=0,
            created_at=datetime.now(UTC),
        )

        assert event.trace_id is None
