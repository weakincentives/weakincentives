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

from datetime import datetime
from typing import Protocol
from uuid import UUID, uuid4

import pytest

from weakincentives.runtime.clock import Clock
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session

pytest_plugins = (
    "tests.plugins.dbc",
    "tests.plugins.dataclass_serde",
    "tests.plugins.threadstress",
    "tests.plugins.tool_contracts",
    "tests.helpers.time",
)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "allow_system_clock: mark test as allowed to use SystemClock (integration tests)",
    )


@pytest.fixture(autouse=True)
def _enforce_fake_clock_in_tests(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent direct SystemClock instantiation in test code.

    Test code should use FakeClock for deterministic time control. This ensures:
    - Deterministic test behavior
    - No time-based race conditions
    - Consistent timestamps across test runs

    Note: Production code default factories (like _default_created_at) are allowed
    to use SystemClock since they're not directly called from test code.

    To allow SystemClock in integration tests, mark the test with:
        @pytest.mark.allow_system_clock
    """
    # Allow escape hatch for integration tests via marker
    if request.node.get_closest_marker("allow_system_clock"):
        return

    # Import here to avoid circular dependency
    import traceback

    from weakincentives.runtime.clock import SystemClock

    original_init = SystemClock.__init__

    def patched_init(self: SystemClock) -> None:
        # Get test name from traceback
        stack = traceback.extract_stack()
        test_info = "unknown"
        called_from_test_code = False

        for frame in reversed(stack):
            # Check if any frame is directly in test code
            if "/tests/" in frame.filename and not frame.filename.endswith(
                "conftest.py"
            ):
                test_file = frame.filename.split("/tests/")[-1]
                test_info = f"{test_file}::{frame.name}"
                # Only flag if this frame is the direct caller (not via production code)
                # Production code paths contain /src/weakincentives/
                called_from_test_code = True
            elif called_from_test_code and (
                "/src/weakincentives/" in frame.filename
                or frame.filename == "<string>"  # dataclass-generated __init__
            ):
                # Production code is in the call chain between test and SystemClock
                # This is allowed (e.g., default factories in dataclasses)
                # Note: <string> is the filename for dataclass-generated __init__ methods
                original_init(self)
                return

        if called_from_test_code:
            raise RuntimeError(
                f"SystemClock instantiated directly in test '{test_info}'. "
                "Tests must use FakeClock for deterministic time control. "
                "If this test requires real blocking behavior, add "
                "@pytest.mark.allow_system_clock decorator."
            )

        # Not called from test code at all, allow it
        original_init(self)

    monkeypatch.setattr(SystemClock, "__init__", patched_init)


class SessionFactory(Protocol):
    def __call__(
        self,
        *,
        session_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> tuple[Session, InProcessDispatcher]:
        """Return a newly constructed session and dispatcher pair."""


@pytest.fixture
def session_factory(clock: Clock) -> SessionFactory:
    """Return a factory that creates session and dispatcher pairs."""

    def factory(
        *,
        session_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> tuple[Session, InProcessDispatcher]:
        dispatcher = InProcessDispatcher()
        resolved_session_id = session_id if session_id is not None else uuid4()
        resolved_created_at = created_at if created_at is not None else clock.now()
        session = Session(
            dispatcher=dispatcher,
            session_id=resolved_session_id,
            created_at=resolved_created_at,
            tags={"suite": "tests"},
            clock=clock,
        )
        return session, dispatcher

    return factory
