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


# Capture the original SystemClock.__init__ at module load time to avoid
# order-dependent issues with monkeypatch capturing a patched version.
def _get_original_system_clock_init() -> object:
    """Get the original SystemClock.__init__ method."""
    from weakincentives.runtime.clock import SystemClock

    return SystemClock.__init__


_ORIGINAL_SYSTEM_CLOCK_INIT = _get_original_system_clock_init()


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

    # Use the module-level captured original to avoid order-dependent issues
    original_init = _ORIGINAL_SYSTEM_CLOCK_INIT

    def patched_init(self: SystemClock) -> None:
        # Get test name from traceback
        stack = traceback.extract_stack()
        test_info = "unknown"
        test_frame_idx = -1

        # First pass: find the test frame
        for idx, frame in enumerate(stack):
            if "/tests/" in frame.filename and not frame.filename.endswith(
                "conftest.py"
            ):
                test_file = frame.filename.split("/tests/")[-1]
                test_info = f"{test_file}::{frame.name}"
                test_frame_idx = idx

        # If no test code in stack, allow it (e.g., module-level imports)
        if test_frame_idx == -1:
            original_init(self)  # type: ignore[misc]
            return

        # Second pass: check frames AFTER the test frame (towards SystemClock)
        # If any of these are production code or dataclass-generated, allow it
        # Note: We exclude clock.py itself since that's where SystemClock is defined
        for frame in stack[test_frame_idx + 1 :]:
            # Skip the clock.py file itself and conftest.py
            if "clock.py" in frame.filename or "conftest.py" in frame.filename:
                continue
            if (
                "/src/weakincentives/" in frame.filename
                or frame.filename == "<string>"  # dataclass-generated __init__
            ):
                # Production code or dataclass machinery is in the call chain
                # between test and SystemClock - this is allowed
                # (e.g., default factories in dataclasses)
                original_init(self)  # type: ignore[misc]
                return

        # Direct call from test code without going through production code
        raise RuntimeError(
            f"SystemClock instantiated directly in test '{test_info}'. "
            "Tests must use FakeClock for deterministic time control. "
            "If this test requires real blocking behavior, add "
            "@pytest.mark.allow_system_clock decorator."
        )

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
