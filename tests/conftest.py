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

from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

import pytest

from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session

pytest_plugins = ["tests.plugins.threadstress"]


class SessionFactory(Protocol):
    def __call__(
        self,
        *,
        session_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> tuple[Session, InProcessEventBus]:
        """Return a newly constructed session and bus pair."""


@pytest.fixture
def session_factory() -> SessionFactory:
    """Return a factory that creates session and bus pairs."""

    def factory(
        *,
        session_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> tuple[Session, InProcessEventBus]:
        bus = InProcessEventBus()
        resolved_session_id = session_id if session_id is not None else uuid4()
        resolved_created_at = (
            created_at if created_at is not None else datetime.now(UTC)
        )
        session = Session(
            bus=bus,
            session_id=resolved_session_id,
            created_at=resolved_created_at,
        )
        return session, bus

    return factory
