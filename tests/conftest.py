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

from typing import Protocol

import pytest

from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session


class SessionFactory(Protocol):
    def __call__(
        self,
        *,
        session_id: str | None = None,
        created_at: str | None = None,
    ) -> tuple[Session, InProcessEventBus]:
        """Return a newly constructed session and bus pair."""


@pytest.fixture
def session_factory() -> SessionFactory:
    """Return a factory that creates session and bus pairs."""

    def factory(
        *,
        session_id: str | None = None,
        created_at: str | None = None,
    ) -> tuple[Session, InProcessEventBus]:
        bus = InProcessEventBus()
        session = Session(bus=bus, session_id=session_id, created_at=created_at)
        return session, bus

    return factory
