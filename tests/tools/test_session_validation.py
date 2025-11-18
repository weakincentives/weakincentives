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

"""Validation coverage for tool session bindings."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pytest

from tests.tools.helpers import build_tool_context
from weakincentives.prompt.tool import ToolContext
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session
from weakincentives.runtime.session.protocols import SessionProtocol
from weakincentives.tools.asteval import AstevalSection
from weakincentives.tools.planning import PlanningToolsSection
from weakincentives.tools.podman import PodmanToolsSection
from weakincentives.tools.vfs import VfsToolsSection


class _DummySession(SessionProtocol):
    def snapshot(self) -> object:  # pragma: no cover - protocol stub
        return object()

    def rollback(self, snapshot: object) -> None:  # pragma: no cover - protocol stub
        return None

    def reset(self) -> None:  # pragma: no cover - protocol stub
        return None


class _Preparable(Protocol):
    def prepare_session(self, *, context: ToolContext) -> Session: ...


@pytest.mark.parametrize(
    "factory",
    (
        lambda bus, session, tmp_path: PlanningToolsSection(session=session),
        lambda bus, session, tmp_path: VfsToolsSection(session=session),
        lambda bus, session, tmp_path: AstevalSection(session=session),
        lambda bus, session, tmp_path: PodmanToolsSection(
            session=session,
            base_url="http://podman.invalid",
            identity=None,
            cache_dir=tmp_path,
            client_factory=lambda: None,
        ),
    ),
)
def test_sections_reject_non_session(
    factory: Callable[[InProcessEventBus, Session, Path], _Preparable], tmp_path: Path
) -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    dummy_session = _DummySession()
    section = factory(bus, session, tmp_path)
    context = build_tool_context(bus, dummy_session)

    with pytest.raises(TypeError):
        section.prepare_session(context=context)
