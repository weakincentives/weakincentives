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

"""Tests for core session operations (construction, cloning, reset)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from tests.helpers.session import ExampleOutput
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import (
    Session,
    replace_latest,
)

if TYPE_CHECKING:
    from tests.conftest import SessionFactory

pytestmark = pytest.mark.core


def test_session_requires_timezone_aware_created_at() -> None:
    dispatcher = InProcessDispatcher()
    naive_timestamp = datetime.now()

    with pytest.raises(ValueError):
        Session(dispatcher=dispatcher, created_at=naive_timestamp)


def test_session_instantiates_default_dispatcher_when_none_provided() -> None:
    session = Session()
    assert isinstance(session.dispatcher, InProcessDispatcher)


def test_reset_clears_registered_slices(session_factory: SessionFactory) -> None:
    from weakincentives.runtime.session import append_all

    session, _ = session_factory()

    session[ExampleOutput].register(ExampleOutput, append_all)

    # Dispatch payload directly (payloads no longer extracted from telemetry events)
    session.dispatch(ExampleOutput(text="first"))
    assert session[ExampleOutput].all()

    session.reset()

    assert session[ExampleOutput].all() == ()

    # Dispatch payload directly
    session.dispatch(ExampleOutput(text="second"))
    assert session[ExampleOutput].all() == (ExampleOutput(text="second"),)


def test_clone_preserves_state_and_reducer_registration(
    session_factory: SessionFactory,
) -> None:
    provided_session_id = uuid4()
    provided_created_at = datetime.now(UTC)
    session, _ = session_factory(
        session_id=provided_session_id, created_at=provided_created_at
    )

    session[ExampleOutput].register(ExampleOutput, replace_latest)

    # Dispatch payload directly (payloads no longer extracted from telemetry events)
    session.dispatch(ExampleOutput(text="first"))

    clone_dispatcher = InProcessDispatcher()
    clone = session.clone(dispatcher=clone_dispatcher)

    assert clone.session_id == provided_session_id
    assert clone.created_at == provided_created_at
    assert clone[ExampleOutput].all() == (ExampleOutput(text="first"),)
    assert session[ExampleOutput].all() == (ExampleOutput(text="first"),)
    assert clone._reducers.keys() == session._reducers.keys()

    # Dispatch to clone session
    clone.dispatch(ExampleOutput(text="second"))

    assert clone[ExampleOutput].all()[-1] == ExampleOutput(text="second")
    assert session[ExampleOutput].all() == (ExampleOutput(text="first"),)

    # Dispatch to original session
    session.dispatch(ExampleOutput(text="third"))

    assert session[ExampleOutput].all()[-1] == ExampleOutput(text="third")
    assert clone[ExampleOutput].all()[-1] == ExampleOutput(text="second")


def test_clone_attaches_to_new_dispatcher_when_provided(
    session_factory: SessionFactory,
) -> None:
    session, _ = session_factory()

    # Dispatch payload directly (payloads no longer extracted from telemetry events)
    session.dispatch(ExampleOutput(text="first"))

    target_dispatcher = InProcessDispatcher()
    clone_session_id = uuid4()
    clone_created_at = datetime.now(UTC)
    clone = session.clone(
        dispatcher=target_dispatcher,
        session_id=clone_session_id,
        created_at=clone_created_at,
    )

    assert clone.session_id == clone_session_id
    assert clone.created_at == clone_created_at
    assert clone[ExampleOutput].all() == session[ExampleOutput].all()

    # Dispatch to clone session
    clone.dispatch(ExampleOutput(text="from clone"))

    assert clone[ExampleOutput].all()[-1] == ExampleOutput(text="from clone")
    assert session[ExampleOutput].all()[-1] == ExampleOutput(text="first")

    # Dispatch to original session
    session.dispatch(ExampleOutput(text="original"))

    assert session[ExampleOutput].all()[-1] == ExampleOutput(text="original")
