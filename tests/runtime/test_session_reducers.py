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

from dataclasses import dataclass
from typing import TYPE_CHECKING

from weakincentives.runtime.session import Session
from weakincentives.runtime.session._types import ReducerContextProtocol
from weakincentives.runtime.session.reducers import replace_latest_by

if TYPE_CHECKING:
    from weakincentives.runtime.events import EventBus


@dataclass(slots=True)
class _Sample:
    key: str
    value: str


@dataclass(slots=True)
class _ReducerEvent:
    value: _Sample


@dataclass(slots=True)
class _Context(ReducerContextProtocol):
    session: Session
    event_bus: EventBus


def test_replace_latest_by_replaces_matching_entry() -> None:
    reducer = replace_latest_by(lambda item: item.key)
    session = Session()
    context = _Context(session=session, event_bus=session.event_bus)
    initial = (_Sample("a", "first"), _Sample("b", "second"))

    updated = reducer(
        initial,
        _ReducerEvent(value=_Sample("a", "updated")),
        context=context,
    )

    assert len(updated) == 2
    assert updated[-1].key == "a"
    assert updated[-1].value == "updated"
    assert any(item.value == "second" for item in updated)
